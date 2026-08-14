'''Batch/cluster port of notebooks/simple-mde-gmn.ipynb, executed by
scripts/batch/simple-mde-gmn.sbatch. Every parameter is read from an
MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
PARAMETERS block) rather than edited here directly.

Pipeline steps (see the notebook for the narrative "why" behind each one
-- docstrings here cover only what's specific to running headless):

  1. MDE forward-selection (skipped if MODE='GMN'): per-target driver
     selection (dimx.MDE). Feeds both the MDE-only diagnostics in step 7
     and, when MODE='MDE-GMN', step 2's depth-0 seed.
  2. Network construction -- two unrelated methods depending on MODE;
     everything after this step is identical either way:
       - MODE='MDE-GMN': build_nested_mde_network() repeatedly runs
         dimx.MDE, turning each selected driver into a new MDE target for
         MDE_GMN_NESTED_M further rounds, rejecting any edge that would
         create a cycle or self-loop as it's proposed. No CCM, no
         apps.InteractionMatrix, no apps.CreateNetwork -- the resulting
         graph IS the network handed to gmn.GMN directly (see
         build_network()'s docstring for why CreateNetwork would only
         be redundant here).
       - MODE='GMN': the original CCM interaction matrix (data-driven
         E/tau grid search over the REGION_CHANNEL_CAP-limited
         observable pool, via apps.InteractionMatrix), then a per-node
         driver-count estimate (EmbedDimension) and apps.CreateNetwork
         with its driver-ranking-bug workaround.
  3. Node generation config: SMap uniformly for every node, never Simplex
     or a per-node mix -- see build_gmn_config()'s docstring for why.
  4. Empirical (E, tau, per-node theta) search maximizing free-running
     Generate() fidelity against the real held-out targets.
  5. Start-point sweep: at the winning (E, tau, theta), tries several
     rows for Generate()/Forecast() to launch from and keeps whichever
     matches the real data best -- see sweep_generate_start()'s docstring.
  6. Final rebuild at the winning parameters and start row, plus the
     Forecast() diagnostic and a Fig-4e-style generative fidelity metric.
  7. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
     univariate E/tau grid) -- skipped if MODE='GMN'.

Differences from the notebook, all consequences of running headless
instead of inside a live Jupyter kernel:
  - apps.InteractionMatrix runs via subprocess.run instead of
    get_ipython().system(), which only exists inside a real kernel.
  - plt.show() calls are replaced with plt.close(fig) after the matching
    savefig, so figures don't accumulate over a long run.
  - All output goes under one flat RESULTS_DIR instead of the notebook's
    separate results/vs/figures/ split, so the sbatch script can copy it
    back with a single `cp -av`.
  - CORES is threaded through every *_cores/numProcess parameter (dimx.MDE,
    apps.InteractionMatrix's -cr, pyEDM's numProcess, gmn.GMN's cores --
    the last is currently a no-op in the installed gmn package, passed
    anyway in case a future version uses it) instead of each tool's own
    hardcoded default, which has no way to know how many cores SLURM
    actually allocated.

FIXED: apps.InteractionMatrix's own worker pool and pyEDM.CCM()'s
internal multiprocessing used to nest inside each other -- CORES=8 could
spawn ~90 live processes, not 8, with each of CCM()'s ~100k+ calls paying
a full process-spawn-and-reimport cost for only 4-8 tiny KDTree tasks.
This is why early jobs took ~2 days regardless of dataset size.
`_sequential_ccm_workers` below forces pyEDM.CCM() onto its in-process
path everywhere in this pipeline, removing the inner pool so only the
outer one provides parallelism -- see that function's and
build_network()'s docstrings for the full evidence trail.

SEARCH REUSE (MDE_GMN_REUSE_BEST_PARAMS, default off): the E/tau/theta
search is deterministic given the same network and data, so resubmitting
identical settings pays for that search twice for no reason. When
enabled, main() builds a fingerprint per network (every config knob that
shapes its topology/search space, a hash of the input data file, and its
own realized node set) and looks for a past job's
best_params_<network_id>.json under MDE_GMN_PREV_RUNS_DIR with a
matching fingerprint -- a hit skips that network's search and reuses its
cached E/tau/theta; any mismatch (different dataset, different MDE_D/
NESTED_M/REGION_CHANNEL_CAP/etc., or the first run with these settings)
just runs the real search, so this is always a safe, redundant-compute
optimization, never a correctness shortcut. See
build_search_fingerprint() and find_cached_best_params().
'''
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd

import dimx as dx
import gmn
from gmn.Common import FunctionType
from apps.CreateNetwork import CreateNetwork
from pyEDM import (
    Simplex, SMap, CCM, ComputeError, Embed, EmbedDimension,
    PredictNonlinear, PredictInterval,
)


def _fixed_node_forecast(self):
    '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
    bugs in the installed package's SMap branch -- the only branch this
    pipeline exercises, since every node uses SMap (see
    build_gmn_config()'s docstring):

    (1) gmn/Forecast.py calls Embed() without importing pyEDM.Embed (it
    only does `from .Common import *`, unlike gmn/Generate.py's identical
    SMap branch, which imports it correctly) -- raises NameError on the
    first Forecast() call against any SMap node.

    (2) `val` is a 2-tuple (time, predictions), then checked for
    divergence via `val.max() > 1E15` -- tuples have no .max(), so this
    raises AttributeError unconditionally, not just on genuine
    divergence. Fixed to check the predictions Series directly (val[1]).

    Every other branch is copied verbatim from gmn/Forecast.py (unused
    here, since every node is SMap, but kept intact so this patch can't
    silently change behavior for a future non-SMap node). Monkeypatched
    onto gmn.Node (gmn/__init__.py's `from .Node import Node` shadows the
    submodule name with the class) rather than editing the installed
    package, so this script stays self-contained on other machines.'''
    if self.args.DEBUG :
        print( '-----> Node:Forecast() : ',self.FunctionType.name, self.name)
        print( self.data.tail( 2 ), flush = True );

    Parameters = self.Parameters
    data       = self.data

    if self.args.DEBUG :
        print( '  data : shape: ', data.shape );
        print( data.tail( 3 ) ); print('', flush = True)

    if self.FunctionType.value == FunctionType.Simplex.value :
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else:
            lib = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )

        S = self.Function( dataFrame       = data,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = Parameters.columns,
                           target          = Parameters.target,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S.iloc[:,0], S['Predictions'] )

    elif self.FunctionType.value == FunctionType.SMap.value :
        df = Embed( dataFrame = data, E = Parameters.E,
                    tau = Parameters.tau, columns = Parameters.columns )

        df = df.dropna()

        df.insert( 0, data.columns[0], data.iloc[:,0] )

        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else :
            lib  = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )

        S = self.Function( dataFrame       = df,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           theta           = Parameters.theta,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = df.columns[1:-1],
                           target          = Parameters.target + '(t-0)',
                           solver          = None,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S['predictions'].iloc[:,0], S['predictions']['Predictions'] )

        if val[1].max() > 1E15 :  # FIX: was val.max() -- val is the tuple above
            raise RuntimeError( "Forecast:SMap() Node: " + self.name +\
                                "   Divergence detected (>1E15)" )

    elif self.FunctionType.value == FunctionType.kedmSimplex.value :
        raise RuntimeError( "Forecast kedmSimplex function not available" )

    elif self.FunctionType.value == FunctionType.kedmSMap.value :
        raise RuntimeError( "Forecast kedmSMap function not available" )

    elif self.FunctionType.value == FunctionType.Linear.value :
        raise RuntimeError( "Forecast Linear function not available" )

    elif self.FunctionType.value == FunctionType.SVR.value :
        raise RuntimeError( "Forecast SVR function not available" )

    elif self.FunctionType.value == FunctionType.knn.value :
        raise RuntimeError( "Forecast knn function not available" )

    if self.args.DEBUG :
        print( self.name, "val:", val )
        print( '<----- Node:Forecast() : ',self.FunctionType.name,self.name )
        print('', flush = True)

    return val


gmn.Node.Forecast = _fixed_node_forecast


def _sequential_ccm_workers(parallel):
    '''Replaces pyEDM.CCM's private _resolve_workers(), forcing every
    CCM().Project() call in this process onto its sequential in-process
    path instead of spawning a fresh multiprocessing Pool.

    This is the fix for the nested-pool blowup described in the module
    docstring: every pyEDM.CCM() call (one per channel pair in
    apps.InteractionMatrix, one per candidate column in dimx.Run) used to
    spawn its own fresh Pool sized to cpu_count(), on top of whichever
    outer pool was already dispatching that call -- confirmed by reading
    pyEDM/CCM.py and by the "No module named 'kedm'" line each spawned
    worker printed on reimport, hundreds of thousands of times in one run.
    Neither apps.InteractionMatrix nor dimx.Run ever pass `parallel=`
    through to CCM(), so this couldn't be tuned via their CLIs.

    Forcing sequential execution is a strict win, not a tradeoff: every
    CCM() call here only ever resolves to 4-8 tiny KDTree tasks (2
    directions x 2-4 libSizes), so a fresh Pool's spawn-and-reimport cost
    dwarfs the actual work. Real parallelism across channel pairs/
    candidates already comes from the outer pool, which this doesn't
    touch. Monkeypatched (rather than passing parallel=False at each call
    site) because apps.InteractionMatrix constructs CCM() internally with
    no way to pass that argument in; dimx.Run's own CCM() call gets a
    second, independent patch below (_sequential_ccm_api) that does force
    parallel=False explicitly, so this override is the sole protection
    for apps.InteractionMatrix and a belt-and-suspenders one for dimx.'''
    return 1


import pyEDM.CCM as _pyedm_ccm_module
_pyedm_ccm_module._resolve_workers = _sequential_ccm_workers

import dimx.Run as _dimx_run_module
_real_pyedm_ccm_api = _dimx_run_module.CCM


def _sequential_ccm_api(*args, **kwargs):
    '''Second, independent layer on top of _sequential_ccm_workers: forces
    parallel=False on pyEDM's top-level CCM() function directly. Patches
    the `CCM` name inside dimx.Run's own namespace (dimx/Run.py imports it
    as `from pyEDM import ... CCM ...`, so `dimx.Run.CCM` is the exact
    binding dimx.MDE.Run() calls) rather than relying only on the
    _resolve_workers indirection above, in case that patch doesn't reach
    every call the same way in every process.'''
    kwargs['parallel'] = False
    return _real_pyedm_ccm_api(*args, **kwargs)


_dimx_run_module.CCM = _sequential_ccm_api


def _mde_wave_worker_init():
    '''ProcessPoolExecutor(initializer=...) hook for
    build_nested_mde_network()'s per-wave worker pool -- reapplies both
    CCM patches above inside every worker process, rather than assuming
    either survives from this module's own top-level monkeypatch lines
    (it doesn't reliably on this cluster).

    The failure mode when it doesn't is a hard crash, not just a slowdown:
    a worker whose dimx.MDE -> CCM() call lands on the unpatched path
    tries to create CCM's own Pool from inside an already-pooled worker
    process, which fails outright with "ChildProcessError: No child
    processes". The exact trigger was never fully pinned down, but
    applying both patches together, fresh inside every worker via this
    initializer, reliably avoided it across repeated test runs, while
    either patch alone did not.'''
    import pyEDM.CCM as _ccm
    _ccm._resolve_workers = _sequential_ccm_workers
    import dimx.Run as _dimx_run
    _dimx_run.CCM = _sequential_ccm_api


# =============================================================================
# Configuration -- every value overridable via MDE_GMN_* environment
# variables; see scripts/batch/simple-mde-gmn.sbatch's MODIFIABLE
# PARAMETERS block for the canonical list with descriptions.
# =============================================================================

def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [x.strip() for x in val.split(',') if x.strip()]


def _env_float_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [float(x.strip()) for x in val.split(',') if x.strip()]


def load_config():
    cfg = {}

    # ---- Data ---------------------------------------------------------
    # NOTE: unlike the notebook (DATA_DIR = '../data/transfer_.../'), the
    # cluster deployment expects data files copied flat into DATA_DIR by
    # the sbatch script -- no nested extraction-folder name.
    cfg['DATA_DIR'] = Path(os.environ.get('MDE_GMN_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('MDE_GMN_DATA_FILE', 'J1620210605_epoch_8_freq_4_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('MDE_GMN_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- Overall pipeline shape ---------------------------------------
    cfg['MODE'] = os.environ.get('MDE_GMN_MODE', 'MDE-GMN')  # 'MDE-GMN' | 'GMN'
    if cfg['MODE'] not in ('MDE-GMN', 'GMN'):
        raise ValueError(f"Unknown MDE_GMN_MODE: {cfg['MODE']!r}")
    cfg['GMN_MODE'] = os.environ.get('MDE_GMN_GMN_MODE', 'combined')  # 'combined' | 'separate'

    cfg['TARGETS'] = _env_list('MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
    cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None

    # Lib/pred split as a fraction of the full dataset -- e.g. LIB_RATIO=0.5
    # means the first 50% of rows are library, the remaining 50% are the
    # held-out pred/Generate() window. PREDICTION_START/PREDICTION_LENGTH
    # (row counts derived from this ratio and the data's length) are filled
    # in by main() once the data is loaded, and are what the rest of the
    # pipeline reads: the MDE lib/pred split, CCM/EmbedDimension calls, and
    # the held-out window Generate() results are validated against.
    cfg['LIB_RATIO'] = float(os.environ.get('MDE_GMN_LIB_RATIO', 0.5))

    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH:
    # every Generate() call (in the E/tau/theta search, the start sweep, and
    # the final rebuild) runs this many steps, not the entire held-out
    # window. Generate() is the search loop's single most expensive step
    # (one full network simulation per candidate), so this shrinks that
    # cost independently of LIB_RATIO/MDE's own held-out window, which the
    # causality gating (CCM, EmbedDimension) still needs to be a meaningful
    # size. Also caps the Forecast() diagnostic's pred window (see
    # build_gmn_config()'s docstring) so actual/Generate()/Forecast() all
    # cover the same window on that plot. 0 (default) means "use the full
    # PREDICTION_LENGTH"; a larger value is clamped down to it in main()
    # once PREDICTION_LENGTH is known (there'd be no real data to score any
    # extra steps against).
    cfg['GENERATE_LENGTH'] = int(os.environ.get('MDE_GMN_GENERATE_LENGTH', 0))

    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
    # as offsets (rows) from PREDICTION_START -- e.g. MIN=0, MAX=200,
    # STEP=50 tries offsets 0, 50, 100, 150 (MAX exclusive, like E_MAX/
    # TAU_MAX below). sweep_generate_start() runs one Generate() call per
    # offset at the already-chosen best E/tau/theta (not re-searched) and
    # keeps whichever start best matches the real data: Generate() is a
    # recursive nonlinear simulation, so which row it launches from can
    # change how well it tracks reality, independent of how good E/tau/
    # theta are. Offsets must be >= 0 (never before PREDICTION_START) so
    # every candidate's comparison data stays genuinely out-of-sample.
    # Default MIN=0, MAX=1 tries only offset 0 (always PREDICTION_START,
    # unchanged pre-existing behavior) -- widen MAX to actually sweep.
    # Offsets that would leave too little data to score GENERATE_LENGTH
    # steps against are dropped once PREDICTION_LENGTH is known (main()).
    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
    cfg['GENERATE_START_STEP'] = int(os.environ.get('MDE_GMN_GENERATE_START_STEP', 50))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # Nested-MDE network construction (MODE='MDE-GMN' only -- see
    # build_nested_mde_network()'s docstring for the full algorithm).
    # Reuses MDE_D as the per-node driver cap N at every nesting level
    # (deliberately not a separate parameter -- MDE_D already means "max
    # # of driver vars MDE can forward-select", which is exactly N here
    # too). NESTED_M is the number of *additional* nested-MDE rounds
    # beyond the initial per-target selection -- total graph depth from
    # each target is 1 + NESTED_M levels. Combinatorially expensive: a
    # naive upper bound on nodes is roughly MDE_D**(1+NESTED_M) before
    # dedup/cycle-rejection collapses it back down to the real (capped)
    # observable pool size -- keep MDE_D and NESTED_M small for a first
    # test run.
    cfg['NESTED_M'] = int(os.environ.get('MDE_GMN_NESTED_M', 1))
    # Candidate observable pool: first N channels of each region. Default
    # 999 (effectively uncapped -- matches the current notebook's default
    # and Park et al. 2026's own full-channel-set construction); cap this
    # down for a fast test job.
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- E/tau/theta search ranges -------------------------------------
    # E >= 2 always: every GMN node uses SMap uniformly (see
    # build_gmn_config's docstring), and SMap's predictor-exclusion
    # convention needs E >= 2 even for leaf/undriven nodes.
    cfg['E_RANGE'] = range(
        int(os.environ.get('MDE_GMN_E_MIN', 3)),
        int(os.environ.get('MDE_GMN_E_MAX', 16)),
        int(os.environ.get('MDE_GMN_E_STEP', 2)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('MDE_GMN_TAU_MIN', -15)),
        int(os.environ.get('MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('MDE_GMN_TAU_STEP', 2)),
    )
    cfg['THETA_RANGE'] = _env_float_list(
        'MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    cfg['COARSE_THETA_RANGE'] = cfg['THETA_RANGE'][::3]
    cfg['SMAP_SEARCH_MODE'] = os.environ.get('MDE_GMN_SMAP_SEARCH_MODE', 'coarse_retune')  # 'coarse_retune' | 'joint'

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('MDE_GMN_RUN_MDE_DIAGNOSTICS', True)

    # ---- Parallelism ----------------------------------------------------
    # On HPC this should be SLURM_CPUS_PER_TASK, not left at each tool's
    # own package default (dimx: 5, apps.InteractionMatrix: 8, pyEDM
    # EmbedDimension/PredictNonlinear: their own internal default) -- see
    # module docstring's FIXED section for why this doesn't scale linearly.
    cfg['CORES'] = int(os.environ.get('MDE_GMN_CORES', os.cpu_count() or 1))
    # Deliberately NOT the same as CORES -- see build_network()'s
    # docstring: apps.InteractionMatrix's -cr and pyEDM.CCM()'s own
    # per-pair internal pool multiply rather than coordinate, so this
    # stays small regardless of how large CORES is.
    cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))

    # ---- Search reuse ---------------------------------------------------
    # See the module docstring's SEARCH REUSE section for the full
    # rationale; short version: reuse a past job's E/tau/theta search
    # result instead of redoing it, when settings match exactly.
    cfg['REUSE_BEST_PARAMS'] = _env_bool('MDE_GMN_REUSE_BEST_PARAMS', False)
    # Where to look for previous jobs' best_params_*.json files. Default
    # mirrors DATA_DIR/RESULTS_DIR's own '../../data'-style fallback, but
    # the sbatch script overrides this to the persistent $PROJ_DIR/results
    # -- unlike RESULTS_DIR, which the sbatch script points at a /flash
    # scratch tempdir deleted at job exit, PREV_RUNS_DIR needs to point at
    # wherever past jobs' outputs actually still exist.
    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('MDE_GMN_PREV_RUNS_DIR', '../../results'))

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg


# =============================================================================
# MDE (mirrors notebook cells 2-5)
# =============================================================================

def region_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]


def observable_columns(df, cap):
    return (
        region_columns(df, 'CA1_', cap)
        + region_columns(df, 'mPFC_', cap)
        + region_columns(df, 'OFC_', cap)
    )


def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                        mde_d, embed_dim_rho_min, cross_map_rho_min, cores, results_dir):
    '''Run dimx.MDE's forward selection for a single target variable.
    See notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
    full rationale (identical logic, just cores threaded through).'''
    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]

    m = dx.MDE( mde_df, target = target, D = mde_d,
                removeColumns = [target],
                lib  = [1, prediction_start],
                pred = [prediction_start + 1, prediction_start + prediction_length],
                embedDimRhoMin = embed_dim_rho_min,
                crossMapRhoMin = cross_map_rho_min,
                cores = cores )

    m.Run()

    m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)

    selected_columns = list(m.MDEcolumns)
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    return {
        'target': target,
        'mde_df': mde_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }


def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['m'].MDEOut
        dims = np.arange(1, len(mde_out) + 1)
        ax.plot(dims, mde_out['rho'], marker='o', label=target, color=target_colors[target])
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Cross-map rho')
    ax.set_title('MDE rho vs. dimension, per target')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_3d_mde_projection(target, second_target, mde_results, df,
                            prediction_start, prediction_length, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    obs = mde_results[target]['SELECTED_COLUMNS'][:2]
    if len(obs) < 2:
        print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.', flush=True)
        return

    window = df.iloc[:prediction_start + prediction_length]
    x = window[target].to_numpy()
    y = window[obs[0]].to_numpy()
    z = window[obs[1]].to_numpy()

    if second_target and second_target in window.columns:
        color_vals = window[second_target].to_numpy()
        color_label = second_target
    else:
        color_vals = np.arange(len(window))
        color_label = 'time index'

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='gray', linewidth=0.6, alpha=0.4)
    scatter = ax.scatter(x, y, z, c=color_vals, cmap='viridis', s=10)
    ax.set_xlabel(target)
    ax.set_ylabel(obs[0])
    ax.set_zlabel(obs[1])
    ax.set_title(f'MDE projection: {target} vs. its top-2 observables')
    fig.colorbar(scatter, ax=ax, shrink=0.7, label=color_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN network construction (mirrors notebook cell 7)
# =============================================================================

def _lib_train_test_ranges(n_rows, train_frac=0.7):
    split = int(n_rows * train_frac)
    return f'1 {split}', f'{split + 1} {n_rows}'


def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                S = Simplex(
                    dataFrame = lib_df, columns = target, target = target,
                    E = E, tau = tau, lib = lib_range, pred = pred_range,
                    embedded = False, showPlot = False,
                )
                rho_grid[i, j] = ComputeError(S['Observations'], S['Predictions'])['rho']
            except Exception:
                pass
    return rho_grid


def estimate_node_dimension(lib_df, node, cores, max_e=15, fallback=4):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = EmbedDimension(
            dataFrame = lib_df, columns = node, target = node,
            maxE = max_e, lib = lib_range, pred = pred_range, showPlot = False,
            numProcess = cores,
        )
        return int(ed.loc[ed['rho'].idxmax(), 'E'])
    except Exception:
        return fallback


def build_nested_mde_network(targets, df, observable_columns, prediction_start, prediction_length,
                              nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
                              results_dir, mde_results):
    '''Nested-MDE network construction -- MODE='MDE-GMN' only. MODE='GMN'
    still uses the original CCM-across-the-full-observable-pool path in
    build_network(), untouched.

    No notebook precedent: replaces "CCM over the union of each target's
    MDE-selected columns" with an algorithm built purely from repeated
    dimx.MDE calls (no CCM, no apps.InteractionMatrix here):

      Step 1 (depth 0): for each target, dimx.MDE selects up to
        `nested_n` (== MDE_D) driver columns from `observable_columns`.
        Reuses the caller's precomputed mde_results entry when available
        -- main() already ran this once per target for the MDE-only
        diagnostics, so it isn't paid for twice.
      Step 2 (depth 1..nested_m): each newly discovered driver becomes a
        new MDE target in turn, again selecting up to `nested_n` drivers
        from the same pool (excluding itself), for `nested_m` further
        rounds -- total graph depth from any target is 1 + nested_m
        levels.

    A branch's nesting stops before nested_m, as ordinary BFS
    termination, when:
      - A node's own dimx.MDE call selects zero drivers (embedDimRhoMin/
        crossMapRhoMin/the CCM slope gate rejected every candidate) --
        nothing gets enqueued for the next wave.
      - A selected driver's marginal MDE rho gain is NaN (a degenerate
        correlation, not the same as dimx.MDE's own gates rejecting it --
        it already passed those, or it wouldn't be in `cols`). Since this
        function's graph IS the final network (build_network() hands it
        to gmn.GMN directly -- no downstream re-filtering by weight), an
        edge with no trustworthy strength must be rejected here or it
        becomes a real edge with a nonsensical weight; nesting deeper
        through it would also be wasted work. `driver` can still enter
        the graph later via a different, valid edge. If every driver a
        node selected is NaN, that node is terminal -- the same
        empty-enqueue outcome as above.

    Cycle/self-loop guarding happens eagerly during discovery: each
    proposed edge is added to a networkx.DiGraph shared across this whole
    call, then checked with nx.is_directed_acyclic_graph() immediately --
    a cycle-causing edge is removed (the driver may still appear
    elsewhere via a different, non-cyclic edge). This is the only such
    guard for MODE='MDE-GMN' (no apps.CreateNetwork call follows), and
    dimx.MDE itself has no notion of the network being built around it,
    so a cycle has to be caught here or never. Self-loop exclusion is
    redundant with dimx.MDE's own removeColumns=[target] gate in the
    ordinary case, kept as an explicit guard anyway.

    Each unique node is MDE'd at most once (memoized via `visited`); it
    can still gain further incoming edges later (ordinary DAG
    driver-sharing), just isn't re-selected as an MDE target.

    Parallelization: every node at the same BFS depth is MDE-independent
    of every other (each dimx.MDE call only reads the static
    df/observable_columns pool). So each "wave" (not-yet-visited nodes
    newly discovered at one depth) is submitted concurrently via
    ProcessPoolExecutor, one process per node. Edge-acceptance/cycle-
    checking stays sequential once a wave's results are back -- cheap
    graph bookkeeping that genuinely needs one consistent DiGraph,
    nothing to gain from parallelizing.

    Outer (per-node) and inner (each node's own CrossMapColumns)
    parallelism are split adaptively per wave rather than both left at
    `cores` -- that would reproduce the outer x inner pool-squaring
    blowup this project hit once already (see the module docstring's
    FIXED section). outer_workers = min(len(wave), cores), inner_cores =
    max(1, cores // outer_workers): a wide wave leans on outer
    parallelism with each MDE call single-threaded; a narrow wave gives
    that one call the full remaining core budget. outer_workers *
    inner_cores never exceeds `cores`. dimx.MDE's own CCM validation is
    forced sequential by `_sequential_ccm_workers`, but that patch
    doesn't reliably survive into a ProcessPoolExecutor worker on its own
    (this cluster's 'forkserver' start method can predate the patch), so
    the wave pool passes `initializer=_mde_wave_worker_init` to reapply
    it inside every worker -- see that function's docstring for the
    crash this avoids.

    Edge weight is the *marginal* MDE rho gain at the step that driver
    was selected (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) --
    not the raw cumulative MDEOut['rho'] (dimx/Run.py cross-maps
    columns=[candidate]+already-selected, so cumulative rho trends
    upward with d regardless of any one driver's importance, which would
    make later, weaker drivers look equal-or-stronger than the first).
    The marginal gain isolates what that driver specifically added on
    top of the previous d-1 -- what "edge strength" should mean here.

    Returns (graph, iMatrix, nested_mde_results):
      graph   -- networkx.DiGraph, acyclic, no self-loops. The actual
                 network build_network() hands to gmn.GMN (see that
                 function's docstring) -- apps.CreateNetwork is not
                 involved.
      iMatrix -- NxN DataFrame (index==columns==graph.nodes): row =
                 driven node, column = driver, value = edge weight (0 =
                 no edge). Exported to CSV by build_network() as a
                 readable record alongside the graph; not fed to
                 apps.CreateNetwork.
      nested_mde_results -- {node: run_mde_for_target() result} for every
                 node this call ran dimx.MDE on. Kept separate from the
                 caller's own mde_results so repeated separate-mode calls
                 can't cross-contaminate a shared intermediate channel
                 name.
    '''
    graph = nx.DiGraph()
    graph.add_nodes_from(targets)

    edge_weight = {}          # (driver, node) -> weight
    nested_mde_results = {}   # node -> run_mde_for_target() result, this call only
    visited = set()
    current_wave = [(t, 0) for t in targets]

    while current_wave:
        # Dedup within the wave itself -- two different nodes discovered
        # in the *previous* wave can independently propose the same
        # not-yet-visited driver as their own new MDE target; each
        # unique node still only gets MDE'd once.
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        # Depth-0 targets reuse the caller's precomputed mde_results
        # directly (see docstring) -- no need to pay for, or parallelize,
        # that MDE call again.
        to_compute = [(node, depth) for node, depth in wave_nodes
                      if not (depth == 0 and node in mde_results)]
        wave_results = {
            node: mde_results[node] for node, depth in wave_nodes
            if depth == 0 and node in mde_results
        }

        if to_compute:
            outer_workers = min(len(to_compute), max(1, cores))
            inner_cores = max(1, cores // outer_workers)
            print(f'nested MDE: wave of {len(to_compute)} node(s) at depth '
                  f'{to_compute[0][1]} -- {outer_workers} parallel worker(s) '
                  f'x {inner_cores} core(s) each', flush=True)

            with ProcessPoolExecutor(max_workers=outer_workers,
                                      mp_context=mp.get_context('fork'),
                                      initializer=_mde_wave_worker_init) as executor:
                futures = {}
                for node, depth in to_compute:
                    candidate_pool = [c for c in observable_columns if c != node]
                    fut = executor.submit(
                        run_mde_for_target, node, df, candidate_pool,
                        prediction_start, prediction_length, nested_n,
                        embed_dim_rho_min, cross_map_rho_min, inner_cores, results_dir,
                    )
                    futures[fut] = node
                for fut in as_completed(futures):
                    wave_results[futures[fut]] = fut.result()

        next_wave = []
        for node, depth in wave_nodes:
            visited.add(node)
            result = wave_results[node]
            nested_mde_results[node] = result

            cols = result['SELECTED_COLUMNS']
            gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))

            for driver, gain in zip(cols, gains):
                if driver == node:
                    continue  # explicit self-loop guard (belt-and-suspenders)

                if np.isnan(gain):
                    # Degenerate marginal-gain computation -- dead-end
                    # this specific edge without nesting further through
                    # it (see this function's docstring for why NaN
                    # weights can't just be added anyway).
                    print(f'nested MDE: rejected {driver} -> {node} '
                          f'(marginal MDE rho gain is NaN -- treating as a dead-end edge, '
                          f'no further nesting from this driver via this path)', flush=True)
                    continue

                graph.add_node(driver)
                graph.add_edge(driver, node)

                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(driver, node)
                    if graph.degree(driver) == 0:
                        graph.remove_node(driver)
                    print(f'nested MDE: rejected {driver} -> {node} (would create a cycle)', flush=True)
                    continue

                edge_weight[(driver, node)] = max(gain, edge_weight.get((driver, node), 0.0))

                if depth < nested_m and driver not in visited:
                    next_wave.append((driver, depth + 1))

        current_wave = next_wave

    all_nodes = list(graph.nodes)
    iMatrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
    for (driver, node), weight in edge_weight.items():
        iMatrix.loc[node, driver] = weight

    print(f'nested MDE: {len(all_nodes)} nodes, {len(edge_weight)} edges '
          f'(N={nested_n}, M={nested_m})', flush=True)

    return graph, iMatrix, nested_mde_results


def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
                   observable_columns=None, nested_n=None, nested_m=None,
                   embed_dim_rho_min=None, cross_map_rho_min=None,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15):
    '''Build one network spanning `targets` and their driver candidates,
    via one of two unrelated constructions depending on `mode`:
      - mode == 'MDE-GMN': build_nested_mde_network()'s own graph IS the
        final network, handed to gmn.GMN directly -- no
        apps.CreateNetwork() call. Per explicit direction: nested-MDE
        already does the network-discovery job CreateNetwork exists for
        (BFS out from targets, rejecting cycles as edges are proposed),
        using real MDE-driven edge selection, so re-deriving a network
        from an interaction matrix afterward would just be redundant
        work. Safe because gmn.Network only ever reads the pickled
        network file's 'Graph' entry (gmn/Network.py:43-46) via
        Graph.predecessors(node) (gmn/Node.py:104) -- the 'Map' entry
        CreateNetwork also writes is explicitly unused (that file's own
        comment) -- so a {'Graph': graph, 'Map': ...} pickle built
        directly from build_nested_mde_network()'s DiGraph is exactly
        what CreateNetwork would have handed back anyway.
        observable_columns/nested_n/nested_m/embed_dim_rho_min/
        cross_map_rho_min are required in this mode. node_dimensions is
        NOT computed here (see gmn_fidelity_rho()'s default_E fallback
        for its one remaining consumer).
      - mode == 'GMN': the original CCM interaction matrix (data-driven
        E/tau grid search over the REGION_CHANNEL_CAP-limited observable
        pool, via apps.InteractionMatrix), then a node-dimension estimate
        and apps.CreateNetwork with its driver-ranking-bug workaround --
        unchanged, since there's no MDE-discovered graph to substitute
        and CreateNetwork's own construction is still needed.

    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network() for
    the mode == 'GMN' branch.

    interaction_matrix_outer_cores is a *separate* knob from `cores`:
    apps.InteractionMatrix's -cr sizes an outer pool across column pairs,
    and its internal pyEDM.CCM() call used to spawn its *own* fresh pool
    per pair sized to cpu_count() (confirmed by reading pyEDM/CCM.py and
    apps/InteractionMatrix.py, neither of which passes `parallel=`).
    Peak concurrent processes was -cr * cpu_count(), not -cr -- setting
    -cr = cores (the natural-seeming choice) squared your core request
    into a process count (-cr=cores on a 10-core machine produced ~90
    live processes). This module now monkeypatches pyEDM.CCM's inner
    pool away entirely (`_sequential_ccm_workers` near the top of this
    file) rather than just avoiding the multiplication -- every CCM()
    call here only ever resolves to 4-8 tiny tasks, so the inner pool was
    pure overhead. interaction_matrix_outer_cores stays a separate knob
    regardless, since it's still what sizes real parallelism across
    column pairs.
    '''
    network_id = '_'.join(targets)
    # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
    # artifacts for the same targets -- these are genuinely different
    # networks (full raw channel pool vs. nested-MDE-selected columns)
    # and must not share a checkpoint file (see the search function's
    # checkpoint resume logic, which only validates e_vals/tau_vals/
    # theta_vals, not network topology).
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    ccm_E = ccm_tau = None
    node_dimensions = {}

    if mode == 'MDE-GMN':
        graph, iMatrix, _ = build_nested_mde_network(
            targets, df, observable_columns, prediction_start, prediction_length,
            nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
            results_dir, mde_results,
        )
        selected_union = [n for n in graph.nodes if n not in targets]
        interaction_prefix = results_dir / f'iMatrix_nestedMDE_N{nested_n}_M{nested_m}_{network_id}'
        iMatrix.to_csv(f'{interaction_prefix}_MDE.csv')
    else:  # mode == 'GMN'
        selected_union = []
        for t in targets:
            for c in mde_results[t]['SELECTED_COLUMNS']:
                if c not in selected_union:
                    selected_union.append(c)

    data_after_mde = (
        df[targets + selected_union]
        .iloc[:prediction_start + prediction_length]
        .reset_index(drop=True)
    )
    data_after_mde.insert(0, 'Time', np.arange(len(data_after_mde)))
    data_after_mde_file = results_dir / f'data_after_mde_{network_id}.csv'
    data_after_mde.to_csv(data_after_mde_file, index=False)

    data_after_mde_lib = data_after_mde.iloc[:prediction_start]
    data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
    data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)

    if mode == 'MDE-GMN':
        # build_nested_mde_network()'s graph already IS the network --
        # see this function's docstring for why apps.CreateNetwork() adds
        # nothing here. 'Map' is written only because CreateNetwork's own
        # pickle format includes it and downstream code below prints from
        # it; gmn itself never reads it (see docstring).
        network_dict = {node: list(graph.predecessors(node)) for node in graph.nodes}
        network = {'Graph': graph, 'Map': network_dict}
        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
        with open(network_file, 'wb') as f:
            pickle.dump(network, f)
    else:  # mode == 'GMN'
        e_vals, tau_vals = list(e_range), list(tau_range)
        target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
        mean_grid = np.nanmean(np.stack(target_grids), axis=0)
        best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
        ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
        print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
              f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)

        interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
        # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
        # invoked via subprocess instead of the notebook's IPython
        # get_ipython().system(), which only exists inside a live kernel.
        # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
        subprocess_env = dict(os.environ, MPLBACKEND='Agg')
        subprocess.run(
            [
                'python', '-m', 'apps.InteractionMatrix',
                '-d', str(data_after_mde_lib_file),
                '-oc', str(interaction_prefix),
                '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
                '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
                '-P', '-cr', str(interaction_matrix_outer_cores),
            ],
            check=True, env=subprocess_env,
        )

        interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
        iMatrix = pd.read_csv(interaction_csv, index_col=0)

        node_dimensions = {
            node: estimate_node_dimension(data_after_mde_lib, node, cores,
                                           max_e=max_node_dimension, fallback=fallback_node_dimension)
            for node in iMatrix.index
        }
        print(f'{network_id}: node N_D =', node_dimensions, flush=True)

        # apps.CreateNetwork() keeps each node's *weakest* above-threshold
        # drivers instead of the strongest (confirmed against the installed
        # apps/CreateNetwork.py:110-116 -- it sorts each row ascending, then
        # truncates to numDrivers). Worked around by pre-selecting the
        # strongest N_D drivers ourselves, then calling CreateNetwork with
        # numDrivers set to the full column count so its buggy truncation
        # never has anything left to remove.
        strongest_iMatrix = iMatrix.copy()
        for node in strongest_iMatrix.index:
            row = strongest_iMatrix.loc[node].drop(node)
            n_drivers = max(1, min(node_dimensions[node], len(row)))
            keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
            drop = row.index.difference(keep)
            strongest_iMatrix.loc[node, drop] = threshold

        network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
        network = CreateNetwork(
            interactionMatrix = strongest_iMatrix,
            targetCols        = list(targets),
            threshold         = threshold,
            numDrivers        = len(strongest_iMatrix.columns),
            outputFile        = str(network_file),
            plotNetwork       = False,
        )

    for t in targets:
        print(f'{t} parents:', network['Map'][t], flush=True)

    return {
        'targets': list(targets),
        'network_id': network_id,
        'DATA_AFTER_MDE': data_after_mde_file,
        'NETWORK_FILE': network_file,
        'network': network,
        'ccm_E': ccm_E,
        'ccm_tau': ccm_tau,
        'node_dimensions': node_dimensions,
    }


def draw_colored_network(net_result, target_colors, save_path):
    graph = net_result['network']['Graph']
    targets = net_result['targets']

    node_colors = [target_colors.get(n, 'lightgray') for n in graph.nodes]
    edge_colors = [target_colors.get(v, 'gray') for _, v in graph.edges]

    pos = nx.arf_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx(
        graph, pos=pos, ax=ax, with_labels=True, arrowsize=10,
        node_size=300, node_color=node_colors, edge_color=edge_colors,
        width=1.2, font_size=9, font_weight='bold',
    )
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                   markerfacecolor=target_colors[t], label=t)
        for t in targets
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"GMN network ({net_result['network_id']})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN node generation config (mirrors notebook cell 8)
# =============================================================================

def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                      mode='Generate', function='SMap', E=3, tau=-3, theta=3,
                      node_theta_overrides=None, variant='', generate_length=None,
                      generate_start=None):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    generate_start (both modes): row index where Generate() starts
    free-running / where Forecast()'s pred window starts, independent of
    prediction_start -- the latter is the fixed MDE lib/pred boundary
    (see load_config()'s LIB_RATIO) that MDE variable selection and
    CCM/EmbedDimension validate against, and stays fixed regardless of
    what row a diagnostic Generate()/Forecast() call happens to start at.
    Defaults to prediction_start when None (unchanged pre-existing
    behavior). Sweeping this is how sweep_generate_start() finds the
    best-matching starting point for a fixed, already-chosen (E, tau,
    theta) -- see that function's docstring.

    generate_length (both modes): number of steps simulated/predicted
    from generate_start, independent of prediction_length -- the latter
    is also the MDE lib/pred split's held-out length, which drives MDE
    variable selection, CCM/EmbedDimension calls, and the search's own
    per-candidate theta retuning, none of which have anything to do with
    how many steps one Generate()/Forecast() call should cover. A full
    Generate() call over the whole held-out window can be the single most
    expensive step in the search loop (one call per (E, tau[, theta])
    candidate), so decoupling this lets you shrink it
    (MDE_GMN_GENERATE_LENGTH) without also shrinking the held-out window
    MDE/CCM validate against. Defaults to prediction_length when None.
    mode='Forecast' honors this same cap (rather than always spanning the
    full held-out prediction_length) so its window matches Generate()'s --
    run_forecast_diagnostic() plots both on the same axes, and a
    Forecast() line running past where Generate() stopped would have
    nothing to visually compare against.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting (confirmed
    by reading gmn/Generate.py directly), so it can't do real Takens
    delay embedding: embedded=True forces pyEDM to overwrite E with the
    raw predecessor-column count, silently discarding tau. gmn's SMap
    branch does call Embed(E, tau, columns) first -- a genuine delay
    embedding. Per explicit direction, this pipeline uses SMap uniformly
    for every node rather than mixing generators per node or patching
    Simplex. Consequence: every node (including leaf/undriven ones)
    needs E >= 2, or SMap's own predictor-column-exclusion convention
    leaves it with zero predictors.

    node_theta_overrides: optional {nodeName: theta} dict -- written as
    per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
    lookup); every other node uses the network-wide `theta`. The trailing
    '/' on configPath below works around gmn/Node.py:78's naive string
    concatenation path lookup (confirmed empirically: without it, every
    override is silently ignored).

    Forecast mode's lib/pred offset: gmn/Forecast.py's SMap branch calls
    Embed(E, tau, columns) then dropna() before predicting, which drops
    the leading (E-1)*abs(tau) NaN rows the time-shift introduces --
    shrinking the dataframe SMap actually sees relative to node.data's
    full row count, which is what `generate_start`/`forecast_length`
    below are expressed in. Since every node uses SMap and E >= 2 always
    (see docstring above), this offset is never zero, and an unadjusted
    lib/pred computed in original-row terms overflows the shrunk frame
    (confirmed empirically: pyEDM.SMap's CreateIndices() raises
    "prediction index N exceeds the number of data rows M" with M
    matching forecast_length - offset exactly). gmn/Generate.py's own
    SMap branch sidesteps this by building a fresh 0-based time index
    after its own equivalent offset slice; gmn/Forecast.py instead reuses
    the node's original (now index-shifted) time column, so the fix here
    is to shift our own explicit lib/pred bounds back by the same offset
    instead.'''
    if mode not in ('Generate', 'Forecast'):
        raise ValueError(f"Unknown mode: {mode!r}")
    Tp = 1
    generate_start = generate_start if generate_start is not None else prediction_start

    if mode == 'Generate':
        gmn_prediction_start = str(generate_start)
        gmn_prediction_length = str(generate_length if generate_length is not None else prediction_length)
        lib_range = ''
        pred_range = ''
    else:
        gmn_prediction_start = '1'
        gmn_prediction_length = '1'
        offset = (E - 1) * abs(tau)
        # Capped to generate_length (same window Generate() free-runs
        # for), not the full held-out prediction_length -- see this
        # function's docstring for why the two modes' windows need to
        # match for run_forecast_diagnostic()'s combined plot to be
        # meaningful.
        forecast_length = generate_length if generate_length is not None else prediction_length
        if offset >= generate_start or offset >= forecast_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'(E={E}, tau={tau}) leaves too few rows in generate_start='
                f'{generate_start}/forecast_length={forecast_length} for '
                f'a valid lib/pred split after SMap\'s Embed()+dropna() shrinks '
                f'the frame -- shrink E/|tau| or grow the prediction window.'
            )
        lib_range = f'1 {generate_start - offset}'
        pred_range = f'{generate_start - offset} {generate_start + forecast_length - Tp - offset}'

    node_cfg_dir = results_dir / f"node_cfgs_{net_result['network_id']}_{variant or mode.lower()}"

    def make_sections(node_theta):
        return {
            'GMN': {
                'mode': mode,
                'predictionStart': gmn_prediction_start,
                'predictionLength': gmn_prediction_length,
                'outPath': str(results_dir),
                'dataOutFile': '',
                'showPlot': 'False',
                'plotType': 'state',
                'plotColumns': '',
                'plotFile': '',
            },
            'Network': {
                'name': 'GMN',
                'targetNode': ','.join(net_result['targets']),
                'file': str(net_result['NETWORK_FILE']),
                'data': str(net_result['DATA_AFTER_MDE']),
            },
            'Node': {
                'info': '',
                'function': function,
                'configPath': str(node_cfg_dir) + '/' if node_theta_overrides else '',
                'data': '',
            },
            'EDM': {
                'lib': lib_range,
                'pred': pred_range,
                'E': str(E),
                'Tp': str(Tp),
                'knn': '0',
                'tau': str(tau),
                'theta': str(node_theta),
                'exclusionRadius': '0',
                'columns': '',
                'target': '',
                'solver': '',
                'embedded': 'True',
                'validLib': '',
                'generateSteps': '0',
                'libSizes': '',
                'sample': '0',
                'random': 'False',
                'includeData': 'False',
                'seed': '0',
            },
            'Scale': {'factor': '1', 'offset': '0'},
        }

    def write_config(sections, path):
        config = ConfigParser()
        for section, values in sections.items():
            config[section] = values
        with open(path, 'w') as f:
            config.write(f)

    suffix = '' if mode == 'Generate' else f'_{mode.lower()}'
    if variant:
        suffix += f'_{variant}'
    config_file = results_dir / f"gmn_network_{net_result['network_id']}{suffix}.cfg"
    write_config(make_sections(theta), config_file)

    if node_theta_overrides:
        node_cfg_dir.mkdir(parents=True, exist_ok=True)
        for node_name, node_theta in node_theta_overrides.items():
            write_config(make_sections(node_theta), node_cfg_dir / f'{node_name}.cfg')

    return config_file


# =============================================================================
# E/tau/theta search (mirrors notebook cell 10)
# =============================================================================

def _hash_file(path, chunk_size=1 << 20):
    '''sha256 of a file's actual bytes, not its name/size/mtime -- used to
    detect a same-named DATA_FILE whose content silently changed between
    job submissions, which name/size/mtime alone can miss (mtime survives
    a `cp`, size collisions are rare but real for CSVs of similar shape).
    Streamed in chunks since this is not IPC-bound, just avoiding loading
    an unnecessarily large file fully into memory at once.'''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def build_search_fingerprint(cfg, net_result, data_file_sha256):
    '''Everything that determines one network's (E, tau, theta) search
    outcome, for MDE_GMN_REUSE_BEST_PARAMS's cache-lookup below: the
    knobs that shape that network's topology (MODE/GMN_MODE/targets/
    MDE_D/NESTED_M/REGION_CHANNEL_CAP/the causality gates/LIB_RATIO), the
    knobs that shape the search itself (the E/tau/theta ranges,
    SMAP_SEARCH_MODE, GENERATE_LENGTH), a hash of the actual input data
    (see _hash_file), and the network's own realized node set as a cheap
    extra check -- if every upstream knob and the data genuinely match,
    the reconstructed network (already built by the time this runs, cost
    already paid) should have exactly the same nodes; disagreement here
    would mean some other nondeterminism crept in upstream that the knob
    list alone didn't catch, and is a signal to NOT trust the cache
    rather than silently reuse a stale search result.

    All values here are already JSON-safe (plain str/int/float/list) so
    the returned dict can be dumped/compared directly with `==` after a
    json round-trip, with no custom (de)serialization needed.'''
    graph = net_result['network']['Graph']
    return {
        'mode': cfg['MODE'],
        'gmn_mode': cfg['GMN_MODE'],
        'targets': sorted(net_result['targets']),
        'mde_d': cfg['MDE_D'],
        'embed_dim_rho_min': cfg['EMBED_DIM_RHO_MIN'],
        'cross_map_rho_min': cfg['CROSS_MAP_RHO_MIN'],
        'nested_m': cfg['NESTED_M'],
        'region_channel_cap': cfg['REGION_CHANNEL_CAP'],
        'lib_ratio': cfg['LIB_RATIO'],
        'generate_length': cfg['GENERATE_LENGTH'],
        'e_vals': list(cfg['E_RANGE']),
        'tau_vals': list(cfg['TAU_RANGE']),
        'theta_vals': list(cfg['THETA_RANGE']),
        'smap_search_mode': cfg['SMAP_SEARCH_MODE'],
        'data_file': cfg['DATA_FILE'],
        'data_file_sha256': data_file_sha256,
        'prediction_start': cfg['PREDICTION_START'],
        'prediction_length': cfg['PREDICTION_LENGTH'],
        'network_nodes': sorted(graph.nodes),
    }


def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
    '''Scans prev_runs_dir (recursively, so it works whether prev_runs_dir
    is the results/ root -- one level of job_<id>/ subdirectories -- or
    already a single job's output dir) for best_params_<network_id>.json
    files, newest (by mtime) first, and returns the first one whose
    stored fingerprint exactly equals `fingerprint`. Returns None if
    prev_runs_dir doesn't exist yet (e.g. a project's very first run) or
    nothing matches -- both are ordinary "no cache hit" outcomes, not
    errors, so the caller falls through to running the real search.'''
    if not prev_runs_dir.exists():
        return None
    candidates = sorted(
        prev_runs_dir.glob(f'**/best_params_{network_id}.json'),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for path in candidates:
        try:
            with open(path) as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if entry.get('fingerprint') == fingerprint:
            return entry['best_E'], entry['best_tau'], entry['best_node_theta'], path
    return None


def save_best_params(results_dir, network_id, fingerprint, best_E, best_tau, best_node_theta, source):
    '''Writes this run's own best_params_<network_id>.json -- the artifact
    a future job's find_cached_best_params() looks for -- regardless of
    whether this run computed it fresh or itself reused a cache hit, so
    the provenance chain (source: 'computed' vs. 'reused_from:<path>')
    stays visible in whichever job's output this file ends up in.'''
    path = results_dir / f'best_params_{network_id}.json'
    with open(path, 'w') as f:
        json.dump({
            'fingerprint': fingerprint,
            'best_E': best_E,
            'best_tau': best_tau,
            'best_node_theta': best_node_theta,
            'source': source,
        }, f, indent=2)
    return path


def retune_theta_for_candidate(data_lib, graph, nodes, E, tau, theta_range, cores,
                                min_pred_rows=20):
    '''For a given (E, tau), finds each node's own best theta (S-map's
    nonlinearity/localization parameter -- see build_gmn_config()'s
    module-level context for what it controls) via PredictNonlinear on a
    70/30 lib/pred split of that node's own delay embedding. Theta is the
    one SMap parameter this pipeline lets vary per node rather than
    network-wide (see run_smap_search()'s docstring), since how nonlinear
    a variable's dynamics are is expected to differ node to node even at
    a shared embedding E/tau. Falls back to theta=3 (a reasonable
    generic default) whenever there's too little data for the split
    (fewer than min_pred_rows pred rows) or PredictNonlinear itself
    fails, rather than letting one bad node abort the whole candidate.'''
    best_theta_by_node = {}

    for node in nodes:
        columns = list(graph.predecessors(node)) + [node]
        embedded = Embed(dataFrame=data_lib, E=E, tau=tau, columns=columns)
        offset = (E - 1) * abs(tau)
        embedded = embedded.iloc[offset:].reset_index(drop=True)
        embedded.insert(0, 'Time', np.arange(len(embedded)))
        embedded[node] = data_lib[node].to_numpy()[offset:]

        n_emb = len(embedded)
        split = int(n_emb * 0.7)
        n_pred = n_emb - split
        if n_pred < min_pred_rows:
            best_theta_by_node[node] = 3
            continue

        lib_range = f'1 {split}'
        pred_range = f'{split + 1} {n_emb}'

        try:
            pn = PredictNonlinear(
                dataFrame=embedded, columns=list(embedded.columns[1:-1]), target=node,
                theta=list(theta_range), lib=lib_range, pred=pred_range,
                Tp=1, embedded=True, showPlot=False, numProcess=cores,
            )
            best_theta_by_node[node] = float(pn.loc[pn['rho'].idxmax(), 'theta'])
        except Exception:
            best_theta_by_node[node] = 3

    return best_theta_by_node


def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
                             E, tau, node_theta, variant, results_dir, cores,
                             generate_length=None, generate_start=None):
    '''Runs one Generate() call and scores it as mean rho, across
    `net_result['targets']` only (not every driver node), between the
    simulated and real held-out series -- this is "generative skill" as
    this pipeline defines it everywhere: the E/tau/theta search
    (run_smap_search) and the start-point sweep (sweep_generate_start)
    both call this same function, at the same objective, so a search
    over either axis is comparable to the other.'''
    generate_start = generate_start if generate_start is not None else prediction_start
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir,
        mode='Generate', E=E, tau=tau,
        node_theta_overrides=node_theta,
        variant=variant, generate_length=generate_length,
        generate_start=generate_start,
    )
    G = gmn.GMN( configFile = str(config_file), cores = cores )
    G.Generate()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    n_gen = len(generated)

    target_rhos = {}
    for target in net_result['targets']:
        actual_vals = actual_df[target].to_numpy()[generate_start: generate_start + n_gen]
        gen_vals = generated[target].to_numpy()
        target_rhos[target] = ComputeError(actual_vals, gen_vals)['rho']

    score = float(np.nanmean(list(target_rhos.values())))
    return score, target_rhos, G


def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
                     theta_range, coarse_theta_range, search_mode, results_dir, cores,
                     generate_length=None, reuse_best_params=False, prev_runs_dir=None,
                     fingerprints=None):
    '''Per network, grid-searches one network-wide (E, tau) over
    e_range x tau_range, scoring each candidate via evaluate_smap_candidate()
    (mean Generate() rho across targets). Two modes for theta:
      - 'coarse_retune' (recommended): retune_theta_for_candidate() finds
        a per-node theta at each (E, tau) via a coarse grid, then one
        Generate() call scores that candidate. len(e_range)*len(tau_range)
        Generate() calls total.
      - 'joint': also sweeps the full theta_range inside the loop, same
        theta for every node. len(e_range)*len(tau_range)*len(theta_range)
        Generate() calls -- far more expensive, deliberate opt-in only.
    Checkpoints score_grid to disk after every candidate so an interrupted
    job can resume instead of restarting; if reuse_best_params and
    fingerprints are given, a network whose fingerprint matches a past
    job's cached result skips this search entirely (see the module
    docstring's SEARCH REUSE section).'''
    smap_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        all_nodes = list(graph.nodes)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        e_vals, tau_vals = list(e_range), list(tau_range)
        theta_vals = list(theta_range) if search_mode == 'joint' else [None]
        n_candidates = len(e_vals) * len(tau_vals) * len(theta_vals)

        fingerprint = fingerprints[network_id] if fingerprints else None
        if reuse_best_params and fingerprint is not None:
            cached = find_cached_best_params(prev_runs_dir, network_id, fingerprint)
            if cached is not None:
                best_E, best_tau, best_node_theta, source = cached
                print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched {source} -- '
                      f'skipping the {n_candidates}-candidate search entirely '
                      f'(best_E={best_E}, best_tau={best_tau})', flush=True)
                smap_search_results[network_id] = {
                    'mode': search_mode, 'e_vals': e_vals, 'tau_vals': tau_vals,
                    'theta_vals': theta_vals, 'score_grid': None,
                    'best_E': best_E, 'best_tau': best_tau,
                    'best_node_theta': best_node_theta,
                    'reused_from': str(source),
                }
                save_best_params(results_dir, network_id, fingerprint, best_E, best_tau,
                                  best_node_theta, source=f'reused_from:{source}')
                continue
            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
                  f'found under {prev_runs_dir} -- running the full search.', flush=True)

        checkpoint_file = results_dir / f'smap_search_checkpoint_{network_id}_{search_mode}.pkl'
        score_grid = np.full((len(e_vals), len(tau_vals), len(theta_vals)), np.nan)
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            # generate_length is part of the validity check alongside
            # e_vals/tau_vals/theta_vals -- each cell's score was computed
            # against a specific Generate() step count, and a resumed run
            # with a different MDE_GMN_GENERATE_LENGTH would otherwise
            # silently mix scores from two different-length simulations
            # into the same grid.
            if (ckpt.get('e_vals') == e_vals and ckpt.get('tau_vals') == tau_vals
                    and ckpt.get('theta_vals') == theta_vals
                    and ckpt.get('generate_length') == generate_length):
                score_grid = ckpt['score_grid']
                print(f'{network_id}: resumed checkpoint, '
                      f'{int(np.sum(~np.isnan(score_grid)))}/{score_grid.size} candidates already done',
                      flush=True)

        print(f'{network_id}: {search_mode} search over {n_candidates} (E, tau) candidates '
              f'{"x " + str(len(theta_vals)) + " theta " if search_mode == "joint" else ""}'
              f'(mode={search_mode})', flush=True)

        start_t = time.monotonic()
        done_at_start = int(np.sum(~np.isnan(score_grid)))
        done = done_at_start

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                if np.all(~np.isnan(score_grid[i, j])):
                    continue

                if search_mode == 'coarse_retune':
                    node_theta_for_candidate = retune_theta_for_candidate(
                        data_after_mde_lib, graph, all_nodes, E, tau, coarse_theta_range, cores,
                    )

                for k, theta in enumerate(theta_vals):
                    if not np.isnan(score_grid[i, j, k]):
                        continue

                    if search_mode == 'joint':
                        node_theta = {n: theta for n in all_nodes}
                    else:
                        node_theta = node_theta_for_candidate

                    variant = (f'smap_search_{search_mode}_E{E}_tau{tau}'
                               + (f'_theta{theta}' if search_mode == 'joint' else ''))
                    try:
                        score, _, _ = evaluate_smap_candidate(
                            net_result, prediction_start, prediction_length,
                            E, tau, node_theta, variant, results_dir, cores,
                            generate_length=generate_length,
                        )
                    except Exception as exc:
                        # gmn.Generate raises RuntimeError on SMap divergence
                        # (blown-up recursive forecast, >1E15) -- an expected
                        # outcome for at least some (E, tau[, theta])
                        # candidates on a large, high-dimensional network, not
                        # a bug to crash the whole search over. Record this
                        # cell as failed (score_grid keeps its NaN default,
                        # same as target_E_tau_rho_grid's own except: pass
                        # elsewhere in this file) and move on to the next
                        # candidate instead of losing the rest of the search.
                        print(f'[PROGRESS] {network_id} candidate {done + 1}/{n_candidates} '
                              f'(E={E}, tau={tau}{f", theta={theta}" if search_mode == "joint" else ""}) '
                              f'FAILED: {exc!r} -- scoring as NaN', flush=True)
                        score = np.nan

                    score_grid[i, j, k] = score

                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'e_vals': e_vals, 'tau_vals': tau_vals,
                                     'theta_vals': theta_vals, 'generate_length': generate_length,
                                     'score_grid': score_grid}, f)

                    done += 1
                    elapsed = time.monotonic() - start_t
                    per_candidate = elapsed / max(1, done - done_at_start)
                    eta = per_candidate * (n_candidates - done)
                    theta_str = f', theta={theta}' if search_mode == 'joint' else ''
                    print(f'[PROGRESS] {network_id} candidate {done}/{n_candidates} '
                          f'(E={E}, tau={tau}{theta_str}) score={score:.4f} '
                          f'(elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s)', flush=True)

        smap_search_results[network_id] = {
            'mode': search_mode,
            'e_vals': e_vals, 'tau_vals': tau_vals, 'theta_vals': theta_vals,
            'score_grid': score_grid,
        }

        flat_grid = score_grid[:, :, 0] if search_mode != 'joint' else np.nanmax(score_grid, axis=2)
        best_i, best_j = np.unravel_index(np.nanargmax(flat_grid), flat_grid.shape)
        best_E, best_tau = e_vals[best_i], tau_vals[best_j]

        if search_mode == 'joint':
            best_k = int(np.nanargmax(score_grid[best_i, best_j]))
            best_theta_global = theta_vals[best_k]
            print(f'{network_id}: best E={best_E}, tau={best_tau}, theta={best_theta_global} '
                  f'(score={score_grid[best_i, best_j, best_k]:.4f})', flush=True)
            smap_search_results[network_id]['best_node_theta'] = {n: best_theta_global for n in all_nodes}
        else:
            best_node_theta = retune_theta_for_candidate(
                data_after_mde_lib, graph, all_nodes, best_E, best_tau, coarse_theta_range, cores,
            )
            print(f'{network_id}: best E={best_E}, tau={best_tau} '
                  f'(theta=coarse-retuned per node, score={flat_grid[best_i, best_j]:.4f})', flush=True)
            smap_search_results[network_id]['best_node_theta'] = best_node_theta

        smap_search_results[network_id]['best_E'] = best_E
        smap_search_results[network_id]['best_tau'] = best_tau

        if fingerprint is not None:
            save_best_params(results_dir, network_id, fingerprint, best_E, best_tau,
                              smap_search_results[network_id]['best_node_theta'], source='computed')

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(flat_grid, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'GMN E/tau search ({search_mode}) -- {network_id}\nmean Generate() rho across targets')
        fig.colorbar(im, ax=ax, label='mean target rho')
        rect = patches.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                  fill=False, edgecolor='red', linewidth=2.5)
        ax.add_patch(rect)
        fig.tight_layout()
        fig.savefig(results_dir / f'gmn_E_tau_search_{search_mode}_{network_id}.png', dpi=300)
        plt.close(fig)

    return smap_search_results


# =============================================================================
# Generate()/Forecast() start-point sweep
# =============================================================================

def sweep_generate_start(gmn_runs, smap_search_results, prediction_start, prediction_length,
                          results_dir, cores, generate_length, start_offsets):
    '''Finds, per network, which row Generate() should start free-running
    from to best match the real held-out data -- at the (E, tau,
    per-node theta) run_smap_search() already chose, not re-searched here.
    GMN's Generate() is a recursive nonlinear simulation, so (like any
    such system) how well it tracks reality can depend on which state it
    launches from, independent of how good E/tau/theta are; this sweep
    answers "given our best model, which starting point does it explain
    best" as a separate question from "what is our best model."

    One evaluate_smap_candidate() call per offset in `start_offsets`
    (each offset a row count added to prediction_start), scored on the
    exact same mean-rho-across-targets objective the E/tau/theta search
    uses -- see that function's docstring. Checkpoints to disk after
    every offset (same resume pattern as run_smap_search's checkpoint)
    since each candidate is its own full Generate() call and this
    shouldn't have to restart from scratch after an interrupted job.

    A single-offset `start_offsets` (the default -- see load_config()'s
    GENERATE_START_MIN/MAX_OFFSET) means the sweep is disabled: that one
    offset is returned directly with no Generate() call at all, since
    rebuild_at_best_params() is about to run Generate() at that exact
    (E, tau, theta, start) anyway -- running it again here first would
    just be a wasted duplicate call.

    Returns {network_id: best_generate_start_row}.'''
    best_starts = {}
    for network_id, run in gmn_runs.items():
        if len(start_offsets) == 1:
            best_starts[network_id] = prediction_start + start_offsets[0]
            continue

        net_result = run['net_result']
        search = smap_search_results[network_id]
        best_E, best_tau, node_theta = search['best_E'], search['best_tau'], search['best_node_theta']

        checkpoint_file = results_dir / f'generate_start_checkpoint_{network_id}.pkl'
        scores = {}
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if (ckpt.get('start_offsets') == start_offsets and ckpt.get('best_E') == best_E
                    and ckpt.get('best_tau') == best_tau and ckpt.get('generate_length') == generate_length):
                scores = ckpt['scores']
                print(f'{network_id}: resumed start-sweep checkpoint, '
                      f'{len(scores)}/{len(start_offsets)} offsets already done', flush=True)

        for offset in start_offsets:
            if offset in scores and not np.isnan(scores[offset]):
                continue

            generate_start = prediction_start + offset
            try:
                score, _, _ = evaluate_smap_candidate(
                    net_result, prediction_start, prediction_length,
                    best_E, best_tau, node_theta, f'startsweep_offset{offset}', results_dir, cores,
                    generate_length=generate_length, generate_start=generate_start,
                )
            except Exception as exc:
                # Same rationale as run_smap_search's own try/except: a
                # divergent SMap Generate() is an expected failure mode
                # for at least some starting points, not a bug to crash
                # the whole sweep over.
                print(f'{network_id}: start offset {offset} FAILED: {exc!r} -- scoring as NaN', flush=True)
                score = np.nan

            scores[offset] = score
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'start_offsets': start_offsets, 'best_E': best_E, 'best_tau': best_tau,
                             'generate_length': generate_length, 'scores': scores}, f)
            print(f'{network_id}: start offset {offset:5d} (row {generate_start}) score={score:.4f}', flush=True)

        valid_offsets = [o for o in start_offsets if not np.isnan(scores.get(o, np.nan))]
        best_offset = max(valid_offsets, key=lambda o: scores[o]) if valid_offsets else start_offsets[0]
        best_starts[network_id] = prediction_start + best_offset
        print(f'{network_id}: best generate start = row {best_starts[network_id]} '
              f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
        ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Start offset from PREDICTION_START (rows)')
        ax.set_ylabel('Generate() rho (mean over targets)')
        ax.set_title(f'Generate() start-point sweep -- {network_id}')
        fig.tight_layout()
        fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
        plt.close(fig)

    return best_starts


# =============================================================================
# Final rebuild + Forecast() diagnostic + fidelity metric
# (mirrors notebook cells 11-13)
# =============================================================================

def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
                            results_dir, cores, generate_length=None, generate_starts=None):
    '''Reruns Generate() at the winning (E, tau, per-node theta) found by
    run_smap_search(), and the winning start row found by
    sweep_generate_start() (generate_starts; defaults to prediction_start
    per network when None). Plotting is deliberately NOT done here -- it
    used to draw its own actual-vs-Generate() figure, but that was a
    strict subset of run_forecast_diagnostic()'s own actual-vs-
    Generate()-vs-Forecast() figure (same two of its three lines, same
    nodes), so the two have been merged into the one figure
    run_forecast_diagnostic() draws, and this function now only reruns
    Generate() (needed downstream by run_forecast_diagnostic,
    gmn_fidelity_rho, and the raw output CSV) and prints the per-node
    rho.'''
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        search = smap_search_results[network_id]
        best_E, best_tau = search['best_E'], search['best_tau']
        node_theta = search['best_node_theta']
        generate_start = generate_starts[network_id] if generate_starts else prediction_start

        config_file = build_gmn_config(
            net_result, prediction_start, prediction_length, results_dir,
            mode='Generate', E=best_E, tau=best_tau,
            node_theta_overrides=node_theta,
            variant='best', generate_length=generate_length, generate_start=generate_start,
        )

        G = gmn.GMN( configFile = str(config_file), cores = cores )
        G.Generate()

        run['G'] = G
        run['best_E'] = best_E
        run['best_tau'] = best_tau
        run['best_node_theta'] = node_theta
        run['generate_start'] = generate_start

        actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
        generated = G.DataOut.reset_index(drop=True)
        nodes = [c for c in generated.columns if c != 'Time']

        print(f"{network_id} Generate() rho, held-out window (start row {generate_start}):", flush=True)
        for node in nodes:
            # Sliced to len(generated), not prediction_length: once
            # generate_length can be shorter than the full held-out window
            # (the whole point of decoupling it), actual_df's slice must
            # match Generate()'s actual output length or ComputeError()
            # compares mismatched-length arrays.
            a = actual_df[node].to_numpy()[generate_start:generate_start + len(generated)]
            r = ComputeError(a, generated[node].to_numpy())['rho']
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)


def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             E, tau, node_theta, results_dir, cores, save_path,
                             generate_length=None, generate_start=None, nodes_per_figure=10):
    '''Runs the Forecast() diagnostic and draws the one actual-vs-
    Generate()-vs-Forecast() figure for this network (see
    rebuild_at_best_params()'s docstring for why there's only one figure
    here rather than a separate Generate()-only one).

    generate_start (defaults to prediction_start when None) anchors both
    Forecast()'s own pred window and the plotted window to the same row
    sweep_generate_start() picked for G's own Generate() run -- so all
    three lines on the figure (actual, Generate(), Forecast()) describe
    the same starting point, "to be consistent with the generative mode."

    generate_length caps both the plotted actual/lib window and (via
    build_gmn_config's own generate_length parameter) Forecast()'s own
    pred window to the same length Generate() free-ran for -- plotting
    actual data all the way to the end of the full held-out
    prediction_length, when neither Generate() nor Forecast() produced
    anything to compare against past generate_length, only made the
    figure longer with no comparison value.

    Split across ceil(len(nodes) / nodes_per_figure) separate files
    (save_path with a _partNN suffix inserted before its extension)
    rather than one figure with a subplot per node -- this network can
    have 100+ nodes (see build_nested_mde_network()'s docstring on
    network size), and a single figure that tall was slow to render and
    load for no benefit over several page-sized ones.'''
    generate_length = generate_length if generate_length is not None else prediction_length
    generate_start = generate_start if generate_start is not None else prediction_start

    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
        E=E, tau=tau, node_theta_overrides=node_theta, generate_length=generate_length,
        generate_start=generate_start,
    )

    Gf = gmn.GMN( configFile = str(config_file), cores = cores )
    Gf.Forecast()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = forecast_out['Time'].to_numpy().astype(int)
    nodes = [c for c in forecast_out.columns if c != 'Time']

    generate_out = G.DataOut.reset_index(drop=True)
    generate_time = np.arange(generate_start, generate_start + len(generate_out))
    plot_end = generate_start + generate_length

    rhos = {}
    n_parts = (len(nodes) + nodes_per_figure - 1) // nodes_per_figure
    for part in range(n_parts):
        part_nodes = nodes[part * nodes_per_figure:(part + 1) * nodes_per_figure]
        fig, axes = plt.subplots(len(part_nodes), 1, figsize=(11, 2.2 * len(part_nodes)), sharex=False)
        if len(part_nodes) == 1:
            axes = [axes]

        for ax, node in zip(axes, part_nodes):
            actual = actual_df[node].to_numpy()
            forecast_pred = forecast_out[node].to_numpy()
            rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']

            ax.plot(np.arange(plot_end), actual[:plot_end], color='tab:blue', lw=1, label='Lib/actual')
            if node in generate_out.columns:
                ax.plot(generate_time, generate_out[node].to_numpy(),
                        color='tab:red', lw=1, label='Generate()')
            ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
            ax.axvline(generate_start, color='gray', ls='--', lw=0.8)
            ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"Generate() vs Forecast() vs actual -- {net_result['network_id']} "
                     f"(nodes {part * nodes_per_figure + 1}-"
                     f"{part * nodes_per_figure + len(part_nodes)} of {len(nodes)})", fontsize=11)
        fig.tight_layout()
        part_path = save_path.with_name(f'{save_path.stem}_part{part + 1:02d}{save_path.suffix}')
        fig.savefig(part_path, dpi=300)
        plt.close(fig)

    return Gf, rhos


def gmn_fidelity_rho(net_result, G, generate_start, default_E=3, tau=-1):
    '''Fig-4e-style generative fidelity: per node, a fresh Simplex predicts
    the real actual series *from* the generated series (not the other way
    around, and not a direct rho(actual, generated) comparison) on a 50/50
    split of the generated window -- this tests whether the generated
    trajectory's own dynamics stay on the same manifold as reality, a
    stricter/different question than pointwise closeness. E defaults to
    node_dimensions (an EmbedDimension-based estimate from build_network(),
    when computed -- see that function's docstring on when it isn't).'''
    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[generate_start: generate_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            S = Simplex(
                dataFrame = pair_df, columns = f'{node}_gen', target = f'{node}_actual',
                E = E, tau = tau, Tp = 0,
                lib  = f'1 {split}', pred = f'{split + 1} {n_gen}',
                embedded = False, showPlot = False,
            )
            fidelity_rho[node] = ComputeError(S['Observations'], S['Predictions'])['rho']
        except Exception:
            fidelity_rho[node] = np.nan
    return fidelity_rho


def plot_fidelity(rho, network_id, save_path):
    order = sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(order)), [rho[n] for n in order], color='tab:green')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel('rho (Simplex prediction of actual from generated embedding)')
    ax.set_title(f'GMN generative fidelity (Fig. 4e-style) -- {network_id}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# MDE-only diagnostics (mirrors notebook cells 15, 17, 19 -- skipped when
# MODE == 'GMN', same guard the notebook uses)
# =============================================================================

def run_mde_evaluate(mde_results, targets, results_dir):
    for target in targets:
        r = mde_results[target]
        Eval = dx.Evaluate(
            dataFrame     = r['mde_df'],
            columns_range = [1, r['mde_df'].shape[1]],
            mde_columns   = r['SELECTED_COLUMNS'],
            predictVar    = target,
            library       = r['m'].args.lib,
            prediction    = r['m'].args.pred,
            Tp            = r['m'].args.Tp,
            tau           = r['m'].args.tau,
            components    = 5,
            dmap_k        = 15,
            dmap_epsilon  = 'bgh_generous',
            figsize       = (7, 6),
        )
        Eval.Run()
        Eval.Plot()
        plt.gcf().savefig(results_dir / f'mde_evaluate_{target}.png', dpi=300)
        plt.close('all')


def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        pi = PredictInterval(
            dataFrame = r['mde_df'],
            columns   = cols,
            target    = target,
            lib       = r['m'].args.lib,
            pred      = r['m'].args.pred,
            E         = len(cols),
            tau       = r['m'].args.tau,
            embedded  = True,
            maxTp     = max_tp,
            showPlot  = False,
        )
        ax.plot(pi['Tp'], pi['rho'], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel('Prediction skill (rho)')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir):
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                S = Simplex(
                    dataFrame = r['mde_df'], columns = target, target = target,
                    E = E, tau = tau, lib = r['m'].args.lib, pred = r['m'].args.pred,
                    embedded = False, showPlot = False,
                )
                err = ComputeError(S['Observations'], S['Predictions'])
                rho_grid[i, j] = err['rho']

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label='rho')

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================

def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")


def main():
    cfg = load_config()

    df = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Data shape:', df.shape, flush=True)

    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))
    cfg['PREDICTION_LENGTH'] = len(df) - cfg['PREDICTION_START']

    # Resolve GENERATE_LENGTH now that PREDICTION_LENGTH is known: 0
    # (unset) means "use the full held-out window", and anything
    # requesting more steps than there is real held-out data to score
    # against gets clamped down to it (see load_config()'s docstring).
    if cfg['GENERATE_LENGTH'] <= 0:
        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']
    elif cfg['GENERATE_LENGTH'] > cfg['PREDICTION_LENGTH']:
        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the held-out window "
              f"({cfg['PREDICTION_LENGTH']} rows); clamping to it.", flush=True)
        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']

    # Resolve the generate-start sweep's candidate offsets now that
    # PREDICTION_LENGTH/GENERATE_LENGTH are known: drop any offset that
    # would leave too little real data to score GENERATE_LENGTH steps
    # against (see load_config()'s docstring on GENERATE_START_MAX_OFFSET).
    max_feasible_offset = cfg['PREDICTION_LENGTH'] - cfg['GENERATE_LENGTH']
    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
                                cfg['GENERATE_START_STEP']))
    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
    if feasible_offsets != start_offsets:
        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]

    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    if cfg['MODE'] == 'MDE-GMN':
        for target in cfg['TARGETS']:
            mde_results[target] = run_mde_for_target(
                target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                cfg['CORES'], cfg['RESULTS_DIR'],
            )

        if cfg['RUN_MDE_DIAGNOSTICS']:
            plot_mde_rho_vs_dimension(
                mde_results, cfg['TARGETS'], cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
            )
            plot_3d_mde_projection(
                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df,
                cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['RESULTS_DIR'] / f"mde_3d_projection_{cfg['TARGETS'][0]}.png",
            )
    else:  # MODE == 'GMN'
        for target in cfg['TARGETS']:
            mde_results[target] = {'target': target, 'SELECTED_COLUMNS': list(obs_cols)}
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} observable columns "
              f"as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['INTERACTION_MATRIX_OUTER_CORES'],
            observable_columns=obs_cols, nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    fingerprints = None
    if cfg['REUSE_BEST_PARAMS']:
        data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE'])
        fingerprints = {
            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256)
            for network_id, run in gmn_runs.items()
        }

    smap_search_results = run_smap_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
        cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
        generate_length=cfg['GENERATE_LENGTH'],
        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        fingerprints=fingerprints,
    )

    generate_starts = sweep_generate_start(
        gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
    )

    rebuild_at_best_params(
        gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'],
        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
    )

    for network_id, run in gmn_runs.items():
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_E'], run['best_tau'], run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
        )
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
        for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
            print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics ------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        run_mde_evaluate(mde_results, cfg['TARGETS'], cfg['RESULTS_DIR'])
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'], cfg['RESULTS_DIR'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'])
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
