'''Batch/cluster port of notebooks/tsls-mde-gmn.ipynb, executed by
scripts/batch/tsls-mde-gmn.sbatch. Every parameter is read from a
TSLS_MDE_GMN_* environment variable (set by the sbatch script's
MODIFIABLE PARAMETERS block) rather than edited here directly.

This is scripts/src/simple-mde-gmn.py's own MDE+GMN pipeline (the
run_mde_for_target / build_network / build_gmn_config / run_smap_search /
rebuild_at_best_params / ... functions below are DUPLICATED from that
file verbatim, not imported -- see notebooks/tsls-mde-gmn.ipynb's own
design-decisions note and this pipeline's own commit for why duplication
was chosen over a shared module or a dynamic import of a hyphenated
filename: it keeps this script fully self-contained and keeps
simple-mde-gmn.py untouched, at the cost of the two copies needing to be
kept in sync by hand if either evolves later), fed by a TSLS conversion
stage instead of simple-mde-gmn.py's own plain CSV load:

  0. Load the raw 40 Hz spike-train file, TSLS-convert every candidate
     CA1/mPFC/OFC channel (external/TSLS/cleaned_helper_functions.py's
     compute_time_since_previous_spike), then burn-in trim: TSLS is
     undefined (NaN) before a channel's own first spike, and
     dimx.MDE/pyEDM (unlike cleaned_helper_functions.py's own
     proper_simplex/proper_mde, which mask NaN per-embedding internally)
     need one fully non-NaN table for the whole lib/pred range, so the
     window can only start once every included channel has fired at
     least once -- see notebooks/tsls-mde-gmn.ipynb's intro cell for the
     empirical evidence this requirement is based on (dimx.MDE was
     confirmed to silently mis-select variables, not raise, when handed
     NaN-laden candidate columns).
  1. MDE forward-selection (skipped entirely if MODE='GMN')
  2. GMN network construction: CCM interaction matrix (data-driven E/tau
     grid search) + per-node driver-count estimate (EmbedDimension) +
     apps.CreateNetwork, with a driver-ranking bug workaround
  3. GMN node generation: SMap uniformly for every node
  4. Empirical (E, tau, per-node theta) search maximizing free-running
     Generate() fidelity against the real held-out targets
  5. Final GMN rebuild at the winning parameters + Forecast() diagnostic
     + Fig-4e-style generative fidelity metric
  6. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
     univariate E/tau grid) -- skipped if MODE='GMN'

Differences from simple-mde-gmn.py, all specific to the TSLS front end:
  - cleaned_helper_functions.py is imported from the same directory as
    this script (the sbatch script copies it there alongside the data
    file), matching scripts/src/tsls_example.py's own convention.
  - lib_ratio is computed against the row count AFTER TSLS conversion +
    burn-in trim, not the raw file's row count -- the usable window is
    itself data-driven (depends on which channel fires slowest), not a
    constant.
  - REGION_CHANNEL_CAP here caps how many raw per-region channels get
    TSLS-converted and offered to MDE/GMN as candidates (before the
    _tsls suffix), the same semantics as simple-mde-gmn.py's own
    REGION_CHANNEL_CAP just applied one step earlier in the pipeline.

Everything else (differences from the notebook it mirrors) matches
simple-mde-gmn.py's own module docstring: subprocess.run instead of
get_ipython().system(), plt.close(fig) instead of plt.show(), one flat
RESULTS_DIR, and CORES threaded through every *_cores/numProcess
parameter. See that file's docstring for the KNOWN CAVEAT on
apps.InteractionMatrix's own process-pool nesting (unaffected by the
TSLS front end -- it is a property of the GMN network-construction step,
which this script duplicates unchanged).
'''

import os
import pickle
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd

import gmn
from gmn.Common import FunctionType
from apps.CreateNetwork import CreateNetwork
# NOTE: dimx (dx.MDE/dx.Evaluate), apps.InteractionMatrix, and pyEDM's
# Simplex/SMap/CCM/ComputeError/EmbedDimension/PredictInterval are no
# longer imported directly -- this pipeline now uses
# cleaned_helper_functions.py's proper_mde/proper_ccm/proper_simplex/
# proper_embed_dimension/proper_predict_interval throughout (see this
# module's docstring and load_config()'s METRIC docstring for why). Embed
# and PredictNonlinear remain: no TSLS-aware equivalent exists for
# gmn.Node.Forecast's own delay-embedding call or for the per-node theta
# search (both pyEDM.SMap-specific concepts with no analog in
# cleaned_helper_functions.py's own KDTree engine).
from pyEDM import Embed, PredictNonlinear

# cleaned_helper_functions.py isn't installed as a package -- the sbatch
# script copies it into this script's own directory (same convention as
# scripts/src/tsls_example.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaned_helper_functions as tsls  # noqa: E402 -- import after sys.path edit above

def _fixed_node_forecast(self):
    '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
    bugs found empirically in the installed package's SMap branch -- the
    only branch this pipeline exercises, since every node uses SMap
    uniformly (see build_gmn_config's docstring):

    (1) that module calls Embed() without ever importing pyEDM.Embed
    (confirmed: gmn/Forecast.py only does `from .Common import *`, while
    gmn/Generate.py's identical SMap branch correctly imports it) --
    raises NameError on the very first Forecast() call against any SMap
    node.

    (2) `val` is assigned as a 2-tuple (time, predictions) and then
    immediately checked via `val.max() > 1E15` for divergence -- tuples
    have no .max() method, so this raises AttributeError unconditionally,
    not just on genuine divergence (confirmed: this fires as soon as (1)
    is worked around, on ordinary non-diverging output). Fixed to check
    the predictions Series specifically (val[1]).

    Every other branch is copied verbatim from the installed
    gmn/Forecast.py (unused by this pipeline, since every node is SMap,
    but kept intact rather than trimmed so this patch doesn't silently
    change behavior for any future non-SMap use). Monkeypatched onto
    gmn.Node (== gmn.Node.Node -- gmn/__init__.py's `from .Node import
    Node` shadows the submodule name with the class in gmn's own
    namespace) rather than editing the installed package, to keep this
    script self-contained on other machines/the cluster.'''
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

    Ported from simple-mde-gmn.py -- this is the actual fix for the
    nested-pool blowup documented in that file's module docstring and in
    build_network()'s docstring below: every single pyEDM.CCM() call --
    one per channel pair in apps.InteractionMatrix, one per MDE candidate
    column in dimx.Run -- creates its *own* fresh `spawn`/`forkserver`
    Pool sized to cpu_count() (CCM.py's `_resolve_workers`, called from
    `Project()`), on top of whichever outer pool (apps.InteractionMatrix's
    ProcessPoolExecutor, dimx's CrossMapColumns Pool) is already
    dispatching that call. Neither apps.InteractionMatrix nor dimx.Run
    ever pass `parallel=` through to CCM(), so this can't be tuned away
    via their CLIs. This module duplicates simple-mde-gmn.py's GMN
    network-construction step (and dimx.MDE call) unchanged, so it is
    exposed to the identical bug -- confirmed empirically there as the
    root cause of jobs running ~2 days regardless of dataset size, MODE,
    or MDE_D, before this patch was added. Without it, this pipeline's
    much larger default row/channel count (see this module's docstring)
    makes hitting the same pathology *more* likely, not less.

    Forcing sequential execution here is strictly a win, not a
    parallelism/wall-time tradeoff: every CCM() call in this pipeline
    resolves to only 4-8 tiny tasks (2 directions x 2-4 libSizes -- see
    apps/InteractionMatrix.py's libSizes=[libMin,libMax] and dimx's
    default pLibSizes=[10,15,85,90]), each a KDTree build+query over a
    handful of embedding rows. Running those 4-8 tasks in-process is
    faster than the Pool spawn + full heavy-import (numpy/scipy/pandas/
    sklearn/pyEDM/gmn) teardown cost alone, before any actual work
    happens -- and the real parallelism across channel pairs/candidates
    already comes from the outer pool, which this doesn't touch.
    Monkeypatched (not `parallel=False` at each call site) because
    apps.InteractionMatrix and dimx.Run construct CCM() internally and
    don't expose that argument.'''
    return 1


import pyEDM.CCM as _pyedm_ccm_module
_pyedm_ccm_module._resolve_workers = _sequential_ccm_workers


# =============================================================================
# Configuration -- every value overridable via TSLS_MDE_GMN_* environment
# variables; see scripts/batch/tsls-mde-gmn.sbatch's MODIFIABLE
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
    # NOTE: as with simple-mde-gmn.py, the cluster deployment expects
    # data files copied flat into DATA_DIR by the sbatch script -- no
    # nested extraction-folder name.
    cfg['DATA_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('TSLS_MDE_GMN_DATA_FILE', 'J1620210605_epoch_8_freq_40_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- Overall pipeline shape ---------------------------------------
    cfg['MODE'] = os.environ.get('TSLS_MDE_GMN_MODE', 'MDE-GMN')  # 'MDE-GMN' | 'GMN'
    if cfg['MODE'] not in ('MDE-GMN', 'GMN'):
        raise ValueError(f"Unknown TSLS_MDE_GMN_MODE: {cfg['MODE']!r}")
    cfg['GMN_MODE'] = os.environ.get('TSLS_MDE_GMN_GMN_MODE', 'combined')  # 'combined' | 'separate'

    cfg['TARGETS'] = _env_list('TSLS_MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
    cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None

    # Lib/pred split, expressed as the fraction of the TSLS-converted,
    # burn-in-trimmed dataset used as library -- computed in main() once
    # that trimmed row count is known (see module docstring), not here.
    cfg['LIB_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_LIB_RATIO', 0.7))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    cfg['MDE_D'] = int(os.environ.get('TSLS_MDE_GMN_D', 20))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # Candidate pool: first N *raw* channels of each region, TSLS-converted
    # and offered to MDE/GMN as candidates. Default 999 (effectively
    # uncapped -- matches notebooks/tsls-mde-gmn.ipynb's own full-channel-
    # set decision); cap this down for a fast test job.
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('TSLS_MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- E/tau/theta search ranges -------------------------------------
    cfg['E_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_E_MIN', 3)),
        int(os.environ.get('TSLS_MDE_GMN_E_MAX', 16)),
        int(os.environ.get('TSLS_MDE_GMN_E_STEP', 2)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_TAU_MIN', -15)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_STEP', 2)),
    )
    cfg['THETA_RANGE'] = _env_float_list(
        'TSLS_MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    cfg['COARSE_THETA_RANGE'] = cfg['THETA_RANGE'][::3]
    cfg['SMAP_SEARCH_MODE'] = os.environ.get('TSLS_MDE_GMN_SMAP_SEARCH_MODE', 'coarse_retune')  # 'coarse_retune' | 'joint'

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('TSLS_MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('TSLS_MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS', True)

    # ---- Parallelism ----------------------------------------------------
    cfg['CORES'] = int(os.environ.get('TSLS_MDE_GMN_CORES', os.cpu_count() or 1))
    cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))

    # ---- TSLS-aware scoring (cleaned_helper_functions.py's proper_mde/
    # proper_ccm/proper_simplex/adaptive_event_metrics_from_tsls, replacing
    # dimx.MDE / apps.InteractionMatrix / pyEDM.EmbedDimension / plain
    # ComputeError rho everywhere in this pipeline) -----------------------
    # 'correlation' (plain Pearson rho, pyEDM's own default) | 'mae' |
    # 'f1' | 'gamma' (Kistler coincidence factor) -- see
    # cleaned_helper_functions.py's _score()/adaptive_event_metrics_from_tsls
    # docstrings. 'f1'/'gamma' score TSLS series by decoding both true and
    # predicted series back to discrete reset/spike events (via a
    # near-zero threshold calibrated from the true series itself) and
    # matching them within an adaptive per-event tolerance window -- this
    # is what actually answers "did we get the spike timing right" instead
    # of Pearson correlation over the whole sawtooth, which is dominated
    # by the smooth (uninformative) ramp segments between spikes.
    cfg['METRIC'] = os.environ.get('TSLS_MDE_GMN_METRIC', 'f1')
    # TARGETS (head_position_x/y) are continuous behavioral traces, not
    # TSLS-encoded spike series -- they never "reset" near zero at
    # discrete events, so 'f1'/'gamma' event-decoding is meaningless for
    # them (confirmed empirically: applying METRIC='f1' to a target's own
    # self-prediction produced NaN for every candidate, since
    # infer_near_zero_threshold_from_true_tsls/detect_reset_events_from_tsls
    # have nothing sensible to key off of on a signal with no genuine
    # reset structure). Always 'correlation' (or 'mae') for anything scored
    # against a TARGETS column; METRIC above is only for scoring
    # predictions of the TSLS-converted neuron columns themselves.
    cfg['TARGET_METRIC'] = os.environ.get('TSLS_MDE_GMN_TARGET_METRIC', 'correlation')
    cfg['GAMMA_DELTA'] = int(os.environ.get('TSLS_MDE_GMN_GAMMA_DELTA', 2))
    # Every CCM convergence gate in this pipeline (proper_mde's own CCM
    # step, and build_interaction_matrix() below) sweeps library sizes as
    # PERCENTAGES of however many rows it's handed. CCM_WINDOW_ROWS bounds
    # that row count -- CRITICAL for calibration, not just a speed knob:
    # confirmed empirically twice that too-large a window makes every
    # tested library size land in the already-converged tail of the real
    # curve, with the gate then (correctly) reporting "no rise" -- not
    # because the underlying relationship isn't real, but because it was
    # never shown the region where the rise happens.
    #   1) job 35513363's preserved iMatrix CSV: apps.InteractionMatrix's
    #      2-point delta gate (libMinFraction=0.05 of ~31k rows -> a
    #      "small" library of ~1567 rows) reported exactly 0 for
    #      virtually every pair.
    #   2) job 35525431: proper_mde's own 4-point slope gate at
    #      CCM_WINDOW_ROWS=3000 (a "small" library of 300 rows) rejected
    #      every candidate for both targets, despite the ranking stage
    #      finding clearly strong candidates (e.g. correlation=0.68).
    # Recalibrated by testing genuinely small (tens of rows) library
    # sizes directly against real data pairs from this dataset: a real,
    # gradual, directionally-asymmetric convergence curve appeared in the
    # ~15-400 row range (e.g. one real pair's forward direction rose
    # 0.075->0.200 smoothly across that range while its reverse direction
    # stayed flat/declining -- the textbook CCM causality signature). The
    # defaults below (CCM_WINDOW_ROWS=800, percentages landing on
    # ~16/24/40/64/96/160/240/400) reproduce that validated range. Keep
    # CCM_WINDOW_ROWS's smallest resulting libSize >= (max E in E_RANGE)+2
    # -- SMap/Simplex's own minimum embedding-plus-one-point requirement.
    cfg['CCM_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_CCM_WINDOW_ROWS', 800))
    cfg['CCM_LIB_SIZE_PCTS'] = tuple(
        _env_float_list('TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS', (2, 3, 5, 8, 12, 20, 30, 50))
    )
    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_MDE_GMN_CCM_SAMPLE', 20))
    cfg['CCM_SLOPE'] = float(os.environ.get('TSLS_MDE_GMN_CCM_SLOPE', 0.01))

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg


# =============================================================================
# TSLS conversion + burn-in trim (mirrors notebooks/tsls-mde-gmn.ipynb
# cells 3-4; see that notebook's intro cell for the full rationale)
# =============================================================================

REGION_PREFIXES = ('CA1_', 'mPFC_', 'OFC_')


def region_neuron_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]


def raw_neuron_columns(df, cap):
    return (
        region_neuron_columns(df, 'CA1_', cap)
        + region_neuron_columns(df, 'mPFC_', cap)
        + region_neuron_columns(df, 'OFC_', cap)
    )


def convert_to_tsls(df_raw, neuron_columns, targets, results_dir):
    '''Spike counts -> TSLS for every candidate channel; behavioral
    TARGETS columns pass through untouched (they are already continuous,
    not spike counts). Returns the full (not yet burn-in-trimmed)
    TSLS DataFrame and the list of its _tsls-suffixed observable columns.'''
    time_grid = df_raw['time_bin_center'].to_numpy()

    data = {'time_bin_center': time_grid}
    for t in targets:
        data[t] = df_raw[t].to_numpy()

    for i, neuron in enumerate(neuron_columns):
        spike_times = df_raw.loc[df_raw[neuron] > 0, 'time_bin_center'].to_numpy()
        data[f'{neuron}_tsls'] = tsls.compute_time_since_previous_spike(spike_times, time_grid)
        if (i + 1) % 100 == 0 or i + 1 == len(neuron_columns):
            print(f'  TSLS-converted {i + 1}/{len(neuron_columns)} channels', flush=True)

    tsls_df_full = pd.DataFrame(data)
    tsls_df_full.to_csv(results_dir / 'tsls_data_full.csv', index=False)
    observable_columns = [f'{n}_tsls' for n in neuron_columns]
    print(f'-> tsls_data_full.csv  shape={tsls_df_full.shape}', flush=True)
    return tsls_df_full, observable_columns


def burn_in_trim(tsls_df_full, observable_columns, results_dir):
    '''Drop leading rows until every observable_columns entry is non-NaN
    (see module docstring for why this is a correctness requirement for
    the dimx.MDE/pyEDM handoff, not just cleanliness). Raises if any
    candidate channel never fires at all in this window.'''
    first_valid_idx = tsls_df_full[observable_columns].notna().idxmax()
    never_valid = [c for c in observable_columns if not tsls_df_full[c].notna().any()]
    if never_valid:
        raise ValueError(
            f'{len(never_valid)} channel(s) never fire in this window, TSLS is '
            f'all-NaN: {never_valid} -- widen the window or drop these channels.'
        )

    burn_in_rows = int(first_valid_idx.max())
    slowest_channel = first_valid_idx.idxmax()
    df = tsls_df_full.iloc[burn_in_rows:].reset_index(drop=True)

    df.to_csv(results_dir / 'tsls_data_trimmed.csv', index=False)
    print(f'Burn-in: dropped first {burn_in_rows}/{len(tsls_df_full)} rows '
          f'({burn_in_rows / len(tsls_df_full):.1%}) -- slowest channel to fire: {slowest_channel}',
          flush=True)
    print(f'-> tsls_data_trimmed.csv  shape={df.shape}', flush=True)
    return df

def _ccm_window(df, lib_ratio, max_rows):
    '''Bound the row count fed into any percentage-based CCM libSize sweep
    to max_rows, preserving lib_ratio's proportion within that smaller
    window. See load_config()'s CCM_WINDOW_ROWS docstring for why this is
    necessary (not optional) for CCM's convergence gate to mean anything
    at this pipeline's row counts.'''
    n = min(len(df), max_rows)
    windowed = df.iloc[:n].reset_index(drop=True)
    lib_end = max(1, int(round(n * lib_ratio)))
    return windowed, lib_end


def _metric_for(column, targets, metric, target_metric):
    '''TARGETS (continuous behavioral traces) always score via
    target_metric ('correlation'); everything else (TSLS-converted neuron
    columns) uses metric ('f1' by default) -- see load_config()'s
    TARGET_METRIC docstring for why this split is required, not
    cosmetic.'''
    return target_metric if column in targets else metric


def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                        mde_d, embed_dim_rho_min, cross_map_rho_min, results_dir,
                        metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
                        ccm_sample, ccm_slope):
    '''Forward-selection for a single target variable via
    cleaned_helper_functions.proper_mde (replaces dimx.MDE -- see
    load_config()'s METRIC docstring for why: proper_mde accepts
    metric='f1'/'gamma', scoring TSLS series by decoded spike-timing
    match instead of plain Pearson rho over the whole sawtooth).

    metric vs. ccmMetric: proper_mde's `metric` scores the RANKING stage
    (predictions of `target`, which is one of TARGETS -- a continuous
    behavioral trace, not TSLS -- so this must be target_metric, never
    METRIC); `ccmMetric` scores the CCM gate (predictions of the
    CANDIDATE column, which IS a TSLS neuron series -- so this is METRIC).
    See load_config()'s TARGET_METRIC docstring and this file's
    _metric_for() -- proper_mde's own docstring anticipates exactly this
    split ("pick [ccmMetric] to match the nature of the candidate
    columns").

    NOTE: proper_mde has no multiprocessing of its own (confirmed against
    cleaned_helper_functions.py -- no Pool/multiprocessing usage anywhere
    in the file), unlike dimx.MDE which threaded `cores` through its own
    worker pool. This pipeline no longer has a way to parallelize the MDE
    forward-selection step; CCM_WINDOW_ROWS keeps each individual
    proper_ccm/Simplex call cheap (bounded row count) to compensate, but a
    very large REGION_CHANNEL_CAP candidate pool will still take
    proportionally longer wall-clock than the old dimx.MDE-based version
    did on a multi-core node.

    CCM gate (noCCM left at its default False, i.e. ENABLED): a job
    submitted with noCCM=True (dropping this gate entirely) produced zero
    selected drivers under one ranking configuration and non-zero drivers
    under another, in a way that was ultimately inconclusive about
    whether the gate itself was miscalibrated at this data's scale or
    correctly rejecting non-causal correlations -- see this project's own
    job-log/dev notes for 2026-07-30. The gate stays on by design (it is
    the only thing distinguishing "these two series are correlated" from
    "one appears to causally drive the other," which matters for a
    pipeline whose output is meant to represent validated causal
    structure); see CCM_WINDOW_ROWS and CCM_LIB_SIZE_PCTS below for the
    calibration knobs to reach for instead of disabling the gate.'''
    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]
    lib_ratio = prediction_start / (prediction_start + prediction_length)
    ccm_df, ccm_lib_end = _ccm_window(mde_df, lib_ratio, ccm_window_rows)

    result = tsls.proper_mde(
        ccm_df, target=target, columns=observable_columns, D=mde_d,
        lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
        tau=-1, Tp=1, E=3,
        pLibSizes=ccm_lib_size_pcts, sample=ccm_sample, ccmSlope=ccm_slope,
        crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
        metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
        verbose=True,
    )

    mde_out = pd.DataFrame({'rho': result['MDErho']})
    mde_out.to_csv(results_dir / f'mde_output_{target}.csv', index=False)

    selected_columns = list(result['MDEcolumns'])
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    return {
        'target': target,
        'mde_df': ccm_df,
        'MDEOut': mde_out,
        'SELECTED_COLUMNS': selected_columns,
        'LIB': [1, ccm_lib_end],
        'PRED': [ccm_lib_end + 1, len(ccm_df)],
        'TAU': -1,
    }


def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['MDEOut']
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
    # [start, stop] list form (1-offset, inclusive) -- both call sites
    # feed these into cleaned_helper_functions.py's proper_simplex/
    # proper_embed_dimension, which expect that form, not pyEDM's own
    # "start stop" string form (the two look similar but proper_simplex's
    # lib[0]/lib[1] indexing on a string silently does the wrong thing
    # and gets swallowed by the caller's try/except -- confirmed
    # empirically as the cause of an all-NaN E/tau grid in production).
    split = int(n_rows * train_frac)
    return [1, split], [split + 1, n_rows]


def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals, metric, gamma_delta):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                out = tsls.proper_simplex(
                    lib_df, columns=target, target=target,
                    E=E, tau=tau, lib=lib_range, pred=pred_range,
                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][metric]
            except Exception:
                pass
    return rho_grid


def estimate_node_dimension(lib_df, node, metric, gamma_delta, max_e=15, fallback=4):
    '''EmbedDimension-equivalent for TSLS-aware metrics via
    proper_embed_dimension (pyEDM's own EmbedDimension is Pearson-rho
    only, so it can't score 'f1'/'gamma').'''
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = tsls.proper_embed_dimension(
            lib_df, columns=node, target=node,
            maxE=max_e, lib=lib_range, pred=pred_range,
            metric=metric, gammaDelta=gamma_delta,
        )
        return int(ed.loc[ed[metric].idxmax(), 'E'])
    except Exception:
        return fallback


def _ccm_converged(mean_vals, direction, ccm_slope):
    '''Two-part convergence test mirroring proper_mde's own CCM gate
    (cleaned_helper_functions.py: endpoint improved AND fitted slope vs.
    normalized libSize position exceeds ccm_slope). Returns (passed, slope).'''
    if np.all(np.isnan(mean_vals)):
        return False, 0.0
    x = np.arange(len(mean_vals), dtype=float)
    x /= x[-1] if x[-1] > 0 else 1.0
    slope = float(np.polyfit(x, mean_vals, 1)[0]) if len(mean_vals) > 1 else 0.0
    endpoint_improved = (direction * (mean_vals[-1] - mean_vals[0])) > 0
    return (endpoint_improved and (direction * slope) > ccm_slope), slope


def build_interaction_matrix(data_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
                              gamma_delta, ccm_window_rows, ccm_lib_size_pcts, ccm_sample,
                              ccm_slope):
    '''NxN interaction matrix via cleaned_helper_functions.proper_ccm,
    replacing apps.InteractionMatrix's CLI subprocess.

    Per-pair metric selection via _metric_for(): `nodes` mixes TSLS-
    converted neuron columns with the (continuous, non-TSLS) TARGETS
    columns, so each proper_ccm call's target_measurement/
    columns_measurement must independently match whichever side (a or b)
    is actually being predicted in that direction -- using METRIC
    ('f1' by default) to score a TARGETS column produced NaN for every
    candidate in production (see load_config()'s TARGET_METRIC
    docstring).

    WHY (empty-network root cause, and why the gate stays on): both
    apps.InteractionMatrix's own CCM convergence gate (InteractFunc, line
    ~430: `CCM_XY = CCM_YX = 0` unless the large-libSize skill beats the
    small-libSize skill by > deltaCCM=0.05) and proper_mde's own 4-point
    slope gate reported ~zero interactions in production (job 35513363's
    preserved iMatrix CSV: 88/90 cells exactly 0.0; job 35525431: MDE
    selected zero drivers for either target) -- not because the gates
    are wrong to require convergence, but because both were fed
    percentage-of-N libSize sweeps where even the *small* end (hundreds
    to thousands of rows, at this pipeline's ~30-45k-row TSLS-converted
    scale) already sits in the flat/converged tail of the real curve.
    Confirmed directly: testing genuinely small (tens of rows) library
    sizes against real candidate pairs from this dataset showed a
    smooth, monotonic, directionally-asymmetric convergence curve in the
    ~15-400 row range (one real pair's forward direction rose
    0.075->0.200 while its reverse direction stayed flat/declining --
    the textbook CCM causality signature) -- i.e. the convergence
    requirement is doing real, correct work here; it just needs to be
    shown the right range of library sizes. See load_config()'s
    CCM_WINDOW_ROWS docstring for the calibrated defaults this function
    (and proper_mde's own CCM gate) now use, matching that validated
    range. An earlier version of this function dropped the convergence
    requirement entirely rather than recalibrating it -- reverted: for a
    pipeline whose output is meant to represent causally-validated
    structure, an uncalibrated gate is a bug to fix, not a check to
    remove.

    Matches apps.CreateNetwork()'s documented convention (confirmed
    empirically against a working simple-mde-gmn network's own iMatrix.csv,
    job 34172242): `iMatrix.loc[driven_node, driver_node]` = skill of
    predicting driven_node FROM driver_node's history. proper_ccm's c2t
    direction (`target` predicted FROM `columns`) maps directly onto that;
    calling it once per *unordered* pair yields both c2t and t2c, matching
    apps.InteractionMatrix's own upper-triangular optimization (its module
    docstring: "CCM returns X:Y and Y:X, also compute pairs"). Non-
    converged directions are left at 0 (the pre-filled default), which is
    what build_network()'s downstream threshold + per-node top-N-drivers
    selection already expects.'''
    windowed, _ = _ccm_window(data_lib, 0.5, ccm_window_rows)
    libSizes = sorted({max(10, int(len(windowed) * p / 100.0)) for p in ccm_lib_size_pcts})

    iMatrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            a_metric = _metric_for(a, targets, metric, target_metric)
            b_metric = _metric_for(b, targets, metric, target_metric)
            try:
                # target_measurement scores c2t (predictions of `target`=a);
                # columns_measurement scores t2c (predictions of `columns`=b)
                # -- each independently matched to whichever of a/b it predicts.
                c2t, t2c, _ = tsls.proper_ccm(
                    windowed, libSizes=libSizes, sample=ccm_sample,
                    tau=ccm_tau, Tp=1, E=ccm_E,
                    target=a, columns=b,
                    target_measurement=a_metric, columns_measurement=b_metric,
                    gammaDelta=gamma_delta,
                )
            except Exception:
                continue

            # c2t: skill of predicting `target`=a FROM `columns`=b's
            # embedding -> iMatrix.loc[a, b] (a driven by b), kept only if
            # this direction shows genuine convergence. t2c mirrors this
            # the other way -> iMatrix.loc[b, a].
            for series, driven, driver, node_metric in (
                (c2t, a, b, a_metric), (t2c, b, a, b_metric),
            ):
                mean_vals = np.nanmean(series, axis=0)
                direction = 1 if node_metric in ('correlation', 'f1', 'gamma') else -1
                passed, slope = _ccm_converged(mean_vals, direction, ccm_slope)
                if passed:
                    iMatrix.loc[driven, driver] = float(mean_vals[-1])

    for n in nodes:
        iMatrix.loc[n, n] = np.nan
    return iMatrix


def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, metric, target_metric, gamma_delta,
                   ccm_window_rows, ccm_lib_size_pcts, ccm_sample, ccm_slope,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15):
    '''Build one interaction matrix + strongest-driver network spanning
    `targets` and the union of their driver candidate columns
    (mde_results[t]['SELECTED_COLUMNS'] -- MDE-selected when mode ==
    'MDE-GMN', the full observable pool when mode == 'GMN'; this function
    itself has no mode-awareness beyond that one flag used for the
    network_id prefix below).

    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network(). See
    that cell's docstring for the CreateNetwork driver-ranking-bug
    workaround rationale (identical here). The interaction matrix itself
    is now built by build_interaction_matrix() (see its docstring) instead
    of apps.InteractionMatrix's subprocess -- fixes the empty-network bug
    documented there, and (as a side effect) removes the
    apps.InteractionMatrix/pyEDM.CCM nested-pool concern this docstring
    used to describe, since proper_ccm has no multiprocessing pool at all.
    '''
    network_id = '_'.join(targets)
    # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
    # artifacts for the same targets -- these are genuinely different
    # networks (full raw channel pool vs. MDE-selected columns) and must
    # not share a checkpoint file (see the search function's checkpoint
    # resume logic, which only validates e_vals/tau_vals/theta_vals, not
    # network topology).
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    selected_union = []
    for t in targets:
        for c in mde_results[t]['SELECTED_COLUMNS']:
            if c not in selected_union:
                selected_union.append(c)

    if not selected_union:
        print(f'{network_id}: WARNING -- no MDE-selected driver columns for any target; '
              f'the network below will only contain {targets} with no real neuron drivers.',
              flush=True)

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

    e_vals, tau_vals = list(e_range), list(tau_range)
    # target_E_tau_rho_grid always scores a TARGETS column's own
    # self-predictability -- always target_metric (continuous), never
    # METRIC (see load_config()'s TARGET_METRIC docstring).
    target_grids = [
        target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, target_metric, gamma_delta)
        for t in targets
    ]
    mean_grid = np.nanmean(np.stack(target_grids), axis=0)
    best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
    ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
    print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
          f'(grid-search {target_metric}={mean_grid[best_i, best_j]:.3f})', flush=True)

    nodes = [c for c in data_after_mde_lib.columns if c != 'Time']
    iMatrix = build_interaction_matrix(
        data_after_mde_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric, gamma_delta,
        ccm_window_rows, ccm_lib_size_pcts, ccm_sample, ccm_slope,
    )
    interaction_csv = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}_CCM.csv'
    iMatrix.to_csv(interaction_csv)

    node_dimensions = {
        node: estimate_node_dimension(
            data_after_mde_lib, node, _metric_for(node, targets, metric, target_metric),
            gamma_delta, max_e=max_node_dimension, fallback=fallback_node_dimension,
        )
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
                      node_theta_overrides=None, variant=''):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting (confirmed
    by reading gmn/Generate.py directly), so it can't do real Takens
    delay embedding: embedded=True forces pyEDM to overwrite E with the
    raw predecessor-column count, silently discarding tau. gmn's SMap
    branch does call Embed(E, tau, columns) first -- a genuine delay
    embedding. Per explicit direction, this pipeline uses SMap uniformly
    for every node rather than mixing generators per node or patching
    Simplex -- see notebooks/simple-mde-gmn.ipynb's build_gmn_config
    docstring for the full evidence trail. Consequence: every node
    (including leaf/undriven ones) needs E >= 2, or SMap's own
    predictor-column-exclusion convention leaves it with zero predictors.

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
    full row count, which is what `prediction_start`/`n_rows` below are
    expressed in. Since every node uses SMap and E >= 2 always (see
    docstring above), this offset is never zero, and an unadjusted
    lib/pred computed in original-row terms overflows the shrunk frame
    (confirmed empirically: pyEDM.SMap's CreateIndices() raises
    "prediction index N exceeds the number of data rows M" with M
    matching n_rows - offset exactly). gmn/Generate.py's own SMap branch
    sidesteps this by building a fresh 0-based time index after its own
    equivalent offset slice; gmn/Forecast.py instead reuses the node's
    original (now index-shifted) time column, so the fix here is to shift
    our own explicit lib/pred bounds back by the same offset instead.'''
    if mode not in ('Generate', 'Forecast'):
        raise ValueError(f"Unknown mode: {mode!r}")

    n_rows = prediction_start + prediction_length
    Tp = 1

    if mode == 'Generate':
        gmn_prediction_start = str(prediction_start)
        gmn_prediction_length = str(prediction_length)
        lib_range = ''
        pred_range = ''
    else:
        gmn_prediction_start = '1'
        gmn_prediction_length = '1'
        offset = (E - 1) * abs(tau)
        if offset >= prediction_start or offset >= prediction_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'(E={E}, tau={tau}) leaves too few rows in prediction_start='
                f'{prediction_start}/prediction_length={prediction_length} for '
                f'a valid lib/pred split after SMap\'s Embed()+dropna() shrinks '
                f'the frame -- shrink E/|tau| or grow the prediction window.'
            )
        lib_range = f'1 {prediction_start - offset}'
        pred_range = f'{prediction_start - offset} {n_rows - Tp - offset}'

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

def retune_theta_for_candidate(data_lib, graph, nodes, E, tau, theta_range, cores,
                                min_pred_rows=20):
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
                             metric, gamma_delta):
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir,
        mode='Generate', E=E, tau=tau,
        node_theta_overrides=node_theta,
        variant=variant,
    )
    G = gmn.GMN( configFile = str(config_file), cores = cores )
    G.Generate()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    n_gen = len(generated)

    # gmn.GMN().Generate() itself still runs pyEDM.SMap internally (no
    # TSLS-aware equivalent exists for that -- out of scope, see this
    # module's docstring); what changes here is only how we SCORE its
    # already-generated output against the real held-out targets.
    # tsls._score is cleaned_helper_functions.py's own metric dispatcher
    # (same one proper_simplex/proper_ccm/proper_mde call internally),
    # reused directly instead of re-implementing correlation/f1/gamma
    # switching here.
    target_rhos = {}
    for target in net_result['targets']:
        actual_vals = actual_df[target].to_numpy()[prediction_start: prediction_start + n_gen]
        gen_vals = generated[target].to_numpy()
        target_rhos[target] = tsls._score(actual_vals, gen_vals, metric, gammaDelta=gamma_delta)

    score = float(np.nanmean(list(target_rhos.values())))
    return score, target_rhos, G


def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
                     theta_range, coarse_theta_range, search_mode, results_dir, cores,
                     target_metric, gamma_delta):
    smap_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        all_nodes = list(graph.nodes)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        e_vals, tau_vals = list(e_range), list(tau_range)
        theta_vals = list(theta_range) if search_mode == 'joint' else [None]
        n_candidates = len(e_vals) * len(tau_vals) * len(theta_vals)

        checkpoint_file = results_dir / f'smap_search_checkpoint_{network_id}_{search_mode}.pkl'
        score_grid = np.full((len(e_vals), len(tau_vals), len(theta_vals)), np.nan)
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if (ckpt.get('e_vals') == e_vals and ckpt.get('tau_vals') == tau_vals
                    and ckpt.get('theta_vals') == theta_vals):
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
                    score, _, _ = evaluate_smap_candidate(
                        net_result, prediction_start, prediction_length,
                        E, tau, node_theta, variant, results_dir, cores,
                        target_metric, gamma_delta,
                    )
                    score_grid[i, j, k] = score

                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'e_vals': e_vals, 'tau_vals': tau_vals,
                                     'theta_vals': theta_vals, 'score_grid': score_grid}, f)

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
# Final rebuild + Forecast() diagnostic + fidelity metric
# (mirrors notebook cells 11-13)
# =============================================================================

def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
                            results_dir, cores, metric, target_metric, gamma_delta):
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        search = smap_search_results[network_id]
        best_E, best_tau = search['best_E'], search['best_tau']
        node_theta = search['best_node_theta']

        config_file = build_gmn_config(
            net_result, prediction_start, prediction_length, results_dir,
            mode='Generate', E=best_E, tau=best_tau,
            node_theta_overrides=node_theta,
            variant='best',
        )

        G = gmn.GMN( configFile = str(config_file), cores = cores )
        G.Generate()

        run['G'] = G
        run['best_E'] = best_E
        run['best_tau'] = best_tau
        run['best_node_theta'] = node_theta

        actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
        generated = G.DataOut.reset_index(drop=True)
        generate_time = np.arange(prediction_start, prediction_start + len(generated))
        n_rows = prediction_start + prediction_length
        nodes = [c for c in generated.columns if c != 'Time']

        fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
        if len(nodes) == 1:
            axes = [axes]

        for ax, node in zip(axes, nodes):
            ax.plot(np.arange(n_rows), actual_df[node].to_numpy(),
                    color='tab:blue', lw=1, label='Lib/actual')
            ax.plot(generate_time, generated[node].to_numpy(),
                    color='tab:red', lw=1.2, label=f'Generate() SMap E={best_E}, tau={best_tau}')
            ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
            ax.set_title(node, fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"GMN Generate() (SMap, {search['mode']} search) -- {network_id}", fontsize=11)
        fig.tight_layout()
        fig.savefig(results_dir / f'gmn_generate_{network_id}.png', dpi=300)
        plt.close(fig)

        print(f"{network_id} Generate() score, held-out window:", flush=True)
        for node in nodes:
            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
            a = actual_df[node].to_numpy()[prediction_start:n_rows]
            r = tsls._score(a, generated[node].to_numpy(), node_metric, gammaDelta=gamma_delta)
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} {node_metric}={r:7.3f}", flush=True)


def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             E, tau, node_theta, results_dir, cores, save_path,
                             metric, target_metric, gamma_delta):
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
        E=E, tau=tau, node_theta_overrides=node_theta,
    )

    Gf = gmn.GMN( configFile = str(config_file), cores = cores )
    Gf.Forecast()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = forecast_out['Time'].to_numpy().astype(int)
    nodes = [c for c in forecast_out.columns if c != 'Time']

    generate_out = G.DataOut.reset_index(drop=True)
    generate_time = np.arange(prediction_start, prediction_start + len(generate_out))
    n_rows = prediction_start + prediction_length

    rhos = {}
    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
    if len(nodes) == 1:
        axes = [axes]

    for ax, node in zip(axes, nodes):
        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
        actual = actual_df[node].to_numpy()
        forecast_pred = forecast_out[node].to_numpy()
        rhos[node] = tsls._score(actual[time_idx], forecast_pred, node_metric, gammaDelta=gamma_delta)

        ax.plot(np.arange(n_rows), actual, color='tab:blue', lw=1, label='Lib/actual')
        if node in generate_out.columns:
            ax.plot(generate_time, generate_out[node].to_numpy(),
                    color='tab:red', lw=1, label='Generate()')
        ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
        ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
        ax.set_title(f"{node}   Forecast {node_metric}={rhos[node]:.3f}", fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f"Forecast() vs Generate() vs actual -- {net_result['network_id']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return Gf, rhos


def gmn_fidelity_rho(net_result, G, prediction_start, metric, target_metric, gamma_delta,
                      default_E=3, tau=-1):
    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[prediction_start: prediction_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            out = tsls.proper_simplex(
                pair_df, columns=f'{node}_gen', target=f'{node}_actual',
                E=E, tau=tau, Tp=0,
                lib=[1, split], pred=[split + 1, n_gen],
                embedded=False, metric=node_metric, gammaDelta=gamma_delta,
            )
            fidelity_rho[node] = out['metrics'][node_metric]
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
# NOTE: the notebook cell 15 diagnostic (dx.Evaluate: MDE selection vs.
# PCA/diffusion-map baselines) has no cleaned_helper_functions.py
# equivalent -- proper_mde replaced dimx.MDE for selection itself (see
# run_mde_for_target), but dx.Evaluate is a standalone dimx tool with no
# TSLS-aware port, and it depended on the dx.MDE result object (r['m'])
# this pipeline no longer constructs. Dropped rather than kept partially
# working against a dimx.MDE object this pipeline doesn't build anymore.

def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir, metric, gamma_delta):
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        if not cols:
            print(f'{target}: no MDE-selected columns, skipping Tp-vs-rho sweep.', flush=True)
            continue
        pi = tsls.proper_predict_interval(
            r['mde_df'], columns=cols, target=target,
            lib=r['LIB'], pred=r['PRED'],
            E=len(cols), tau=r['TAU'],
            maxTp=max_tp, metric=metric, gammaDelta=gamma_delta,
        )
        ax.plot(pi['Tp'], pi[metric], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel(f'Prediction skill ({metric})')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir,
                               metric, gamma_delta):
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                out = tsls.proper_simplex(
                    r['mde_df'], columns=target, target=target,
                    E=E, tau=tau, lib=r['LIB'], pred=r['PRED'],
                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][metric]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label=metric)

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)

def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")



# =============================================================================
# Orchestration
# =============================================================================

def main():
    cfg = load_config()
    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    df_raw = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Raw data shape:', df_raw.shape, flush=True)

    neuron_columns = raw_neuron_columns(df_raw, cfg['REGION_CHANNEL_CAP'])
    print(f'{len(neuron_columns)} raw spike-count channels to TSLS-convert', flush=True)
    tsls_df_full, obs_cols = convert_to_tsls(df_raw, neuron_columns, cfg['TARGETS'], cfg['RESULTS_DIR'])
    df = burn_in_trim(tsls_df_full, obs_cols, cfg['RESULTS_DIR'])

    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))
    cfg['PREDICTION_LENGTH'] = len(df) - cfg['PREDICTION_START']
    print(f"Usable rows N={len(df)}; lib_ratio={cfg['LIB_RATIO']} -> "
          f"PREDICTION_START={cfg['PREDICTION_START']} (lib), "
          f"PREDICTION_LENGTH={cfg['PREDICTION_LENGTH']} (pred)", flush=True)

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    if cfg['MODE'] == 'MDE-GMN':
        for target in cfg['TARGETS']:
            mde_results[target] = run_mde_for_target(
                target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                cfg['RESULTS_DIR'],
                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
                cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
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
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} TSLS-converted observable "
              f"columns as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'],
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
            cfg['CCM_WINDOW_ROWS'], cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    smap_search_results = run_smap_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
        cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
        cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
    )

    rebuild_at_best_params(
        gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
    )

    for network_id, run in gmn_runs.items():
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_E'], run['best_tau'], run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_forecast_vs_generate_{network_id}.png',
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        )
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        print(f"{network_id} Forecast() score per node (weakest first):", flush=True)
        for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
            print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'],
                                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity {cfg["METRIC"]} (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics -------------------------------------------
    # run_mde_evaluate (dx.Evaluate: MDE vs PCA/diffusion-map) dropped --
    # see the "MDE-only diagnostics" section docstring above
    # run_univariate_e_tau_grid for why.
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        # Both score a TARGETS column's own predictability -- always
        # TARGET_METRIC (continuous), never METRIC (see load_config()'s
        # TARGET_METRIC docstring).
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
                             cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
                                   cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
