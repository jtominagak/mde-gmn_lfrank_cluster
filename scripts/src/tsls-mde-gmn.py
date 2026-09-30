'''Batch/cluster port of notebooks/tsls-mde-gmn.ipynb, executed by
scripts/batch/tsls-mde-gmn.sbatch. Every parameter is read from a
TSLS_MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
PARAMETERS block) rather than edited here directly.

This is scripts/src/simple-mde-gmn.py's own two-stage nested-MDE + GMN
pipeline, ported wholesale (not imported -- see that file's own module
docstring for why duplication over a shared module was chosen; same
reasoning applies here), fed by a TSLS conversion stage instead of
simple-mde-gmn.py's own plain CSV load, and scored throughout via
cleaned_helper_functions.py's TSLS-aware metrics instead of dimx/pyEDM/
apps.InteractionMatrix directly (see the METRIC/TARGET_METRIC docstring in
load_config() below for why TSLS-encoded spike-timing series need a
different scoring function than plain Pearson rho).

Pipeline steps (see simple-mde-gmn.py's own module docstring for the
narrative behind each one -- this docstring only covers what's specific to
the TSLS front end and scoring):

  0. Load the raw 40 Hz spike-train file, TSLS-convert every candidate
     CA1/mPFC/OFC channel (external/TSLS/cleaned_helper_functions.py's
     compute_time_since_previous_spike), then burn-in trim: TSLS is
     undefined (NaN) before a channel's own first spike, and gmn's own
     Generate()/Forecast() SMap calls (unlike cleaned_helper_functions.py's
     own proper_simplex/proper_mde, which mask NaN per-embedding internally)
     need one fully non-NaN table for the whole lib/pred range, so the
     window can only start once every included channel has fired at least
     once.
  1. MDE forward-selection: two-stage nested-MDE network construction
     (skipped if MODE='GMN') via cleaned_helper_functions.py's proper_mde
     in place of dimx.MDE -- see run_mde_for_target()'s docstring.
  2. Network construction -- two unrelated methods depending on MODE,
     identical branching to simple-mde-gmn.py's own build_network():
       - MODE='MDE-GMN': build_nested_mde_network(), two serial phases --
         Stage 1 (real-observable-only BFS, run to full completion) then
         Stage 2 (per-node lagged-embedding attachment from the CCM-pre-
         filtered observable universe, seeded with every Stage 1 node) --
         so a Stage 2 real-observable selection recurses into its own
         further Stage 2 round exactly like a Stage 1 one recurses into
         further Stage 1 rounds, instead of terminating as a leaf -- both
         stages call the TSLS-flavored run_mde_for_target() below instead
         of dimx.MDE. No CCM interaction matrix, no apps.InteractionMatrix,
         no apps.CreateNetwork -- the resulting MultiDiGraph IS the network
         handed to gmn.GMN directly.
       - MODE='GMN': the original CCM interaction matrix, now via
         cleaned_helper_functions.proper_ccm (build_interaction_matrix())
         instead of apps.InteractionMatrix's CLI subprocess -- see that
         function's own docstring for the empty-network bug this already
         fixed, and why proper_ccm has no multiprocessing-pool-nesting
         concern at all (no Pool usage in cleaned_helper_functions.py).
  3. Node generation config: SMap uniformly for every node, via this
     pipeline's own generalized per-driver-delay embedding -- ported
     unchanged from simple-mde-gmn.py (see the gmn.Node monkeypatches right
     below this docstring), since it operates purely on the network graph
     and gmn's own data structures, with no dependency on what the columns
     represent.
  4. Theta search: retunes each node's own S-map theta once per network,
     scored via cleaned_helper_functions.tsls._score(metric=TARGET_METRIC)
     instead of pyEDM's plain ComputeError rho -- E and every driver's delay
     come directly from step 2's network, never searched.
  5. Start-point sweep: at the winning theta, tries several rows for
     Generate()/Forecast() to launch from and keeps whichever matches the
     real data best.
  6. Final rebuild at the winning theta and start row, plus the Forecast()
     diagnostic and a Fig-4e-style generative fidelity metric.
  7. MDE-only diagnostics (Tp-vs-rho sweep, univariate E/tau grid) --
     skipped if MODE='GMN'. The PCA/diffusion-map Evaluate comparison
     (dx.Evaluate) simple-mde-gmn.py still has here has NO TSLS port and
     stays dropped, exactly as this file's own architecture has always
     established: dx.Evaluate depends on a live dimx.MDE result object this
     pipeline doesn't build (proper_mde replaces dimx.MDE for selection
     itself, but is not a drop-in for dx.Evaluate specifically).

Differences from simple-mde-gmn.py, all specific to the TSLS front end and
scoring (see load_config()'s own docstrings for the full rationale on each):
  - cleaned_helper_functions.py is imported from the same directory as this
    script (the sbatch script copies it there alongside the data file).
  - lib_ratio (LIB_RATIO/MDE_VALIDATION_RATIO's three-way split) is computed
    against the row count AFTER TSLS conversion + burn-in trim, not the raw
    file's row count -- the usable window is itself data-driven (depends on
    which channel fires slowest), not a constant.
  - REGION_CHANNEL_CAP here caps how many raw per-region channels get
    TSLS-converted and offered to MDE/GMN as candidates (before the _tsls
    suffix), the same semantics as simple-mde-gmn.py's own REGION_CHANNEL_CAP
    just applied one step earlier in the pipeline.
  - Every MDE/CCM/embedding-search call goes through cleaned_helper_
    functions.py's proper_mde/proper_ccm/proper_simplex/proper_embed_
    dimension/proper_predict_interval instead of dimx.MDE/apps.
    InteractionMatrix/pyEDM.EmbedDimension/ComputeError/PredictInterval --
    these accept TSLS-appropriate metrics ('f1'/'gamma' event-timing scores,
    not just Pearson rho) and mask NaN per-embedding internally.
    Consequently: MDE_GMN_EXCLUSION_RADIUS's dynamic 'auto' mode (simple-
    mde-gmn.py's per-dimx.Run()-internal-call recomputation) has no
    equivalent -- proper_mde and friends are single opaque calls taking one
    static exclusionRadius, not a staged sequence of internal calls to
    intercept. TSLS_MDE_GMN_EXCLUSION_RADIUS is a plain integer here, no
    'auto' choice. Likewise, proper_mde has no multiprocessing of its own
    (confirmed: no Pool usage anywhere in cleaned_helper_functions.py), so
    Stage 1/Stage 2's wave pools use *only* outer (per-node)
    ProcessPoolExecutor parallelism -- the inner cores-per-worker split that
    fed dimx.MDE's own `cores=` argument has nothing to feed into here, and
    is dropped. The same applies to MODE='GMN's interaction-matrix
    construction: proper_ccm has no pool either, so
    MDE_GMN_INTERACTION_MATRIX_OUTER_CORES has no TSLS equivalent and isn't
    exposed.
  - METRIC/TARGET_METRIC/GAMMA_DELTA (see load_config()) pick which of
    cleaned_helper_functions.py's scoring functions every rho-like value in
    this pipeline is computed with -- TARGETS (continuous behavioral
    traces) always via TARGET_METRIC ('correlation'), every TSLS-converted
    neuron column via METRIC ('f1' by default, decoding both series back to
    discrete spike/reset events and matching them within an adaptive
    per-event tolerance window, since plain correlation over the whole TSLS
    sawtooth is dominated by the uninformative ramp segments between
    spikes).
  - CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS/CCM_SAMPLE/CCM_SLOPE calibrate every
    CCM convergence gate in this pipeline (proper_mde's own gate,
    build_interaction_matrix()'s) to this pipeline's much larger row counts
    -- see load_config()'s own docstring for the two production failures
    (empty networks / zero selected drivers) this calibration fixed.
  - GMN_MODE='combined' (MODE='MDE-GMN' only): the top-level, per-TARGETS
    MDE selection itself (main()'s own MDE block) runs ONCE via
    cleaned_helper_functions.multi_proper_mde -- forward-selecting drivers
    that jointly predict every TARGETS entry at once, via
    run_multi_mde_for_targets() -- instead of GMN_MODE='separate's one
    independent proper_mde call per target (run_mde_for_target(), unchanged
    from before multi_proper_mde was wired in). MULTI_MDE_E/MULTI_MDE_
    TARGET_AGGREGATE (see load_config()) configure that joint call; both
    are no-ops under GMN_MODE='separate'. GMN_MODE already independently
    controlled whether network construction spans one shared graph or one
    graph per target (target_groups_for_mode()) -- this extends that same
    "combined" meaning to the MDE stage feeding it.

Everything else (differences from the notebook it mirrors) matches
simple-mde-gmn.py's own module docstring: subprocess.run instead of
get_ipython().system() (MODE='GMN' no longer needs this at all here --
proper_ccm has no CLI to shell out to), plt.close(fig) instead of
plt.show(), one flat RESULTS_DIR, and CORES threaded through every
*_cores/numProcess parameter that actually has one to feed.

SEARCH REUSE / caching architecture (TSLS_MDE_GMN_REUSE_BEST_PARAMS,
default off): ported unchanged from simple-mde-gmn.py -- see that file's own
module docstring's SEARCH REUSE section for the full rationale. Every
genuinely expensive stage (MDE forward-selection, network construction,
theta search, the generate-start sweep) is deterministic given the same
inputs, so a fingerprint match against a past job's cache skips that stage
entirely. TSLS_MDE_GMN_PREV_RUNS_DIR/TSLS_MDE_GMN_RUN_CACHE_DIR point at a
durable bucket-storage pool (see tsls-mde-gmn.sbatch's CACHE_ROOT/
RUN_CACHE_DIR comment), not RESULTS_DIR -- every save_json_to_cache()/
save_pickle_to_cache() call writes to RUN_CACHE_DIR *immediately*, right
after its stage finishes, so a stage's result survives even if this job
later crashes or times out.
'''

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gmn


# cleaned_helper_functions.py isn't installed as a package -- the sbatch
# script copies it into this script's own directory (same convention as
# scripts/src/tsls_example.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaned_helper_functions as tsls  # noqa: E402 -- import after sys.path edit above

# ---- this pipeline's own package (see mde_gmn/__init__.py for the
# module map and dependency direction) ----------------------------------
from mde_gmn.caching import (
    _hash_file, _load_pickle_cache_entry, build_code_fingerprint,
    build_network_fingerprint, build_search_fingerprint, build_start_sweep_fingerprint,
    find_cached, find_cached_best_params, save_best_params, save_json_to_cache,
    save_pickle_to_cache)
from mde_gmn.columns import DELAY_MARKER, derive_lag_set, effective_mde_dimension
from mde_gmn.diagnostics import (
    run_mde_evaluate, run_tp_vs_rho_sweep, run_univariate_e_tau_grid)
from mde_gmn.generate import (
    gmn_fidelity_rho, plot_fidelity, rebuild_at_best_params, run_forecast_diagnostic,
    sweep_generate_start)
from mde_gmn.gmn_build import build_gmn_config, build_network
from mde_gmn.gmn_embedding import (          # noqa: F401 -- patches gmn on import
    _embedding_spec_for, _generalized_embed, network_max_embedding_offset)
from mde_gmn.mde import (
    _failed_mde_result, _failed_multi_mde_result, ccm_prefilter_observables,
    plot_3d_mde_projection, plot_mde_rho_vs_dimension, run_mde_for_target,
    run_multi_mde_for_targets)
from mde_gmn.network import (
    _simulated_nodes, build_nested_mde_network, partition_network, report_partition)
from mde_gmn.plots import draw_colored_network, plot_3d_network
from mde_gmn.progress import _format_eta, _log_progress
from mde_gmn.theta import (
    evaluate_smap_candidate, proper_predict_nonlinear, retune_theta_for_all_nodes,
    run_theta_search)
from mde_gmn.tsls_data import (
    _ccm_window, _metric_for, _preview_window_rows, _resolve_target_ranking_window,
    burn_in_trim, convert_to_tsls, raw_neuron_columns)


# =============================================================================
# Configuration -- every value overridable via TSLS_MDE_GMN_* environment
# variables; see scripts/batch/tsls-mde-gmn.sbatch's MODIFIABLE PARAMETERS
# block for the canonical list with descriptions.
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


def _env_int_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [int(x.strip()) for x in val.split(',') if x.strip()]


def load_config():
    cfg = {}

    # ---- Data ---------------------------------------------------------
    # NOTE: as with simple-mde-gmn.py, the cluster deployment expects data
    # files copied flat into DATA_DIR by the sbatch script -- no nested
    # extraction-folder name.
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

    # Three-way temporal split of the full (TSLS-converted, burn-in-trimmed)
    # dataset -- ported from simple-mde-gmn.py's own load_config() docstring,
    # resolved by main() once the trimmed row count is known:
    #   [0, A)      MDE library    -- LIB_RATIO fraction of the trimmed dataset
    #   [A, B)      MDE validation -- proper_mde's own held-out `pred` window
    #               (CCM/embedDimRhoMin/crossMapRhoMin gates, both nested-MDE
    #               stages) -- MDE_VALIDATION_RATIO fraction of the
    #               *remainder* after A
    #   [B, end)    Generate/Forecast validation -- generate_start floats
    #               only within this segment (GENERATE_START_MIN/MAX/STEP_
    #               OFFSET below), for GENERATE_LENGTH steps
    # This exists so network construction (both nested-MDE stages) and the
    # reported Generate()/Forecast() fidelity never touch the same rows --
    # see run_mde_for_target()'s docstring for the mechanics and main()'s
    # window-resolution block for the actual arithmetic. Unlike
    # simple-mde-gmn.py, computed against the row count AFTER TSLS
    # conversion + burn-in trim (main()), not the raw file's row count --
    # see this module's own docstring.
    cfg['LIB_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_LIB_RATIO', 0.7))
    cfg['MDE_VALIDATION_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_MDE_VALIDATION_RATIO', 0.5))
    if not (0.0 < cfg['MDE_VALIDATION_RATIO'] < 1.0):
        raise ValueError(
            f"TSLS_MDE_GMN_MDE_VALIDATION_RATIO must be in (0, 1): {cfg['MDE_VALIDATION_RATIO']!r}"
        )

    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH
    # -- see simple-mde-gmn.py's own load_config() docstring for the full
    # rationale (identical here). 0 (default) means "use the full held-out
    # window"; a larger value is clamped down to it in main().
    cfg['GENERATE_LENGTH'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_LENGTH', 0))

    # Theiler-window / neighbor-exclusion radius for gmn's own recursive
    # Generate()/Forecast() nearest-neighbor search (SMap), passed through
    # as Parameters.exclusionRadius by build_gmn_config() -- a plain static
    # knob (confirmed against build_gmn_config()'s own docstring and every
    # consumption site in simple-mde-gmn.py), NOT the dynamic/auto
    # mechanism TSLS_MDE_GMN_EXCLUSION_RADIUS below uses -- these are two
    # independent EDM contexts (MDE's forward selection vs. GMN's own
    # Generate()/Forecast()) with their own separate embeddings, so there's
    # no single shared "auto" value that would make sense for both.
    # Deliberately not TSLS_MDE_GMN_-prefixed, matching simple-mde-gmn.py's
    # own GMN_EXCLUSION_RADIUS naming (kept visually distinct from the
    # unrelated MDE-side parameter).
    cfg['GMN_EXCLUSION_RADIUS'] = int(os.environ.get('GMN_EXCLUSION_RADIUS', 0))

    # Plotted x-axis range for the final Generate()-vs-Forecast()-vs-actual
    # diagnostic figure -- see run_forecast_diagnostic()'s plot_range_mode
    # docstring. Ported unchanged from simple-mde-gmn.py.
    cfg['FORECAST_PLOT_RANGE'] = os.environ.get('TSLS_MDE_GMN_FORECAST_PLOT_RANGE', 'full').strip().lower()
    if cfg['FORECAST_PLOT_RANGE'] not in ('full', 'windowed'):
        raise ValueError(
            f"TSLS_MDE_GMN_FORECAST_PLOT_RANGE must be 'full' or 'windowed', "
            f"got {cfg['FORECAST_PLOT_RANGE']!r}"
        )

    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
    # as offsets (rows) from GENERATE_VALIDATION_START -- see
    # simple-mde-gmn.py's own load_config() docstring for the full rationale
    # (identical here). Default MIN=0, MAX=1 tries only offset 0.
    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
    cfg['GENERATE_START_STEP'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_STEP', 50))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    # MDE_D counts distinct *real* observables, not delay-expanded candidate
    # columns -- reused as the per-round cap for both nested-MDE stages
    # (Stage 1: real-observable-only, so the multiplier is always 1;
    # Stage 2: lagged-only, multiplied by however many lag values
    # LAG_MODE/LAG_DENSITY/LAG_STEP below resolve to) -- see
    # effective_mde_dimension()'s docstring. Ported unchanged from
    # simple-mde-gmn.py.
    cfg['MDE_D'] = int(os.environ.get('TSLS_MDE_GMN_D', 20))
    # Prediction interval every proper_mde forward-selection call (both the
    # top-level per-target search and every nested-MDE per-node search)
    # ranks/validates every candidate driver against -- ported from
    # simple-mde-gmn.py's own TP parameter (proper_mde accepts Tp as a real
    # argument, confirmed), replacing this pipeline's own previous hardcoded
    # Tp=1.
    cfg['TP'] = int(os.environ.get('TSLS_MDE_GMN_TP', 1))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # CCM pre-filter (see ccm_prefilter_observables()'s docstring): a cheap
    # CCM(observable -> target) skill check run once per target group,
    # right before either nested-MDE stage, to shrink REGION_CHANNEL_CAP's
    # candidate pool down to only observables that cross-map at least one
    # target above this threshold -- cuts cost on both Stage 1 (real
    # observables) and Stage 2 (their lags, since Stage 2 draws from the
    # same filtered pool) for free. 0 disables the filter entirely (pool
    # passed through unchanged); >0 enables it as the rho cutoff.
    cfg['CCM_PREFILTER_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN', 0.2))
    # Static embedding dimension for the CCM pre-filter's own CCM calls --
    # its OWN parameter, deliberately NOT a reuse of MULTI_MDE_E below: this
    # pipeline's data is TSLS-converted from a 40 Hz recording vs.
    # simple-mde-gmn.py's 1 Hz input (40x more rows per unit of real time),
    # and both pipelines fix tau=-1 (one sample step), so one embedding
    # step here covers only 1/40th the real time one step covers in
    # simple-mde-gmn.py -- matching that in raw sample-count terms would
    # need an impractical E. See simple-mde-gmn.sbatch's own
    # CCM_PREFILTER_E comment for the full simple-mde-gmn-side reasoning.
    # 7 here is a deliberately modest compromise: the middle value of this
    # pipeline's own already-team-vetted E_RANGE search below (E_MIN=3/
    # E_MAX=13/E_STEP=4 -> 3,7,11), not an untested extrapolation, and
    # still leaves large margin against the CCM_WINDOW_ROWS/LIBSIZE_RATIOS
    # -driven smallest-libSize-vs-E feasibility ceiling.
    cfg['CCM_PREFILTER_E'] = int(os.environ.get('TSLS_MDE_GMN_CCM_PREFILTER_E', 7))
    # Library sizes for the CCM pre-filter, as fractions of the same
    # CCM_WINDOW_ROWS-bounded window (see that parameter's own docstring
    # below) every other CCM call in this pipeline already uses, rather
    # than absolute row counts -- that window's actual row count depends on
    # CCM_WINDOW_ROWS/the data itself. Converted to actual libSizes (deduped,
    # floored at 10 rows, same floor convention build_interaction_matrix()'s
    # own CCM_LIB_SIZE_PCTS conversion uses) once the window is known -- see
    # ccm_prefilter_observables()'s own docstring.
    cfg['LIBSIZE_RATIOS'] = _env_float_list(
        'TSLS_MDE_GMN_LIBSIZE_RATIOS', (0.05, 0.10, 0.20, 0.40, 0.6, 0.8, 0.9),
    )
    # Only consulted when GMN_MODE='combined' (see target_groups_for_mode()
    # and main()'s own MDE block): in that mode, the depth-0 (top-level,
    # per-TARGETS) MDE selection runs once via
    # cleaned_helper_functions.multi_proper_mde -- forward-selecting drivers
    # that jointly predict every TARGETS column at once -- instead of the
    # GMN_MODE='separate' behaviour of one independent
    # cleaned_helper_functions.proper_mde call per target.
    #
    # UPDATED: multi_proper_mde now DOES have a per-candidate auto-select
    # path (multi_proper_embed_dimension() -- a joint rho-vs-E sweep scored
    # via _simplex_multi_metric, the same per-target-then-aggregate
    # primitive the ranking stage already used), mirroring single-target
    # proper_mde's own E<=0 sentinel convention. main()'s own depth-0 seed
    # call always passes E=0 (auto-select unconditionally -- same
    # "hardcoded at the one call site" precedent as the single-target path,
    # see run_mde_for_target()'s own max_e docstring) and reuses
    # MULTI_MDE_E as the sweep's own ceiling (max_e) rather than a fixed
    # value -- the same "reuse an already-exposed budget rather than
    # introduce a second one" convention MDE_D already established for the
    # single-target path. MULTI_MDE_TARGET_AGGREGATE picks how per-target
    # scores are combined at all three stages now (ranking, the E-select
    # sweep, and the CCM gate): 'mean' avoids penalizing a driver that's
    # strong for one target and weak for another; 'min' only accepts
    # drivers that are simultaneously good for every target.
    cfg['MULTI_MDE_E'] = int(os.environ.get('TSLS_MDE_GMN_MULTI_MDE_E', 3))
    cfg['MULTI_MDE_TARGET_AGGREGATE'] = os.environ.get('TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE', 'mean')
    if cfg['MULTI_MDE_TARGET_AGGREGATE'] not in ('mean', 'min'):
        raise ValueError(
            f"TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE must be 'mean' or 'min': "
            f"{cfg['MULTI_MDE_TARGET_AGGREGATE']!r}"
        )
    # Theiler-window / neighbor-exclusion radius for every proper_mde/
    # proper_ccm/proper_simplex/proper_embed_dimension call this pipeline
    # makes for MDE forward-selection. UNLIKE simple-mde-gmn.py, this is a
    # PLAIN STATIC INTEGER ONLY -- no 'auto' dynamic mode. simple-mde-gmn.py's
    # 'auto' mode recomputes exclusionRadius fresh at every one of dimx.Run's
    # own internal CrossMapColumns/EmbedDimension/CCM calls (a staged
    # sequence of internal calls this pipeline's own forward-selection makes
    # as it grows a candidate embedding one column at a time); proper_mde
    # and friends are each a single opaque call taking one exclusionRadius
    # value for their own whole internal search, with no staged internal
    # calls to intercept the same way. See this module's own docstring for
    # the full explanation.
    cfg['EXCLUSION_RADIUS_VALUE'] = int(os.environ.get('TSLS_MDE_GMN_EXCLUSION_RADIUS', 0))
    # Nested-MDE network construction (MODE='MDE-GMN' only). Reuses MDE_D as
    # the per-node driver cap N at every nesting level (deliberately not a
    # separate parameter). NESTED_M is the number of *additional* nested-MDE
    # rounds beyond each target's own initial selection -- total graph depth
    # from each target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for
    # "uncapped". R13 (docs/review-actions-v1.md): confirmed this is a
    # WHOLE-NETWORK budget, shared jointly by Stage 1 and Stage 2 -- a
    # node's depth is its distance from the target regardless of which
    # stage discovered it (build_nested_mde_network()'s own `node_depth`),
    # not two independent per-stage counters. This was already true before
    # R13 (and before R12); R13 only confirmed and documented it, no
    # arithmetic changed. Ported unchanged from simple-mde-gmn.py -- see
    # build_nested_mde_network()'s docstring for the full algorithm.
    cfg['NESTED_M'] = int(os.environ.get('TSLS_MDE_GMN_NESTED_M', 1))
    # First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted
    # and offered as MDE candidates (or, when MODE='GMN', used directly as
    # the interaction-matrix's full node set). 999 is effectively uncapped
    # (matches notebooks/tsls-mde-gmn.ipynb's own full-channel-set decision).
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('TSLS_MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- Two-stage nested-MDE network construction (MODE='MDE-GMN' only) --
    # See build_nested_mde_network() for the algorithm: Stage 1 builds the
    # real-observable backbone recursively (NESTED_M rounds), then Stage 2
    # attaches lagged drivers as leaf nodes in a single non-recursive pass.
    # DELAY_SOURCE/LAG_MODE/LAG_DENSITY/LAG_STEP/DELAY_SET below define which
    # lags Stage 2 may choose from.
    #
    # TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES is retired (2026-09-30): Stage
    # 2 considers lagged columns only, so there is nothing for it to select
    # recursively. The env var is rejected rather than ignored, so an sbatch
    # still setting it fails loudly instead of silently doing something else.
    if 'TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES' in os.environ:
        raise ValueError(
            'TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES is retired: Stage 2 now considers '
            'lagged columns only and never recurses (see build_nested_mde_network). '
            'Remove it from the sbatch script.')

    # R14 (docs/review-actions-v1.md): a zero-driver leaf is partitioned out
    # of the simulated network as an "external input" (partition_network())
    # rather than getting the old unconditional self-history embedding.
    # EXTERNAL_INPUT_MODE picks what Generate()/Forecast() actually do with
    # such a node -- see _generalized_node_generate()'s own docstring for
    # the full rationale (this was an explicit ask-before-implementing
    # decision, not a default guessed at):
    #   'real_data' (default): the node is genuinely exogenous -- its
    #     Generate() call short-circuits to the real recorded value at each
    #     step (self.Network.data, already loaded in full), never running
    #     an SMap self-regression. The INTERIOR network still free-runs
    #     autonomously; only external inputs are fed from real data. This
    #     is what "external input" means in the standard control-theory
    #     sense, and is the reading docs/tsls-mde-gmn-pipeline.pdf's own
    #     future doc-rewrite pass should adopt for its Sec 1.1 claim.
    #   'self_simulate': keeps today's exact pre-R14 behavior for these
    #     nodes (unconditional self_delay_set embedding, genuinely self-
    #     predicted every step, never touching real data past the initial
    #     seed) -- R14's partition becomes reporting/visualization-only in
    #     this mode, with no behavior change from before R14.
    # Either way, a TARGET that itself selects zero drivers is NEVER
    # classified as an external input regardless of this setting --
    # partition_network() raises instead (feeding a target's own real
    # values back in as its "simulation" and then scoring fidelity against
    # those same real values would be circular, producing a spuriously
    # perfect score for a network that is actually a failure).
    cfg['EXTERNAL_INPUT_MODE'] = os.environ.get('TSLS_MDE_GMN_EXTERNAL_INPUT_MODE', 'real_data')
    if cfg['EXTERNAL_INPUT_MODE'] not in ('real_data', 'self_simulate'):
        raise ValueError(
            f"TSLS_MDE_GMN_EXTERNAL_INPUT_MODE must be 'real_data' or 'self_simulate': "
            f"{cfg['EXTERNAL_INPUT_MODE']!r}"
        )

    cfg['DELAY_SOURCE'] = os.environ.get('TSLS_MDE_GMN_DELAY_SOURCE', 'lag_mode').strip().lower()
    if cfg['DELAY_SOURCE'] not in ('explicit', 'lag_mode'):
        raise ValueError(f"TSLS_MDE_GMN_DELAY_SOURCE must be 'explicit' or 'lag_mode': {cfg['DELAY_SOURCE']!r}")

    cfg['LAG_MODE'] = os.environ.get('TSLS_MDE_GMN_LAG_MODE', 'generation_length_adjusted').strip().lower()
    if cfg['LAG_MODE'] not in ('generation_length_adjusted', 'every_lag_possible'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_MODE must be 'generation_length_adjusted' or "
                          f"'every_lag_possible': {cfg['LAG_MODE']!r}")

    cfg['LAG_DENSITY'] = os.environ.get('TSLS_MDE_GMN_LAG_DENSITY', 'strided').strip().lower()
    if cfg['LAG_DENSITY'] not in ('strided', 'dense'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_DENSITY must be 'strided' or 'dense': {cfg['LAG_DENSITY']!r}")
    cfg['LAG_STEP'] = int(os.environ.get('TSLS_MDE_GMN_LAG_STEP', 1))
    if cfg['LAG_STEP'] < 1:
        raise ValueError(f"TSLS_MDE_GMN_LAG_STEP must be >= 1: {cfg['LAG_STEP']!r}")

    cfg['DELAY_SET'] = _env_int_list('TSLS_MDE_GMN_DELAY_SET', [0, -1, -3, -5, -10])
    if any(d > 0 for d in cfg['DELAY_SET']):
        raise ValueError(f"TSLS_MDE_GMN_DELAY_SET must be <= 0 (no future data): {cfg['DELAY_SET']!r}")

    # ---- E/tau search range (MODE='GMN' only) --------------------------
    # Used exclusively by build_network()'s mode == 'GMN' branch (the
    # proper_ccm-based interaction-matrix E/tau grid search, via
    # target_E_tau_rho_grid()) -- MODE='MDE-GMN' no longer has a
    # network-wide E/tau at all. Also feeds run_theta_search()'s own theta
    # retuning range regardless of mode (THETA_RANGE below).
    cfg['E_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_E_MIN', 3)),
        int(os.environ.get('TSLS_MDE_GMN_E_MAX', 13)),
        int(os.environ.get('TSLS_MDE_GMN_E_STEP', 4)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_TAU_MIN', -9)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_STEP', 4)),
    )
    # Theta (SMap's nonlinearity/localization parameter) search range, used
    # by run_theta_search() for both MODE='MDE-GMN' and MODE='GMN' -- ported
    # unchanged from simple-mde-gmn.py. Replaces this pipeline's own
    # previous SMAP_SEARCH_MODE='coarse_retune'/'joint' combined (E, tau,
    # theta) grid search entirely (see run_theta_search()'s own docstring):
    # E and every driver's delay now come from the network itself (MODE=
    # 'MDE-GMN': nested-MDE; MODE='GMN': the E_RANGE/TAU_RANGE grid search
    # above), leaving theta as the only thing still searched per node.
    cfg['THETA_RANGE'] = _env_float_list(
        'TSLS_MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('TSLS_MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('TSLS_MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS', True)
    # RUN_MDE_EVALUATE: gates ONLY run_mde_evaluate() (R19's PCA/diffusion-
    # map baseline comparison) -- deliberately separate from
    # RUN_MDE_DIAGNOSTICS above, which still gates run_tp_vs_rho_sweep()/
    # run_univariate_e_tau_grid()/plot_3d_mde_projection() (unaffected;
    # verified under R7, not implicated in the finding below). Default
    # False: docs/review-findings.md's "diffusion-map crash and hang"
    # finding confirmed pydiffmap's own diffusion-map fit, on real TSLS
    # data, can both raise (ValueError: Input X contains infinity...,
    # reproduced in 1.8s at an 800-row scale) and hang indefinitely
    # (reproduced past 74 CPU-minutes with no completion, at a real
    # ~31,000-row scale) -- the mechanism for the hang specifically is NOT
    # diagnosed (see that finding's own "flag for the documentation pass"
    # note). run_mde_evaluate() itself still has its own exception guard
    # AND a wall-clock bound around each baseline independently (see that
    # function's own docstring) for whoever turns this back on, but the
    # toggle is what actually protects a run while it's off -- a guard is
    # only as good as remembering every place that could need one, so
    # skipping this by default until the mechanism is understood is the
    # deliberate choice here, not a workaround. R19 is optional tooling,
    # not part of the causal claim (docs/tsls-mde-gmn-pipeline.tex's own
    # sec:diagnostics linknote) -- skipping it is a legitimate
    # configuration. Logged once per run when off, so its absence is
    # never silent (see docs/review-findings.md's own §18.6 cross-
    # reference for why a swallowed diagnostic staying invisible is
    # exactly the failure mode to avoid repeating).
    cfg['RUN_MDE_EVALUATE'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_EVALUATE', False)

    # ---- Parallelism ----------------------------------------------------
    # NOTE: no TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES here -- unlike
    # simple-mde-gmn.py's own MODE='GMN' branch (still shells out to
    # apps.InteractionMatrix, whose own outer pool that knob sizes), this
    # pipeline's MODE='GMN' path uses proper_ccm (build_interaction_matrix()
    # below), which -- like proper_mde -- has no multiprocessing pool of its
    # own at all (no Pool usage anywhere in cleaned_helper_functions.py), so
    # there's nothing for a knob like this to size. See this module's own
    # docstring.
    cfg['CORES'] = int(os.environ.get('TSLS_MDE_GMN_CORES', os.cpu_count() or 1))

    # ---- TSLS-aware scoring (cleaned_helper_functions.py's proper_mde/
    # proper_ccm/proper_simplex/adaptive_event_metrics_from_tsls, replacing
    # dimx.MDE / apps.InteractionMatrix / pyEDM.EmbedDimension / plain
    # ComputeError rho everywhere in this pipeline that scores a TSLS-
    # converted column) -----------------------------------------------------
    # 'correlation' (plain Pearson rho, pyEDM's own default) | 'mae' |
    # 'f1' | 'gamma' (Kistler coincidence factor) -- see
    # cleaned_helper_functions.py's _score()/adaptive_event_metrics_from_tsls
    # docstrings. 'f1'/'gamma' score TSLS series by decoding both true and
    # predicted series back to discrete reset/spike events (via a near-zero
    # threshold calibrated from the true series itself) and matching them
    # within an adaptive per-event tolerance window -- this is what actually
    # answers "did we get the spike timing right" instead of Pearson
    # correlation over the whole sawtooth, which is dominated by the smooth
    # (uninformative) ramp segments between spikes.
    cfg['METRIC'] = os.environ.get('TSLS_MDE_GMN_METRIC', 'f1')
    # TARGETS (head_position_x/y) are continuous behavioral traces, not
    # TSLS-encoded spike series -- they never "reset" near zero at discrete
    # events, so 'f1'/'gamma' event-decoding is meaningless for them. Always
    # 'correlation' (or 'mae') for anything scored against a TARGETS column;
    # METRIC above is only for scoring predictions of the TSLS-converted
    # neuron columns themselves.
    cfg['TARGET_METRIC'] = os.environ.get('TSLS_MDE_GMN_TARGET_METRIC', 'correlation')
    cfg['GAMMA_DELTA'] = int(os.environ.get('TSLS_MDE_GMN_GAMMA_DELTA', 2))
    # MODE='GMN' only (build_interaction_matrix() below) -- sweeps library
    # sizes as PERCENTAGES of however many rows it's handed. CCM_WINDOW_ROWS
    # bounds that row count -- CRITICAL for calibration, not just a speed
    # knob: too large a window makes every tested library size land in the
    # already-converged tail of the real curve, with the gate then
    # (correctly) reporting "no rise" even though the underlying
    # relationship is real. Fixed two past production failures this way
    # (apps.InteractionMatrix reporting ~all-zero interactions; proper_mde
    # rejecting every candidate for both targets) -- recalibrated against
    # real small-libSize data from this dataset. Keep CCM_WINDOW_ROWS's
    # smallest resulting libSize >= (max E in E_RANGE)+2.
    #
    # MODE='MDE-GMN' (run_mde_for_target()/run_multi_mde_for_targets(),
    # the pipeline's own default) no longer uses this knob at all -- see
    # TARGET_RANKING_WINDOW_ROWS/MIN_PRED_EVENTS/MAX_WINDOW_ROWS/
    # CCM_LIB_SIZES_ROWS below, and docs/review-findings.md's own
    # window-sizing section for why: a single fixed window/percentage-based
    # library grid made f1-scored ranking degenerate for real neuron
    # targets whose own reset events are too sparse within any window small
    # enough to keep the CCM gate's own convergence curve meaningful.
    cfg['CCM_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_CCM_WINDOW_ROWS', 800))
    # R17 (docs/review-actions-v1.md), gated on R9's own calibration: R9's
    # window sweep measured, at this pipeline's CCM_WINDOW_ROWS=800 default
    # (whose own smallest percentage below, 2%, is 16 rows), that 35% of a
    # 54-pair stratified sample were still rising at 40 rows (the 5%
    # point) -- a real but NOT majority fraction (R9's own decision rule:
    # "most sampled pairs" would mean adopting the reviewer's coarser
    # [5,20,40,60,80] grid outright, which drops everything below 40 rows
    # entirely; 35% doesn't clear that bar). So: coarsen only the upper
    # end, keeping the small (16/24/40/64-row) percentages that still
    # carry real rising-curve signal for a meaningful minority of pairs,
    # and only dropping percentages from the already-converged tail
    # (12%, 30% removed; 20%/50% kept) -- see docs/review-findings.md's
    # R17 section for the measured compute/selection effect on real data.
    cfg['CCM_LIB_SIZE_PCTS'] = tuple(
        _env_float_list('TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS', (2, 3, 5, 8, 20, 50))
    )
    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_MDE_GMN_CCM_SAMPLE', 20))
    cfg['CCM_SLOPE'] = float(os.environ.get('TSLS_MDE_GMN_CCM_SLOPE', 0.01))

    # Window-sizing fix (docs/review-findings.md) -- MODE='MDE-GMN' only.
    # Replaces CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS for run_mde_for_target()/
    # run_multi_mde_for_targets()'s own ranking-stage/E-select window and
    # the CCM gate's own library sweep, decoupled from each other (see
    # _resolve_target_ranking_window()'s and proper_mde()'s own `libSizes`
    # docstrings for the full reasoning -- a single fixed window/percentage
    # grid made f1-scored ranking degenerate for real sparse-firing
    # neurons).
    #
    # TARGET_RANKING_WINDOW_ROWS: the depth-0 seed's own fixed
    # ranking/E-select window (behavioral targets have no reset-event
    # concept, so there's nothing to extend against) -- arbitrary, set to
    # 1000 for now; larger than the old CCM_WINDOW_ROWS=800 default, no
    # other rationale yet.
    cfg['TARGET_RANKING_WINDOW_ROWS'] = int(
        os.environ.get('TSLS_MDE_GMN_TARGET_RANKING_WINDOW_ROWS', 1000))
    # MIN_PRED_EVENTS: for every Stage 1/Stage 2 (neuron) node, its own
    # ranking window is extended -- never past the leakage boundary B --
    # until its own PRED portion (the last (1 - lib_ratio) fraction of ROWS
    # of the window, the part `f1`-scoring actually touches) contains at
    # least this many of that neuron's real reset events, counted strictly
    # within [0, B). A plain fixed integer, not a fraction of anything
    # neuron-specific (see docs/review-findings.md for why an earlier
    # "half of this neuron's own total events" design was abandoned: it
    # made a busy neuron face a stricter test than a quiet one for no
    # stated reason, and -- checked against the whole window rather than
    # its pred portion -- was found on real data to pass while the pred
    # portion itself still had too few events to score anything on).
    #
    # Calibrated (not guessed) via r9_ccm_window_e_calibration.py's own
    # MIN_PRED_EVENTS section: f1 between a real TSLS slice and a lightly
    # perturbed copy of itself, >=200 draws per point, stratified by
    # firing-rate tercile. Result: a smooth ~1/sqrt(N) std decay with NO
    # natural elbow (5->0.146, 10->0.104, 20->0.070, 40->0.050, 80->0.036,
    # 160->0.025, pooled) -- so 20 is a threshold choice justified by the
    # cost/reachability tradeoff below, not a discovered stabilisation
    # point. At 20, std~0.07 is still ~3x smaller than CROSS_MAP_RHO_MIN's
    # own 0.2 default, and real data (REGION_CHANNEL_CAP=15, this session)
    # showed only 3/45 channels unreachable within the leakage-safe range
    # (median 3,593 rows needed) -- 40 and 80 buy modestly less noise at a
    # steep cost in both unreachable channels (8/45, 20/45) and window
    # length (median ~10,400 rows either way).
    cfg['MIN_PRED_EVENTS'] = int(os.environ.get('TSLS_MDE_GMN_MIN_PRED_EVENTS', 20))
    # MAX_WINDOW_ROWS: a plain compute ceiling on the search above -- NOT
    # calibrated (unlike MIN_PRED_EVENTS): if a node can't reach
    # MIN_PRED_EVENTS within min(MAX_WINDOW_ROWS, B), it's skipped (logged,
    # with the shortfall) rather than scored on too few events.
    # MIN_PRED_EVENTS is the statistical floor; this is the compute
    # ceiling -- the two express one thing each. 30000 is comfortably
    # above the largest real row count MIN_PRED_EVENTS=20 was observed to
    # need (26,748) and safely below this dataset's own leakage boundary
    # (~38,067) -- a backstop, not a tuned value.
    cfg['MAX_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_MAX_WINDOW_ROWS', 30000))
    # CCM_LIB_SIZES_ROWS: explicit, absolute row counts (never a percentage
    # of the -- now node-specific, potentially much larger -- ranking
    # window above) for the CCM gate's own convergence-curve library sweep.
    # Starts from R9/R17's own empirically-calibrated percentage grid
    # (2,3,5,8,20,50% of the old 800-row CCM_WINDOW_ROWS = 16,24,40,64,160,
    # 400 rows), plus one additional point (800) for better plateau
    # visibility now that the surrounding frame can be much larger --
    # re-validated at both an 800-row and an 8000-row frame on real data
    # (docs/review-findings.md): the same absolute values showed the same
    # qualitative "rising, then plateauing" shape at both scales, since
    # that shape is a property of the absolute library size, not its ratio
    # to the frame.
    cfg['CCM_LIB_SIZES_ROWS'] = tuple(
        _env_int_list('TSLS_MDE_GMN_CCM_LIB_SIZES_ROWS', (16, 24, 40, 64, 160, 400, 800))
    )

    # ---- Search reuse ---------------------------------------------------
    # See simple-mde-gmn.py's own module docstring's SEARCH REUSE section
    # for the full rationale; ported unchanged. Short version: reuse a past
    # job's E/tau/theta search result instead of redoing it, when settings
    # match exactly.
    cfg['REUSE_BEST_PARAMS'] = _env_bool('TSLS_MDE_GMN_REUSE_BEST_PARAMS', False)
    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_PREV_RUNS_DIR', '../../results'))
    cfg['RUN_CACHE_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_RUN_CACHE_DIR', cfg['RESULTS_DIR']))
    cfg['RUN_CACHE_DIR'].mkdir(parents=True, exist_ok=True)

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg
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

    # Progress-monitoring log (see HEARTBEAT_SECONDS/_log_progress()'s own
    # docstrings): cfg['RESULTS_DIR'] is already on /flash for the whole
    # run's duration (TSLS_MDE_GMN_RESULTS_DIR -> $OUT_DIR in the sbatch
    # script, archived to bucket only at job end), so this needs no new
    # path plumbing. Opened in truncate mode -- each job already gets its
    # own fresh RESULTS_DIR, so there is no earlier content to preserve.
    progress_log_path = cfg['RESULTS_DIR'] / 'progress.log'
    open(progress_log_path, 'w').close()
    pipeline_start = time.perf_counter()
    _log_progress(progress_log_path, 'PIPELINE START')

    df_raw = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Raw data shape:', df_raw.shape, flush=True)

    neuron_columns = raw_neuron_columns(df_raw, cfg['REGION_CHANNEL_CAP'])
    print(f'{len(neuron_columns)} raw spike-count channels to TSLS-convert', flush=True)
    tsls_df_full, obs_cols = convert_to_tsls(df_raw, neuron_columns, cfg['TARGETS'], cfg['RESULTS_DIR'])
    df = burn_in_trim(tsls_df_full, obs_cols, cfg['RESULTS_DIR'])

    # Three-way A/B/end split -- see load_config()'s docstring for the full
    # rationale (identical to simple-mde-gmn.py, ported here), computed
    # against the TSLS-converted, burn-in-trimmed row count (len(df)), not
    # the raw file's row count.
    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))                          # A
    post_lib_remainder = len(df) - cfg['PREDICTION_START']
    cfg['PREDICTION_LENGTH'] = int(round(post_lib_remainder * cfg['MDE_VALIDATION_RATIO']))   # B - A
    cfg['GENERATE_VALIDATION_START'] = cfg['PREDICTION_START'] + cfg['PREDICTION_LENGTH']     # B
    cfg['GENERATE_VALIDATION_LENGTH'] = len(df) - cfg['GENERATE_VALIDATION_START']            # end - B

    if cfg['GENERATE_LENGTH'] <= 0:
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']
    elif cfg['GENERATE_LENGTH'] > cfg['GENERATE_VALIDATION_LENGTH']:
        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the Generate/Forecast "
              f"validation segment ({cfg['GENERATE_VALIDATION_LENGTH']} rows); clamping "
              f"to it.", flush=True)
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']

    max_feasible_offset = cfg['GENERATE_VALIDATION_LENGTH'] - cfg['GENERATE_LENGTH']
    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
                                cfg['GENERATE_START_STEP']))
    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
    if feasible_offsets != start_offsets:
        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]

    cfg['LAG_SET'] = derive_lag_set(
        cfg['DELAY_SOURCE'], cfg['DELAY_SET'], cfg['LAG_MODE'], cfg['LAG_DENSITY'], cfg['LAG_STEP'],
        cfg['GENERATE_LENGTH'], cfg['PREDICTION_START'],
    )

    # VERIFY (see this pipeline's own design-decision record for why this
    # guard exists here and not in simple-mde-gmn.py): Stage 2's lag-
    # embedding depth must fit within PREDICTION_START (A), the MDE library
    # segment -- an implicit assumption in simple-mde-gmn.py too, but
    # TSLS's own PREDICTION_START is data-driven (burn_in_trim() shrinks
    # the usable row count depending on which channel fires slowest, unlike
    # simple-mde-gmn.py's fixed-length 1 Hz input), so this fails fast with
    # a clear message instead of a confusing downstream all-NaN/shape
    # error deep inside proper_mde or gmn's own SMap calls.
    max_lag_depth = max((abs(d) for d in cfg['LAG_SET']), default=0)
    if max_lag_depth >= cfg['PREDICTION_START']:
        raise ValueError(
            f"Stage 2's lag-embedding depth ({max_lag_depth} rows, from LAG_SET= "
            f"{cfg['LAG_SET']}) does not fit within PREDICTION_START "
            f"({cfg['PREDICTION_START']} rows, the MDE library segment) -- every "
            f"lagged candidate column would be all-NaN across the entire library. "
            f"This is data-driven for TSLS (burn-in trim shrinks the usable row "
            f"count depending on which channel fires slowest), unlike "
            f"simple-mde-gmn.py's fixed-length input. Narrow "
            f"TSLS_MDE_GMN_LAG_MODE/LAG_STEP, shrink TSLS_MDE_GMN_GENERATE_LENGTH, "
            f"or raise TSLS_MDE_GMN_LIB_RATIO to fix."
        )

    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE']) if cfg['REUSE_BEST_PARAMS'] else None
    code_fingerprint = build_code_fingerprint(__file__) if cfg['REUSE_BEST_PARAMS'] else None

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    filtered_pools = {}
    if cfg['MODE'] == 'MDE-GMN':
        # CCM pre-filter runs once per target_groups_for_mode() group, here,
        # BEFORE depth-0 -- so depth-0's own candidate pool is already the
        # filtered one, same as every later nested-MDE round. See this
        # pipeline's own build_network() docstring: that function no longer
        # runs the pre-filter itself, precisely because main() now always
        # does it first.
        for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
            if cfg['CCM_PREFILTER_RHO_MIN'] > 0:
                filtered_columns, _ = ccm_prefilter_observables(
                    df, obs_cols, group, cfg['PREDICTION_START'],
                    cfg['CCM_PREFILTER_RHO_MIN'], cfg['CCM_PREFILTER_E'], cfg['TP'],
                    cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
                    cfg['LIBSIZE_RATIOS'], cfg['CORES'],
                )
                print(f'CCM pre-filter: kept {len(filtered_columns)}/{len(obs_cols)} '
                      f'observables (rho >= {cfg["CCM_PREFILTER_RHO_MIN"]}) for {group}', flush=True)
            else:
                filtered_columns = obs_cols
            filtered_pools[tuple(group)] = filtered_columns

        # Stage 1 seed: real-observable-only MDE, no delay expansion --
        # matches build_nested_mde_network()'s own Stage-1 candidate pool
        # exactly, so its depth-0 reuse of this result stays valid.
        if cfg['GMN_MODE'] == 'combined':
            # One joint multi_proper_mde call across every target, instead
            # of GMN_MODE='separate's per-target proper_mde loop below --
            # see run_multi_mde_for_targets()'s and load_config()'s own
            # MULTI_MDE_E docstrings. The single shared result is fanned out
            # into one per-target *view* below (same 'm'/'mde_df'/
            # 'SELECTED_COLUMNS' object, just re-keyed with a 'target' entry
            # each) so every downstream consumer that indexes
            # mde_results[target] -- build_nested_mde_network()'s depth-0
            # reuse (adds the SAME joint drivers as edges into every target
            # node in the group, which is the intended "combined" MDE
            # behaviour), and the diagnostics plots/sweeps below -- keeps
            # working completely unchanged.
            candidate_pool = [c for c in filtered_pools[tuple(cfg['TARGETS'])] if c not in cfg['TARGETS']]
            effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
            depth0_start = time.perf_counter()
            _log_progress(progress_log_path, f"depth0 joint MDE START -- targets={cfg['TARGETS']}")
            try:
                shared_result = run_multi_mde_for_targets(
                    cfg['TARGETS'], df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                    effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                    tp=cfg['TP'],
                    metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                    gamma_delta=cfg['GAMMA_DELTA'],
                    target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                    ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                    ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                    reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                    run_cache_dir=cfg['RUN_CACHE_DIR'],
                    data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                    code_fingerprint=code_fingerprint,
                    E=0, max_e=cfg['MULTI_MDE_E'], target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
                )
            except Exception as exc:
                # Same failure mode _failed_mde_result() guards against below
                # -- this is the depth-0 seed for the Stage-1 BFS (shared
                # across every target in the group), so it needs the
                # identical guard or the exact crash just relocates one
                # level up.
                print(f"{cfg['TARGETS']}: joint (multi-target) MDE call FAILED: {exc!r} -- "
                      f'treating as a dead-end (zero drivers selected for every target)', flush=True)
                shared_result = _failed_multi_mde_result(cfg['TARGETS'])
            _log_progress(progress_log_path,
                          f'depth0 joint MDE END -- {time.perf_counter() - depth0_start:.1f}s '
                          f"selected={shared_result['SELECTED_COLUMNS']}")
            for target in cfg['TARGETS']:
                mde_results[target] = {**shared_result, 'target': target}
        else:
            for target in cfg['TARGETS']:
                candidate_pool = [c for c in filtered_pools[(target,)] if c != target]
                effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
                depth0_start = time.perf_counter()
                _log_progress(progress_log_path, f'depth0 MDE START -- {target}')
                try:
                    mde_results[target] = run_mde_for_target(
                        target, df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                        effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                        cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                        tp=cfg['TP'],
                        metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                        gamma_delta=cfg['GAMMA_DELTA'],
                        is_behavioral_target=True,
                        target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                        min_pred_events=cfg['MIN_PRED_EVENTS'],
                        max_window_rows=cfg['MAX_WINDOW_ROWS'],
                        ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                        ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                        run_cache_dir=cfg['RUN_CACHE_DIR'],
                        data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                        code_fingerprint=code_fingerprint,
                        max_e=cfg['MDE_D'],
                        # This loop is sequential over cfg['TARGETS'] (one
                        # target computed at a time, no outer pool sharing
                        # cores with it) -- so unlike build_nested_mde_
                        # network()'s own per-wave outer_workers-based
                        # split, this depth-0 seed call gets the FULL core
                        # budget for its own inner pool.
                        inner_workers=max(1, cfg['CORES']),
                    )
                except Exception as exc:
                    # Same failure mode _failed_mde_result() guards against
                    # in build_nested_mde_network() below -- this is the
                    # depth-0 seed for that same Stage-1 BFS, so it needs
                    # the identical guard or the exact crash just
                    # relocates one level up.
                    print(f'{target}: top-level MDE call FAILED: {exc!r} -- treating as a dead-end '
                          f'(zero drivers selected for this target)', flush=True)
                    mde_results[target] = _failed_mde_result(target)
                _log_progress(progress_log_path,
                              f'depth0 MDE END -- {target} {time.perf_counter() - depth0_start:.1f}s '
                              f"selected={mde_results[target]['SELECTED_COLUMNS']}")

        if cfg['RUN_MDE_DIAGNOSTICS']:
            plot_mde_rho_vs_dimension(
                mde_results, cfg['TARGETS'], cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
            )
            plot_3d_mde_projection(
                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df, obs_cols,
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
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
            cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
            target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
            min_pred_events=cfg['MIN_PRED_EVENTS'],
            max_window_rows=cfg['MAX_WINDOW_ROWS'],
            ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
            observable_columns=filtered_pools.get(tuple(group), obs_cols),
            nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
            exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
            lag_set=cfg['LAG_SET'], tp=cfg['TP'],
            reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
            run_cache_dir=cfg['RUN_CACHE_DIR'],
            data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
            code_fingerprint=code_fingerprint,
            gmn_mode=cfg['GMN_MODE'], multi_mde_e=cfg['MULTI_MDE_E'],
            multi_mde_target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
            external_input_mode=cfg['EXTERNAL_INPUT_MODE'],
            progress_log_path=progress_log_path,
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        if cfg['MODE'] == 'MDE-GMN':
            plot_3d_network(
                net_result, cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / f"network_3d_{net_result['network_id']}.html",
            )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    fingerprints = None
    if cfg['REUSE_BEST_PARAMS']:
        fingerprints = {
            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256, code_fingerprint)
            for network_id, run in gmn_runs.items()
        }

    theta_search_results = run_theta_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['THETA_RANGE'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'],
        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'],
        fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    generate_starts = sweep_generate_start(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
        cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'], fingerprints=fingerprints,
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    rebuild_at_best_params(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    for network_id, run in gmn_runs.items():
        if 'G' not in run:
            # rebuild_at_best_params() already logged why and skipped this
            # network -- nothing downstream has anything to work with
            # without G.
            print(f'{network_id}: no successful Generate() rebuild -- skipping '
                  f'Forecast()/fidelity/output for this network', flush=True)
            continue

        forecast_start = time.perf_counter()
        _log_progress(progress_log_path, f'{network_id}: Forecast() diagnostic START')
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
            plot_range_mode=cfg['FORECAST_PLOT_RANGE'],
            generate_validation_start=cfg['GENERATE_VALIDATION_START'], total_rows=len(df),
        )
        _log_progress(progress_log_path,
                      f'{network_id}: Forecast() diagnostic END -- '
                      f'{time.perf_counter() - forecast_start:.1f}s')
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        if Gf is None:
            print(f'{network_id}: no Forecast() rho to report for this network', flush=True)
        else:
            print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
            for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
                print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'],
                                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics -------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        _log_progress(progress_log_path, 'R7 diagnostics START (tp-vs-rho sweep, E/tau grid)')
        r7_start = time.perf_counter()
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
                             cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
                                   cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        _log_progress(progress_log_path,
                      f'R7 diagnostics END -- {time.perf_counter() - r7_start:.1f}s')
        # R19 (docs/review-actions-v1.md): per constructed network -- needs
        # net_result['network']['Graph'] for D (direct-driver count), which
        # mde_results (depth-0-only) doesn't carry. Gated on its own
        # RUN_MDE_EVALUATE, deliberately separate from RUN_MDE_DIAGNOSTICS
        # above (see that config's own docstring) -- R7's two diagnostics
        # just above are unaffected and always run when RUN_MDE_DIAGNOSTICS
        # is set, regardless of this toggle.
        if cfg['RUN_MDE_EVALUATE']:
            for network_id, run in gmn_runs.items():
                r19_start = time.perf_counter()
                _log_progress(progress_log_path, f'{network_id}: R19 run_mde_evaluate START')
                run_mde_evaluate(run['net_result'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                                  cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                                  exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
                _log_progress(progress_log_path,
                              f'{network_id}: R19 run_mde_evaluate END -- '
                              f'{time.perf_counter() - r19_start:.1f}s')
        else:
            # Logged once per run, unconditionally -- a skipped diagnostic
            # that never appears anywhere is exactly how §18.6's swallowed
            # exception went unnoticed for so long (docs/review-findings.md).
            msg = ("R19 run_mde_evaluate SKIPPED by configuration "
                   "(RUN_MDE_EVALUATE=false) -- see docs/review-findings.md's "
                   "'diffusion-map crash and hang' finding for why this defaults off")
            print(msg, flush=True)
            _log_progress(progress_log_path, msg)
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    _log_progress(progress_log_path,
                  f'PIPELINE COMPLETE -- {time.perf_counter() - pipeline_start:.1f}s total')
    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
