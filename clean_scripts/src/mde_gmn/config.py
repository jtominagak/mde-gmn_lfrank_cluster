"""Every knob, in one place.

load_config() reads the whole configuration from TSLS_MDE_GMN_* environment
variables, applies defaults, and returns it as a plain dict. Nothing else in
the pipeline reads os.environ, so this file is the complete list of what can
be changed from outside. Each entry's comment says what it does and what it
costs; the longer "why this number" evidence lives in
clean_scripts/docs/parameters.md, so this file stays skimmable.

The batch scripts set these from the submitting side; their MODIFIABLE
PARAMETERS block is the same list.

The _env_* helpers just parse: '1'/'true'/'yes'/'on' for booleans,
comma-separated for lists, with a default when the variable is unset.

Retired settings RAISE rather than being ignored, so a stale sbatch fails
immediately instead of running with a value nobody honours any more.

NO HOST-SPECIFIC PATHS ARE BAKED IN. Every directory below comes from an
environment variable with a relative default, so this file runs anywhere. The
cluster paths live in the batch scripts and in saion/cluster_env.sh.

One oddity: matplotlib is imported here only because TARGET_COLORS picks each
target's plotting colour off the 'tab10' colormap. That is a presentation
detail living in configuration, but it is where the colours have always been
assigned and every figure reads them from the config dict.

Depends on: nothing in this package; matplotlib (for TARGET_COLORS only).
Imported by: the entry scripts.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # cluster jobs have no display
import matplotlib.pyplot as plt


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

# The ways Stage 2's lag set can be defined, in the order derive_lag_set()
# documents them (which quantity each one pins: depth, count, every value).
# Named here rather than spelled out at the validation site because
# pipeline.py's lag-depth error message keys its remedy advice off the same
# set -- one list, so the two cannot drift apart.
DELAY_SOURCES = ('lag_mode', 'tau_count', 'explicit')


def load_config():
    cfg = {}

    # ---- Data ---------------------------------------------------------
    # The cluster deployment expects data files copied FLAT into DATA_DIR by
    # the sbatch script -- no nested extraction-folder name.
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
    # dataset --
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
    # see run_mde_for_target()'s docstring for the mechanics and pipeline.py's
    # window-resolution block for the arithmetic. Computed against the row count
    # AFTER TSLS conversion and burn-in trim, not the raw file's row count,
    # because the trim decides how much data is really usable.
    cfg['LIB_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_LIB_RATIO', 0.7))
    cfg['MDE_VALIDATION_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_MDE_VALIDATION_RATIO', 0.5))
    if not (0.0 < cfg['MDE_VALIDATION_RATIO'] < 1.0):
        raise ValueError(
            f"TSLS_MDE_GMN_MDE_VALIDATION_RATIO must be in (0, 1): {cfg['MDE_VALIDATION_RATIO']!r}"
        )

    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH
    # -- 0 (default) means "use the full held-out
    # window"; a larger value is clamped down to it in main().
    cfg['GENERATE_LENGTH'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_LENGTH', 0))

    # Theiler-window / neighbor-exclusion radius for gmn's own recursive
    # Generate()/Forecast() nearest-neighbor search (SMap), passed through
    # as Parameters.exclusionRadius by build_gmn_config() -- a plain static
    # knob, NOT the dynamic/auto mechanism TSLS_MDE_GMN_EXCLUSION_RADIUS below
    # uses -- these are two
    # independent EDM contexts (MDE's forward selection vs. GMN's own
    # Generate()/Forecast()) with their own separate embeddings, so there's
    # no single shared "auto" value that would make sense for both.
    # Deliberately NOT TSLS_MDE_GMN_-prefixed, to keep it visually distinct
    # from the unrelated MDE-side parameter of almost the same name.
    cfg['GMN_EXCLUSION_RADIUS'] = int(os.environ.get('GMN_EXCLUSION_RADIUS', 0))

    # Plotted x-axis range for the final Generate()-vs-Forecast()-vs-actual
    # diagnostic figure -- see run_forecast_diagnostic()'s plot_range_mode
    # docstring.
    cfg['FORECAST_PLOT_RANGE'] = os.environ.get('TSLS_MDE_GMN_FORECAST_PLOT_RANGE', 'full').strip().lower()
    if cfg['FORECAST_PLOT_RANGE'] not in ('full', 'windowed'):
        raise ValueError(
            f"TSLS_MDE_GMN_FORECAST_PLOT_RANGE must be 'full' or 'windowed', "
            f"got {cfg['FORECAST_PLOT_RANGE']!r}"
        )

    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
    # as offsets (rows) from GENERATE_VALIDATION_START. Default MIN=0, MAX=1 tries only offset 0.
    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
    cfg['GENERATE_START_STEP'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_STEP', 50))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    # MDE_D counts distinct *real* observables, not delay-expanded candidate
    # columns -- reused as the per-round cap for both nested-MDE stages
    # (Stage 1: real-observable-only, so the multiplier is always 1;
    # Stage 2: lagged-only, multiplied by however many lag values
    # DELAY_SOURCE's chosen path below resolves to) -- see
    # effective_mde_dimension()'s docstring. Ported unchanged from
    cfg['MDE_D'] = int(os.environ.get('TSLS_MDE_GMN_D', 20))
    # Prediction interval every proper_mde forward-selection call (both the
    # top-level per-target search and every nested-MDE per-node search)
    # ranks and validates every candidate driver against.
    cfg['TP'] = int(os.environ.get('TSLS_MDE_GMN_TP', 1))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # The two gates above are applied to whichever metric scores the node, and
    # the two metrics do not share a ceiling. `correlation`, which scores a
    # behavioural target, tops out at 1.0. `f1`, which scores a TSLS neuron
    # node, does not: r9_min_pred_events_curve.csv measured a NEAR-PERFECT
    # predictor of a real TSLS channel at f1 ~ 0.49-0.54 on average (never
    # above 0.69). A single 0.2 bar therefore demanded ~38% of achievable
    # skill from a neuron and 20% from a behavioural target.
    #
    # 0.1 restores parity in those terms (0.20 x 0.52 ~ 0.104), and is the
    # looser gate for neuronal data that the asymmetry calls for. It is not
    # free: r9 puts the f1 estimator's own std at ~0.069 at MIN_PRED_EVENTS=20,
    # so 0.1 is only ~1.4 sigma there -- it falls to ~0.052 at 40 events and
    # ~0.028 at 160, and real candidate windows in job 4774136 carried 67-188
    # events, so the floor is the worst case rather than the typical one.
    # Raise this first if the network starts admitting links that do not
    # replicate.
    cfg['EMBED_DIM_RHO_MIN_F1'] = float(os.environ.get('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN_F1', 0.1))
    cfg['CROSS_MAP_RHO_MIN_F1'] = float(os.environ.get('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1', 0.1))
    # What actually travels to the MDE calls: _rho_min_for_metric() picks the
    # entry matching the node's own metric, 'default' for anything else.
    cfg['EMBED_DIM_RHO_MIN_BY_METRIC'] = {'default': cfg['EMBED_DIM_RHO_MIN'],
                                          'f1': cfg['EMBED_DIM_RHO_MIN_F1']}
    cfg['CROSS_MAP_RHO_MIN_BY_METRIC'] = {'default': cfg['CROSS_MAP_RHO_MIN'],
                                          'f1': cfg['CROSS_MAP_RHO_MIN_F1']}
    # CCM pre-filter (see ccm_prefilter_observables()'s docstring): a cheap
    # CCM(observable -> target) skill check run once per target group,
    # right before either nested-MDE stage, to shrink REGION_CHANNEL_CAP's
    # candidate pool down to only observables that cross-map at least one
    # target above this threshold -- cuts cost on both Stage 1 (real
    # observables) and Stage 2 (their lags, since Stage 2 draws from the
    # same filtered pool) for free. 0 disables the filter entirely (pool
    # passed through unchanged); >0 enables it as the rho cutoff.
    cfg['CCM_PREFILTER_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN', 0.2))
    # Static embedding dimension for the CCM pre-filter's own CCM calls.
    # Its OWN parameter, deliberately not a reuse of MULTI_MDE_E below: at
    # 40 Hz one embedding step at tau=-1 covers 1/40th the real time it
    # covers at 1 Hz, so a value tuned for 1 Hz data does not transfer.
    # 7 is the middle of E_RANGE's own search below. See
    # docs/parameters.md.
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
    # edm.multi_proper_mde -- forward-selecting drivers
    # that jointly predict every TARGETS column at once -- instead of the
    # GMN_MODE='separate' behaviour of one independent
    # edm.proper_mde call per target.
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
    # makes for MDE forward-selection. A PLAIN STATIC INTEGER -- there is no
    # 'auto' mode here. An auto mode would have to recompute the radius at each
    # internal cross-map as the embedding grows, and proper_mde and friends are
    # each a single opaque call taking one radius for their whole internal
    # search, with no staged internal calls to intercept.
    cfg['EXCLUSION_RADIUS_VALUE'] = int(os.environ.get('TSLS_MDE_GMN_EXCLUSION_RADIUS', 0))
    # Nested-MDE network construction (MODE='MDE-GMN' only). Reuses MDE_D as
    # the per-node driver cap N at every nesting level (deliberately not a
    # separate parameter). NESTED_M is the number of *additional* nested-MDE
    # rounds beyond each target's own initial selection -- total graph depth
    # from each target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for
    # "uncapped".
    #
    # It is a WHOLE-NETWORK budget, shared by Stage 1 and Stage 2: a node's
    # depth is its distance from the target regardless of which stage
    # discovered it (the builder's `node_depth`), not two independent per-stage
    # counters. See build_nested_mde_network()'s docstring.
    cfg['NESTED_M'] = int(os.environ.get('TSLS_MDE_GMN_NESTED_M', 1))
    # First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted
    # and offered as MDE candidates (or, when MODE='GMN', used directly as
    # the interaction-matrix's full node set). 999 is effectively uncapped
    #.
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('TSLS_MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- Two-stage nested-MDE network construction (MODE='MDE-GMN' only) --
    # See build_nested_mde_network() for the algorithm: Stage 1 builds the
    # real-observable backbone recursively (NESTED_M rounds), then Stage 2
    # attaches lagged drivers as leaf nodes in a single non-recursive pass.
    # DELAY_SOURCE below picks which of THREE ways defines the lag set Stage 2
    # may choose from -- they differ only in which quantity you pin, see
    # derive_lag_set():
    #   'lag_mode'  (default) pins the DEPTH, the count follows
    #               (LAG_MODE/LAG_DENSITY/LAG_STEP)
    #   'tau_count'           pins the COUNT, the depth follows
    #               (LAG_STEP as tau + LAG_COUNT)
    #   'explicit'            pins EVERY VALUE (DELAY_SET)
    # Reach for 'tau_count' or 'explicit' when the auto-derived count is too
    # large to finish: Stage 2's candidate pool is (observables x lags), and
    # Stage 2 is where essentially all of a run's time goes.
    #
    # TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES is retired: Stage 2 considers
    # lagged columns only, so there is nothing for it to select recursively. The
    # variable is REJECTED rather than ignored, so an sbatch still setting it
    # fails loudly instead of quietly doing something else.
    if 'TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES' in os.environ:
        raise ValueError(
            'TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES is retired: Stage 2 now considers '
            'lagged columns only and never recurses (see build_nested_mde_network). '
            'Remove it from the sbatch script.')

    # A zero-driver leaf is partitioned out of the simulated network as an
    # "external input" (partition_network()) rather than being given a
    # self-history embedding. EXTERNAL_INPUT_MODE decides what
    # Generate()/Forecast() then do with such a node -- see
    # _generalized_node_generate()'s docstring:
    #   'real_data' (default): the node is genuinely exogenous -- its
    #     Generate() call short-circuits to the real recorded value at each
    #     step (self.Network.data, already loaded in full), never running
    #     an SMap self-regression. The INTERIOR network still free-runs
    #     autonomously; only external inputs are fed from real data. This
    #     is what "external input" means in the standard control-theory
    #     sense.
    #   'self_simulate': the older behaviour -- such a node gets an
    #     unconditional self-history embedding and is genuinely self-predicted
    #     every step, never touching real data past the initial seed. The
    #     partition then only affects reporting and the figures.
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
    if cfg['DELAY_SOURCE'] not in DELAY_SOURCES:
        raise ValueError(f"TSLS_MDE_GMN_DELAY_SOURCE must be one of "
                          f"{', '.join(DELAY_SOURCES)}: {cfg['DELAY_SOURCE']!r}")

    cfg['LAG_MODE'] = os.environ.get('TSLS_MDE_GMN_LAG_MODE', 'generation_length_adjusted').strip().lower()
    if cfg['LAG_MODE'] not in ('generation_length_adjusted', 'every_lag_possible'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_MODE must be 'generation_length_adjusted' or "
                          f"'every_lag_possible': {cfg['LAG_MODE']!r}")

    cfg['LAG_DENSITY'] = os.environ.get('TSLS_MDE_GMN_LAG_DENSITY', 'strided').strip().lower()
    if cfg['LAG_DENSITY'] not in ('strided', 'dense'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_DENSITY must be 'strided' or 'dense': {cfg['LAG_DENSITY']!r}")
    # LAG_STEP is the spacing between consecutive lags, and it means the same
    # thing under both auto-derived sources: the sampling stride of LAG_MODE's
    # bound under DELAY_SOURCE='lag_mode', and tau under 'tau_count'. One knob
    # rather than two so the two paths cannot disagree about the stride.
    cfg['LAG_STEP'] = int(os.environ.get('TSLS_MDE_GMN_LAG_STEP', 1))
    if cfg['LAG_STEP'] < 1:
        raise ValueError(f"TSLS_MDE_GMN_LAG_STEP must be >= 1: {cfg['LAG_STEP']!r}")

    # LAG_COUNT: how many lags, used only when DELAY_SOURCE='tau_count'. The
    # lag set is then exactly LAG_COUNT lags spaced LAG_STEP apart, so the
    # deepest is LAG_STEP * LAG_COUNT (which still has to fit inside
    # PREDICTION_START -- main() checks that for every source alike).
    # LAG_DENSITY does not apply on this path; LAG_STEP=1 gives consecutive
    # lags. This is the knob that makes the Stage 2 pool size a decision
    # instead of a consequence of GENERATE_LENGTH.
    cfg['LAG_COUNT'] = int(os.environ.get('TSLS_MDE_GMN_LAG_COUNT', 10))
    if cfg['LAG_COUNT'] < 1:
        raise ValueError(f"TSLS_MDE_GMN_LAG_COUNT must be >= 1: {cfg['LAG_COUNT']!r}")

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
    # by run_theta_search() for both MODE='MDE-GMN' and MODE='GMN'. Theta is the
    # ONLY thing searched per node: E and every driver's delay come from the
    # network itself (MODE='MDE-GMN': nested-MDE; MODE='GMN': the
    # E_RANGE/TAU_RANGE grid search above), which is what makes a per-node
    # search affordable at all.
    cfg['THETA_RANGE'] = _env_float_list(
        'TSLS_MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('TSLS_MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('TSLS_MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS', True)
    # RUN_MDE_EVALUATE gates ONLY run_mde_evaluate(), the PCA/diffusion-map
    # baseline comparison. Kept separate from RUN_MDE_DIAGNOSTICS above, which
    # still gates the two sweeps and the 3D projection; those are unaffected by
    # the problem below.
    #
    # DEFAULTS OFF because pydiffmap's diffusion-map fit, on real TSLS data,
    # both raises (ValueError: Input X contains infinity..., in 1.8s at an
    # 800-row scale) and HANGS (past 74 CPU-minutes with no completion, at the
    # real ~31,000-row scale). The mechanism of the hang is not diagnosed.
    # run_mde_evaluate() does guard each baseline with an exception handler and a
    # wall-clock bound (see diagnostics.py), but a guard only helps where someone
    # remembered to put one -- so this stays off until the hang is understood.
    # Nothing in the causal claim depends on it; it is optional tooling.
    # Its absence is logged once per run, so a skipped diagnostic is never
    # silent.
    cfg['RUN_MDE_EVALUATE'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_EVALUATE', False)

    # ---- Parallelism ----------------------------------------------------
    # CORES is the whole parallelism budget. There is no separate knob for an
    # inner pool because there is no inner pool to size: proper_ccm and
    # proper_mde have no multiprocessing of their own (nothing in
    # edm.py uses a Pool). The builder splits CORES into
    # outer processes and inner threads per wave -- see its docstring.
    cfg['CORES'] = int(os.environ.get('TSLS_MDE_GMN_CORES', os.cpu_count() or 1))

    # ---- TSLS-aware scoring (edm.py's proper_mde/
    # proper_ccm/proper_simplex/adaptive_event_metrics_from_tsls, replacing
    # dimx.MDE / apps.InteractionMatrix / pyEDM.EmbedDimension / plain
    # ComputeError rho everywhere in this pipeline that scores a TSLS-
    # converted column) -----------------------------------------------------
    # 'correlation' (plain Pearson rho, pyEDM's own default) | 'mae' |
    # 'f1' | 'gamma' (Kistler coincidence factor) -- see
    # edm.py's _score()/adaptive_event_metrics_from_tsls
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
    # MODE='MDE-GMN' -- the default -- does not use this knob at all. See
    # TARGET_RANKING_WINDOW_ROWS/MIN_PRED_EVENTS/MAX_WINDOW_ROWS/
    # CCM_LIB_SIZES_ROWS below. A single fixed window with a percentage-based
    # library grid made f1-scored ranking degenerate for real neurons: their
    # reset events are too sparse within any window small enough to keep the CCM
    # gate's convergence curve meaningful.
    cfg['CCM_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_CCM_WINDOW_ROWS', 800))
    # The library sizes for the CCM convergence curve, as percentages of
    # CCM_WINDOW_ROWS. Calibrated, not guessed: in a 54-pair stratified sample at
    # the 800-row default, 35% of pairs were still rising at 40 rows (the 5%
    # point). A real but minority fraction -- enough to keep the small
    # percentages (2/3/5/8%, i.e. 16/24/40/64 rows), which still carry rising-
    # curve signal for those pairs, while coarsening the already-converged upper
    # end (12% and 30% dropped, 20% and 50% kept).
    cfg['CCM_LIB_SIZE_PCTS'] = tuple(
        _env_float_list('TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS', (2, 3, 5, 8, 20, 50))
    )
    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_MDE_GMN_CCM_SAMPLE', 20))
    cfg['CCM_SLOPE'] = float(os.environ.get('TSLS_MDE_GMN_CCM_SLOPE', 0.01))

    # The next four settings replace CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS for
    # MODE='MDE-GMN'. They separate two things the single old window conflated:
    # how much data a node is SCORED over, and which library sizes the CCM gate
    # sweeps. See _resolve_target_ranking_window() and proper_mde()'s `libSizes`.
    #
    # TARGET_RANKING_WINDOW_ROWS: the depth-0 seed's own fixed
    # ranking/E-select window (behavioral targets have no reset-event
    # concept, so there's nothing to extend against) -- arbitrary, set to
    # 1000 for now; larger than the old CCM_WINDOW_ROWS=800 default, no
    # other rationale yet.
    cfg['TARGET_RANKING_WINDOW_ROWS'] = int(
        os.environ.get('TSLS_MDE_GMN_TARGET_RANKING_WINDOW_ROWS', 1000))
    # MIN_PRED_EVENTS: how many of a neuron's own real reset events must
    # fall in its scoring window's PRED portion before f1 is treated as
    # signal rather than noise. The window grows until it does -- never past
    # the leakage boundary B. A plain fixed integer: it answers a question
    # about the METRIC, not about the neuron, so it is not a fraction of
    # anything neuron-specific. Empirically calibrated; see
    # docs/parameters.md for the measurements and why 20 rather than 40.
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
    # These are CCM_LIB_SIZE_PCTS' calibrated percentages converted to absolute
    # rows against the old 800-row window (16, 24, 40, 64, 160, 400), plus one
    # more point (800) so the plateau stays visible now that the surrounding
    # frame can be much larger. Re-validated at both an 800-row and an 8000-row
    # frame on real data: the same absolute values gave the same "rising, then
    # plateauing" shape at both, because that shape depends on the absolute
    # library size, not its ratio to the frame.
    cfg['CCM_LIB_SIZES_ROWS'] = tuple(
        _env_int_list('TSLS_MDE_GMN_CCM_LIB_SIZES_ROWS', (16, 24, 40, 64, 160, 400, 800))
    )

    # ---- Search reuse ---------------------------------------------------
    # Reuse a past job's MDE/network/theta search results instead of redoing
    # them, but ONLY when every setting that could change the outcome matches
    # exactly -- caching.py builds the fingerprints that decide that.
    cfg['REUSE_BEST_PARAMS'] = _env_bool('TSLS_MDE_GMN_REUSE_BEST_PARAMS', False)
    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_PREV_RUNS_DIR', '../../results'))
    cfg['RUN_CACHE_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_RUN_CACHE_DIR', cfg['RESULTS_DIR']))
    cfg['RUN_CACHE_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- GPU (torch) engine ---------------------------------------------
    # GPU_ENABLED gates the whole torch/CUDA path. Off by default, and when
    # off the pipeline behaves exactly as it did before the flag existed:
    # fork-based worker pools, pyEDM/scipy engines throughout.
    #
    # It also decides the multiprocessing start method, because CUDA does NOT
    # support fork() -- forking a process holding a CUDA context is undefined
    # behaviour in the child, usually surfacing as "Cannot re-initialize CUDA
    # in forked subprocess". Spawn is not free (a spawn worker re-imports
    # pandas/numpy/matplotlib/pyEDM/gmn from scratch, ~18s measured), which is
    # why it is conditional rather than always on: CPU-only runs get no
    # benefit, so they keep fork. See docs/parameters.md.
    cfg['GPU_ENABLED'] = _env_bool('TSLS_MDE_GMN_GPU_ENABLED', False)
    cfg['MP_START_METHOD'] = 'spawn' if cfg['GPU_ENABLED'] else 'fork'

    # ---- Parallelized-MDE scope (parallel-MDE pipeline only) -------------
    # Read by both pipelines but used only by network_parallel.py; the serial
    # builder takes no scope. Both values produce the
    # IDENTICAL network -- this trades wall-clock against wasted compute,
    # nothing else, which is why it is deliberately NOT part of
    # build_network_fingerprint() (a cached network built under one scope is
    # legitimately reusable by the other).
    #
    #   'union'  ONE MDE per node, over real AND lagged candidates together,
    #            for the entire node universe, submitted to a single pool.
    #            Exactly one barrier in the whole build, and real and lagged
    #            candidates compete directly inside each forward selection.
    #            This is NOT the same algorithm as the parent pipeline's two
    #            stages, so it produces a different (not a wrong) network.
    #   'staged' (recommended for comparability) -- the parent pipeline's
    #            algorithm, minus the per-wave barriers: Stage 1 over the
    #            whole universe, the instant Stage 1 replay, then Stage 2
    #            (lagged-only) for exactly Stage 1's visited set. Builds the
    #            SAME network as run_serial_mde.py
    #            (tests/test_parallel_mde_equivalence.py).
    #
    # Either scope may have to RECOMPUTE a node serially: Phase 0 cannot know
    # which observables will end up upstream, so a precomputed result whose
    # real picks turn out to be cycle-closing is discarded and recomputed
    # with those candidates excluded -- see _mde_result_for_node().
    cfg['PARALLEL_MDE_SCOPE'] = os.environ.get(
        'TSLS_MDE_GMN_PARALLEL_MDE_SCOPE', 'union').strip().lower()
    if cfg['PARALLEL_MDE_SCOPE'] not in ('union', 'staged'):
        raise ValueError(
            f"TSLS_MDE_GMN_PARALLEL_MDE_SCOPE must be 'union' or 'staged', got "
            f"{cfg['PARALLEL_MDE_SCOPE']!r}")

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg
