#!/bin/bash
#SBATCH --job-name=run_tsls_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64
# ADJUST: must be a literal absolute path, not a shell variable -- Slurm
# parses #SBATCH directives itself before this script (or any shell) runs,
# so no shell variable expansion happens here. Pointed directly at
# /flash/FukaiU/j-tominaga/pao_rotation_results/ (a fixed path that must
# already exist -- created once via `mkdir -p`, persists across jobs the
# same way $FLASH_BASE itself already does) rather than .../job_%j/ because
# Slurm opens these files before the job starts, i.e. before $SLURM_JOB_ID
# is known and before this script's own `mkdir -p "$OUT_DIR"` below could
# create that subdirectory. The EXIT trap further down (search SLURM LOG
# RELOCATION) moves these into $OUT_DIR once it exists.
# On /flash, not /home, and shared with simple-mde-gmn.sbatch's own
# pao_rotation_results/ directory -- same reasoning as that file: home
# storage is limited, and Slurm logs alone have reached tens of MB per job
# on the sibling pipeline.
#SBATCH --output=/flash/FukaiU/j-tominaga/pao_rotation_results/%x-%j.out
#SBATCH --error=/flash/FukaiU/j-tominaga/pao_rotation_results/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
# NOTE ON --time/--mem/--cpus-per-task ABOVE: this pipeline is
# scripts/src/simple-mde-gmn.py's own two-stage nested-MDE + GMN pipeline,
# ported wholesale (see that file's own module docstring for the full
# design), fed by a TSLS-converted 40 Hz recording instead of a pre-binned
# 1 Hz one -- a MUCH larger usable row count by default than
# simple-mde-gmn.sbatch's own ~1100 rows.
#
# UPDATED: cleaned_helper_functions.proper_mde now takes an optional `pool`
# (a ThreadPoolExecutor -- every hot loop it reaches, pyEDM.Simplex and the
# KDTree build/query behind proper_ccm/proper_embed_dimension, is a compiled
# extension that releases the GIL, so threads give real parallelism without
# a second multiprocessing pool). run_mde_for_target() in tsls-mde-gmn.py
# opens exactly ONE such pool per node, sized `inner_workers = cores //
# outer_workers` and recomputed fresh every wave from that wave's own real
# size (same formula build_nested_mde_network()'s/attach_lag_embeddings()'s
# own outer_workers already used) -- so this run is the first real exercise
# of that inner pool on cluster data, on top of being this config's own
# validation-scale smoke test. No pool-nesting risk: outer_workers *
# inner_workers is bounded by `cores` by construction every wave, the same
# invariant that guards simple-mde-gmn.sbatch's own FIXED CAVEAT case
# (apps.InteractionMatrix x pyEDM.CCM's nested pools) -- see
# run_mde_for_target()'s own INNER POOL docstring for the full design.
# --cpus-per-task here now sizes BOTH layers: OUTER (per-node, per-wave)
# ProcessPoolExecutor parallelism AND, through inner_workers, each node's
# own proper_mde() ranking loop / CCM gate, plus (unchanged) gmn.GMN's own
# Generate()/Forecast() core count (cores=) and PredictNonlinear's
# numProcess= (theta retuning).
#
# --cpus-per-task=16 / --mem=32G / --time=24:00:00 below are a deliberate
# STEP UP from the REGION_CHANNEL_CAP=3 SMOKE TEST this file was most
# recently sized for, to a MID-SCALE CALIBRATION RUN
# (REGION_CHANNEL_CAP=15/MDE_D=4/NESTED_M=1/GENERATE_LENGTH=100 below) --
# deliberately reproducing job 38976891's own historical configuration, but
# now against the current code, after the smoke test (job 40385630)
# confirmed a clean "Pipeline complete" at REGION_CHANNEL_CAP=3 with every
# fix in docs/review-findings.md landed (metric-routing, per-node window
# resolution, CCM-gate absolute library sizes, the RUN_MDE_EVALUATE
# toggle/guard/timeout, and the progress-monitoring log). This is the first
# time this exact scale has run under any of that code -- these numbers are
# still an estimate, not a measurement, reasoned from what evidence exists:
#   - job 38976891 itself, at this identical REGION_CHANNEL_CAP=15/MDE_D=4/
#     NESTED_M=1 configuration, timed out after 4+ hours on a single
#     depth-1 wave of just 3 nodes under the OLD (fixed 800-row window,
#     Pearson-correlation-scored-even-for-neurons) design -- confirmed
#     still doing real work throughout, not hung. That history is the
#     floor this sizing has to respect, not a number this run is assumed
#     to beat.
#   - Both directions of change since then are real, and point opposite
#     ways on cost, so neither is treated as dominant by assumption:
#     R1's 186x compute_time_since_previous_spike speedup, the inner
#     ThreadPoolExecutor per node, and the CCM pre-filter shrinking the
#     candidate pool before any MDE call all cut cost; but the window-
#     sizing redesign lets a single slow-firing neuron node resolve a
#     ranking window up to MAX_WINDOW_ROWS=30000 rows (vs. the old fixed
#     800) specifically so f1-scoring has enough real events to be
#     meaningful (docs/tsls-mde-gmn-pipeline.pdf's own per-node
#     ranking-window-sizing section) -- more real, necessary compute per
#     node, not less, and this scale has never been measured under it.
#   - Direct single-node timing this session (verify_window_fix_real.py,
#     scratchpad, not committed) used this exact REGION_CHANNEL_CAP=15
#     candidate pool (44 real channels) and measured 25.5s wall-clock for
#     one full run_mde_for_target() call at a 3,971-row resolved window --
#     but every candidate failed crossMapRhoMin before reaching the more
#     expensive CCM gate in that measurement, and a slower-firing node here
#     could resolve a window up to ~7.5x larger (30,000 rows) with real
#     candidates actually reaching the CCM gate's own KD-tree cost -- so
#     that 25.5s figure is a floor on per-node cost at this scale, not a
#     representative average.
#   - cpus-per-task=16 (up from the smoke test's 8): --cpus-per-task sizes
#     BOTH layers (outer per-wave ProcessPoolExecutor parallelism AND,
#     through inner_workers, each node's own proper_mde() ranking loop /
#     CCM gate -- see UPDATED note above). REGION_CHANNEL_CAP=15's own
#     45-channel pool produced 3-node waves historically (job 38976891) --
#     16 gives real headroom for a somewhat wider wave than that without
#     requesting far more than this scale can plausibly use.
#   - mem=32G (up from 16G): still generous headroom, not a tight fit --
#     this scale's own TSLS-converted DataFrame is still at most tens of
#     MB -- but per-node working data (expanded delay columns, a resolved
#     window up to 30,000 rows) is real enough now to warrant doubling the
#     smoke test's own margin rather than assuming it's unaffected.
#   - time=24:00:00 (up from 2 hours): six times job 38976891's own 4-hour
#     budget that this identical configuration already exhausted once,
#     under the OLD design -- generous enough to survive the window-sizing
#     redesign genuinely costing more per node at this scale (the
#     opposite-direction possibility above), without committing to the
#     full 4-day VALIDATION-tier budget this file used before the smoke-test
#     shrink. If this run also times out, that is itself the concrete
#     "isolate Stage 1's real cost driver at REGION_CHANNEL_CAP=15" data
#     point docs/tsls-mde-gmn-pipeline.pdf's own Future Directions already
#     calls for -- not a sizing failure to silently resubmit past.
#
# GENERATE_LENGTH is set to 100 (job 38976891's own historical value, up
# from the smoke test's 40) rather than left at its own 0 sentinel --
# LAG_MODE=generation_length_adjusted derives Stage 2's lag-candidate bound
# directly from it (derive_lag_set()), and 0 resolves in main() to the FULL
# held-out window (thousands of rows at this dataset's scale) before that
# happens -- unrelated to, and not implied by, REGION_CHANNEL_CAP/MDE_D
# below. Leaving it at 0 here would silently make Stage 2 search thousands
# of lag columns at this scale.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"   # default: the 40 Hz (25ms-bin) spike-train file

# ---- Overall pipeline shape ------------------------------------------------
# MODE=MDE-GMN: the network is built in two stages, no CCM anywhere -- see
#   build_nested_mde_network() (Stage 1) and attach_lag_embeddings()
#   (Stage 2) in tsls-mde-gmn.py for the full algorithm (ported from
#   simple-mde-gmn.py, TSLS-scored throughout).
#   Stage 1: nested-MDE over *real* (delay=0) TSLS-converted observables
#     only. Each target's MDE-selected drivers (up to MDE_D of them)
#     themselves become new MDE targets in turn, for NESTED_M further
#     rounds. Any driver that would create a cycle or a self-loop is
#     rejected as soon as it's proposed.
#   Stage 2: for every Stage-1 node, one non-recursive MDE round over the
#     *lagged* candidate pool (LAG_MODE/LAG_DENSITY/LAG_STEP/DELAY_SOURCE
#     below) spanning the whole observable universe -- attaches each node's
#     own generalized (multi-delay) embedding coordinates.
# MODE=GMN: skip MDE entirely; the interaction matrix is built directly
#   from the full REGION_CHANNEL_CAP-limited, TSLS-converted observable set
#   via cleaned_helper_functions.proper_ccm (build_interaction_matrix() in
#   tsls-mde-gmn.py) -- this pipeline's own existing replacement for
#   apps.InteractionMatrix's CLI subprocess, unaffected by the MDE-GMN-side
#   two-stage rework.
MODE="MDE-GMN"             # default: MDE-GMN. Other value: GMN
# GMN_MODE="combined": network construction spans every TARGETS entry in one
#   shared graph (target_groups_for_mode() in tsls-mde-gmn.py), AND (MODE=
#   MDE-GMN only) the top-level, per-TARGETS MDE selection itself runs ONCE
#   via cleaned_helper_functions.multi_proper_mde -- forward-selecting
#   drivers that jointly predict every target at once (MULTI_MDE_E/
#   MULTI_MDE_TARGET_AGGREGATE below configure that call) -- instead of one
#   independent proper_mde call per target. GMN_MODE="separate": one
#   independent network AND one independent per-target proper_mde call per
#   TARGETS entry, exactly as before multi_proper_mde was wired in.
GMN_MODE="combined"           # default: combined. Other value: separate
TARGETS="head_position_x,head_position_y"   # comma-separated target column names (never TSLS-converted)

# ---- MDE (variable selection; unused when MODE=GMN) ------------------------
# MDE_D counts distinct *real* observables, not delay-expanded candidate
# columns, and is reused as the per-round cap for *both* nested-MDE stages
# (Stage 1: real-observable-only, so the multiplier is always 1; Stage 2:
# lagged-only, multiplied by however many lag values LAG_MODE/LAG_DENSITY/
# LAG_STEP below resolve to) -- see effective_mde_dimension() in
# tsls-mde-gmn.py. Also reused (as the RAW, un-multiplied value -- see
# run_mde_for_target()'s own max_e docstring) as the ceiling on
# proper_mde's per-candidate embedding-dimension auto-select sweep in its
# CCM causality gate -- no separate maxE knob is exposed here, per
# explicit direction to reuse this already-exposed budget rather than
# introduce a second one.
MDE_D=6                        # MID-SCALE CALIBRATION RUN (see NOTE above) -- default: 20
# Prediction interval every proper_mde forward-selection call ranks/
# validates every candidate driver against -- ported from
# simple-mde-gmn.py's own TP parameter (ranking, not just final scoring).
TP=20                          # default: 1 (proper_mde's own default)
EMBED_DIM_RHO_MIN=0.2         # default: 0.2, embedding-dimension skill gate
CROSS_MAP_RHO_MIN=0.2         # default: 0.2, cross-map skill gate
# CCM pre-filter: a cheap CCM(observable -> target) skill check run once
# per target group (see GMN_MODE above), right before either nested-MDE
# stage -- shrinks REGION_CHANNEL_CAP's candidate pool down to only
# observables that cross-map at least one target in that group's own
# rho >= this threshold, before Stage 1 (real observables) or Stage 2
# (their lags, which draw from this same filtered pool) ever see it. 0
# disables the filter entirely (candidate pool passed through unchanged);
# set >0 to enable it as the rho cutoff. See ccm_prefilter_observables() in
# tsls-mde-gmn.py.
CCM_PREFILTER_RHO_MIN=0.2     # default: 0.2. 0 disables the CCM pre-filter entirely
# Static embedding dimension for the CCM pre-filter's own CCM calls --
# deliberately its OWN parameter, not a reuse of MULTI_MDE_E below (an
# earlier version of this pipeline did reuse it, but the two knobs need to
# be independently tunable): this pipeline's own data is TSLS-converted
# from a 40 Hz recording, vs. simple-mde-gmn.sbatch's 1 Hz input -- 40x more
# rows per unit of real time. Both pipelines fix tau=-1 (one sample step),
# so one embedding step here covers only 1/40th the real time that one step
# covers in simple-mde-gmn.py; matching that in raw sample-count terms
# would need a wildly impractical E (~40x higher, both computationally and
# against the CCM_WINDOW_ROWS/LIBSIZE_RATIOS-driven smallest-libSize-vs-E
# feasibility floor -- see CCM_WINDOW_ROWS below). 7 is a deliberately
# modest compromise instead: the middle value of this pipeline's own
# already-team-vetted E_RANGE search below (E_MIN=3/E_MAX=13/E_STEP=4 ->
# 3,7,11), rather than an untested extrapolation, and still leaves large
# margin against the ~38 (== CCM_WINDOW_ROWS=800 x smallest LIBSIZE_RATIOS
# entry 0.05, minus the +2 floor) feasibility ceiling.
CCM_PREFILTER_E=7             # default: 7 -- deliberately different from simple-mde-gmn.sbatch's CCM_PREFILTER_E=3 (40x sampling-rate difference, see comment above)
# Library sizes for the CCM pre-filter's own CCM calls, as fractions of the
# same CCM_WINDOW_ROWS-bounded window every other CCM call in this
# pipeline already uses (see CCM_WINDOW_ROWS below) rather than absolute
# row counts, since that window's actual row count depends on
# CCM_WINDOW_ROWS/the data itself. Converted to actual libSizes (deduped,
# floored at 10 rows) once the window is known -- see
# ccm_prefilter_observables()'s own docstring.
LIBSIZE_RATIOS="0.05,0.10,0.20,0.40,0.6,0.8,0.9"
# Only consulted when GMN_MODE=combined (see GMN_MODE's own comment above).
# UPDATED: multi_proper_mde now auto-selects E per candidate (a joint
# rho-vs-E sweep via multi_proper_embed_dimension(), mirroring
# single-target proper_mde's own auto-select convention) -- main()'s own
# depth-0 seed call always requests auto-select, and this value is reused
# as that sweep's own ceiling (max_e), not a fixed E, the same "reuse an
# already-exposed budget" convention MDE_D already established for the
# single-target path.
MULTI_MDE_E=8                 # default: 3. Ceiling for multi_proper_mde's per-candidate auto-selected E (GMN_MODE=combined only)
# How multi_proper_mde combines each candidate's per-target scores at both
# the ranking and CCM-gate stages. 'mean': don't penalize a driver that's
# strong for one target and weak for another. 'min': only accept drivers
# that are simultaneously good for every target.
MULTI_MDE_TARGET_AGGREGATE="mean"   # default: mean. Other value: min (GMN_MODE=combined only)
# Theiler-window / neighbor-exclusion radius FLOOR for every proper_mde/
# proper_ccm/proper_simplex/proper_embed_dimension call this pipeline makes
# for MDE forward-selection. UNLIKE simple-mde-gmn.py's own
# MDE_GMN_EXCLUSION_RADIUS, this is a PLAIN STATIC INTEGER, no 'auto'
# STRING option -- proper_mde has no equivalent to dimx.Run's own staged
# sequence of internal CrossMapColumns/EmbedDimension/CCM calls that
# simple-mde-gmn.py's 'auto' mode intercepts to recompute a radius between
# stages. This does NOT mean the radius is never recomputed here, though
# (R21, docs/review-actions-v1.md): proper_mde grows the *effective*
# radius per forward-selection dimension on top of this floor --
# exclusionRadius_d = max(EXCLUSION_RADIUS, deepest |delay| among columns
# already selected earlier in that same call) -- via its own new
# column_delay parameter, not by hooking into staged calls the way the
# sibling's 'auto' mode does. So EXCLUSION_RADIUS=0 sets the FLOOR to 0;
# it does not mean no dynamic adjustment ever happens. See
# cleaned_helper_functions.proper_mde()'s own column_delay docstring and
# docs/tsls-mde-gmn-pipeline.pdf's own R21 section for the full mechanism.
EXCLUSION_RADIUS=0            # default: 0. Static floor; proper_mde grows the effective per-dimension radius on top of it (R21) -- see NOTE above
# Nested-MDE network depth (MODE=MDE-GMN only -- ignored for MODE=GMN).
# MDE_D above doubles as N (the per-node driver cap) at every nesting
# level. NESTED_M is the number of *additional* nested-MDE rounds beyond
# each target's own initial selection -- total graph depth from each
# target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for "uncapped":
# nesting keeps expanding round after round with no fixed depth limit,
# until it runs out on its own, rather than stopping after a fixed number
# of rounds. Combinatorially expensive -- see simple-mde-gmn.sbatch's own
# NESTED_M comment for the full rationale (identical here); keep both
# MDE_D and NESTED_M small for a first test run.
NESTED_M=4                    # MID-SCALE CALIBRATION RUN (see NOTE above) -- default: 1, additional nested-MDE rounds beyond each target's own selection. 0 = uncapped
# First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted and
# offered as MDE candidates (or, when MODE=GMN, used directly as the
# interaction-matrix's full node set). 999 is effectively uncapped (matches
# notebooks/tsls-mde-gmn.ipynb's own full-channel-set decision).
REGION_CHANNEL_CAP=30         # MID-SCALE CALIBRATION RUN (see NOTE above) -- default: 999

# Stage 2's (and every node's self_delay_set's) nonzero-delay list (in
# rows, always negative) -- MODE=MDE-GMN only. Each real observable is
# offered to Stage 2's forward selection as one column per delay, so MDE
# can pick specific (observable, delay) pairs rather than being stuck with
# one fixed lag. This is a generalized/non-uniform Takens embedding (Deyle
# & Sugihara 2011) -- see simple-mde-gmn.sbatch's own DELAY_SOURCE/
# LAG_MODE/LAG_DENSITY/LAG_STEP comments for the full rationale (identical
# here, ported unchanged -- pure column-name/DataFrame machinery with no
# TSLS-specific behavior).
#   'explicit' (legacy): the hand-picked DELAY_SET list below.
#   'lag_mode' (default, recommended): auto-derived from LAG_MODE/
#     LAG_DENSITY/LAG_STEP below instead -- DELAY_SET is ignored.
DELAY_SOURCE="lag_mode"       # default: lag_mode. Other value: explicit
# LAG_MODE (only used when DELAY_SOURCE=lag_mode) sets the *bound* (deepest
# lag magnitude, in rows) the auto-derived delay set reaches:
#   'generation_length_adjusted' (default): bound = GENERATE_LENGTH below.
#   'every_lag_possible': bound = as deep as the MDE library segment
#     structurally allows -- exhaustive, and correspondingly the more
#     expensive of the two.
LAG_MODE="generation_length_adjusted"   # default: generation_length_adjusted. Other value: every_lag_possible
# LAG_DENSITY (only used when DELAY_SOURCE=lag_mode) samples LAG_MODE's
# bound at LAG_STEP-row intervals ('strided', default) rather than every
# single integer lag ('dense').
LAG_DENSITY="strided"         # default: strided. Other value: dense
# LAG_STEP=1 makes LAG_MODE's bound dense regardless of LAG_DENSITY above
# (every consecutive row is its own candidate delay). Cost is real and
# load-bearing on --cpus-per-task/--mem/--time above -- see
# simple-mde-gmn.sbatch's own LAG_STEP comment for the full rationale.
LAG_STEP=4                    # default: 5, only used when LAG_DENSITY=strided
# Explicit delay list -- only used when DELAY_SOURCE=explicit above.
DELAY_SET="0,-1,-3,-5,-10"
# NOTE: STAGE2_ALLOW_REAL_OBSERVABLES is retired (2026-09-30). Stage 2's
# candidate pool is lagged columns only, and every lagged pick becomes a leaf
# node, so there is nothing for Stage 2 to select recursively. load_config()
# REJECTS the env var rather than ignoring it, so re-adding it here fails the
# run loudly instead of quietly changing nothing.

# ---- Lib/pred split ----------------------------------------------------------
# Three-way split of the full (TSLS-converted, burn-in-trimmed) dataset --
# ported from simple-mde-gmn.py's own three-way A/B/end split (see that
# file's load_config() docstring), computed against the row count AFTER
# TSLS conversion + burn-in trim, not the raw file's row count -- the
# usable window is itself data-driven (depends on which channel fires
# slowest), not a constant. TSLS is undefined (NaN) before each channel's
# own first spike, so the usable window starts only once every included
# channel has fired at least once -- that trim happens inside the script,
# before this ratio is applied.
LIB_RATIO=0.7                  # default: 0.7, fraction of usable rows used as library (A)
# Fraction of the *post-library* remainder given to the MDE validation
# segment [A, B) rather than the Generate/Forecast validation segment
# [B, end) -- see simple-mde-gmn.sbatch's own MDE_VALIDATION_RATIO comment
# for the full rationale (identical here).
MDE_VALIDATION_RATIO=0.5        # default: 0.5

# Number of steps each GMN Generate() call free-runs for, decoupled from
# the LIB_RATIO/MDE_VALIDATION_RATIO-derived held-out window -- see
# simple-mde-gmn.sbatch's own GENERATE_LENGTH comment for the full
# rationale (identical here). 0 means "use the entire held-out window".
GENERATE_LENGTH=400             # MID-SCALE CALIBRATION RUN -- job 38976891's own value, see NOTE above -- default: 0 (= full held-out window)

# Theiler-window / neighbor-exclusion radius for GMN's own recursive
# Generate()/Forecast() SMap nearest-neighbor search -- a PLAIN STATIC
# integer (confirmed against tsls-mde-gmn.py's build_gmn_config() docstring
# and every consumption site: no autodiscovery logic anywhere), NOT the
# dynamic/auto mechanism EXCLUSION_RADIUS above uses -- these are two
# independent EDM contexts (MDE's forward selection vs. GMN's own
# Generate()/Forecast()) with their own separate embeddings. Deliberately
# not TSLS_MDE_GMN_-prefixed, matching simple-mde-gmn.sbatch's own
# GMN_EXCLUSION_RADIUS naming.
GMN_EXCLUSION_RADIUS=0            # default: 0. Deliberately not TSLS_MDE_GMN_-prefixed

# Plotted x-axis range for the final Generate()-vs-Forecast()-vs-actual
# diagnostic figure(s) -- see simple-mde-gmn.sbatch's own
# FORECAST_PLOT_RANGE comment for the full behavior of each option
# (identical here, ported unchanged).
FORECAST_PLOT_RANGE="windowed"        # default: full. Other value: windowed

# ---- Generate()/Forecast() start-point sweep --------------------------------
# Tries starting Generate() (and, to match it, Forecast()) at several rows
# instead of always the Generate/Forecast validation segment's own start
# (B), and keeps whichever one best matches the real held-out data -- see
# simple-mde-gmn.sbatch's own GENERATE_START_MIN/MAX/STEP_OFFSET comment
# for the full rationale (identical here, ported unchanged -- this sweep
# doesn't exist in this pipeline's own previous architecture at all).
# Defaults to MIN=0, MAX=1 -- only offset 0. Widen MAX to actually sweep.
GENERATE_START_MIN_OFFSET=0
GENERATE_START_MAX_OFFSET=20
GENERATE_START_STEP=1

# ---- E/tau search (MODE=GMN only) --------------------------------------------
# Used exclusively by that mode's own interaction-matrix construction
# (data-driven E/tau grid search over the network's driver pool via
# proper_ccm) -- MODE=MDE-GMN has no network-wide E/tau at all. Also feeds
# run_theta_search()'s own theta retuning range regardless of mode
# (THETA_RANGE below). E_MIN must stay >= 2 (SMap's own predictor-exclusion
# convention needs E >= 2 even for leaf/undriven nodes).
E_MIN=3      # lower bound (inclusive) of embedding-dimension E search; must stay >= 2
E_MAX=13     # upper bound (EXCLUSIVE -- python range()) of E search
E_STEP=4     # step between candidate E values
TAU_MIN=-9   # lower bound (inclusive) of time-delay tau (in samples) search
TAU_MAX=0    # upper bound (EXCLUSIVE -- python range()) of tau search
TAU_STEP=4   # step between candidate tau values
# Theta (SMap's nonlinearity/localization parameter) search range, used by
# run_theta_search() for both MODE=MDE-GMN and MODE=GMN -- replaces this
# pipeline's own previous SMAP_SEARCH_MODE='coarse_retune'/'joint' combined
# (E, tau, theta) grid search entirely (E and every driver's delay now come
# from the network itself). Comma-separated list of candidate theta values.
THETA_RANGE="0.01,0.1,0.3,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9"

# ---- TSLS-aware scoring -------------------------------------------------------
# See tsls-mde-gmn.py's load_config()'s METRIC docstring for the full
# rationale. 'f1' (recommended for TSLS/spike-timing data) scores
# predictions by decoding both true and predicted series back to discrete
# spike/reset events (via a near-zero threshold calibrated from the true
# series itself) and matching them within an adaptive per-event tolerance
# window -- i.e. it specifically credits getting spike *timing* right, not
# just overall curve shape (which plain correlation over the whole TSLS
# sawtooth would reward even for the uninformative ramp segments between
# spikes). Other values: 'gamma' (Kistler coincidence factor, alternative
# event-timing metric), 'correlation' (plain Pearson rho), 'mae'.
METRIC="f1"
# TARGETS (head_position_x/y) are continuous behavioral traces with no
# genuine spike/reset structure -- 'f1'/'gamma' event-decoding on them
# produced NaN for every candidate in production. Always 'correlation' (or
# 'mae'); this is NOT the knob to change for TSLS-aware scoring, use METRIC
# above for that.
TARGET_METRIC="correlation"
GAMMA_DELTA=2                  # coincidence window (bins), only used when METRIC=gamma
# Bounds the row count fed into every CCM convergence gate in this
# pipeline (proper_mde's own CCM step, and build_interaction_matrix()'s) --
# CRITICAL for calibration, not just a speed knob. Both gates require
# skill to visibly RISE as library size grows from small to large; too
# large a CCM_WINDOW_ROWS means even the smallest tested library is
# already past where real relationships in this data saturate, so the
# gate finds no rise regardless of whether a real link exists -- confirmed
# as the direct cause of two separate production failures
# (apps.InteractionMatrix reporting ~all-zero interactions at job 35513363;
# proper_mde selecting zero drivers for either target at job 35525431).
# Recalibrated by testing genuinely small (tens of rows) library sizes
# directly against real data: a real, smooth, directionally-asymmetric
# convergence curve appeared in the ~15-400 row range. CCM_WINDOW_ROWS=800
# with the percentages below reproduces that validated range -- don't
# widen this back toward "percentage of the full ~30-45k-row dataset"
# without re-validating against real small-libSize data first, and keep
# the smallest resulting libSize >= (max E in E_RANGE)+2.
# MODE='GMN' only now (build_interaction_matrix()) -- MODE='MDE-GMN' (this
# pipeline's own default) no longer uses either of these two; see the
# window-sizing block below.
CCM_WINDOW_ROWS=800
CCM_LIB_SIZE_PCTS="2,3,5,8,12,20,30,50"  # -> ~16,24,40,64,96,160,240,400 rows at CCM_WINDOW_ROWS=800
CCM_SAMPLE=20                    # random-library resamples averaged per libSize -- higher = less noisy, slower
CCM_SLOPE=0.01                   # convergence-slope threshold, used by both proper_mde's own CCM gate and build_interaction_matrix()

# ---- Window sizing (MODE='MDE-GMN' only, docs/review-findings.md) -----------
# Replaces CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS above for run_mde_for_target()'s/
# run_multi_mde_for_targets()'s own ranking-stage/E-select window and the CCM
# gate's own library sweep -- decoupled from each other, per-node rather than
# one fixed global window. See load_config()'s own docstrings for the full
# reasoning and calibration evidence behind each default below.
TARGET_RANKING_WINDOW_ROWS=1000  # depth-0 (behavioral target) seed's own fixed window, arbitrary
# MIN_PRED_EVENTS: how many real reset events the target must have in a
# candidate window's own PRED portion before f1 is treated as signal, not
# noise. Empirically calibrated (r9_ccm_window_e_calibration.py's own
# MIN_PRED_EVENTS section, docs/review-findings.md): a smooth ~1/sqrt(N)
# noise-floor decay with no natural elbow, so 20 is a threshold choice
# (std~0.07, ~3x below CROSS_MAP_RHO_MIN's 0.2) justified by the
# cost/reachability tradeoff (only 3/45 real channels unreachable at this
# value, vs 8/45 and 20/45 at 40/80) -- not a discovered stabilisation point.
MIN_PRED_EVENTS=20
MAX_WINDOW_ROWS=30000             # compute ceiling on the search above -- not calibrated, just a backstop
CCM_LIB_SIZES_ROWS="16,24,40,64,160,400,800"  # absolute row counts for the CCM gate's own library sweep, never a percentage now

# ---- MDE-only diagnostics (skipped when MODE=GMN) ---------------------------
MAX_TP=20                        # max forecast horizon (steps) for the Tp-vs-rho sweep
GRID_RHO_THRESHOLD=0.9           # highlight threshold for the univariate E/tau grid figure
RUN_MDE_DIAGNOSTICS=true         # default: true. Set false to skip the MDE-only diagnostic figures entirely
# RUN_MDE_EVALUATE: gates ONLY run_mde_evaluate() (R19's PCA/diffusion-map
# baseline comparison) -- deliberately separate from RUN_MDE_DIAGNOSTICS
# above, which still gates run_tp_vs_rho_sweep()/run_univariate_e_tau_grid()
# regardless of this setting. Default false: docs/review-findings.md's
# "diffusion-map crash and hang" finding -- pydiffmap's own diffusion-map
# fit, on real TSLS data, has been observed to both raise (a ValueError,
# reproduced in 1.8s at an 800-row scale) and hang indefinitely (past 74
# CPU-minutes with no completion, at a real ~31,000-row scale), and the
# mechanism for the hang specifically is NOT yet diagnosed. Both an
# exception guard and a wall-clock bound now exist around this call for
# whoever turns it back on, but the toggle is what actually protects a run
# while it stays off. R19 is optional tooling, not part of the causal
# claim -- skipping it is a legitimate configuration, not a workaround, and
# its absence is logged once per run rather than silent either way.
RUN_MDE_EVALUATE=false           # default: false. See the finding above before setting true

# ---- Parallelism -------------------------------------------------------------
# NOTE: no TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES here -- unlike
# simple-mde-gmn.sbatch's own MODE=GMN branch (still shells out to
# apps.InteractionMatrix, whose own outer pool that knob sizes), this
# pipeline's MODE=GMN path uses proper_ccm, which -- like proper_mde -- has
# no multiprocessing pool of its own at all, so there's nothing for a knob
# like this to size. See this file's own NOTE at the top and
# tsls-mde-gmn.py's own module docstring.

# ---- Search reuse -------------------------------------------------------------
# The per-node theta search above (one Generate() call per network, the
# single most expensive part of this pipeline) is entirely deterministic
# given the same network + data -- see simple-mde-gmn.sbatch's own
# REUSE_BEST_PARAMS comment for the full rationale (identical here, ported
# unchanged). Set REUSE_BEST_PARAMS=true to have the pipeline look for a
# matching past run's cache file under CACHE_ROOT before searching -- an
# exact match skips that stage entirely; no match just runs the full
# computation, so this is always safe to leave on.
REUSE_BEST_PARAMS=true      # default: false. Set true to reuse a past exact-match search instead of re-running it

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: only needed if the cluster doesn't expose Python 3 in PATH by
# default on compute nodes.
module load python/3.11

# ADJUST: absolute path to your uv venv directory.
VENV_DIR="$HOME/pao_rotation/.venv"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
# gmn.GMN / PredictNonlinear already parallelize across SLURM_CPUS_PER_TASK
# worker *processes* (passed through as TSLS_MDE_GMN_CORES below); leaving
# BLAS/OpenMP free to also multithread within each worker oversubscribes
# the node, so pin those to 1 rather than mirroring SLURM_CPUS_PER_TASK the
# way a single-process job would.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/pao_rotation"

# ADJUST: <GROUP> to this lab's actual flash allocation. Defined here
# (rather than down in the /flash scratch setup section, where it used to
# live) because OUT_DIR needs it too -- same reasoning as
# simple-mde-gmn.sbatch: home storage is limited, so every output this
# pipeline produces now lands on /flash instead. Only JOB_LOG.md and the
# small .snapshots/ job-description cache stay on $PROJ_DIR -- both are
# small, human/git-facing provenance records, not pipeline output data.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

# Live for the run's whole duration (not the ephemeral per-job tempdir
# further below, which is deleted on every exit including a timeout) --
# matches $FLASH_BASE/pao_rotation_results, the literal path the #SBATCH
# --output/--error directives above are hardcoded to. Only persistent past
# this job's own lifetime on a crash/timeout, though, not on a clean exit
# -- see the `cleanup` EXIT trap below and OUTPUT_ARCHIVE_DIR's own comment
# for the full archive-then-clear behavior on success.
OUT_DIR="$FLASH_BASE/pao_rotation_results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ADJUST: durable cross-job cache pool for expensive, fingerprint-reusable
# pipeline stages (MDE forward-selection, network construction, theta
# search, generate-start sweep -- see tsls-mde-gmn.py's own module
# docstring's SEARCH REUSE section). Deliberately NOT under $OUT_DIR
# (/flash) or $PROJ_DIR (/home) -- a growing pool of machine-only cache
# artifacts across every run ever submitted, not this project's own
# human-facing per-job output; bucket storage has the headroom (and
# durability -- unlike /flash, not subject to the same scratch-cleanup
# expectations) for that instead. Separate cache pool from
# simple-mde-gmn.sbatch's own CACHE_ROOT -- these two pipelines' cache
# artifacts must never collide, even by coincidence of network_id/node
# naming, since they're fingerprinted against genuinely different code and
# data.
CACHE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache"
RUN_CACHE_DIR="$CACHE_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_CACHE_DIR"

# Separate from CACHE_ROOT/RUN_CACHE_DIR above -- that pool holds only the
# fingerprinted, cross-job-reusable *cache* artifacts, not this run's
# actual deliverables (network edge CSVs, every diagnostic PNG, Generate()
# output CSVs, fidelity plots -- everything else OUT_DIR collects). This is
# the full-output durable archive instead: on a successful exit, the
# `cleanup` trap below copies everything under $OUT_DIR here, then deletes
# the /flash copy -- /flash is scratch, not permanent storage, now that
# bucket has the headroom, while still staying in place on /flash (not
# archived, not deleted) if the job crashes/times out/gets cancelled, so a
# failed run's partial output is still there to inspect. Same architecture
# as simple-mde-gmn.sbatch's own OUTPUT_ARCHIVE_DIR, separate pool.
OUTPUT_ARCHIVE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs"
OUTPUT_ARCHIVE_DIR="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ARCHIVE_DIR"

# ==============================================================================
# SLURM LOG RELOCATION + OUTPUT ARCHIVING
# ==============================================================================
# The #SBATCH --output/--error above write to $FLASH_BASE/
# pao_rotation_results/ (flat, not $OUT_DIR -- see the ADJUST comment on
# those directives for why). Move them into $OUT_DIR once it exists, via a
# single combined EXIT trap (merged with the /flash tempdir cleanup trap
# below, since bash only honors the most recently registered EXIT trap).
# Runs on every exit path, including failure/timeout/cancel.
#
# Also decides $OUT_DIR's own fate, based on PY_STATUS (set around the
# `python3 -u tsls-mde-gmn.py` call below -- unset/non-"0" here means
# either the pipeline itself failed, or this trap fired before that line
# ever ran): "0" means the pipeline completed cleanly, so this run's full
# output is copied to the durable OUTPUT_ARCHIVE_DIR (bucket) and the
# /flash copy is then removed; anything else leaves $OUT_DIR exactly where
# it is on /flash, untouched, for post-mortem inspection -- crashes/OOMs/
# timeouts are exactly the case where you want that data still sitting on
# fast, directly-browsable storage rather than needing to dig it out of the
# archive. Same mechanism as simple-mde-gmn.sbatch's own `cleanup` trap.
SLURM_OUT="$FLASH_BASE/pao_rotation_results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$FLASH_BASE/pao_rotation_results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
# How the job actually ended, appended to the progress-monitoring log
# (RESULTS_DIR/progress.log -- see tsls-mde-gmn.py's own _log_progress()
# docstring) so a crash/timeout/OOM has an explicit, unambiguous final
# line instead of the file just going silent. This is the one piece of
# that log that CANNOT be written by Python itself: a hard kill means the
# Python process never gets to write anything of its own. `$?` alone
# (PY_STATUS below) can't distinguish OOM from a plain crash from a
# timeout, so this queries sacct -- SLURM's own authoritative record --
# with a couple of retries since its accounting can lag a few seconds
# right at job-end. Honest limitation, not hidden: a true instantaneous
# whole-cgroup SIGKILL (rare) can prevent this trap itself from running --
# `sacct -j <job_id>` run separately, after the fact, is the authoritative
# fallback in that case regardless of what this managed to write.
log_final_outcome() {
    local progress_log="$OUT_DIR/progress.log"
    [ -f "$progress_log" ] || return 0
    local slurm_state=""
    for _ in 1 2 3; do
        slurm_state="$(sacct -j "$SLURM_JOB_ID" --format=State -P -n 2>/dev/null | head -1 | tr -d ' ')"
        [ -n "$slurm_state" ] && break
        sleep 2
    done
    echo "$(date -Iseconds) JOB ENDED: PY_STATUS=${PY_STATUS:-unset} SLURM_STATE=${slurm_state:-unknown (sacct had no record yet -- check \`sacct -j $SLURM_JOB_ID\` manually)}" \
        >> "$progress_log"
}
cleanup() {
    log_final_outcome
    [ -f "$SLURM_OUT" ] && mv -f "$SLURM_OUT" "$OUT_DIR"/
    [ -f "$SLURM_ERR" ] && mv -f "$SLURM_ERR" "$OUT_DIR"/
    if [ "${PY_STATUS:-1}" = "0" ] && [ -d "$OUT_DIR" ]; then
        if cp -a "$OUT_DIR"/. "$OUTPUT_ARCHIVE_DIR"/; then
            rm -rf "$OUT_DIR"
        else
            echo "WARNING: archive copy from $OUT_DIR to $OUTPUT_ARCHIVE_DIR failed -- leaving the /flash copy in place rather than risk deleting the only one." >&2
        fi
    fi
    [ -n "${tempdir:-}" ] && rm -rf "$tempdir"
    return 0
}
trap cleanup EXIT

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Diffs this run's tsls-mde-gmn.py + tsls-mde-gmn.sbatch against the
# rolling snapshot cache of the last tsls-mde-gmn submission
# ($PROJ_DIR/results/.snapshots/ -- keyed by filename, so this never gets
# compared against a simple-mde-gmn job by mistake), inserting a new entry
# into $PROJ_DIR/JOB_LOG.md, then updates that cache to this run's files.
# Runs before the pipeline itself so the entry is written even if the run
# later fails or times out. In the common case (submitted via submit.sh)
# this call is a no-op -- submit.sh already wrote this job's JOB_LOG.md
# entry at submission time -- but still runs here so a job started via
# bare `sbatch` still gets logged.
#
# sbatch_src is "$0", not a hardcoded path to this canonical file -- sbatch
# copies whatever file it's given verbatim into the running job, and $0
# reflects that actual submitted path, which matters if the live canonical
# file at scripts/batch/tsls-mde-gmn.sbatch gets edited again while this
# job sits queued.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/tsls-mde-gmn.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# FLASH_BASE itself is defined earlier now (OUT_DIR needs it too); only the
# ephemeral per-job tempdir is set up here. Only ever holds the staged
# source/helper-module/data copy below -- actual pipeline output no longer
# passes through here at all (see TSLS_MDE_GMN_RESULTS_DIR below), so this
# directory being wiped on every exit, including a timeout, is not a
# data-loss concern.
tempdir="$(mktemp -d "${FLASH_BASE}/run_tsls_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
# Cleanup on exit (including this tempdir) is handled by the combined
# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION + OUTPUT
# ARCHIVING section above -- deliberately not a second `trap ... EXIT`
# here, which would silently replace that one instead of stacking.

# ==============================================================================
# COPY SOURCE FILE + HELPER MODULE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/scripts/src/tsls-mde-gmn.py" \
    "$PROJ_DIR/scripts/src/cleaned_helper_functions.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export TSLS_MDE_GMN_DATA_DIR="$tempdir/data"
export TSLS_MDE_GMN_DATA_FILE="${DATA_FILE}"
# Points directly at OUT_DIR (not a subdirectory of the ephemeral
# $tempdir) -- every pipeline output, including the theta search's/
# generate-start sweep's own intra-job checkpoint files, survives a
# timeout/crash rather than being deleted with $tempdir before ever
# reaching $OUT_DIR. What happens to OUT_DIR itself after the run --
# archived to bucket and cleared from /flash on success, left alone on
# /flash on a crash/timeout -- is the `cleanup` EXIT trap's job (see
# OUTPUT_ARCHIVE_DIR above); nothing here needs to change based on that.
export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR"
export TSLS_MDE_GMN_MODE="${MODE}"
export TSLS_MDE_GMN_GMN_MODE="${GMN_MODE}"
export TSLS_MDE_GMN_TARGETS="${TARGETS}"
export TSLS_MDE_GMN_D="${MDE_D}"
export TSLS_MDE_GMN_TP="${TP}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN="${CCM_PREFILTER_RHO_MIN}"
export TSLS_MDE_GMN_CCM_PREFILTER_E="${CCM_PREFILTER_E}"
export TSLS_MDE_GMN_LIBSIZE_RATIOS="${LIBSIZE_RATIOS}"
export TSLS_MDE_GMN_MULTI_MDE_E="${MULTI_MDE_E}"
export TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE="${MULTI_MDE_TARGET_AGGREGATE}"
export TSLS_MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
export TSLS_MDE_GMN_NESTED_M="${NESTED_M}"
export TSLS_MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export TSLS_MDE_GMN_DELAY_SOURCE="${DELAY_SOURCE}"
export TSLS_MDE_GMN_LAG_MODE="${LAG_MODE}"
export TSLS_MDE_GMN_LAG_DENSITY="${LAG_DENSITY}"
export TSLS_MDE_GMN_LAG_STEP="${LAG_STEP}"
export TSLS_MDE_GMN_DELAY_SET="${DELAY_SET}"
export TSLS_MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export TSLS_MDE_GMN_MDE_VALIDATION_RATIO="${MDE_VALIDATION_RATIO}"
export TSLS_MDE_GMN_GENERATE_LENGTH="${GENERATE_LENGTH}"
export GMN_EXCLUSION_RADIUS="${GMN_EXCLUSION_RADIUS}"
export TSLS_MDE_GMN_FORECAST_PLOT_RANGE="${FORECAST_PLOT_RANGE}"
export TSLS_MDE_GMN_GENERATE_START_MIN_OFFSET="${GENERATE_START_MIN_OFFSET}"
export TSLS_MDE_GMN_GENERATE_START_MAX_OFFSET="${GENERATE_START_MAX_OFFSET}"
export TSLS_MDE_GMN_GENERATE_START_STEP="${GENERATE_START_STEP}"
export TSLS_MDE_GMN_E_MIN="${E_MIN}"
export TSLS_MDE_GMN_E_MAX="${E_MAX}"
export TSLS_MDE_GMN_E_STEP="${E_STEP}"
export TSLS_MDE_GMN_TAU_MIN="${TAU_MIN}"
export TSLS_MDE_GMN_TAU_MAX="${TAU_MAX}"
export TSLS_MDE_GMN_TAU_STEP="${TAU_STEP}"
export TSLS_MDE_GMN_THETA_RANGE="${THETA_RANGE}"
export TSLS_MDE_GMN_MAX_TP="${MAX_TP}"
export TSLS_MDE_GMN_GRID_RHO_THRESHOLD="${GRID_RHO_THRESHOLD}"
export TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS="${RUN_MDE_DIAGNOSTICS}"
export TSLS_MDE_GMN_RUN_MDE_EVALUATE="${RUN_MDE_EVALUATE}"
export TSLS_MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export TSLS_MDE_GMN_METRIC="${METRIC}"
export TSLS_MDE_GMN_TARGET_METRIC="${TARGET_METRIC}"
export TSLS_MDE_GMN_GAMMA_DELTA="${GAMMA_DELTA}"
export TSLS_MDE_GMN_CCM_WINDOW_ROWS="${CCM_WINDOW_ROWS}"
export TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
export TSLS_MDE_GMN_CCM_SAMPLE="${CCM_SAMPLE}"
export TSLS_MDE_GMN_CCM_SLOPE="${CCM_SLOPE}"
export TSLS_MDE_GMN_TARGET_RANKING_WINDOW_ROWS="${TARGET_RANKING_WINDOW_ROWS}"
export TSLS_MDE_GMN_MIN_PRED_EVENTS="${MIN_PRED_EVENTS}"
export TSLS_MDE_GMN_MAX_WINDOW_ROWS="${MAX_WINDOW_ROWS}"
export TSLS_MDE_GMN_CCM_LIB_SIZES_ROWS="${CCM_LIB_SIZES_ROWS}"
export TSLS_MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
# Separate from TSLS_MDE_GMN_RESULTS_DIR (also persistent now, but per-job
# and on /flash) -- these point at the durable *cross-job* bucket cache
# pool set up above so a later job can actually find a past run's cached
# results to reuse, not just this run's own.
export TSLS_MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export TSLS_MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

python3 -V

# Deliberately not left under `set -e` for this one call -- PY_STATUS
# needs to be captured either way (success or failure) so the `cleanup`
# EXIT trap above knows whether to archive-and-clear $OUT_DIR or leave it
# on /flash; under plain `set -e`, a nonzero exit here would jump straight
# to the trap without ever reaching the capture line below.
set +e
python3 -u tsls-mde-gmn.py
PY_STATUS=$?
set -e

if [ "$PY_STATUS" -ne 0 ]; then
    echo "tsls-mde-gmn.py exited with status ${PY_STATUS} -- leaving output on /flash at ${OUT_DIR} for inspection (not archived to bucket)." >&2
    exit "$PY_STATUS"
fi

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
# No cp step needed here anymore -- TSLS_MDE_GMN_RESULTS_DIR points
# directly at $OUT_DIR now, so the pipeline has already been writing
# straight to /flash throughout the run. What happens next -- archived to
# bucket and removed from /flash, since PY_STATUS=0 here means the run
# completed cleanly -- is handled by the `cleanup` EXIT trap above.

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Archiving outputs to bucket and clearing the /flash copy:"
echo "  ${OUT_DIR} -> ${OUTPUT_ARCHIVE_DIR}"
echo "========================================================"
