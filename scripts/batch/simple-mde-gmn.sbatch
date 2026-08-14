#!/bin/bash
#SBATCH --job-name=run_simple_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=4-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
# ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
# parses #SBATCH directives itself before this script (or any shell)
# runs, so no shell variable expansion happens here. Pointed directly at
# results/ (a fixed path that already exists) rather than results/job_%j/
# because Slurm opens these files before the job starts, i.e. before
# $SLURM_JOB_ID is known and before this script's own `mkdir -p
# "$OUT_DIR"` below could create that subdirectory -- a per-job
# subdirectory in this path would fail with "couldn't open output file"
# since Slurm doesn't create missing directories. The EXIT trap further
# down (search SLURM LOG RELOCATION) moves these into $OUT_DIR once it
# exists, so they still end up alongside the rest of that job's output.
#SBATCH --output=/home/j/j-tominaga/pao_rotation/results/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/pao_rotation/results/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
# NOTE ON --time/--mem/--cpus-per-task ABOVE: this pipeline now includes a
# data-driven CCM interaction-matrix step and a per-node Generate()-based
# theta search -- both can legitimately take hours, not minutes (the old
# 1-hour budget this file used to have was sized for a much simpler
# fixed-parameter pipeline). These are starting points, not maxed-out
# requests against this partition's limits (40-128 cores/node, 500G
# mem/node, 4-day cap):
#   --cpus-per-task=32: most of this pipeline's parallel steps (dimx.MDE,
#     EmbedDimension, PredictNonlinear, gmn.GMN) scale close to linearly
#     with cores with no nesting issue. The one exception is the CCM
#     interaction-matrix step -- see INTERACTION_MATRIX_OUTER_CORES and
#     FIXED CAVEAT below for why *that* step doesn't benefit from a
#     larger --cpus-per-task past a point, and why raising it further is
#     mostly wasted allocation rather than wasted time.
#   --mem=112G: this pipeline's actual data (a few hundred channels x a
#     few thousand rows) is a few MB; memory use is driven by per-process
#     overhead (~100-150MB/worker observed) at whatever process count
#     --cpus-per-task and INTERACTION_MATRIX_OUTER_CORES produce, not by
#     data size. Was 48G, but job 36515744 (REGION_CHANNEL_CAP=999,
#     NESTED_M=3, MDE_D=5, --cpus-per-task=32) was OOM-killed at ~49.75G
#     RSS after ~5h42m -- that combination fans out more concurrent
#     nested-MDE/theta-search workers than the original 48G estimate
#     assumed, so per-process overhead alone pushed past the request with
#     almost no headroom. Raised to 112G for real margin above observed
#     usage without requesting enough to slow down scheduling (larger
#     --mem requests generally mean fewer eligible nodes and a longer
#     queue wait).
#   --time=2-00:00:00: a generous but not maxed (4-day cap) starting
#     point. Tighten it once you've timed an actual run of your chosen
#     ranges below -- e.g. via a cheap calibration job with a small
#     REGION_CHANNEL_CAP and a 1x1 E/tau search, timed end-to-end, then
#     scaled up to your real channel count / search grid size.
# See FIXED CAVEAT further down for why wall time won't scale linearly
# with --cpus-per-task regardless.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_1_Hz.csv"   # default: J1620210605_epoch_8_freq_4_Hz.csv

# ---- Overall pipeline shape ------------------------------------------------
# MODE=MDE-GMN: the network is built entirely from nested dimx.MDE calls
#   over a delay-expanded candidate pool (DELAY_SET below), no CCM
#   anywhere -- see build_nested_mde_network() in simple-mde-gmn.py for
#   the full algorithm. In short: each target's MDE-selected drivers (up
#   to MDE_D of them, each at its own selected delay) themselves become
#   new MDE targets in turn *only if selected at delay 0* (a driver
#   selected at a nonzero delay is a terminal placeholder edge, not a new
#   target -- see that function's docstring), for NESTED_M further
#   rounds, growing the network one level at a time; any driver that
#   would create a cycle or a self-loop is rejected as soon as it's
#   proposed, before it's ever written into the graph. Total network
#   depth from each target is 1 + NESTED_M levels. The same observable
#   can drive one node at several different delays at once (each its own
#   graph edge, visualized as separate colored arcs -- see the network
#   figure's legend).
# MODE=GMN: skip MDE entirely; GMN's interaction matrix is built directly
#   from the full REGION_CHANNEL_CAP-limited observable set via CCM
#   (Park et al. 2026 Section 2.3's standard construction). Much more
#   expensive -- see FIXED CAVEAT below. Unchanged from before the
#   nested-MDE rework above.
MODE="MDE-GMN"             # default: MDE-GMN. Other value: GMN
# GMN_MODE=combined: one network spanning every TARGETS entry + the union
#   of their drivers. GMN_MODE=separate: one independent network per target.
GMN_MODE="separate"           # default: combined. Other value: separate
TARGETS="head_position_x,head_position_y"   # comma-separated target column names

# ---- MDE (variable selection; unused when MODE=GMN) ------------------------
# MDE_D counts distinct *real* observables, not delay-expanded candidate
# columns -- the pipeline internally multiplies it by len(DELAY_SET)
# before handing it to dimx.MDE as the actual dimension cap D, since each
# real observable spans len(DELAY_SET) separate (observable, delay)
# candidate columns and a naive D could otherwise be exhausted by several
# delays of the very same observable instead of covering MDE_D distinct
# ones (see effective_mde_dimension() in simple-mde-gmn.py). Setting
# MDE_D to a number >= the actual real-observable count in play (e.g. 999
# against 500 real observables) makes MDE consider all of them -- it gets
# capped down to that real count rather than inflating D past what the
# candidate pool can supply.
MDE_D=999                     # default: 20, max # of distinct real observables MDE can forward-select per target
# Prediction interval dimx.MDE forward-selection ranks/validates every
# candidate driver against -- every CrossMapColumns/EmbedDimension/CCM
# call inside run_mde_for_target() (both the top-level per-target search
# and every nested-MDE per-node search) uses this same value, so it
# shapes which drivers get selected, not just how they're scored
# afterward. dimx.MDE's own default (unset before this parameter existed)
# is 1 -- drivers were being chosen purely for 1-step-ahead skill, with
# nothing in the selection ever rewarding stability over a long recursive
# Generate() run (GENERATE_LENGTH below). See JOB_LOG.md's diagnosis of
# job 36786285 (fidelity rho ~0.03 / -0.13 despite R~0.98 at the Tp=1 MDE
# step) for why this is being swept alongside MDE_D.
TP=30                         # default: 1 (dimx.MDE's own default), prediction interval for MDE candidate ranking/validation
EMBED_DIM_RHO_MIN=0.1         # default: 0.2, CCM embedding-dimension rho gate
CROSS_MAP_RHO_MIN=0.1         # default: 0.2, simplex cross-map rho gate
# Theiler-window / neighbor-exclusion radius for every EDM nearest-
# neighbor search inside dimx.MDE (CrossMapColumns' candidate ranking,
# EmbedDimension's E search, and CCM's own causality-validation call --
# see simple-mde-gmn.py's monkeypatch block, right above
# _sequential_ccm_api, for exactly which calls this reaches and why).
# Guards against a candidate looking like a driver purely because
# nearby-in-time rows of a slowly-varying/autocorrelated signal are also
# nearby in embedding space -- not because of any real recurrence in the
# dynamics. The classic guideline is exclusionRadius >= E*tau, but this
# pipeline's nested-MDE embeddings are *generalized* (DELAY_SET lets
# different selected columns carry different delays), so E and tau are
# confounded rather than fixed scalars -- see
# _dynamic_exclusion_radius()'s docstring for the full reasoning.
# 'auto' (default, recommended): recomputes exclusionRadius fresh at
# every one of dimx.MDE's internal EDM calls, from that call's own
# actually-selected columns (dimension count x largest DELAY_SET-encoded
# delay among them) -- always exactly matches the embedding actually in
# play, growing across MDE's forward-selection steps as more columns get
# added, rather than one static a-priori value. Set an explicit integer
# instead to fall back to dimx.MDE's ordinary fixed-exclusionRadius
# behavior (0 reproduces this pipeline's behavior before this parameter
# existed -- no temporal exclusion at all beyond exact self-matches).
EXCLUSION_RADIUS="auto"   # default: auto. Set an integer (e.g. 0) to disable the dynamic per-call computation
# Nested-MDE network depth (MODE=MDE-GMN only -- ignored for MODE=GMN).
# MDE_D above doubles as N (the per-node driver cap, in real-observable
# terms) at every nesting level, not just the top level. NESTED_M is the
# number of *additional* nested-MDE rounds beyond each target's own
# initial selection -- total graph depth from each target is 1 + NESTED_M
# levels. NESTED_M=0 is a sentinel for "uncapped": nesting keeps
# expanding round after round with no fixed depth limit, until it runs
# out on its own -- every reachable real observable already visited, or
# every remaining branch dead-ends via dimx.MDE's own gates/cycle
# rejection -- rather than stopping after a fixed number of rounds.
# Combinatorially expensive: a naive upper bound on node count is roughly
# MDE_D**(1+NESTED_M) (or, with NESTED_M=0, bounded only by the real
# observable pool size) before dedup/cycle-rejection collapses it back
# down to the real (REGION_CHANNEL_CAP-limited) observable pool size --
# keep both MDE_D and NESTED_M small for a first test run (e.g. MDE_D=4,
# NESTED_M=1, REGION_CHANNEL_CAP=15, matching the scale of past
# small-scale validation runs of this pipeline) before scaling up to
# NESTED_M=0 for a full uncapped expansion.
NESTED_M=3                    # default: 1, additional nested-MDE rounds beyond each target's own selection. 0 = uncapped (expand until real observables are exhausted)
# First N channels per region (CA1_/mPFC_/OFC_) offered as MDE candidates,
# or (when MODE=GMN) used directly as the interaction-matrix's full node
# set. 999 is effectively uncapped (matches the full recording). Cap this
# down for a fast test job -- see FIXED CAVEAT below for why an uncapped
# MODE=GMN run is an O(channels^2) interaction matrix.
REGION_CHANNEL_CAP=999 # default: 999

# Delays (in rows, <= 0 -- 0 = contemporaneous, negative = that many rows
# in the past) nested-MDE searches per candidate observable (MODE=MDE-GMN
# only). Each observable is offered to dimx.MDE's forward selection as
# len(DELAY_SET) separate columns, one per delay, so MDE can pick
# specific (observable, delay) pairs -- e.g. 200 observables x 5 delays =
# 1000 candidate columns -- rather than being stuck with one fixed
# contemporaneous copy of every observable. This is a generalized/
# non-uniform Takens embedding (Deyle & Sugihara 2011): different driving
# variables, and even repeats of the *same* driver, can each carry their
# own delay instead of sharing one embedding depth/lag network-wide.
# Widening this multiplies the candidate pool nested-MDE searches (and
# so its cost) by len(DELAY_SET); keep it modest for a first test run.
DELAY_SET="0,-1,-3,-5,-10"

# ---- GMN Generate() window --------------------------------------------------
# Lib/pred split as a fraction of the full dataset (no more separate
# start-row/duration row counts): LIB_RATIO=0.7 means the first 70% of
# rows are library, the remaining 30% are the held-out pred/Generate()
# window. The entire input file is used -- row counts are derived inside
# the script from this ratio and the loaded data's length. This split is
# also what MDE variable selection and CCM/EmbedDimension validate
# against -- it does NOT control how many steps a single Generate() call
# actually simulates; see GENERATE_LENGTH below for that.
LIB_RATIO=0.75                  # default: 0.5, fraction of rows used as library

# Number of steps each GMN Generate() call free-runs for, decoupled from
# the LIB_RATIO-derived held-out window above. Every Generate() call
# (the theta search below, the start-point sweep, and the final rebuild)
# always still starts right where the library ends; only the *length* is
# independent. Generate() is the single most expensive step in this
# pipeline, so shrinking this alone (without also shrinking the held-out
# window MDE/CCM validate against) is the cheapest way to cut wall time.
# Also sizes the final Forecast() diagnostic's own window -- see
# FORECAST_PLOT_RANGE below for exactly how, since that now depends on
# which of its two modes is selected. 0 means "use the entire held-out
# window" (unchanged pre-existing behavior); a value larger than the
# held-out window is automatically clamped down to it (a Generate()
# longer than there's real data to score it against isn't meaningful).
GENERATE_LENGTH=300              # default: 0 (= full held-out window). Set e.g. 50-200 to speed up the theta search

# Theiler-window / neighbor-exclusion radius for gmn's own recursive
# Generate()/Forecast() SMap nearest-neighbor search -- passed straight
# through as Parameters.exclusionRadius to every pyEDM SMap call gmn
# itself makes (theta search, start-point sweep, final rebuild, and the
# Forecast() diagnostic all use it, via build_gmn_config()). Independent
# of MDE_GMN_EXCLUSION_RADIUS above, which only affects dimx.MDE's own
# forward-selection search -- these are two separate EDM contexts with
# their own separate embeddings. Unlike MDE's 'auto' mode, this is a
# plain static integer: every node here shares one theta-driven SMap
# embedding (see build_gmn_config()'s function='SMap' docstring), so
# there's no per-node generalized-embedding confound to dynamically
# track. 0 (default) matches this pipeline's original hardcoded
# behavior (no temporal exclusion beyond exact self-matches) -- raise it
# to test whether Generate()/Forecast() skill is being inflated by
# temporally-adjacent library points on this recording's autocorrelation
# timescale.
GMN_EXCLUSION_RADIUS=0            # default: 0. Deliberately not MDE_GMN_-prefixed -- separate from MDE_GMN_EXCLUSION_RADIUS above

# Plotted x-axis range for the final Generate()-vs-Forecast()-vs-actual
# diagnostic figure(s) (gmn_generate_forecast_<network_id>_partNN.png --
# already split across multiple files when a network has more nodes than
# fit in one figure, see run_forecast_diagnostic()'s nodes_per_figure).
# 'full' (default, unchanged pre-existing behavior): actual/lib plotted
# from row 0 through generate_start+GENERATE_LENGTH; Forecast() computes
# exactly GENERATE_LENGTH steps -- all three traces (actual, Generate(),
# Forecast()) share one right edge.
# 'windowed': narrows the window to 2x GENERATE_LENGTH on each side of
# generate_start (clamped to available data) -- e.g. GENERATE_LENGTH=300
# shows 600 rows before generate_start and extends to 600 rows after it.
# actual/lib fills the entire window (real data, always available, so it
# ends up spanning wider than Generate()'s own window); Forecast() is
# separately run out to that same 2x-GENERATE_LENGTH length just for
# this plot (a direct per-row SMap prediction against real held-out
# data, not a recursive simulation like Generate(), so extending it here
# is cheap and doesn't compound error) -- so its trace always spans
# exactly twice Generate()'s own window. Generate() itself (the
# recursive simulation everything else in this pipeline scores against)
# is NOT extended -- unchanged at GENERATE_LENGTH regardless of this
# setting. See run_forecast_diagnostic()'s plot_range_mode docstring for
# the full reasoning.
FORECAST_PLOT_RANGE="windowed"        # default: full. Other value: windowed

# ---- Generate()/Forecast() start-point sweep --------------------------------
# GMN's Generate() is a recursive nonlinear simulation, so which row it
# launches from can change how well it tracks reality, separately from
# how good its embedding/theta are -- the same way a chaotic system's
# trajectory depends on its initial condition. This sweep tries starting
# Generate() (and, to match it, Forecast()) at several rows instead of
# always PREDICTION_START, and keeps whichever one best matches the real
# held-out data, using the already-chosen best theta from the search
# below (not re-searched here).
# Candidates are expressed as offsets in rows from PREDICTION_START:
# MIN/MAX/STEP work like E_MIN/E_MAX/E_STEP below (MAX EXCLUSIVE). Every
# offset must be >= 0 (an offset before PREDICTION_START would compare
# against data the model's own library already used, which isn't a fair
# out-of-sample test) and leave enough rows for GENERATE_LENGTH to still
# fit inside the held-out window -- offsets outside that range are
# dropped automatically once the dataset is loaded.
# Defaults to MIN=0, MAX=1 -- only offset 0 (i.e. always start at
# PREDICTION_START), the same as before this sweep existed. Widen MAX to
# actually sweep; each extra offset costs one more full Generate() call,
# so widen gradually (e.g. STEP=50 across a few hundred rows) rather than
# scanning every row.
GENERATE_START_MIN_OFFSET=0
GENERATE_START_MAX_OFFSET=20
GENERATE_START_STEP=2

# ---- E/tau search (MODE=GMN only) --------------------------------------------
# Used exclusively by that mode's own CCM interaction-matrix construction
# (data-driven E/tau grid search over the network's driver pool) --
# MODE=MDE-GMN has no network-wide E/tau at all: each node's embedding
# comes from nested-MDE's own discovered per-driver delays (DELAY_SET
# above) instead, so these ranges are simply unused in that mode. E_MIN
# must stay >= 2 (SMap's own predictor-exclusion convention needs E >= 2
# even for leaf/undriven nodes).
E_MIN=3      # lower bound (inclusive) of embedding-dimension E search; must stay >= 2
E_MAX=16     # upper bound (EXCLUSIVE -- python range()) of E search; 16 itself is never tried
E_STEP=2     # step between candidate E values, e.g. 3,5,7,...,15 with the defaults above
TAU_MIN=-15  # lower bound (inclusive) of time-delay tau (in samples) search
TAU_MAX=0    # upper bound (EXCLUSIVE -- python range()) of tau search; 0 itself is never tried
TAU_STEP=2   # step between candidate tau values, e.g. -15,-13,...,-1 with the defaults above

# ---- Parallelism -------------------------------------------------------------
# Kept deliberately small and separate from --cpus-per-task -- see FIXED
# CAVEAT below for why raising this to match --cpus-per-task backfires
# (it multiplies against, rather than shares, pyEDM.CCM()'s own per-pair
# worker pool). 2-4 is a reasonable range; there's rarely a reason to go
# higher regardless of how many cores you request.
INTERACTION_MATRIX_OUTER_CORES=4

# ---- Search reuse -------------------------------------------------------------
# The per-node theta search above (one Generate() call per network, the
# single most expensive part of this pipeline) is entirely deterministic
# given the same network + data -- if a previous job already searched
# the exact same DATA_FILE/MODE/GMN_MODE/TARGETS/MDE_D/NESTED_M/
# REGION_CHANNEL_CAP/LIB_RATIO/GENERATE_LENGTH/DELAY_SET/E-tau/theta-range
# combination, redoing that search here is pure wasted compute. Set
# REUSE_BEST_PARAMS=true to have the pipeline look for a matching
# results/job_*/best_params_<network_id>.json from a past run before
# searching -- an exact match skips that network's theta retuning
# entirely and reuses its winning per-node theta; no match (different
# settings, or simply the first run with these settings) just runs the
# full search as normal, so this is always safe to leave on. Every run
# (fresh or reused) writes its own best_params_<network_id>.json, so a
# chain of jobs with identical settings only ever pays for the search
# once.
REUSE_BEST_PARAMS=true      # default: false. Set true to reuse a past exact-match search instead of re-running it

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: only needed if the cluster doesn't expose Python 3 in PATH by
# default on compute nodes. If your uv venv bundles its own Python
# interpreter (created with `uv python install` / `uv venv --python ...`)
# you may not need this line at all -- try commenting it out if you get
# "module not found" errors.
module load python/3.11

# ADJUST: absolute path to your uv venv directory.
VENV_DIR="$HOME/pao_rotation/.venv"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
# dimx.MDE / apps.InteractionMatrix / pyEDM's EmbedDimension and
# PredictNonlinear already parallelize across SLURM_CPUS_PER_TASK worker
# *processes* (passed through as MDE_GMN_CORES below); leaving BLAS/OpenMP
# free to also multithread within each worker oversubscribes the node, so
# pin those to 1 rather than mirroring SLURM_CPUS_PER_TASK the way a
# single-process job would.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# FIXED CAVEAT (root-caused after several full jobs each ran ~2 days
# regardless of dataset size, MODE, or MDE_D): apps.InteractionMatrix's
# own outer worker pool (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
# below) calls pyEDM.CCM() once per channel pair (dimx.MDE similarly once
# per candidate column), and CCM() used to spawn its *own* fresh
# multiprocessing pool sized to the full visible cpu_count() on every
# call, for work that only ever amounts to 4-8 tiny KDTree tasks -- an
# 8-core request could produce ~90 live processes, not 8, since the two
# pools didn't coordinate (peak processes was outer-cores x cpu_count()).
# With REGION_CHANNEL_CAP uncapped (~470 real channels) that's ~100k+
# CCM() calls each paying a full spawn-and-reimport cost -- process-spawn
# overhead dominating actual compute by orders of magnitude, which is why
# cutting DATA_FILE resolution or MDE_D didn't help wall time.
# scripts/src/simple-mde-gmn.py now monkeypatches pyEDM.CCM's inner pool
# away entirely (`_sequential_ccm_workers`), so this is no longer live --
# kept here for provenance. MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
# (default 2) is still a separate knob from --cpus-per-task since it's
# the one thing sizing real parallelism across column pairs now.
#
# REMAINING CAVEAT (unfixed, just made avoidable): even with the above
# fixed, the per-node theta search in run_theta_search() -- one
# GENERATE_LENGTH-step recursive Generate() call across every node -- is
# still the single most expensive part of this pipeline. That cost is
# unavoidable the *first* time a given network/data combination is
# searched, but resubmitting the identical combination used to redo it
# every time regardless. REUSE_BEST_PARAMS below skips it on an exact
# repeat -- see that parameter's own comment. (This used to also include
# a len(E) x len(tau) grid on top of the per-node theta retuning --
# removed per explicit direction once E and each driver's delay started
# coming from nested-MDE's own network instead of a separate search; see
# scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that version.)

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/pao_rotation"
OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ADJUST: durable cross-job cache pool for expensive, fingerprint-reusable
# pipeline stages (MDE forward-selection, nested-MDE/CCM network
# construction, theta search, generate-start sweep -- see the module
# docstring's SEARCH REUSE section and each stage's own docstring).
# Deliberately NOT under $PROJ_DIR/results (/home) -- that quota is meant
# for this project's own human-facing per-job output, not a growing pool
# of machine-only cache artifacts across every run ever submitted. Bucket
# storage has the headroom for that instead. CACHE_ROOT is the read side
# (searched recursively, across every past job's subdirectory, by
# find_cached_best_params()-style lookups); RUN_CACHE_DIR is this job's
# own write side -- each cacheable stage syncs its result here
# *immediately* upon completing (not just at job end like $OUT_DIR),
# so it survives even if this job later crashes or times out.
CACHE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/mde_gmn_cache"
RUN_CACHE_DIR="$CACHE_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_CACHE_DIR"

# ==============================================================================
# SLURM LOG RELOCATION
# ==============================================================================
# The #SBATCH --output/--error above write to $PROJ_DIR/results/ (flat,
# not $OUT_DIR -- see the ADJUST comment on those directives for why).
# Move them into $OUT_DIR once it exists, via a single combined EXIT trap
# (merged with the /flash tempdir cleanup trap below, since bash only
# honors the most recently registered EXIT trap -- a second `trap ...
# EXIT` later in this script would silently replace this one rather than
# stacking). Runs on every exit path, including failure/timeout/cancel,
# which is exactly when you most want the log sitting next to the rest
# of that run's output instead of loose in results/.
SLURM_OUT="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
cleanup() {
    [ -n "${tempdir:-}" ] && rm -rf "$tempdir"
    [ -f "$SLURM_OUT" ] && mv -f "$SLURM_OUT" "$OUT_DIR"/
    [ -f "$SLURM_ERR" ] && mv -f "$SLURM_ERR" "$OUT_DIR"/
    return 0
}
trap cleanup EXIT

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Diffs this run's simple-mde-gmn.py + simple-mde-gmn.sbatch against the
# rolling snapshot cache of the last simple-mde-gmn submission
# ($PROJ_DIR/results/.snapshots/ -- not a per-job copy under $OUT_DIR;
# see generate_job_description.sh's own module docstring for why),
# inserting a new entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes
# as a table, everything else as unified diffs), then updates that cache
# to this run's files. Runs before the pipeline itself so the entry is
# written even if the run later fails or times out.
#
# sbatch_src is "$0", not a hardcoded path to this canonical file: sbatch
# copies whatever file it's given verbatim into the running job, and $0
# reflects that actual submitted path (confirmed via `scontrol show job`'s
# Command= field) -- this matters if the live canonical file at
# scripts/batch/simple-mde-gmn.sbatch gets edited again while this job
# sits queued. A hardcoded canonical path here would silently diff
# whatever that file happens to contain *right now* instead of what this
# job actually launched with.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/simple-mde-gmn.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
# Cleanup on exit (including this tempdir) is handled by the combined
# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION section
# above -- deliberately not a second `trap ... EXIT` here, which would
# silently replace that one instead of stacking.

# NOTE: the theta search below checkpoints to disk after every network
# so a resubmitted job can resume instead of restarting a multi-hour
# search from scratch (see the script's run_theta_search()) -- but
# $tempdir lives under $FLASH_BASE and is deleted by the `trap` above
# the moment this script exits, including on a SLURM timeout/cancellation
# (SIGTERM), *before* results get copied to $OUT_DIR. If your search
# might not finish within --time, checkpoints will be lost across
# resubmissions under this scratch layout -- consider pointing
# MDE_GMN_RESULTS_DIR directly at a subdirectory of $OUT_DIR instead of
# $tempdir/simple_mde_gmn_outputs if that matters for your run (trades
# some I/O speed on /flash for surviving a timeout).

# ==============================================================================
# COPY SOURCE FILE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/scripts/src/simple-mde-gmn.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export MDE_GMN_DATA_DIR="$tempdir/data"
export MDE_GMN_DATA_FILE="${DATA_FILE}"
export MDE_GMN_RESULTS_DIR="$tempdir/simple_mde_gmn_outputs"
export MDE_GMN_MODE="${MODE}"
export MDE_GMN_GMN_MODE="${GMN_MODE}"
export MDE_GMN_TARGETS="${TARGETS}"
export MDE_GMN_D="${MDE_D}"
export MDE_GMN_TP="${TP}"
export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
export MDE_GMN_NESTED_M="${NESTED_M}"
export MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export MDE_GMN_DELAY_SET="${DELAY_SET}"
export MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export MDE_GMN_GENERATE_LENGTH="${GENERATE_LENGTH}"
export GMN_EXCLUSION_RADIUS="${GMN_EXCLUSION_RADIUS}"
export MDE_GMN_FORECAST_PLOT_RANGE="${FORECAST_PLOT_RANGE}"
export MDE_GMN_GENERATE_START_MIN_OFFSET="${GENERATE_START_MIN_OFFSET}"
export MDE_GMN_GENERATE_START_MAX_OFFSET="${GENERATE_START_MAX_OFFSET}"
export MDE_GMN_GENERATE_START_STEP="${GENERATE_START_STEP}"
export MDE_GMN_E_MIN="${E_MIN}"
export MDE_GMN_E_MAX="${E_MAX}"
export MDE_GMN_E_STEP="${E_STEP}"
export MDE_GMN_TAU_MIN="${TAU_MIN}"
export MDE_GMN_TAU_MAX="${TAU_MAX}"
export MDE_GMN_TAU_STEP="${TAU_STEP}"
export MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export MDE_GMN_INTERACTION_MATRIX_OUTER_CORES="${INTERACTION_MATRIX_OUTER_CORES}"
export MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
# Unlike MDE_GMN_RESULTS_DIR (pointed at $tempdir on /flash scratch,
# deleted at job exit -- see that export above / the /flash scratch
# setup section's own comment), these must point at the durable bucket
# cache pool set up above so a later job can actually find a past run's
# cached results to reuse, and so this run's own results survive even a
# crash partway through (see CACHE_ROOT/RUN_CACHE_DIR's own comment).
export MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

python3 -V

python3 -u simple-mde-gmn.py

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

cp -av simple_mde_gmn_outputs/. "$OUT_DIR"/

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Outputs saved to:"
echo "  ${OUT_DIR}"
echo "========================================================"
