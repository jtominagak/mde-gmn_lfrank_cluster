#!/bin/bash
#SBATCH --job-name=run_simple_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=64
# ADJUST: must be a literal absolute path, not a shell variable -- Slurm
# parses #SBATCH directives itself before this script (or any shell)
# runs, so no shell variable expansion happens here (this is why this
# can't just read $FLASH_BASE/pao_rotation_results below, even though
# that's the exact same literal value). Pointed directly at
# /flash/FukaiU/j-tominaga/pao_rotation_results/ (a fixed path that must
# already exist -- created once via `mkdir -p`, persists across jobs the
# same way $FLASH_BASE itself already does) rather than .../job_%j/
# because Slurm opens these files before the job starts, i.e. before
# $SLURM_JOB_ID is known and before this script's own `mkdir -p
# "$OUT_DIR"` below could create that subdirectory -- a per-job
# subdirectory in this path would fail with "couldn't open output file"
# since Slurm doesn't create missing directories. The EXIT trap further
# down (search SLURM LOG RELOCATION) moves these into $OUT_DIR once it
# exists, so they still end up alongside the rest of that job's output.
# On /flash, not /home (per explicit direction: home storage is limited,
# so every output this pipeline produces -- not just scratch -- now
# lands on /flash instead; see OUT_DIR/MDE_GMN_RESULTS_DIR below for the
# rest of that change) -- these two files alone can reach tens of MB per
# job on their own (job 37939058's .err was ~23MB, almost entirely
# pyEDM UserWarning spam from Stage 2's exhaustive per-node candidate
# pools), which was a real, measured contributor to the home-quota
# problem, not a hypothetical one.
#SBATCH --output=/flash/FukaiU/j-tominaga/pao_rotation_results/%x-%j.out
#SBATCH --error=/flash/FukaiU/j-tominaga/pao_rotation_results/%x-%j.err
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
# fixed-parameter pipeline).
#   --cpus-per-task=64, --mem=300G (both scaled back down from 128/500G --
#     job 38870714 sat queued under the 128/500G request with Reason=Priority
#     while it was the ONLY pending job on the entire compute partition;
#     `sinfo` at the time showed why -- 128 cores + 500G is a whole node,
#     and at that moment ZERO of this partition's 128-core nodes were fully
#     idle (all either partially "mix"-allocated or "drng"/draining), so
#     the job needed one to drain completely rather than ever fitting into
#     existing free capacity. 64 cores + 300G, by contrast, already fit on
#     ~23 currently-"mix" nodes at that same moment -- OverSubscribe=OK on
#     this partition means jobs share nodes rather than needing exclusive
#     ones, so a request sized to fit alongside other jobs schedules far
#     sooner than one that needs an entire node to itself). This reduction
#     is grounded in actual measured usage, not just scheduling
#     convenience: the one full run that completed cleanly on this
#     pipeline's architecture (job 36786285, pre-two-stage but the closest
#     real comparison available) peaked at ~58G RSS on 32 cores; even
#     job 37946756 -- the MDE_D=999 pathological run (200-300+ drivers
#     selected per node) that got 1d11h into its very first depth-1 wave
#     at --cpus-per-task=128 before hitting the (now-fixed) all-NaN crash,
#     not an OOM -- peaked at only ~135G RSS and averaged just ~57 of its
#     128 allocated cores actually busy (TotalCPU/Elapsed), because
#     build_nested_mde_network()'s/attach_lag_embeddings()'s outer(per-node)
#     x inner(per-node's own CrossMapColumns) core split is bounded by
#     however many nodes a given wave actually contains and by MDE_D
#     itself, not by --cpus-per-task alone -- so above some point, more
#     cores requested just means more idle ones. MDE_D=30 (down from the
#     999 that produced that 135G/57-core reading) shrinks the very
#     candidate pool that split parallelizes over, so actual usage at this
#     config should be well under even that pathological run's numbers;
#     300G/64 cores is real headroom above it, not a tight fit. Raise both
#     back toward the partition's 40-128 core / 500G ceiling if a run at
#     this MDE_D actually gets OOM-killed or shows every core saturated --
#     but that hasn't happened at MDE_D=30 yet, so there's no measured
#     basis for holding a whole node reserved on the chance it might.
#   --time=4-00:00:00: left at this partition's hard cap regardless of the
#     above -- unlike cores/mem, a job already at the front of the queue
#     (as this one now should be, being the only one pending) starts as
#     soon as a fitting node exists, so a shorter --time buys no further
#     scheduling advantage here; it would only add real risk of a
#     mid-run timeout with nothing gained. See LAG_STEP's own comment for
#     why full completion within one submission is still not guaranteed,
#     and MDE_GMN_REUSE_BEST_PARAMS/RUN_CACHE_DIR below for the
#     resubmission-resumes-from-cache fallback this pipeline already has
#     for exactly that situation.
# See FIXED CAVEAT below for why wall time won't scale linearly with
# --cpus-per-task regardless.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_1_Hz.csv"   # default: J1620210605_epoch_8_freq_4_Hz.csv

# ---- Overall pipeline shape ------------------------------------------------
# MODE=MDE-GMN: the network is built in two stages, no CCM anywhere --
#   see build_nested_mde_network() (Stage 1) and attach_lag_embeddings()
#   (Stage 2) in simple-mde-gmn.py for the full algorithm.
#   Stage 1: nested-MDE over *real* (delay=0) observables only, straight
#     against the raw candidate pool (no delay expansion). Each target's
#     MDE-selected drivers (up to MDE_D of them) themselves become new
#     MDE targets in turn, for NESTED_M further rounds, growing the
#     "main node" backbone one level at a time; any driver that would
#     create a cycle or a self-loop is rejected as soon as it's proposed,
#     before it's ever written into the graph. Total backbone depth from
#     each target is 1 + NESTED_M levels.
#   Stage 2: for every Stage-1 node, one non-recursive MDE round over the
#     *lagged* candidate pool (LAG_MODE/LAG_DENSITY/LAG_STEP/DELAY_SOURCE
#     below) spanning the whole observable universe -- attaches each
#     node's own generalized (multi-delay) embedding coordinates. Never
#     spawns new backbone nodes, regardless of what delay a selected
#     driver carries (STAGE2_ALLOW_REAL_OBSERVABLES below controls
#     whether it can even select a delay=0 driver at all) -- Stage 1
#     alone owns which real observables become backbone/hub nodes. The
#     same observable can drive one node at several different delays at
#     once (each its own graph edge, visualized as separate colored arcs
#     -- see the network figure's legend).
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
# columns, and is reused as the per-round cap for *both* nested-MDE
# stages (Stage 1: real-observable-only, so the multiplier below is
# always 1; Stage 2: lagged-only, multiplied by however many lag values
# LAG_MODE/LAG_DENSITY/LAG_STEP below resolve to) -- the pipeline
# internally multiplies it by that stage's own number-of-candidate-
# columns-per-real-observable before handing it to dimx.MDE as the
# actual dimension cap D, since each real observable can span several
# (observable, delay) candidate columns and a naive D could otherwise be
# exhausted by several delays of the very same observable instead of
# covering MDE_D distinct ones (see effective_mde_dimension() in
# simple-mde-gmn.py). Setting MDE_D to a number >= the actual real-
# observable count in play (e.g. 999 against 500 real observables) makes
# MDE consider all of them -- it gets capped down to that real count
# rather than inflating D past what the candidate pool can supply.
# MDE_D=999 (picking essentially every real observable, ~500 at
# REGION_CHANNEL_CAP=999) is the eventual/ideal target, but per explicit
# direction is restricted to 30 for now: job 37946756 (MDE_D=999) got
# through its depth-0 seed with individual nodes already selecting
# 200-300+ drivers each, and only made it partway through the very first
# depth-1 wave (325 nodes) in 1d11h before failing -- CrossMapColumns'
# own per-step ranking cost grows with how many columns are already
# selected, so a node selecting hundreds of drivers is not just slower
# per candidate, it's calling CrossMapColumns hundreds of times at
# steadily growing dimension. 30 keeps each node's own candidate ranking
# bounded to a manageable size while still exercising real (not
# artificially tiny) multi-driver selection.
MDE_D=10                      # default: 20, max # of distinct real observables/lag columns MDE can forward-select per round, per stage
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
TP=10                         # default: 1 (dimx.MDE's own default), prediction interval for MDE candidate ranking/validation
EMBED_DIM_RHO_MIN=0.1         # default: 0.2, CCM embedding-dimension rho gate
CROSS_MAP_RHO_MIN=0.1         # default: 0.2, simplex cross-map rho gate
# CCM pre-filter: a cheap CCM(observable -> target) skill check run once
# per target group (see GMN_MODE above), right before either nested-MDE
# stage -- shrinks REGION_CHANNEL_CAP's candidate pool down to only
# observables that cross-map at least one target in that group's own
# rho >= this threshold, before Stage 1 (real observables) or Stage 2
# (their lags, which draw from this same filtered pool) ever see it. Cuts
# nested-MDE cost on both stages for a REGION_CHANNEL_CAP that would
# otherwise be too expensive to run uncapped. 0 disables the filter
# entirely (candidate pool passed through unchanged, byte-identical to
# this parameter never having existed); set >0 to enable it as the rho
# cutoff. See ccm_prefilter_observables() in simple-mde-gmn.py.
CCM_PREFILTER_RHO_MIN=0.2     # default: 0.2. 0 disables the CCM pre-filter entirely
# Static embedding dimension for the CCM pre-filter's own CCM calls -- no
# per-candidate EmbedDimension auto-select here (would defeat the point of
# a *cheap* pre-filter). Default matches TSLS_MDE_GMN_MULTI_MDE_E's own
# default in tsls-mde-gmn.sbatch (same static-E-for-a-bulk-CCM-gate
# rationale), for cross-pipeline consistency even though these are
# separate parameters (this pipeline has no per-candidate auto-select E to
# reuse the way tsls-mde-gmn's own MDE forward-selection now does).
CCM_PREFILTER_E=3             # default: 3
# Library sizes for the CCM pre-filter's own CCM calls, as fractions of
# PREDICTION_START (the leakage-safe MDE library window -- see the
# GENERATE()/Forecast() window section below for the full three-way A/B/
# end split) rather than absolute row counts, since PREDICTION_START's
# actual row count is data-dependent. E.g. 0.05 -> 5% of PREDICTION_START's
# rows as one candidate libSize. Converted to actual libSizes (deduped,
# floored at 10 rows) once PREDICTION_START is known -- see
# ccm_prefilter_observables()'s own docstring.
LIBSIZE_RATIOS="0.05,0.10,0.20,0.40,0.6,0.8,0.9"
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
NESTED_M=5                    # default: 1, additional nested-MDE rounds beyond each target's own selection. 0 = uncapped (expand until real observables are exhausted)
# Whether Stage 2's own candidate pool also includes delay=0 (real-
# observable) columns, in addition to lagged ones -- i.e. whether Stage
# 2 can introduce a real observable Stage 1 never reached. false
# (default): Stage 2 stays lagged-only, so Stage 1 remains the sole
# authority over which real observables become backbone/hub nodes -- a
# real observable worth hub status belongs in a wider Stage-1 MDE_D/
# NESTED_M search, not as a Stage-2-only leaf that's discovered but never
# itself expanded. true: a Stage-2-selected delay=0 driver is still
# subject to the same eager cycle-rejection as every other edge, but is
# never enqueued for its own further nesting the way a Stage-1 delay=0
# selection would be.
STAGE2_ALLOW_REAL_OBSERVABLES=false   # default: false. Other value: true
# First N channels per region (CA1_/mPFC_/OFC_) offered as MDE candidates,
# or (when MODE=GMN) used directly as the interaction-matrix's full node
# set. 999 is effectively uncapped (matches the full recording). Cap this
# down for a fast test job -- see FIXED CAVEAT below for why an uncapped
# MODE=GMN run is an O(channels^2) interaction matrix.
REGION_CHANNEL_CAP=30 # default: 999

# Stage 2's (and every node's self_delay_set's) nonzero-delay list (in
# rows, always negative -- that many rows in the past) -- MODE=MDE-GMN
# only. Each real observable is offered to Stage 2's forward selection
# as one column per delay, so MDE can pick specific (observable, delay)
# pairs -- e.g. 200 observables x 5 delays = 1000 candidate columns --
# rather than being stuck with one fixed lag. This is a generalized/
# non-uniform Takens embedding (Deyle & Sugihara 2011): different driving
# variables, and even repeats of the *same* driver, can each carry their
# own delay instead of sharing one embedding depth/lag network-wide.
# DELAY_SOURCE picks what actually determines this list:
#   'explicit' (legacy): the hand-picked DELAY_SET list below.
#   'lag_mode' (default, recommended): auto-derived from LAG_MODE/
#     LAG_DENSITY/LAG_STEP below instead -- DELAY_SET is ignored.
DELAY_SOURCE="lag_mode"       # default: lag_mode. Other value: explicit
# LAG_MODE (only used when DELAY_SOURCE=lag_mode) sets the *bound*
# (deepest lag magnitude, in rows) the auto-derived delay set reaches:
#   'generation_length_adjusted' (default): bound = GENERATE_LENGTH below
#     -- ties the deepest embedding coordinate to the length of one full
#     Generate() run.
#   'every_lag_possible': bound = as deep as the MDE library segment
#     structurally allows (its own row count minus a small fixed safety
#     buffer) -- exhaustive, and correspondingly the more expensive of
#     the two.
LAG_MODE="generation_length_adjusted"   # default: generation_length_adjusted. Other value: every_lag_possible
# LAG_DENSITY (only used when DELAY_SOURCE=lag_mode) samples LAG_MODE's
# bound at LAG_STEP-row intervals ('strided', default) rather than every
# single integer lag ('dense') -- density is tunable independently of
# the bound for compute tractability: e.g. every_lag_possible against a
# several-thousand-row library would otherwise mean tens of thousands of
# candidate columns per Stage-2 node. 'dense' accepts that cost -- keep
# REGION_CHANNEL_CAP/MDE_D/GENERATE_LENGTH small if using it.
LAG_DENSITY="strided"         # default: strided. Other value: dense
# LAG_STEP=1 makes LAG_MODE's bound dense regardless of LAG_DENSITY
# above (every consecutive row is its own candidate delay -- the classic
# Takens-embedding tau IS exactly this step size) -- forced per explicit
# direction, this initial two-stage implementation never uses any other
# step. Cost is real and load-bearing on --cpus-per-task/--mem/--time
# above: job 37939058 (a smoke test at REGION_CHANNEL_CAP=15/
# GENERATE_LENGTH=20, 900 candidate columns/node) still timed out at 30
# minutes mid-Stage-2. At full scale (REGION_CHANNEL_CAP=999/
# GENERATE_LENGTH=300, ~470 real channels x 300 lags =~141k candidate
# columns/node -- roughly 150x that smoke test's own pool, against which
# even a ~4x --cpus-per-task speedup falls well short) full completion
# within one 4-day submission -- this partition's hard cap, see above --
# is genuinely uncertain, not just untested. MDE_GMN_REUSE_BEST_PARAMS/
# RUN_CACHE_DIR below already checkpoint every completed per-node MDE
# selection to durable bucket storage as it finishes, so a timed-out job
# resubmitted unchanged (same settings/data/code) picks back up node by
# node rather than restarting Stage 2 from scratch -- the practical path
# to a fully-populated network if one submission isn't enough.
LAG_STEP=1                    # default: 5, only used when LAG_DENSITY=strided
# Explicit delay list -- only used when DELAY_SOURCE=explicit above.
# Widening this multiplies the candidate pool Stage 2 searches (and so
# its cost) by len(DELAY_SET); keep it modest for a first test run.
DELAY_SET="0,-1,-3,-5,-10"

# ---- GMN Generate() window --------------------------------------------------
# Three-way split of the full dataset (row counts derived inside the
# script from these ratios and the loaded data's length):
#   [0, A)      MDE library    -- LIB_RATIO fraction of the full dataset
#   [A, B)      MDE validation -- both nested-MDE stages' own held-out
#               window (CCM/EmbedDimension/crossMapRhoMin gates) --
#               MDE_VALIDATION_RATIO fraction of the *remainder* after A
#   [B, end)    Generate/Forecast validation -- every Generate()/
#               Forecast() call (theta search, start-point sweep, final
#               rebuild, Forecast() diagnostic) is scored only here, via
#               GENERATE_LENGTH/GENERATE_START_MIN/MAX/STEP_OFFSET below
# This exists so network construction (both nested-MDE stages) and the
# reported Generate()/Forecast() fidelity never touch the same rows --
# see run_mde_for_target()'s LEAKAGE docstring in simple-mde-gmn.py for
# the actual mechanics (mde_df truncated to end exactly at B, plus a
# Tp-row buffer before B).
LIB_RATIO=0.75                  # default: 0.5, fraction of rows used as library (A)
# Fraction of the *post-library* remainder (i.e. of [A, end)) given to
# the MDE validation segment [A, B) rather than the Generate/Forecast
# validation segment [B, end) -- e.g. LIB_RATIO=0.75, MDE_VALIDATION_
# RATIO=0.5 on a 1000-row dataset: A=750, remainder=250, B=750+125=875,
# so MDE validation gets rows 750-874 and Generate/Forecast validation
# gets rows 875-999. Widen this (toward 1.0) if MDE's own CCM/rho gates
# need a larger held-out window to be meaningful; narrow it (toward 0.0)
# to leave more room for GENERATE_LENGTH/the start-point sweep.
MDE_VALIDATION_RATIO=0.5        # default: 0.5, fraction of the post-library remainder given to MDE's own validation segment

# Number of steps each GMN Generate() call free-runs for, decoupled from
# the LIB_RATIO/MDE_VALIDATION_RATIO-derived held-out window above. Every
# Generate() call (the theta search below, the start-point sweep, and
# the final rebuild) always still starts right at the beginning of the
# Generate/Forecast validation segment (B) unless the start-point sweep
# below picks a later row within that same segment; only the *length* is
# independent otherwise. Generate() is the single most expensive step in this
# pipeline, so shrinking this alone (without also shrinking the held-out
# window MDE/CCM validate against) is the cheapest way to cut wall time.
# Also sizes the final Forecast() diagnostic's own window -- see
# FORECAST_PLOT_RANGE below for exactly how, since that now depends on
# which of its two modes is selected. 0 means "use the entire held-out
# window" (unchanged pre-existing behavior); a value larger than the
# held-out window is automatically clamped down to it (a Generate()
# longer than there's real data to score it against isn't meaningful).
GENERATE_LENGTH=40              # default: 0 (= full held-out window). Set e.g. 50-200 to speed up the theta search

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
# always the Generate/Forecast validation segment's own start (B), and
# keeps whichever one best matches the real held-out data, using the
# already-chosen best theta from the search below (not re-searched here).
# Candidates are expressed as offsets in rows from B (never from
# PREDICTION_START -- that anchors the unrelated MDE validation segment):
# MIN/MAX/STEP work like E_MIN/E_MAX/E_STEP below (MAX EXCLUSIVE). Every
# offset must be >= 0 (an offset before B would compare against data the
# MDE validation segment already used, which isn't a fair out-of-sample
# test) and leave enough rows for GENERATE_LENGTH to still fit inside the
# Generate/Forecast validation segment -- offsets outside that range are
# dropped automatically once the dataset is loaded.
# Defaults to MIN=0, MAX=1 -- only offset 0 (i.e. always start at B), the
# same as before this sweep existed. Widen MAX to actually sweep; each
# extra offset costs one more full Generate() call, so widen gradually
# (e.g. STEP=50 across a few hundred rows) rather than scanning every row.
GENERATE_START_MIN_OFFSET=0
GENERATE_START_MAX_OFFSET=20
GENERATE_START_STEP=2

# ---- E/tau search (MODE=GMN only) --------------------------------------------
# Used exclusively by that mode's own CCM interaction-matrix construction
# (data-driven E/tau grid search over the network's driver pool) --
# MODE=MDE-GMN has no network-wide E/tau at all: each node's embedding
# comes from nested-MDE's own discovered per-driver delays (Stage 2's
# lag set, DELAY_SOURCE/LAG_MODE/LAG_DENSITY/LAG_STEP/DELAY_SET above)
# instead, so these ranges are simply unused in that mode. E_MIN
# must stay >= 2 (SMap's own predictor-exclusion convention needs E >= 2
# even for leaf/undriven nodes).
E_MIN=3      # lower bound (inclusive) of embedding-dimension E search; must stay >= 2
E_MAX=16     # upper bound (EXCLUSIVE -- python range()) of E search; 16 itself is never tried
E_STEP=2     # step between candidate E values, e.g. 3,5,7,...,15 with the defaults above
TAU_MIN=-15  # lower bound (inclusive) of time-delay tau (in samples) search
TAU_MAX=0    # upper bound (EXCLUSIVE -- python range()) of tau search; 0 itself is never tried
TAU_STEP=2   # step between candidate tau values, e.g. -15,-13,...,-1 with the defaults above

# RUN_MDE_EVALUATE: exposed here for consistency with tsls-mde-gmn.sbatch's
# own knob of the same name, NOT currently wired into simple-mde-gmn.py --
# this pipeline's own run_mde_evaluate() (line ~4207) is a different
# implementation (dx.Evaluate-based, on 1 Hz raw spike-count data) already
# gated by RUN_MDE_DIAGNOSTICS above, and has not been confirmed to share
# tsls-mde-gmn.py's own "diffusion-map crash and hang" finding
# (docs/review-findings.md) -- that finding's own suspected mechanism
# (TSLS's long inter-spike ramps producing heavy near-duplicate structure)
# doesn't obviously carry over to raw counts. Left False (matching the
# TSLS-side default) as a placeholder should this pipeline's own
# run_mde_evaluate() ever need the same protection -- read as inert until
# someone wires it in, not as evidence this pipeline has the same risk.
RUN_MDE_EVALUATE=false

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
VENV_DIR="/home/j/j-tominaga/pao_rotation-exp-general/.venv"
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
PROJ_DIR="/home/j/j-tominaga/pao_rotation-exp-general"

# ADJUST: <GROUP> to this lab's actual flash allocation. Defined here
# (rather than down in the /flash scratch setup section below, where it
# used to live) because OUT_DIR now needs it too -- per explicit
# direction, ALL of this pipeline's output now lands on /flash, not
# /home ($PROJ_DIR/results): home storage is limited, and this isn't
# hypothetical -- job 37939058 alone (a tiny smoke test) left a ~23MB
# .err file (see the #SBATCH --error comment above). Only JOB_LOG.md and
# the small .snapshots/ job-description cache (see JOB DESCRIPTION below)
# stay on $PROJ_DIR -- both are small, human/git-facing provenance
# records, not pipeline output data.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

# Live for the run's whole duration (not the ephemeral per-job tempdir
# further below, which is deleted on every exit including a timeout) --
# matches $FLASH_BASE/pao_rotation_results, the literal path the #SBATCH
# --output/--error directives above are hardcoded to, so this job's
# Slurm logs and its actual pipeline output end up side by side the same
# way they always have on /home. Only persistent past this job's own
# lifetime on a crash/timeout, though, not on a clean exit -- see the
# `cleanup` EXIT trap in the SLURM LOG RELOCATION + OUTPUT ARCHIVING
# section below and OUTPUT_ARCHIVE_DIR's own comment for the full
# archive-then-clear behavior on success. /flash is scratch again for
# this pipeline's purposes; the durable copy is OUTPUT_ARCHIVE_DIR
# (bucket).
OUT_DIR="$FLASH_BASE/pao_rotation_results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ADJUST: durable cross-job cache pool for expensive, fingerprint-reusable
# pipeline stages (MDE forward-selection, nested-MDE/CCM network
# construction, theta search, generate-start sweep -- see the module
# docstring's SEARCH REUSE section and each stage's own docstring).
# Deliberately NOT under $OUT_DIR (/flash) or $PROJ_DIR (/home) -- this
# is a growing pool of machine-only cache artifacts across every run ever
# submitted, not this project's own human-facing per-job output; bucket
# storage has the headroom (and durability -- unlike /flash, not subject
# to the same scratch-cleanup expectations) for that instead. CACHE_ROOT
# is the read side (searched recursively, across every past job's
# subdirectory, by find_cached_best_params()-style lookups); RUN_CACHE_DIR
# is this job's own write side -- each cacheable stage syncs its result
# here *immediately* upon completing (not just at job end like $OUT_DIR),
# so it survives even if this job later crashes or times out.
CACHE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/mde_gmn_cache"
RUN_CACHE_DIR="$CACHE_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_CACHE_DIR"

# Separate from CACHE_ROOT/RUN_CACHE_DIR above -- that pool holds only the
# fingerprinted, cross-job-reusable *cache* artifacts (per-node MDE
# selection pickles, best_params/generate_start_sweep json), not this
# run's actual deliverables (network edge CSVs, every diagnostic PNG,
# Generate() output CSVs, fidelity plots -- everything else OUT_DIR
# collects). This is the full-output durable archive instead: on a
# successful exit, the SLURM LOG RELOCATION section's `cleanup` trap below
# copies everything under $OUT_DIR here, then deletes the /flash copy --
# per explicit direction, /flash should go back to being genuine scratch
# (not permanent storage) now that bucket has the headroom, while still
# staying in place on /flash (not archived, not deleted) if the job
# crashes/times out/gets cancelled, so a failed run's partial output is
# still there to inspect.
OUTPUT_ARCHIVE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/mde_gmn_outputs"
OUTPUT_ARCHIVE_DIR="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ARCHIVE_DIR"

# ==============================================================================
# SLURM LOG RELOCATION + OUTPUT ARCHIVING
# ==============================================================================
# The #SBATCH --output/--error above write to $FLASH_BASE/
# pao_rotation_results/ (flat, not $OUT_DIR -- see the ADJUST comment on
# those directives for why). Move them into $OUT_DIR once it exists, via
# a single combined EXIT trap (merged with the /flash tempdir cleanup
# trap below, since bash only honors the most recently registered EXIT
# trap -- a second `trap ... EXIT` later in this script would silently
# replace this one rather than stacking). Runs on every exit path,
# including failure/timeout/cancel, which is exactly when you most want
# the log sitting next to the rest of that run's output instead of loose
# in pao_rotation_results/.
#
# Also decides $OUT_DIR's own fate, based on PY_STATUS (set around the
# `python3 -u simple-mde-gmn.py` call below -- unset/non-"0" here means
# either the pipeline itself failed, or this trap fired before that line
# ever ran, e.g. an earlier setup failure): "0" means the pipeline
# completed cleanly, so this run's full output is copied to the durable
# OUTPUT_ARCHIVE_DIR (bucket -- see that variable's own comment) and the
# /flash copy is then removed, since /flash is meant to be scratch again
# now that bucket holds the real copy; anything else leaves $OUT_DIR
# exactly where it is on /flash, untouched, for post-mortem inspection --
# per explicit direction, crashes/OOMs/timeouts are exactly the case
# where you want that data still sitting on fast, directly-browsable
# storage rather than needing to dig it out of the archive.
SLURM_OUT="$FLASH_BASE/pao_rotation_results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$FLASH_BASE/pao_rotation_results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
cleanup() {
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
# Diffs this run's simple-mde-gmn.py + simple-mde-gmn.sbatch against the
# rolling snapshot cache of the last simple-mde-gmn submission
# ($PROJ_DIR/results/.snapshots/ on /home, deliberately small and not a
# per-job copy -- see generate_job_description.sh's own module docstring
# for why),
# inserting a new entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes
# as a table, everything else as unified diffs), then updates that cache
# to this run's files. Runs before the pipeline itself so the entry is
# written even if the run later fails or times out. In the common case
# (submitted via submit.sh) this call is a no-op -- submit.sh already
# wrote this job's JOB_LOG.md entry at submission time, and this script
# checks for that first (see generate_job_description.sh's own
# idempotency comment) -- but still runs here so a job started via bare
# `sbatch` (skipping submit.sh) still gets logged.
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
    "$PROJ_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/simple-mde-gmn.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# FLASH_BASE itself is defined earlier now (OUT_DIR needs it too -- see
# that section's own comment); only the ephemeral per-job tempdir is set
# up here.
tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
# Cleanup on exit (including this tempdir) is handled by the combined
# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION section
# above -- deliberately not a second `trap ... EXIT` here, which would
# silently replace that one instead of stacking. Only ever holds the
# staged source/data copy below (COPY SOURCE FILE + INPUT DATA) -- actual
# pipeline output no longer passes through here at all (see
# MDE_GMN_RESULTS_DIR below), so this directory being wiped on every
# exit, including a timeout, is no longer a data-loss concern the way it
# used to be.

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
# Points directly at OUT_DIR (not a subdirectory of the ephemeral
# $tempdir) -- see the /flash scratch setup section's own comment: every
# pipeline output, including the theta search's/generate-start sweep's
# own intra-job checkpoint files, now survives a timeout/crash rather
# than being deleted with $tempdir before ever reaching $OUT_DIR. What
# happens to OUT_DIR itself after the run -- archived to bucket and
# cleared from /flash on success, left alone on /flash on a
# crash/timeout -- is the `cleanup` EXIT trap's job (see OUTPUT_ARCHIVE_DIR
# above); nothing here needs to change based on that.
export MDE_GMN_RESULTS_DIR="$OUT_DIR"
export MDE_GMN_MODE="${MODE}"
export MDE_GMN_GMN_MODE="${GMN_MODE}"
export MDE_GMN_TARGETS="${TARGETS}"
export MDE_GMN_D="${MDE_D}"
export MDE_GMN_TP="${TP}"
export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export MDE_GMN_CCM_PREFILTER_RHO_MIN="${CCM_PREFILTER_RHO_MIN}"
export MDE_GMN_CCM_PREFILTER_E="${CCM_PREFILTER_E}"
export MDE_GMN_LIBSIZE_RATIOS="${LIBSIZE_RATIOS}"
export MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
export MDE_GMN_NESTED_M="${NESTED_M}"
export MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES="${STAGE2_ALLOW_REAL_OBSERVABLES}"
export MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export MDE_GMN_DELAY_SOURCE="${DELAY_SOURCE}"
export MDE_GMN_LAG_MODE="${LAG_MODE}"
export MDE_GMN_LAG_DENSITY="${LAG_DENSITY}"
export MDE_GMN_LAG_STEP="${LAG_STEP}"
export MDE_GMN_DELAY_SET="${DELAY_SET}"
export MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export MDE_GMN_MDE_VALIDATION_RATIO="${MDE_VALIDATION_RATIO}"
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
export MDE_GMN_RUN_MDE_EVALUATE="${RUN_MDE_EVALUATE}"
export MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export MDE_GMN_INTERACTION_MATRIX_OUTER_CORES="${INTERACTION_MATRIX_OUTER_CORES}"
export MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
# Separate from MDE_GMN_RESULTS_DIR (also persistent now, but per-job and
# on /flash -- see that export's own comment) -- these point at the
# durable *cross-job* bucket cache pool set up above so a later job can
# actually find a past run's cached results to reuse, not just this run's
# own (see CACHE_ROOT/RUN_CACHE_DIR's own comment).
export MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

python3 -V

# Deliberately not left under `set -e` for this one call -- PY_STATUS
# needs to be captured either way (success or failure) so the `cleanup`
# EXIT trap above knows whether to archive-and-clear $OUT_DIR or leave it
# on /flash; under plain `set -e`, a nonzero exit here would jump straight
# to the trap without ever reaching the capture line below.
set +e
python3 -u simple-mde-gmn.py
PY_STATUS=$?
set -e

if [ "$PY_STATUS" -ne 0 ]; then
    echo "simple-mde-gmn.py exited with status ${PY_STATUS} -- leaving output on /flash at ${OUT_DIR} for inspection (not archived to bucket)." >&2
    exit "$PY_STATUS"
fi

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
# No cp step needed here anymore -- MDE_GMN_RESULTS_DIR points directly
# at $OUT_DIR now (see that export's own comment), so the pipeline has
# already been writing straight to /flash throughout the run. What
# happens next -- archived to bucket and removed from /flash, since
# PY_STATUS=0 here means the run completed cleanly -- is handled by the
# `cleanup` EXIT trap above (see that section's own comment for why it's
# there rather than here: it has to run on every exit path, not just this
# clean one).

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Archiving outputs to bucket and clearing the /flash copy:"
echo "  ${OUT_DIR} -> ${OUTPUT_ARCHIVE_DIR}"
echo "========================================================"
