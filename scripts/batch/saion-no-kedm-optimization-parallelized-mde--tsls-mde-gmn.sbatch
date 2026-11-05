#!/bin/bash -l
#SBATCH --job-name=saion_pmde_half
# Either restricted GPU partition (your allocation is 36 cores + 4 GPUs PER
# partition, 7-day max); the cu126 torch build in .venv-gpu runs unchanged on
# both P100 and V100 nodes. `--partition=gpu-p100,gpu-v100` is rejected for
# this account ("Multiple partition job request not supported when a partition
# is set in the association"), so scripts/saion/submit_saion.sh picks the one
# with the earlier estimated start and overrides this line. The value here is
# only the fallback for a plain `sbatch`.
#SBATCH --partition=gpu-p100
# A HALF-NODE SHARE (raised from a quarter, 2026-10-05). The allocation is
# 36 cores + 4 GPUs per partition, so 2 GPUs + 18 cores + ~240 GB is the
# balanced half of it, and still leaves half for everyone else -- the point of
# the original quarter-share note was not to occupy a whole node while
# stranding GPUs nobody else could then use, and that still holds at a half.
#
# WHY THIS WENT UP -- job 4741044. The claim this block used to make, that
# "9 workers on one 16 GB card is comfortably inside the measured capacity
# (preflight ran 36 with 0 errors at 92% of device memory; the safe cap is
# ~31)", was wrong, and wrong in a way the preflight could not have caught.
# The preflight measured the ~288 MiB CUDA CONTEXT each worker costs just to
# attach to the device, and divided 16 GiB by it. It never measured the
# WORKING SET of a real cross-map: the (M, M) distance matrix plus the
# gathered (B, M, L) tensor run to GIGABYTES -- one process in 4741044 was
# logged holding 8.51 GiB, ~30x its context. Nine of those do not fit. The
# card peaked at 16203/16384 MiB and 51 CCM calls across 85 nodes failed,
# each one silently recorded as a dead end.
#
# Capacity is no longer asserted here at all. scripts/src/gpu_budget.py sizes
# a per-device budget AT RUNTIME from what the device reports, and workers
# queue for it, so this file no longer has to predict what fits -- which is
# what it got wrong. The second GPU is throughput: it restores roughly the
# concurrency the broken version had by accident, without the failures.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
# See SIZING NOTE below for how --mem/--time were derived (doubled with the
# core count: memory here scales per spawn worker, each holding its own
# pickled copy of the Stage 2 DataFrame).
#SBATCH --mem=240G
#SBATCH --time=6-00:00:00
# 30 min before the time limit Slurm sends USR1 to this batch shell only;
# the handler below stops Python cleanly so there is time to stage results
# and the MDE cache out to /bucket/FukaiU/tominaga/pao_rotation before the hard kill.
#SBATCH --signal=B:USR1@1800
# Absolute paths under /home/j/j-tominaga (guide's recommendation; /flash does not exist on
# Saion). The directory must exist before submitting -- see the runbook.
#SBATCH --output=/home/j/j-tominaga/saion_slurm_logs/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/saion_slurm_logs/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

# =============================================================================
# parallel-MDE GPU pipeline -- SAION restricted-partition variant
# =============================================================================
# Runs scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py
# (UNCHANGED) with the same TSLS_MDE_GMN_* parameters as
# no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch. What differs is
# only what Saion's rules require (https://scda-guide.oist.jp/saion-system.html,
# summarised in scripts/saion/saion_env.sh):
#
#   DATA FLOW
#     /bucket/FukaiU/tominaga/pao_rotation/data/<DATA_FILE>  read directly (TSLS_MDE_GMN_DATA_DIR)
#     /home/j/j-tominaga/pao_rotation (code only) --copy--> /work/FukaiU/j-tominaga/pmde.<job>.XXXX/
#         src/      pipeline + helper modules
#         results/  TSLS_MDE_GMN_RESULTS_DIR   (all pipeline output)
#         cache/    TSLS_MDE_GMN_RUN_CACHE_DIR (this run's cache entries)
#     /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache   TSLS_MDE_GMN_PREV_RUNS_DIR, READ-ONLY
#     cache/   --hourly + at exit, via login node--> /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache/job_<id>/
#     results/ --at exit, via login node-->          /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs/job_<id>/
#                                                    (or job_<id>_FAIL/ if the run failed)
#     The pmde.<job>.XXXX dir is removed only after BOTH copies succeed.
#     Every mkdir/rm/stage-out is guarded to stay inside those three roots
#     (path_under/safe_rm_work in scripts/saion/saion_env.sh).
#
#   OUTCOME  results/STATUS.txt says COMPLETED or FAIL plus the reason
#     (python exit status, time-limit, CUDA errors, Slurm state), and reports
#     CCM_DEAD_ENDS -- candidates genuinely evaluated and rejected, which are
#     findings, not failures.
#     As of 2026-10-05 a GPU memory failure no longer reaches this check as a
#     silent shrinkage: proper_ccm/multi_proper_ccm re-raise it instead of
#     recording the candidate as a zero-driver dead end (see gpu_budget.py),
#     so the run stops at the failure rather than exiting 0 with a quietly
#     smaller network. The CUDA-error count is kept as a second line of
#     defence and now also catches the allocator's own warning spelling.
#
#   RESUMING  Every finished MDE node is cached immediately and synced to
#     /bucket/FukaiU/tominaga/pao_rotation hourly, so resubmitting this same file after a timeout/crash
#     skips all completed nodes (REUSE_BEST_PARAMS=true). Only nodes that were
#     in flight at the kill are recomputed.
#
#   GRACEFUL CANCEL  `scancel --batch --signal=USR1 <jobid>` stops the run AND
#     stages everything out. Plain `scancel <jobid>` leaves only ~30 s, so
#     rely on the hourly cache sync + the manual recovery command it prints.
#
# Submit from the Saion login node, from /home/j/j-tominaga, with
#     cd /home/j/j-tominaga && /home/j/j-tominaga/pao_rotation/scripts/saion/submit_saion.sh \
#         saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch \
#         no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py
# (picks gpu-p100 or gpu-v100, then goes through submit.sh for JOB_LOG.md)
# Full procedure: docs/saion-parallelized-mde-runbook.md
# =============================================================================

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
#
# SIZING NOTE (Saion, "half production scale", 2026-09-16)
# ---------------------------------------------------------
# "Production" = this pipeline's documented defaults: REGION_CHANNEL_CAP=999
# (all 468 real channels: CA1 147, OFC 234, mPFC 87), MDE_D=20, NESTED_M=1,
# GENERATE_LENGTH=0 (full held-out window). Half of that, as set below:
#   REGION_CHANNEL_CAP=78   -> 78+78+78 = 234 channels = exactly half of 468
#   MDE_D=10                -> half of 20
#   NESTED_M=1              -> already the default; the next step down (0)
#                              means UNCAPPED, not smaller
#   GENERATE_LENGTH=400     -> NOT halved from the default, deliberately:
#       0 resolves to the full held-out window (~6,700 rows here), and
#       LAG_MODE=generation_length_adjusted turns that bound into Stage 2's
#       lag set. Half of it (~3,300 rows / LAG_STEP=8) would be ~410 lags per
#       channel -- a ~8x wider Stage 2 candidate pool than every run so far,
#       which already produced 36 h single nodes. 400 (-> 50 lags at
#       LAG_STEP=8 below) is the value every recent run used. Raise
#       deliberately, not via "half".
#       NOTE (2026-10-01): the lag count no longer has to be controlled this
#       way. DELAY_SOURCE=tau_count states it directly (LAG_STEP x LAG_COUNT),
#       which frees GENERATE_LENGTH to mean only how long Generate() runs.
#       This file still uses lag_mode/400 so its numbers stay comparable to
#       the runs the evidence below came from -- switch deliberately, since
#       changing the resolved lag set invalidates the MDE cache.
#
# Evidence these numbers are sized from -- Deigo job 40684253 (COMPLETED,
# CAP=30 -> 48/90 observables kept by the CCM pre-filter, MDE_D=6, lag set of
# 101, 44,449 usable rows, 64 cores, fork workers):
#   - Stage 1 nodes: < 1 s each.  Stage 2 nodes: 11 s ... 132,219 s (36.7 h),
#     heavy-tailed. Stage 2 is where all the time goes.
#   - batch-step MaxRSS 20.7 GB with 7 concurrent workers on a ~1.7 GB
#     Stage 2 DataFrame (~2.4 GB per worker).
# Scaled to CAP=78 (~125 observables after the same ~53% pre-filter keep
# rate): Stage 2 DataFrame ~125 x 101 cols x 44k rows x 8 B ~= 4.5 GB, and
# every spawn worker gets its own pickled copy (no fork copy-on-write under
# GPU_ENABLED). ~6.5 GB/worker + ~1 GB torch/CUDA import, x 9 workers,
# + ~15 GB main process ~= 83 GB  =>  --mem=120G, ~45% headroom on an
# extrapolation that has never been measured at this scale.
# If the job is OOM-killed: lower --cpus-per-task (= worker count), not MDE_D.
#
# TIME, AND WHAT THE QUARTER-NODE SHARE COSTS. 7-00:00:00 is the partition
# maximum; this asks for 6 days.
#
# Per-node Stage 2 cost is unchanged by the core count -- one node's work is one
# node's work, and Deigo saw single Stage 2 nodes take up to 36.7 h at a
# narrower pool than this. What the core count changes is how many nodes run at
# once: 9 instead of 36.
#
# For the CCM gate that costs nothing, which is the measured part. Saion
# preflight 4724497 on a P100 found total throughput flat at ~58 CCM sweeps/s
# for 1, 9, 18 and 36 workers alike (per-sweep time scaled exactly with the
# worker count) -- one card is already saturated by a single worker, so sharing
# it between 36 buys no GPU throughput over 9. That preflight also measured
# 9 workers at 3,785 MiB of the card's 16,384, i.e. 23%, against 92% for 36:
# this shape has large VRAM margin where the 36-worker shape had almost none.
#
# What does slow down is the CPU-side work -- the ranking stage and everything
# outside the GPU kernel -- by up to ~4x wherever Phase 0 has more tasks than
# workers.
#
# PARALLEL_MDE_SCOPE='staged' (set below) bounds the Stage 2 task list: Stage 2
# attaches lagged leaves and never recurses, so its tasks are exactly Stage 1's
# visited set -- a few dozen nodes, not all ~125 observables. Beyond that, the
# per-node cache synced hourly is what protects the run: resubmitting resumes
# from every finished node, so exceeding 6 days costs a resubmission, not the
# work.
#
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
MDE_D=10                       # HALF PRODUCTION (see SIZING NOTE above) -- default: 20
# Prediction interval every proper_mde forward-selection call ranks/
# validates every candidate driver against -- ported from
# simple-mde-gmn.py's own TP parameter (ranking, not just final scoring).
TP=20                          # default: 1 (proper_mde's own default)
EMBED_DIM_RHO_MIN=0.2         # default: 0.2, embedding-dimension skill gate -- CORRELATION-scored nodes (the behavioural targets)
CROSS_MAP_RHO_MIN=0.2         # default: 0.2, cross-map skill gate -- CORRELATION-scored nodes
# The same two gates for F1-SCORED nodes, i.e. every TSLS neuron. Separate
# because the metrics do not share a ceiling: correlation tops out at 1.0,
# while r9_min_pred_events_curve.csv measured a NEAR-PERFECT predictor of a
# real TSLS channel at f1 ~ 0.49-0.54 (never above 0.69). One 0.2 bar asked a
# neuron for ~38% of achievable skill and a behavioural target for 20%, which
# is part of why 101 nodes in job 4774136 had NO candidate clear the gate.
# 0.1 restores parity (0.20 x 0.52 ~ 0.104). Cost: r9 puts the f1 estimator's
# std at ~0.069 at MIN_PRED_EVENTS=20, so 0.1 is ~1.4 sigma at the event
# floor (std falls to ~0.028 by 160 events; real windows in job 4774136 had
# 67-188). Raise these first if admitted links stop replicating.
EMBED_DIM_RHO_MIN_F1=0.1      # default: 0.1, embedding-dimension gate for f1-scored (neuronal) nodes
CROSS_MAP_RHO_MIN_F1=0.1      # default: 0.1, cross-map gate for f1-scored (neuronal) nodes
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
# RAISED TO 0 = UNCAPPED (2026-11-05). At NESTED_M=1 the BFS expanded the
# targets and their drivers and then stopped, so everything one step further
# out was classified as an external input no matter what the evidence said.
# Job 4774136 ended with 6 interior nodes and 23 external ones, and 5 of those
# 23 HAD discovered drivers in the precomputed table -- CA1_85_tsls found six
# -- which the depth cap then discarded (`realized max depth=1`).
#
# Uncapping is nearly free for this pipeline. Both scopes precompute MDE for
# the WHOLE node universe up front, so the depth budget only governs how much
# of that existing table the BFS reads: job 4774136 precomputed 247 results
# and consulted 21. The exception is scope='staged' WITH lags, where a larger
# visited set means more Stage 2 calls -- so revisit this if you re-enable
# lags on the staged scope.
NESTED_M="${NESTED_M:-0}"      # overridable; 0 = UNCAPPED (see above). Code default: 1, additional nested-MDE rounds beyond each target's own selection
# First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted and
# offered as MDE candidates (or, when MODE=GMN, used directly as the
# interaction-matrix's full node set). 999 is effectively uncapped (matches
# notebooks/tsls-mde-gmn.ipynb's own full-channel-set decision).
# Overridable (2026-10-27), like DELAY_SOURCE/DELAY_SET/LAG_STEP below: it was
# hardcoded, so `sbatch --export=...,REGION_CHANNEL_CAP=999` was silently
# clobbered here and this file could only ever mean half production.
# REGION_CHANNEL_CAP=999 is the edge case that unlocks: the full 468-channel
# candidate pool, with the CCM pre-filter above still the only gate on it.
# The cap is in the per-node cache fingerprint, so such a run shares nothing
# with a capped one.
REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP:-78}"   # overridable; default here: 78 = HALF PRODUCTION (see SIZING NOTE above), 78x3 = 234 of 468 channels -- code default: 999

# Stage 2's (and every node's self_delay_set's) nonzero-delay list (in
# rows, always negative) -- MODE=MDE-GMN only. Each real observable is
# offered to Stage 2's forward selection as one column per delay, so MDE
# can pick specific (observable, delay) pairs rather than being stuck with
# one fixed lag. This is a generalized/non-uniform Takens embedding (Deyle
# & Sugihara 2011) -- see simple-mde-gmn.sbatch's own DELAY_SOURCE/
# LAG_MODE/LAG_DENSITY/LAG_STEP comments for the full rationale (identical
# here, ported unchanged -- pure column-name/DataFrame machinery with no
# TSLS-specific behavior).
# THREE WAYS TO DEFINE THE LAG SET. They differ only in which quantity you
# pin -- a lag set is a stride and how far back it reaches, so fixing any two
# of {stride, count, depth} determines the third:
#   'lag_mode' (default): pins the DEPTH, the count follows. Auto-derived from
#     LAG_MODE/LAG_DENSITY/LAG_STEP below; DELAY_SET and LAG_COUNT are ignored.
#   'tau_count': pins the COUNT, the depth follows. Exactly LAG_COUNT lags
#     spaced LAG_STEP apart (LAG_STEP is the tau), so the deepest lag is
#     LAG_STEP x LAG_COUNT. LAG_MODE/LAG_DENSITY/DELAY_SET are ignored.
#   'explicit': pins EVERY VALUE -- the hand-picked DELAY_SET list below.
# WHICH TO USE. 'lag_mode' when you want the lag set to follow the data.
# 'tau_count' when the count is a BUDGET you need to hold: Stage 2 offers MDE
# one candidate column per (real observable x lag) and Stage 2 is where
# essentially all of this job's time goes, so the count directly sets the bill.
# With the values set here ('lag_mode', GENERATE_LENGTH=400, LAG_STEP=8) the
# auto path resolves to 50 lags per observable; the same shape stated
# directly is DELAY_SOURCE=tau_count with LAG_STEP=8, LAG_COUNT=50. Narrowing
# it to, say, LAG_COUNT=25 is then a one-knob edit that does NOT also shorten
# the Generate() run -- which is what makes this different from shrinking
# GENERATE_LENGTH to get the same effect (see the pitfall note there).
DELAY_SOURCE="${DELAY_SOURCE:-lag_mode}"       # overridable; default: lag_mode. Other values: tau_count, explicit
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
# LAG_STEP is the spacing between consecutive lags and means the same thing
# under both auto paths: the sampling stride of LAG_MODE's bound under
# DELAY_SOURCE=lag_mode, and tau under DELAY_SOURCE=tau_count. One knob rather
# than two so the two paths cannot disagree about the stride.
LAG_STEP="${LAG_STEP:-8}"     # overridable; default: 1. The sampling stride (lag_mode) / tau (tau_count)
# LAG_COUNT: how many lags -- only used when DELAY_SOURCE=tau_count above.
# The deepest lag is then LAG_STEP x LAG_COUNT, which still has to fit inside
# the MDE library segment (the pipeline checks that for every source alike and
# fails at startup with the numbers, not mid-run).
LAG_COUNT="${LAG_COUNT:-10}"  # overridable; default: 10, only used when DELAY_SOURCE=tau_count
# Explicit delay list -- only used when DELAY_SOURCE=explicit above.
# Overridable (2026-10-22), like DELAY_SOURCE/LAG_STEP/LAG_COUNT above: it was
# hardcoded, so `sbatch --export=...,DELAY_SET=...` was silently clobbered here
# and DELAY_SOURCE=explicit could only ever mean this one list.
#
# DELAY_SET=0 is the useful edge case it unlocks: derive_lag_set() strips every
# 0 entry on the explicit path, so that resolves to an EMPTY lag set -- no
# lagged candidates, hence no Stage 2 picks and no lagged leaf nodes at all.
# Both builders SKIP Stage 2 entirely when the lag set is empty (b5b7da0,
# pinned by tests/test_zero_lag_stage2_skip.py), and lag_set is in the
# per-node cache fingerprint, so such a run shares nothing with a lagged one.
#
# That skip is why this is safe, and it was added only after this comment's
# earlier claim -- "nothing downstream trips on it" -- turned out to be false.
# Both builders used to run Stage 2 over the empty pool and divide by
# len(lag_set) in effective_mde_dimension(): job 4774133 (2026-10-27, staged)
# died there with ZeroDivisionError after Stage 1 had completed. Only
# scope='union' had ever exercised zero lags before (job 4769464), and union
# is immune because its num_delays counts the real observables too. Do not
# re-weaken this note without re-reading that test.
DELAY_SET="${DELAY_SET:-0,-1,-3,-5,-10}"
# NOTE: STAGE2_ALLOW_REAL_OBSERVABLES is retired (2026-09-30). Stage 2's
# candidates are lagged columns only, and every lagged pick becomes a leaf
# node, so there is nothing for Stage 2 to select recursively. load_config()
# REJECTS the env var rather than ignoring it.

# ---- Parallelized-MDE scope (THIS PIPELINE ONLY) -----------------------------
# Not a knob tsls-mde-gmn.py has. The two values are DIFFERENT ALGORITHMS
# (2026-09-30), not just different batching:
#
#   staged  The parent pipeline's algorithm without its per-wave barriers:
#           Stage 1 (real observables, recursive) for the whole node universe
#           in one pool, then the instant Stage 1 replay, then Stage 2
#           (lagged columns only, attached as leaf nodes) for EXACTLY Stage
#           1's visited set -- no Stage 2 overshoot at all, since Stage 2
#           never recurses. Builds the SAME network as tsls-mde-gmn.py,
#           asserted by tests/test_parallel_mde_equivalence.py. Two barriers.
#   union   ONE MDE per node over real AND lagged candidates TOGETHER, for the
#           whole universe, in a single pool: one barrier, and real and lagged
#           candidates compete directly inside each forward selection. This is
#           a different network, not a wrong one -- and a much more expensive
#           call per node, since every node's pool is (1 + len(LAG_SET))x
#           wider from the first selection round.
#
# Either scope may RECOMPUTE a node serially: the up-front phase cannot know
# which observables will end up upstream, so a precomputed result whose real
# picks turn out to be cycle-closing is discarded and recomputed with those
# candidates excluded (logged as "must be RECOMPUTED").
# SAION HALF-PRODUCTION CHOICE: staged -- it is the CPU pipeline's own
# algorithm (so results are comparable), it computes Stage 2 for only the
# backbone nodes, and union's much wider per-node pool is not affordable
# inside the 7-day partition limit.
# OVERRIDABLE FROM THE SUBMITTING SHELL. The value here is the default;
# `sbatch --export=ALL,PARALLEL_MDE_SCOPE=union <this file>` runs the other
# scope from this same file, which is how scripts/saion/submit_saion_scope_pair.sh
# submits the staged/union pair as one chain. Keeping it one file matters: the
# two scopes must otherwise agree on every knob, or the comparison is not one.
PARALLEL_MDE_SCOPE="${PARALLEL_MDE_SCOPE:-staged}"   # code default: union. Here: staged

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
GENERATE_LENGTH=400             # deliberately NOT halved from 0 -- see SIZING NOTE above -- default: 0 (= full held-out window)

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
CCM_SAMPLE=10                    # random-library resamples averaged per libSize -- higher = less noisy, slower
# 20 -> 10 (2026-10-22), to bring union scope inside a 2-day wall. This is
# the BATCH dimension of the GPU cross-map, so it buys time AND memory: by
# crossmap_bytes()'s own terms the B-dependent terms are ~80% of a call, so
# halving B takes the worst observed union call (M=28251, ~14.9 GiB live in
# job 4744295) to roughly 9 GiB. It costs ESTIMATOR PRECISION, not model
# space -- no observable, driver, depth, lag, generative window or data row
# is given up, the CCM rho at each libSize is just averaged over 10 random
# library draws instead of 20, so the convergence slope is noisier.
# NOTE: ccm_sample is part of the per-node cache fingerprint
# (caching.py build_mde_selection_fingerprint), so this correctly MISSES
# every node cached at 20 rather than silently reusing it.
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
CCM_LIB_SIZES_ROWS="16,40,160,400,800"  # absolute row counts for the CCM gate's own library sweep, never a percentage now
# 7 entries -> 5 (2026-10-22), same 2-day motivation. The sweep is
# sequential within one reservation, so this is ~5/7 of the time and does
# NOT change peak memory. 24 and 64 were dropped and the ENDPOINTS kept:
# proper_ccm's verdict needs both the endpoint-to-endpoint change and the
# fitted slope to agree, so 16 and 800 are the two that must stay, and what
# is left (16,40,160,400,800) is still a near-geometric spread -- five
# points is ample to fit a line through. Also in the node fingerprint.

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
# Deliberately no `set -e`: every failure path below must still reach the
# EXIT trap's stage-out, and the steps that must not fail check explicitly.
set -uo pipefail

# Defines the scoped roots PROJ_DIR, SLURM_LOG_DIR, WORK_BASE, BUCKET_PROJECT.
# shellcheck source=../saion/saion_env.sh
source "/home/j/j-tominaga/pao_rotation/scripts/saion/saion_env.sh"
PY_NAME="no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py"
VENV_DIR="$PROJ_DIR/.venv-gpu"

die() { echo "FATAL: $*" >&2; exit 2; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch, not bash"
[ -d "$(dirname "$WORK_BASE")" ] || die "$(dirname "$WORK_BASE") not found -- this script is for Saion compute nodes only"
[ -x "$VENV_DIR/bin/python3" ] || die "$VENV_DIR missing -- run scripts/saion/setup_gpu_venv.sh on the login node first"

# No `module load python` -- the venv's interpreter is uv-managed under /home/j/j-tominaga.
# No `module load cuda` -- the cu126 torch wheel bundles its own CUDA runtime;
# only the node's driver is used.
source "${VENV_DIR}/bin/activate"
export MPLBACKEND=Agg
# Workers are processes (TSLS_MDE_GMN_CORES); keep BLAS/OpenMP single-threaded
# inside each, so N workers don't each spin up N threads and oversubscribe the
# cores this job was actually given.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Let the CUDA caching allocator grow one segment instead of many fixed
# blocks. torch's own OOM message in job 4741044 recommended exactly this;
# it does not create capacity, it stops capacity being lost to
# fragmentation between the several workers sharing each card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Bucket targets, all inside $BUCKET_PROJECT (written only through the login
# node; stage_dir refuses anything outside it).
CACHE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_cache"
OUTPUT_ARCHIVE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_outputs"
CACHE_DEST="$CACHE_ROOT/job_${SLURM_JOB_ID}"

# Per-job scratch inside $WORK_BASE (guide: unique mktemp dir, deleted at the end).
mkdir -p "$WORK_BASE" || die "cannot create $WORK_BASE"
jobdir="$(mktemp -d "$WORK_BASE/pmde.${SLURM_JOB_ID}.XXXXXX")" || die "mktemp in $WORK_BASE failed"
OUT_DIR="$jobdir/results"
RUN_CACHE_DIR="$jobdir/cache"
STAGE_SRC="$jobdir/src"
mkdir -p "$OUT_DIR" "$RUN_CACHE_DIR" "$STAGE_SRC" "$jobdir/tmp" || die "cannot populate $jobdir"
# Temp files (Python tempfile, mktemp, matplotlib/torch scratch) stay inside
# this job's dir instead of the node-wide /tmp, and are removed with it.
export TMPDIR="$jobdir/tmp"
export MPLCONFIGDIR="$jobdir/tmp/matplotlib"

SLURM_OUT="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

PY_PID=""
PY_STATUS=""
STOP_REASON=""
BG_PIDS=()

# ==============================================================================
# STOP / OUTCOME / STAGE-OUT
# ==============================================================================
stop_python() {
    [ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null || return 0
    # Spawn workers are children of the Python process -- stop them too.
    pkill -TERM -P "$PY_PID" 2>/dev/null
    kill -TERM "$PY_PID" 2>/dev/null
    for _ in $(seq 60); do
        kill -0 "$PY_PID" 2>/dev/null || return 0
        sleep 2
    done
    pkill -KILL -P "$PY_PID" 2>/dev/null
    kill -KILL "$PY_PID" 2>/dev/null
}

on_usr1() {
    STOP_REASON="time limit approaching (or scancel --signal=USR1): stopped cleanly to stage out"
    echo "$(date -Iseconds) USR1 received -- ${STOP_REASON}" >&2
    stop_python
}
trap on_usr1 USR1
trap 'STOP_REASON="${STOP_REASON:-SIGTERM received (scancel or node shutdown)}"; stop_python; exit 143' TERM

cleanup() {
    trap - EXIT USR1 TERM
    stop_python
    for pid in "${BG_PIDS[@]}"; do
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done

    local slurm_state="" gpu_errors budget_advisories networks_skipped mde_failures ccm_dead_ends status reason dest stage_ok=1
    for _ in 1 2 3; do
        slurm_state="$(sacct -j "$SLURM_JOB_ID" --format=State -P -n 2>/dev/null | head -1 | tr -d ' ')"
        [ -n "$slurm_state" ] && break
        sleep 2
    done
    # 'memory allocation failed with OOM' is the CUDACachingAllocator's own
    # spelling and matched NONE of the other patterns -- job 4741044 logged
    # 47 of those lines and this check saw zero of them, so the early
    # warning that the card was saturating never reached STATUS.txt.
    #
    # 'UNDER-ESTIMATED' is deliberately NOT in this list (it was, 2026-10-05
    # to 2026-10-22). gpu_budget.reserve() prints it AFTER a reservation has
    # been granted and the call has run, to say the shape formula predicted
    # less than the allocator went on to hold. Nothing failed. Counting it
    # here stamped job 4744294 FAIL on 93 of those advisories with zero real
    # CUDA errors in the whole 19h run, which buried the one thing that did
    # go wrong. It gets its own line below instead.
    gpu_errors="$(cat "$SLURM_OUT" "$SLURM_ERR" 2>/dev/null \
        | grep -c -E 'CUDA out of memory|OutOfMemoryError|CUDA error|CUBLAS_STATUS|cuDNN error|memory allocation failed with OOM')"
    budget_advisories="$(cat "$SLURM_OUT" "$SLURM_ERR" 2>/dev/null | grep -c 'UNDER-ESTIMATED')"
    # A network whose Generate() never rebuilt produces NO plots, fidelity or
    # output, and the pipeline says so and carries on to exit 0. Job 4744294
    # lost head_position_x_head_position_y exactly that way and reported
    # success. Counted, and FAILed on below, so it cannot pass unnoticed again.
    networks_skipped="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'no successful Generate() rebuild')"
    mde_failures="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'MDE call FAILED')"
    # Candidates that were genuinely evaluated and found wanting (short
    # data, no convergence). These are FINDINGS, not failures -- counted
    # so they stop being indistinguishable from the GPU-memory drops that
    # used to print through the same clause.
    ccm_dead_ends="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'CCM dead end:')"

    if [ "${PY_STATUS:-unset}" = "0" ] && [ -z "$STOP_REASON" ] && [ "$gpu_errors" -eq 0 ] \
            && [ "$networks_skipped" -eq 0 ]; then
        status=COMPLETED; reason="pipeline exited 0"
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}"
    else
        status=FAIL
        if [ -n "$STOP_REASON" ]; then reason="$STOP_REASON"
        elif [ "${PY_STATUS:-unset}" = "0" ] && [ "$gpu_errors" -gt 0 ]; then reason="pipeline exited 0 but the log has ${gpu_errors} CUDA error line(s) -- MDE nodes that hit them were silently recorded as dead ends"
        elif [ "${PY_STATUS:-unset}" = "0" ]; then reason="pipeline exited 0 but ${networks_skipped} network(s) produced no output (Generate() never rebuilt) -- see the '*: FINAL Generate() rebuild FAILED' line(s) in the log"
        elif [ -z "${PY_STATUS:-}" ]; then reason="job ended before the pipeline started (setup failure)"
        else reason="pipeline exited with status ${PY_STATUS}"
        fi
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}_FAIL"
    fi

    {
        echo "STATUS: $status"
        echo "REASON: $reason"
        echo "PY_STATUS: ${PY_STATUS:-unset}"
        echo "SLURM_STATE_AT_EXIT: ${slurm_state:-unknown (check: sacct -j $SLURM_JOB_ID)}"
        echo "HOST: $(hostname)  PARTITION: ${SLURM_JOB_PARTITION:-?}  GPUS: ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}}"
        echo "CUDA_ERROR_LINES: $gpu_errors"
        echo "GPU_BUDGET_ADVISORIES: $budget_advisories  (UNDER-ESTIMATED; informational, not failures)"
        echo "NETWORKS_SKIPPED: $networks_skipped"
        echo "CCM_DEAD_ENDS: $ccm_dead_ends"
        echo "MDE_CALL_FAILED_LINES: $mde_failures"
        echo "ENDED: $(date -Iseconds)"
        echo "CACHE_ARCHIVE: $CACHE_DEST"
    } > "$OUT_DIR/STATUS.txt"
    [ -f "$OUT_DIR/progress.log" ] && \
        echo "$(date -Iseconds) JOB ENDED: $status -- $reason (PY_STATUS=${PY_STATUS:-unset} SLURM_STATE=${slurm_state:-unknown})" \
        >> "$OUT_DIR/progress.log"
    # cp, not mv: Slurm is still appending to these until the job exits.
    [ -f "$SLURM_OUT" ] && cp -f "$SLURM_OUT" "$OUT_DIR"/
    [ -f "$SLURM_ERR" ] && cp -f "$SLURM_ERR" "$OUT_DIR"/
    [ -f "$0" ] && cp -f "$0" "$OUT_DIR/submitted.sbatch"

    echo "========================================================"
    echo "Job ${SLURM_JOB_ID}: $status -- $reason"
    echo "Staging out through ${STAGE_HOST}:"
    echo "  $OUT_DIR -> $dest"
    echo "  $RUN_CACHE_DIR -> $CACHE_DEST"
    stage_dir "$OUT_DIR" "$dest" || { stage_ok=0; echo "WARNING: output stage-out FAILED" >&2; }
    stage_dir "$RUN_CACHE_DIR" "$CACHE_DEST" || { stage_ok=0; echo "WARNING: cache stage-out FAILED" >&2; }

    if [ "$stage_ok" -eq 1 ]; then
        safe_rm_work "$jobdir" && echo "Stage-out OK; removed $jobdir."
    else
        cat >&2 <<EOF
WARNING: stage-out incomplete -- NOT deleting $jobdir (it holds the only copy).
Recover from the Saion LOGIN node (it can write $BUCKET_PROJECT directly):
    mkdir -p "$dest" "$CACHE_DEST"
    rsync -a "$OUT_DIR"/ "$dest"/
    rsync -a "$RUN_CACHE_DIR"/ "$CACHE_DEST"/
    rm -rf "$jobdir"
EOF
    fi
    echo "========================================================"
}
trap cleanup EXIT

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Same JOB_LOG.md entry as the Deigo pipelines (no-op when submitted via
# submit.sh, which already wrote it). Non-fatal: provenance, not the run.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/$PY_NAME" \
    || echo "WARNING: generate_job_description.sh failed (non-fatal)" >&2

# ==============================================================================
# STAGE IN: code + helper modules -> $jobdir (inside $WORK_BASE)
# ==============================================================================
# gpu_edm.py must be staged: the pipeline puts its own directory at the head
# of sys.path, and a missing gpu_edm would turn every GPU MDE call into an
# ImportError-recorded dead end (an empty network, not a crash).
# gpu_budget.py likewise, and it fails harder: cleaned_helper_functions
# imports it on the GPU path and again from both CCM except-clauses, so a
# missing copy is an unhandled ImportError mid-run rather than a quiet
# degradation. It is a separate module from gpu_edm for the same reason
# gpu_edm is separate -- it must import cleanly with no torch on Deigo.
# The mde_gmn/ PACKAGE is staged too, not just the entry script: the pipeline
# is split across its modules (see scripts/src/mde_gmn/__init__.py). Without
# it the entry script fails at import.
cp -a \
    "$PROJ_DIR/scripts/src/$PY_NAME" \
    "$PROJ_DIR/scripts/src/cleaned_helper_functions.py" \
    "$PROJ_DIR/scripts/src/gpu_edm.py" \
    "$PROJ_DIR/scripts/src/gpu_budget.py" \
    "$STAGE_SRC"/
cp -a "$PROJ_DIR/scripts/src/mde_gmn" "$STAGE_SRC"/ || die "staging source files failed"
[ -r "$BUCKET_DATA_DIR/${DATA_FILE}" ] || die "input data not found on Bucket: $BUCKET_DATA_DIR/${DATA_FILE}"
cd "$STAGE_SRC" || die "cd $STAGE_SRC failed"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK} gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}}"
staged_commit="$(git -C "$PROJ_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
echo "commit=${staged_commit:0:12} jobdir=$jobdir"
# SCOPE_PAIR_COMMIT is set by submit_saion_scope_pair.sh at submission time. The
# sources above were copied out of $PROJ_DIR just now, not when this job was
# queued, so a job that waited days behind another may have staged different
# code than its partner -- which would quietly invalidate a staged-vs-union
# comparison. Warn rather than die: the runbook's resume path legitimately
# resubmits after a code fix, and dying here would throw away the queue wait.
if [ -n "${SCOPE_PAIR_COMMIT:-}" ] && [ "$SCOPE_PAIR_COMMIT" != "$staged_commit" ]; then
    echo "WARNING: staged sources are at ${staged_commit:0:12} but this job was" >&2
    echo "         submitted expecting ${SCOPE_PAIR_COMMIT:0:12}. The repository changed" >&2
    echo "         while this job was queued, so it is NOT running the same code as the" >&2
    echo "         other job in its pair. Results are not directly comparable." >&2
fi
nvidia-smi || echo "WARNING: nvidia-smi failed" >&2

# ==============================================================================
# BACKGROUND: GPU usage log + hourly cache sync
# ==============================================================================
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 60 > "$OUT_DIR/gpu_usage.csv" 2>&1 &
BG_PIDS+=($!)
nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
    --format=csv -l 60 > "$OUT_DIR/gpu_processes.csv" 2>&1 &
BG_PIDS+=($!)
(
    while sleep 3600; do
        stage_dir "$RUN_CACHE_DIR" "$CACHE_DEST" >/dev/null 2>&1 \
            || echo "$(date -Iseconds) WARNING: hourly cache sync to $CACHE_DEST failed (will retry)" >&2
    done
) &
BG_PIDS+=($!)

# ==============================================================================
# RUN
# ==============================================================================
export TSLS_MDE_GMN_GPU_ENABLED=1

export TSLS_MDE_GMN_DATA_DIR="$BUCKET_DATA_DIR"
export TSLS_MDE_GMN_DATA_FILE="${DATA_FILE}"
export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR"
export TSLS_MDE_GMN_MODE="${MODE}"
export TSLS_MDE_GMN_GMN_MODE="${GMN_MODE}"
export TSLS_MDE_GMN_TARGETS="${TARGETS}"
export TSLS_MDE_GMN_D="${MDE_D}"
export TSLS_MDE_GMN_TP="${TP}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN_F1="${EMBED_DIM_RHO_MIN_F1}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1="${CROSS_MAP_RHO_MIN_F1}"
export TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN="${CCM_PREFILTER_RHO_MIN}"
export TSLS_MDE_GMN_CCM_PREFILTER_E="${CCM_PREFILTER_E}"
export TSLS_MDE_GMN_LIBSIZE_RATIOS="${LIBSIZE_RATIOS}"
export TSLS_MDE_GMN_MULTI_MDE_E="${MULTI_MDE_E}"
export TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE="${MULTI_MDE_TARGET_AGGREGATE}"
export TSLS_MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
export TSLS_MDE_GMN_NESTED_M="${NESTED_M}"
export TSLS_MDE_GMN_PARALLEL_MDE_SCOPE="${PARALLEL_MDE_SCOPE}"
export TSLS_MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export TSLS_MDE_GMN_DELAY_SOURCE="${DELAY_SOURCE}"
export TSLS_MDE_GMN_LAG_MODE="${LAG_MODE}"
export TSLS_MDE_GMN_LAG_DENSITY="${LAG_DENSITY}"
export TSLS_MDE_GMN_LAG_STEP="${LAG_STEP}"
export TSLS_MDE_GMN_LAG_COUNT="${LAG_COUNT}"
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
# Lookups scan the durable bucket pool (read-only is all a lookup needs);
# this run's own new entries go to $jobdir/cache and are staged out to CACHE_DEST.
export TSLS_MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export TSLS_MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

python3 -V

# Python runs in the background so the USR1 trap can fire while it runs
# (bash defers traps until a FOREGROUND child exits). `wait` returns early
# when a trap fires (status > 128 while Python is still alive), hence the
# loop: keep waiting until the process is really gone, and keep the status
# of the wait that saw it go.
python3 -u "$PY_NAME" &
PY_PID=$!
while :; do
    wait "$PY_PID"
    rc=$?
    if ! kill -0 "$PY_PID" 2>/dev/null; then
        PY_STATUS=$rc
        break
    fi
done
echo "$PY_NAME exited with status ${PY_STATUS}"
# Outcome, STATUS.txt, stage-out and $jobdir cleanup: the EXIT trap above.
exit "$PY_STATUS"
