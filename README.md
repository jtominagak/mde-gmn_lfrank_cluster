# mde-gmn_lfrank_cluster

Cluster (SLURM) pipeline for building a **Generalized Manifold Network (GMN)**
over multi-region electrophysiology data, with an optional **MDE**
(forward-selection) pre-filtering step. This is a batch port of an
exploratory notebook (`notebooks/simple-mde-gmn.ipynb`, not included in this
repo) — every parameter is read from an environment variable so the pipeline
can be tuned without editing Python.

## Contents

```
scripts/
  batch/
    simple-mde-gmn.sbatch   # SLURM submission script — all tunable parameters live here
  src/
    simple-mde-gmn.py       # the actual pipeline, driven entirely by MDE_GMN_* env vars
pyproject.toml, uv.lock      # exact dependency set (managed with uv)
```

Two custom, non-PyPI-standard packages this depends on: `dimx` (MDE forward
selection) and `gmn` (network construction/generation), plus `pyEDM` for the
underlying empirical dynamic modeling primitives (CCM, Simplex, SMap,
EmbedDimension, ...). All three, plus the rest of the pinned environment, are
declared in `pyproject.toml`.

## What the pipeline does

Given a table of per-timestep channel recordings (e.g. LFP power per
electrode across CA1/mPFC/OFC) and one or more **target** columns to
explain/predict (e.g. `head_position_x`, `head_position_y`), the script:

1. **(optional) MDE** — for each target, forward-selects up to `MDE_D`
   candidate driver channels via convergent cross mapping, gated by
   `EMBED_DIM_RHO_MIN` / `CROSS_MAP_RHO_MIN`. Skipped entirely when
   `MODE=GMN` (see below) — in that case every candidate channel is used
   directly instead of a pre-filtered subset.
2. **CCM interaction matrix** — grid-searches embedding dimension `E` and lag
   `tau` for the channel set, then computes a full pairwise CCM interaction
   matrix (which channels causally drive which) via `apps.InteractionMatrix`.
3. **Per-node dimension estimate** — for every channel/node, estimates its
   own embedding dimension via `pyEDM.EmbedDimension`, then keeps each node's
   *strongest* above-threshold drivers (a manual workaround for a real bug in
   the installed `apps.CreateNetwork`, which keeps the *weakest* drivers —
   see the code comments).
4. **Network construction** — builds the directed driver graph with
   `apps.CreateNetwork`.
5. **GMN (E, tau, per-node theta) search** — every node in the network is
   generated with SMap (see "Why SMap for every node" below). This step
   grid-searches network-wide `E`/`tau` and per-node `theta`, scoring each
   candidate by actually running a full free-running `Generate()` over the
   network and comparing it to the real held-out target trajectory.
6. **Final rebuild** at the winning parameters, plus a `Forecast()` (one-step
   -ahead) diagnostic and a Fig-4e-style generative fidelity metric
   (Simplex-predicting the real trajectory from the generated one).
7. **(optional) MDE-only diagnostics** — PCA/diffusion-map comparison,
   Tp-vs-rho sweep, univariate E/tau grid. Skipped when `MODE=GMN`.

### Why SMap for every node, not Simplex

`gmn`'s Simplex branch never calls `pyEDM.Embed()` before predicting, so it
can't do a real delay embedding — `embedded=True` silently discards `tau`.
The SMap branch does embed properly. Per explicit project direction, this
pipeline therefore uses SMap uniformly for every node rather than mixing
generators. One consequence: every node (including undriven leaf nodes)
needs `E >= 2`, or SMap's own predictor-exclusion convention leaves it with
zero predictors — hence `E_MIN` must stay `>= 2`.

The Python script also monkeypatches a real bug in the installed
`gmn.Node.Forecast` (missing `Embed` import + a tuple/Series `.max()` bug on
the divergence check) rather than editing the installed package, so the
script stays self-contained across machines.

## Running on the cluster

```bash
# one-time setup
module load python/3.11
cd ~/pao_rotation
uv sync                     # creates .venv from pyproject.toml/uv.lock

# submit
sbatch scripts/batch/simple-mde-gmn.sbatch
```

All tunable parameters (data file, mode, targets, MDE thresholds, channel
cap, E/tau/theta search ranges, core counts) live in the **MODIFIABLE
PARAMETERS** block at the top of `simple-mde-gmn.sbatch` — edit that block,
don't touch the Python file. The script expects a `uv` venv at
`~/pao_rotation/.venv` (`VENV_DIR`) and the input CSV under
`~/pao_rotation/data/` (`DATA_FILE`); adjust both if your checkout layout
differs.

### Important cluster considerations

- **This submits one SLURM job, not a job array.** There is no built-in way
  to fan the parameter search out across multiple SLURM jobs — the entire
  E/tau/theta search below runs as a sequential loop inside a single job's
  single Python process. If you want several configurations (e.g. different
  `MODE`, `TARGETS`, or `DATA_FILE`) run concurrently, submit multiple
  `sbatch` calls with edited copies of the script.
- **Two independent layers of parallelism, on purpose kept apart.**
  `--cpus-per-task` (default 32) sizes `MDE_GMN_CORES`, which most
  parallel steps (`dimx.MDE`, `EmbedDimension`, `PredictNonlinear`) use
  directly and scale near-linearly with. The CCM interaction-matrix step is
  the exception: `apps.InteractionMatrix`'s own outer worker pool
  (`INTERACTION_MATRIX_OUTER_CORES`, default **2**, deliberately *not* tied
  to `--cpus-per-task`) calls `pyEDM.CCM()` once per channel pair, and
  `CCM()` creates its *own* fresh multiprocessing pool sized to the node's
  full visible core count on every call. The two pools don't coordinate, so
  peak concurrent processes is `outer_cores × cpu_count()`, not
  `cpu_count()` — setting the outer count equal to `--cpus-per-task` would
  *square* your core request into a process count (confirmed empirically:
  an 8-core request produced ~90 live processes on a 10-core test machine).
  Don't raise `INTERACTION_MATRIX_OUTER_CORES` to match `--cpus-per-task`.
- **Wall time won't scale linearly with `--cpus-per-task`** for the
  interaction-matrix step regardless, since every channel pair pays a fresh
  pool creation/teardown cost. If a job runs far longer than `--time`
  allows, check actual CPU utilization (`srun --jobid=<id> top`) before
  assuming the pipeline is stuck.
- **`--time`/`--mem`/`--cpus-per-task` are starting points, not tuned
  values.** Time the interaction-matrix + search steps on a cheap
  calibration job first (small `REGION_CHANNEL_CAP`, narrow E/tau ranges),
  then scale up to your real channel count / search grid before requesting
  a multi-day allocation for a full run.
- **Checkpoints live on ephemeral `/flash` scratch, not in the results
  directory.** The E/tau/theta search checkpoints to disk after every
  candidate so a resubmitted job can resume a multi-hour search — but the
  scratch directory (`$FLASH_BASE/run_simple_mde_gmn.$SLURM_JOB_ID.*`) is
  deleted by an `EXIT` trap the moment the job exits, *including* on a
  SLURM timeout (`SIGTERM`), before results get copied to `$OUT_DIR`. If a
  search might not finish inside `--time`, checkpoints from that attempt
  will be lost when the job is killed. If resuming across timeouts matters
  for your run, point `MDE_GMN_RESULTS_DIR` at a subdirectory of `$OUT_DIR`
  directly instead of the scratch tempdir (trades some `/flash` I/O speed
  for surviving a timeout).
- **`GROUP`/allocation path (`FLASH_BASE`) and `module load python/3.11`
  are placeholders** — adjust `FLASH_BASE="/flash/FukaiU/${USER}"` to your
  lab's actual flash allocation, and drop the `module load` line if your
  `uv` venv already bundles its own interpreter.

## What actually runs, with the current default parameters

The sbatch script as checked in currently sets `MODE=GMN`,
`GMN_MODE=combined`, `TARGETS=head_position_x,head_position_y`,
`DATA_FILE=J1620210605_epoch_8_freq_40_Hz.csv`, and leaves
`REGION_CHANNEL_CAP` at its uncapped default (999). Against this repo's
actual data file, that means:

- **1 SLURM job**, requesting 32 cores / 48G / up to 2 days.
- **No MDE** (`MODE=GMN` skips it) — the candidate driver pool is instead
  *every* channel in the data: 147 `CA1_*` + 87 `mPFC_*` + 234 `OFC_*` =
  **468 channels**.
- **`GMN_MODE=combined`** means both targets are folded into **one**
  network (not two independent ones), so the network has up to
  468 + 2 = **470 nodes**.
- **Interaction matrix**: a CCM run over essentially all directed pairs of
  those 470 nodes — on the order of **2×10⁵ CCM channel-pair computations**.
  This is by far the most expensive and highest-risk step at this channel
  count (see "Important cluster considerations" above for why it doesn't
  parallelize cleanly). **Recommend lowering `REGION_CHANNEL_CAP`** (e.g. to
  20-30) for an initial test run before attempting the uncapped run.
- **Per-node dimension estimate**: ~470 sequential `EmbedDimension` calls
  (each itself parallelized across `CORES`).
- **GMN (E, tau, theta) search**: with the default ranges
  (`E_MIN=3, E_MAX=16, E_STEP=2` → E ∈ {3,5,7,9,11,13,15}, 7 values;
  `TAU_MIN=-15, TAU_MAX=0, TAU_STEP=2` → tau ∈ {-15,-13,...,-1}, 8 values),
  that's **7 × 8 = 56 (E, tau) candidates**, run **sequentially within the
  same job** (not as separate SLURM jobs/tasks). `SMAP_SEARCH_MODE`
  defaults to `coarse_retune`: at each of the 56 candidates, every node's
  `theta` is re-tuned over a coarse 5-value grid, then **one full
  `Generate()` call** is run over the whole 470-node network — so 56 full
  network `Generate()` calls total. (The alternative `joint` mode would
  instead sweep the full 15-value theta grid inside the loop —
  56 × 15 = 840 `Generate()` calls, ~15× the cost — and is only meant to be
  turned on deliberately.)
- **+2 more full `Generate()`/`Forecast()` calls** for the final rebuild at
  the winning parameters and the Forecast() diagnostic.
- **No MDE-only diagnostics** (skipped when `MODE=GMN`).

### Why each parameter sweep exists

- **CCM E/tau grid** (`target_E_tau_rho_grid`, cheap — plain `Simplex`
  calls, not full `Generate()`): picks the delay-embedding dimension and lag
  used to build the interaction matrix itself. CCM's ability to detect a
  real causal/driving relationship between two channels depends on `E`/`tau`
  matching the underlying dynamics' actual embedding dimension — a fixed,
  guessed `E`/`tau` risks systematically missing or fabricating edges in the
  network. This sweep is picked once per network (maximizing mean cross-map
  rho across targets), not per channel pair.
- **Per-node dimension estimate** (`EmbedDimension`): each node gets its own
  driver count (`node_dimensions`) rather than one global value, because
  different channels/regions genuinely have different intrinsic
  dimensionality — using one global count would over- or under-fit whichever
  nodes it doesn't match.
- **GMN-wide (E, tau, theta) search** (the expensive one): pairwise CCM
  skill from step 2 says nothing about whether the *whole coupled network*,
  run free-running for `PREDICTION_LENGTH` steps, stays accurate — errors
  compound across nodes and coupling can destabilize dynamics that looked
  fine pairwise. There's no way to infer the network-optimal `E`/`tau` from
  pairwise statistics alone, so each candidate is scored by literally
  running `Generate()` and comparing it against the real held-out target
  trajectory. `theta` (SMap's nonlinearity/localization parameter) is
  re-tuned at every `(E, tau)` candidate rather than fixed, because the
  locally-linear-vs-nonlinear tradeoff SMap is interpolating depends on the
  embedding geometry, which changes with `E`/`tau`.

## Outputs

Everything lands under `results/job_<SLURM_JOB_ID>/` on the login/head node
after the job completes (copied back from `/flash` scratch at the end of the
script): MDE selection CSVs (if `MODE=MDE-GMN`), the interaction matrix and
network pickle, per-candidate search checkpoints, generated/forecast CSVs,
and all diagnostic figures (rho-vs-dimension, E/tau heatmaps, network graph,
`Generate()` vs. actual overlays, fidelity bar chart).

## Known caveats

See the inline comments in both `simple-mde-gmn.sbatch` (MODIFIABLE
PARAMETERS block, KNOWN CAVEAT section) and `simple-mde-gmn.py` (module
docstring, `build_network()` docstring) for the full evidence trail behind
the `CreateNetwork` driver-ranking bug workaround, the `gmn.Node.Forecast`
monkeypatch, and the interaction-matrix process-count caveat summarized
above.
