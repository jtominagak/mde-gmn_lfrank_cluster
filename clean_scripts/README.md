# mde_gmn — nested-MDE network discovery with GMN simulation

Takes a multi-channel spike-train recording and a behavioural signal, and asks:
**which neurons drive the behaviour, at what delays, and does a network built
from those answers reproduce the recording when you let it run on its own?**

This is a cleaned-up copy of the pipeline. It is behaviour-for-behaviour the
same as the original — the same algorithm, the same gates, the same outputs —
restructured so the logic is followable, with the redundant compute removed.
[What changed](#what-changed-from-the-original) lists every difference.

---

## The idea, in one page

The input is a 40 Hz recording: a few hundred neurons, each a sparse train of
spikes, plus the animal's head position.

**1. Spike trains are a bad variable for delay embedding.** Mostly zero,
occasionally one — every neighbour in the embedding space collapses onto the
"no spike" state. So each channel is recoded as **time since last spike**
(TSLS): a sawtooth that counts up between spikes and drops to zero at each one.
Same information (the drop times *are* the spike times), but now a continuous
signal with local geometry an embedding can use.

**2. For each node, find what predicts it.** Forward selection: score every
candidate, keep the best, re-score the rest *alongside* the one kept, repeat.
A candidate that predicts well is not necessarily *driving*, so each kept
candidate also has to pass a causality gate — convergent cross mapping, which
checks that prediction keeps improving as more data is used. That is the
signature of a real causal link rather than a coincidence.

**3. Assemble those answers into one acyclic network.** Starting at the
behavioural targets and working outwards. Real observables form a backbone;
delayed copies attach as leaves, standing for input the network does not
explain.

**4. Hand it to GMN and let it run.** Each node predicts itself from its own
discovered drivers, at their own delays. Then free-run the whole thing from a
seed and compare against data it never saw.

Because these signals cannot be judged by Pearson correlation — it is dominated
by the smooth ramps between spikes — scoring goes through event-timing metrics
that decode predictions back to spike times and match them. See
[docs/parameters.md §6](docs/parameters.md).

---

## Layout

```
clean_scripts/
  src/
    run_parallel_mde.py   entry point: parallel network builder (the production one)
    run_serial_mde.py     entry point: serial builder (the reference implementation)
    edm.py                the TSLS-aware EDM engine: proper_mde, proper_ccm, proper_simplex
    gpu_edm.py            optional CUDA cross-map primitive
    gpu_budget.py         cross-process GPU memory admission control
    mde_gmn/              the pipeline, one module per stage
  batch/
    saion_parallel_mde.sbatch   GPU production run
    serial_mde.sbatch           CPU reference run
    submit.sh                   submit + write the JOB_LOG.md entry
    generate_job_description.sh what changed since the last job
  saion/
    cluster_env.sh              storage roots and stage-out helpers
    submit_saion.sh             pick the GPU partition that starts soonest
    submit_saion_scope_pair.sh  submit both parallel scopes as one comparison
    setup_gpu_venv.sh           build the CUDA venv
  tests/                 24 scripts, no pytest needed
  docs/
    parameters.md        where every number comes from, and which are guesses
    deployment.md        running it, reading the outcome, resuming, troubleshooting
```

### Reading order

1. `src/mde_gmn/__init__.py` — the module map and a dependency-safe reading
   order.
2. `src/mde_gmn/network.py` — the algorithm the whole project is about: what the
   graph means, what the two stages do, and how it stays acyclic.
3. `src/mde_gmn/pipeline.py` — `main()`, the run step by step.

Then whichever stage you need. Every module's docstring explains the idea before
the mechanics.

---

## Running it

```bash
# On a cluster (see docs/deployment.md for one-time setup)
cd ~
<checkout>/clean_scripts/saion/submit_saion.sh saion_parallel_mde.sbatch run_parallel_mde.py

# The tests. Never on a login node.
srun -p <partition> -t 0:30:00 -c 4 --mem=8G bash -c '
  cd <checkout> && fail=0
  for t in clean_scripts/tests/test_*.py; do
    .venv/bin/python3 "$t" >/dev/null 2>&1 || { fail=1; echo "FAIL $t"; }
  done; echo "SUITE-EXIT=$fail"'
```

**Nothing is configured by editing Python.** Every parameter is a
`TSLS_MDE_GMN_*` environment variable; `src/mde_gmn/config.py` is the complete
list, and the sbatch `MODIFIABLE PARAMETERS` block is the same list from the
submitting side. Retired settings *raise* rather than being ignored, so a stale
sbatch fails at startup instead of running with a value nobody honours.

### Three things that will surprise you

1. **Importing `gmn_embedding` patches `gmn`.** Three `gmn.Node` methods are
   replaced at import time, so GMN honours this pipeline's per-node embeddings
   instead of one global `(E, tau)`. Anything that constructs `gmn` objects must
   import it first.
2. **The network builders reach MDE through the module object** — `from mde_gmn
   import mde`, then `mde.run_mde_for_target(...)`. The tests install a fake by
   assigning to `mde_gmn.mde`; importing the name directly would bind the real
   function and the fake would be silently bypassed.
3. **The cache key includes the entry script's path.** That is why `main()`
   takes `entry_path`. Without it the serial and parallel pipelines would share
   one cache namespace, because every other input to the fingerprint is
   identical.

### Two builders, same network

`run_serial_mde.py` runs MDE one breadth-first wave at a time. Correct, and the
reference — but each wave ends in a barrier, and a wave is often two nodes, so
most of the machine idles.

`run_parallel_mde.py` computes MDE for every node that could be needed up
front, with no barriers, then replays the graph walk against those results. It
computes more than it consults; that is the trade — wasted compute bought with
wall-clock. Under `PARALLEL_MDE_SCOPE=staged` it builds the **same network**,
edge for edge, which `tests/test_parallel_mde_equivalence.py` asserts.

---

## What changed from the original

Grouped by what each change was for. **No change alters what the pipeline
computes**, except where explicitly noted, and the full test suite passes
identically before and after (24/24).

### Redundant compute removed

**Every CCM call computed both cross-map directions and discarded one.** CCM
measures `columns → target` (c2t) and `target → columns` (t2c) at once. The MDE
causality gate — the pipeline's hottest call by a wide margin, once per
surviving candidate, per dimension, per node — reads **only t2c**. The CCM
pre-filter reads **only c2t**. Each discarded direction cost a KDTree build and
query per (library size × sample) cell, and on the GPU a second (M, M) distance
matrix held for the whole call.

`proper_ccm`/`multi_proper_ccm` now take a `directions` argument and compute
only what the caller will read. The library draws still happen for both, in the
same order, so a given seed produces **bit-identical numbers** for the
requested direction — verified directly against the original before anything
else was changed. On the GPU path it also halves `crossmap_bytes()`'s dominant
term, which is what decides how many workers fit on a card.

Under the default `GMN_MODE=combined` the saving is larger still: the forward
pass of `multi_proper_ccm` cross-maps *every target separately*
(`len(targets) × libSizes × sample` predictions per candidate) and
`multi_proper_mde` never looks at it.

**A full network free-run was computed twice and the first result thrown away.**
`run_theta_search()` ran `Generate()` purely to print a score it then discarded.
With the default single-offset configuration, `sweep_generate_start()` skipped
straight past and `rebuild_at_best_params()` ran the *identical* call again —
same network, same theta, same start row, same length. A full free-run is the
single most expensive step in the pipeline. The throwaway call is gone; the
final rebuild already prints strictly more (every node's score, targets
included, at exactly those parameters).

**The per-network data frame was re-read and re-parsed four times.** The theta
search, the final rebuild, the Forecast diagnostic and the fidelity scoring each
did `pd.read_csv(net_result['DATA_AFTER_MDE'])` on the same
multi-hundred-megabyte file. `build_network()` now keeps the frame it just
wrote, and `tsls_data.network_data_frame()` falls back to reading the file when
a `net_result` was assembled by hand, so nothing depends on the optimisation.

**Smaller ones:** `observable_and_delay()` was called twice per candidate column
where one pass gives both halves; Stage 2 re-derived each candidate's underlying
observable once per node instead of once per stage; `proper_mde` rebuilt a
reset-event count over a whole column for every candidate at every dimension,
for a log line (now memoised); the ranking stage built a dict whose string keys
nothing ever read.

### Dead code removed

~860 lines from the EDM engine that nothing in this pipeline reaches:
`mean_in_time_grid`, `multi_simplex`, `rolling_simplex`, `top_neurons`.

`_compute_time_since_previous_spike_reference` was removed and then **restored**
— it looks unused, but it is what `tests/test_r1_tsls_vectorized.py` checks the
vectorised implementation against, so it is test infrastructure, not dead code.

### Duplication collapsed

| Was | Now |
|---|---|
| Stage 1 and Stage 2 each had its own ~110-line pool/heartbeat/failure block in `network.py`, and the parallel builder a third copy | one `waves.run_mde_batch()`, which all three use |
| `run_mde_for_target` and `run_multi_mde_for_targets` each had their own copy of the cache load, cache save and result-namespace construction | shared helpers; the two functions are now just their own forward-selection calls |
| `_failed_mde_result` / `_failed_multi_mde_result` were the same dict twice | one builder, two thin wrappers |
| Three convergence tests, two of them identical | `edm.converged()`, taking the x-scale as an argument (see below) |
| `_generalized_node_forecast` / `_generalized_node_generate` were two long `if/elif` chains over node function type, with the one live branch buried in the middle | one small function per type plus a dispatch table; every branch kept verbatim |
| The depth-0 seed in `pipeline.py` was two ~50-line branches differing in which MDE function they call | one loop, one `if joint:` at the call |
| ~60 near-identical `export TSLS_MDE_GMN_*=` lines in the sbatch | one loop over a parameter list (verified to export the identical 60 variables) |

**The three convergence tests were deliberately *not* unified.** They look
interchangeable and are not: `proper_mde`'s gate checks the fitted slope alone
against library *values*, while the interaction matrix and `pass_fail` check
the endpoint change *and* the slope against the library *index*. Three things
differ, and each changes the verdict. `edm.converged()` shares the
endpoint-and-slope form and takes `x` as a parameter; `proper_mde`'s gate was
left exactly as it was. Flattening them would have silently moved both
thresholds. This is noted here because it is the one place where the obvious
cleanup is wrong.

### A real defect fixed

`generate_job_description.sh` diffed only the **entry script** to report
"Python pipeline changes". That script is fifteen lines of boilerplate — all the
real code is in the package beside it. Every entry in the existing `JOB_LOG.md`
therefore says `_No changes._` while the pipeline underneath was being
rewritten. It now snapshots and diffs **every `.py` under `src/`**.

### Reproducibility

Stage 2's node order was the iteration order of a Python `set`, which is not
reproducible across processes. It only ever affected the order of rows in the
edge CSV and columns in the exported frame — never the graph — but both builders
now sort it the same way, so those outputs are stable and the two builders agree
node for node.

### Readability

Renamed for what things are: `cleaned_helper_functions.py` → `edm.py`;
`no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py` →
`run_parallel_mde.py`; `tsls-mde-gmn.py` → `run_serial_mde.py`. Names used
across module boundaries are no longer underscore-private
(`failed_mde_result`, `embedding_spec_for`, `generalized_embed`,
`simulated_nodes`, `GPU_MEMORY_STOP_ADVICE`).

The 953-line sbatch is 508 lines: the parameter block keeps a one-line comment
each, and the sizing evidence, calibration data and job archaeology moved to
`docs/parameters.md` and `docs/deployment.md` — where it is organised by topic
and actually findable, rather than interleaved with the values it justifies.
Nothing was discarded.

### Publishability

No account-specific path is baked into any script. Everything comes from an
environment variable with a working default, and `cluster_env.sh` derives the
project root from its own location. The exceptions are six `#SBATCH
--output`/`--error`/`--mail-user` lines, which **cannot** be variables — Slurm
parses its directives before any shell runs. They are grouped at the top of each
sbatch under an `EDIT FOR YOUR ACCOUNT` banner.

### Scope

This is the parallel pipeline and its serial reference. Deliberately not
included: the separate 1 Hz `simple-mde-gmn.py` pipeline (a different, older
algorithm, out of scope), four near-duplicate sbatch variants differing in two
or three parameter lines, superseded backups, one-off GPU prototypes, and the
calibration script whose results are written up in `docs/parameters.md`.

The `tests/` directory is a copy of the repo's suite adapted to this tree —
paths, the renamed modules, and the one test that asserted on the old
executor structure (rewritten to assert the same contract against
`waves.run_mde_batch`). The halves that exercised the out-of-scope 1 Hz pipeline
were removed.

---

## ⚠️ The cache does not carry over

Cache entries are fingerprinted on a hash of the **whole package's
docstring-stripped AST** plus the installed dependency versions. That is
deliberate: a real code change must miss the cache rather than silently reuse a
result computed by different code.

Removing redundant compute changes function bodies, so **the first run from
`clean_scripts/` recomputes from scratch**, even though it computes the same
answers. There is no way around it that does not weaken the safety property —
the fingerprint cannot know that a change was behaviour-preserving.

Point `TSLS_MDE_GMN_PREV_RUNS_DIR` at a fresh cache root for this tree, and
budget the first run accordingly.
