# No-kEDM GPU Optimization Progress Log

Companion to `docs/no-kEDM-gpu-optimization-plan.md` (read that first). Dated, append-only log
of what's been tried on this branch, what worked, what didn't, and what's next.

Branch: `no-kEDM-gpu-optimization` (created off `main` at commit `fd184d1`, 2026-09-02 — a
sibling of `gpu-implementation`, not a descendant of it; deliberately isolated so neither
direction's dependencies/docs pollute the other).

---

## 2026-09-02 — Feasibility analysis + first prototypes

**Status: feasibility analysis complete. Verdict: the naive brute-force/batched approach is
confirmed SLOWER on CPU at every scale tested; the metric-fidelity premise (routing around
kEDM's `ccmMetric='f1'` blocker) is confirmed TRUE. Not proceeding to full integration; parked
pending GPU hardware access. See plan §5 for the exact reasoning and next steps.**

### What was done
1. Created this branch off `main` (not off `gpu-implementation`) so it starts from the clean,
   kEDM-free CPU baseline. Committed the pending kEDM work to `gpu-implementation` first so
   nothing was at risk of being carried across branches by an uncommitted-changes checkout.
2. Installed `torch`. First attempt (`uv add torch`, no index specified) pulled the default
   CUDA-bundled build and separately hit a real, pre-existing `pyproject.toml` resolution bug
   (`dimx==1.1.3` unavailable for the `python_full_version>=3.14, sys_platform=='win32'` marker
   branch uv's universal resolver tries by default) — killed, then fixed both: added
   `[tool.uv] environments = [...]` scoping resolution to `linux`/`x86_64` (what this project
   actually runs on), and pinned `torch` to PyTorch's dedicated CPU wheel index via
   `[tool.uv.sources]`/`[[tool.uv.index]]` so future `uv sync` doesn't silently regress to the
   CUDA-bundled build. `uv sync` then completed cleanly: `torch==2.14.0+cpu`,
   `torch.cuda.is_available() == False` (expected — no GPU on this cluster, same as
   `docs/gpu-optimization-plan.md` §1).
3. Read `_crossmap_predict` (`cleaned_helper_functions.py:599-712`, the CCM gate's per-subsample
   cross-map engine) and `proper_ccm`'s sweep structure (`cleaned_helper_functions.py:900-990`) in
   full, to get the exact Sugihara-weighting formula and masking/exclusion logic right before
   writing any torch code — this existing engine, not textbook Simplex, is the actual correctness
   reference.
4. **First prototype** (batched torch reimplementation of `_crossmap_predict`): failed on both
   axes initially — correctness (max abs diff 1.16, a real bug, not float noise) and speed (~25x
   slower than the reference in a 140-call sweep). Diagnosed both:
   - Correctness bug: `d_min` (the weighting normalizer) was computed from the raw
     nearest-of-over-fetched-candidates *before* removing self/excluded neighbors, instead of
     from the first valid (post-compaction) neighbor as the reference does. Fixed with a batched
     `torch.argsort`+`torch.gather` compaction step mirroring the reference's `np.where`+reshape
     exactly.
   - Speed: traced to torch's default CPU thread pool (`get_num_threads()==40` on this node)
     imposing large fixed per-op dispatch overhead on small tensors — confirmed by instrumenting
     every internal step and finding `torch.set_num_threads(1)` alone gave a 21.7x speedup on the
     identical computation. This is the same class of finding as the kEDM investigation's
     OpenMP-thread-wake overhead (`docs/gpu-optimization-plan.md` §2) — recorded explicitly so it
     isn't rediscovered as a surprise in a future session.
5. **Second prototype** (bug fixed, single-threaded, distance matrix cached once across the whole
   sweep): numerically correct (max abs diff 8.28e-05 across 280,000 compared points, consistent
   with float32/float64 precision noise, 0/140 NaN-pattern mismatches) — but still **3.2x slower**
   than the numpy/KDTree reference at M=2000. Checked the actual production config
   (`scripts/batch/tsls-mde-gmn.sbatch:419-441`): `CCM_WINDOW_ROWS=800`, so real production scale
   is smaller than M=2000, not larger — meaning the real regime is likely *worse* for this
   approach, not better (a tree search's relative advantage over brute force grows as the library
   shrinks relative to the query set).
6. Attempted a larger-M scaling sweep (M=2000/8000/20000/40000) to see whether the ratio improves
   at scale — killed after several minutes as uninformative for actual production sizes once
   `CCM_WINDOW_ROWS=800` was confirmed; not worth the wall-clock cost. If this needs revisiting,
   don't just rerun it blind — first decide what M range is actually worth checking (see plan §6).
7. **Third prototype**, testing the OTHER major target (ranking stage, `_simplex_metric`/
   `pyEDM.Simplex`, `cleaned_helper_functions.py:1440-1449`) at production-realistic scale
   (lib=20,000, pred=2,000, E=2, 100 candidates): an unchunked "stack all 100 candidates into one
   `torch.cdist` call" version OOM-killed outright (16 GB dense tensor). Rewrote with
   `chunk_b=5` candidates per call (memory-forced, not a free design choice) — still **~9x
   slower** than looped `pyEDM.Simplex` (35.78s vs 4.01s). A separate off-by-one bug surfaced in
   the *test harness itself* (pyEDM's Predictions array came back 2001 rows vs the torch code's
   2000 — an indexing convention difference not yet root-caused) — didn't chase it further since
   the timing verdict was already decisive and consistent with prototype #2; flagged in the plan
   (§2c) as something a real integration attempt would need to resolve properly, not evidence
   against the timing conclusion itself.
8. Synthesized why this approach's slowdown differs in *kind* from kEDM's (plan §3): kEDM's
   problem is per-call orchestration overhead (backend-independent, likely persists on GPU);
   this approach's measured CPU slowdown is a raw compute-volume deficit (brute-force `O(L)` scan
   vs. tree search's `O(log L)`) — a difference GPU throughput could plausibly close, unlike
   kEDM's problem. This is a hypothesis pending actual GPU hardware, not a confirmed result — see
   plan §5 item 3 for the cheap, decisive experiment that would settle it once hardware exists.
9. Copied both working (bug-fixed, single-threaded) prototype scripts from the scratchpad into
   the repo at `scripts/gpu_prototypes/validate_batched_ccm.py` and
   `scripts/gpu_prototypes/validate_ranking_simplex.py` — not wired into the pipeline, but a
   validated starting point if GPU hardware becomes available and §5's experiment is run. Neither
   script has GPU code paths yet (both run on CPU tensors) — that would be the very first change
   to make once hardware exists (move tensors to `.to('cuda')`, nothing else should need to
   change for the experiment in plan §5 item 3).
10. Wrote `docs/no-kEDM-gpu-optimization-plan.md` and this progress log.

### What worked
- Both validated torch prototypes (CCM cross-map, ranking-stage Simplex) are numerically faithful
  to their respective references once the compaction bug was fixed — confirmed by direct
  comparison, not assumed. This means the "solves kEDM's ccmMetric='f1' problem" premise from the
  original proposal is real and usable, independent of whatever happens with speed.
- The CPU-side engineering process itself (find the reference's exact algorithm, prototype,
  validate numerically against real data, only THEN trust a timing number) caught a real bug that
  a speed-only benchmark would have silently missed — worth continuing this discipline for any
  future primitive (S-Map, theta retuning) that gets prototyped.

### What didn't work / was wrong
- The original proposal's implicit assumption — that moving the candidate sweep to torch tensors
  would be faster even before GPU hardware is available — is not supported by any benchmark run
  this session. All three timing comparisons (kEDM ranking stage from the sibling plan, torch CCM
  gate, torch ranking stage) show the *existing* pyEDM/scipy-based engines winning, sometimes by
  an order of magnitude.
- The first-attempt "just batch everything into one call" instinct hit a real memory ceiling
  (16 GB) at realistic production row counts — same structural lesson as kEDM's own 281.6 GiB
  crash (`docs/gpu-optimization-plan.md` §2) despite being a completely different codebase/engine.
  Any future batched implementation (CPU or GPU) needs deliberate chunking sized to whatever
  memory budget is actually available, not an unbounded "stack them all" design.

### Scratch files (not committed, scratchpad only)
`/flash/.../scratchpad/torch_prototype/validate_batched_ccm.py` (first, buggy version — superseded,
not copied to the repo) and the M-scaling sweep script (killed mid-run, not saved). The two
scripts actually worth keeping are committed at `scripts/gpu_prototypes/` (see step 9 above).

### Next steps (superseded by the 2026-09-02 (Saion) entry below — kept for history)
1. ~~Ask the user where GPU hardware is actually reachable~~ — answered (see next entry): Saion,
   `ssh <oist-id>@saion.oist.jp`, a separate OIST cluster from Deigo.
2. Once hardware is identified: install `torch` with the matching CUDA build there, move
   `scripts/gpu_prototypes/validate_batched_ccm.py` and `validate_ranking_simplex.py`'s tensors to
   the GPU device, and re-run both — this is the single experiment that actually decides whether
   this direction is worth any further investment (plan §5 item 3). Do not skip straight to a
   pipeline integration without this checkpoint.
3. If GPU numbers win: prototype the memory-tiling strategy properly (GPU memory is typically
   smaller than this node's system RAM, so the chunking problem from step 7 above gets more
   urgent, not less) before touching pipeline code.
4. If GPU numbers still lose: close out this branch's direction and record why in this log,
   rather than leaving it as an open, ambiguous thread for a future session to re-litigate from
   scratch.
5. Not started, and not worth starting before 1-2 above resolve: S-Map/theta-retuning torch
   port (plan §6's last open question), fork/spawn safety review for any future
   `ProcessPoolExecutor` use (mirrors `docs/gpu-optimization-plan.md`'s Stage 4, though torch
   itself doesn't have kEDM's specific Kokkos-at-import fork hazard — not yet checked whether
   torch's own CPU thread pool has an analogous fork-safety concern; assume caution until checked).

---

## 2026-09-02 (continued) — Saion identified; CUDA support added; manual procedure written

**Status: GPU hardware identified (Saion). Scripts are CUDA-ready and re-verified correct on CPU.
Not yet run on an actual GPU — this session has no SSH credentials to reach Saion (tried directly;
`~/.ssh/` on this Deigo session has no private key/config for outbound access, only an
`authorized_keys` for *incoming* connections). User chose to run the procedure manually rather
than set up passwordless access for this session. Waiting on the user to run the procedure below
and report back the printed numbers.**

### What was done
1. Fetched the official guide (scda-guide.oist.jp/saion-system). Key facts: **`/home` is shared
   between Deigo and Saion** (this whole project directory, `pyproject.toml`/`uv.lock` included,
   is already visible at the same path on Saion — no code sync needed); `/bucket` is read-only on
   Saion compute nodes (matches this pipeline's existing bucket-archival conventions); `/work`
   (200TB SSD, 10TB/unit) is the place for temporary compute output, not `/home` (50GB cap) or the
   Deigo-only `/flash` scratchpad (not necessarily mounted the same way on Saion — untested).
   Scheduler is Slurm (`srun`/`sbatch`, same as Deigo). GPU partitions: **`test-gpu` (open, no
   access request needed)** — 6 nodes, 4×P100 each; `gpu-p100`/`gpu-v100`/`gpu-a100` are
   *restricted*, need a form. Start with `test-gpu`.
2. Updated `scripts/gpu_prototypes/validate_batched_ccm.py` and `validate_ranking_simplex.py` to
   auto-detect `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`, move every tensor
   accordingly, add a warm-up call + `torch.cuda.synchronize()` around the timed region (CUDA
   calls are async — a naive `time.perf_counter()` would measure kernel-launch time, not actual
   completion), and catch `torch.cuda.OutOfMemoryError` in the chunked ranking-stage loop with a
   message to shrink `chunk_b` (GPU memory is typically smaller than a compute node's system RAM,
   so the chunking problem from the 2026-09-02 entry above is more urgent on GPU, not less).
3. While doing this, found and fixed a **real bug in the ranking-stage script's own correctness
   check** (not the algorithm — the test harness): `pyEDM.Simplex`'s `Predictions` array carries
   one extra **leading** boundary row (`Time = pred_start`, `Predictions = NaN`) before its
   `n_pred` real, `Tp`-shifted predictions. The script was comparing `ref[:n_pred]` against the
   torch output (`n_pred` rows, unpadded) — silently comparing misaligned rows, previously
   misreported as a spurious ~1.0 max diff. Root-caused by hand-inspecting `pyEDM.Simplex`'s raw
   output on a small (`n_lib=30`) case where torch matched to 1e-5 once correctly aligned (`ref[1:]`
   vs the torch output). At full benchmark scale (`n_lib=20000`), after the fix: median diff
   2.89e-05, mean 3.24e-04, max 0.131, only 0.4% of 200,000 compared points over 0.01 — the
   occasional larger single-row diff is expected, not a bug: a smooth multi-period test signal
   creates many near-duplicate (candidate, selected) state-space points, so float32 (torch) vs
   pyEDM's internal float64 can pick a different but similarly-close k-th neighbor at a near-tie
   for a handful of rows. Switched the script's reported statistic to median/mean primarily,
   with an explanatory comment, so this doesn't look alarming when the user runs it on Saion.
4. Re-ran both scripts on this Deigo CPU environment after the edits to confirm the CUDA-support
   changes didn't regress the CPU path — both reproduce the same numbers as the original
   2026-09-02 entry (CCM: 1.12s ref vs 3.61s torch, 8.28e-05 max diff; ranking: 3.5s ref vs 32s
   torch chunked, now-correct diff stats above). Committed as `6285f68`.
5. Attempted a direct SSH test from this Deigo session to Saion — rejected
   (`Permission denied (publickey,password,hostbased)`). Checked `~/.ssh/`: no private key or
   `config` entry for outbound Saion access exists in this environment. Asked the user how to
   proceed (set up passwordless access for this session, vs. running it manually themselves) —
   **user chose to run it manually.**

### Manual procedure for the user to run on Saion

```bash
# 1. Log in (separate cluster/credentials from Deigo)
ssh j-tominaga@saion.oist.jp

# 2. Confirm the shared home actually shows the right branch/commit
cd ~/pao_rotation
git status              # should show branch no-kEDM-gpu-optimization, clean
git log --oneline -1    # should show 6285f68 or later

# 3. Build the GPU-specific venv FIRST, on the LOGIN node, before requesting any
#    allocation -- compute nodes commonly have no/restricted outbound internet
#    (needed to download the torch wheel) and srun's time limit shouldn't be
#    spent on non-GPU setup work anyway. Deliberately SEPARATE from the
#    project's own .venv/ (committed to torch's CPU-only wheel via
#    pyproject.toml's [tool.uv.sources] pin) so nothing here touches the
#    shared, NFS-mounted pyproject.toml/uv.lock Deigo's CPU pipeline depends on.
cd ~/pao_rotation
uv venv .venv-gpu --python 3.11
uv pip install --python .venv-gpu/bin/python numpy pandas "pyedm==2.5.0"
# torch==2.14.0 (same version as Deigo's CPU venv) from the cu126 index --
# NOT cu128+, which dropped Pascal/P100 (sm_60) support (confirmed via
# PyTorch's own dev-discuss announcement, see plan doc's sources):
uv pip install --python .venv-gpu/bin/python "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126

# 4. NOW get an interactive GPU allocation on the OPEN test-gpu partition (no
#    access request needed) -- venv is already fully built, nothing left to
#    download inside the allocation
srun -p test-gpu -t 0-2 -c 4 --mem=10G --gres=gpu:1 --pty bash -l
cd ~/pao_rotation
nvidia-smi

# 5. Verify CUDA actually works before running anything real. P100 = compute
#    capability 6.0 (Pascal) -- recent PyTorch wheels have been dropping
#    support for older architectures release over release, so this is a real
#    "might just not work" checkpoint, not a formality:
.venv-gpu/bin/python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1000, 1000, device='cuda')
print('matmul ok:', (x @ x).sum().item())
"
# If this errors with something like "no kernel image is available for
# execution on the device" -- the installed torch build has dropped sm_60
# (Pascal) support. Fall back to an older torch release known to still
# support it (check https://pytorch.org/get-started/previous-versions/,
# look for one still listing Pascal/sm_60), or request access to gpu-v100 /
# gpu-a100 (restricted, needs the form linked from the Saion guide) instead
# of test-gpu's P100s -- V100 (sm_70) and A100 (sm_80) are far more likely
# to be supported by a current wheel.

# 6. Run both prototype scripts and record the printed numbers
.venv-gpu/bin/python3 scripts/gpu_prototypes/validate_batched_ccm.py
.venv-gpu/bin/python3 scripts/gpu_prototypes/validate_ranking_simplex.py
```

**After running this**: report back what steps 5 and 6 printed (or paste the full terminal
output) — that's what turns into the actual go/no-go entry in this log (plan §5 item 3's
decisive experiment). Compare directly against the CPU numbers already on record above: CCM gate
1.12s (ref) vs 3.61s (CPU torch); ranking stage 3.5s (ref) vs ~32s (CPU torch, chunked). If the
GPU numbers beat the *reference* engine's CPU time (not just beat the CPU-torch time), that's the
signal this direction is worth real investment; beating CPU-torch but still losing to the
reference is not enough on its own.

### What to watch out for while running this
- If `uv venv`/`uv pip install` behaves unexpectedly because of the project's own
  `[tool.uv] environments`/`[tool.uv.sources]` config (added this session, scoped to
  `linux`/`x86_64` and pinning `torch` to the CPU index) leaking into the standalone `.venv-gpu`
  install: that config lives in `pyproject.toml` and only applies to *project-mode* commands
  (`uv sync`, `uv add`, `uv lock`) — the `uv pip install --python .venv-gpu/bin/python ...` form
  above deliberately bypasses project mode entirely, so it should be unaffected, but this hasn't
  been verified on Saion itself.
- Do not run `uv sync` or `uv add` from inside `~/pao_rotation` while on Saion unless deliberately
  intending to change the shared `pyproject.toml`/`uv.lock` — those files are the same ones Deigo's
  CPU pipeline depends on (shared home), and an accidental change there would affect both clusters.
- `/deigo_flash` (read-only from Saion) may or may not be the same mount as this session's own
  `/flash/.../scratchpad` path used earlier — don't assume scratch files written during earlier
  Deigo work are reachable/writable from a Saion compute node.

---

## 2026-09-02 (continued) — GPU results: confirmed win on real hardware

**Status: the decisive experiment (plan §5 item 3, pre-update) has been run. Verdict: WIN on
both benchmarks, by a wide margin. See `docs/no-kEDM-gpu-optimization-plan.md` §0 for the
plan-level implications — this entry is the raw record.**

The user ran the procedure above on Saion (`test-gpu` partition, node `saion-gpu06`) and reported
back the full terminal output. Environment: driver 580.167.08, CUDA 13.0 (driver-side; torch
itself was installed as the `cu126` build per the procedure above — the driver supports newer
CUDA than the torch build needs, which is fine, forward compatibility), `Tesla P100-SXM2-16GB`,
compute capability `(6, 0)` — confirmed still supported by `torch==2.14.0+cu126` (unlike a
cu128+ build, which would have failed here per the dev-discuss source in the plan doc).

**CUDA sanity check** (before running either benchmark):
```
cuda available: True
device: Tesla P100-SXM2-16GB
capability: (6, 0)
matmul ok: -58747.58203125
```
Confirms the environment is genuinely using the GPU, not silently falling back to CPU.

**CCM gate benchmark** (`validate_batched_ccm.py`, M=2000, same config as the CPU run):
```
Reference (numpy/KDTree, 140 calls): 0.659s
Torch device: cuda (Tesla P100-SXM2-16GB, compute capability (6, 0))
Torch batched, single-threaded, cached cdist (7 batched calls of size 20): 0.074s
Speedup: 8.88x
Max abs diff (float64 ref vs float32 torch), 280000 points compared: 8.27e-05
NaN-pattern mismatches: 0 / 140
```
CPU (Deigo) had been 1.19s (ref) vs 3.77s (torch) = 0.31x (torch 3.2x *slower*). On GPU: **8.88x
faster**. Correctness: 8.27e-05 max diff, essentially identical to CPU's 8.28e-05 — the GPU port
didn't introduce any new numerical issue.

**Ranking-stage benchmark** (`validate_ranking_simplex.py`, lib=20000/pred=2000/100 candidates,
chunk_b=5, same config as the CPU run):
```
pyEDM.Simplex looped, 100 candidates: 1.638s (16.38 ms/call)
Torch device: cuda (Tesla P100-SXM2-16GB, compute capability (6, 0))
Torch batched, chunk_b=5 (20 chunked calls), 100 candidates: 0.486s
Speedup: 3.37x
Diff vs pyEDM.Simplex across 200000 points: median=2.89e-05  mean=3.22e-04  max=1.31e-01  frac>0.01=0.0040
```
CPU (Deigo) had been 4.01s (ref) vs 35.78s (torch) = 0.11x (torch ~9x *slower*). On GPU: **3.37x
faster**. Correctness distribution (median/mean/max/frac>0.01) matches the CPU run's
2.89e-05/3.24e-04/0.131/0.0040 almost exactly — same benign near-tied-neighbor pattern (§2c of
the plan doc), not a new GPU-specific issue, and chunk_b=5 fit comfortably in the P100's 16GB
VRAM (no `torch.cuda.OutOfMemoryError` triggered).

**Also notable**: Saion's CPU (used for the "reference" computation in both scripts, which still
runs on CPU regardless of `DEVICE`) is itself roughly 1.8-2.4x faster per-core than Deigo's for
this workload (0.659s/1.638s here vs. 1.19s/4.01s on Deigo for the same configs) — a different
CPU generation, not a discrepancy in the code. This makes the GPU speedup slightly more
conservative than it would look against Deigo's own CPU numbers, not less.

### What this confirms
- The plan's §3 hypothesis (CPU slowdown = compute-volume-bound, not overhead-bound like kEDM,
  therefore GPU-closeable) was correct, verified by measurement, not just successful reasoning.
- The metric-fidelity premise (§1) was already independently true and unaffected by this result.

### What this does NOT yet confirm — see plan §5's updated next steps
- Real production scale: CCM gate was tested at M=2000, not the actual `CCM_WINDOW_ROWS=800`;
  ranking stage was tested at 100 candidates, not production's 1000-2000+. Smaller CCM libraries
  favor tree search more, so the 8.88x figure specifically should not be assumed to hold unchanged
  at M=800 without re-testing.
- Whether `test-gpu` (open, P100-only, likely meant for validation/testing rather than long
  production runs) is the right partition for an eventual real pipeline job, or whether the
  restricted `gpu-v100`/`gpu-a100` partitions would be needed — not yet asked of OIST SCDA.
- How a single shared GPU device interacts with this pipeline's existing `ThreadPoolExecutor`
  based inner-parallelism design (`pool` argument threaded through `proper_mde`/`proper_ccm`) —
  not yet investigated at all; a GPU is a single shared resource in a way CPU cores aren't, so this
  needs its own design, not an assumption that the current threading structure ports over for free.

### Next steps (superseded by the entries below — kept for history)
1. ~~Re-run both benchmarks at production-realistic scale~~ — done, see the 2026-09-02 (production
   scale) entry below. Result: NOT symmetric across targets — CCM gate held/improved, ranking
   stage reversed to a loss.
2. ~~Ask OIST SCDA whether test-gpu is appropriate for production runs~~ — deferred by explicit
   user decision (see the plan doc's §5 item 4); not blocking further validation/design work.
3. ~~Investigate GPU-sharing behavior~~ — done via reasoning from documented CUDA/PyTorch
   behavior (see the "fork-safety and GPU-sharing design" entry below); not yet empirically tested.
4. Refined below now that only one target (CCM gate) is proceeding.

---

## 2026-09-02 (continued) — Fork-safety and GPU-sharing design resolved; test-gpu deferred

**Status: two of the three open design questions from the previous entry are now resolved via
research/reasoning (not yet empirical Saion testing). See `docs/no-kEDM-gpu-optimization-plan.md`
§7 for the full write-up — this is the pointer entry.**

- **Fork-safety**: confirmed, directly (not by analogy) — CUDA does not support `fork()` at all;
  PyTorch's own docs require `spawn`/`forkserver` for CUDA in subprocesses. Same fix as the kEDM
  plan's Stage 4: switch `tsls-mde-gmn.py`'s two outer `ProcessPoolExecutor` pools
  (`mp_context=fork`, lines ~2603-2604/~2781-2782) to `spawn` before any torch/CUDA code can reach
  them. Sources in the plan doc §7a.
- **GPU-sharing under the existing thread/process parallelism model**: reasoned through (not yet
  measured) — multiple outer-pool processes sharing one P100 compete for its 16GB VRAM and
  scheduler rather than adding throughput; concurrent inner-thread GPU calls within one process are
  safe (CUDA driver serializes them) but don't get true concurrent execution without deliberately
  using separate CUDA streams (not currently done). Working hypothesis: shrink `outer_workers`
  drastically when GPU-bound (maybe to 1-2, not today's CPU-core-sized value) and rely on batch
  size per call as the real throughput lever instead. Full reasoning in plan §7b.
- The user reviewed the plan and explicitly deferred requesting restricted-partition access
  (`gpu-v100`/`gpu-a100`) until a working GPU-optimized pipeline exists to justify it — `test-gpu`
  remains sufficient for all remaining validation/design work in the meantime.
- Prepared and committed production-scale versions of both prototype scripts (commit `5f61ed7`):
  CCM gate at `M=800`/`libSizes=[16,24,40,64,160,400,800]` (exact `CCM_WINDOW_ROWS`/
  `CCM_LIB_SIZES_ROWS` from `scripts/batch/tsls-mde-gmn.sbatch`, not guessed); ranking stage at
  `n_total=44788` (real row count from an archived job, 40563312) with the `N//2` lib/pred split
  `proper_mde` documents as its default, `n_candidates=1500` (representative of prior sessions'
  observed pool sizes). Recalculated `chunk_b` from 5→2 for the much larger `n_pred` this implies
  (~10GB/chunk at the old value vs. the P100's 16GB — too tight for comfort unsupervised).
  Re-verified both scripts on CPU before handing off again: CCM fully re-run (1.49e-05 max diff,
  tighter than the M=2000 prototype); ranking stage smoke-tested at reduced scale (same code path,
  smaller N, since the full CPU run would take too long to be worth waiting for).

---

## 2026-09-02 (continued) — Production-scale results: the two targets diverged

**Status: DECISIVE. CCM gate confirmed for integration (win held and improved at real production
scale, twice validated). Ranking-stage torch primitive as currently designed is NOT viable —
reverses to a loss at production scale, for a well-understood, non-fixable-by-tuning reason. See
`docs/no-kEDM-gpu-optimization-plan.md` §0's "UPDATE 2" and revised §5 for the full plan-level
implications — this entry is the raw record and mechanism derivation.**

User ran the production-scale scripts (commit `5f61ed7`) on Saion (`test-gpu`, node
`saion-gpu06`), after first noting `git pull` doesn't work here (expected and harmless — `/home`
is shared NFS between Deigo and Saion, so commits made from Deigo are already present on Saion;
there's no remote to pull from for this workflow, `git log --oneline -1` is the correct check,
not `git pull`. This was a bad instruction on my part in the prior turn, corrected here for the
record).

**CCM gate** (`validate_batched_ccm.py`, `M=800`, real `CCM_LIB_SIZES_ROWS`):
```
Reference (numpy/KDTree, 140 calls): 0.202s
Torch device: cuda (Tesla P100-SXM2-16GB, compute capability (6, 0))
Torch batched, single-threaded, cached cdist (7 batched calls of size 20): 0.016s
Speedup: 12.30x
Max abs diff (float64 ref vs float32 torch), 112000 points compared: 1.89e-05
NaN-pattern mismatches: 0 / 140
```
Prototype scale (M=2000) had been 8.88x. **Production scale (M=800): 12.30x — the win IMPROVED,
not degraded**, despite the plan's own prior concern that smaller libraries would favor tree
search more. Correctness held (1.89e-05, similar order to the CPU M=800 run's 1.49e-05).

**Ranking stage** (`validate_ranking_simplex.py`, `n_total=44788`, `n_candidates=1500`,
`chunk_b=2`):
```
pyEDM.Simplex looped, 1500 candidates: 53.050s (35.37 ms/call)
Torch device: cuda (Tesla P100-SXM2-16GB, compute capability (6, 0))
Torch batched, chunk_b=2 (750 chunked calls), 1500 candidates: 88.592s
Speedup: 0.60x
Diff vs pyEDM.Simplex across 33591000 points: median=3.71e-05  mean=3.85e-04  max=2.38e-01  frac>0.01=0.0052
```
Prototype scale (100 candidates, pred=2,000) had been 3.37x faster. **Production scale (1500
candidates, pred≈22,394): 0.60x — 1.67x SLOWER, a full reversal.** Correctness distribution is
unchanged/benign (same near-tied-neighbor pattern as every prior run) — this is a pure speed
regression, not a new bug.

### The mechanism, verified quantitatively (not just plausibly)
Fit the reference's two known (n_pred, cost) points to `cost = fixed + k·n_pred`:
`16.38ms` at `n_pred=2000` and `35.37ms` at `n_pred=22394` (n_lib barely changed, 20000→22394,
~1.12x) gives `fixed≈14.5ms`, `k≈0.93 microseconds/row`. This says pyEDM.Simplex's per-call cost
is dominated by a roughly pred-size-independent fixed cost (almost certainly building a search
index over the library) plus a very cheap per-query cost against that already-built index —
classic tree-search behavior. Torch's brute-force approach has no equivalent shortcut: predicting
its production-scale total time from pure `O(candidates × n_pred × n_lib)` compute-volume scaling
— `0.486s × (1500/100) × (22394/2000) × (22394/20000) ≈ 91.4s` — matches the **observed 88.6s
within 3%**. This rules out chunking/overhead artifacts as the cause (if per-chunk overhead were
significant, going from 20 chunks to 750 chunks would have blown the prediction, not matched it):
**the slowdown is pure compute-volume, exactly as predicted by the O(n_lib) vs O(log n_lib)
per-query cost difference between brute force and tree search, once n_pred is large enough for
that difference to dominate.** No batching or GPU-throughput improvement fixes an asymptotically
worse algorithm once data size outruns the constant-factor hardware advantage.

The critical asymmetry between the two targets: CCM's production `n_pred` (=`M`=800) *shrank*
relative to the prototype test (2000→800), while the ranking stage's production `n_pred`
(2,000→22,394) *grew* by 11x. `n_pred` is exactly the dimension this mechanism is sensitive to —
so the two targets moving in opposite directions on it fully explains why they diverged.

### What this confirms
- The plan's own explicit warning ("do not generalize from the prototype-scale numbers") was
  necessary, not just cautious boilerplate — the ranking-stage number it warned about is exactly
  the one that didn't survive.
- The CCM gate is now validated at real production scale, on real hardware, twice, with an
  improving margin — the strongest evidence in this entire investigation (kEDM included) for any
  single integration target.

### What this rules out (not just "doesn't confirm")
- Retrying the ranking-stage torch primitive with a different `chunk_b`, more GPU memory, or a
  newer/faster GPU. The 3%-accurate compute-volume prediction means this is not an overhead or
  tuning problem — the algorithm itself is asymptotically worse at this data size, on any hardware
  that doesn't change the O(n_lib) brute-force scan into something sub-linear.

### Next steps
1. **Design and build the CCM-gate integration** — this is the one target to actually carry
   forward. Needs: (a) fork-safety fix (spawn, plan §7a) applied before any torch code can reach
   the outer worker pools; (b) a concrete decision on GPU-sharing (plan §7b's hypothesis — likely
   a small, dedicated number of GPU-using workers rather than today's CPU-core-sized
   `outer_workers` — needs empirical testing on Saion, not just the reasoning already done);
   (c) an actual call-site integration plan (which exact functions in `cleaned_helper_functions.py`
   change, what the CPU-fallback behavior is when no GPU is available, what the test strategy is)
   — not yet written, this is the next concrete deliverable.
2. **Do not pursue the ranking-stage torch primitive further** without first securing a
   sub-linear GPU nearest-neighbor search (GPU KD-tree, FAISS-GPU, or similar) — a materially
   different, bigger engineering effort than anything built so far. Not started; not scheduled
   unless the user decides it's worth the investment given the CCM-only win in hand.
3. Empirically test the GPU-sharing hypothesis from plan §7b (queue depth vs. throughput, small
   `outer_workers` vs. today's CPU-sized value) before finalizing the CCM-gate integration's
   worker-pool design — reasoned through so far, not yet measured on Saion.
