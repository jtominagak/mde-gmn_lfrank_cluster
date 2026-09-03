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

---

## 2026-09-03 — Fork-safety implemented; GPU-sharing experiment written; integration spec drafted

**Status: all three of the previous entry's next steps are addressed. One (the GPU-sharing
measurement) is written but NOT yet run — it needs Saion GPU access, which this session does not
have. Integration coding (plan §8.3 onward) has not started.**

### 1. Fork-safety fix — DONE and tested
Implemented as a *conditional* switch rather than a blanket one, because measurement showed spawn
is expensive here: **a spawn worker re-imports this module in 18.3s** (pandas/numpy/matplotlib/
pyEDM/gmn), and the outer pools are recreated per BFS wave, so CPU-only runs would pay that
repeatedly for a GPU path they never use.

- `load_config()`: new `GPU_ENABLED` (`TSLS_MDE_GMN_GPU_ENABLED`, default False) and derived
  `MP_START_METHOD` (`'spawn'` when GPU on, else `'fork'`).
- `build_network()` / `build_nested_mde_network()`: new `mp_start_method='fork'` keyword
  (defaulted, so existing callers are unaffected), threaded from `main()`; both outer
  `ProcessPoolExecutor` pools now use it instead of a hardcoded `'fork'`.
- Verified before changing anything: gmn monkeypatching is at module scope (lines 600/809 —
  re-executes correctly on re-import in a spawn worker), there is a proper `__main__` guard
  (line ~5686), and there are no `global` statements. Without these, spawn would have broken.
- New test `tests/test_gpu_fork_safety.py` — all pass: default stays fork; flag flips to spawn;
  `mp_start_method` genuinely reaches both pools (asserts the hardcoded `'fork'` is gone and both
  pools reference the parameter); and a real spawn-context `ProcessPoolExecutor` runs while
  round-tripping the argument *types* the real `submit()` passes (DataFrame/Path/dict/list/
  ndarray). That last check is the one fork would silently let a regression through on.
- Existing suite re-run green after the change: `test_r12_stage_serialization`,
  `test_r13_shared_depth_budget`, `test_metric_routing_depth_ge1`.

### 2. GPU-sharing experiment — written, not yet run
`scripts/gpu_prototypes/measure_gpu_sharing.py`. States plan §7b's claims as four falsifiable
hypotheses and prints directly interpretable results, using the CCM-gate primitive at production
scale (M=800):
- **H1** batch size is the real throughput lever (expect per-sweep time to drop as batch grows).
- **H2** more *processes* sharing one GPU = contention, not throughput (fixed total work split
  across 1/2/4 spawn processes; expect throughput ~flat or worse, unlike CPU cores).
- **H3** more *threads* in one process likewise don't add GPU parallelism (kernels on the default
  stream serialize).
- **H4** CUDA cannot fork — expected to FAIL loudly; that failure is the on-hardware confirmation
  of §7a and of why `GPU_ENABLED=1` switches to spawn. Timeout-guarded (60s) because undefined
  behaviour can hang rather than raise, and the script explicitly warns against relaxing the spawn
  requirement even if the fork probe unexpectedly succeeds.
- Smoke-tested end-to-end on Deigo CPU (runs, all sections execute, H4 skips cleanly without a
  GPU). The CPU numbers it printed are meaningless for the hypotheses — the script says so itself
  in that case.

### 3. Code-level integration spec — written (plan §8)
Scope deliberately narrowed to the **CCM gate only**; the ranking stage is explicitly excluded per
the production-scale result. Key decisions recorded there:
- New isolated module `scripts/src/gpu_edm.py` (keeps the torch import optional; must import
  cleanly on a GPU-less machine), porting the already-validated batched primitive unchanged —
  including the two hard-won details (first-`k`-valid compaction before `d_min`, and
  `set_num_threads(1)`).
- Integration point is `proper_ccm`'s existing task grid, which is already ideally structured:
  all `rng.choice` draws happen up-front and sequentially, so the torch path reuses the *same*
  draws and reproducibility per `seed` is preserved; and `_score()` (with `f1`/`gamma`) stays on
  CPU untouched, which is exactly what makes this viable where kEDM's `ccm()` wasn't.
- Mandatory transparent CPU fallback: `use_gpu=True` without a GPU must fall back, not raise, so
  the same code runs on Deigo and Saion with only the flag differing.
- Test strategy centred on a `proper_ccm`-level equivalence test (GPU vs CPU, same seed, matching
  outputs) that skips cleanly without a GPU — the check that would catch a silent scientific
  regression rather than just a broken primitive.

### Next steps
1. Run `scripts/gpu_prototypes/measure_gpu_sharing.py` on Saion (`test-gpu`, same venv/procedure
   as before). Its H2 result is a prerequisite for choosing `outer_workers` under
   `GPU_ENABLED=1`; H4 confirms the fork hazard on this actual hardware.
2. Then implement plan §8.3-§8.4 (`gpu_edm.py`, `proper_ccm(use_gpu=...)`, flag threading) with
   the §8.5 tests.
3. Then the §8.6 step-5 end-to-end small-scale A/B on Saion (`GPU_ENABLED=1` vs `0`, same config)
   before any long production run.

---

## 2026-09-03 — GPU-sharing measured: two hypotheses falsified, integration simplified

**Status: `measure_gpu_sharing.py` run on Saion (`saion-gpu05`, P100). Two of four hypotheses
falsified, one confirmed, one inconclusive due to a test-design flaw. Net effect: the integration
spec got SIMPLER, and no worker-sizing change is needed. Plan §7c has the analysis; this is the
raw record. Next: build the integration (plan §8.6 step 3) — not yet started.**

### Raw output
```
device: cuda (Tesla P100-SXM2-16GB, capability (6, 0), 17.1 GB)

--- H1: batch size as the throughput lever ---
  batch= 1  calls= 64  sweeps= 64  total=0.947s  per-sweep=14.80ms
  batch= 2  calls= 32  sweeps= 64  total=0.875s  per-sweep=13.67ms
  batch= 4  calls= 16  sweeps= 64  total=0.858s  per-sweep=13.41ms
  batch= 8  calls=  8  sweeps= 64  total=0.849s  per-sweep=13.27ms
  batch=16  calls=  4  sweeps= 64  total=0.844s  per-sweep=13.19ms

--- H3: thread count vs. throughput (fixed total work, one process) ---
  threads=1  sweeps/thread=32  total_sweeps=32  wall=0.47s  throughput=67.67 sweeps/s
  threads=2  sweeps/thread=16  total_sweeps=32  wall=0.47s  throughput=68.32 sweeps/s
  threads=4  sweeps/thread=8  total_sweeps=32  wall=0.47s  throughput=68.26 sweeps/s
  threads=8  sweeps/thread=4  total_sweeps=32  wall=0.47s  throughput=67.93 sweeps/s

--- H2: process count vs. throughput (fixed total work) ---
  procs=1  sweeps/proc=32  total_sweeps=32  wall=28.28s  throughput=1.13 sweeps/s
  procs=2  sweeps/proc=16  total_sweeps=32  wall=10.75s  throughput=2.98 sweeps/s
  procs=4  sweeps/proc=8  total_sweeps=32  wall=9.81s  throughput=3.26 sweeps/s

--- H4: CUDA + fork (expected to FAIL -- that is the confirmation) ---
  forked child failed as expected: RuntimeError: Cannot re-initialize CUDA in forked subprocess.
  To use CUDA with multiprocessing, you must use the 'spawn' start method

peak GPU memory this run: 1.27 GB
```

### H4 — CONFIRMED on hardware
The exact predicted error. This independently validates the §8.1 fork→spawn change against the
real GPU, rather than resting on documentation alone.

### H3 — CONFIRMED
Throughput dead flat (67.7 → 68.3 → 68.3 → 67.9 sweeps/s) across 1→8 threads. Extra inner threads
add no GPU parallelism — as reasoned. Usefully, they don't *degrade* it either, so the existing
inner `ThreadPoolExecutor` needs no redesign.

### H1 — FALSIFIED (my §7b reasoning was wrong)
I had claimed batch size was "the real throughput lever." It bought **11%** across a 16x batch
increase. The correct reading: total FLOPs don't change with chunking, so batching only helps when
per-call overhead is significant relative to per-call work — at M=800 the workload is already
compute-bound, so consolidating 448 kernel launches into 28 changed almost nothing. Recorded
explicitly because it's the kind of plausible-sounding claim that would otherwise get
reintroduced later by first-principles reasoning.

### H2 — INCONCLUSIVE (my test design was flawed, not just the hypothesis)
The apparent 3x throughput gain from 1→4 processes is not a real scaling result. H3 establishes
one process does ~68 sweeps/s, so H2's 32 sweeps is **~0.47s of actual compute** against wall
times of 9.8-28.3s — meaning H2 measured spawn startup (torch import + CUDA context), not GPU
contention. The monotonic improvement is also consistent with page-cache warming, since `procs=1`
ran first and cold. A valid contention test would need per-worker GPU work large enough to
dominate startup; deferred to the real end-to-end A/B instead of rebuilding the microbenchmark.

**Important calibration note**: the 10-28s spawn startup looks alarming against a 0.47s benchmark
workload, but production per-node `run_mde_for_target` runtimes are minutes to many hours (job
40582445's `head_position_x`: 49,238s for a single node) — so it's 0.02-0.3% overhead in reality.
Do not "fix" spawn startup on the strength of this benchmark; it isn't a real problem.

### What this changed in the plan
Three pieces of planned complexity were dropped (plan §8.4b), and no sizing change is needed:
- **No candidate-level batching tier** (H1) — the validated primitive's existing per-libSize
  batching is sufficient.
- **No threading redesign** (H3) — existing inner `ThreadPoolExecutor` stays as-is.
- **No VRAM tiling design** (peak 1.27 GB of 17.1 GB) — the `chunk_b`-style tiling the
  ranking-stage prototype needed is irrelevant here. Retires plan §5 item 6.
- **No `outer_workers` shrink** — §7b's hypothesis that it would need to shrink drastically did
  not survive measurement; keep current sizing, evaluate contention in the §8.6 step-5 A/B.

### Next steps (superseded by the entry below — kept for history)
1. ~~Build `scripts/src/gpu_edm.py` + `proper_ccm(use_gpu=...)` + the equivalence test~~ — done,
   see the 2026-09-03 entry below.
2. ~~Thread `GPU_ENABLED` through the call chain~~ — done, see the entry below.
3. End-to-end small-scale A/B on Saion (`GPU_ENABLED=1` vs `0`, same config), which also serves
   as the real contention measurement H2 couldn't provide. Still not started — this session has
   no GPU/no Saion access (see that entry's own note).

---

## 2026-09-03 (continued) — `gpu_edm.py` built, `proper_ccm(use_gpu=...)` implemented, wired through, tested

**Status: plan §8.6 steps 3-4 done and committed. Step 5 (the real Saion A/B) is next and is the
first time any of this touches actual GPU hardware — everything below was built and verified on
Deigo (no GPU), which bounds how much confidence it earns on its own. See "What this does NOT yet
confirm" below before treating this as more validated than it is.**

### What was done

1. **`scripts/src/gpu_edm.py`** (new file). `is_available()` (never raises, False whenever torch
   itself fails to import — wrapped `import torch` in `try/except`, `torch = None` sentinel) and
   `batched_crossmap_predict(embed, pred_vals, lib_idx_batch, k, exclusionRadius, time_indices,
   full_dist=None, device=None)`, ported from
   `scripts/gpu_prototypes/validate_batched_ccm.py::batched_crossmap_predict_torch` with the
   algorithm UNCHANGED — same first-k-valid compaction (`torch.argsort`+`torch.gather`) before
   `d_min`, same `torch.set_num_threads(1)` (called once at module import, guarded against
   `torch is None`). One deliberate API deviation from plan §8.3's one-line signature sketch:
   returns `(preds, full_dist)` instead of just `preds` — `full_dist` is an opaque cached (M, M)
   torch tensor the caller can pass back into the next call against the SAME `embed` to skip
   recomputing the O(M²) pairwise distance matrix, which is otherwise the single most expensive
   step and doesn't depend on `lib_idx_batch` at all. This mirrors the validated prototype's own
   "cached cdist" usage (its main() computes `full_dist` once and reuses it across every libSize
   in the sweep) rather than dropping that optimization — recorded here explicitly since the plan
   text's signature line didn't spell out the tuple return, so a future reader diffing against the
   plan should know this was intentional, not a drift.

2. **`proper_ccm(..., use_gpu=False)`** (`cleaned_helper_functions.py`). `use_gpu=False` (the
   default) never imports `gpu_edm` at all — the existing `mapper(_run_one_cell, tasks)` path is
   reached completely unchanged. When `use_gpu=True`, checks `gpu_edm.is_available()`; if False
   (no GPU), falls through to that exact same unchanged path — no exception, no behavior
   difference from `use_gpu=False`. If True, groups the already-drawn `tasks` list (built earlier,
   completely unmodified — the `rng.choice` draw loop is untouched by any of this) by libSize
   index `i`: every task at a given `i` has the same fwd library length
   (`min(libSizes[i], M_fwd)`) and same rev library length (`min(libSizes[i], M_rev)`) across
   every sample `t`, which is exactly the `(B, L)` shape `batched_crossmap_predict` needs. Issues
   one batched call per `(direction, i)` group (reusing a separately-cached `full_dist` per
   direction — fwd and rev use different embeddings, never shared), then scores each resulting row
   through the SAME unchanged `_score()` calls the CPU path uses. Verified this reuses the exact
   same `rng`-drawn library indices regardless of `use_gpu` (see test 3 below) and that
   `use_gpu=False` is byte-for-byte identical to before this change (verified directly — compared
   against a no-`use_gpu`-kwarg call and a `pool=ThreadPoolExecutor(...)` call, both
   `np.array_equal`).

3. **`tests/test_gpu_ccm_equivalence.py`** (new). Three tests, matching plan §8.5 items 1-3
   exactly:
   - Equivalence (`use_gpu=False` vs `True`, same seed) — skips clean on Deigo (no CUDA), the only
     one of the three that actually needs GPU hardware to mean anything.
   - Fallback (`use_gpu=True` with no GPU == `use_gpu=False`, byte-identical, no exception) — runs
     and PASSES on Deigo; skips clean instead on a machine that DOES have a GPU (that combination
     is what the equivalence test covers instead — the two tests are complementary, gated on
     opposite `is_available()` outcomes).
   - Reproducibility — stronger than comparing final scores, per the task instructions: patches
     `numpy.random.default_rng` (an ordinary Python function) to return a recording wrapper that
     logs every `.choice()` call, then asserts the recorded draw SEQUENCE is identical between
     `use_gpu=False` and `use_gpu=True`. (Note: `numpy.random.Generator.choice` itself can't be
     monkeypatched directly — it's a Cython extension type method, raises "cannot set 'choice'
     attribute of immutable type" — hence patching `default_rng`'s return value instead.) Runs and
     PASSES on Deigo; doesn't need a GPU at all, since the draw loop runs unconditionally before
     the `use_gpu` branch.
   All three use `target_measurement='correlation'`/`columns_measurement='correlation'` rather
   than `'f1'`/`'gamma'` deliberately — those event-timing metrics threshold a TSLS-shaped
   (sawtooth) signal, which the test's synthetic sine-wave data isn't, and thresholding is exactly
   where a tiny float32-vs-float64 prediction difference could flip a boolean outcome for reasons
   unrelated to whether the GPU port itself is correct. `'correlation'` is a smooth function of
   the raw predictions, so it isolates the thing actually under test.
   Separately (not committed — a throwaway scratchpad script, deleted after use): directly compared
   `gpu_edm.batched_crossmap_predict(..., device='cpu')` against `_crossmap_predict` at production
   scale (M=800, same libSizes/sample as the validated prototype) to catch any transcription bug
   in the port independent of needing real GPU hardware. Result: max abs diff 1.49e-05 across
   112,000 compared points, 0/140 NaN-pattern mismatches — matching the original prototype's own
   CPU-production-scale number (progress log, 2026-09-02 entry) almost exactly, which is strong
   evidence the port (numpy/torch boundary, `full_dist` tuple-threading, `k`→`k_eff` rename) didn't
   introduce a bug, independent of anything CUDA-specific.

4. **Call-chain wiring** (plan §8.4): `use_gpu=False` added as a defaulted keyword at every hop —
   `main()` → `build_network()` → `build_nested_mde_network()` → `run_mde_for_target()` →
   `proper_mde()` → `proper_ccm()` — mirroring exactly how `mp_start_method` was threaded in §8.1
   (same call sites, same "defaulted so no existing caller changes behaviour" pattern).
   `main()` passes `cfg['GPU_ENABLED']` at both of its two call sites (the depth-0 seed
   `run_mde_for_target` call and the `build_network` call).
   **Scope decision, not in the original plan text**: `multi_proper_mde`/`multi_proper_ccm` (the
   `GMN_MODE='combined'` multi-target path, driven by `run_multi_mde_for_targets` — a genuinely
   separate function from `run_mde_for_target`, not a branch inside it, confirmed by reading both)
   were deliberately left untouched. `multi_proper_ccm` has a materially different sweep structure
   from `proper_ccm` — per-target draws interleaved inside the same `t`/`L` loop (multiple
   `_crossmap_predict` calls per `t` for the fwd direction, one per target) rather than one flat
   pre-drawn task list, and it doesn't even accept a `pool` parameter today. Neither plan §8.2 nor
   §8.3 specified a batched design for that structure, and the task instructions' own deliverable
   list named `proper_ccm` specifically, not `multi_proper_ccm`. `GPU_ENABLED=1` therefore
   currently only accelerates `GMN_MODE='separate'`/single-target runs. Recorded here so this
   doesn't read as an oversight if `GMN_MODE='combined'` runs are GPU-enabled later without effect
   — it would need its own scoped design, not an assumption that `proper_ccm`'s grouping strategy
   ports over unchanged.

5. **Existing suite**: full `tests/test_*.py` (18 files, including the new one) run individually,
   all PASS — in particular the four named as must-stay-green: `test_gpu_fork_safety`,
   `test_r12_stage_serialization`, `test_r13_shared_depth_budget`, `test_metric_routing_depth_ge1`.
   Two shared test fakes needed updating to accept the new `use_gpu` kwarg
   (`tests/_pipeline_test_utils.py`'s `make_fake_run_mde_for_target_tsls`, and
   `tests/test_metric_routing_depth_ge1.py`'s own local `_make_fake_run_mde_for_target`) — both
   stand in for the real `run_mde_for_target()` and would otherwise raise
   `TypeError: unexpected keyword argument 'use_gpu'` the moment `build_nested_mde_network()`'s
   `executor.submit(run_mde_for_target, ..., use_gpu=use_gpu)` calls reached them. Caught by
   actually running the suite, not by inspection — worth remembering next time any
   `run_mde_for_target` signature change lands, since these fakes have no automatic sync with the
   real signature.

### What this does NOT yet confirm
- Actual GPU numerics beyond the CPU-device port-validation check in item 3 above. That check
  proves the algorithm transcription is faithful; it says nothing about anything CUDA-specific
  (kernel behavior, float32 rounding on real hardware, `torch.cuda.synchronize()` correctness).
  Only a real Saion run (§8.6 step 5, next) tests that.
- Multi-worker GPU contention under `outer_workers`-sized concurrent spawn processes each holding
  their own CUDA context and calling into this new code path — still exactly the open question
  §7c's H2 couldn't answer, deferred to the same step-5 A/B as before.
- Anything about `GMN_MODE='combined'` runs (see the scope decision in item 4 above) — those are
  simply not GPU-accelerated by this work at all yet.

### Next steps (superseded by the entry below for the `GMN_MODE='combined'` gap — kept for history)
1. ~~Run the plan §8.6 step-5 A/B on Saion~~ — still not run (no GPU/no Saion access this session),
   but the scope it needs to cover grew: see the entry below, then the updated command block in the
   chat turn that entry corresponds to.

---

## 2026-09-03 (continued) — GPU path extended to `GMN_MODE='combined'` (plan section 9)

**Status: plan section 9 done and committed. `multi_proper_ccm(use_gpu=...)` now exists alongside
`proper_ccm(use_gpu=...)`, closing the gap the previous entry's item 4 scope note flagged —
`GMN_MODE='combined'` (the pipeline default) now gets GPU acceleration too. Same caveat as before:
built and verified entirely on Deigo (no GPU); the Saion A/B (next) is what actually proves this on
real hardware, for both `GMN_MODE` values now.**

### What was done

1. **`multi_proper_ccm(..., use_gpu=False)`** (`cleaned_helper_functions.py`). The real work here
   was structural, not algorithmic (section 9.1 was right that the underlying primitive applies
   unchanged) — three differences from `proper_ccm` drove it, matching plan section 9.2 exactly:
   - **Draws collected up front.** Unlike `proper_ccm` (already structured that way),
     `multi_proper_ccm`'s original code interleaved `rng_fwd.choice`/`rng_rev.choice` calls with
     `_crossmap_predict`/`_score` compute, one `(i, t)` cell at a time, with an explicit code
     comment about preserving consumption order/count. Rewrote as: one pass that draws every
     `(i, t[, target])` library index into `fwd_draws`/`rev_draws` dicts, in EXACTLY the original
     per-libSize `rng_fwd`/`rng_rev` Generator creation and consumption order — the same `for i:
     rng_fwd = default_rng(...); rng_rev = default_rng(...); for t: for tname: ...; then rev`
     structure, just recording instead of computing — followed by separate compute passes (CPU
     loop or GPU-batched) that only ever read from those already-drawn values.
   - **Fwd batches per `(libSize, target)` pair.** Each target has its own predicted values
     (`obs_fwd_dict[tname]`), and `batched_crossmap_predict` takes one `pred_vals` array per call
     (not one per batch row) — so unlike `proper_ccm`'s one-call-per-libSize, this is one call per
     `(i, tname)` pair, batching across `sample`. A single cached `(M_fwd, M_fwd)` distance matrix
     is still reused across every one of those calls (embed_fwd never changes across the whole
     function, regardless of `use_full_pool`). Scores are computed per target immediately after
     each batched call, then aggregated via `target_aggregate` (`mean`/`min`) in a final pass —
     aggregating scores, never predictions, per section 9.2 item 2. Moved the
     `target_aggregate not in ('mean', 'min')` check to the top of the function (was previously
     raised lazily inside the per-cell loop) — same error, now fail-fast instead of after
     potentially wasted compute; a strict improvement, not a behavior change for valid inputs.
   - **Rev batches only when `use_full_pool`.** `target_embed` is constant across every sample only
     when `target_select_E is None` (the full joint pool) — exactly the precondition the cached
     distance-matrix trick needs. When `target_select_E` is an int, `target_embed` is rebuilt from
     a freshly-drawn `col_idx` every `(i, t)`, so per section 9.2 item 3 this stays on the CPU
     `_crossmap_predict` path unconditionally (not yet measured as worth batching per distinct
     embedding — explicitly deferred, matching the "don't add complexity before measuring" lesson
     section 7c's H1 already established for a different case).
   - `multi_proper_ccm` still has no `pool` parameter (per the guardrail) — not added.

2. **Verification beyond the committed test suite** (two scratchpad-only scripts, deleted after
   use, not committed): (a) loaded the PRE-this-session committed version of
   `cleaned_helper_functions.py` (via `git show HEAD:...`) side by side with the edited working-tree
   version and called `multi_proper_ccm(..., use_gpu=False)` on both across five configurations
   (`use_full_pool` True/False, `n_extra` zero/nonzero, 1-3 targets, `mean`/`min` aggregate) —
   **all five produced byte-identical `c2t`/`t2c`/`libSizes`**, direct confirmation the
   draw-then-compute restructuring didn't change CPU-path numerics, not just an argument that it
   shouldn't have. (b) Monkeypatched `gpu_edm.is_available()` to return `True` and
   `gpu_edm.batched_crossmap_predict` to force `device='cpu'` (both late-imported inside
   `multi_proper_ccm`, so patching the `gpu_edm` module's attributes before calling it redirects
   those lookups) to actually exercise the new GPU-batched fwd/rev code paths on this GPU-less
   machine, compared against the `use_gpu=False` CPU path across four configurations — max diffs
   3.6e-06/3.0e-08 (`use_full_pool=True`, both directions batched) down to exactly 0.0 for the rev
   direction when `use_full_pool=False` (confirms rev genuinely never touches the GPU path in that
   case, as designed). This is the same class of check as `gpu_edm.py`'s own CPU-device port
   validation from the previous session (1.49e-05 max diff there) — independent evidence the new
   wiring is correct, without needing real CUDA hardware to get it.

3. **Call-chain wiring** (mirroring section 8.4's pattern exactly): `use_gpu=False` added to
   `multi_proper_mde()`, threaded to its `multi_proper_ccm()` call; added to
   `run_multi_mde_for_targets()` (`tsls-mde-gmn.py`), threaded to its `tsls.multi_proper_mde()`
   call; `main()`'s `GMN_MODE='combined'` depth-0 seed call site now passes
   `use_gpu=cfg['GPU_ENABLED']`. Confirmed by reading the call graph directly that
   `run_multi_mde_for_targets`/`multi_proper_mde`/`multi_proper_ccm` are reached ONLY from this one
   depth-0 seed call, not from `build_nested_mde_network()`'s Stage 1/Stage 2 BFS (that BFS always
   uses `run_mde_for_target`/`proper_mde`/`proper_ccm` regardless of `GMN_MODE` — confirmed by
   grepping every `gmn_mode`/`GMN_MODE` reference in `tsls-mde-gmn.py`, not assumed). So this
   session's work specifically closes the ONE call that was previously unaccelerated under
   `GMN_MODE='combined'`, not a whole separate code path running throughout the network build.

4. **`tests/test_gpu_multi_ccm_equivalence.py`** (new) — the `multi_proper_ccm` counterpart to
   `tests/test_gpu_ccm_equivalence.py`, same three checks: equivalence (skips clean on Deigo),
   fallback (runs and PASSES on Deigo), reproducibility (runs and PASSES on Deigo, recording both
   the `rng_fwd` and `rng_rev` draw streams via the same `numpy.random.default_rng`-patching
   technique as the `proper_ccm` test — `numpy.random.Generator.choice` itself can't be
   monkeypatched directly, Cython extension type). Equivalence and reproducibility both run against
   BOTH `use_full_pool` True and False (target_select_E=None vs an int), since those are genuinely
   different code paths inside the function (rev GPU-batched vs rev CPU-only) and both need
   hardware coverage — the equivalence test additionally asserts `t2c` is BYTE-IDENTICAL (not just
   close) between `use_gpu=False`/`True` in the `use_full_pool=False` case specifically, since rev
   never touches the GPU path there regardless of the flag.

5. **Existing suite**: full `tests/test_*.py` (20 files now, including the new one) run
   individually, all PASS — in particular every test named as must-stay-green:
   `test_gpu_ccm_equivalence`, `test_gpu_fork_safety`, `test_r12_stage_serialization`,
   `test_r13_shared_depth_budget`, `test_metric_routing_depth_ge1`. Also specifically re-checked
   `test_multi_mde_auto_embed_dimension.py` (calls `multi_proper_mde`/`multi_proper_ccm` directly,
   not via a fake) and `test_r19_mde_evaluate.py` (exercises `GMN_MODE='combined'` end-to-end via
   `run_mde_evaluate`) since those were the tests most likely to be sensitive to this session's
   restructuring — both PASS unchanged. No test fakes needed updating this time (unlike the
   previous entry's `run_mde_for_target` fakes) — nothing in `tests/` stands in for
   `run_multi_mde_for_targets`/`multi_proper_mde`/`multi_proper_ccm`.

### What this does NOT yet confirm
- Actual GPU numerics on real hardware — same caveat as the previous entry, now for this path too.
  The CPU-forced monkeypatch validation in item 2 above proves the wiring/grouping logic is
  correct; it says nothing CUDA-specific.
- Multi-worker GPU contention specifically for the `GMN_MODE='combined'` depth-0 seed call, which
  runs BEFORE `build_nested_mde_network()`'s own `ProcessPoolExecutor` waves start (single-process,
  no `pool` at all) — a different concurrency shape than the Stage 1/Stage 2 contention question
  §7c's H2 left open. Not yet reasoned through in detail; likely low-risk (it's one process, not
  many, sharing the GPU during this specific call) but not measured.

### Next steps
1. **Run the plan §8.6 step-5 A/B on Saion, now covering BOTH `GMN_MODE` values**:
   `GPU_ENABLED=1` vs `GPU_ENABLED=0`, small scale (`REGION_CHANNEL_CAP` low), once with
   `GMN_MODE=separate` and once with `GMN_MODE=combined` — diff each pair's MDE selections/edges,
   confirm both `tests/test_gpu_ccm_equivalence.py` and `tests/test_gpu_multi_ccm_equivalence.py`
   actually PASS (not skip) there. This session still has no GPU and no Saion SSH credentials, so
   this is handed to the user as a command block (see the chat turn this entry corresponds to).
