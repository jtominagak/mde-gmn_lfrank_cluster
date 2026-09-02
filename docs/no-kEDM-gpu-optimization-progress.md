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

### Next steps
1. **Ask the user where GPU hardware is actually reachable** — not yet asked, same open item as
   `docs/gpu-optimization-plan.md`'s own Stage 3. Nothing past this point is actionable without an
   answer.
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
