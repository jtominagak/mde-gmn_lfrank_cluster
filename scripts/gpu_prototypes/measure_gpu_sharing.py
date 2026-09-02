'''Empirically test the GPU-sharing hypotheses in
docs/no-kEDM-gpu-optimization-plan.md section 7b, using the CCM-gate primitive
(the one target validated for integration) at production scale (M=800).

The hypotheses, stated so the output can falsify them:

  H1 "batch size is the real lever": throughput should improve substantially as
     more independent sub-problems are batched into one call, up to a
     memory/diminishing-returns ceiling.

  H2 "more processes = contention, not throughput": splitting a FIXED total
     workload across N concurrent processes sharing one GPU should NOT scale
     down wall-clock the way it would on N CPU cores. If total time stays flat
     (or worsens) as N grows, H2 holds -- and the pipeline's CPU-core-sized
     `outer_workers` should shrink drastically when GPU-bound.

  H3 "more inner threads don't add GPU parallelism": same fixed workload split
     across N threads in ONE process (matching this pipeline's inner
     ThreadPoolExecutor) should likewise not scale down much, since kernels
     issued to the same default stream serialize.

  H4 "CUDA cannot fork" (plan section 7a): a forked child touching CUDA after
     the parent has initialised a context should fail. This one is expected to
     FAIL LOUDLY -- that failure is the confirmation, and it's why
     GPU_ENABLED=1 switches the pipeline's outer pools to spawn. Timeout-guarded
     because "undefined behaviour" can mean a hang rather than a clean error.

Run on a GPU node (see docs/no-kEDM-gpu-optimization-progress.md for the
srun line and venv):
    .venv-gpu/bin/python3 scripts/gpu_prototypes/measure_gpu_sharing.py
'''
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_batched_ccm import batched_crossmap_predict_torch  # noqa: E402

torch.set_num_threads(1)

# Production-scale CCM-gate parameters (same sourcing as validate_batched_ccm.py:
# CCM_WINDOW_ROWS=800, CCM_LIB_SIZES_ROWS, CCM_SAMPLE=20).
M = 800
E = 3
K = E + 1
EXCL = 2
LIB_SIZES = [16, 24, 40, 64, 160, 400, 800]
N_SAMPLE = 20


def _make_inputs(device, seed=0):
    rng = np.random.default_rng(seed)
    embed = rng.normal(0, 1, (M, E)).astype(np.float32)
    pred_vals = (np.sin(np.linspace(0, 40, M)) + rng.normal(0, 0.1, M)).astype(np.float32)
    embed_t = torch.tensor(embed, device=device)
    pred_vals_t = torch.tensor(pred_vals, device=device)
    time_idx_t = torch.arange(M, device=device)
    lib_batches = {}
    for L in LIB_SIZES:
        idx = np.stack([np.sort(rng.choice(M, size=L, replace=False)) for _ in range(N_SAMPLE)])
        lib_batches[L] = torch.tensor(idx, dtype=torch.long, device=device)
    return embed_t, pred_vals_t, time_idx_t, lib_batches


def _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device):
    '''One candidate's full CCM sweep (all libSizes x samples) -- the unit of
    work the CCM gate actually performs per candidate.'''
    full_dist = torch.cdist(embed_t.unsqueeze(0), embed_t.unsqueeze(0)).squeeze(0)
    for L in LIB_SIZES:
        batched_crossmap_predict_torch(embed_t, pred_vals_t, lib_batches[L],
                                       K, EXCL, time_idx_t, full_dist=full_dist)
    if device == 'cuda':
        torch.cuda.synchronize()


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------- H1: batching
def h1_batch_scaling(device, n_candidates=64):
    '''Same total work (n_candidates sweeps), batched B sweeps per call.
    Here "batching" means stacking B candidates' sample-batches into one bigger
    lib_idx batch, so one call does B times the work.'''
    print('\n--- H1: batch size as the throughput lever ---')
    embed_t, pred_vals_t, time_idx_t, lib_batches = _make_inputs(device)

    # warm-up
    _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device)

    for B in (1, 2, 4, 8, 16):
        big = {L: lib_batches[L].repeat(B, 1) for L in LIB_SIZES}  # B x more rows per call
        n_calls = max(1, n_candidates // B)
        _sync(device)
        t0 = time.perf_counter()
        full_dist = torch.cdist(embed_t.unsqueeze(0), embed_t.unsqueeze(0)).squeeze(0)
        for _ in range(n_calls):
            for L in LIB_SIZES:
                batched_crossmap_predict_torch(embed_t, pred_vals_t, big[L],
                                               K, EXCL, time_idx_t, full_dist=full_dist)
        _sync(device)
        dt = time.perf_counter() - t0
        eff = n_calls * B
        print(f'  batch={B:>2}  calls={n_calls:>3}  sweeps={eff:>3}  '
              f'total={dt:.3f}s  per-sweep={dt/eff*1000:.2f}ms')
    print('  H1 holds if per-sweep time drops materially as batch grows.')


# ------------------------------------------------------- H2: process contention
def _proc_worker(args):
    device, n_sweeps, seed = args
    torch.set_num_threads(1)
    embed_t, pred_vals_t, time_idx_t, lib_batches = _make_inputs(device, seed=seed)
    _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device)
    return time.perf_counter() - t0


def h2_process_scaling(device, total_sweeps=32):
    '''FIXED total work split across N concurrent processes sharing one GPU.
    On N CPU cores this would scale down ~linearly; the hypothesis is that on a
    single GPU it will not.'''
    print('\n--- H2: process count vs. throughput (fixed total work) ---')
    ctx = mp.get_context('spawn')  # CUDA cannot fork -- see H4 / plan 7a
    for n_proc in (1, 2, 4):
        per = max(1, total_sweeps // n_proc)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as ex:
            list(ex.map(_proc_worker, [(device, per, 100 + i) for i in range(n_proc)]))
        dt = time.perf_counter() - t0
        print(f'  procs={n_proc}  sweeps/proc={per}  total_sweeps={per*n_proc}  '
              f'wall={dt:.2f}s  throughput={per*n_proc/dt:.2f} sweeps/s')
    print('  H2 holds if throughput stays ~flat (or drops) as procs grow.')
    print('  NOTE: each spawn worker re-imports torch + re-creates its own CUDA')
    print('        context; that fixed startup is included in wall time here,')
    print('        which is itself part of the real cost of GPU-enabled workers.')


# --------------------------------------------------------- H3: thread contention
def h3_thread_scaling(device, total_sweeps=32):
    '''Same fixed work split across N threads in ONE process (one CUDA context),
    matching this pipeline's inner ThreadPoolExecutor design.'''
    print('\n--- H3: thread count vs. throughput (fixed total work, one process) ---')
    embed_t, pred_vals_t, time_idx_t, lib_batches = _make_inputs(device)
    _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device)  # warm-up

    def run_n(n):
        for _ in range(n):
            _one_sweep(embed_t, pred_vals_t, time_idx_t, lib_batches, device)

    for n_thr in (1, 2, 4, 8):
        per = max(1, total_sweeps // n_thr)
        _sync(device)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            list(ex.map(run_n, [per] * n_thr))
        _sync(device)
        dt = time.perf_counter() - t0
        print(f'  threads={n_thr}  sweeps/thread={per}  total_sweeps={per*n_thr}  '
              f'wall={dt:.2f}s  throughput={per*n_thr/dt:.2f} sweeps/s')
    print('  H3 holds if throughput stays ~flat as threads grow.')


# ------------------------------------------------------------- H4: fork + CUDA
def _fork_child_probe(q):
    try:
        x = torch.randn(10, 10, device='cuda')
        q.put(('ok', float(x.sum().item())))
    except Exception as exc:  # noqa: BLE001 -- reporting the failure IS the result
        q.put(('error', f'{type(exc).__name__}: {exc}'))


def h4_fork_fails(device):
    '''Confirm on THIS hardware what plan 7a claims from documentation: a forked
    child cannot use CUDA once the parent has a context. Expected to fail --
    that failure is the point.'''
    print('\n--- H4: CUDA + fork (expected to FAIL -- that is the confirmation) ---')
    if device != 'cuda':
        print('  skipped (no GPU)')
        return
    torch.cuda.init()
    _ = torch.randn(8, 8, device='cuda')  # ensure parent holds a live CUDA context
    torch.cuda.synchronize()

    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(target=_fork_child_probe, args=(q,))
    p.start()
    p.join(timeout=60)  # guarded: undefined behaviour can hang rather than raise
    if p.is_alive():
        p.terminate()
        p.join()
        print('  forked child HUNG (killed after 60s) -- worst case of the same')
        print('  hazard: confirms fork is unusable with CUDA, silently.')
        return
    if q.empty():
        print(f'  forked child died with exitcode={p.exitcode} and no message')
        print('  -- confirms fork is unusable with CUDA.')
        return
    status, detail = q.get()
    if status == 'error':
        print(f'  forked child failed as expected: {detail}')
        print('  -> confirms plan 7a on this hardware; GPU_ENABLED=1 correctly')
        print('     switches the pipeline outer pools to spawn.')
    else:
        print(f'  UNEXPECTED: forked child succeeded ({detail}).')
        print('  Do NOT relax the spawn requirement on this basis alone -- CUDA+fork')
        print('  is documented as undefined behaviour, and "worked once" is not a')
        print('  safety guarantee. Report this result; it is worth understanding.')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)}, '
          f'capability {torch.cuda.get_device_capability(0)}, '
          f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)'
          if device == 'cuda' else ''))
    if device != 'cuda':
        print('WARNING: no GPU visible -- results below measure CPU behaviour and do')
        print('         NOT test the hypotheses. Run this on a GPU node.')

    h1_batch_scaling(device)
    h3_thread_scaling(device)
    h2_process_scaling(device)
    h4_fork_fails(device)

    if device == 'cuda':
        print(f'\npeak GPU memory this run: '
              f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')


if __name__ == '__main__':
    main()
