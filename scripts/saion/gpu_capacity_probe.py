'''Can N spawned pipeline workers share ONE GPU? Measured, not assumed.

Run by scripts/batch/saion-preflight.sbatch on a Saion GPU node (1 GPU). It
answers the question the multi-GPU decision was deferred on: the parallel-MDE
fork starts `--cpus-per-task` spawn workers from the first second, and every
one of them that touches the CCM gate creates its OWN CUDA context on the same
device (plus the main process's context for the depth-0 joint seed). Plan
section 7c measured 1.27 GB peak for ONE process's whole CCM workload but
never measured N concurrent contexts, and CUDA context overhead alone is
typically a few hundred MB per process on Pascal -- 36 of them may or may not
fit in 16 GB.

For each worker count in --workers (default 1,9,18,<cpus>), it spawns that
many processes the same way the pipeline does (multiprocessing 'spawn'), each
of which:
  - imports torch + scripts/src/gpu_edm.py and initialises CUDA,
  - reports ready, then waits for a shared start signal,
  - runs full CCM-gate sweeps through the REAL primitive
    (gpu_edm.batched_crossmap_predict, numpy in/out, cached full_dist) at
    production size (M=CCM_WINDOW_ROWS=800, CCM_LIB_SIZES_ROWS, CCM_SAMPLE=20,
    E=MDE_D) for --duration seconds.
Meanwhile the parent polls nvidia-smi for device memory.used.

Reported per worker count: device memory with every context initialised but
idle (=> per-context overhead), peak device memory under load, OOM/CUDA
errors, and median seconds per sweep (=> contention slowdown vs. 1 worker).
The verdict at the end is what the multi-GPU decision should be based on.

    .venv-gpu/bin/python3 scripts/saion/gpu_capacity_probe.py --json out.json
'''
import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'

# Production CCM-gate shape -- keep in sync with the saion sbatch's
# CCM_WINDOW_ROWS / CCM_LIB_SIZES_ROWS / CCM_SAMPLE.
M = 800
LIB_SIZES = [16, 24, 40, 64, 160, 400, 800]
N_SAMPLE = 20


def _gpu_memory_used_mib():
    '''memory.used of the first visible GPU (Slurm's cgroup normally exposes
    only the allocated device to nvidia-smi). None if nvidia-smi fails.'''
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()
        used, total = (int(x) for x in out[0].split(','))
        return used, total
    except Exception:
        return None


def _worker(rank, E, duration, ready_q, result_q, go):
    try:
        sys.path.insert(0, str(SRC_DIR))
        import numpy as np
        import torch
        import gpu_edm

        torch.zeros(1, device='cuda')   # force context creation now, not mid-run
        rng = np.random.default_rng(rank)
        embed = rng.normal(0, 1, (M, E)).astype(np.float32)
        pred_vals = (np.sin(np.linspace(0, 40, M)) + rng.normal(0, 0.1, M)).astype(np.float32)
        time_idx = np.arange(M)
        lib_batches = {L: np.stack([np.sort(rng.choice(M, size=L, replace=False))
                                    for _ in range(N_SAMPLE)]) for L in LIB_SIZES}
        ready_q.put(rank)
        go.wait()

        sweeps = []
        end = time.monotonic() + duration
        while time.monotonic() < end:
            t0 = time.perf_counter()
            full_dist = None
            for L in LIB_SIZES:
                _, full_dist = gpu_edm.batched_crossmap_predict(
                    embed, pred_vals, lib_batches[L], E + 1, 0, time_idx,
                    full_dist=full_dist, device='cuda')
            sweeps.append(time.perf_counter() - t0)
        result_q.put({'rank': rank, 'ok': True, 'sweeps': len(sweeps),
                      'median_s': statistics.median(sweeps) if sweeps else None,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
    except BaseException as exc:   # noqa: BLE001 -- report everything, incl. CUDA OOM
        try:
            ready_q.put(rank)
        except Exception:
            pass
        result_q.put({'rank': rank, 'ok': False, 'error': repr(exc)})


def run_level(n, E, duration, ready_timeout):
    ctx = mp.get_context('spawn')   # CUDA cannot fork -- same as the pipeline
    ready_q, result_q, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    before = _gpu_memory_used_mib()
    t_spawn = time.monotonic()
    procs = [ctx.Process(target=_worker, args=(r, E, duration, ready_q, result_q, go))
             for r in range(n)]
    for p in procs:
        p.start()

    ready = 0
    deadline = time.monotonic() + ready_timeout
    while ready < n and time.monotonic() < deadline:
        try:
            ready_q.get(timeout=5)
            ready += 1
        except Exception:
            pass
    spawn_s = time.monotonic() - t_spawn
    time.sleep(3)
    idle = _gpu_memory_used_mib()

    peak = [idle[0] if idle else 0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            s = _gpu_memory_used_mib()
            if s:
                peak[0] = max(peak[0], s[0])
            stop.wait(1.0)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    go.set()

    results = []
    deadline = time.monotonic() + duration + 300
    while len(results) < n and time.monotonic() < deadline:
        try:
            results.append(result_q.get(timeout=10))
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
    stop.set()
    poller.join()
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()

    ok = [r for r in results if r.get('ok')]
    errors = [r['error'] for r in results if not r.get('ok')]
    lost = n - len(results)
    total = (idle or before or (None, None))[1]
    level = {
        'workers': n, 'ready': ready, 'spawn_s': round(spawn_s, 1),
        'mem_before_mib': before[0] if before else None,
        'mem_idle_all_contexts_mib': idle[0] if idle else None,
        'per_context_mib': (round((idle[0] - before[0]) / max(ready, 1)) if idle and before else None),
        'mem_peak_mib': peak[0], 'mem_total_mib': total,
        'median_sweep_s': (statistics.median(r['median_s'] for r in ok if r['median_s'])
                           if ok else None),
        'max_peak_reserved_mib': max((r['peak_reserved_mib'] for r in ok), default=None),
        'workers_ok': len(ok), 'workers_error': len(errors), 'workers_lost': lost,
        'errors': sorted(set(errors))[:5],
    }
    return level


def main():
    ap = argparse.ArgumentParser()
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '36'))
    ap.add_argument('--workers', default=f'1,9,18,{cpus}')
    ap.add_argument('--E', type=int, default=10, help='embedding dim; MDE_D is the ceiling in the pipeline')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--ready-timeout', type=float, default=900.0)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print('FAIL: no CUDA device visible')
        sys.exit(2)
    print(f'device: {torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}  '
          f'torch={torch.__version__}', flush=True)

    levels = []
    for n in [int(x) for x in args.workers.split(',') if x.strip()]:
        print(f'\n--- {n} concurrent spawn worker(s) on one GPU, {args.duration:.0f}s ---', flush=True)
        lv = run_level(n, args.E, args.duration, args.ready_timeout)
        levels.append(lv)
        for k, v in lv.items():
            print(f'  {k}: {v}', flush=True)

    base = levels[0]['median_sweep_s'] if levels and levels[0]['median_sweep_s'] else None
    print('\n=== SUMMARY ===')
    print(f'{"workers":>8} {"ok":>4} {"err":>4} {"ctx MiB":>8} {"peak MiB":>9} {"/total":>7} '
          f'{"s/sweep":>8} {"slowdown":>8}')
    for lv in levels:
        slow = (lv['median_sweep_s'] / base) if base and lv['median_sweep_s'] else None
        lv['slowdown_vs_1'] = round(slow, 2) if slow else None
        sweep_txt = '%.3f' % lv['median_sweep_s'] if lv['median_sweep_s'] else '-'
        slow_txt = '%.2fx' % slow if slow else '-'
        print(f'{lv["workers"]:>8} {lv["workers_ok"]:>4} {lv["workers_error"] + lv["workers_lost"]:>4} '
              f'{str(lv["per_context_mib"]):>8} {lv["mem_peak_mib"]:>9} {str(lv["mem_total_mib"]):>7} '
              f'{sweep_txt:>8} {slow_txt:>8}')

    top = levels[-1]
    fits = (top['workers_error'] == 0 and top['workers_lost'] == 0 and top['mem_total_mib']
            and top['mem_peak_mib'] + 1024 <= 0.9 * top['mem_total_mib'])
    # +1 context: the pipeline's main process also holds one.
    print('\n=== VERDICT ===')
    if fits:
        print(f'FITS: {top["workers"]} workers (+1 main-process context) on one GPU with '
              f'{top["mem_total_mib"] - top["mem_peak_mib"]} MiB headroom. Single-GPU job is OK; '
              f'multi-GPU assignment is only worth it if slowdown_vs_1 at this level is large.')
    else:
        # Size from memory per worker UNDER LOAD (context + working memory),
        # not the idle context alone: on P100 the idle context was 288 MiB but
        # a loaded worker ~420 MiB, and the idle figure over-promised by ~40%.
        # Same threshold as `fits`, minus one slot for the main-process context.
        per = ((top['mem_peak_mib'] - (top['mem_before_mib'] or 0)) / top['workers']
               if top['workers_ok'] else None)
        if per and top['mem_total_mib']:
            cap = max(0, int((0.9 * top['mem_total_mib'] - 1024) // per) - 1)
            print(f'DOES NOT FIT: at {top["workers"]} workers -> errors={top["errors"]}, '
                  f'peak={top["mem_peak_mib"]}/{top["mem_total_mib"]} MiB. '
                  f'~{per:.0f} MiB per loaded worker => about {cap} workers per GPU. '
                  f'Either lower --cpus-per-task to <= {cap} or implement per-worker GPU assignment '
                  f'and request more GPUs.')
        else:
            print(f'DOES NOT FIT / INCONCLUSIVE: {top}')
    if args.json:
        Path(args.json).write_text(json.dumps({'levels': levels, 'fits': bool(fits)}, indent=2))
    sys.exit(0 if fits else 1)


if __name__ == '__main__':
    main()
