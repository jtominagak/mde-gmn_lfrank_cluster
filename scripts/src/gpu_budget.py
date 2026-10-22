# ====================================================================
# gpu_budget.py
# ====================================================================
# Cross-process GPU memory admission control for the CCM gate.
#
# WHY THIS EXISTS -- job 4741044. Eight worker processes each launched
# their GPU cross-map blindly and discovered there was no room by
# raising; proper_ccm's caller caught the exception, printed a line and
# `continue`d, so 51 candidate connections across 85 nodes were never
# evaluated and were indistinguishable downstream from "we checked and
# found nothing". The GPU peaked at 16203/16384 MiB. The preflight that
# cleared "9 workers on one 16 GB card" had measured only the ~288 MiB
# CUDA *context* cost per worker, not the multi-GB working set a real
# cross-map allocates.
#
# THE RULE THIS MODULE ENFORCES: a worker does not start GPU work until
# the device provably has room for it. Reserve first, allocate second.
# An out-of-memory failure stops being something to recover from and
# becomes something that cannot arise.
#
# Three properties, in the order they matter:
#
#   1. NO SILENT LOSS. reserve() either grants the memory or blocks.
#      It never declines, and callers never get a "couldn't run" result
#      they might mistake for a scientific negative. If the GPU truly
#      cannot serve a request even with the device to itself, the OOM
#      surfaces as a raised exception that stops the run -- see
#      is_gpu_memory_error()/retry_on_oom() and the call sites in
#      cleaned_helper_functions.py, which re-raise rather than skip.
#
#   2. NO STARVATION. Admission is strict FIFO by ticket. A large
#      request cannot be indefinitely overtaken by a queue of small
#      ones. FIFO costs less parallelism than it sounds: when the head
#      of the queue fits it is served immediately and the next head is
#      considered in the same notify, so a run of small requests is
#      admitted back-to-back and then runs concurrently. The queue only
#      stalls behind a request that does not currently fit -- which is
#      exactly when stalling is the point.
#
#   3. NO DEADLOCK. Each reservation is taken in one atomic step (never
#      "some now, more later", which is how counting-semaphore schemes
#      deadlock). A request larger than the whole budget is clamped to
#      the whole budget, i.e. it runs with the device to itself, rather
#      than waiting forever for an amount that can never be free.
#      Nested reserve() in one process is a pass-through, since the
#      outer reservation already covers it.
#
# Per DEVICE, not per job: with --gres=gpu:N each device gets its own
# independent budget and its own queue, and init_worker() pins each
# worker to one device round-robin.
#
# This module must import cleanly with no torch and no CUDA (Deigo) --
# every entry point degrades to a transparent no-op when there is no
# budget configured, which is also what makes the serial code paths and
# the unit tests work unchanged.
# ====================================================================

import os
import time
from contextlib import contextmanager

try:
    import torch
except Exception:                                      # pragma: no cover
    torch = None


# Budget bookkeeping is in whole tokens so the shared counter stays a
# small integer. 32 MiB is fine-grained enough that a per-call estimate
# rounds up by at most ~1.5% of a 2 GB request, and coarse enough that a
# 16 GiB card is only ~500 tokens.
TOKEN_BYTES = 32 * 1024 * 1024

# Measured by scripts/batch/saion-preflight.sbatch: a worker process that
# merely attaches to the device costs ~288 MiB before it allocates
# anything. Rounded up -- this is subtracted from the budget once per
# worker sharing the device, because that memory is gone the moment the
# worker starts, whatever it goes on to do.
CONTEXT_BYTES = 320 * 1024 * 1024

# Held back on top of the per-worker contexts: driver/allocator overhead,
# fragmentation, and the non-PyTorch share nvidia-smi attributes to the
# process. Deliberately generous -- the cost of over-reserving is some
# serialisation, the cost of under-reserving is the failure mode this
# module exists to remove.
SAFETY_BYTES = 1024 * 1024 * 1024

# How long a single wait may go unreported. Purely observational: a long
# wait is correct behaviour (something big is in front of you), but a
# silent multi-hour wait is indistinguishable from a hang in the log.
WAIT_LOG_SECONDS = 300.0


class GpuBudgetError(RuntimeError):
    '''Raised for a request this module can prove is unservable -- as
    opposed to a torch OOM, which it cannot prove anything about.'''


# --------------------------------------------------------------------
# Recognising a memory failure
# --------------------------------------------------------------------

# Spellings seen in job 4741044's own logs, from both layers that can
# report exhaustion: torch's allocator
# ('CUDA out of memory', 'memory allocation failed with OOM') and the
# driver underneath it ('CUDA error: out of memory'). Matched on the
# message because torch raises plain RuntimeError for several of these,
# so the exception type alone does not separate "no memory" from "your
# library size exceeds your data" -- a distinction the call sites depend
# on, since the latter is a real dead end and the former never is.
_OOM_MARKERS = (
    'cuda out of memory',
    'out of memory',
    'memory allocation failed with oom',
    'cublas_status_alloc_failed',
    'cudnn_status_alloc_failed',
)


def is_gpu_memory_error(exc):
    '''True when `exc` reports GPU memory exhaustion rather than a
    genuine analytical dead end.

    The whole point of separating these: "Maximum libSize 800 exceeds
    data size 703" is a real finding about short data and should be
    recorded as a dead end, while "CUDA out of memory" says nothing
    about the science and must never be allowed to look like one. In
    job 4741044 both went down the same `except Exception: continue`.'''
    if torch is not None:
        oom_cls = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    text = f'{type(exc).__name__}: {exc}'.lower()
    return any(marker in text for marker in _OOM_MARKERS)


# --------------------------------------------------------------------
# Estimating what a cross-map call will cost
# --------------------------------------------------------------------

def crossmap_bytes(m_rows, batch, lib_len, k_query, dist_matrices=2):
    '''Peak device bytes for one proper_ccm GPU section, from the tensor
    shapes gpu_edm.batched_crossmap_predict actually allocates.

    Three terms, largest first in the regimes this pipeline runs in:

      dist_matrices * m_rows^2 * 4
          the cached (M, M) float32 pairwise distance matrices. proper_ccm
          holds the forward and reverse ones SIMULTANEOUSLY across the
          whole libSize loop (they are different embeddings and must never
          be shared), hence the default of 2.

      batch * m_rows * lib_len * 4
          `dist_b`, the gather of those distances down to the B library
          index sets. This is the term that actually blew up job 4741044:
          at B=20, M=15654, L=800 it is ~1.0 GB on its own, and it scales
          linearly in all three.

      batch * m_rows * k_query * 80
          the topk/compaction family -- nn_dist, nn_local, glob, the three
          boolean masks, cs, first_k, sort_idx, compact_idx, nn_dist_c,
          glob_c, pred_at_nn and the weight temporaries. 80 B/neighbour is
          the summed per-element width of those, rounded up, with k_eff
          bounded by k_query.

    The 1.5 factor covers allocator rounding, the transient overlap
    between the forward direction's working set and the reverse
    direction's, and -- the reason it is not smaller -- the gap between
    LIVE tensors and RESERVED memory. torch's caching allocator keeps
    freed blocks rather than returning them to the driver, and what the
    driver (and therefore every other process) sees as unavailable is the
    reserved figure. In job 4741044 a process whose live tensors come to
    ~3.6 GB by the sum above was reported holding 8.51 GiB. reserve()
    closes most of that gap by emptying the cache when a reservation is
    released; this factor covers the rest, within a reservation.

    An estimate, not a measurement: it is meant to be comfortably high.
    If it is ever too low the result is an ordinary OOM, which
    retry_on_oom() turns into a retry and then, if that fails, a raised
    exception -- never a dropped candidate.'''
    m_rows = max(int(m_rows), 1)
    batch = max(int(batch), 1)
    lib_len = max(int(lib_len), 1)
    k_query = max(int(k_query), 1)

    dist_cache = dist_matrices * m_rows * m_rows * 4
    gathered = batch * m_rows * lib_len * 4
    neighbours = batch * m_rows * k_query * 80
    return int(1.5 * (dist_cache + gathered + neighbours))


# --------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------

class _DeviceBudget:
    '''One device's queue. Plain attributes holding multiprocessing
    primitives, so the whole object travels to a spawned worker through
    ProcessPoolExecutor's initargs (which are passed at process
    construction, the only way mp primitives can be shared).'''

    def __init__(self, ctx, index, total_tokens):
        self.index = index
        self.total_tokens = total_tokens
        self.cond = ctx.Condition()
        self.avail = ctx.Value('l', total_tokens, lock=False)
        self.next_ticket = ctx.Value('l', 0, lock=False)
        self.now_serving = ctx.Value('l', 0, lock=False)


# Set by init_worker() in each pool worker; stays None in the parent and
# in any process that never joined a pool, where reserve() is a no-op
# because there is nobody to contend with.
_MY_BUDGET = None
_MY_DEVICE = None
_RESERVE_DEPTH = 0


def device_count():
    '''Visible CUDA devices, or 0. Never raises.'''
    if torch is None:
        return 0
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def create_budgets(ctx, workers, devices=None, verbose=True):
    '''Build one _DeviceBudget per visible device, sized from what the
    device actually reports rather than a hardcoded card size -- the job
    is submitted to whichever of gpu-p100/gpu-v100 starts sooner (see
    scripts/saion/submit_saion.sh) and those cards need not have the same
    memory. Returns [] when there is no GPU, which makes every other
    entry point here a no-op.

    `workers` is the number of worker processes that will share the
    devices; each device's budget is reduced by the context cost of the
    workers assigned to it, because that memory is spoken for as soon as
    they start.'''
    n_dev = devices if devices is not None else device_count()
    if not n_dev:
        return []

    budgets = []
    for i in range(n_dev):
        try:
            total = torch.cuda.get_device_properties(i).total_memory
        except Exception:
            continue
        on_this_device = max(1, -(-workers // n_dev))   # ceil, workers are round-robined
        usable = total - on_this_device * CONTEXT_BYTES - SAFETY_BYTES
        tokens = int(usable // TOKEN_BYTES)
        if tokens < 1:
            raise GpuBudgetError(
                f'GPU {i}: {total / 2**30:.1f} GiB total leaves nothing usable after '
                f'{on_this_device} worker context(s) and the safety reserve. '
                f'Lower --cpus-per-task or raise --gres=gpu:N.')
        budgets.append(_DeviceBudget(ctx, i, tokens))
        if verbose:
            print(f'GPU budget: device {i} -- {total / 2**30:.1f} GiB total, '
                  f'{on_this_device} worker(s), '
                  f'{tokens * TOKEN_BYTES / 2**30:.1f} GiB admissible', flush=True)
    return budgets


def init_worker(budgets, counter):
    '''ProcessPoolExecutor initializer: pin this worker to one device
    round-robin and remember that device's queue.

    Pinning matters for more than tidiness. torch.cuda.set_device() makes
    the plain string 'cuda' resolve to this worker's own card, so
    gpu_edm.batched_crossmap_predict needs no change to become
    multi-GPU-aware, and -- more to the point -- a worker can only ever
    consume the budget it is queueing against.'''
    global _MY_BUDGET, _MY_DEVICE
    if not budgets:
        return
    with counter.get_lock():
        slot = counter.value
        counter.value = slot + 1
    _MY_BUDGET = budgets[slot % len(budgets)]
    _MY_DEVICE = _MY_BUDGET.index
    if torch is not None:
        try:
            torch.cuda.set_device(_MY_DEVICE)
        except Exception:
            pass


def active_device():
    '''This worker's device index, or None when unpinned.'''
    return _MY_DEVICE


# --------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------

@contextmanager
def reserve(nbytes, label=''):
    '''Block until this process may hold `nbytes` on its device, then
    yield. Releases on the way out, including on exception.

    Does NOT allocate anything -- it reserves the right to. The caller
    allocates inside the block, and the arithmetic only holds if the
    caller's own estimate is honest; crossmap_bytes() is deliberately
    generous for that reason.

    No-op (yields immediately) when there is no budget for this process:
    no GPU, or a serial path with no pool, where there is no second
    consumer to contend with. Also a pass-through when already inside a
    reserve() on this process -- the outer block's tokens cover it, and
    re-entering the FIFO behind yourself is a self-deadlock.'''
    global _RESERVE_DEPTH

    budget = _MY_BUDGET
    if budget is None or nbytes <= 0 or _RESERVE_DEPTH > 0:
        _RESERVE_DEPTH += 1
        try:
            yield
        finally:
            _RESERVE_DEPTH -= 1
        return

    need = max(1, -(-int(nbytes) // TOKEN_BYTES))       # ceil, never 0
    clamped = False
    if need > budget.total_tokens:
        # Cannot ever be satisfied as asked, so asking would hang. Take
        # the device exclusively instead and let the allocation itself
        # decide: if it fits with nobody else present it proceeds, and
        # if it does not, retry_on_oom() raises and the run stops with a
        # real message. Either beats waiting for memory that will never
        # exist.
        need = budget.total_tokens
        clamped = True

    waited_from = time.monotonic()
    with budget.cond:
        ticket = budget.next_ticket.value
        budget.next_ticket.value = ticket + 1
        reported = False
        while not (budget.now_serving.value == ticket and budget.avail.value >= need):
            budget.cond.wait(timeout=WAIT_LOG_SECONDS)
            waited = time.monotonic() - waited_from
            if not reported and waited >= WAIT_LOG_SECONDS:
                reported = True
                print(f'GPU budget: waiting {waited:.0f}s for '
                      f'{need * TOKEN_BYTES / 2**30:.2f} GiB on device {budget.index} '
                      f'({budget.avail.value * TOKEN_BYTES / 2**30:.2f} GiB free, '
                      f'queue position {ticket - budget.now_serving.value}) -- {label}',
                      flush=True)
        budget.avail.value -= need
        # Advance the head as soon as it is served, not when it finishes,
        # so the next ticket is evaluated in this same notify and a run of
        # small requests is admitted back-to-back.
        budget.now_serving.value = ticket + 1
        budget.cond.notify_all()

    if clamped:
        # "To itself" means no other worker is doing GPU WORK. It does not mean
        # the card is empty: every sibling worker keeps its ~CONTEXT_BYTES CUDA
        # context resident for its whole life, which is why that cost was
        # already subtracted from total_tokens. A request that exceeds the
        # context-adjusted budget therefore wants memory that does not exist on
        # this card, and the OOM that usually follows is a sizing shortfall, not
        # a transient -- job 4744295 clamped 30 times and lost 20 candidates.
        print(f'GPU budget: {label or "request"} estimated at '
              f'{nbytes / 2**30:.2f} GiB exceeds device {budget.index}\'s '
              f'{budget.total_tokens * TOKEN_BYTES / 2**30:.2f} GiB budget -- '
              f'running with the device to itself. Sibling CUDA contexts stay '
              f'resident, so this may still OOM; if it does, raise --gres=gpu:N, '
              f'lower --cpus-per-task, or reduce MAX_WINDOW_ROWS/CCM_SAMPLE.',
              flush=True)

    # Measure what this reservation actually costs, so an estimate that is
    # too low shows up in the log as a number instead of as a mysterious OOM
    # three hours later. crossmap_bytes() is a formula over tensor shapes;
    # this is what it gets checked against, and the two diverging is the
    # signal to re-derive it.
    #
    # It is the GROWTH over this reservation, not the absolute peak, and that
    # distinction is the whole reliability of the number.
    # max_memory_reserved() is a PROCESS-WIDE high-water mark, and
    # reset_peak_memory_stats() rebases it only to what the allocator is
    # ALREADY holding -- so the absolute figure charges this call for every
    # block its process cached earlier. Job 4744295 reported 73x
    # under-estimates on M=1003 calls whose own tensors come to ~0.1 GiB:
    # that reading was an earlier, far larger call's cache, not this one's,
    # and it made the advisory useless exactly where it mattered. Subtracting
    # the baseline attributes to this reservation only what it added.
    baseline = 0
    if torch is not None:
        try:
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_reserved()
        except Exception:
            baseline = 0

    _RESERVE_DEPTH += 1
    try:
        yield
    finally:
        _RESERVE_DEPTH -= 1
        if torch is not None:
            try:
                actual = max(torch.cuda.max_memory_reserved() - baseline, 0)
            except Exception:
                actual = 0
            if actual > nbytes:
                print(f'GPU budget: UNDER-ESTIMATED {label or "request"} -- grew '
                      f'{actual / 2**30:.2f} GiB against an estimate of '
                      f'{nbytes / 2**30:.2f} GiB ({actual / max(nbytes, 1):.2f}x), '
                      f'over a {baseline / 2**30:.2f} GiB baseline. Admission '
                      f'control is running looser than intended on device '
                      f'{budget.index}; raise crossmap_bytes()\'s headroom factor. '
                      f'This is an advisory, NOT a failure -- the reservation was '
                      f'granted and the call ran.',
                      flush=True)
        # Hand the memory back to the DRIVER before handing the tokens
        # back to the queue. torch's caching allocator would otherwise
        # keep the freed blocks, so the next worker admitted against
        # those tokens would find the card still occupied by a process
        # that is, as far as this module is concerned, finished. Cheap
        # at this granularity: one reservation spans a whole proper_ccm
        # section, not a single kernel.
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        with budget.cond:
            budget.avail.value += need
            budget.cond.notify_all()


# --------------------------------------------------------------------
# The backstop
# --------------------------------------------------------------------

def retry_on_oom(fn, *args, attempts=3, pause=5.0, label='', **kwargs):
    '''Call fn(*args, **kwargs), retrying only a GPU-memory failure.

    Admission control should make this unreachable; it is here because
    "should" is not "does" -- fragmentation, a stray process on a shared
    card, or an estimate that came in low can all still produce an OOM.

    Between attempts it empties torch's cache, which returns blocks the
    allocator is holding but not using to the driver, and waits for other
    workers to release theirs.

    On the final failure it RAISES. That is the whole contract: a
    candidate is either evaluated or the run stops and says why. It is
    never quietly skipped, because a skipped candidate is a missing edge
    that looks exactly like a measured absence.

    Anything that is not a memory failure propagates immediately and
    untouched -- a real dead end must not be retried, let alone
    swallowed here.'''
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_gpu_memory_error(exc):
                raise
            last = exc
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if attempt < attempts:
                print(f'GPU OOM on attempt {attempt}/{attempts} for {label or fn!r} '
                      f'(device {_MY_DEVICE}) -- freeing cache and retrying in {pause:.0f}s: '
                      f'{exc}', flush=True)
                time.sleep(pause)

    raise GpuBudgetError(
        f'GPU out of memory after {attempts} attempts for {label or fn!r} on device '
        f'{_MY_DEVICE}. This candidate was NOT evaluated, and the run is stopping '
        f'rather than recording it as a dead end -- a silently skipped candidate is '
        f'indistinguishable from a measured absence (see job 4741044, which lost 51 '
        f'of them this way). Raise --gres=gpu:N, lower --cpus-per-task, or reduce '
        f'MAX_WINDOW_ROWS/CCM_SAMPLE. Underlying error: {last}') from last


def env_summary():
    '''One line for the job log, so a run records what it was working
    with rather than what the sbatch hoped for.'''
    n = device_count()
    if not n:
        return 'GPU budget: no CUDA device visible -- admission control inactive'
    parts = []
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            parts.append(f'{i}:{p.name} {p.total_memory / 2**30:.0f}GiB')
        except Exception:
            parts.append(f'{i}:?')
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')
    return f'GPU budget: {n} device(s) [{", ".join(parts)}] PYTORCH_CUDA_ALLOC_CONF={alloc_conf}'
