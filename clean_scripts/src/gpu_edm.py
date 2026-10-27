# ====================================================================
# gpu_edm.py
# ====================================================================
# Isolated GPU (torch/CUDA) primitive for the CCM gate only -- see
# <repo>/docs/no-kEDM-gpu-optimization-plan.md section 8. Kept as its own module
# (not folded into edm.py) so the torch import stays
# optional and contained: this file must import cleanly on a machine with
# no GPU and no CUDA (Deigo), and equally cleanly wherever CUDA IS
# available (Saion). Every public function here takes/returns numpy
# arrays -- callers never need to import torch themselves or handle torch
# types.
#
# Ported UNCHANGED in algorithm from
# the project's GPU prototype (not part of this tree),
# which is already numerically validated against
# edm._crossmap_predict (max abs diff 1.89e-05 at
# production scale M=800 across 112,000 compared points, 0/140 NaN-pattern
# mismatches -- see <repo>/docs/no-kEDM-gpu-optimization-progress.md's
# "production-scale results" entry). Two details here are load-bearing --
# each was a real, silent correctness bug once when it was missing (see
# that progress log's 2026-09-02 entries):
#   - the first-k-valid COMPACTION (torch.argsort + torch.gather) done
#     BEFORE computing d_min, mirroring the reference's np.where+reshape
#     compaction exactly. Computing d_min from the raw nearest-of-
#     over-fetched neighbors instead (skipping compaction) silently let a
#     self-match or excluded near-in-time neighbor become the distance-
#     weighting reference point -- errors up to 1.16 absolute, far beyond
#     float32-vs-float64 noise.
#   - torch.set_num_threads(1) -- torch's default per-node thread count
#     (one thread per core) imposes large fixed per-op dispatch overhead
#     on tensors this small; leaving it at the default measured 21.7x
#     slower on CPU for reasons unrelated to any GPU work.
# ====================================================================

import numpy as np

try:
    import torch
except Exception:
    torch = None
else:
    torch.set_num_threads(1)


def is_available():
    """True only if torch imported successfully AND a CUDA device is
    visible. Never raises -- a GPU-less or torch-less environment must be
    able to call this safely and get False back."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def batched_crossmap_predict(embed, pred_vals, lib_idx_batch, k, exclusionRadius,
                              time_indices, full_dist=None, device=None):
    """GPU-batched equivalent of edm._crossmap_predict,
    run over B libraries that all share the same length L at once (the
    batch dimension). Algorithm matches that reference function line for
    line -- see this module's own docstring above for the two details
    that must not be dropped.

    INPUT
    -----
    embed : (M, E) numpy array -- same meaning as _crossmap_predict's own
        `embed` (the full embedding, not just the library).
    pred_vals : (M,) numpy array -- same meaning as _crossmap_predict's
        own `pred_vals`.
    lib_idx_batch : (B, L) int array -- B independently-drawn library
        index sets, ALL THE SAME LENGTH L (this is what makes batching
        valid: every row must be a `lib_idx` _crossmap_predict could have
        been called with individually). Callers are responsible for the
        grouping that guarantees this -- see proper_ccm's own use_gpu
        branch, which groups the already-drawn tasks by libSize index for
        exactly this reason.
    k, exclusionRadius, time_indices : same meaning as
        _crossmap_predict's own arguments.
    full_dist : optional cached (M, M) torch tensor (opaque to callers --
        never inspect or construct one directly) from a PRIOR call
        against this exact SAME `embed`. Passing it back in skips
        recomputing the O(M^2) pairwise distance matrix, which is
        otherwise the single most expensive step and does not depend on
        `lib_idx_batch`/`k` at all -- this is what the validated
        prototype's "cached cdist" timing numbers reflect (progress log,
        2026-09-02 production-scale entry: 7 batched calls sharing one
        cdist). Passing a `full_dist` computed against a DIFFERENT embed
        (e.g. reusing the forward direction's cache for the reverse
        direction) silently produces wrong results -- never do that.
    device : 'cuda' / 'cpu' / None. None (default) resolves to 'cuda' if
        `is_available()` else 'cpu'.

    OUTPUT
    ------
    (preds, full_dist) : preds is a (B, M) float64 numpy array, NaN
        wherever that (batch row, query point) pair couldn't get k valid
        neighbors -- same convention as _crossmap_predict's own (M,)
        return. full_dist is the (possibly newly computed) cached tensor
        described above; pass it back into the next call against the
        same `embed` to reuse it.
    """
    if torch is None:
        raise RuntimeError('gpu_edm.batched_crossmap_predict called but torch is not installed')
    if device is None:
        device = 'cuda' if is_available() else 'cpu'

    embed_t = torch.as_tensor(np.asarray(embed, dtype=np.float32), device=device)
    pred_vals_t = torch.as_tensor(np.asarray(pred_vals, dtype=np.float32), device=device)
    lib_idx_t = torch.as_tensor(np.asarray(lib_idx_batch, dtype=np.int64), device=device)
    time_indices_t = torch.as_tensor(np.asarray(time_indices, dtype=np.int64), device=device)

    M = embed_t.shape[0]
    B, L = lib_idx_t.shape
    k_eff = min(k, L - 1)
    if k_eff < 1:
        preds = torch.full((B, M), float('nan'), device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        return preds.cpu().numpy().astype(np.float64), full_dist

    # Over-fetch neighbors: +1 for the query point possibly being its own
    # nearest neighbor, +2*exclusionRadius as headroom for however many
    # additional near-in-time neighbors get excluded below -- identical
    # reasoning to _crossmap_predict's own k_query.
    k_query = k_eff + 1
    if exclusionRadius > 0:
        k_query += 2 * exclusionRadius
    k_query = min(k_query, L)

    if full_dist is None:
        full_dist = torch.cdist(embed_t.unsqueeze(0), embed_t.unsqueeze(0)).squeeze(0)

    lib_cols = lib_idx_t.unsqueeze(1).expand(B, M, L)
    dist_b = torch.gather(full_dist.unsqueeze(0).expand(B, M, M), 2, lib_cols)

    nn_dist, nn_local = torch.topk(dist_b, k_query, dim=2, largest=False, sorted=True)
    glob = torch.gather(lib_idx_t.unsqueeze(1).expand(B, M, L), 2, nn_local)

    r_M = torch.arange(M, device=device).view(1, M, 1)
    is_self = glob == r_M
    if exclusionRadius > 0:
        qt = time_indices_t.view(1, M, 1)
        nt = time_indices_t[glob]
        mask_bad = is_self | (torch.abs(qt - nt) <= exclusionRadius)
    else:
        mask_bad = is_self

    valid = ~mask_bad
    cs = torch.cumsum(valid.long(), dim=2)
    first_k = valid & (cs <= k_eff)
    insufficient = cs[:, :, -1] < k_eff  # (B, M)

    # Compact: gather exactly the first-k valid neighbors into dense
    # (B, M, k) slots, matching the reference's np.where+reshape
    # compaction -- NOT zero-out-in-place, which would leave d_min
    # computed from an uncompacted (possibly invalid/self) neighbor. See
    # this module's own docstring for why this step is load-bearing.
    sort_idx = torch.argsort((~first_k).long(), dim=2, stable=True)  # False(valid&first-k)=0 sorts first
    compact_idx = sort_idx[:, :, :k_eff]
    nn_dist_c = torch.gather(nn_dist, 2, compact_idx)
    glob_c = torch.gather(glob, 2, compact_idx)

    d_min = nn_dist_c[:, :, 0:1]
    d_min_nz = torch.where(d_min > 0, d_min, torch.ones_like(d_min))
    w = torch.exp(-nn_dist_c / d_min_nz)
    zero = d_min == 0
    w = torch.where(zero, torch.where(nn_dist_c == 0, torch.ones_like(w), torch.zeros_like(w)), w)
    w_sum = w.sum(dim=2, keepdim=True)
    w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
    w = w / w_sum

    pred_at_nn = pred_vals_t[glob_c]
    preds = (w * pred_at_nn).sum(dim=2)
    preds = torch.where(insufficient, torch.full_like(preds, float('nan')), preds)

    if device == 'cuda':
        torch.cuda.synchronize()
    return preds.cpu().numpy().astype(np.float64), full_dist
