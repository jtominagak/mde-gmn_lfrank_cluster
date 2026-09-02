import sys
import time
import numpy as np
import torch
torch.set_num_threads(1)  # confirmed: default 40 threads causes catastrophic per-op dispatch overhead at this tensor size

sys.path.insert(0, "/home/j/j-tominaga/pao_rotation/scripts/src")
from cleaned_helper_functions import _crossmap_predict


def batched_crossmap_predict_torch(embed_t, pred_vals_t, lib_idx_batch, k, exclusionRadius, time_indices_t, full_dist=None):
    M = embed_t.shape[0]
    B, L = lib_idx_batch.shape
    k = min(k, L - 1)
    if k < 1:
        return torch.full((B, M), float("nan"))

    k_query = k + 1
    if exclusionRadius > 0:
        k_query += 2 * exclusionRadius
    k_query = min(k_query, L)

    if full_dist is None:
        full_dist = torch.cdist(embed_t.unsqueeze(0), embed_t.unsqueeze(0)).squeeze(0)

    lib_cols = lib_idx_batch.unsqueeze(1).expand(B, M, L)
    dist_b = torch.gather(full_dist.unsqueeze(0).expand(B, M, M), 2, lib_cols)

    nn_dist, nn_local = torch.topk(dist_b, k_query, dim=2, largest=False, sorted=True)
    glob = torch.gather(lib_idx_batch.unsqueeze(1).expand(B, M, L), 2, nn_local)

    r_M = torch.arange(M).view(1, M, 1)
    is_self = glob == r_M
    if exclusionRadius > 0:
        qt = time_indices_t.view(1, M, 1)
        nt = time_indices_t[glob]
        mask_bad = is_self | (torch.abs(qt - nt) <= exclusionRadius)
    else:
        mask_bad = is_self

    valid = ~mask_bad
    cs = torch.cumsum(valid.long(), dim=2)
    first_k = valid & (cs <= k)
    insufficient = cs[:, :, -1] < k  # (B, M)

    # Compact: gather exactly the first-k valid neighbors into dense (B,M,k)
    # slots, matching the reference's np.where+reshape compaction -- NOT
    # zero-out-in-place, which would leave d_min computed from an
    # uncompacted (possibly invalid/self) nearest neighbor.
    sort_idx = torch.argsort((~first_k).long(), dim=2, stable=True)  # False(valid&first-k)=0 sorts first
    compact_idx = sort_idx[:, :, :k]
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
    preds = torch.where(insufficient, torch.full_like(preds, float("nan")), preds)
    return preds


def main():
    rng = np.random.default_rng(0)
    M = 2000
    E = 3
    embed = rng.normal(0, 1, (M, E)).astype(np.float64)
    pred_vals = np.sin(np.linspace(0, 40, M)) + rng.normal(0, 0.1, M)
    time_indices = np.arange(M)
    k = E + 1
    exclusionRadius = 2

    libSizes = [200, 400, 800, 1200, 1600, 1800, 1900]
    n_sample = 20

    t0 = time.perf_counter()
    ref_results = {}
    for L in libSizes:
        for s in range(n_sample):
            lrng = np.random.default_rng(1000 * L + s)
            lib_idx = np.sort(lrng.choice(M, size=L, replace=False))
            preds = _crossmap_predict(embed, pred_vals, lib_idx, k, exclusionRadius, time_indices)
            ref_results[(L, s)] = (lib_idx, preds)
    t_ref = time.perf_counter() - t0
    print(f"Reference (numpy/KDTree, {len(libSizes)*n_sample} calls): {t_ref:.3f}s")

    embed_t = torch.tensor(embed, dtype=torch.float32)
    pred_vals_t = torch.tensor(pred_vals, dtype=torch.float32)
    time_indices_t = torch.tensor(time_indices, dtype=torch.long)

    t0 = time.perf_counter()
    full_dist = torch.cdist(embed_t.unsqueeze(0), embed_t.unsqueeze(0)).squeeze(0)  # cached ONCE across all libSizes/samples
    torch_results = {}
    for L in libSizes:
        lib_idx_batch = np.stack([ref_results[(L, s)][0] for s in range(n_sample)])
        lib_idx_batch_t = torch.tensor(lib_idx_batch, dtype=torch.long)
        preds_b = batched_crossmap_predict_torch(embed_t, pred_vals_t, lib_idx_batch_t, k, exclusionRadius, time_indices_t, full_dist=full_dist)
        for s in range(n_sample):
            torch_results[(L, s)] = preds_b[s].numpy()
    t_torch = time.perf_counter() - t0
    print(f"Torch batched, single-threaded, cached cdist ({len(libSizes)} batched calls of size {n_sample}): {t_torch:.3f}s")
    print(f"Speedup: {t_ref/t_torch:.2f}x")

    max_abs_diff = 0.0
    n_compared = 0
    n_nan_mismatch = 0
    for L in libSizes:
        for s in range(n_sample):
            ref_preds = ref_results[(L, s)][1]
            tor_preds = torch_results[(L, s)]
            ref_nan = np.isnan(ref_preds)
            tor_nan = np.isnan(tor_preds)
            if not np.array_equal(ref_nan, tor_nan):
                n_nan_mismatch += 1
                continue
            valid_mask = ~ref_nan
            diff = np.abs(ref_preds[valid_mask] - tor_preds[valid_mask])
            if diff.size:
                max_abs_diff = max(max_abs_diff, diff.max())
                n_compared += diff.size
    print(f"Max abs diff (float64 ref vs float32 torch), {n_compared} points compared: {max_abs_diff:.2e}")
    print(f"NaN-pattern mismatches: {n_nan_mismatch} / {len(libSizes)*n_sample}")


if __name__ == "__main__":
    main()
