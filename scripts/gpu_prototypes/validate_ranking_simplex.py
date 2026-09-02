"""Realistic ranking-stage benchmark: many candidates, each appending one new
column to a shared 'selected' embedding, predicting `target` via Simplex.
Compares looped pyEDM.Simplex (current) against ONE batched torch.cdist call
across all candidates at once (single-threaded), at production-realistic
lib/pred sizes. Validates numerical parity against pyEDM.Simplex first.
"""
import time
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from pyEDM import Simplex

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batched_ranking_simplex_torch(lib_states, pred_states, target_lib_vals, Tp):
    """lib_states/pred_states: (B, n, E) float32 -- B independent candidate
    embeddings sharing the same row ranges. target_lib_vals: (n_lib,) the
    target series over the FULL series (so target[t+Tp] can be read off).
    Returns (B, n_result) predictions, matching pyEDM.Simplex(embedded=True)'s
    own convention: neighbors found in lib_states, weighted-average of
    target's value at (neighbor_lib_row + Tp).
    """
    B, n_lib, E = lib_states.shape
    n_pred = pred_states.shape[1]
    k = E + 1

    dist = torch.cdist(pred_states, lib_states)  # (B, n_pred, n_lib)
    n_valid_lib = n_lib - Tp  # library rows whose target[row+Tp] exists
    dist = dist[:, :, :n_valid_lib]

    nn_dist, nn_idx = torch.topk(dist, k, dim=2, largest=False, sorted=True)
    d_min = nn_dist[:, :, 0:1]
    d_min_nz = torch.where(d_min > 0, d_min, torch.ones_like(d_min))
    w = torch.exp(-nn_dist / d_min_nz)
    zero = d_min == 0
    w = torch.where(zero, torch.where(nn_dist == 0, torch.ones_like(w), torch.zeros_like(w)), w)
    w_sum = w.sum(dim=2, keepdim=True)
    w = w / torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))

    tgt_at_nn = target_lib_vals[nn_idx + Tp]  # (B, n_pred, k)
    preds = (w * tgt_at_nn).sum(dim=2)
    return preds


def main():
    rng = np.random.default_rng(0)
    n_lib = 20000
    n_pred = 2000
    n_total = n_lib + n_pred
    Tp = 1
    n_candidates = 100

    selected = np.sin(np.linspace(0, 400, n_total)) + rng.normal(0, 0.05, n_total)
    target = np.roll(selected, 5) + rng.normal(0, 0.05, n_total)
    cand_cols = {f"c{i}": np.roll(selected, (i % 30) - 15) + rng.normal(0, 0.1, n_total) for i in range(n_candidates)}
    df = pd.DataFrame({"Time": np.arange(1, n_total + 1), "target": target, "selected": selected, **cand_cols})

    lib_str = f"1 {n_lib}"
    pred_str = f"{n_lib+1} {n_total}"

    # ---- reference: looped pyEDM.Simplex (current production engine) ----
    t0 = time.perf_counter()
    ref_preds = {}
    for i in range(n_candidates):
        out = Simplex(dataFrame=df, lib=lib_str, pred=pred_str, tau=1, Tp=Tp,
                       target="target", columns=[f"c{i}", "selected"], E=2,
                       embedded=True, showPlot=False, noTime=True)
        ref_preds[i] = out["Predictions"].to_numpy(dtype=np.float64)
    t_ref = time.perf_counter() - t0
    print(f"pyEDM.Simplex looped, {n_candidates} candidates: {t_ref:.3f}s ({t_ref/n_candidates*1000:.2f} ms/call)")

    print(f"Torch device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)}, "
          f"compute capability {torch.cuda.get_device_capability(0)})" if DEVICE == "cuda" else ""))

    # ---- batched torch, one cdist call per chunk of candidates ----
    lib_sel = selected[:n_lib].astype(np.float32)
    pred_sel = selected[n_lib:].astype(np.float32)
    target_t = torch.tensor(target.astype(np.float32), device=DEVICE)

    # A full (n_candidates, n_pred, n_lib) dense distance tensor would be
    # 100*2000*20000*4 bytes = 16 GB -- OOMs outright on CPU (confirmed: this
    # exact unchunked version was SIGKILLed) and GPU memory is typically
    # SMALLER than this node's system RAM, so chunking matters even more here.
    # If this OOMs on GPU (torch.cuda.OutOfMemoryError), reduce chunk_b -- try
    # 1 before concluding the approach doesn't fit.
    chunk_b = 5

    def make_chunk_tensors(idx):
        lib_states = np.stack([np.stack([cand_cols[f"c{i}"][:n_lib], lib_sel], axis=1) for i in idx])
        pred_states = np.stack([np.stack([cand_cols[f"c{i}"][n_lib:], pred_sel], axis=1) for i in idx])
        return (torch.tensor(lib_states, dtype=torch.float32, device=DEVICE),
                torch.tensor(pred_states, dtype=torch.float32, device=DEVICE))

    if DEVICE == "cuda":
        lib_states_t, pred_states_t = make_chunk_tensors(range(0, min(chunk_b, n_candidates)))
        _ = batched_ranking_simplex_torch(lib_states_t, pred_states_t, target_t, Tp)
        torch.cuda.synchronize()

    try:
        t0 = time.perf_counter()
        preds_chunks = []
        for start in range(0, n_candidates, chunk_b):
            idx = range(start, min(start + chunk_b, n_candidates))
            lib_states_t, pred_states_t = make_chunk_tensors(idx)
            preds_chunks.append(batched_ranking_simplex_torch(lib_states_t, pred_states_t, target_t, Tp))
        preds_b = torch.cat(preds_chunks, dim=0)
        if DEVICE == "cuda":
            torch.cuda.synchronize()  # CUDA calls are async -- must drain the queue before stopping the clock
        t_torch = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at chunk_b={chunk_b} on {DEVICE} -- rerun with a smaller chunk_b (try 1) before concluding anything.")
        raise
    preds_b = preds_b.cpu()
    print(f"Torch batched, chunk_b={chunk_b} ({-(-n_candidates // chunk_b)} chunked calls), {n_candidates} candidates: {t_torch:.3f}s")
    print(f"Speedup: {t_ref/t_torch:.2f}x")

    # pyEDM.Simplex's Predictions array carries one extra LEADING boundary row
    # (Time = pred_start, Predictions = NaN -- a padding artifact, not a real
    # prediction) before its `n_pred` real, Tp-shifted predictions. Confirmed
    # by inspecting out['Time']/out['Predictions'] directly: row 0 is the only
    # NaN-prediction row, and dropping it aligns 1:1 with this script's own
    # (unpadded) torch output. Comparing without this offset silently compares
    # misaligned rows (max diff ~1.0, not a real discrepancy).
    #
    # A handful of individual rows (typically <1%) can still show a larger
    # (e.g. 0.01-0.1) diff even when correctly aligned: with a smooth,
    # multi-period signal, many (candidate, selected) state-space points sit
    # near-duplicate distances apart, so float32 (torch) vs pyEDM's internal
    # float64 precision can pick a DIFFERENT but similarly-close k-th
    # neighbor at a near-tie, producing an occasional larger single-row
    # difference despite the underlying algorithm being correct -- confirmed
    # by a small (n_lib=30) hand-checked case matching to 1e-5 exactly, and
    # by this same full-scale check with a different candidate column
    # showing median diff 2.5e-05 with only 8/2000 rows over 0.01. Judge
    # correctness by the MEDIAN, not the max, for this reason.
    all_diffs = []
    for i in range(n_candidates):
        a, b = ref_preds[i][1:], preds_b[i].numpy()
        assert len(a) == len(b), f"still misaligned: {len(a)} vs {len(b)}"
        all_diffs.append(np.abs(a - b))
    all_diffs = np.concatenate(all_diffs)
    print(f"Diff vs pyEDM.Simplex across {len(all_diffs)} points: "
          f"median={np.nanmedian(all_diffs):.2e}  mean={np.nanmean(all_diffs):.2e}  "
          f"max={np.nanmax(all_diffs):.2e}  frac>0.01={(all_diffs > 0.01).mean():.4f}")


if __name__ == "__main__":
    main()
