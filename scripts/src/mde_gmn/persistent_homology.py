"""Vietoris-Rips persistent homology in dimensions 0 and 1, dependency-free.

exp/spike-conversion-general. The cluster venvs carry no TDA library (no
gudhi, ripser) and must not be modified, so this is a small exact
implementation for the point-cloud sizes the experiment uses (~50-120
landmarks after max-min subsampling).

    maxmin_landmarks()  greedy farthest-point subsample; covers the cloud's
                        shape far better than a random subsample of the same
                        size, which over-represents dense regions.
    rips_diagrams()     (H0, H1) persistence pairs of the Rips filtration.
                        H0 from the minimum spanning tree (single linkage);
                        H1 by the standard Z/2 column reduction of the
                        triangle boundary matrix, with triangles restricted
                        to diameter <= max_scale. Above the enclosing radius
                        (min_i max_j d_ij) the Rips complex is a cone, so no
                        H1 class can be born there and the default max_scale
                        loses nothing; an H1 class still open at max_scale
                        is given death = max_scale.
    wasserstein()       p-Wasserstein distance between two diagrams, exact,
                        via an assignment problem with diagonal projections.
    betti_curve()       number of bars alive at each scale.

Depends on: numpy, scipy. Imported by: autonomous_eval, tests.
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def maxmin_landmarks(X, n, start=None):
    '''Indices of n farthest-point landmarks of X (rows). Deterministic: starts
    from the point farthest from the centroid unless `start` is given.'''
    X = np.asarray(X, dtype=float)
    if len(X) <= n:
        return np.arange(len(X))
    if start is None:
        start = int(np.argmax(((X - X.mean(0)) ** 2).sum(1)))
    idx = [start]
    d = np.sqrt(((X - X[start]) ** 2).sum(1))
    for _ in range(n - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.sqrt(((X - X[j]) ** 2).sum(1)))
    return np.array(idx)


def _h0(D):
    mst = minimum_spanning_tree(D).toarray()
    deaths = np.sort(mst[mst > 0])
    return np.column_stack([np.zeros(len(deaths)), deaths])


def _h1(D, max_scale):
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = D[iu, ju]
    # Edge order: by length, ties broken by index (a valid filtration order).
    order = np.lexsort((np.arange(len(w)), w))
    rank = np.empty(len(w), dtype=np.int64)
    rank[order] = np.arange(len(w))
    edge_id = np.full((n, n), -1, dtype=np.int64)
    edge_id[iu, ju] = rank
    edge_id[ju, iu] = rank
    edge_len = w[order]

    tris = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    e_ab = edge_id[tris[:, 0], tris[:, 1]]
    e_ac = edge_id[tris[:, 0], tris[:, 2]]
    e_bc = edge_id[tris[:, 1], tris[:, 2]]
    E3 = np.stack([e_ab, e_ac, e_bc], axis=1)
    tri_max_rank = E3.max(axis=1)
    tri_val = edge_len[tri_max_rank]
    keep = tri_val <= max_scale
    E3, tri_max_rank, tri_val = E3[keep], tri_max_rank[keep], tri_val[keep]
    # Triangle order: by diameter, then by its youngest edge, then by the
    # remaining edges -- any refinement of the diameter order is valid.
    srt = np.lexsort((np.sort(E3, axis=1)[:, 1], tri_max_rank, tri_val))
    E3, tri_val = E3[srt], tri_val[srt]

    # Edges that merge components are H0 deaths, never H1 births.
    mst = minimum_spanning_tree(D).tocoo()
    negative = set(int(edge_id[i, j]) for i, j in zip(mst.row, mst.col))

    pivot_owner = {}
    reduced = {}
    pairs = []
    for t in range(len(E3)):
        col = set(int(e) for e in E3[t])
        while col:
            piv = max(col)
            owner = pivot_owner.get(piv)
            if owner is None:
                break
            col ^= reduced[owner]
        if col:
            piv = max(col)
            pivot_owner[piv] = t
            reduced[t] = col
            b, d = edge_len[piv], tri_val[t]
            if d > b:
                pairs.append((b, d))
    paired = set(pivot_owner)
    for e in range(len(edge_len)):
        if e in paired or e in negative or edge_len[e] > max_scale:
            continue
        if edge_len[e] < max_scale:
            pairs.append((edge_len[e], max_scale))
    return np.array(pairs, dtype=float).reshape(-1, 2)


def enclosing_radius(D):
    return float(np.min(np.max(D, axis=1)))


def rips_diagrams(X, max_scale=None, metric='euclidean'):
    '''(H0, H1) persistence diagrams of the Rips filtration of point cloud X,
    each an (k, 2) array of (birth, death). H0 omits the one infinite bar.'''
    X = np.asarray(X, dtype=float)
    D = squareform(pdist(X, metric=metric))
    if max_scale is None:
        max_scale = enclosing_radius(D)
    return _h0(D), _h1(D, max_scale)


def wasserstein(A, B, p=1.0):
    '''Exact p-Wasserstein distance between diagrams A and B ((k, 2) arrays),
    with the L-infinity ground metric and free matching to the diagonal.'''
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    B = np.asarray(B, dtype=float).reshape(-1, 2)
    na, nb = len(A), len(B)
    if na == 0 and nb == 0:
        return 0.0
    da = (A[:, 1] - A[:, 0]) / 2.0          # L-inf distance to the diagonal
    db = (B[:, 1] - B[:, 0]) / 2.0
    big = np.zeros((na + nb, na + nb))
    if na and nb:
        big[:na, :nb] = np.maximum(np.abs(A[:, None, 0] - B[None, :, 0]),
                                   np.abs(A[:, None, 1] - B[None, :, 1])) ** p
    if na:
        big[:na, nb:] = np.inf
        big[np.arange(na), nb + np.arange(na)] = da ** p
    if nb:
        big[na:, :nb] = np.inf
        big[na + np.arange(nb), np.arange(nb)] = db ** p
    # diagonal-to-diagonal: free
    big[na:, nb:] = 0.0
    r, c = linear_sum_assignment(big)
    return float(big[r, c].sum() ** (1.0 / p))


def betti_curve(diagram, scales):
    d = np.asarray(diagram, dtype=float).reshape(-1, 2)
    s = np.asarray(scales, dtype=float)[:, None]
    return ((d[None, :, 0] <= s) & (d[None, :, 1] > s)).sum(axis=1)


def diagram_summary(diagram, min_persistence=0.0):
    d = np.asarray(diagram, dtype=float).reshape(-1, 2)
    pers = d[:, 1] - d[:, 0]
    big = pers > min_persistence
    return {
        'n_bars': int(big.sum()),
        'total_persistence': float(pers[big].sum()),
        'max_persistence': float(pers.max()) if len(pers) else 0.0,
    }
