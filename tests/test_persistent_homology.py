'''exp/spike-conversion-general: the dependency-free Rips persistence in
scripts/src/mde_gmn/persistent_homology.py, on clouds with known homology.

  * a noisy circle has exactly one long-lived H1 bar, born near the sample
    spacing and dying near sqrt(3) x radius (the inscribed equilateral
    triangle);
  * two well-separated blobs have one long H0 bar (the gap) and no long H1;
  * a figure-eight has two long H1 bars;
  * Wasserstein distance: 0 to itself, symmetric, and the circle is farther
    from the blobs than from a second circle sample;
  * H1 agrees with an independent brute-force rank computation (Z/2 ranks of
    the boundary matrices at each scale) on a small random cloud.

Run: python3 tests/test_persistent_homology.py
'''
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'src'))
from mde_gmn import persistent_homology as PH  # noqa: E402


def _circle(n, r=1.0, noise=0.02, seed=0, center=(0, 0)):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)]) + center + rng.normal(0, noise, (n, 2))


def test_circle_one_h1():
    X = _circle(300)
    L = X[PH.maxmin_landmarks(X, 60)]
    h0, h1 = PH.rips_diagrams(L)
    pers = np.sort(h1[:, 1] - h1[:, 0])[::-1]
    assert pers[0] > 1.0, pers[:3]
    assert len(pers) == 1 or pers[1] < 0.2, pers[:3]
    assert abs(h1[np.argmax(h1[:, 1] - h1[:, 0]), 1] - np.sqrt(3)) < 0.15
    assert h0[:, 1].max() < 0.3


def test_blobs_h0_gap_no_h1():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (60, 2)), rng.normal((5, 0), 0.1, (60, 2))])
    h0, h1 = PH.rips_diagrams(X[PH.maxmin_landmarks(X, 50)])
    d = np.sort(h0[:, 1])[::-1]
    assert d[0] > 4.0 and d[1] < 0.6, d[:3]
    assert len(h1) == 0 or (h1[:, 1] - h1[:, 0]).max() < 0.3


def test_figure_eight_two_h1():
    X = np.vstack([_circle(200, seed=2), _circle(200, seed=3, center=(2.0, 0))])
    h0, h1 = PH.rips_diagrams(X[PH.maxmin_landmarks(X, 80)])
    pers = np.sort(h1[:, 1] - h1[:, 0])[::-1]
    assert pers[0] > 0.8 and pers[1] > 0.8 and (len(pers) < 3 or pers[2] < 0.3), pers[:4]


def test_wasserstein_properties():
    A = PH.rips_diagrams(_circle(200, seed=4)[PH.maxmin_landmarks(_circle(200, seed=4), 50)])[1]
    Xb = _circle(200, seed=5)
    B = PH.rips_diagrams(Xb[PH.maxmin_landmarks(Xb, 50)])[1]
    rng = np.random.default_rng(6)
    Xc = np.vstack([rng.normal(0, 0.1, (60, 2)), rng.normal((3, 0), 0.1, (60, 2))])
    Cd = PH.rips_diagrams(Xc[PH.maxmin_landmarks(Xc, 50)])[1]
    assert PH.wasserstein(A, A) < 1e-12
    assert abs(PH.wasserstein(A, B) - PH.wasserstein(B, A)) < 1e-9
    assert PH.wasserstein(A, Cd) > 3 * PH.wasserstein(A, B)
    assert PH.wasserstein(np.empty((0, 2)), np.array([[0.0, 2.0]])) == 1.0


def _gf2_rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = np.flatnonzero(M[r:, c])
        if len(piv) == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        for i in np.flatnonzero(M[:, c]):
            if i != r:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def _betti1_bruteforce(D, s):
    n = len(D)
    edges = [(i, j) for i, j in itertools.combinations(range(n), 2) if D[i, j] <= s]
    tris = [t for t in itertools.combinations(range(n), 3)
            if max(D[t[0], t[1]], D[t[0], t[2]], D[t[1], t[2]]) <= s]
    eidx = {e: k for k, e in enumerate(edges)}
    d1 = np.zeros((n, len(edges)), dtype=np.uint8)
    for k, (i, j) in enumerate(edges):
        d1[i, k] = d1[j, k] = 1
    d2 = np.zeros((len(edges), len(tris)), dtype=np.uint8)
    for k, (a, b, c) in enumerate(tris):
        for e in ((a, b), (a, c), (b, c)):
            d2[eidx[e], k] = 1
    r1 = _gf2_rank(d1) if len(edges) else 0
    r2 = _gf2_rank(d2) if len(tris) else 0
    return len(edges) - r1 - r2


def test_h1_against_bruteforce_ranks():
    rng = np.random.default_rng(7)
    X = np.vstack([_circle(9, noise=0.05, seed=8), rng.normal(0, 0.6, (5, 2))])
    D = squareform(pdist(X))
    _, h1 = PH.rips_diagrams(X)
    for s in np.linspace(0.05, PH.enclosing_radius(D) - 1e-9, 25):
        expected = _betti1_bruteforce(D, s)
        got = int(PH.betti_curve(h1, [s])[0])
        assert got == expected, (s, got, expected)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('ALL PASSED')
