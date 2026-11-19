'''exp/spike-conversion-general: the closed-form event-to-continuous operators
(scripts/src/mde_gmn/universal_point_process_embeddings.py).

Checks, each against an independent computation rather than the module's own
recursion:
  * Gamma filter: grid samples equal the analytic convolution sum_k g_m(t - t_k)
    exactly, for any event train; the single-event response is continuous
    (g_m(0) = 0) and C^2-smooth for m = 3 (no detectable jumps), while TSLS
    jumps at every event.
  * Strict causality for every operator: changing events after row n0 leaves
    rows <= n0 bit-identical.
  * Legendre memory: matches a fine-step ODE integration, and decodes a
    delayed copy of a smooth input (its defining property).
  * Fading log-signature: equals the log of the signature of the explicitly
    re-weighted path computed from scratch by Chen products over all its
    segments (no recursion), satisfies the level-2 shuffle identity (it is a
    genuine signature), and the Levy area separates "event early in the
    window" from "event late in the window" -- which the fading count alone
    cannot.
  * Takens-lag selection on series with known answers; observability filter.

Run: python3 tests/test_universal_embeddings.py   (no pytest in this venv)
'''
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'src'))
from mde_gmn import universal_point_process_embeddings as U  # noqa: E402

DT = 0.025


def _random_train(T=1500, C=4, p=0.03, seed=0):
    rng = np.random.default_rng(seed)
    c = (rng.random((T, C)) < p).astype(float)
    c[rng.random((T, C)) < 0.003] += 1.0          # a few double events
    return c


def test_gamma_exact_against_convolution():
    counts = _random_train()
    for order in (0, 2, 3):
        for tau in (0.1, 0.5):
            y = U.gamma_filter(counts, DT, tau, order)
            T = counts.shape[0]
            t = np.arange(T) * DT
            for ch in range(counts.shape[1]):
                ev = np.flatnonzero(counts[:, ch])
                ref = np.zeros(T)
                for k in ev:
                    ref += counts[k, ch] * U.gamma_kernel(t - t[k], tau, order)
                assert np.allclose(y[:, ch], ref, rtol=1e-9, atol=1e-9), (order, tau, ch)
    # unit area: a single event integrates to 1 event
    c = np.zeros((4000, 1)); c[10, 0] = 1
    y = U.gamma_filter(c, DT, 0.25, 3)
    assert abs(y.sum() * DT - 1.0) < 1e-3


def test_gamma_smoothness_vs_tsls():
    c = np.zeros((400, 1)); c[100, 0] = 1
    y = U.gamma_filter(c, DT, 0.25, 3)[:, 0]
    assert y[100] == 0.0                      # g_m(0) = 0: continuous at the event
    # m=3: first and second differences are both small at the event onset;
    # for m=0 (a plain exponential) the onset step IS the largest step
    d1 = np.diff(y); d2 = np.diff(y, 2)
    assert abs(d1[100]) < 0.05 * np.abs(d1).max()
    assert abs(d2[99]) < 0.1 * np.abs(d2).max()
    y0 = U.gamma_filter(c, DT, 0.25, 0)[:, 0]
    assert abs(np.diff(y0)[99]) == np.abs(np.diff(y0)).max()
    # the sawtooth jumps at every event; the gamma output never moves by more
    # than a fraction of its spread in one sample
    import cleaned_helper_functions as tsls
    counts = _random_train(C=1, p=0.02)
    t = np.arange(len(counts)) * DT
    saw = tsls.compute_time_since_previous_spike(t[counts[:, 0] > 0], t)
    saw = saw[~np.isnan(saw)]
    g = U.gamma_filter(counts, DT, 0.25, 3)
    assert U.max_step_ratio(g)[0] < 0.5 < U.max_step_ratio(saw[:, None])[0], (
        U.max_step_ratio(g), U.max_step_ratio(saw[:, None]))
    assert U.roughness(g)[0] < 0.2 < U.roughness(saw[:, None])[0]


def _all_operators():
    return {
        'gamma3': lambda c: U.gamma_filter(c, DT, 0.25, 3),
        'gamma_ms_log': lambda c: U.log_compress(U.multiscale_gamma(c, DT, [0.1, 0.5], 3),
                                                 np.full(c.shape[1], 2.0)),
        'lmu': lambda c: U.legendre_memory(c, DT, 1.0, 4),
        'lmu_zoh': lambda c: U.legendre_memory(c, DT, 1.0, 4, discretization='zoh'),
        'lmu_pre': lambda c: U.legendre_memory(c, DT, 1.0, 4, presmooth_order=2),
        'logsig': lambda c: U.fading_log_signature(c, DT, 1.0, depth=3),
        'logsig_pre': lambda c: U.fading_log_signature(c, DT, 1.0, depth=3, presmooth_order=3),
    }


def test_strict_causality():
    counts = _random_train(T=800)
    n0 = 500
    alt = counts.copy()
    rng = np.random.default_rng(5)
    alt[n0 + 1:] = (rng.random(alt[n0 + 1:].shape) < 0.2).astype(float)
    for name, op in _all_operators().items():
        a, b = op(counts), op(alt)
        assert np.array_equal(a[:n0 + 1], b[:n0 + 1]), name
        assert not np.array_equal(a[n0 + 1:], b[n0 + 1:]), name
    # an event AT row n0 must reach row n0 (no artificial one-row delay) for
    # every impulse-form operator with an instantaneous path
    single = np.zeros((50, 1)); single[20, 0] = 1
    assert U.legendre_memory(single, DT, 1.0, 3)[20, 0, 0] != 0
    assert U.fading_log_signature(single, DT, 1.0, 2)[20, 0, 0] != 0
    assert U.legendre_memory(single, DT, 1.0, 3)[19, 0, 0] == 0


def test_lmu_matches_ode_and_decodes_delay():
    D, theta = 6, 1.0
    A, B = U.lmu_matrices(D)
    # smooth input given as a marked train with weight u(t) dt per bin
    T = 1200
    t = np.arange(T) * DT
    u = np.sin(2 * np.pi * t / 2.0) + 0.5 * np.cos(2 * np.pi * t / 3.1)
    u[0] = 0.0      # bin 0 covers (-dt, 0], before the ODE's t=0 start
    m = U.legendre_memory((u * DT)[:, None], DT, theta, D, discretization='zoh')[:, 0, :]
    # independent: fine-step integration of theta m' = A m + B u(t), with u
    # held over each bin (the ZOH reading)
    def rhs(s, z):
        k = min(int(np.ceil(s / DT - 1e-12)), T - 1)
        return (A @ z + B * u[k]) / theta
    sol = solve_ivp(rhs, (0, t[400]), np.zeros(D), t_eval=t[:401], max_step=DT / 4,
                    rtol=1e-8, atol=1e-10)
    assert np.allclose(m[:401], sol.y.T, atol=2e-3), np.abs(m[:401] - sol.y.T).max()
    # defining property: decode u(t - s*theta) from m(t)
    for s in (0.25, 0.5, 0.75):
        P = U.shifted_legendre(D, [s])[0]
        lag = int(round(s * theta / DT))
        dec = m[200:] @ P
        ref = u[200 - lag:T - lag]
        assert np.corrcoef(dec, ref)[0, 1] > 0.98, s


def _sig_from_scratch(segments, depth):
    '''Signature of a piecewise-linear path in R^2 given its segment vectors,
    by Chen products of exp(segment) from the identity.'''
    S = (np.zeros((1, 2)), np.zeros((1, 2, 2)), np.zeros((1, 2, 2, 2)))
    for v in segments:
        seg = U._tensor_exp_seg(np.array([v[0]]), np.array([v[1]]), 3)
        S = U._chen(S, seg, 3)
    return S


def test_logsig_matches_explicit_reweighted_path():
    W = 0.5
    counts = np.zeros((60, 1))
    counts[[5, 17, 18, 40], 0] = [1, 1, 2, 1]
    out, sigs = U.fading_log_signature(counts, DT, W, depth=3, return_signature=True)
    lam = math.exp(-DT / W)
    for n in (17, 41, 59):
        # explicit re-weighted path Y_t, t = row n: each bin k contributes a
        # straight time segment of length (1-lam)*lam^(n-k) (time in units of
        # W) and then, if it had events, a jump of c_k * lam^(n-k)
        segs = []
        for k in range(n + 1):
            wk = lam ** (n - k)
            segs.append((wk * (1 - lam), 0.0))
            if counts[k, 0]:
                segs.append((0.0, counts[k, 0] * wk))
        S = _sig_from_scratch(segs, 3)
        L1, L2, L3 = U._tensor_log(S, 3)
        ref = [L1[0, 1], L2[0, 0, 1], L3[0, 0, 0, 1], L3[0, 0, 1, 1]]
        assert np.allclose(out[n, 0], ref, rtol=1e-9, atol=1e-12), (n, out[n, 0], ref)
        # group-like: the symmetric part of level 2 is S1 (x) S1 / 2
        S1, S2, _ = sigs[n]
        assert np.allclose(S2 + S2.transpose(0, 2, 1), np.einsum("ci,cj->cij", S1, S1))
        # log level 2 is antisymmetric
        assert np.allclose(L2[0], -L2[0].T)


def test_logsig_area_separates_event_timing():
    W = 1.0
    n = 79
    early = np.zeros((80, 1)); early[n - 30, 0] = 1
    late = np.zeros((80, 1)); late[n - 2, 0] = 1
    # scale so the fading counts are EQUAL: then only the area can tell them apart
    lam = math.exp(-DT / W)
    early[n - 30, 0] = lam ** (-30 + 2)
    oe = U.fading_log_signature(early, DT, W, 2)[n, 0]
    ol = U.fading_log_signature(late, DT, W, 2)[n, 0]
    assert abs(oe[0] - ol[0]) < 1e-12
    # [t,N] > 0: the window's events are recent (time first, then the jump);
    # < 0: they are old (jump first, then time)
    assert ol[1] > oe[1] + 1e-3, (oe, ol)
    assert oe[1] < 0 < ol[1]


def test_embedding_lag_known_cases():
    t = np.arange(8000)
    x = (np.sin(2 * np.pi * t / 40.0)                  # period 40 rows
         + 0.1 * np.random.default_rng(2).normal(size=len(t)))
    lag, rule = U.embedding_lag(x, max_lag=30, method='ami')
    assert rule == 'ami' and 8 <= lag <= 12, (lag, rule)  # ~ quarter period
    rng = np.random.default_rng(1)
    phi = math.exp(-1 / 10.0)                          # AR(1), 1/e lag = 10
    e = rng.normal(size=60000)
    y = np.zeros_like(e)
    for i in range(1, len(e)):
        y[i] = phi * y[i - 1] + e[i]
    lag, rule = U.embedding_lag(y, max_lag=40, method='acf')
    assert rule == 'acf' and 9 <= lag <= 11, (lag, rule)


def test_observability_filter_and_timescales():
    c = np.zeros((1000, 3))
    c[::10, 0] = 1        # 100 events
    c[::50, 1] = 1        # 20 events
    c[::5, 2] = 1         # 200 events, but only 100 before row 500
    keep, n = U.observable_channels(c, 500, 50)
    assert list(keep) == [0, 2] and list(n) == [50, 10, 100]
    assert abs(U.median_inter_event_interval(c, DT) - 10 * DT) < 1e-12


def test_reductions():
    st = np.random.default_rng(0).normal(size=(100, 3, 4))
    r = U.reduce_to_scalar(st, 'lmu_norm')
    assert np.allclose(r, np.sqrt(1e-18 + (st ** 2 / np.array([1, 3, 5, 7])).sum(2)))
    rms = U.coordinate_rms(st, 50)
    r2 = U.reduce_to_scalar(st, 'rms_norm', rms=rms)
    assert r2.shape == (100, 3) and np.all(r2 > 0)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('ALL PASSED')
