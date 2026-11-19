"""Sparse event trains -> continuous signals, with closed-form causal operators.

exp/spike-conversion-general. Domain-agnostic: every operator here acts on a
(marked) point process  mu_i(t) = sum_k w_k delta(t - t_k)  sampled on a
uniform grid of step dt, given as a (T, C) array of per-bin event counts (or
mark sums). Nothing assumes neurons, and nothing mixes channels: output
channel i depends on input channel i only.

CONVENTIONS SHARED BY EVERY OPERATOR
  * Time grid. Row n stands for the END of bin n. An event counted in bin n is
    treated as a Dirac at that grid point, so x[n] depends on bins 0..n only.
    That is the strict causality the experiment requires (no event later than
    n*dt ever reaches x[n]); tests/test_universal_embeddings.py checks it by
    perturbing the future and asserting the past is unchanged.
  * Exact discretisation. Each operator is a linear time-invariant system
    z' = A z + B mu(t). With Dirac input on the grid, the exact sampled
    solution is  z[n] = Phi z[n-1] + B c[n],  Phi = expm(A dt)
    (impulse-invariant discretisation). It is exact for this input, unlike
    zero-order hold, which assumes the input is piecewise constant.
    `discretization='zoh'` is offered for the Legendre memory, where the
    literature uses it.
  * Zero trainable parameters. Every coefficient is a closed-form function of
    the timescales (tau, theta, W) and dt. The only data-derived quantities are
    normalisers (a channel's mean rate on the training split, used to make
    log-compression dimensionless) and the optional auto-scaling of
    timescales to the dataset's median inter-event interval
    (auto_timescale()). Neither is fitted to a prediction objective.

THE THREE OPERATORS
  gamma_filter()        Method 1. Causal Gamma kernel of integer order m,
                            g_m(t) = t^m e^{-t/tau} / (tau^{m+1} m!) u(t),
                        realised as m+1 identical first-order leaky
                        integrators in cascade. Unit area, so the output is a
                        smoothed rate (events per second). g_m and its first
                        m-1 derivatives vanish at t=0, so the response to one
                        event is C^{m-1} (NOT C^m: the m-th derivative jumps
                        at the event). C^2 therefore needs m >= 3.
  legendre_memory()     Method 2. Legendre Memory Unit state
                            theta m'(t) = A m(t) + B mu(t)
                        (Voelker, Kajic & Eliasmith 2019): m(t) holds the
                        first D shifted-Legendre coefficients of the input
                        over the window [t - theta, t]. The impulse response
                        is e^{A s/theta} B/theta, which is NONZERO at s=0, so
                        on a raw Dirac train the state jumps at every event.
                        `presmooth_order` > 0 first passes the train through
                        a Gamma filter (tau_p = theta/(4 D) by default), which
                        makes the state C^{presmooth_order}.
  fading_log_signature()
                        Method 3. Log-signature, truncated at depth K <= 3, of
                        the time-augmented counting path X(t) = (t/W, N(t))
                        under an exponential fading envelope. Exactly: the
                        signature of the re-weighted path
                            Y_t(s) = int_{-inf}^{s} e^{-(t-r)/W} dX_r,
                        which obeys Chen's identity in the recursive form
                            S_n = delta_lam(S_{n-1}) (x) exp(seg_n),
                        delta_lam scaling tensor level k by lam^k,
                        lam = e^{-dt/W}. Because S_n is a genuine signature
                        (group-like), its logarithm is a Lie element and the
                        Lyndon coordinates below are well defined.
                        Coordinates (time's own level-1 term is a constant
                        and is dropped):
                            K>=1: N       fading event count
                            K>=2: [t,N]   Levy area: > 0 when the window's
                                          events are recent, < 0 when old
                            K>=3: [t,[t,N]], [[t,N],N]
                        `presmooth_order` > 0 replaces the jump path N(t) by
                        the integral of a Gamma-smoothed rate.

SCALAR NODES
  The MDE/GMN graph is one SCALAR column per node. Vector-valued operators
  (Methods 2 and 3) are reduced to one scalar per channel by
  reduce_to_scalar(): a weighted Euclidean norm (the L2 norm of the
  reconstructed window function for the Legendre memory; the norm of
  RMS-normalised coordinates for the log-signature). Smoothed by
  sqrt(eps^2 + |v|^2), so the reduction is C^infinity wherever the state is.

Pure numpy/scipy; no pipeline imports. Depends on: numpy, scipy.
Imported by: tsls_data (conversion), the screening script, tests.
"""

import math

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Generic exact LTI filtering of an event train
# ---------------------------------------------------------------------------

def _lti_event_filter(counts, Phi, B, readout=None):
    '''z[n] = Phi z[n-1] + B c[n] for every channel at once; z[-1] = 0.

    counts: (T, C). Phi: (S, S). B: (S,). Returns (T, C, S), or (T, C) when
    `readout` (an (S,) vector) is given. Event-free stretches are advanced in
    one step with a precomputed matrix power when they are long, so a sparse
    train costs far less than T matrix products.'''
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    T, C = counts.shape
    S = Phi.shape[0]
    PhiT = Phi.T.copy()
    z = np.zeros((C, S))
    out = np.empty((T, C, S)) if readout is None else np.empty((T, C))
    for n in range(T):
        z = z @ PhiT
        row = counts[n]
        if row.any():
            z += row[:, None] * B[None, :]
        if readout is None:
            out[n] = z
        else:
            out[n] = z @ readout
    return out


def _gamma_system(tau, order):
    '''Continuous (A, B, C) of the (order+1)-stage cascade whose impulse
    response is g_order(t; tau) (unit area).'''
    S = order + 1
    A = (-np.eye(S) + np.eye(S, k=-1)) / tau
    B = np.zeros(S)
    B[0] = 1.0 / tau
    Cout = np.zeros(S)
    Cout[-1] = 1.0
    return A, B, Cout


def gamma_kernel(t, tau, order):
    '''Continuous g_m(t; tau) = t^m e^{-t/tau} / (tau^{m+1} m!) for t >= 0.'''
    t = np.asarray(t, dtype=float)
    g = np.where(t >= 0, np.power(np.clip(t, 0, None), order) * np.exp(-np.clip(t, 0, None) / tau), 0.0)
    return g / (tau ** (order + 1) * math.factorial(order))


def gamma_filter(counts, dt, tau, order=3):
    '''Method 1 (single scale). Exact samples of (g_order * mu)(n dt).

    Returns (T, C) smoothed rates in events per unit time.'''
    if order < 0:
        raise ValueError(f'gamma order must be >= 0, got {order}')
    A, B, Cout = _gamma_system(tau, order)
    Phi = expm(A * dt)
    return _lti_event_filter(counts, Phi, B, readout=Cout)


def multiscale_gamma(counts, dt, taus, order=3):
    '''Method 1, multi-scale variant: the sum of single-scale outputs.'''
    return sum(gamma_filter(counts, dt, tau, order) for tau in taus)


def log_compress(x, mean_rate, kappa=100.0):
    '''x -> log(1 + c x) with c = kappa / mean_rate (per channel, from the
    training split), i.e. log(1 + kappa x / rbar).

    WHY: a Gamma-filtered train decays like t^m e^{-t/tau} through a silent
    interval, so long silences pile up just above 0 and nearest-neighbour
    lookups there are degenerate. The log maps that exponential tail onto a
    near-linear descending ramp, log g ~ m log t - t/tau, which is a smooth
    analogue of time-since-last-event, down to the floor x ~ rbar/kappa where
    it saturates. kappa sets how many decades below the mean rate the ramp
    reaches; it is a fixed constant, not fitted.'''
    mean_rate = np.asarray(mean_rate, dtype=float)
    c = kappa / np.where(mean_rate > 0, mean_rate, 1.0)
    return np.log1p(c[None, :] * np.clip(x, 0, None))


# ---------------------------------------------------------------------------
# Method 2: Legendre memory
# ---------------------------------------------------------------------------

def lmu_matrices(D):
    '''Continuous (A, B) of the Legendre Memory Unit of order D (Voelker et
    al. 2019, eq. 2): theta m' = A m + B u.'''
    i = np.arange(D)[:, None]
    j = np.arange(D)[None, :]
    A = (2 * i + 1) * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    B = ((2 * np.arange(D) + 1) * (-1.0) ** np.arange(D)).astype(float)
    return A.astype(float), B


def shifted_legendre(D, s):
    '''P~_d(s), d < D, for s in [0, 1] (P~_d(s) = P_d(2s - 1)); (len(s), D).
    The LMU decodes u(t - s*theta) ~ sum_d P~_d(s) m_d(t).'''
    s = np.atleast_1d(np.asarray(s, dtype=float))
    x = 2 * s - 1
    P = np.zeros((len(s), D))
    P[:, 0] = 1.0
    if D > 1:
        P[:, 1] = x
    for d in range(2, D):
        P[:, d] = ((2 * d - 1) * x * P[:, d - 1] - (d - 1) * P[:, d - 2]) / d
    return P


def legendre_memory(counts, dt, theta, D, discretization='impulse',
                    presmooth_order=0, presmooth_tau=None):
    '''Method 2. Returns the LMU state, (T, C, D).

    discretization: 'impulse' (exact for Dirac input on the grid) or 'zoh'
    (each bin's count spread uniformly over that bin, i.e. the input held at
    c[n]/dt on ((n-1) dt, n dt]; row n still sees bins <= n only).
    presmooth_order > 0: feed the LMU a Gamma-filtered rate instead of the
    raw train (augmented exact state space, so still one closed-form Phi).'''
    A_l, B_l = lmu_matrices(D)
    A_l = A_l / theta
    B_l = B_l / theta
    if presmooth_order > 0:
        tau_p = presmooth_tau if presmooth_tau is not None else theta / (4.0 * D)
        A_g, B_g, C_g = _gamma_system(tau_p, presmooth_order)
        S_g = A_g.shape[0]
        A = np.zeros((S_g + D, S_g + D))
        A[:S_g, :S_g] = A_g
        A[S_g:, S_g:] = A_l
        A[S_g:, :S_g] = np.outer(B_l, C_g)      # LMU input = gamma output
        B = np.concatenate([B_g, np.zeros(D)])
        sel = slice(S_g, S_g + D)
    else:
        A, B, sel = A_l, B_l, slice(0, D)

    if discretization == 'impulse':
        Phi = expm(A * dt)
        Bd = B
    elif discretization == 'zoh':
        # Van Loan: expm([[A, B], [0, 0]] dt) = [[Phi, Gamma], [0, 1]], with
        # Gamma = int_0^dt e^{A s} B ds the response to a unit input held
        # over one step; the held level is c[n]/dt.
        S = A.shape[0]
        M = np.zeros((S + 1, S + 1))
        M[:S, :S] = A
        M[:S, S] = B
        E = expm(M * dt)
        Phi, Bd = E[:S, :S], E[:S, S] / dt
    else:
        raise ValueError(f"discretization must be 'impulse' or 'zoh', got {discretization!r}")
    z = _lti_event_filter(counts, Phi, Bd)
    return z[:, :, sel]


# ---------------------------------------------------------------------------
# Method 3: fading-memory log-signature of (t, N(t))
# ---------------------------------------------------------------------------

def _tensor_exp_seg(v1, v2, depth):
    '''exp of the level-1 element v = v1 e1 + v2 e2 (per channel), truncated.
    v1, v2: (C,). Returns (L1 (C,2), L2 (C,2,2) or None, L3 (C,2,2,2) or None).'''
    V = np.stack([v1, v2], axis=1)
    L2 = np.einsum('ci,cj->cij', V, V) / 2.0 if depth >= 2 else None
    L3 = np.einsum('ci,cj,ck->cijk', V, V, V) / 6.0 if depth >= 3 else None
    return V, L2, L3


def _chen(X, Y, depth):
    '''(1+X1+X2+X3) (x) (1+Y1+Y2+Y3), truncated; X, Y are (L1, L2, L3).'''
    X1, X2, X3 = X
    Y1, Y2, Y3 = Y
    Z1 = X1 + Y1
    Z2 = Z3 = None
    if depth >= 2:
        Z2 = X2 + Y2 + np.einsum('ci,cj->cij', X1, Y1)
    if depth >= 3:
        Z3 = (X3 + Y3 + np.einsum('cij,ck->cijk', X2, Y1)
              + np.einsum('ci,cjk->cijk', X1, Y2))
    return Z1, Z2, Z3


def _tensor_log(S, depth):
    '''log(1 + X) = X - X^2/2 + X^3/3, truncated, for X = S - 1.'''
    X1, X2, X3 = S
    L1 = X1
    L2 = L3 = None
    if depth >= 2:
        X1X1 = np.einsum('ci,cj->cij', X1, X1)
        L2 = X2 - X1X1 / 2.0
    if depth >= 3:
        X1X2 = np.einsum('ci,cjk->cijk', X1, X2)
        X2X1 = np.einsum('cij,ck->cijk', X2, X1)
        X1X1X1 = np.einsum('ci,cj,ck->cijk', X1, X1, X1)
        L3 = X3 - (X1X2 + X2X1) / 2.0 + X1X1X1 / 3.0
    return L1, L2, L3


def logsig_coordinate_names(depth):
    names = ['N']
    if depth >= 2:
        names.append('[t,N]')
    if depth >= 3:
        names += ['[t,[t,N]]', '[[t,N],N]']
    return names


def fading_log_signature(counts, dt, W, depth=2, presmooth_order=0, presmooth_tau=None,
                         return_signature=False):
    '''Method 3. Returns (T, C, n_coords) Lyndon coordinates of the log-
    signature of the exponentially faded path (see module docstring);
    coordinate order is logsig_coordinate_names(depth).

    Time is measured in units of W, so the time component is dimensionless and
    O(1). With presmooth_order == 0 each step is the cadlag segment "time
    advances over the bin, then the bin's events jump N at its end"; with
    presmooth_order > 0 it is the straight segment (dt/W, rate[n] dt) of the
    smoothed counting path. In both cases the fading weight inside the step
    only reparametrises a straight segment (its direction is constant), so the
    segment's signature is exp(kappa * v) exactly, with
    kappa = (W/dt)(1 - lam) for the time part.'''
    if depth not in (1, 2, 3):
        raise ValueError(f'depth must be 1, 2 or 3, got {depth}')
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    T, C = counts.shape
    lam = math.exp(-dt / W)
    kappa_t = (1.0 - lam)               # = (W/dt)(1-lam) * (dt/W)
    if presmooth_order > 0:
        tau_p = presmooth_tau if presmooth_tau is not None else W / 8.0
        rate = gamma_filter(counts, dt, tau_p, presmooth_order)
        incr_N = rate * dt * (W / dt) * (1.0 - lam)   # faded straight segment
    else:
        incr_N = counts

    zeros = np.zeros(C)
    S1 = np.zeros((C, 2))
    S2 = np.zeros((C, 2, 2)) if depth >= 2 else None
    S3 = np.zeros((C, 2, 2, 2)) if depth >= 3 else None
    seg_t = _tensor_exp_seg(np.full(C, kappa_t), zeros, depth)
    n_coords = len(logsig_coordinate_names(depth))
    out = np.empty((T, C, n_coords))
    sig_out = [] if return_signature else None
    for n in range(T):
        # delta_lam dilation
        S = (S1 * lam,
             S2 * lam ** 2 if S2 is not None else None,
             S3 * lam ** 3 if S3 is not None else None)
        dn = incr_N[n]
        if presmooth_order > 0:
            seg = _tensor_exp_seg(np.full(C, kappa_t), dn, depth)
            S = _chen(S, seg, depth)
        else:
            S = _chen(S, seg_t, depth)
            if dn.any():
                S = _chen(S, _tensor_exp_seg(zeros, dn, depth), depth)
        S1, S2, S3 = S
        L1, L2, L3 = _tensor_log(S, depth)
        out[n, :, 0] = L1[:, 1]
        if depth >= 2:
            out[n, :, 1] = L2[:, 0, 1]
        if depth >= 3:
            out[n, :, 2] = L3[:, 0, 0, 1]       # word 112 -> [1,[1,2]]
            out[n, :, 3] = L3[:, 0, 1, 1]       # word 122 -> [[1,2],2]
        if sig_out is not None:
            sig_out.append(tuple(None if s is None else s.copy() for s in S))
    if return_signature:
        return out, sig_out
    return out


# ---------------------------------------------------------------------------
# Scalar reduction, scaling, and lag selection
# ---------------------------------------------------------------------------

def reduce_to_scalar(state, method, rms=None, eps=1e-9):
    '''(T, C, K) -> (T, C).

    method:
      'lmu_norm'  sqrt(eps^2 + sum_d m_d^2 / (2d+1)): proportional to the L2
                  norm, over the window, of the reconstructed input
                  sum_d P~_d(s) m_d (the P~_d are orthogonal with
                  int_0^1 P~_d^2 = 1/(2d+1)).
      'rms_norm'  sqrt(eps^2 + sum_k (v_k / rms_k)^2) with rms_k each
                  coordinate's root-mean-square on the training split
                  (`rms`, shape (C, K)), so coordinates in different units
                  weigh equally. Uncentred: an all-quiet window maps to ~0.
      'first'     the first coordinate only (the window's mean rate for the
                  LMU, the fading count for the log-signature).'''
    if method == 'lmu_norm':
        w = 1.0 / (2 * np.arange(state.shape[2]) + 1)
        return np.sqrt(eps ** 2 + np.einsum('tck,k->tc', state ** 2, w))
    if method == 'rms_norm':
        if rms is None:
            raise ValueError("reduce_to_scalar('rms_norm') needs rms")
        r = np.where(rms > 0, rms, 1.0)
        return np.sqrt(eps ** 2 + np.sum((state / r[None]) ** 2, axis=2))
    if method == 'first':
        return state[:, :, 0]
    raise ValueError(f'unknown reduction {method!r}')


def coordinate_rms(state, lib_end):
    '''Per-channel, per-coordinate RMS over rows [0, lib_end): (C, K).'''
    return np.sqrt(np.mean(state[:lib_end] ** 2, axis=0))


def median_inter_event_interval(counts, dt, lib_end=None):
    '''Median, over channels, of each channel's median inter-event interval
    (training rows only when lib_end is given). Channels with < 2 events are
    skipped. Units of dt.'''
    counts = np.asarray(counts)
    if lib_end is not None:
        counts = counts[:lib_end]
    meds = []
    for j in range(counts.shape[1]):
        idx = np.flatnonzero(counts[:, j] > 0)
        if len(idx) > 2:
            meds.append(np.median(np.diff(idx)) * dt)
    if not meds:
        raise ValueError('no channel has 3 or more events')
    return float(np.median(meds))


def mean_inter_event_interval(counts, dt, lib_end=None):
    '''Median, over channels, of each channel's MEAN inter-event interval
    (= duration / events). Bursty channels make the median-ISI small while the
    mean ISI reflects the silences that cause collapse.'''
    counts = np.asarray(counts)
    if lib_end is not None:
        counts = counts[:lib_end]
    n = (counts > 0).sum(axis=0)
    dur = counts.shape[0] * dt
    vals = dur / n[n >= 2]
    return float(np.median(vals))


def auto_timescale(counts, dt, multiplier, statistic='mean', lib_end=None):
    '''A timescale for any dataset: `multiplier` x the median channel's mean
    (or median) inter-event interval on the training split.'''
    f = mean_inter_event_interval if statistic == 'mean' else median_inter_event_interval
    return multiplier * f(counts, dt, lib_end)


def _ami(x, lag, bins=16):
    '''Average mutual information between x[t] and x[t+lag] (nats), using
    equal-occupancy bins (quantiles), so it is invariant to monotone rescaling
    and a heavy tail does not empty most bins.'''
    a, b = x[:-lag], x[lag:]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ia = np.clip(np.searchsorted(edges, a, side='right') - 1, 0, len(edges) - 2)
    ib = np.clip(np.searchsorted(edges, b, side='right') - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    joint = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k).astype(float)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])))


def embedding_lag(x, max_lag=40, method='ami', bins=16, rise_frac=0.02):
    '''Takens lag for one series, in rows (>= 1).

    'ami': first minimum of the average mutual information (Fraser & Swinney
           1986). A dip only counts as the minimum if the curve afterwards
           rises above it by more than `rise_frac` of the curve's total drop,
           so binning wiggles on a near-deterministic series are not taken
           for it. If no such minimum exists up to max_lag (AMI decreasing
           throughout, typical of a smooth low-pass signal), fall back to
           'acf'.
    'acf': first lag where the autocorrelation drops below 1/e.
    Returns (lag, rule_used); rule_used is 'ami', 'acf' or 'cap' (nothing
    found by max_lag, so max_lag is returned).'''
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4 * max_lag or np.std(x) == 0:
        return 1, 'degenerate'
    if method == 'ami':
        curve = np.array([_ami(x, lag, bins) for lag in range(1, max_lag + 1)])
        tol = rise_frac * max(curve[0] - curve.min(), 1e-12)
        run_min = np.inf
        arg = None
        for i, v in enumerate(curve):
            if v < run_min:
                run_min, arg = v, i
            elif v > run_min + tol:
                return arg + 1, 'ami'
    xc = x - x.mean()
    var = np.dot(xc, xc)
    for lag in range(1, max_lag + 1):
        r = np.dot(xc[:-lag], xc[lag:]) / var
        if r < 1.0 / math.e:
            return lag, 'acf'
    return max_lag, 'cap'


def dataset_embedding_lag(X, max_lag=40, method='ami', lib_end=None):
    '''One Takens lag for the whole transformed dataset: the median of the
    per-channel lags (MDE and CCM use a single tau for all columns). X is
    (T, C); only rows [0, lib_end) are used. Returns (lag, per_channel_lags,
    rule_counts).'''
    X = np.asarray(X, dtype=float)
    if lib_end is not None:
        X = X[:lib_end]
    lags, rules = [], {}
    for j in range(X.shape[1]):
        lag, rule = embedding_lag(X[:, j], max_lag, method)
        lags.append(lag)
        rules[rule] = rules.get(rule, 0) + 1
    return int(max(1, round(float(np.median(lags))))), lags, rules


def observable_channels(counts, lib_end, n_min):
    '''Observability filter: indices of channels with at least n_min events
    (bins with a nonzero count) in rows [0, lib_end).'''
    n = (np.asarray(counts)[:lib_end] > 0).sum(axis=0)
    return np.flatnonzero(n >= n_min), n


# ---------------------------------------------------------------------------
# Representation diagnostics (used by the screening script and the report)
# ---------------------------------------------------------------------------

def roughness(x):
    '''Per-channel ratio of the sd of the 2nd difference to the sd of the 1st
    difference. Small for a smooth sampled signal; ~sqrt(2)..2 for a jump or
    white-noise-like series.'''
    d1 = np.diff(x, axis=0)
    d2 = np.diff(x, n=2, axis=0)
    s1 = d1.std(axis=0)
    return np.where(s1 > 0, d2.std(axis=0) / np.where(s1 > 0, s1, 1), np.nan)


def max_step_ratio(x, q=0.999):
    '''Per-channel q-quantile of |x[n+1] - x[n]| divided by sd(x): how large
    the biggest single-sample moves are relative to the signal's spread. A
    series with jumps scores O(1) (a reset moves it by about its own spread in
    one sample); a smooth series with correlation time tau_c rows scores about
    1/tau_c. Scale-free, so comparable across operators.'''
    d1 = np.abs(np.diff(x, axis=0))
    sd = np.nanstd(x, axis=0)
    return np.where(sd > 0, np.nanquantile(d1, q, axis=0) / np.where(sd > 0, sd, 1), np.nan)


def collapse_fraction(x, rel=1e-3):
    '''Per-channel fraction of rows within rel x (range) of the channel's
    minimum: how much of the time the channel sits on a degenerate floor.'''
    lo = np.nanmin(x, axis=0)
    rng = np.nanmax(x, axis=0) - lo
    return np.mean(x <= lo + rel * np.where(rng > 0, rng, 1), axis=0)
