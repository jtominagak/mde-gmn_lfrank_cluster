"""Spike train -> continuous single-unit state (exp/spike-conversion-neuro).

One function per representation. Each maps a (T, N) array of spike COUNTS per
25 ms bin to a (T, N) array of continuous states, one column per recorded unit
-- never a mixture of units (constraint 1: every network node is one physical
unit).

    'tsls'          the baseline: time since the previous spike (sawtooth),
                    via cleaned_helper_functions.compute_time_since_previous_spike.
                    Causal, but jumps back to 0 at every spike.
    'calcium'       Model A: two-stage linear cascade
                        dg/dt = -g/tau_syn + S(t)
                        dc/dt = -c/tau_ca  + g(t)        output c
    'adapt_nr'      Model B: Model A plus slow adaptation w and a Naka-Rushton
                    saturation
                        tau_w dw/dt = a c - w + b S(t)
                        y = c + lambda w
                        r = y^n / (sigma^n + y^n)        output r in [0, 1)
    'theta_osc'     Model C: spike-kicked damped oscillator
                        x'' + 2 zeta w0 x' + w0^2 x = S(t)   output x

HOW THE ODES ARE INTEGRATED. Every model is linear in its state (Model B's
saturation is applied afterwards, pointwise), driven by S(t) = sum_k
delta(t - t_k). Spikes are taken to occur at the bin times. Between two bins
the system is free, so the exact update over one bin is
    x[n] = Phi x[n-1] + B S[n],     Phi = expm(A dt)
where B is the jump a unit impulse gives the state. There is no Euler error
and no dt stability limit, whatever the time constants. The output reported
at bin n is the state just AFTER bin n's spikes. It depends on S[0..n] only,
so the transform is strictly causal: no lookahead. tests/test_spike_filters.py
pins this by changing future spikes and checking the past does not move.

SMOOTHNESS, STATED PRECISELY. With delta inputs, the first state an impulse
reaches jumps (g in A and B; dx/dt in C). The output one integration further
down is therefore continuous with a bounded derivative but a kink at each
spike: c (A), x (C), and r (B, through c and w) are C^0 and piecewise C^inf.
That is the same smoothness order as one more integration of the sawtooth,
and strictly better than the sawtooth, which jumps. It is not literally C^2;
that would need a third cascade stage. At 40 Hz sampling the delay embedding
sees samples, not derivatives, so what matters in practice is that there are
no jumps.

SIGMA (Model B) is a per-unit SCALE, not a fitted parameter: the median or 75th
percentile of the unit's 'active' drive y over the library rows only, so no
held-out row informs it. 'Active' means y > ACTIVE_FRACTION x that unit's
99th percentile. y decays exponentially and is never exactly 0 after the first
spike, so a literal y > 0 test would make sigma a statistic of decay tails.

Depends on: numpy, scipy.linalg; cleaned_helper_functions (baseline only).
Imported by: tsls_data.
"""

import numpy as np
from scipy.linalg import expm

METHODS = ('tsls', 'calcium', 'adapt_nr', 'theta_osc')
ACTIVE_FRACTION = 0.01


def _integrate_impulse_lti(spikes, A, B, dt):
    '''Exact discrete-time solution of x' = A x + B S(t), S a train of unit
    impulses at the bin times (counts allowed). spikes: (T, N). Returns the
    (T, N, k) state just after each bin's impulses, zero initial state.'''
    spikes = np.asarray(spikes, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    T, N = spikes.shape
    k = A.shape[0]
    phi_t = expm(A * dt).T
    out = np.empty((T, N, k))
    x = np.zeros((N, k))
    for n in range(T):
        x = x @ phi_t + spikes[n][:, None] * B[None, :]
        out[n] = x
    return out


def calcium_states(spikes, dt, tau_syn, tau_ca):
    '''Model A states (g, c), shape (T, N, 2). Time constants in seconds.'''
    A = [[-1.0 / tau_syn, 0.0],
         [1.0, -1.0 / tau_ca]]
    return _integrate_impulse_lti(spikes, A, [1.0, 0.0], dt)


def calcium(spikes, dt, tau_syn, tau_ca):
    '''Model A output c(t).'''
    return calcium_states(spikes, dt, tau_syn, tau_ca)[:, :, 1]


def adaptation_states(spikes, dt, tau_syn, tau_ca, tau_w, a, b):
    '''Model B linear part: states (g, c, w), shape (T, N, 3).
    tau_w dw/dt = a c - w + b S  =>  dw/dt = (a/tau_w) c - w/tau_w + (b/tau_w) S.'''
    A = [[-1.0 / tau_syn, 0.0, 0.0],
         [1.0, -1.0 / tau_ca, 0.0],
         [0.0, a / tau_w, -1.0 / tau_w]]
    return _integrate_impulse_lti(spikes, A, [1.0, 0.0, b / tau_w], dt)


def naka_rushton_sigma(y, fit_rows, stat):
    '''Per-unit sigma from rows `fit_rows` (a slice) of drive y (T, N).
    stat: 'median' | 'p75'. Units with no active rows get the overall median
    of the others (they are silent in the library; any positive scale works).'''
    q = {'median': 50.0, 'p75': 75.0}[stat]
    fit = y[fit_rows]
    sig = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        col = fit[:, j]
        top = np.percentile(col, 99.0) if col.size else 0.0
        active = col[col > ACTIVE_FRACTION * top] if top > 0 else col[:0]
        if active.size:
            sig[j] = np.percentile(active, q)
    if np.all(np.isnan(sig)):
        return np.ones(y.shape[1])
    sig[np.isnan(sig)] = np.nanmedian(sig)
    return sig


def naka_rushton(y, sigma, n):
    y = np.clip(y, 0.0, None)
    yn = y ** n
    return yn / (sigma[None, :] ** n + yn)


def adapt_nr(spikes, dt, tau_syn, tau_ca, tau_w, a, b, lam, n, sigma_stat, fit_rows):
    '''Model B output r(t) in [0, 1). Returns (r, sigma).'''
    st = adaptation_states(spikes, dt, tau_syn, tau_ca, tau_w, a, b)
    y = st[:, :, 1] + lam * st[:, :, 2]
    sigma = naka_rushton_sigma(y, fit_rows, sigma_stat)
    return naka_rushton(y, sigma, n), sigma


def theta_oscillator(spikes, dt, f0_hz, zeta):
    '''Model C output x(t): x'' + 2 zeta w0 x' + w0^2 x = S. A unit impulse
    sets x' += 1 (x itself is continuous).'''
    w0 = 2.0 * np.pi * f0_hz
    A = [[0.0, 1.0],
         [-w0 * w0, -2.0 * zeta * w0]]
    return _integrate_impulse_lti(spikes, A, [0.0, 1.0], dt)[:, :, 0]


def transform_spikes(method, spikes, time_grid, params, fit_rows):
    '''Dispatch: one method name -> (T, N) continuous array, plus a dict of
    per-unit fitted scales (only Model B has any) for the run log.

    spikes: (T, N) counts. time_grid: (T,) bin centres in seconds (used by the
    sawtooth baseline, and for dt). params: the cfg['SPIKE_PARAMS'] dict.
    fit_rows: slice of library rows, for anything estimated from data.'''
    if method not in METHODS:
        raise ValueError(f'unknown spike transform {method!r}; expected one of {METHODS}')
    time_grid = np.asarray(time_grid, dtype=float)
    dt = float(np.median(np.diff(time_grid)))
    if method == 'tsls':
        import cleaned_helper_functions as tsls
        cols = []
        for j in range(spikes.shape[1]):
            spike_times = time_grid[spikes[:, j] > 0]
            cols.append(tsls.compute_time_since_previous_spike(spike_times, time_grid))
        return np.column_stack(cols) if cols else np.empty((len(time_grid), 0)), {}
    if method == 'calcium':
        return calcium(spikes, dt, params['TAU_SYN'], params['TAU_CA']), {}
    if method == 'adapt_nr':
        r, sigma = adapt_nr(spikes, dt, params['TAU_SYN'], params['TAU_CA'], params['TAU_W'],
                            params['ADAPT_A'], params['ADAPT_B'], params['NR_LAMBDA'],
                            params['NR_N'], params['NR_SIGMA'], fit_rows)
        return r, {'sigma': sigma}
    return theta_oscillator(spikes, dt, params['OSC_F0'], params['OSC_ZETA']), {}
