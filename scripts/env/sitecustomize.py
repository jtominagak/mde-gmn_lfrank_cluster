'''Project-local performance patch for this venv's installed pyEDM.

Auto-imported by Python's `site` module at interpreter startup for every
process that uses this venv -- critically, that includes BOTH
scripts/src/simple-mde-gmn.py's own process AND the `python -m
apps.InteractionMatrix` subprocess it launches via subprocess.run() (see
that script's module docstring). The latter is why this can't just be a
monkeypatch inside simple-mde-gmn.py itself: a subprocess starts a fresh
interpreter that re-imports pyEDM from disk, unpatched, regardless of
what the parent process did in its own memory. This file is the only
patch location that reaches both.

Root cause (see simple-mde-gmn.py's module docstring and
`_sequential_ccm_workers` for the original single-function diagnosis;
this file is the fuller fix once the subprocess gap was found): four
pyEDM functions -- CCM (via Project()), EmbedDimension, PredictInterval,
PredictNonlinear -- each create a *brand-new* multiprocessing Pool
(spawn/forkserver context, full reimport of numpy/scipy/pandas/sklearn/
pyEDM/gmn per worker) on *every single call*. CCM gates this behind a
patchable `_resolve_workers()` helper; the other three inline the Pool
creation with no such hook, so those three are replaced wholesale below
instead. Every call site in this pipeline (apps.InteractionMatrix: CCM
once per channel pair; dimx.MDE: CCM + EmbedDimension once per candidate
column; this pipeline's own E/tau/theta search: PredictNonlinear once per
node per (E,tau) candidate; EmbedDimension once per node) only ever
resolves to a handful of tiny per-value tasks (Evals/theta/Tp lists
capped at 15-20 entries in this pipeline), so Pool spawn + teardown cost
dwarfs the actual work every time -- confirmed by the "No module named
'kedm'" line (printed once per freshly spawned worker re-importing gmn)
appearing hundreds of thousands of times in a single job's .out log, and
by several full jobs each running ~2 days regardless of dataset size,
MODE, or MDE_D (none of which touch the actual driver: REGION_CHANNEL_CAP
x call count).

Forcing all four onto their sequential in-process path is a strict win,
not a parallelism tradeoff: task counts here never benefit from a pool,
and the real parallelism across channel pairs / candidates / nodes
already comes from each caller's own OUTER pool (apps.InteractionMatrix's
ProcessPoolExecutor, dimx's CrossMapColumns Pool, this pipeline's own
per-node/per-candidate loops under GMN's `cores`), none of which this
file touches.

Tradeoff accepted deliberately: importing pyEDM (and therefore numpy/
scipy/pandas) adds real startup latency to *every* python invocation
under this venv, even ones that never touch pyEDM. This venv
($HOME/pao_rotation/.venv) is dedicated to this one pipeline, so that
cost is paid once per job/subprocess regardless, and simplicity here
(patch eagerly at startup, one file, no import hooks) was chosen over a
lazy-patch-on-first-import mechanism.'''

import warnings

import pyEDM.API as _pyedm_api
import pyEDM.PoolFunc as _pyedm_poolfunc
import pyEDM.CCM as _pyedm_ccm_module


def _sequential_ccm_workers(parallel):
    return 1


_pyedm_ccm_module._resolve_workers = _sequential_ccm_workers


def _sequential_embed_dimension(
    dataFrame=None, columns="", target="", maxE=10, lib="", pred="",
    Tp=1, tau=-1, exclusionRadius=0, embedded=False, validLib=[],
    noTime=False, ignoreNan=True, verbose=False, numProcess=4,
    mpMethod=None, chunksize=1, showPlot=True,
):
    '''Sequential drop-in for pyEDM.EmbedDimension -- see module docstring.'''
    from pandas import DataFrame

    Evals = [E for E in range(1, maxE + 1)]
    args = {
        'columns': columns, 'target': target, 'lib': lib, 'pred': pred,
        'Tp': Tp, 'tau': tau, 'exclusionRadius': exclusionRadius,
        'embedded': embedded, 'validLib': validLib, 'noTime': noTime,
        'ignoreNan': ignoreNan,
    }
    rhoList = [_pyedm_poolfunc.EmbedDimSimplexFunc(E, dataFrame, args) for E in Evals]
    df = DataFrame({'E': Evals, 'rho': rhoList})

    if showPlot:
        from matplotlib.pyplot import show
        title = "Tp=" + str(Tp)
        ax = df.plot('E', 'rho', title=title, linewidth=3)
        ax.set(xlabel="Embedding Dimension", ylabel="Prediction Skill ρ")
        show()

    return df


def _sequential_predict_interval(
    dataFrame=None, columns="", target="", lib="", pred="", maxTp=10,
    E=1, tau=-1, exclusionRadius=0, embedded=False, validLib=[],
    noTime=False, ignoreNan=True, verbose=False, numProcess=4,
    mpMethod=None, chunksize=1, showPlot=True,
):
    '''Sequential drop-in for pyEDM.PredictInterval -- see module docstring.'''
    from pandas import DataFrame

    Tps = [Tp for Tp in range(1, maxTp + 1)]
    args = {
        'columns': columns, 'target': target, 'lib': lib, 'pred': pred,
        'E': E, 'tau': tau, 'exclusionRadius': exclusionRadius,
        'embedded': embedded, 'validLib': validLib, 'noTime': noTime,
        'ignoreNan': ignoreNan,
    }
    rhoList = [_pyedm_poolfunc.PredictIntervalSimplexFunc(Tp, dataFrame, args) for Tp in Tps]
    df = DataFrame({'Tp': Tps, 'rho': rhoList})

    if showPlot:
        from matplotlib.pyplot import show
        if embedded:
            from pyEDM.AuxFunc import IsIterable
            E = len(columns) if IsIterable(columns) else len(columns.split())
        title = "E=" + str(E)
        ax = df.plot('Tp', 'rho', title=title, linewidth=3)
        ax.set(xlabel="Forecast Interval", ylabel="Prediction Skill ρ")
        show()

    return df


def _sequential_predict_nonlinear(
    dataFrame=None, columns="", target="", theta=None, lib="", pred="",
    E=1, Tp=1, knn=0, tau=-1, exclusionRadius=0, solver=None,
    embedded=False, validLib=[], noTime=False, ignoreNan=True,
    verbose=False, numProcess=4, mpMethod=None, chunksize=1, showPlot=True,
):
    '''Sequential drop-in for pyEDM.PredictNonlinear -- see module docstring.'''
    from pandas import DataFrame
    from pyEDM.AuxFunc import IsIterable

    if theta is None:
        theta = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
    elif not IsIterable(theta):
        theta = [float(t) for t in theta.split()]

    args = {
        'columns': columns, 'target': target, 'lib': lib, 'pred': pred,
        'E': E, 'Tp': Tp, 'knn': knn, 'tau': tau,
        'exclusionRadius': exclusionRadius, 'solver': solver,
        'embedded': embedded, 'validLib': validLib, 'noTime': noTime,
        'ignoreNan': ignoreNan,
    }
    rhoList = [_pyedm_poolfunc.PredictNLSMapFunc(th, dataFrame, args) for th in theta]
    df = DataFrame({'theta': theta, 'rho': rhoList})

    if showPlot:
        from matplotlib.pyplot import show
        if embedded:
            E = len(columns) if IsIterable(columns) else len(columns.split())
        title = "E=" + str(E)
        ax = df.plot('theta', 'rho', title=title, linewidth=3)
        ax.set(xlabel="S-map Localisation (θ)", ylabel="Prediction Skill ρ")
        show()

    return df


# Patch every location the original could already be bound to. This file
# runs before any project code imports pyEDM, so `pyEDM.API.X = ...` and
# `pyEDM.X = ...` are sufficient for anything that does `from pyEDM import
# X` *after* this point (dimx.Run, simple-mde-gmn.py, apps.InteractionMatrix
# all do exactly that) -- there is no already-bound-reference gap here the
# way there would be if this patch ran from inside project code instead.
_pyedm_api.EmbedDimension = _sequential_embed_dimension
_pyedm_api.PredictInterval = _sequential_predict_interval
_pyedm_api.PredictNonlinear = _sequential_predict_nonlinear

import pyEDM as _pyedm_pkg
_pyedm_pkg.EmbedDimension = _sequential_embed_dimension
_pyedm_pkg.PredictInterval = _sequential_predict_interval
_pyedm_pkg.PredictNonlinear = _sequential_predict_nonlinear


# Second, unrelated fix, same file for the same subprocess-reach reason as
# the Pool patches above: pyEDM/Neighbors.py's FindNeighbors() emits a
# UserWarning (two variants, both prefixed '<name>: FindNeighbors() :')
# every time a prediction has fewer valid neighbors than requested outside
# the current exclusionRadius -- purely informational, the prediction still
# runs with however many neighbors it found. This pipeline's own
# EXCLUSION_RADIUS='auto' (simple-mde-gmn.py) / dynamic per-call
# exclusionRadius recomputation deliberately varies exclusionRadius on
# nearly every call, so the interpolated exclusionRadius/knn values baked
# into the message text make each occurrence a "new" message to Python's
# default warning dedup (which keys on exact text), so it never collapses
# repeats -- every single occurrence gets printed. Confirmed as the actual
# driver behind job_38871293 and its Tp-sweep siblings' 68-71GB .err files
# (job_39025340's smaller-scale run: 23GB) -- see JOB_LOG.md's entry for
# this fix's own submission for the full investigation, including the
# slurmstepd "Detected 1 oom_kill event" log line those jobs hit partway
# through, plausibly connected to this volume. 'ignore' (not 'once' --
# 'once' still dedupes on exact text, so it wouldn't help here either) is
# safe: nothing in this pipeline branches on this specific warning, only on
# the actual returned rho/columns values, which are unaffected either way.
warnings.filterwarnings(
    'ignore', category=UserWarning,
    message=r'.*FindNeighbors\(\)\s*:',
)
