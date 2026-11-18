'''Startup patch for the installed pyEDM: no per-call process pools.

Python's `site` module imports this file automatically, at interpreter
startup, for every process that uses the venv it is installed in -- the
pipeline itself and every worker it spawns. That is why it is a
sitecustomize.py and not a patch inside the pipeline: a spawned worker is a
fresh interpreter that re-imports pyEDM from disk, unpatched, whatever the
parent did in its own memory. Install it with env/README.md.

What it fixes. Four pyEDM functions -- CCM (via Project()), EmbedDimension,
PredictInterval and PredictNonlinear -- each create a brand-new
multiprocessing Pool, with a full re-import of numpy/scipy/pandas/sklearn/
pyEDM/gmn per worker, on every single call. Every call this pipeline makes
resolves to a handful of tiny tasks (E/theta/Tp lists of 15-20 entries), so
pool spawn and teardown dwarf the work: without this file, a job runs for
days regardless of dataset size or MDE_D, and the log fills with one
"No module named 'kedm'" line per freshly spawned worker. CCM exposes a
patchable `_resolve_workers()` hook; the other three create their pool
inline, so they are replaced wholesale below with sequential equivalents.

Sequential is a strict win, not a trade: the real parallelism is the
pipeline's own outer process pool (mde_gmn/waves.py), which this file does
not touch.

It also filters pyEDM's FindNeighbors() "fewer valid neighbors" UserWarning.
The pipeline varies exclusionRadius on nearly every call, so the
interpolated numbers in the message defeat Python's exact-text dedup and
every occurrence prints -- tens of GB of .err log per job.

Cost accepted deliberately: importing pyEDM adds startup latency to every
Python process in the venv, even ones that never use it. The venv is
dedicated to this pipeline, so that is paid once per process regardless.'''

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
