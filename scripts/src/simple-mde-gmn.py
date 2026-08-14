'''Batch/cluster port of notebooks/simple-mde-gmn.ipynb, executed by
scripts/batch/simple-mde-gmn.sbatch. Every parameter is read from an
MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
PARAMETERS block) rather than edited here directly.

Pipeline steps (see the notebook for the narrative "why" behind each one
-- docstrings here cover only what's specific to running headless):

  1. MDE forward-selection (skipped if MODE='GMN'): per-target driver
     selection (dimx.MDE). Feeds both the MDE-only diagnostics in step 7
     and, when MODE='MDE-GMN', step 2's depth-0 seed.
  2. Network construction -- two unrelated methods depending on MODE;
     everything after this step is identical either way:
       - MODE='MDE-GMN': build_nested_mde_network() repeatedly runs
         dimx.MDE over a delay-expanded candidate pool (expand_delay_
         columns(), MDE_GMN_DELAY_SET -- each observable offered at every
         configured delay as its own candidate column, a generalized/
         non-uniform Takens embedding per explicit direction), turning
         each *contemporaneous* (delay=0) selected driver into a new MDE
         target for MDE_GMN_NESTED_M further rounds -- a driver selected
         at a nonzero delay becomes a terminal graph edge instead, never
         itself re-MDE'd (see that function's docstring for why),
         rejecting any edge that would create a cycle or self-loop as
         it's proposed. No CCM, no apps.InteractionMatrix, no
         apps.CreateNetwork -- the resulting MultiDiGraph (parallel
         edges let the same observable drive one node at several delays
         at once) IS the network handed to gmn.GMN directly (see
         build_network()'s docstring for why CreateNetwork would only
         be redundant here).
       - MODE='GMN': the original CCM interaction matrix (data-driven
         E/tau grid search over the REGION_CHANNEL_CAP-limited
         observable pool, via apps.InteractionMatrix), then a per-node
         driver-count estimate (EmbedDimension) and apps.CreateNetwork
         with its driver-ranking-bug workaround.
  3. Node generation config: SMap uniformly for every node, never Simplex
     or a per-node mix -- see build_gmn_config()'s docstring for why.
     Each node's actual embedding is no longer a shared network-wide
     (E, tau): gmn's own Embed(E, tau, columns) call is replaced by this
     pipeline's own per-driver-delay embedding, built straight from the
     network graph -- see _embedding_spec_for()'s and
     _generalized_embed()'s docstrings, and the gmn.Node.__init__/
     Forecast/Generate monkeypatches right below this docstring.
  4. Theta search: retunes each node's own S-map nonlinearity/
     localization parameter once per network (run_theta_search(),
     retune_theta_for_all_nodes()), then scores the result with one
     Generate() call. E and every driver's delay are never searched --
     they come directly from step 2's network -- so this replaces what
     used to be a combined (E, tau, per-node theta) grid search; see the
     SEARCH REUSE paragraph below and run_theta_search()'s docstring.
  5. Start-point sweep: at the winning theta, tries several rows for
     Generate()/Forecast() to launch from and keeps whichever matches
     the real data best -- see sweep_generate_start()'s docstring.
  6. Final rebuild at the winning theta and start row, plus the
     Forecast() diagnostic and a Fig-4e-style generative fidelity metric.
  7. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
     univariate E/tau grid) -- skipped if MODE='GMN'.

Differences from the notebook, all consequences of running headless
instead of inside a live Jupyter kernel:
  - apps.InteractionMatrix runs via subprocess.run instead of
    get_ipython().system(), which only exists inside a real kernel.
  - plt.show() calls are replaced with plt.close(fig) after the matching
    savefig, so figures don't accumulate over a long run.
  - All output goes under one flat RESULTS_DIR instead of the notebook's
    separate results/vs/figures/ split, so the sbatch script can copy it
    back with a single `cp -av`.
  - CORES is threaded through every *_cores/numProcess parameter (dimx.MDE,
    apps.InteractionMatrix's -cr, pyEDM's numProcess, gmn.GMN's cores --
    the last is currently a no-op in the installed gmn package, passed
    anyway in case a future version uses it) instead of each tool's own
    hardcoded default, which has no way to know how many cores SLURM
    actually allocated.

FIXED: apps.InteractionMatrix's own worker pool and pyEDM.CCM()'s
internal multiprocessing used to nest inside each other -- CORES=8 could
spawn ~90 live processes, not 8, with each of CCM()'s ~100k+ calls paying
a full process-spawn-and-reimport cost for only 4-8 tiny KDTree tasks.
This is why early jobs took ~2 days regardless of dataset size.
`_sequential_ccm_workers` below forces pyEDM.CCM() onto its in-process
path everywhere in this pipeline, removing the inner pool so only the
outer one provides parallelism -- see that function's and
build_network()'s docstrings for the full evidence trail.

SEARCH REUSE (MDE_GMN_REUSE_BEST_PARAMS, default off): every genuinely
expensive stage -- MDE forward-selection (run_mde_for_target(), step 1),
network construction (build_network()/build_nested_mde_network(),
step 2), theta search (run_theta_search(), step 4), and the generate-
start sweep (sweep_generate_start(), step 5) -- is deterministic given
the same inputs, so resubmitting identical settings pays for all of them
twice for no reason. When enabled, each stage builds its own fingerprint
(every config knob that shapes its outcome, plus a hash of the input data
file -- build_search_fingerprint()/build_network_fingerprint()/
build_mde_selection_fingerprint()/build_start_sweep_fingerprint()) and
looks for a past job's matching cache file (best_params_<network_id>.json
/ network_cache_<network_id>.pkl / mde_selection_<node>.pkl /
generate_start_sweep_<network_id>.json) under MDE_GMN_PREV_RUNS_DIR via
find_cached() -- a hit skips that stage's own expensive work entirely and
reuses the cached result; any mismatch (different dataset, different
MDE_D/NESTED_M/REGION_CHANNEL_CAP/DELAY_SET/etc., or the first run with
these settings) just runs the real computation, so this is always a safe,
redundant-compute optimization, never a correctness shortcut.

MDE_GMN_PREV_RUNS_DIR/MDE_GMN_RUN_CACHE_DIR deliberately point at a
durable bucket-storage pool (see simple-mde-gmn.sbatch's CACHE_ROOT/
RUN_CACHE_DIR comment), not RESULTS_DIR -- RESULTS_DIR lives on a /flash
scratch tempdir deleted the instant this script exits, including on a
crash, before its contents ever reach persistent storage (see that
script's /flash scratch setup section). Every save_json_to_cache()/
save_pickle_to_cache() call therefore writes to RUN_CACHE_DIR
*immediately*, right after its stage finishes, in addition to the
ordinary results_dir write -- so a stage's result is durable the moment
it completes, not contingent on the whole job finishing successfully.
This is what actually fixes a 23-hour job losing everything to a crash in
its very last step (see JOB_LOG.md job 36533849): before this, only
theta search had a cache at all, and even that cache's write only ever
reached persistent storage via RESULTS_DIR's own end-of-job copy, so it
never survived a crash either.

Before this pipeline's generalized (per-driver-delay) embedding existed,
this same mechanism also cached a searched network-wide (E, tau) -- see
scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that version,
kept for reference/comparison per explicit direction.
'''
import ast
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict

import dimx as dx
import gmn
from gmn.Common import FunctionType
from apps.CreateNetwork import CreateNetwork
from pyEDM import (
    Simplex, ComputeError, EmbedDimension,
    PredictNonlinear, PredictInterval,
)


def _embedding_spec_for(graph, node_name):
    '''Builds one node's (column, delay) embedding coordinates directly
    from the network graph -- a pure function, reusable both by
    _node_embedding_spec() (which caches this on a live gmn.Node object,
    for gmn's own monkeypatched Forecast()/Generate()) and by
    retune_theta_for_all_nodes() (which builds embeddings directly with
    pandas/pyEDM outside gmn's Node/Network machinery entirely, before
    any gmn.GMN object exists).

    MultiDiGraph (MODE='MDE-GMN', built by build_nested_mde_network()):
    self always gets the full self_delay_set (graph.graph['self_delay_set'],
    set once in build_network()) regardless of how many external drivers
    nested-MDE found for this node -- the same unconditional fallback a
    driverless leaf node already needed, just now built from the same
    delay machinery as everything else rather than a special case. Each
    real in-edge contributes one coordinate at that edge's own
    discovered delay (see build_nested_mde_network()'s docstring for why
    the same driver can appear more than once, at different delays).

    Plain DiGraph (MODE='GMN', from apps.CreateNetwork -- no per-edge
    delay data exists in that mode at all): falls back to the original
    uniform E-lagged-copies-of-every-column scheme, using
    graph.graph['fallback_E']/['fallback_tau'] (set in build_network()
    from that mode's own CCM E/tau grid search) in place of a per-node
    nested-MDE result it doesn't have.

    Either way, this is what replaced gmn's own Embed(E, tau, columns)
    call -- see _generalized_embed()'s docstring -- since a single
    uniform (E, tau) can't express a per-driver, per-delay embedding.'''
    if isinstance(graph, nx.MultiDiGraph):
        self_delays = graph.graph.get('self_delay_set', [0])
        spec = [(node_name, d) for d in self_delays]
        spec += [(u, data['delay']) for u, v, data in graph.in_edges(node_name, data=True)]
    else:
        E = graph.graph.get('fallback_E', 3)
        tau = graph.graph.get('fallback_tau', -1)
        cols = list(graph.predecessors(node_name)) + [node_name]
        spec = [(c, k * tau) for c in cols for k in range(E)]
    return spec


def network_max_embedding_offset(graph):
    '''Largest per-node embedding offset across the whole network --
    since nodes can each have their own distinct embedding now (unlike
    the old shared (E, tau)), this is used by build_gmn_config()'s
    Forecast mode to size a single, network-wide lib/pred window that
    stays valid for every node (see that function's docstring for the
    row-alignment tradeoff this implies for lower-offset nodes).'''
    if isinstance(graph, nx.MultiDiGraph):
        self_delays = graph.graph.get('self_delay_set', [0])
        self_max = max(abs(d) for d in self_delays)
        edge_max = max((abs(data['delay']) for _, _, data in graph.edges(data=True)), default=0)
        return max(self_max, edge_max)
    E = graph.graph.get('fallback_E', 3)
    tau = graph.graph.get('fallback_tau', -1)
    return (E - 1) * abs(tau)


def _node_embedding_spec(node):
    '''Cached (on node._embedding_spec) wrapper around _embedding_spec_for()
    for a live gmn.Node object -- see that function's docstring for the
    full construction.'''
    if not hasattr(node, '_embedding_spec'):
        node._embedding_spec = _embedding_spec_for(node.Network.Graph, node.name)
    return node._embedding_spec


def _generalized_embed(data, embedding_spec):
    '''Builds a dataframe with one column per (col, delay) pair in
    embedding_spec, each holding that column's value abs(delay) rows in
    the past (every delay is <= 0). Named to match pyEDM.Embed()'s own
    "col(t-k)" convention (k = abs(delay)) so downstream code that looks
    up Parameters.target + '(t-0)' for the current-value column keeps
    working unchanged. Returns (embedded_df, offset) with a fresh
    0-based index -- offset is the largest lag magnitude across the
    whole spec, i.e. how many leading rows are dropped as NaN once every
    (col, delay) pair has been shifted into place (mirrors the
    (E-1)*abs(tau)-style offset gmn's own Embed()-based code computed,
    generalized to a non-uniform delay set).'''
    columns = {
        f'{col}(t-{abs(delay)})': data[col].shift(-delay).to_numpy()
        for col, delay in embedding_spec
    }
    embedded = pd.DataFrame(columns, index=data.index)
    offset = max(abs(delay) for _, delay in embedding_spec)
    return embedded.iloc[offset:].reset_index(drop=True), offset


_real_node_init = gmn.Node.__init__


def _generalized_node_init(self, args, Network, nodeName):
    '''Wraps gmn.Node.__init__ to correct self.libEnd_i (the EDM library
    end index) for SMap nodes after construction -- the installed
    package computes it as self.data.shape[0] - Parameters.E *
    abs(Parameters.tau), which assumes one shared, uniform embedding;
    the generalized embedding this pipeline builds instead (see
    _node_embedding_spec()'s docstring) has its own, generally
    different, per-node offset. Recomputed exactly once here, right
    after __init__ sets the node's pristine self.data, since libEnd_i is
    meant to reflect that original library size before any values get
    appended during a Generate() run (gmn/Node.py's own docstring: "the
    data library is limited to Parameters.predictionStart... does not
    grow") -- not something to recompute on every later call.'''
    _real_node_init(self, args, Network, nodeName)
    if self.FunctionType == FunctionType.SMap:
        spec = _node_embedding_spec(self)
        offset = max(abs(delay) for _, delay in spec)
        self.libEnd_i = self.data.shape[0] - offset


gmn.Node.__init__ = _generalized_node_init


def _generalized_node_forecast(self):
    '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
    pre-existing bugs (below) and replacing its SMap branch's
    Embed(E, tau, columns) call with _generalized_embed() -- the
    per-driver-delay embedding built from the network graph (see
    _node_embedding_spec()'s docstring) instead of one uniform (E, tau)
    applied to every predictor. This is the only branch this pipeline
    exercises, since every node uses SMap (see build_gmn_config()'s
    docstring).

    (1) gmn/Forecast.py calls Embed() without importing pyEDM.Embed (it
    only does `from .Common import *`, unlike gmn/Generate.py's identical
    SMap branch, which imports it correctly) -- raised NameError on the
    first Forecast() call against any SMap node. Moot now that this
    branch no longer calls pyEDM's Embed() at all, but noted since the
    original bug is what first justified monkeypatching this method.

    (2) `val` is a 2-tuple (time, predictions), then checked for
    divergence via `val.max() > 1E15` -- tuples have no .max(), so this
    raised AttributeError unconditionally, not just on genuine
    divergence. Fixed to check the predictions Series directly (val[1]).

    The time column is inserted explicitly from data's own offset-sliced,
    reset-index time values (data[data.columns[0]].iloc[offset:].reset_
    index(drop=True)) rather than gmn's own `df.insert(0, ...,
    data.iloc[:,0])`, which relied on pandas' implicit index-alignment
    between an un-reset Embed()+dropna() frame and the untouched `data`
    -- since _generalized_embed() already resets its own index to 0-based,
    that implicit alignment would silently line up the wrong rows here.

    Every other branch is copied verbatim from gmn/Forecast.py (unused
    here, since every node is SMap, but kept intact so this patch can't
    silently change behavior for a future non-SMap node). Monkeypatched
    onto gmn.Node (gmn/__init__.py's `from .Node import Node` shadows the
    submodule name with the class) rather than editing the installed
    package, so this script stays self-contained on other machines.'''
    if self.args.DEBUG :
        print( '-----> Node:Forecast() : ',self.FunctionType.name, self.name)
        print( self.data.tail( 2 ), flush = True );

    Parameters = self.Parameters
    data       = self.data

    if self.args.DEBUG :
        print( '  data : shape: ', data.shape );
        print( data.tail( 3 ) ); print('', flush = True)

    if self.FunctionType.value == FunctionType.Simplex.value :
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else:
            lib = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )

        S = self.Function( dataFrame       = data,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = Parameters.columns,
                           target          = Parameters.target,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S.iloc[:,0], S['Predictions'] )

    elif self.FunctionType.value == FunctionType.SMap.value :
        spec = _node_embedding_spec(self)
        df, offset = _generalized_embed(data, spec)

        df.insert( 0, data.columns[0], data[data.columns[0]].iloc[offset:].reset_index(drop=True) )

        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else :
            lib  = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )

        S = self.Function( dataFrame       = df,
                           lib             = lib,
                           pred            = pred,
                           E               = len(spec),
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = 1,
                           theta           = Parameters.theta,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = df.columns[1:-1],
                           target          = Parameters.target + '(t-0)',
                           solver          = None,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S['predictions'].iloc[:,0], S['predictions']['Predictions'] )

        if val[1].max() > 1E15 :  # FIX: was val.max() -- val is the tuple above
            raise RuntimeError( "Forecast:SMap() Node: " + self.name +\
                                "   Divergence detected (>1E15)" )

    elif self.FunctionType.value == FunctionType.kedmSimplex.value :
        raise RuntimeError( "Forecast kedmSimplex function not available" )

    elif self.FunctionType.value == FunctionType.kedmSMap.value :
        raise RuntimeError( "Forecast kedmSMap function not available" )

    elif self.FunctionType.value == FunctionType.Linear.value :
        raise RuntimeError( "Forecast Linear function not available" )

    elif self.FunctionType.value == FunctionType.SVR.value :
        raise RuntimeError( "Forecast SVR function not available" )

    elif self.FunctionType.value == FunctionType.knn.value :
        raise RuntimeError( "Forecast knn function not available" )

    if self.args.DEBUG :
        print( self.name, "val:", val )
        print( '<----- Node:Forecast() : ',self.FunctionType.name,self.name )
        print('', flush = True)

    return val


gmn.Node.Forecast = _generalized_node_forecast


def _generalized_node_generate(self, lastDataOut):
    '''Corrected copy of gmn.Node.Generate (gmn/Generate.py), replacing
    its SMap branch's Embed(E, tau, columns) call with
    _generalized_embed() -- see _generalized_node_forecast()'s and
    _node_embedding_spec()'s docstrings for why. Unlike Forecast()'s
    SMap branch, gmn's own Generate() code never tried to reuse real
    time values for its inserted time column (just a fresh synthetic
    0-based arange()), so there's no index-alignment subtlety to correct
    here -- _generalized_embed()'s own 0-based index is used as-is.
    Every other branch is copied verbatim (unused here, since every node
    is SMap). Not a pre-existing gmn bug fix like the Forecast patch --
    added because gmn/Generate.py's SMap branch needed the same
    Embed()-call replacement, and this pipeline had no reason to patch
    Node.Generate at all before this pipeline's generalized-embedding
    feature existed.'''
    if self.args.DEBUG :
        print( '-----> Node:Generate() : ', self.FunctionType.name, self.name )
        print( self.data.tail( 2 ) );
        print( 'lastDataOut:' ); print( lastDataOut, flush = True )

    if not ( lastDataOut is None ):
        self.data = pd.concat( [ self.data, lastDataOut ] )

    Parameters = self.Parameters
    data       = self.data

    if self.args.DEBUG :
        print( '  Appended data : shape: ', data.shape );
        print( data.tail( 3 ) ); print('', flush = True)

    if self.FunctionType.value == FunctionType.Simplex.value :
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else:
            lib = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )

        S = self.Function( dataFrame       = data,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = Parameters.columns,
                           target          = Parameters.target,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = S['Predictions'].iloc[-1]

    elif self.FunctionType.value == FunctionType.SMap.value :
        spec = _node_embedding_spec(self)
        df, offset = _generalized_embed(data, spec)

        timeColumn = np.arange(df.shape[0])
        df.insert(0, 'time', timeColumn, allow_duplicates=True)

        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else :
            lib  = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )

        S = self.Function( dataFrame       = df,
                           lib             = lib,
                           pred            = pred,
                           E               = len(spec),
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = 1,
                           theta           = Parameters.theta,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = df.columns[1:-1],
                           target          = Parameters.target + '(t-0)',
                           solver          = None,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = S['predictions']['Predictions'].iloc[-1]

        if abs( val ) > 1E15 :
            raise RuntimeError( "Generate:SMap() Node: " + self.name +\
                                "   Divergence detected (>1E15)" )

    elif self.FunctionType.value == FunctionType.kedmSimplex.value :
        data_ = data[ Parameters.columns ] # DataFrame; includes target

        targetLen = Parameters.E * abs( Parameters.tau )
        pred    = data_.iloc[ -targetLen : ]
        library = data_.iloc[ : self.libEnd_i ]
        target  = library[ Parameters.target ]

        S = self.Function( lib    = library.to_numpy(),
                           pred   = pred.to_numpy(),
                           target = target.to_numpy(),
                           E      = Parameters.E,
                           tau    = abs( Parameters.tau ),
                           Tp     = Parameters.Tp )

        val = S[-1]

    elif self.FunctionType.value == FunctionType.kedmSMap.value :
        data_ = data[ self.name ] # Series

        targetLen = Parameters.E * abs( Parameters.tau )
        pred    = data_.iloc[ -targetLen : ]
        library = data_.iloc[ : self.libEnd_i ]
        target  = library

        S = self.Function( lib    = library.to_numpy(),
                           pred   = pred.to_numpy(),
                           target = target.to_numpy(),
                           E      = Parameters.E,
                           tau    = abs( Parameters.tau ),
                           Tp     = Parameters.Tp,
                           theta  = Parameters.theta )

        val = S[-1]

    elif self.FunctionType.value == FunctionType.Linear.value :
        X = data[ Parameters.columns ]
        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        lr = self.Function()
        lr.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = lr.predict( nextX )
        val   = pred[ 0 ]

    elif self.FunctionType.value == FunctionType.SVR.value :
        X = data[ Parameters.columns ]
        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        svr = self.Function( kernel = 'rbf', gamma = 'scale', tol = 0.001,
                             epsilon = 0.1, shrinking = True, cache_size = 200,
                             verbose = False, max_iter = -1 )
        svr.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = svr.predict( nextX )
        val   = pred[ 0 ]

    elif self.FunctionType.value == FunctionType.knn.value :
        X = data[ Parameters.columns ]

        if len( Parameters.columns ) > 1 :
            X  = X.drop( self.name, axis = 'columns' )

        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        knn = self.Function( n_neighbors = 10, weights = 'distance' )
        knn.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = knn.predict( nextX )
        val   = pred[ 0 ]

    if self.args.DEBUG :
        print( self.name, "val:", val )
        print( '<----- Node:Generate() : ',self.FunctionType.name,self.name )
        print('', flush = True)

    return val


gmn.Node.Generate = _generalized_node_generate


def _sequential_ccm_workers(parallel):
    '''Replaces pyEDM.CCM's private _resolve_workers(), forcing every
    CCM().Project() call in this process onto its sequential in-process
    path instead of spawning a fresh multiprocessing Pool.

    This is the fix for the nested-pool blowup described in the module
    docstring: every pyEDM.CCM() call (one per channel pair in
    apps.InteractionMatrix, one per candidate column in dimx.Run) used to
    spawn its own fresh Pool sized to cpu_count(), on top of whichever
    outer pool was already dispatching that call -- confirmed by reading
    pyEDM/CCM.py and by the "No module named 'kedm'" line each spawned
    worker printed on reimport, hundreds of thousands of times in one run.
    Neither apps.InteractionMatrix nor dimx.Run ever pass `parallel=`
    through to CCM(), so this couldn't be tuned via their CLIs.

    Forcing sequential execution is a strict win, not a tradeoff: every
    CCM() call here only ever resolves to 4-8 tiny KDTree tasks (2
    directions x 2-4 libSizes), so a fresh Pool's spawn-and-reimport cost
    dwarfs the actual work. Real parallelism across channel pairs/
    candidates already comes from the outer pool, which this doesn't
    touch. Monkeypatched (rather than passing parallel=False at each call
    site) because apps.InteractionMatrix constructs CCM() internally with
    no way to pass that argument in; dimx.Run's own CCM() call gets a
    second, independent patch below (_sequential_ccm_api) that does force
    parallel=False explicitly, so this override is the sole protection
    for apps.InteractionMatrix and a belt-and-suspenders one for dimx.'''
    return 1


import pyEDM.CCM as _pyedm_ccm_module
_pyedm_ccm_module._resolve_workers = _sequential_ccm_workers

import dimx.Run as _dimx_run_module
_real_pyedm_ccm_api = _dimx_run_module.CCM
_real_cross_map_columns_api = _dimx_run_module.CrossMapColumns
_real_embed_dimension_api = _dimx_run_module.EmbedDimension

# Set from load_config() (MDE_GMN_EXCLUSION_RADIUS=auto, the default) --
# gates the three dynamic-exclusion-radius wrappers below. Module-level
# rather than threaded through call args because dimx.Run's own internal
# CrossMapColumns/EmbedDimension/CCM calls are what's being intercepted
# here, not something this pipeline calls directly. A plain global is
# safe across the nested-MDE wave pool: that pool is always created with
# mp_context=mp.get_context('fork') (see build_nested_mde_network()),
# which clones this process's memory -- including this flag's already-
# resolved value -- rather than re-executing imports the way forkserver
# would.
_EXCLUSION_RADIUS_AUTO = False


def _delay_from_column_name(name):
    '''DELAY_MARKER/_parse_delayed_column_name are defined later in this
    file (expand_delay_columns() section) -- fine as a forward reference
    since this is only ever called at runtime, long after the whole
    module has finished loading. Delay=0 (contemporaneous) columns have
    no ::delay suffix at all (see expand_delay_columns()'s docstring), so
    absence of DELAY_MARKER means delay 0, not "not a candidate column"
    -- every name reaching here (from dimx.Run's own `columns` args) is
    already known to be one.'''
    if DELAY_MARKER in name:
        return _parse_delayed_column_name(name)[1]
    return 0


def _dynamic_exclusion_radius(columns):
    '''Auto exclusion-radius (MDE_GMN_EXCLUSION_RADIUS=auto, the
    default): approximates the classic Theiler-window guideline
    exclusionRadius >= E*tau for dimx.MDE's forward selection over this
    pipeline's *generalized* (multi-delay, DELAY_SET-driven) embedding --
    where E and tau are confounded, per explicit direction, because
    dimx.Run() grows a genuinely non-uniform embedding one selected
    column at a time, and each column can carry its own independent
    DELAY_SET-encoded delay rather than one lag shared network-wide.
    Adding a new delayed observable always grows E by exactly 1, but the
    embedding's own effective "tau" is whatever the single largest delay
    among ALL its dimensions happens to be -- not a fixed per-step
    increment -- so a single static E*tau computed once upfront can't
    track it correctly.

    Rather than a fixed a-priori bound, this is called fresh at every
    dimx.Run() internal CrossMapColumns()/EmbedDimension()/CCM() call
    (from the monkeypatched wrappers below, replacing dimx.Run's own
    module-level names -- same pattern as _sequential_ccm_api above) and
    reads dimension/delay directly off whatever `columns` argument that
    specific call was made with:
      - CrossMapColumns' `columns` is dimx.Run's own nested list of
        [candidate] + already-selected-MDEcolumns lists (one per
        candidate still being ranked at the current forward-selection
        step) -- exactly "the multivariate embedding as it's being built
        up", so this naturally grows across dimx.Run()'s d=1..D loop as
        MDEcolumns accumulates: always the largest tau *actually selected
        so far*, not a worst-case bound.
      - EmbedDimension/CCM's `columns` is a single already-chosen
        candidate column name (dimx.Run validates one column at a time
        here) -- treated as its own 1-dimensional embedding.
    Delays are parsed straight from each column's own ::delay<N> suffix
    (_delay_from_column_name()) -- exactly the same DELAY_SET-driven
    lag every column already carries via expand_delay_columns(), not a
    separately-modeled quantity.

    Returns dim * max_abs_delay, where dim is the size of the largest
    embedding evaluated in this call and max_abs_delay is the largest
    delay (in rows) among every column in it.'''
    if isinstance(columns, str):
        groups = [[columns]]
    elif columns and isinstance(columns[0], str):
        groups = [list(columns)]
    else:
        groups = [list(group) for group in columns]

    max_dim = max((len(group) for group in groups), default=0)
    max_delay = max(
        (abs(_delay_from_column_name(c)) for group in groups for c in group),
        default=0,
    )
    return max_dim * max_delay


def _auto_exclusion_radius_cross_map_columns(*args, **kwargs):
    '''Replaces dimx.Run's CrossMapColumns binding -- see
    _dynamic_exclusion_radius()'s docstring. A no-op passthrough when
    MDE_GMN_EXCLUSION_RADIUS is a fixed integer rather than 'auto' (the
    ordinary case dimx.MDE's own exclusionRadius kwarg already handles
    correctly on its own).'''
    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
    return _real_cross_map_columns_api(*args, **kwargs)


def _auto_exclusion_radius_embed_dimension(*args, **kwargs):
    '''Replaces dimx.Run's EmbedDimension binding -- see
    _auto_exclusion_radius_cross_map_columns()'s docstring, same
    reasoning, applied to the single-column E-search dimx.Run() runs on
    each forward-selection step's winning candidate.'''
    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
    return _real_embed_dimension_api(*args, **kwargs)


def _sequential_ccm_api(*args, **kwargs):
    '''Second, independent layer on top of _sequential_ccm_workers: forces
    parallel=False on pyEDM's top-level CCM() function directly. Patches
    the `CCM` name inside dimx.Run's own namespace (dimx/Run.py imports it
    as `from pyEDM import ... CCM ...`, so `dimx.Run.CCM` is the exact
    binding dimx.MDE.Run() calls) rather than relying only on the
    _resolve_workers indirection above, in case that patch doesn't reach
    every call the same way in every process. Also applies the dynamic
    exclusion radius (see _auto_exclusion_radius_cross_map_columns()
    above) to dimx.Run's own CCM() causality-validation call, same gating
    on _EXCLUSION_RADIUS_AUTO.'''
    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
    kwargs['parallel'] = False
    return _real_pyedm_ccm_api(*args, **kwargs)


_dimx_run_module.CCM = _sequential_ccm_api
_dimx_run_module.CrossMapColumns = _auto_exclusion_radius_cross_map_columns
_dimx_run_module.EmbedDimension = _auto_exclusion_radius_embed_dimension


def _mde_wave_worker_init():
    '''ProcessPoolExecutor(initializer=...) hook for
    build_nested_mde_network()'s per-wave worker pool -- reapplies all
    three dimx.Run patches above (CCM parallelism + exclusion radius,
    CrossMapColumns/EmbedDimension exclusion radius) inside every worker
    process, rather than assuming they survive from this module's own
    top-level monkeypatch lines (they don't reliably on this cluster).

    The failure mode when it doesn't is a hard crash, not just a slowdown:
    a worker whose dimx.MDE -> CCM() call lands on the unpatched path
    tries to create CCM's own Pool from inside an already-pooled worker
    process, which fails outright with "ChildProcessError: No child
    processes". The exact trigger was never fully pinned down, but
    applying these patches together, fresh inside every worker via this
    initializer, reliably avoided it across repeated test runs, while
    applying only a subset did not.'''
    import pyEDM.CCM as _ccm
    _ccm._resolve_workers = _sequential_ccm_workers
    import dimx.Run as _dimx_run
    _dimx_run.CCM = _sequential_ccm_api
    _dimx_run.CrossMapColumns = _auto_exclusion_radius_cross_map_columns
    _dimx_run.EmbedDimension = _auto_exclusion_radius_embed_dimension


# =============================================================================
# Configuration -- every value overridable via MDE_GMN_* environment
# variables; see scripts/batch/simple-mde-gmn.sbatch's MODIFIABLE
# PARAMETERS block for the canonical list with descriptions.
# =============================================================================

def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [x.strip() for x in val.split(',') if x.strip()]


def _env_float_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [float(x.strip()) for x in val.split(',') if x.strip()]


def _env_int_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [int(x.strip()) for x in val.split(',') if x.strip()]


def load_config():
    cfg = {}

    # ---- Data ---------------------------------------------------------
    # NOTE: unlike the notebook (DATA_DIR = '../data/transfer_.../'), the
    # cluster deployment expects data files copied flat into DATA_DIR by
    # the sbatch script -- no nested extraction-folder name.
    cfg['DATA_DIR'] = Path(os.environ.get('MDE_GMN_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('MDE_GMN_DATA_FILE', 'J1620210605_epoch_8_freq_4_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('MDE_GMN_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- Overall pipeline shape ---------------------------------------
    cfg['MODE'] = os.environ.get('MDE_GMN_MODE', 'MDE-GMN')  # 'MDE-GMN' | 'GMN'
    if cfg['MODE'] not in ('MDE-GMN', 'GMN'):
        raise ValueError(f"Unknown MDE_GMN_MODE: {cfg['MODE']!r}")
    cfg['GMN_MODE'] = os.environ.get('MDE_GMN_GMN_MODE', 'combined')  # 'combined' | 'separate'

    cfg['TARGETS'] = _env_list('MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
    cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None

    # Lib/pred split as a fraction of the full dataset -- e.g. LIB_RATIO=0.5
    # means the first 50% of rows are library, the remaining 50% are the
    # held-out pred/Generate() window. PREDICTION_START/PREDICTION_LENGTH
    # (row counts derived from this ratio and the data's length) are filled
    # in by main() once the data is loaded, and are what the rest of the
    # pipeline reads: the MDE lib/pred split, CCM/EmbedDimension calls, and
    # the held-out window Generate() results are validated against.
    cfg['LIB_RATIO'] = float(os.environ.get('MDE_GMN_LIB_RATIO', 0.5))

    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH:
    # every Generate() call (in the theta search, the start sweep, and
    # the final rebuild) runs this many steps, not the entire held-out
    # window. Generate() is the single most expensive step in this
    # pipeline (one full network simulation per call), so this shrinks that
    # cost independently of LIB_RATIO/MDE's own held-out window, which the
    # causality gating (CCM, EmbedDimension) still needs to be a meaningful
    # size. Also caps the Forecast() diagnostic's pred window (see
    # build_gmn_config()'s docstring) so actual/Generate()/Forecast() all
    # cover the same window on that plot. 0 (default) means "use the full
    # PREDICTION_LENGTH"; a larger value is clamped down to it in main()
    # once PREDICTION_LENGTH is known (there'd be no real data to score any
    # extra steps against).
    cfg['GENERATE_LENGTH'] = int(os.environ.get('MDE_GMN_GENERATE_LENGTH', 0))

    # Theiler-window / neighbor-exclusion radius for gmn's own recursive
    # Generate()/Forecast() nearest-neighbor search (SMap), passed through
    # as Parameters.exclusionRadius by build_gmn_config() -- independent
    # of MDE_GMN_EXCLUSION_RADIUS above, which only affects dimx.MDE's own
    # forward-selection search. Named without the MDE_GMN_ prefix
    # (deliberately, per explicit direction) to keep it visually distinct
    # from that unrelated MDE-side parameter. Static -- unlike MDE's
    # 'auto' mode, there's no equivalent generalized/non-uniform embedding
    # confound here since every node uses one shared theta-driven SMap
    # embedding (see build_gmn_config()'s function='SMap' docstring), so
    # 0 (default, matches this pipeline's pre-existing hardcoded
    # behavior) is a reasonable starting point -- change it to explore
    # sensitivity to temporal-neighbor exclusion in the generative step.
    cfg['GMN_EXCLUSION_RADIUS'] = int(os.environ.get('GMN_EXCLUSION_RADIUS', 0))

    # Plotted x-axis range for the final Generate()-vs-Forecast()-vs-
    # actual diagnostic figure (run_forecast_diagnostic()) -- see that
    # function's plot_range_mode docstring for the full behavior of each
    # option. 'full' (default, unchanged pre-existing behavior): actual/
    # lib plotted from row 0 through generate_start+GENERATE_LENGTH,
    # Forecast() computes exactly GENERATE_LENGTH steps. 'windowed': the
    # window narrows to 2x GENERATE_LENGTH on each side of generate_start
    # (clamped to available data) -- actual/lib fills the whole window
    # (cheap, real data), Forecast() is run out to that same 2x length
    # (cheap, a direct per-row SMap prediction against real held-out
    # data, not a recursive simulation), Generate() itself is left
    # unchanged. Only affects this one diagnostic plot/Forecast() call --
    # the theta search, start-point sweep, and Generate() rebuild all use
    # GENERATE_LENGTH exactly as before regardless of this setting.
    cfg['FORECAST_PLOT_RANGE'] = os.environ.get('MDE_GMN_FORECAST_PLOT_RANGE', 'full').strip().lower()
    if cfg['FORECAST_PLOT_RANGE'] not in ('full', 'windowed'):
        raise ValueError(
            f"MDE_GMN_FORECAST_PLOT_RANGE must be 'full' or 'windowed', "
            f"got {cfg['FORECAST_PLOT_RANGE']!r}"
        )

    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
    # as offsets (rows) from PREDICTION_START -- e.g. MIN=0, MAX=200,
    # STEP=50 tries offsets 0, 50, 100, 150 (MAX exclusive, like E_MAX/
    # TAU_MAX below). sweep_generate_start() runs one Generate() call per
    # offset at the already-chosen best theta (E and every driver's delay
    # come from nested-MDE itself -- see the DELAY_SET comment below --
    # not re-searched) and keeps whichever start best matches the real
    # data: Generate() is a recursive nonlinear simulation, so which row
    # it launches from can change how well it tracks reality, independent
    # of how good the embedding/theta are. Offsets must be >= 0 (never
    # before PREDICTION_START) so every candidate's comparison data stays
    # genuinely out-of-sample. Default MIN=0, MAX=1 tries only offset 0
    # (always PREDICTION_START, unchanged pre-existing behavior) -- widen
    # MAX to actually sweep. Offsets that would leave too little data to
    # score GENERATE_LENGTH steps against are dropped once
    # PREDICTION_LENGTH is known (main()).
    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
    cfg['GENERATE_START_STEP'] = int(os.environ.get('MDE_GMN_GENERATE_START_STEP', 50))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    # MDE_D counts *real* observables, not delay-expanded candidate
    # columns -- e.g. with DELAY_SET holding 5 delays, MDE_D=5 means "up
    # to 5 distinct real observables", not "5 candidate columns" (which
    # could all be different delays of the very same observable).
    # effective_mde_dimension() converts this into the actual D passed to
    # dimx.MDE (MDE_D * len(DELAY_SET), capped at however many real
    # observables that call's own candidate pool actually has) -- see
    # that function's docstring. Setting MDE_D >= the real observable
    # count (e.g. 999 against 500 real observables) makes MDE consider
    # all of them, capped down rather than inflating D past what the pool
    # can supply.
    cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
    # Prediction interval every dimx.MDE forward-selection call (both the
    # top-level per-target search and every nested-MDE per-node search --
    # see run_mde_for_target()) ranks/validates candidate drivers against.
    # dimx.MDE's own default is 1 -- unset before this parameter existed,
    # every driver in this pipeline was being selected purely for 1-step-
    # ahead skill, with nothing in the selection ever rewarding stability
    # over a long recursive Generate() run. See simple-mde-gmn.sbatch's TP
    # comment and JOB_LOG.md's job 36786285 diagnosis for the full
    # rationale.
    cfg['TP'] = int(os.environ.get('MDE_GMN_TP', 1))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # 'auto' (default): dynamic per-dimx.Run()-call exclusion radius --
    # see _dynamic_exclusion_radius()'s docstring near this file's
    # monkeypatch block (just above _sequential_ccm_api) for the full
    # E*tau-confounded-by-DELAY_SET rationale. Set an explicit integer
    # instead to disable that and use dimx.MDE's ordinary fixed
    # exclusionRadius on every call.
    _raw_exclusion_radius = os.environ.get('MDE_GMN_EXCLUSION_RADIUS', 'auto').strip()
    cfg['EXCLUSION_RADIUS_AUTO'] = _raw_exclusion_radius.lower() == 'auto'
    cfg['EXCLUSION_RADIUS_VALUE'] = (
        0 if cfg['EXCLUSION_RADIUS_AUTO'] else int(_raw_exclusion_radius)
    )
    global _EXCLUSION_RADIUS_AUTO
    _EXCLUSION_RADIUS_AUTO = cfg['EXCLUSION_RADIUS_AUTO']
    # Nested-MDE network construction (MODE='MDE-GMN' only -- see
    # build_nested_mde_network()'s docstring for the full algorithm).
    # Reuses MDE_D as the per-node driver cap N at every nesting level
    # (deliberately not a separate parameter -- MDE_D already means "max
    # # of real observables MDE can forward-select", which is exactly N
    # here too). NESTED_M is the number of *additional* nested-MDE rounds
    # beyond the initial per-target selection -- total graph depth from
    # each target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for
    # "uncapped": nesting keeps going, round after round, until it runs
    # out on its own (every reachable real observable already visited, or
    # dimx.MDE/cycle-rejection naturally dead-ends every remaining
    # branch) instead of stopping at a fixed round count -- see
    # build_nested_mde_network()'s docstring. Combinatorially expensive: a
    # naive upper bound on nodes is roughly MDE_D**(1+NESTED_M) (or, when
    # NESTED_M=0, bounded only by the real observable pool size) before
    # dedup/cycle-rejection collapses it back down to the real (capped)
    # observable pool size -- keep MDE_D and NESTED_M small for a first
    # test run.
    cfg['NESTED_M'] = int(os.environ.get('MDE_GMN_NESTED_M', 1))
    # Candidate observable pool: first N channels of each region. Default
    # 999 (effectively uncapped -- matches the current notebook's default
    # and Park et al. 2026's own full-channel-set construction); cap this
    # down for a fast test job.
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('MDE_GMN_REGION_CHANNEL_CAP', 999))

    # Delay set nested-MDE searches per observable (MODE='MDE-GMN' only):
    # each candidate observable is offered to dimx.MDE's forward selection
    # as len(DELAY_SET) separate columns, one per delay -- e.g. 200
    # observables x 5 delays = 1000 candidate columns -- so MDE picks
    # specific (observable, delay) pairs rather than being stuck with one
    # fixed contemporaneous (delay=0) copy of every observable. This is a
    # generalized/non-uniform Takens embedding (Deyle & Sugihara 2011):
    # different driving variables, and even repeats of the *same* driver,
    # can each carry their own delay rather than sharing one embedding
    # depth/lag network-wide. Values must be <= 0 (0 = contemporaneous,
    # negative = that many rows in the past) -- a positive delay would
    # need future data that doesn't exist during a real free-running
    # Generate() call. See build_nested_mde_network()'s docstring for how
    # a node's per-driver delay is recorded and how nesting only recurses
    # through delay=0 selections. 0 is always included in every node's
    # own self-embedding regardless of whether it's listed here (see
    # build_network()'s self_delay_set comment) -- a node's present
    # value is the anchor SMap predicts from/to, not an optional driver
    # delay choice, so it can't be omitted even from a DELAY_SET that
    # doesn't otherwise include 0.
    cfg['DELAY_SET'] = _env_int_list('MDE_GMN_DELAY_SET', [0, -1, -3, -5, -10])
    if any(d > 0 for d in cfg['DELAY_SET']):
        raise ValueError(f"MDE_GMN_DELAY_SET must be <= 0 (no future data): {cfg['DELAY_SET']!r}")

    # ---- E/tau search range (MODE='GMN' only) --------------------------
    # Used exclusively by build_network()'s mode == 'GMN' branch (the CCM
    # interaction-matrix E/tau grid search, via target_E_tau_rho_grid()) --
    # MODE='MDE-GMN' no longer has a network-wide E/tau at all (E and each
    # driver's delay come from nested-MDE/DELAY_SET above instead; see
    # this module's SEARCH REUSE section and run_theta_search()'s
    # docstring for what replaced the old combined E/tau/theta search).
    cfg['E_RANGE'] = range(
        int(os.environ.get('MDE_GMN_E_MIN', 3)),
        int(os.environ.get('MDE_GMN_E_MAX', 16)),
        int(os.environ.get('MDE_GMN_E_STEP', 2)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('MDE_GMN_TAU_MIN', -15)),
        int(os.environ.get('MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('MDE_GMN_TAU_STEP', 2)),
    )

    # Theta (SMap's nonlinearity/localization parameter -- see
    # retune_theta_for_all_nodes()'s docstring) search range, used by
    # run_theta_search() for both MODE='MDE-GMN' and MODE='GMN'.
    cfg['THETA_RANGE'] = _env_float_list(
        'MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('MDE_GMN_RUN_MDE_DIAGNOSTICS', True)

    # ---- Parallelism ----------------------------------------------------
    # On HPC this should be SLURM_CPUS_PER_TASK, not left at each tool's
    # own package default (dimx: 5, apps.InteractionMatrix: 8, pyEDM
    # EmbedDimension/PredictNonlinear: their own internal default) -- see
    # module docstring's FIXED section for why this doesn't scale linearly.
    cfg['CORES'] = int(os.environ.get('MDE_GMN_CORES', os.cpu_count() or 1))
    # Deliberately NOT the same as CORES -- see build_network()'s
    # docstring: apps.InteractionMatrix's -cr and pyEDM.CCM()'s own
    # per-pair internal pool multiply rather than coordinate, so this
    # stays small regardless of how large CORES is.
    cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))

    # ---- Search reuse ---------------------------------------------------
    # See the module docstring's SEARCH REUSE section for the full
    # rationale; short version: reuse a past job's E/tau/theta search
    # result instead of redoing it, when settings match exactly.
    cfg['REUSE_BEST_PARAMS'] = _env_bool('MDE_GMN_REUSE_BEST_PARAMS', False)
    # Where to look for previous jobs' cache artifacts (best_params_*.json,
    # mde_selection_*.pkl, Network_*.pkl, generate_start_*.json -- see the
    # module docstring's SEARCH REUSE section). Default mirrors DATA_DIR/
    # RESULTS_DIR's own '../../data'-style fallback, but the sbatch script
    # overrides this to the durable bucket cache pool -- unlike RESULTS_DIR
    # (a /flash scratch tempdir deleted at job exit), PREV_RUNS_DIR needs
    # to point at wherever past jobs' cache output actually still exists.
    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('MDE_GMN_PREV_RUNS_DIR', '../../results'))
    # This job's own write destination for those same cache artifacts,
    # synced immediately after each cacheable stage completes (not just at
    # job end like RESULTS_DIR->$OUT_DIR) -- see save_to_cache()'s
    # docstring. Falls back to RESULTS_DIR (no eager off-scratch sync) for
    # ad hoc/local runs outside the sbatch script, where durability across
    # a crash isn't a concern.
    cfg['RUN_CACHE_DIR'] = Path(os.environ.get('MDE_GMN_RUN_CACHE_DIR', cfg['RESULTS_DIR']))
    cfg['RUN_CACHE_DIR'].mkdir(parents=True, exist_ok=True)

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg


# =============================================================================
# MDE (mirrors notebook cells 2-5)
# =============================================================================

def region_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]


def observable_columns(df, cap):
    return (
        region_columns(df, 'CA1_', cap)
        + region_columns(df, 'mPFC_', cap)
        + region_columns(df, 'OFC_', cap)
    )


DELAY_MARKER = '::delay'


def _delayed_column_name(observable, delay):
    return f'{observable}{DELAY_MARKER}{delay}'


def _parse_delayed_column_name(name):
    observable, _, delay_str = name.rpartition(DELAY_MARKER)
    return observable, int(delay_str)


def observable_and_delay(column_name, raw_columns):
    '''Maps a (possibly delay-expanded) candidate column name back to its
    underlying (observable, delay) pair. Delay=0 candidates are just the
    raw observable name itself (see expand_delay_columns()'s docstring
    for why that delay isn't also given a suffixed alias) -- checked
    against `raw_columns` rather than by string shape alone, in case a
    real channel name were ever to collide with the delay-suffix marker.
    '''
    if column_name in raw_columns:
        return column_name, 0
    return _parse_delayed_column_name(column_name)


def expand_delay_columns(df, columns, delay_set):
    '''Adds one new column per (observable, delay) pair in `columns` x
    `delay_set` (delay=0 excluded -- the observable's own raw column
    already IS that pair, no separate alias needed) to a copy of `df`,
    each holding that observable's value abs(delay) rows in the past
    (pandas .shift() -- every delay is <= 0, see load_config()'s
    DELAY_SET docstring). This is what lets dimx.MDE's forward selection
    treat each (observable, delay) combination as its own independent
    candidate driver, instead of being stuck with one fixed lag applied
    uniformly to every column -- see build_nested_mde_network()'s
    docstring for how the selected pairs become graph edges.

    Returns (expanded_df, expanded_columns) -- expanded_columns is
    `columns` (representing delay=0) plus one new name per nonzero delay,
    in the format observable_and_delay() parses back.'''
    expanded = df.copy()
    expanded_columns = list(columns)
    for observable in columns:
        for delay in delay_set:
            if delay == 0:
                continue
            name = _delayed_column_name(observable, delay)
            expanded[name] = df[observable].shift(-delay)
            expanded_columns.append(name)
    return expanded, expanded_columns


def effective_mde_dimension(mde_d, candidate_columns, num_delays):
    '''MDE_D names the desired count of distinct *real* observables
    dimx.MDE should be able to select -- not delay-expanded columns. Each
    real observable occupies `num_delays` separate candidate columns (one
    per DELAY_SET entry, see expand_delay_columns()), so the D actually
    handed to dimx.MDE is scaled up by num_delays: even in the worst case
    of every selection slot landing on a different delay of the same
    observable before forward selection moves on to the next one, mde_d
    distinct real observables are still guaranteed to be reachable.

    Capped at the number of real observables actually present in this
    call's own candidate pool (`len(candidate_columns) // num_delays` --
    exact, not approximate: expand_delay_columns() always emits exactly
    num_delays columns per real observable, and candidate-pool filtering
    upstream (observable_and_delay()-based) only ever drops a real
    observable's full column group at once, never a partial one) -- so
    setting MDE_D larger than the pool's real observable count (e.g. 999
    against a 500-observable pool) just means "select all of them" rather
    than inflating D past what the pool can supply.'''
    num_real_observables_available = len(candidate_columns) // num_delays
    return min(mde_d, num_real_observables_available) * num_delays


def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                        mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                        cores, results_dir, delay_set, tp=1,
                        reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                        data_file=None, data_file_sha256=None, code_fingerprint=None):
    '''Run dimx.MDE's forward selection for a single target variable.
    `mde_d` here is already the *effective* dimension count (see
    effective_mde_dimension()) -- callers compute that from the
    real-observable-count MDE_D before calling this, since only they know
    each call's own candidate pool / DELAY_SET. `tp` (default 1, matching
    dimx.MDE's own default) is the prediction interval every candidate
    driver is ranked/validated against -- see load_config()'s TP
    docstring for why this is swept alongside MDE_D. `exclusion_radius` is
    only actually used as-is when MDE_GMN_EXCLUSION_RADIUS is a fixed
    integer -- when it's 'auto' (the default), the module-level
    _EXCLUSION_RADIUS_AUTO-gated monkeypatches (see that flag's docstring
    near the top of this file) override whatever value dimx.MDE is
    called with here on every one of its own internal
    CrossMapColumns/EmbedDimension/CCM calls, so this argument is a
    harmless placeholder (0) in that case. See
    notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
    original single-delay rationale (identical logic otherwise, just
    cores threaded through).

    CACHE (reuse_cache, gated by MDE_GMN_REUSE_BEST_PARAMS same as every
    other stage -- see the module docstring's SEARCH REUSE section):
    every downstream consumer of the returned 'm' object only ever reads
    m.MDEcolumns / m.MDErho / m.MDEOut / m.args.lib / m.args.pred /
    m.args.tau / m.args.Tp (confirmed by exhaustive search across this
    file) -- never anything else, never a second .Run(). So a cache hit
    reconstructs a types.SimpleNamespace exposing exactly those seven
    attributes instead of a live dx.MDE object, and skips calling
    dx.MDE(...).Run() entirely. Called both from main()'s top-level
    per-target loop and from every per-node wave inside
    build_nested_mde_network()'s ProcessPoolExecutor, so caching here
    covers both at full per-node granularity for free.'''
    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]

    # expand_delay_columns() shifts each observable forward by abs(delay)
    # rows, leaving that many leading rows NaN (nothing precedes the start
    # of the recording to shift in). Starting `lib` at row 1 regardless
    # used to let those NaNs into the library window -- harmless for MDE's
    # own NaN-tolerant pyEDM calls, but fatal once mde_df's library slice
    # reaches run_mde_evaluate()'s PCA fit (sklearn has zero NaN
    # tolerance). Skipping the warm-up rows keeps every delay column fully
    # populated for the entire library range.
    warmup = max((abs(d) for d in delay_set), default=0)

    fingerprint = None
    cache_filename = f'mde_selection_{target}.pkl'
    if reuse_cache:
        fingerprint = build_mde_selection_fingerprint(
            target, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            data_file, data_file_sha256, code_fingerprint,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
                           load_one=_load_pickle_cache_entry)
        if hit is not None:
            entry, source = hit
            print(f'{target}: MDE_GMN_REUSE_BEST_PARAMS matched cached MDE selection '
                  f'{source} -- skipping forward-selection entirely', flush=True)
            m = types.SimpleNamespace(
                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
                                            tau=entry['tau'], Tp=entry['Tp']),
            )
            m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)
            selected_columns = list(m.MDEcolumns)
            print(f'MDE selected columns for {target}:', selected_columns, flush=True)
            return {
                'target': target,
                'mde_df': mde_df,
                'm': m,
                'SELECTED_COLUMNS': selected_columns,
            }
        print(f'{target}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached MDE selection '
              f'found under {prev_runs_dir} -- running forward-selection.', flush=True)

    m = dx.MDE( mde_df, target = target, D = mde_d,
                removeColumns = [target],
                lib  = [warmup + 1, prediction_start],
                pred = [prediction_start + 1, prediction_start + prediction_length],
                Tp = tp,
                embedDimRhoMin = embed_dim_rho_min,
                crossMapRhoMin = cross_map_rho_min,
                exclusionRadius = exclusion_radius,
                cores = cores )

    m.Run()

    m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)

    selected_columns = list(m.MDEcolumns)
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    if fingerprint is not None:
        save_pickle_to_cache(
            cache_filename,
            {
                'fingerprint': fingerprint,
                'MDEcolumns': m.MDEcolumns, 'MDErho': m.MDErho, 'MDEOut': m.MDEOut,
                'lib': m.args.lib, 'pred': m.args.pred, 'tau': m.args.tau, 'Tp': m.args.Tp,
            },
            results_dir, run_cache_dir,
        )

    return {
        'target': target,
        'mde_df': mde_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }


def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['m'].MDEOut
        dims = np.arange(1, len(mde_out) + 1)
        ax.plot(dims, mde_out['rho'], marker='o', label=target, color=target_colors[target])
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Cross-map rho')
    ax.set_title('MDE rho vs. dimension, per target')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_3d_mde_projection(target, second_target, mde_results, df, raw_columns,
                            prediction_start, prediction_length, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    # SELECTED_COLUMNS can now be delay-suffixed names (see
    # expand_delay_columns()'s docstring) that don't exist in the raw
    # `df` this function indexes into -- mapped back to their underlying
    # raw observable (the delay itself isn't meaningful for this plot,
    # a visual trajectory of the real, undelayed driver values).
    obs = list(dict.fromkeys(
        observable_and_delay(c, raw_columns)[0] for c in mde_results[target]['SELECTED_COLUMNS']
    ))[:2]
    if len(obs) < 2:
        print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.', flush=True)
        return

    window = df.iloc[:prediction_start + prediction_length]
    x = window[target].to_numpy()
    y = window[obs[0]].to_numpy()
    z = window[obs[1]].to_numpy()

    if second_target and second_target in window.columns:
        color_vals = window[second_target].to_numpy()
        color_label = second_target
    else:
        color_vals = np.arange(len(window))
        color_label = 'time index'

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='gray', linewidth=0.6, alpha=0.4)
    scatter = ax.scatter(x, y, z, c=color_vals, cmap='viridis', s=10)
    ax.set_xlabel(target)
    ax.set_ylabel(obs[0])
    ax.set_zlabel(obs[1])
    ax.set_title(f'MDE projection: {target} vs. its top-2 observables')
    fig.colorbar(scatter, ax=ax, shrink=0.7, label=color_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN network construction (mirrors notebook cell 7)
# =============================================================================

def _lib_train_test_ranges(n_rows, train_frac=0.7):
    split = int(n_rows * train_frac)
    return f'1 {split}', f'{split + 1} {n_rows}'


def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                S = Simplex(
                    dataFrame = lib_df, columns = target, target = target,
                    E = E, tau = tau, lib = lib_range, pred = pred_range,
                    embedded = False, showPlot = False,
                )
                rho_grid[i, j] = ComputeError(S['Observations'], S['Predictions'])['rho']
            except Exception:
                pass
    return rho_grid


def estimate_node_dimension(lib_df, node, cores, max_e=15, fallback=4):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = EmbedDimension(
            dataFrame = lib_df, columns = node, target = node,
            maxE = max_e, lib = lib_range, pred = pred_range, showPlot = False,
            numProcess = cores,
        )
        return int(ed.loc[ed['rho'].idxmax(), 'E'])
    except Exception:
        return fallback


def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
                              prediction_start, prediction_length, nested_n, nested_m,
                              embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
                              results_dir, mde_results, delay_set, tp=1,
                              reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                              data_file=None, data_file_sha256=None, code_fingerprint=None):
    '''Nested-MDE network construction -- MODE='MDE-GMN' only. MODE='GMN'
    still uses the original CCM-across-the-full-observable-pool path in
    build_network(), untouched.

    No notebook precedent: replaces "CCM over the union of each target's
    MDE-selected columns" with an algorithm built purely from repeated
    dimx.MDE calls (no CCM, no apps.InteractionMatrix here). Per explicit
    direction, this searches a *generalized* (non-uniform, multi-delay)
    embedding rather than a single contemporaneous one: `expanded_df`
    (built by expand_delay_columns() in main(), from DELAY_SET) offers
    each raw observable at every delay in DELAY_SET as its own separate
    candidate column, so dimx.MDE's forward selection can pick specific
    (observable, delay) pairs -- e.g. CA1_55 at delay -5 and CA1_78 at
    delay -1 as two independent drivers of the same node, rather than
    every driver sharing one embedding depth/lag network-wide (Deyle &
    Sugihara 2011's generalized embedding theorem). `raw_columns` is the
    unexpanded observable list, needed to map a selected (possibly
    delay-suffixed) column name back to its underlying observable via
    observable_and_delay(), and to exclude a node's own raw+delayed
    variants from its own candidate pool.

      Step 1 (depth 0): for each target, dimx.MDE selects up to
        `nested_n` (== MDE_D) distinct *real* observables from
        `expanded_columns` (excluding the target's own raw+delayed
        variants) -- the actual D handed to dimx.MDE is
        effective_mde_dimension(nested_n, candidate_pool, num_delays), not
        nested_n itself, since each real observable spans num_delays
        candidate columns; see that function's docstring. Reuses the
        caller's precomputed mde_results entry when available -- main()
        already ran this once per target for the MDE-only diagnostics
        (against the same expanded pool), so it isn't paid for twice.
      Step 2 (depth 1..nested_m): each driver selected *at delay 0* (see
        below for why only these) becomes a new MDE target in turn,
        again selecting up to `nested_n` real observables from the same
        expanded pool (excluding itself), for `nested_m` further rounds
        -- total graph depth from any target is 1 + nested_m levels.
        nested_m == 0 is a sentinel for "uncapped": the depth check is
        skipped entirely and nesting continues until it runs out on its
        own (every reachable real observable already visited, or one of
        the ordinary BFS termination conditions below), rather than
        stopping after a fixed number of rounds.

    Per explicit direction, only delay=0 selections continue the nested
    search -- a driver selected at a nonzero delay is added to the graph
    as a terminal edge, never itself MDE'd as a new target. A delayed
    coordinate stands in for an unobserved dimension of the true state
    space (a "placeholder" for something the recording didn't directly
    measure, per generalized embedding theory), not a physical variable
    whose own drivers are worth discovering -- so nesting into it would
    search for the causal history of a lag, not of a real signal. Nodes
    reached only via delay!=0 edges still simulate fine in gmn: they
    always get an unconditional self-embedding (see build_network()'s
    self_delay_set docstring) even with zero further discovered drivers,
    the same fallback every driverless leaf node already relies on.

    A branch's nesting stops before nested_m (or, when nested_m == 0,
    stops at all), as ordinary BFS termination, when:
      - A node's own dimx.MDE call selects zero drivers (embedDimRhoMin/
        crossMapRhoMin/the CCM slope gate rejected every candidate) --
        nothing gets enqueued for the next wave.
      - A selected driver's marginal MDE rho gain is NaN (a degenerate
        correlation, not the same as dimx.MDE's own gates rejecting it --
        it already passed those, or it wouldn't be in `cols`). Since this
        function's graph IS the final network (build_network() hands it
        to gmn.GMN directly -- no downstream re-filtering by weight), an
        edge with no trustworthy strength must be rejected here or it
        becomes a real edge with a nonsensical weight; nesting deeper
        through it would also be wasted work. `driver` can still enter
        the graph later via a different, valid edge. If every driver a
        node selected is NaN, that node is terminal -- the same
        empty-enqueue outcome as above.
      - Every driver a node selected was picked at a nonzero delay (the
        new rule above) -- also the same empty-enqueue outcome.

    Cycle/self-loop guarding happens eagerly during discovery: each
    proposed edge is added to a networkx.MultiDiGraph shared across this
    whole call, then checked with nx.is_directed_acyclic_graph()
    immediately -- a cycle-causing edge is removed by its specific key
    (MultiDiGraph.remove_edge() removes an arbitrary parallel edge if no
    key is given, which would risk removing a different, already-valid
    delay edge between the same driver/node pair). This is the only such
    guard for MODE='MDE-GMN' (no apps.CreateNetwork call follows), and
    dimx.MDE itself has no notion of the network being built around it,
    so a cycle has to be caught here or never. Self-loop exclusion is
    redundant with dimx.MDE's own removeColumns=[target]-equivalent
    exclusion (observable_and_delay()-filtered candidate pool) in the
    ordinary case, kept as an explicit guard anyway. A MultiDiGraph is
    used (rather than DiGraph) specifically so the same observable can
    drive one node at multiple delays simultaneously, each its own edge
    -- per explicit direction, this is expected and kept, not deduped.

    Each unique node is MDE'd at most once (memoized via `visited`); it
    can still gain further incoming edges later (ordinary DAG
    driver-sharing), just isn't re-selected as an MDE target.

    Parallelization: every node at the same BFS depth is MDE-independent
    of every other (each dimx.MDE call only reads the static
    expanded_df/expanded_columns pool). So each "wave" (not-yet-visited
    nodes newly discovered at one depth) is submitted concurrently via
    ProcessPoolExecutor, one process per node. Edge-acceptance/cycle-
    checking stays sequential once a wave's results are back -- cheap
    graph bookkeeping that genuinely needs one consistent MultiDiGraph,
    nothing to gain from parallelizing.

    Outer (per-node) and inner (each node's own CrossMapColumns)
    parallelism are split adaptively per wave rather than both left at
    `cores` -- that would reproduce the outer x inner pool-squaring
    blowup this project hit once already (see the module docstring's
    FIXED section). outer_workers = min(len(wave), cores), inner_cores =
    max(1, cores // outer_workers): a wide wave leans on outer
    parallelism with each MDE call single-threaded; a narrow wave gives
    that one call the full remaining core budget. outer_workers *
    inner_cores never exceeds `cores`. dimx.MDE's own CCM validation is
    forced sequential by `_sequential_ccm_workers`, but that patch
    doesn't reliably survive into a ProcessPoolExecutor worker on its own
    (this cluster's 'forkserver' start method can predate the patch), so
    the wave pool passes `initializer=_mde_wave_worker_init` to reapply
    it inside every worker -- see that function's docstring for the
    crash this avoids.

    Edge weight is the *marginal* MDE rho gain at the step that driver
    was selected (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) --
    not the raw cumulative MDEOut['rho'] (dimx/Run.py cross-maps
    columns=[candidate]+already-selected, so cumulative rho trends
    upward with d regardless of any one driver's importance, which would
    make later, weaker drivers look equal-or-stronger than the first).
    The marginal gain isolates what that driver specifically added on
    top of the previous d-1 -- what "edge strength" should mean here.

    Returns (graph, edges_df, nested_mde_results):
      graph      -- networkx.MultiDiGraph, acyclic, no self-loops. Every
                 edge has 'delay' (<= 0) and 'weight' attributes. The
                 actual network build_network() hands to gmn.GMN (see
                 that function's docstring) -- apps.CreateNetwork is not
                 involved.
      edges_df   -- DataFrame with columns ['driver', 'node', 'delay',
                 'weight'], one row per edge -- an NxN interaction matrix
                 can't represent a node having multiple edges from the
                 same driver at different delays, so this replaces that
                 format. Exported to CSV by build_network() as a
                 human-readable record alongside the graph; not fed to
                 apps.CreateNetwork.
      nested_mde_results -- {node: run_mde_for_target() result} for every
                 node this call ran dimx.MDE on. Kept separate from the
                 caller's own mde_results so repeated separate-mode calls
                 can't cross-contaminate a shared intermediate channel
                 name.
    '''
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    # expanded_columns is raw_columns with num_delays-1 extra columns
    # appended per real observable (see expand_delay_columns()) -- exact,
    # not approximate, so this recovers num_delays without needing
    # delay_set threaded through as its own parameter. Used by
    # effective_mde_dimension() below to convert nested_n (a real-
    # observable count) into the actual D handed to dimx.MDE.
    num_delays = len(expanded_columns) // len(raw_columns)

    edges = []                # list of {driver, node, delay, weight} dicts
    nested_mde_results = {}   # node -> run_mde_for_target() result, this call only
    visited = set()
    current_wave = [(t, 0) for t in targets]

    while current_wave:
        # Dedup within the wave itself -- two different nodes discovered
        # in the *previous* wave can independently propose the same
        # not-yet-visited driver as their own new MDE target; each
        # unique node still only gets MDE'd once.
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        # Depth-0 targets reuse the caller's precomputed mde_results
        # directly (see docstring) -- no need to pay for, or parallelize,
        # that MDE call again.
        to_compute = [(node, depth) for node, depth in wave_nodes
                      if not (depth == 0 and node in mde_results)]
        wave_results = {
            node: mde_results[node] for node, depth in wave_nodes
            if depth == 0 and node in mde_results
        }

        if to_compute:
            outer_workers = min(len(to_compute), max(1, cores))
            inner_cores = max(1, cores // outer_workers)
            print(f'nested MDE: wave of {len(to_compute)} node(s) at depth '
                  f'{to_compute[0][1]} -- {outer_workers} parallel worker(s) '
                  f'x {inner_cores} core(s) each', flush=True)

            with ProcessPoolExecutor(max_workers=outer_workers,
                                      mp_context=mp.get_context('fork'),
                                      initializer=_mde_wave_worker_init) as executor:
                futures = {}
                for node, depth in to_compute:
                    candidate_pool = [c for c in expanded_columns
                                       if observable_and_delay(c, raw_columns)[0] != node]
                    effective_d = effective_mde_dimension(nested_n, candidate_pool, num_delays)
                    fut = executor.submit(
                        run_mde_for_target, node, expanded_df, candidate_pool,
                        prediction_start, prediction_length, effective_d,
                        embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                        inner_cores, results_dir, delay_set, tp,
                        reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                        code_fingerprint,
                    )
                    futures[fut] = node
                for fut in as_completed(futures):
                    wave_results[futures[fut]] = fut.result()

        next_wave = []
        for node, depth in wave_nodes:
            visited.add(node)
            result = wave_results[node]
            nested_mde_results[node] = result

            cols = result['SELECTED_COLUMNS']
            gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))

            for col, gain in zip(cols, gains):
                driver, delay = observable_and_delay(col, raw_columns)

                if driver == node:
                    continue  # explicit self-loop guard (belt-and-suspenders)

                if np.isnan(gain):
                    # Degenerate marginal-gain computation -- dead-end
                    # this specific edge without nesting further through
                    # it (see this function's docstring for why NaN
                    # weights can't just be added anyway).
                    print(f'nested MDE: rejected {driver} (delay {delay}) -> {node} '
                          f'(marginal MDE rho gain is NaN -- treating as a dead-end edge, '
                          f'no further nesting from this driver via this path)', flush=True)
                    continue

                graph.add_node(driver)
                key = graph.add_edge(driver, node, delay=delay, weight=gain)

                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(driver, node, key=key)
                    if graph.degree(driver) == 0:
                        graph.remove_node(driver)
                    print(f'nested MDE: rejected {driver} (delay {delay}) -> {node} '
                          f'(would create a cycle)', flush=True)
                    continue

                edges.append({'driver': driver, 'node': node, 'delay': delay, 'weight': gain})

                # Only a delay=0 (contemporaneous) selection continues the
                # nested search -- see this function's docstring for why a
                # delayed coordinate is a terminal placeholder, not a new
                # MDE target. nested_m == 0 is the "uncapped" sentinel --
                # see this function's docstring -- so the depth check is
                # skipped entirely in that case and nesting only stops via
                # the other ordinary BFS termination conditions (empty
                # wave once every reachable real observable is visited,
                # zero-driver/NaN/nonzero-delay/cycle rejections).
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited:
                    next_wave.append((driver, depth + 1))

        current_wave = next_wave

    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])

    nested_m_display = 'unlimited' if nested_m == 0 else nested_m
    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
          f'(N={nested_n}, M={nested_m_display})', flush=True)

    return graph, edges_df, nested_mde_results


def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
                   observable_columns=None, nested_n=None, nested_m=None,
                   embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
                   expanded_df=None, expanded_observable_columns=None, delay_set=None, tp=1,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15,
                   reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                   data_file=None, data_file_sha256=None, code_fingerprint=None):
    '''Build one network spanning `targets` and their driver candidates,
    via one of two unrelated constructions depending on `mode`:
      - mode == 'MDE-GMN': build_nested_mde_network()'s own graph IS the
        final network, handed to gmn.GMN directly -- no
        apps.CreateNetwork() call. Per explicit direction: nested-MDE
        already does the network-discovery job CreateNetwork exists for
        (BFS out from targets, rejecting cycles as edges are proposed),
        using real MDE-driven edge selection, so re-deriving a network
        from an interaction matrix afterward would just be redundant
        work. Safe because gmn.Network only ever reads the pickled
        network file's 'Graph' entry (gmn/Network.py:43-46) via
        Graph.predecessors(node) (gmn/Node.py:104) -- the 'Map' entry
        CreateNetwork also writes is explicitly unused (that file's own
        comment) -- so a {'Graph': graph, 'Map': ...} pickle built
        directly from build_nested_mde_network()'s MultiDiGraph is
        exactly what CreateNetwork would have handed back anyway.
        observable_columns/nested_n/nested_m/embed_dim_rho_min/
        cross_map_rho_min/expanded_df/expanded_observable_columns/
        delay_set are required in this mode -- expanded_df/
        expanded_observable_columns are built once in main() via
        expand_delay_columns() (see build_nested_mde_network()'s
        docstring for why the search needs a delay-expanded pool rather
        than the raw `df`/`observable_columns`, which are still used
        as-is below for the real data every node actually simulates
        from). delay_set is stored on the returned graph
        (graph.graph['self_delay_set']) so gmn's own generalized-
        embedding patches (see the module docstring) know which delays
        to always give each node's own self-history, regardless of
        what nested-MDE discovered for its external drivers.
        node_dimensions is NOT computed here (see gmn_fidelity_rho()'s
        default_E fallback for its one remaining consumer).
      - mode == 'GMN': the original CCM interaction matrix (data-driven
        E/tau grid search over the REGION_CHANNEL_CAP-limited observable
        pool, via apps.InteractionMatrix), then a node-dimension estimate
        and apps.CreateNetwork with its driver-ranking-bug workaround --
        unchanged, since there's no MDE-discovered graph to substitute
        and CreateNetwork's own construction is still needed.

    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network() for
    the mode == 'GMN' branch.

    interaction_matrix_outer_cores is a *separate* knob from `cores`:
    apps.InteractionMatrix's -cr sizes an outer pool across column pairs,
    and its internal pyEDM.CCM() call used to spawn its *own* fresh pool
    per pair sized to cpu_count() (confirmed by reading pyEDM/CCM.py and
    apps/InteractionMatrix.py, neither of which passes `parallel=`).
    Peak concurrent processes was -cr * cpu_count(), not -cr -- setting
    -cr = cores (the natural-seeming choice) squared your core request
    into a process count (-cr=cores on a 10-core machine produced ~90
    live processes). This module now monkeypatches pyEDM.CCM's inner
    pool away entirely (`_sequential_ccm_workers` near the top of this
    file) rather than just avoiding the multiplication -- every CCM()
    call here only ever resolves to 4-8 tiny tasks, so the inner pool was
    pure overhead. interaction_matrix_outer_cores stays a separate knob
    regardless, since it's still what sizes real parallelism across
    column pairs.
    '''
    network_id = '_'.join(targets)
    # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
    # artifacts for the same targets -- these are genuinely different
    # networks (full raw channel pool vs. nested-MDE-selected columns)
    # and must not share a checkpoint file (see the search function's
    # checkpoint resume logic, which only validates e_vals/tau_vals/
    # theta_vals, not network topology).
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    ccm_E = ccm_tau = None
    node_dimensions = {}

    # CACHE (reuse_cache, gated by MDE_GMN_REUSE_BEST_PARAMS same as every
    # other stage -- see the module docstring's SEARCH REUSE section):
    # this is the single most expensive stage in the whole pipeline
    # (build_nested_mde_network()'s BFS-over-hundreds-of-nodes for
    # MODE='MDE-GMN', or the apps.InteractionMatrix CCM subprocess for
    # MODE='GMN'), so a fingerprint match here skips it entirely rather
    # than just resuming partway through. Computed *before* construction
    # (unlike build_search_fingerprint(), which needs the already-built
    # network for its node-set sanity check) since the whole point is
    # deciding whether to build it at all.
    fingerprint = None
    cache_filename = f'network_cache_{network_id}.pkl'
    cache_hit = None
    if reuse_cache:
        fingerprint = build_network_fingerprint(
            mode, targets, observable_columns, nested_n, nested_m, tp,
            embed_dim_rho_min, cross_map_rho_min, exclusion_radius, delay_set,
            e_range, tau_range, prediction_start, prediction_length,
            data_file, data_file_sha256, code_fingerprint,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint, load_one=_load_pickle_cache_entry)
        if hit is not None:
            cache_hit, source = hit
            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched cached network {source} '
                  f'-- skipping network construction entirely', flush=True)
        else:
            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached network '
                  f'found under {prev_runs_dir} -- building from scratch.', flush=True)

    if mode == 'MDE-GMN':
        if cache_hit is not None:
            graph = cache_hit['graph']
        else:
            graph, _, _ = build_nested_mde_network(
                targets, expanded_df, observable_columns, expanded_observable_columns,
                prediction_start, prediction_length,
                nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
                results_dir, mde_results, delay_set, tp,
                reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                code_fingerprint,
            )
        # 0 is forced into self_delay_set regardless of DELAY_SET: gmn's
        # own SMap target-lookup convention (Parameters.target + '(t-0)',
        # mirrored by this pipeline's _generalized_node_forecast/Generate)
        # requires a node's own present-value coordinate to exist in its
        # embedding -- this isn't a driver delay choice, it's the anchor
        # point being predicted, so it can't be left out even if a user
        # configures a DELAY_SET without 0 in it.
        graph.graph['self_delay_set'] = sorted(set(delay_set) | {0}, reverse=True)
        selected_union = [n for n in graph.nodes if n not in targets]
        # Rebuilt from the graph itself (rather than build_nested_mde_
        # network()'s own returned edges_df) so this works identically on
        # both the cache-hit and freshly-built path -- every edge that
        # function's own BFS discovers and keeps is exactly what ends up
        # in `graph` (cycle-rejected edges are never added in the first
        # place), so the two are equivalent in content either way.
        edges_file = results_dir / f'edges_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.csv'
        edges_rows = [{'driver': u, 'node': v, 'delay': d['delay'], 'weight': d['weight']}
                      for u, v, d in graph.edges(data=True)]
        pd.DataFrame(edges_rows, columns=['driver', 'node', 'delay', 'weight']).to_csv(
            edges_file, index=False)
    else:  # mode == 'GMN'
        if cache_hit is not None:
            network = cache_hit['network']
            ccm_E, ccm_tau, node_dimensions = (
                cache_hit['ccm_E'], cache_hit['ccm_tau'], cache_hit['node_dimensions'])
        selected_union = []
        for t in targets:
            for c in mde_results[t]['SELECTED_COLUMNS']:
                if c not in selected_union:
                    selected_union.append(c)

    data_after_mde = (
        df[targets + selected_union]
        .iloc[:prediction_start + prediction_length]
        .reset_index(drop=True)
    )
    data_after_mde.insert(0, 'Time', np.arange(len(data_after_mde)))
    data_after_mde_file = results_dir / f'data_after_mde_{network_id}.csv'
    data_after_mde.to_csv(data_after_mde_file, index=False)

    data_after_mde_lib = data_after_mde.iloc[:prediction_start]
    data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
    data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)

    if mode == 'MDE-GMN':
        # build_nested_mde_network()'s graph already IS the network --
        # see this function's docstring for why apps.CreateNetwork() adds
        # nothing here. 'Map' is written only because CreateNetwork's own
        # pickle format includes it and downstream code below prints from
        # it; gmn itself never reads it (see docstring). Rebuilt from
        # `graph` unconditionally (not skipped on a cache hit) since it's
        # a trivial dict comprehension -- not worth a second code path.
        network_dict = {node: list(graph.predecessors(node)) for node in graph.nodes}
        network = {'Graph': graph, 'Map': network_dict}
        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
        with open(network_file, 'wb') as f:
            pickle.dump(network, f)

        if fingerprint is not None and cache_hit is None:
            save_pickle_to_cache(cache_filename, {'fingerprint': fingerprint, 'graph': graph},
                                  results_dir, run_cache_dir)
    else:  # mode == 'GMN'
        if cache_hit is None:
            e_vals, tau_vals = list(e_range), list(tau_range)
            target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals)
                             for t in targets]
            mean_grid = np.nanmean(np.stack(target_grids), axis=0)
            best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
            ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
            print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
                  f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)

            interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
            # apps.InteractionMatrix is a CLI-only tool (no in-memory API)
            # -- invoked via subprocess instead of the notebook's IPython
            # get_ipython().system(), which only exists inside a live
            # kernel. MPLBACKEND=Agg keeps -P from trying to open a GUI
            # window.
            subprocess_env = dict(os.environ, MPLBACKEND='Agg')
            subprocess.run(
                [
                    'python', '-m', 'apps.InteractionMatrix',
                    '-d', str(data_after_mde_lib_file),
                    '-oc', str(interaction_prefix),
                    '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
                    '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
                    '-P', '-cr', str(interaction_matrix_outer_cores),
                ],
                check=True, env=subprocess_env,
            )

            interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
            iMatrix = pd.read_csv(interaction_csv, index_col=0)

            node_dimensions = {
                node: estimate_node_dimension(data_after_mde_lib, node, cores,
                                               max_e=max_node_dimension, fallback=fallback_node_dimension)
                for node in iMatrix.index
            }
            print(f'{network_id}: node N_D =', node_dimensions, flush=True)

            # apps.CreateNetwork() keeps each node's *weakest* above-
            # threshold drivers instead of the strongest (confirmed
            # against the installed apps/CreateNetwork.py:110-116 -- it
            # sorts each row ascending, then truncates to numDrivers).
            # Worked around by pre-selecting the strongest N_D drivers
            # ourselves, then calling CreateNetwork with numDrivers set to
            # the full column count so its buggy truncation never has
            # anything left to remove.
            strongest_iMatrix = iMatrix.copy()
            for node in strongest_iMatrix.index:
                row = strongest_iMatrix.loc[node].drop(node)
                n_drivers = max(1, min(node_dimensions[node], len(row)))
                keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
                drop = row.index.difference(keep)
                strongest_iMatrix.loc[node, drop] = threshold

            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
            network = CreateNetwork(
                interactionMatrix = strongest_iMatrix,
                targetCols        = list(targets),
                threshold         = threshold,
                numDrivers        = len(strongest_iMatrix.columns),
                outputFile        = str(network_file),
                plotNetwork       = False,
            )
        else:
            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'

        # gmn.Network.__init__ reloads the graph fresh from network_file's
        # own pickle (gmn/Network.py:43-46), not from this in-memory
        # `network` dict -- CreateNetwork already wrote that file above
        # (fresh path only), so fallback_E/fallback_tau have to be added
        # to the graph and the file (re-)pickled here regardless of
        # cache-hit/fresh, or _node_embedding_spec()'s plain-DiGraph
        # fallback (see its docstring) would never see them.
        network['Graph'].graph['fallback_E'] = ccm_E
        network['Graph'].graph['fallback_tau'] = ccm_tau
        with open(network_file, 'wb') as f:
            pickle.dump(network, f)

        if fingerprint is not None and cache_hit is None:
            save_pickle_to_cache(cache_filename, {
                'fingerprint': fingerprint, 'network': network,
                'ccm_E': ccm_E, 'ccm_tau': ccm_tau, 'node_dimensions': node_dimensions,
            }, results_dir, run_cache_dir)

    for t in targets:
        print(f'{t} parents:', network['Map'][t], flush=True)

    return {
        'targets': list(targets),
        'network_id': network_id,
        'DATA_AFTER_MDE': data_after_mde_file,
        'NETWORK_FILE': network_file,
        'network': network,
        'ccm_E': ccm_E,
        'ccm_tau': ccm_tau,
        'node_dimensions': node_dimensions,
    }


def draw_colored_network(net_result, target_colors, save_path):
    '''Draws the network graph, nodes colored by which target ultimately
    depends on them (unchanged). Edges are colored by delay, with a
    legend mapping color -> delay, whenever the graph carries per-edge
    delay data (a MultiDiGraph built by build_nested_mde_network() for
    MODE='MDE-GMN' -- see that function's docstring on why the same
    observable can drive one node via several parallel edges, each its
    own delay). MODE='GMN' networks (a plain DiGraph from
    apps.CreateNetwork, no delay concept at all) fall back to the
    original by-target edge coloring instead.

    Edges are drawn individually (nx.draw_networkx_edges per edge, not
    one nx.draw_networkx call) with a curvature that increases per
    parallel edge -- otherwise multiple edges between the same two nodes
    (the whole point of a MultiDiGraph here) would overlap into one
    indistinguishable line.'''
    graph = net_result['network']['Graph']
    targets = net_result['targets']
    has_delays = isinstance(graph, nx.MultiDiGraph)

    node_colors = [target_colors.get(n, 'lightgray') for n in graph.nodes]

    pos = nx.arf_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=300, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=9, font_weight='bold')

    if has_delays:
        all_delays = sorted({data['delay'] for _, _, data in graph.edges(data=True)})
        cmap = plt.get_cmap('viridis')
        delay_color = {d: cmap(i / max(1, len(all_delays) - 1)) for i, d in enumerate(all_delays)}

        parallel_count = defaultdict(int)
        for u, v, data in graph.edges(data=True):
            i = parallel_count[(u, v)]
            parallel_count[(u, v)] += 1
            rad = 0.15 * ((i + 1) // 2) * (1 if i % 2 == 0 else -1)
            nx.draw_networkx_edges(
                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
                edge_color=[delay_color[data['delay']]],
                connectionstyle=f'arc3,rad={rad}',
            )

        edge_handles = [
            plt.Line2D([0], [0], color=delay_color[d], lw=2, label=f'delay={d}')
            for d in all_delays
        ]
        legend_title = 'Edge delay / target'
    else:
        for u, v in graph.edges():
            nx.draw_networkx_edges(
                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
                edge_color=[target_colors.get(v, 'gray')],
            )
        edge_handles = []
        legend_title = 'Target'

    node_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                   markerfacecolor=target_colors[t], label=t)
        for t in targets
    ]
    ax.legend(handles=edge_handles + node_handles, loc='upper left',
              bbox_to_anchor=(1.0, 1.0), title=legend_title)
    ax.set_title(f"GMN network ({net_result['network_id']})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN node generation config (mirrors notebook cell 8)
# =============================================================================

def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                      mode='Generate', function='SMap', theta=3,
                      node_theta_overrides=None, variant='', generate_length=None,
                      generate_start=None, max_offset=None, exclusion_radius=0):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    exclusion_radius: written into the [EDM] section below and read
    straight through by gmn's own (monkeypatched) Node.Forecast/Generate
    as `Parameters.exclusionRadius`, passed to every pyEDM Simplex/SMap
    call gmn itself makes (see this pipeline's _generalized_node_generate/
    Node.Forecast monkeypatches near the top of this file) -- the same
    Theiler-window neighbor-exclusion mechanism as dimx.MDE's own
    MDE_GMN_EXCLUSION_RADIUS (see that parameter's docstring for the
    general rationale), just applied to GMN's own recursive Generate()/
    Forecast() nearest-neighbor search rather than MDE's forward
    selection -- these are two independent EDM contexts with their own
    separate embeddings, so there's no single shared "auto" value that
    would make sense for both; GMN_EXCLUSION_RADIUS is a plain static
    knob, not dynamically recomputed the way MDE's is.

    E/tau are no longer parameters here -- every node's real embedding
    now comes from _node_embedding_spec()/_embedding_spec_for()
    (per-driver delays off the network graph, plus self_delay_set or
    fallback_E/fallback_tau -- see those functions' docstrings), not a
    shared network-wide (E, tau) written into this .cfg. Placeholder
    'E'/'tau' values are still written into the [EDM] section below
    purely because gmn.Node.__init__'s own (unpatched) arithmetic reads
    them before _generalized_node_init immediately corrects
    self.libEnd_i -- see that function's docstring; their actual values
    are otherwise inert.

    generate_start (both modes): row index where Generate() starts
    free-running / where Forecast()'s pred window starts, independent of
    prediction_start -- the latter is the fixed MDE lib/pred boundary
    (see load_config()'s LIB_RATIO) that MDE variable selection and
    CCM/EmbedDimension validate against, and stays fixed regardless of
    what row a diagnostic Generate()/Forecast() call happens to start at.
    Defaults to prediction_start when None (unchanged pre-existing
    behavior). Sweeping this is how sweep_generate_start() finds the
    best-matching starting point for a fixed, already-chosen theta --
    see that function's docstring.

    generate_length (both modes): number of steps simulated/predicted
    from generate_start, independent of prediction_length -- the latter
    is also the MDE lib/pred split's held-out length, which drives MDE
    variable selection and CCM/EmbedDimension calls, none of which have
    anything to do with how many steps one Generate()/Forecast() call
    should cover. A full Generate() call over the whole held-out window
    can be the single most expensive step in the search loop (one call
    per candidate), so decoupling this lets you shrink it
    (MDE_GMN_GENERATE_LENGTH) without also shrinking the held-out window
    MDE/CCM validate against. Defaults to prediction_length when None.
    mode='Forecast' honors this same cap (rather than always spanning the
    full held-out prediction_length) so its window matches Generate()'s --
    run_forecast_diagnostic() plots both on the same axes, and a
    Forecast() line running past where Generate() stopped would have
    nothing to visually compare against.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting (confirmed
    by reading gmn/Generate.py directly), so it can't do real Takens
    delay embedding: embedded=True forces pyEDM to overwrite E with the
    raw predecessor-column count, silently discarding tau. gmn's SMap
    branch does call Embed(E, tau, columns) first (now replaced by this
    pipeline's own generalized embedding -- see the module docstring).
    Per explicit direction, this pipeline uses SMap uniformly for every
    node rather than mixing generators per node or patching Simplex.

    node_theta_overrides: optional {nodeName: theta} dict -- written as
    per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
    lookup); every other node uses the network-wide `theta`. The trailing
    '/' on configPath below works around gmn/Node.py:78's naive string
    concatenation path lookup (confirmed empirically: without it, every
    override is silently ignored).

    max_offset (mode='Forecast' only): the network's largest per-node
    embedding offset (network_max_embedding_offset()), used to size a
    single, network-wide lib/pred window valid for every node's own
    embedding. Since nodes can each have a different real offset now
    (unlike the old shared (E, tau)), a node whose own offset is smaller
    than max_offset gets evaluated a few rows further into the window
    than intended -- accepted as reasonable slop for this
    diagnostic-only step (run_forecast_diagnostic()). Generate()'s own
    per-step recursive simulation, used everywhere else, has no such
    issue: libEnd_i is computed correctly per node in
    _generalized_node_init(), not derived from a shared cfg string. This
    mirrors gmn/Forecast.py's SMap branch, which calls Embed(E, tau,
    columns) then dropna() before predicting, dropping (E-1)*abs(tau)
    leading NaN rows the time-shift introduces and shrinking the
    dataframe SMap actually sees relative to node.data's full row count
    -- confirmed empirically that an unadjusted lib/pred computed in
    original-row terms overflows the shrunk frame (pyEDM.SMap's
    CreateIndices() raises "prediction index N exceeds the number of
    data rows M" with M matching forecast_length - offset exactly), so
    this function's own lib/pred bounds are shifted back by max_offset
    to compensate.'''
    if mode not in ('Generate', 'Forecast'):
        raise ValueError(f"Unknown mode: {mode!r}")
    Tp = 1
    generate_start = generate_start if generate_start is not None else prediction_start

    if mode == 'Generate':
        gmn_prediction_start = str(generate_start)
        gmn_prediction_length = str(generate_length if generate_length is not None else prediction_length)
        lib_range = ''
        pred_range = ''
    else:
        gmn_prediction_start = '1'
        gmn_prediction_length = '1'
        offset = max_offset if max_offset is not None else 0
        # Capped to generate_length (same window Generate() free-runs
        # for), not the full held-out prediction_length -- see this
        # function's docstring for why the two modes' windows need to
        # match for run_forecast_diagnostic()'s combined plot to be
        # meaningful.
        forecast_length = generate_length if generate_length is not None else prediction_length
        if offset >= generate_start or offset >= forecast_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'leaves too few rows in generate_start={generate_start}/'
                f'forecast_length={forecast_length} for a valid lib/pred split '
                f'-- shrink MDE_GMN_DELAY_SET or grow the prediction window.'
            )
        lib_range = f'1 {generate_start - offset}'
        pred_range = f'{generate_start - offset} {generate_start + forecast_length - Tp - offset}'

    node_cfg_dir = results_dir / f"node_cfgs_{net_result['network_id']}_{variant or mode.lower()}"

    def make_sections(node_theta):
        return {
            'GMN': {
                'mode': mode,
                'predictionStart': gmn_prediction_start,
                'predictionLength': gmn_prediction_length,
                'outPath': str(results_dir),
                'dataOutFile': '',
                'showPlot': 'False',
                'plotType': 'state',
                'plotColumns': '',
                'plotFile': '',
            },
            'Network': {
                'name': 'GMN',
                'targetNode': ','.join(net_result['targets']),
                'file': str(net_result['NETWORK_FILE']),
                'data': str(net_result['DATA_AFTER_MDE']),
            },
            'Node': {
                'info': '',
                'function': function,
                'configPath': str(node_cfg_dir) + '/' if node_theta_overrides else '',
                'data': '',
            },
            'EDM': {
                'lib': lib_range,
                'pred': pred_range,
                'E': '2',    # placeholder, inert -- see this function's docstring
                'Tp': str(Tp),
                'knn': '0',
                'tau': '-1',  # placeholder, inert -- see this function's docstring
                'theta': str(node_theta),
                'exclusionRadius': str(exclusion_radius),
                'columns': '',
                'target': '',
                'solver': '',
                'embedded': 'True',
                'validLib': '',
                'generateSteps': '0',
                'libSizes': '',
                'sample': '0',
                'random': 'False',
                'includeData': 'False',
                'seed': '0',
            },
            'Scale': {'factor': '1', 'offset': '0'},
        }

    def write_config(sections, path):
        config = ConfigParser()
        for section, values in sections.items():
            config[section] = values
        with open(path, 'w') as f:
            config.write(f)

    suffix = '' if mode == 'Generate' else f'_{mode.lower()}'
    if variant:
        suffix += f'_{variant}'
    config_file = results_dir / f"gmn_network_{net_result['network_id']}{suffix}.cfg"
    write_config(make_sections(theta), config_file)

    if node_theta_overrides:
        node_cfg_dir.mkdir(parents=True, exist_ok=True)
        for node_name, node_theta in node_theta_overrides.items():
            write_config(make_sections(node_theta), node_cfg_dir / f'{node_name}.cfg')

    return config_file


# =============================================================================
# E/tau/theta search (mirrors notebook cell 10)
# =============================================================================

def _hash_file(path, chunk_size=1 << 20):
    '''sha256 of a file's actual bytes, not its name/size/mtime -- used to
    detect a same-named DATA_FILE whose content silently changed between
    job submissions, which name/size/mtime alone can miss (mtime survives
    a `cp`, size collisions are rare but real for CSVs of similar shape).
    Streamed in chunks since this is not IPC-bound, just avoiding loading
    an unnecessarily large file fully into memory at once.'''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def _strip_docstrings(tree):
    '''Removes every module/class/function docstring (the first statement
    of a body, when it's a bare string-literal expression) from an AST in
    place. Used by _pipeline_code_fingerprint() below so that hash is
    insensitive to prose -- by far the most common kind of edit to this
    file (nearly every function here carries a long "why" docstring that
    gets revised/expanded on its own, unrelated to any change in what the
    function actually computes) -- while still changing on any real
    change to a function's statements/expressions. Comments need no
    equivalent handling: they're never part of the AST at all.'''
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return tree


def _pipeline_code_fingerprint():
    '''SHA-256 of this file's own normalized AST (docstrings stripped via
    _strip_docstrings(); comments already excluded from the AST). Folded
    into every cache fingerprint (see build_code_fingerprint() and the
    module docstring's SEARCH REUSE section) because a config-knob-only
    fingerprint has no way to tell "the code that computes this stage's
    result changed" from "nothing changed" -- without this, editing the
    actual computation (fixing a bug, changing a formula, reordering a
    calculation) would silently keep matching old cached results computed
    under the previous, different logic. A pure docstring/comment edit --
    demonstrably the dominant kind of change to this file -- does NOT
    change this hash and so does not cost a cache miss; any real change
    to a statement or expression does, whole-file (not per-function): the
    simpler, safer choice over curating a per-stage dependency list that
    could silently go stale as this file grows -- see the SEARCH REUSE
    section for the "occasional unnecessary miss, never an incorrect hit"
    tradeoff this accepts, same as every other fingerprint field here.'''
    tree = ast.parse(Path(__file__).read_text())
    _strip_docstrings(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _environment_fingerprint():
    '''Captures the installed numerical/algorithmic dependencies this
    pipeline's cached stages actually run through -- an upgraded
    dimx/gmn/pyEDM/etc. version could change a cached stage's numerical
    output just as much as an edit to this file would, and neither a
    config-knob fingerprint nor _pipeline_code_fingerprint() (this file's
    own source only) can notice that. importlib.metadata.version() covers
    every properly-installed package; `apps` (apps.CreateNetwork /
    apps.InteractionMatrix -- see build_network()'s docstring, MODE='GMN'
    only) ships without package metadata in this venv, so its two
    actually-used modules are content-hashed directly instead, same
    approach as _hash_file(). Always includes every entry regardless of
    MODE, same "simpler than branching, costs only an occasional
    unnecessary miss" rationale as build_search_fingerprint()'s own
    docstring.'''
    versions = {}
    for pkg in ('dimx', 'gmn', 'pyEDM', 'networkx', 'numpy', 'pandas'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    for mod_name in ('apps.CreateNetwork', 'apps.InteractionMatrix'):
        mod = importlib.import_module(mod_name)
        versions[mod_name] = _hash_file(Path(mod.__file__))
    return versions


def build_code_fingerprint():
    '''Combines _pipeline_code_fingerprint() and _environment_fingerprint()
    into the single value every cache fingerprint embeds (see the module
    docstring's SEARCH REUSE section). Computed once per job in main()
    (gated by MDE_GMN_REUSE_BEST_PARAMS, same as data_file_sha256 --
    parsing this file's AST and querying installed package versions has a
    real, if small, cost not worth paying when caching is off) and
    threaded down as a plain argument rather than recomputed per call --
    run_mde_for_target() alone can run 400+ times in one job (once per
    nested-MDE node), so recomputing this per call would add up.'''
    return {'source': _pipeline_code_fingerprint(), 'environment': _environment_fingerprint()}


def build_search_fingerprint(cfg, net_result, data_file_sha256, code_fingerprint):
    '''Everything that determines one network's theta search outcome
    (run_theta_search()), for MDE_GMN_REUSE_BEST_PARAMS's cache-lookup
    below: the knobs that shape that network's topology (MODE/GMN_MODE/
    targets/MDE_D/NESTED_M/REGION_CHANNEL_CAP/the causality gates/
    LIB_RATIO/DELAY_SET -- the last only actually matters for
    MODE='MDE-GMN', and e_vals/tau_vals only for MODE='GMN', but
    including all of them regardless of mode is simpler than branching
    the fingerprint by mode, and only costs an occasional unnecessary
    cache miss, never an incorrect hit), the knobs that shape the search
    itself (the theta range, GENERATE_LENGTH), a hash of the actual
    input data (see _hash_file), and the network's own realized node set
    as a cheap extra check -- if every upstream knob and the data
    genuinely match, the reconstructed network (already built by the
    time this runs, cost already paid) should have exactly the same
    nodes; disagreement here would mean some other nondeterminism crept
    in upstream that the knob list alone didn't catch, and is a signal
    to NOT trust the cache rather than silently reuse a stale search
    result.

    All values here are already JSON-safe (plain str/int/float/list) so
    the returned dict can be dumped/compared directly with `==` after a
    json round-trip, with no custom (de)serialization needed.'''
    graph = net_result['network']['Graph']
    return {
        'mode': cfg['MODE'],
        'gmn_mode': cfg['GMN_MODE'],
        'targets': sorted(net_result['targets']),
        'mde_d': cfg['MDE_D'],
        'embed_dim_rho_min': cfg['EMBED_DIM_RHO_MIN'],
        'cross_map_rho_min': cfg['CROSS_MAP_RHO_MIN'],
        'exclusion_radius': 'auto' if cfg['EXCLUSION_RADIUS_AUTO'] else cfg['EXCLUSION_RADIUS_VALUE'],
        'gmn_exclusion_radius': cfg['GMN_EXCLUSION_RADIUS'],
        'nested_m': cfg['NESTED_M'],
        'region_channel_cap': cfg['REGION_CHANNEL_CAP'],
        'lib_ratio': cfg['LIB_RATIO'],
        'generate_length': cfg['GENERATE_LENGTH'],
        'delay_set': list(cfg['DELAY_SET']),
        'e_vals': list(cfg['E_RANGE']),
        'tau_vals': list(cfg['TAU_RANGE']),
        'theta_vals': list(cfg['THETA_RANGE']),
        'data_file': cfg['DATA_FILE'],
        'data_file_sha256': data_file_sha256,
        'prediction_start': cfg['PREDICTION_START'],
        'prediction_length': cfg['PREDICTION_LENGTH'],
        'network_nodes': sorted(graph.nodes),
        'code_fingerprint': code_fingerprint,
        'stage': 'theta_search',
    }


def build_network_fingerprint(mode, targets, observable_columns, nested_n, nested_m, tp,
                               embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                               delay_set, e_range, tau_range, prediction_start,
                               prediction_length, data_file, data_file_sha256, code_fingerprint):
    '''Everything that determines one group's network-construction outcome
    (build_nested_mde_network(), or the CCM/apps.CreateNetwork path for
    MODE='GMN') -- unlike build_search_fingerprint() above, computable
    *before* that construction runs, since the whole point is deciding
    whether to skip it. Takes build_network()'s own already-resolved
    parameters directly (e.g. `observable_columns` -- REGION_CHANNEL_CAP's
    effect is already fully captured by exactly which channels ended up
    in that list, so the raw cap number doesn't need its own field; same
    for LIB_RATIO via prediction_start/prediction_length) rather than raw
    cfg knobs, so this needs no `cfg` dependency. Skips
    build_search_fingerprint()'s network_nodes sanity check (nothing
    built yet to check) and GMN_MODE (irrelevant to one already-resolved
    `targets` group's own topology -- see target_groups_for_mode()).
    'stage' is what keeps this from ever being confused with a
    theta_search/mde_selection fingerprint even if the remaining fields
    happened to collide.'''
    return {
        'mode': mode,
        'targets': sorted(targets),
        'observable_columns': sorted(observable_columns),
        'mde_d': nested_n,
        'nested_m': nested_m,
        'tp': tp,
        'embed_dim_rho_min': embed_dim_rho_min,
        'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius,
        'delay_set': list(delay_set) if delay_set is not None else None,
        'e_vals': list(e_range),
        'tau_vals': list(tau_range),
        'prediction_start': prediction_start,
        'prediction_length': prediction_length,
        'data_file': data_file,
        'data_file_sha256': data_file_sha256,
        'code_fingerprint': code_fingerprint,
        'stage': 'network',
    }


def build_mde_selection_fingerprint(node, candidate_pool, mde_d, tp, embed_dim_rho_min,
                                     cross_map_rho_min, exclusion_radius, delay_set,
                                     prediction_start, prediction_length,
                                     data_file, data_file_sha256, code_fingerprint):
    '''Everything that determines one run_mde_for_target() call's outcome,
    computable before that call runs. `candidate_pool` is the exact
    (possibly delay-expanded) column list dx.MDE searched over, kept in
    order rather than sorted -- dx.MDE's forward selection can tie-break
    on column order, so the same columns in a different order aren't
    guaranteed to select the same result. `exclusion_radius` is recorded
    as passed (not normalized to 'auto' the way build_network_fingerprint
    does) since run_mde_for_target() itself has no visibility into
    EXCLUSION_RADIUS_AUTO -- see that function's own docstring on why the
    argument is a harmless placeholder in AUTO mode; at worst this costs
    an occasional unnecessary cache miss, never an incorrect hit.'''
    return {
        'node': node,
        'candidate_pool': list(candidate_pool),
        'mde_d': mde_d,
        'tp': tp,
        'embed_dim_rho_min': embed_dim_rho_min,
        'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius,
        'delay_set': list(delay_set),
        'prediction_start': prediction_start,
        'prediction_length': prediction_length,
        'data_file': data_file,
        'data_file_sha256': data_file_sha256,
        'code_fingerprint': code_fingerprint,
        'stage': 'mde_selection',
    }


def _load_json_cache_entry(path):
    with open(path) as f:
        return json.load(f)


def _load_pickle_cache_entry(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_cached(prev_runs_dir, filename, fingerprint, load_one=_load_json_cache_entry):
    '''Generic cross-job cache lookup, shared by every cacheable stage
    (theta search, MDE forward-selection, network construction, the
    generate-start sweep -- see the module docstring's SEARCH REUSE
    section). Scans prev_runs_dir recursively -- works whether
    prev_runs_dir is the cache pool's root (one level of job_<id>/
    subdirectories, see simple-mde-gmn.sbatch's CACHE_ROOT/RUN_CACHE_DIR
    comment) or already a single job's own cache dir -- for files named
    `filename`, newest (by mtime) first, and returns the first
    (entry, path) whose entry['fingerprint'] exactly equals `fingerprint`.
    `load_one(path)` does the actual deserialization: JSON by default for
    small metadata (theta, MDE column lists), or _load_pickle_cache_entry
    for DataFrames/graphs too structured for JSON (see
    save_pickle_to_cache()). Returns None if prev_runs_dir doesn't exist
    yet (e.g. this project's very first run) or nothing matches -- both
    ordinary "no cache hit" outcomes, not errors; every caller falls
    through to computing the real result on None.'''
    if not prev_runs_dir.exists():
        return None
    candidates = sorted(
        prev_runs_dir.glob(f'**/{filename}'),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for path in candidates:
        try:
            entry = load_one(path)
        except (OSError, EOFError, json.JSONDecodeError, pickle.UnpicklingError):
            continue
        if entry.get('fingerprint') == fingerprint:
            return entry, path
    return None


def _save_to_cache(filename, results_dir, run_cache_dir, write_one):
    '''Calls write_one(path) once per unique directory in
    {results_dir, run_cache_dir} (deduplicated, order-preserving, so the
    RUN_CACHE_DIR==RESULTS_DIR fallback for ad hoc runs outside the sbatch
    script -- see load_config()'s docstring -- writes exactly once, not
    twice). results_dir is this run's ordinary per-job record, unchanged
    in behavior from before this cache existed: it still only reaches
    persistent storage at job end, on success, via simple-mde-gmn.sbatch's
    final `cp -av`. run_cache_dir is the durable bucket pool
    (MDE_GMN_RUN_CACHE_DIR) a future job's find_cached() searches --
    writing there *immediately*, right after this stage finishes, is what
    lets its result survive even if this job crashes or times out later in
    the pipeline, instead of being lost with the rest of the /flash
    scratch tempdir (see the sbatch script's /flash scratch setup
    section's own comment on exactly this gap). Returns the results_dir
    path, for callers that log it.'''
    written = []
    for directory in dict.fromkeys([results_dir, run_cache_dir]):
        path = directory / filename
        write_one(path)
        written.append(path)
    return written[0]


def save_json_to_cache(filename, payload, results_dir, run_cache_dir):
    '''JSON variant of _save_to_cache() -- `payload` must already carry
    its own 'fingerprint' key (see find_cached()).'''
    def write_one(path):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)


def save_pickle_to_cache(filename, payload, results_dir, run_cache_dir):
    '''Pickle variant of _save_to_cache(), for artifacts too large/
    structured for JSON (a network graph, an MDE run's DataFrame output).
    `payload` must already carry its own 'fingerprint' key (see
    find_cached()).'''
    def write_one(path):
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)


def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
    '''Theta search's own cache lookup -- thin find_cached() wrapper kept
    as a named function since callers want (best_node_theta, path), not
    the raw cache entry dict.'''
    hit = find_cached(prev_runs_dir, f'best_params_{network_id}.json', fingerprint)
    if hit is None:
        return None
    entry, path = hit
    return entry['best_node_theta'], path


def save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta, source):
    '''Writes this run's own best_params_<network_id>.json -- the artifact
    a future job's find_cached_best_params() looks for -- regardless of
    whether this run computed it fresh or itself reused a cache hit, so
    the provenance chain (source: 'computed' vs. 'reused_from:<path>')
    stays visible in whichever job's output this file ends up in.'''
    return save_json_to_cache(
        f'best_params_{network_id}.json',
        {'fingerprint': fingerprint, 'best_node_theta': best_node_theta, 'source': source},
        results_dir, run_cache_dir,
    )


def retune_theta_for_all_nodes(data_lib, graph, nodes, theta_range, cores, min_pred_rows=20):
    '''Finds each node's own best theta (S-map's nonlinearity/
    localization parameter -- see build_gmn_config()'s module-level
    context for what it controls) via PredictNonlinear on a 70/30
    lib/pred split of that node's own generalized embedding
    (_embedding_spec_for()/_generalized_embed() -- built directly with
    pandas/pyEDM here, not through gmn's Node/Network machinery, since
    this runs once per network before any gmn.GMN object exists yet).
    Theta is the only thing this pipeline still searches per node (see
    run_theta_search()'s docstring for why E and each driver's delay no
    longer need searching at all). Falls back to theta=3 (a reasonable
    generic default) whenever there's too little data for the split
    (fewer than min_pred_rows pred rows) or PredictNonlinear itself
    fails, rather than letting one bad node abort the whole search.'''
    best_theta_by_node = {}

    for node in nodes:
        spec = _embedding_spec_for(graph, node)
        embedded, offset = _generalized_embed(data_lib, spec)
        embedded.insert(0, 'Time', np.arange(len(embedded)))
        embedded[node] = data_lib[node].to_numpy()[offset:]

        n_emb = len(embedded)
        split = int(n_emb * 0.7)
        n_pred = n_emb - split
        if n_pred < min_pred_rows:
            best_theta_by_node[node] = 3
            continue

        lib_range = f'1 {split}'
        pred_range = f'{split + 1} {n_emb}'

        try:
            pn = PredictNonlinear(
                dataFrame=embedded, columns=list(embedded.columns[1:-1]), target=node,
                theta=list(theta_range), lib=lib_range, pred=pred_range,
                Tp=1, embedded=True, showPlot=False, numProcess=cores,
            )
            best_theta_by_node[node] = float(pn.loc[pn['rho'].idxmax(), 'theta'])
        except Exception:
            best_theta_by_node[node] = 3

    return best_theta_by_node


def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
                             node_theta, variant, results_dir, cores,
                             generate_length=None, generate_start=None, exclusion_radius=0):
    '''Runs one Generate() call and scores it as mean rho, across
    `net_result['targets']` only (not every driver node), between the
    simulated and real held-out series -- this is "generative skill" as
    this pipeline defines it everywhere: the theta search
    (run_theta_search) and the start-point sweep (sweep_generate_start)
    both call this same function, at the same objective, so a search
    over either axis is comparable to the other.'''
    generate_start = generate_start if generate_start is not None else prediction_start
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir,
        mode='Generate', node_theta_overrides=node_theta,
        variant=variant, generate_length=generate_length,
        generate_start=generate_start, exclusion_radius=exclusion_radius,
    )
    G = gmn.GMN( configFile = str(config_file), cores = cores )
    G.Generate()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    n_gen = len(generated)

    target_rhos = {}
    for target in net_result['targets']:
        actual_vals = actual_df[target].to_numpy()[generate_start: generate_start + n_gen]
        gen_vals = generated[target].to_numpy()
        target_rhos[target] = ComputeError(actual_vals, gen_vals)['rho']

    score = float(np.nanmean(list(target_rhos.values())))
    return score, target_rhos, G


def run_theta_search(gmn_runs, prediction_start, prediction_length, theta_range, results_dir,
                      cores, generate_length=None, reuse_best_params=False, prev_runs_dir=None,
                      run_cache_dir=None, fingerprints=None, exclusion_radius=0):
    '''Retunes each network's per-node theta once
    (retune_theta_for_all_nodes()), then scores the result with a single
    Generate() call -- replacing this pipeline's old combined (E, tau,
    theta) grid search (see the module docstring's SEARCH REUSE section,
    and scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that
    implementation, kept for reference/comparison per explicit
    direction). E and each driver's delay are no longer search
    parameters at all: for MODE='MDE-GMN' they come directly from
    nested-MDE's own discovered network (MultiDiGraph edges' delay
    attribute plus self_delay_set -- see build_nested_mde_network()'s
    and _embedding_spec_for()'s docstrings); for MODE='GMN' they still
    come from that mode's own CCM E/tau grid search
    (fallback_E/fallback_tau, set in build_network()). Either way there
    is nothing left to grid-search over except theta -- a genuine
    reduction in search cost, not just a relocation of it, since the old
    search's dominant cost (len(E) x len(tau) full Generate() calls) is
    now a single Generate() call per network.

    reuse_best_params/prev_runs_dir/fingerprints: the same cross-job
    reuse mechanism as before (see the module docstring's SEARCH REUSE
    section) -- a network whose fingerprint matches a past job's cached
    result skips even this one retuning-and-scoring step entirely.'''
    theta_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        all_nodes = list(graph.nodes)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        fingerprint = fingerprints[network_id] if fingerprints else None
        if reuse_best_params and fingerprint is not None:
            cached = find_cached_best_params(prev_runs_dir, network_id, fingerprint)
            if cached is not None:
                best_node_theta, source = cached
                print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched {source} -- '
                      f'skipping theta retuning entirely', flush=True)
                theta_search_results[network_id] = {
                    'best_node_theta': best_node_theta, 'reused_from': str(source),
                }
                save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                                  source=f'reused_from:{source}')
                continue
            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
                  f'found under {prev_runs_dir} -- retuning theta.', flush=True)

        print(f'{network_id}: retuning theta for {len(all_nodes)} node(s)', flush=True)
        best_node_theta = retune_theta_for_all_nodes(
            data_after_mde_lib, graph, all_nodes, theta_range, cores,
        )

        try:
            score, _, _ = evaluate_smap_candidate(
                net_result, prediction_start, prediction_length, best_node_theta,
                'theta_search', results_dir, cores, generate_length=generate_length,
                exclusion_radius=exclusion_radius,
            )
            print(f'{network_id}: theta search score={score:.4f}', flush=True)
        except Exception as exc:
            # gmn.Generate raises RuntimeError on SMap divergence (blown-up
            # recursive forecast, >1E15) -- possible even at a retuned
            # theta on a large, high-dimensional network, not a bug to
            # crash the whole run over.
            print(f'{network_id}: theta search candidate FAILED: {exc!r}', flush=True)

        theta_search_results[network_id] = {'best_node_theta': best_node_theta}

        if fingerprint is not None:
            save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                              source='computed')

    return theta_search_results


# =============================================================================
# Generate()/Forecast() start-point sweep
# =============================================================================

def build_start_sweep_fingerprint(network_fingerprint, node_theta, start_offsets,
                                   generate_length, exclusion_radius):
    '''Everything that determines one network's start-point sweep outcome
    (sweep_generate_start()): the upstream network+theta identity
    (network_fingerprint -- see build_search_fingerprint(), None if
    MDE_GMN_REUSE_BEST_PARAMS wasn't on for that stage, in which case this
    sweep's own cache is skipped too, since there'd be nothing to pin the
    "which model" half of this fingerprint to) plus node_theta itself
    (two different theta results for the very same network fingerprint --
    e.g. a nondeterministic PredictNonlinear retune -- must not share a
    sweep cache entry) and the sweep's own knobs.'''
    return {
        'network_fingerprint': network_fingerprint,
        'node_theta': {k: node_theta[k] for k in sorted(node_theta)},
        'start_offsets': list(start_offsets),
        'generate_length': generate_length,
        'exclusion_radius': exclusion_radius,
        'stage': 'start_sweep',
    }


def sweep_generate_start(gmn_runs, theta_search_results, prediction_start, prediction_length,
                          results_dir, cores, generate_length, start_offsets, exclusion_radius=0,
                          reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                          fingerprints=None):
    '''Finds, per network, which row Generate() should start free-running
    from to best match the real held-out data -- at the per-node theta
    run_theta_search() already chose, not re-searched here (E and each
    driver's delay are never searched at all -- see that function's
    docstring). GMN's Generate() is a recursive nonlinear simulation, so
    (like any such system) how well it tracks reality can depend on
    which state it launches from, independent of how good the embedding/
    theta are; this sweep answers "given our best model, which starting
    point does it explain best" as a separate question from "what is our
    best model."

    One evaluate_smap_candidate() call per offset in `start_offsets`
    (each offset a row count added to prediction_start), scored on the
    exact same mean-rho-across-targets objective the theta search uses
    -- see that function's docstring. Checkpoints to disk after every
    offset (same resume pattern as run_theta_search's cache) since each
    candidate is its own full Generate() call and this shouldn't have to
    restart from scratch after an interrupted job -- that checkpoint is
    intra-job only (results_dir), though; reuse_cache/prev_runs_dir/
    run_cache_dir/fingerprints add the same cross-job fingerprinted cache
    every other stage has (see the module docstring's SEARCH REUSE
    section) via build_start_sweep_fingerprint(), so a full resubmission
    after a crash can skip the whole sweep, not just resume within one
    still-running job.

    A single-offset `start_offsets` (the default -- see load_config()'s
    GENERATE_START_MIN/MAX_OFFSET) means the sweep is disabled: that one
    offset is returned directly with no Generate() call at all, since
    rebuild_at_best_params() is about to run Generate() at that exact
    (theta, start) anyway -- running it again here first would just be a
    wasted duplicate call.

    Returns {network_id: best_generate_start_row}.'''
    best_starts = {}
    for network_id, run in gmn_runs.items():
        if len(start_offsets) == 1:
            best_starts[network_id] = prediction_start + start_offsets[0]
            continue

        net_result = run['net_result']
        node_theta = theta_search_results[network_id]['best_node_theta']
        network_fingerprint = fingerprints[network_id] if fingerprints else None

        sweep_fingerprint = None
        cache_filename = f'generate_start_sweep_{network_id}.json'
        if reuse_cache and network_fingerprint is not None:
            sweep_fingerprint = build_start_sweep_fingerprint(
                network_fingerprint, node_theta, start_offsets, generate_length, exclusion_radius,
            )
            hit = find_cached(prev_runs_dir, cache_filename, sweep_fingerprint)
            if hit is not None:
                entry, source = hit
                scores = {int(k): v for k, v in entry['scores'].items()}
                best_offset = entry['best_offset']
                best_starts[network_id] = prediction_start + best_offset
                print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched cached start sweep '
                      f'{source} -- skipping sweep entirely (best offset {best_offset}, '
                      f'score={scores.get(best_offset, float("nan")):.4f})', flush=True)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
                ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
                ax.set_xlabel('Start offset from PREDICTION_START (rows)')
                ax.set_ylabel('Generate() rho (mean over targets)')
                ax.set_title(f'Generate() start-point sweep -- {network_id} (cached)')
                fig.tight_layout()
                fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
                plt.close(fig)
                continue
            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached start '
                  f'sweep found under {prev_runs_dir} -- running sweep.', flush=True)

        checkpoint_file = results_dir / f'generate_start_checkpoint_{network_id}.pkl'
        scores = {}
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if (ckpt.get('start_offsets') == start_offsets
                    and ckpt.get('generate_length') == generate_length):
                scores = ckpt['scores']
                print(f'{network_id}: resumed start-sweep checkpoint, '
                      f'{len(scores)}/{len(start_offsets)} offsets already done', flush=True)

        for offset in start_offsets:
            if offset in scores and not np.isnan(scores[offset]):
                continue

            generate_start = prediction_start + offset
            try:
                score, _, _ = evaluate_smap_candidate(
                    net_result, prediction_start, prediction_length,
                    node_theta, f'startsweep_offset{offset}', results_dir, cores,
                    generate_length=generate_length, generate_start=generate_start,
                    exclusion_radius=exclusion_radius,
                )
            except Exception as exc:
                # Same rationale as run_theta_search's own try/except: a
                # divergent SMap Generate() is an expected failure mode
                # for at least some starting points, not a bug to crash
                # the whole sweep over.
                print(f'{network_id}: start offset {offset} FAILED: {exc!r} -- scoring as NaN', flush=True)
                score = np.nan

            scores[offset] = score
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'start_offsets': start_offsets,
                             'generate_length': generate_length, 'scores': scores}, f)
            print(f'{network_id}: start offset {offset:5d} (row {generate_start}) score={score:.4f}', flush=True)

        valid_offsets = [o for o in start_offsets if not np.isnan(scores.get(o, np.nan))]
        best_offset = max(valid_offsets, key=lambda o: scores[o]) if valid_offsets else start_offsets[0]
        best_starts[network_id] = prediction_start + best_offset
        print(f'{network_id}: best generate start = row {best_starts[network_id]} '
              f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)

        if sweep_fingerprint is not None:
            # JSON object keys must be strings -- offsets round-trip back
            # to int in the cache-hit branch above.
            save_json_to_cache(
                cache_filename,
                {
                    'fingerprint': sweep_fingerprint,
                    'scores': {str(k): v for k, v in scores.items()},
                    'best_offset': best_offset,
                },
                results_dir, run_cache_dir,
            )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
        ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Start offset from PREDICTION_START (rows)')
        ax.set_ylabel('Generate() rho (mean over targets)')
        ax.set_title(f'Generate() start-point sweep -- {network_id}')
        fig.tight_layout()
        fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
        plt.close(fig)

    return best_starts


# =============================================================================
# Final rebuild + Forecast() diagnostic + fidelity metric
# (mirrors notebook cells 11-13)
# =============================================================================

def rebuild_at_best_params(gmn_runs, theta_search_results, prediction_start, prediction_length,
                            results_dir, cores, generate_length=None, generate_starts=None,
                            exclusion_radius=0):
    '''Reruns Generate() at the winning per-node theta found by
    run_theta_search(), and the winning start row found by
    sweep_generate_start() (generate_starts; defaults to prediction_start
    per network when None). Plotting is deliberately NOT done here -- it
    used to draw its own actual-vs-Generate() figure, but that was a
    strict subset of run_forecast_diagnostic()'s own actual-vs-
    Generate()-vs-Forecast() figure (same two of its three lines, same
    nodes), so the two have been merged into the one figure
    run_forecast_diagnostic() draws, and this function now only reruns
    Generate() (needed downstream by run_forecast_diagnostic,
    gmn_fidelity_rho, and the raw output CSV) and prints the per-node
    rho.

    Unlike run_theta_search()'s/sweep_generate_start()'s own probe
    Generate() calls, this one used to be completely unguarded --
    gmn.Generate() can still raise RuntimeError on SMap divergence
    (>1E15) here even at an already-retuned theta and swept start row
    (both only ever *tested* a subset of what gets rebuilt: a theta that
    diverged during search can still be handed here if run_theta_search()
    swallowed that failure and kept it anyway -- see that function's own
    try/except), and this is the one Generate() call everything
    downstream (run_forecast_diagnostic, gmn_fidelity_rho, the raw output
    CSV) depends on. Now try/except-guarded per network: a failure here
    is logged and that network is skipped (run['G'] stays unset) rather
    than crashing the whole job and losing every other, unrelated
    network's already-finished or still-pending work -- callers must
    check for run.get('G') before using a network's result (main() does).'''
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        node_theta = theta_search_results[network_id]['best_node_theta']
        generate_start = generate_starts[network_id] if generate_starts else prediction_start

        try:
            config_file = build_gmn_config(
                net_result, prediction_start, prediction_length, results_dir,
                mode='Generate', node_theta_overrides=node_theta,
                variant='best', generate_length=generate_length, generate_start=generate_start,
                exclusion_radius=exclusion_radius,
            )
            G = gmn.GMN( configFile = str(config_file), cores = cores )
            G.Generate()
        except Exception as exc:
            print(f'{network_id}: FINAL Generate() rebuild FAILED: {exc!r} -- '
                  f'skipping this network (no plots/fidelity/output for it)', flush=True)
            continue

        run['G'] = G
        run['best_node_theta'] = node_theta
        run['generate_start'] = generate_start

        actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
        generated = G.DataOut.reset_index(drop=True)
        nodes = [c for c in generated.columns if c != 'Time']

        print(f"{network_id} Generate() rho, held-out window (start row {generate_start}):", flush=True)
        for node in nodes:
            # Sliced to len(generated), not prediction_length: once
            # generate_length can be shorter than the full held-out window
            # (the whole point of decoupling it), actual_df's slice must
            # match Generate()'s actual output length or ComputeError()
            # compares mismatched-length arrays.
            a = actual_df[node].to_numpy()[generate_start:generate_start + len(generated)]
            r = ComputeError(a, generated[node].to_numpy())['rho']
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)


def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             node_theta, results_dir, cores, save_path,
                             generate_length=None, generate_start=None, nodes_per_figure=10,
                             exclusion_radius=0, plot_range_mode='full'):
    '''Runs the Forecast() diagnostic and draws the one actual-vs-
    Generate()-vs-Forecast() figure for this network (see
    rebuild_at_best_params()'s docstring for why there's only one figure
    here rather than a separate Generate()-only one).

    generate_start (defaults to prediction_start when None) anchors both
    Forecast()'s own pred window and the plotted window to the same row
    sweep_generate_start() picked for G's own Generate() run -- so all
    three lines on the figure (actual, Generate(), Forecast()) describe
    the same starting point, "to be consistent with the generative mode."
    See GENERATE_START_MIN/MAX/STEP_OFFSET in load_config() for how a
    swept starting point ends up here: sweep_generate_start() only ever
    changes this one row number (per network) that both Generate() (G,
    already run by rebuild_at_best_params() by the time this is called)
    and this function's own Forecast() call anchor to -- it does not
    otherwise reshape either trace or this plot.

    generate_length is the length Generate() (G, passed in already built)
    free-ran for -- the length actually used for Forecast()'s own pred
    window and for how far both traces get *plotted* depends on
    plot_range_mode, not directly on this value alone; see that
    parameter.

    plot_range_mode -- two options, per explicit direction, both anchored
    at generate_start:
      'full' (default, previous unconditional behavior): actual/lib is
        plotted from row 0 through generate_start+generate_length;
        Forecast() itself computes exactly generate_length steps, same as
        Generate(). All three traces effectively share one right edge.
      'windowed': plotted window is asymmetric around generate_start --
        2*generate_length rows before it (actual/lib only -- Generate()
        never ran that far back) and 2*generate_length rows after it,
        clamped to the data actually available (can't plot before row 0,
        and can't plot past this network's held-out
        prediction_start+prediction_length rows -- see actual_df's own
        construction in build_network()). Forecast() (this function's own
        call, independent of Generate()'s own generate_length -- Forecast
        is a direct per-row SMap prediction against real held-out
        predictor data, not a recursive simulation like Generate(), so
        asking it for more steps here doesn't compound error the way
        widening Generate() would) is deliberately run out to that same
        2*generate_length-clamped length instead of generate_length, so
        its trace visibly spans twice Generate()'s own length rather than
        stopping at the window's midpoint. Net effect: actual/lib always
        spans wider than Generate()'s own window (it's real data, cheap
        to show further out); Forecast() always spans exactly twice
        Generate()'s window (real computation, capped by available
        data); Generate() itself is unchanged, still only generate_length
        long -- the three traces' differing plotted extents are the
        point, not a bug: it isolates "how far can this network's own
        embedding usefully predict" (Forecast, cheap to extend) from "how
        far did we actually ask the recursive simulation to run"
        (Generate, deliberately unchanged here).

    Split across ceil(len(nodes) / nodes_per_figure) separate files
    (save_path with a _partNN suffix inserted before its extension)
    rather than one figure with a subplot per node -- this network can
    have 100+ nodes (see build_nested_mde_network()'s docstring on
    network size), and a single figure that tall was slow to render and
    load for no benefit over several page-sized ones.

    Returns (None, {}) instead of raising if the Forecast() build/call
    itself fails -- either gmn.Forecast() raising RuntimeError on SMap
    divergence (>1E15, the same failure mode Generate() risks, per that
    function's own docstring -- Forecast() isn't recursive so it can't
    compound error the way Generate() does, but its own local SMap fits
    can still occasionally blow up) or build_gmn_config()'s own explicit
    ValueError (an infeasible offset/forecast_length combination, more
    reachable than before now that plot_range_mode='windowed' can request
    a longer forecast_compute_length than the 'full' default ever did).
    Previously unguarded -- see rebuild_at_best_params()'s docstring for
    the same rationale, applied here: a bad Forecast() must not crash
    every other, unrelated network in the same job. Callers must check
    for a None first return value before using it.'''
    generate_length = generate_length if generate_length is not None else prediction_length
    generate_start = generate_start if generate_start is not None else prediction_start
    total_rows = prediction_start + prediction_length  # == len(actual_df) -- see build_network()

    if plot_range_mode == 'windowed':
        forecast_compute_length = min(2 * generate_length, max(total_rows - generate_start, 0))
    else:
        forecast_compute_length = generate_length

    try:
        max_offset = network_max_embedding_offset(net_result['network']['Graph'])
        config_file = build_gmn_config(
            net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
            node_theta_overrides=node_theta, generate_length=forecast_compute_length,
            generate_start=generate_start, max_offset=max_offset,
            exclusion_radius=exclusion_radius,
        )
        Gf = gmn.GMN( configFile = str(config_file), cores = cores )
        Gf.Forecast()
    except Exception as exc:
        print(f"{net_result['network_id']}: Forecast() diagnostic FAILED: {exc!r} -- "
              f"skipping its plot/rho for this network (Generate()-based outputs are unaffected)",
              flush=True)
        return None, {}

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = forecast_out['Time'].to_numpy().astype(int)
    nodes = [c for c in forecast_out.columns if c != 'Time']

    generate_out = G.DataOut.reset_index(drop=True)
    generate_time = np.arange(generate_start, generate_start + len(generate_out))

    if plot_range_mode == 'windowed':
        plot_start = generate_start - min(2 * generate_length, generate_start)
        plot_end = generate_start + forecast_compute_length
    else:
        plot_start = 0
        plot_end = generate_start + generate_length

    rhos = {}
    n_parts = (len(nodes) + nodes_per_figure - 1) // nodes_per_figure
    for part in range(n_parts):
        part_nodes = nodes[part * nodes_per_figure:(part + 1) * nodes_per_figure]
        fig, axes = plt.subplots(len(part_nodes), 1, figsize=(11, 2.2 * len(part_nodes)), sharex=False)
        if len(part_nodes) == 1:
            axes = [axes]

        for ax, node in zip(axes, part_nodes):
            actual = actual_df[node].to_numpy()
            forecast_pred = forecast_out[node].to_numpy()
            rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']

            # alpha=0.6 (not fully opaque) so the overlaid Generate()/
            # Forecast() lines stay easy to pick out against the real
            # trace -- per explicit direction.
            ax.plot(np.arange(plot_start, plot_end), actual[plot_start:plot_end],
                     color='tab:blue', lw=1, alpha=0.6, label='Lib/actual')
            if node in generate_out.columns:
                ax.plot(generate_time, generate_out[node].to_numpy(),
                        color='tab:red', lw=1, label='Generate()')
            ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
            ax.axvline(generate_start, color='gray', ls='--', lw=0.8)
            ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"Generate() vs Forecast() vs actual -- {net_result['network_id']} "
                     f"(nodes {part * nodes_per_figure + 1}-"
                     f"{part * nodes_per_figure + len(part_nodes)} of {len(nodes)})", fontsize=11)
        fig.tight_layout()
        part_path = save_path.with_name(f'{save_path.stem}_part{part + 1:02d}{save_path.suffix}')
        fig.savefig(part_path, dpi=300)
        plt.close(fig)

    return Gf, rhos


def gmn_fidelity_rho(net_result, G, generate_start, default_E=3, tau=-1):
    '''Fig-4e-style generative fidelity: per node, a fresh Simplex predicts
    the real actual series *from* the generated series (not the other way
    around, and not a direct rho(actual, generated) comparison) on a 50/50
    split of the generated window -- this tests whether the generated
    trajectory's own dynamics stay on the same manifold as reality, a
    stricter/different question than pointwise closeness. E defaults to
    node_dimensions (an EmbedDimension-based estimate from build_network(),
    when computed -- see that function's docstring on when it isn't).'''
    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[generate_start: generate_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            S = Simplex(
                dataFrame = pair_df, columns = f'{node}_gen', target = f'{node}_actual',
                E = E, tau = tau, Tp = 0,
                lib  = f'1 {split}', pred = f'{split + 1} {n_gen}',
                embedded = False, showPlot = False,
            )
            fidelity_rho[node] = ComputeError(S['Observations'], S['Predictions'])['rho']
        except Exception:
            fidelity_rho[node] = np.nan
    return fidelity_rho


def plot_fidelity(rho, network_id, save_path):
    order = sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(order)), [rho[n] for n in order], color='tab:green')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel('rho (Simplex prediction of actual from generated embedding)')
    ax.set_title(f'GMN generative fidelity (Fig. 4e-style) -- {network_id}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# MDE-only diagnostics (mirrors notebook cells 15, 17, 19 -- skipped when
# MODE == 'GMN', same guard the notebook uses)
# =============================================================================

def run_mde_evaluate(mde_results, targets, results_dir):
    for target in targets:
        r = mde_results[target]
        Eval = dx.Evaluate(
            dataFrame     = r['mde_df'],
            columns_range = [1, r['mde_df'].shape[1]],
            mde_columns   = r['SELECTED_COLUMNS'],
            predictVar    = target,
            library       = r['m'].args.lib,
            prediction    = r['m'].args.pred,
            Tp            = r['m'].args.Tp,
            tau           = r['m'].args.tau,
            components    = 5,
            dmap_k        = 15,
            dmap_epsilon  = 'bgh_generous',
            figsize       = (7, 6),
        )
        Eval.Run()
        Eval.Plot()
        plt.gcf().savefig(results_dir / f'mde_evaluate_{target}.png', dpi=300)
        plt.close('all')


def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        pi = PredictInterval(
            dataFrame = r['mde_df'],
            columns   = cols,
            target    = target,
            lib       = r['m'].args.lib,
            pred      = r['m'].args.pred,
            E         = len(cols),
            tau       = r['m'].args.tau,
            embedded  = True,
            maxTp     = max_tp,
            showPlot  = False,
        )
        ax.plot(pi['Tp'], pi['rho'], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel('Prediction skill (rho)')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir):
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                S = Simplex(
                    dataFrame = r['mde_df'], columns = target, target = target,
                    E = E, tau = tau, lib = r['m'].args.lib, pred = r['m'].args.pred,
                    embedded = False, showPlot = False,
                )
                err = ComputeError(S['Observations'], S['Predictions'])
                rho_grid[i, j] = err['rho']

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label='rho')

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================

def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")


def main():
    cfg = load_config()

    df = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Data shape:', df.shape, flush=True)

    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))
    cfg['PREDICTION_LENGTH'] = len(df) - cfg['PREDICTION_START']

    # Resolve GENERATE_LENGTH now that PREDICTION_LENGTH is known: 0
    # (unset) means "use the full held-out window", and anything
    # requesting more steps than there is real held-out data to score
    # against gets clamped down to it (see load_config()'s docstring).
    if cfg['GENERATE_LENGTH'] <= 0:
        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']
    elif cfg['GENERATE_LENGTH'] > cfg['PREDICTION_LENGTH']:
        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the held-out window "
              f"({cfg['PREDICTION_LENGTH']} rows); clamping to it.", flush=True)
        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']

    # Resolve the generate-start sweep's candidate offsets now that
    # PREDICTION_LENGTH/GENERATE_LENGTH are known: drop any offset that
    # would leave too little real data to score GENERATE_LENGTH steps
    # against (see load_config()'s docstring on GENERATE_START_MAX_OFFSET).
    max_feasible_offset = cfg['PREDICTION_LENGTH'] - cfg['GENERATE_LENGTH']
    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
                                cfg['GENERATE_START_STEP']))
    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
    if feasible_offsets != start_offsets:
        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]

    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])

    # Computed once, up front, and threaded through every cacheable stage
    # below (MDE forward-selection, network construction, theta search,
    # the generate-start sweep -- see the module docstring's SEARCH REUSE
    # section) rather than per-stage, so they all key off the exact same
    # data/code/environment identity. Only actually paid for when caching
    # is enabled -- skip the cost otherwise. code_fingerprint in
    # particular is what lets a cache safely survive a purely cosmetic
    # edit to this file (a docstring, say) while still invalidating on any
    # real change to the actual computation -- see build_code_fingerprint()
    # and _pipeline_code_fingerprint()'s docstrings.
    data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE']) if cfg['REUSE_BEST_PARAMS'] else None
    code_fingerprint = build_code_fingerprint() if cfg['REUSE_BEST_PARAMS'] else None

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    expanded_df = expanded_obs_cols = None  # MDE-GMN only -- see below
    if cfg['MODE'] == 'MDE-GMN':
        # Built once, shared by every depth-0 target below and by
        # build_nested_mde_network()'s own nested calls (via build_network()
        # -- see expand_delay_columns()'s and build_nested_mde_network()'s
        # docstrings) -- both need to search the exact same delay-expanded
        # candidate pool for the depth-0 reuse (build_nested_mde_network's
        # own docstring) to be valid.
        expanded_df, expanded_obs_cols = expand_delay_columns(df, obs_cols, cfg['DELAY_SET'])

        for target in cfg['TARGETS']:
            candidate_pool = [c for c in expanded_obs_cols
                              if observable_and_delay(c, obs_cols)[0] != target]
            effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, len(cfg['DELAY_SET']))
            mde_results[target] = run_mde_for_target(
                target, expanded_df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                cfg['EXCLUSION_RADIUS_VALUE'], cfg['CORES'], cfg['RESULTS_DIR'], cfg['DELAY_SET'],
                tp=cfg['TP'],
                reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                run_cache_dir=cfg['RUN_CACHE_DIR'],
                data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                code_fingerprint=code_fingerprint,
            )

        if cfg['RUN_MDE_DIAGNOSTICS']:
            plot_mde_rho_vs_dimension(
                mde_results, cfg['TARGETS'], cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
            )
            plot_3d_mde_projection(
                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df, obs_cols,
                cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['RESULTS_DIR'] / f"mde_3d_projection_{cfg['TARGETS'][0]}.png",
            )
    else:  # MODE == 'GMN'
        for target in cfg['TARGETS']:
            mde_results[target] = {'target': target, 'SELECTED_COLUMNS': list(obs_cols)}
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} observable columns "
              f"as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['INTERACTION_MATRIX_OUTER_CORES'],
            observable_columns=obs_cols, nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
            exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
            expanded_df=expanded_df, expanded_observable_columns=expanded_obs_cols,
            delay_set=cfg['DELAY_SET'], tp=cfg['TP'],
            reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
            run_cache_dir=cfg['RUN_CACHE_DIR'],
            data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
            code_fingerprint=code_fingerprint,
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    fingerprints = None
    if cfg['REUSE_BEST_PARAMS']:
        fingerprints = {
            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256, code_fingerprint)
            for network_id, run in gmn_runs.items()
        }

    theta_search_results = run_theta_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['THETA_RANGE'],
        cfg['RESULTS_DIR'], cfg['CORES'],
        generate_length=cfg['GENERATE_LENGTH'],
        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'],
        fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
    )

    generate_starts = sweep_generate_start(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'], fingerprints=fingerprints,
    )

    rebuild_at_best_params(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'],
        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
    )

    for network_id, run in gmn_runs.items():
        if 'G' not in run:
            # rebuild_at_best_params() already logged why and skipped
            # this network -- nothing downstream (Forecast, fidelity, the
            # output CSV) has anything to work with without G.
            print(f'{network_id}: no successful Generate() rebuild -- skipping '
                  f'Forecast()/fidelity/output for this network', flush=True)
            continue

        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
            plot_range_mode=cfg['FORECAST_PLOT_RANGE'],
        )
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        if Gf is None:
            # run_forecast_diagnostic() already logged why -- Generate()
            # (run['G']) still succeeded, so fidelity/the output CSV
            # below are unaffected and still worth producing.
            print(f'{network_id}: no Forecast() rho to report for this network', flush=True)
        else:
            print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
            for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
                print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics ------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        run_mde_evaluate(mde_results, cfg['TARGETS'], cfg['RESULTS_DIR'])
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'], cfg['RESULTS_DIR'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'])
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
