"""Per-node embeddings, and the gmn patches that make GMN honour them.

GMN (the simulator this pipeline hands its network to) assumes every node uses
the same embedding: E coordinates spaced tau apart. This pipeline does not. A
node's coordinates come from its own in-edges, each with its own delay, so two
nodes can have different numbers of coordinates and different spacings. This
module builds those per-node embeddings and makes GMN use them.

    _embedding_spec_for()   a node's (column, delay) coordinates, read straight
                            off its in-edges. Lagged leaves contribute their
                            underlying observable plus their lag.
    _generalized_embed()    materialises those coordinates as shifted columns.
    network_max_embedding_offset()
                            the largest delay anywhere in the network, i.e. how
                            many leading rows no node can predict yet.

!!! IMPORTING THIS MODULE PATCHES gmn !!!
    At the bottom, three gmn.Node methods are replaced: __init__ (to fix the
    library-end index, which the installed package derives from E * tau),
    Forecast and Generate (to embed per node instead of uniformly). The
    patches are applied at import time, so:
      * anything that constructs or runs gmn objects must import this module
        first -- importing it late means some nodes were built unpatched;
      * importing it is not side-effect-free, which is why it is a module of
        its own rather than part of gmn_build.py.
    _real_node_init keeps a reference to the original gmn.Node.__init__, which
    the replacement still calls.

The functions themselves are pure graph/pandas mechanics: they care only about
edge delays and column names, never about what the signals mean, so nothing
here is specific to TSLS-converted data.

Depends on: columns; gmn, networkx, numpy, pandas.
Imported by: gmn_build, theta, generate, pipeline, the entry scripts.
"""

from gmn.Common import FunctionType
import gmn
import networkx as nx
import numpy as np
import pandas as pd


def _embedding_spec_for(graph, node_name):
    '''Builds one node's (column, delay) embedding coordinates directly
    from the network graph -- a pure function, reusable both by
    _node_embedding_spec() (which caches this on a live gmn.Node object,
    for gmn's own monkeypatched Forecast()/Generate()) and by
    retune_theta_for_all_nodes() (which builds embeddings directly with
    pandas/pyEDM outside gmn's Node/Network machinery entirely, before any
    gmn.GMN object exists).

    MultiDiGraph (MODE='MDE-GMN', built by build_nested_mde_network()):
    R14 (docs/review-actions-v1.md), corrected after this pipeline's own
    R14 test found the original "built only from discovered drivers"
    framing doesn't survive contact with pyEDM: self no longer
    unconditionally gets the FULL self_delay_set (the deep lags -- t-1,
    t-3, t-5, ... -- that used to ride along regardless of what nested-MDE
    actually found) -- those are removed. But self ALWAYS keeps its own
    delay=0 "anchor" coordinate unconditionally, on every node, for two
    reasons that are both load-bearing, not cosmetic: (1) gmn's own SMap
    call sets `target = node_name + '(t-0)'`, which pyEDM requires to
    exist as an actual column in the embedded dataframe regardless of
    what the predictor `columns` are -- with no self-anchor and no
    in-edge ever able to supply it (self-loops are excluded, see
    _accept_mde_result_edges()'s docstring), every interior node would
    have no target column at all and pyEDM would raise outright, not just
    on a small-driver-count edge case; (2) this is also standard
    generalized-embedding practice, not merely a workaround -- a node's
    own present value is conventionally one of its own state-space
    coordinates alongside whichever other coordinates come from selected
    drivers (the classic Takens embedding is literally E lags of a
    variable's own history; substituting some of those coordinates for
    other variables' lags, per Deyle & Sugihara's generalized embedding,
    doesn't mean dropping the variable's own present-value coordinate
    too). So: self(t-0) unconditional (double duty as both the SMap
    target and a real predictor coordinate -- not excluded from
    `columns`), plus one coordinate per discovered driver in-edge at that
    edge's own delay; deeper self-lags are admissible only when
    nested-MDE genuinely selects them as an actual driver (i.e., appear
    only via an in-edge, never via an unconditional default). See
    partition_network()'s docstring for the interior (>=1 driver) vs.
    external-input (0 drivers) split this enables, and
    EXTERNAL_INPUT_MODE's own load_config() docstring for what happens to
    a zero-driver node: under 'self_simulate' it reaches this function
    normally and, having no in-edges by definition, ends up with the
    anchor-only spec [(node_name, 0)] -- no special-casing needed here,
    the same formula below already produces that; under the 'real_data'
    default, such a node never reaches this function at all --
    _generalized_node_generate()/_generalized_node_forecast() short-circuit
    it before any embedding is built, and _generalized_node_init() skips
    the libEnd_i correction below that would otherwise need a spec to
    exist at all.

    On the historical "leaf duplicate" degeneracy this decision's
    rationale originally cited (a real observable B and "a lagged copy of
    B" ending up as two separate zero-driver nodes with identical
    embeddings): confirmed against _accept_mde_result_edges() and
    observable_and_delay() that this does not describe this graph.
    observable_and_delay() always resolves a candidate column back to its
    RAW observable name (delay lives separately), and
    _accept_mde_result_edges() calls graph.add_node(driver) using only
    that raw name -- so there is exactly one graph node per raw
    observable regardless of how many different delays reference it; "a
    lagged copy of B" is never a second node, only a (B, some-negative-
    delay) edge into whatever node selected it. The surviving
    justification for partitioning out zero-driver nodes is simpler and
    still correct on its own: a zero-driver node isn't something
    nested-MDE discovered as a network member at all, so counting it as
    an interior (simulated) node overstates what was actually found.

    Plain DiGraph (MODE='GMN', from apps.CreateNetwork -- no per-edge delay
    data exists in that mode at all): falls back to the original uniform
    E-lagged-copies-of-every-column scheme, using
    graph.graph['fallback_E']/['fallback_tau'] (set in build_network() from
    that mode's own CCM E/tau grid search) in place of a per-node nested-MDE
    result it doesn't have. R14 doesn't touch this branch -- MODE='GMN'
    never runs build_nested_mde_network() at all, so it has no nested-MDE
    "discovered driver" concept to partition on in the first place.

    Either way, this is what replaced gmn's own Embed(E, tau, columns) call
    -- see _generalized_embed()'s docstring -- since a single uniform
    (E, tau) can't express a per-driver, per-delay embedding.'''
    if isinstance(graph, nx.MultiDiGraph):
        spec = [(node_name, 0)]
        # An in-edge from a lagged LEAF node carries that leaf's own name
        # (e.g. 'CA1_3_tsls::delay-9'), but an embedding coordinate has to
        # name a real data column, so resolve it back to the raw observable
        # (the delay is already on the edge). Real-observable in-edges pass
        # through unchanged. See _accept_mde_result_edges().
        for u, _v, data in graph.in_edges(node_name, data=True):
            spec.append((graph.nodes[u].get('lagged_of', u), data['delay']))
    else:
        E = graph.graph.get('fallback_E', 3)
        tau = graph.graph.get('fallback_tau', -1)
        cols = list(graph.predecessors(node_name)) + [node_name]
        spec = [(c, k * tau) for c in cols for k in range(E)]
    return spec

def network_max_embedding_offset(graph):
    '''Largest per-node embedding offset across the whole network -- since
    nodes can each have their own distinct embedding now (unlike the old
    shared (E, tau)), this is used by build_gmn_config()'s Forecast mode to
    size a single, network-wide lib/pred window that stays valid for every
    node (see that function's docstring for the row-alignment tradeoff this
    implies for lower-offset nodes). Still safe post-R14: self_max stays an
    upper bound on any node's real offset even though interior nodes no
    longer unconditionally carry self_delay_set themselves (their own
    in-edge delays are, structurally, never deeper than the delays
    self_delay_set/LAG_SET already span) -- this can only over-count, never
    under-count, so the window stays valid.'''
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
    embedding_spec, each holding that column's value abs(delay) rows in the
    past (every delay is <= 0). Named to match pyEDM.Embed()'s own
    "col(t-k)" convention (k = abs(delay)) so downstream code that looks up
    Parameters.target + '(t-0)' for the current-value column keeps working
    unchanged. Returns (embedded_df, offset) with a fresh 0-based index --
    offset is the largest lag magnitude across the whole spec, i.e. how many
    leading rows are dropped as NaN once every (col, delay) pair has been
    shifted into place.'''
    columns = {
        f'{col}(t-{abs(delay)})': data[col].shift(-delay).to_numpy()
        for col, delay in embedding_spec
    }
    embedded = pd.DataFrame(columns, index=data.index)
    offset = max(abs(delay) for _, delay in embedding_spec)
    return embedded.iloc[offset:].reset_index(drop=True), offset

_real_node_init = gmn.Node.__init__

def _generalized_node_init(self, args, Network, nodeName):
    '''Wraps gmn.Node.__init__ to correct self.libEnd_i (the EDM library end
    index) for SMap nodes after construction -- the installed package
    computes it as self.data.shape[0] - Parameters.E * abs(Parameters.tau),
    which assumes one shared, uniform embedding; the generalized embedding
    this pipeline builds instead (see _node_embedding_spec()'s docstring)
    has its own, generally different, per-node offset. Recomputed exactly
    once here, right after __init__ sets the node's pristine self.data,
    since libEnd_i is meant to reflect that original library size before
    any values get appended during a Generate() run.

    R14: skipped entirely for a real-data-mode external input (see
    partition_network()'s docstring) -- such a node has no generalized
    embedding at all (_embedding_spec_for() is never reached for it; its
    Generate()/Forecast() short-circuit to a real-data passthrough), so
    there is no spec to size libEnd_i from. `_real_node_init`'s own
    (otherwise-uncorrected) formula is left in place; it's inert dead data
    for a node that never runs an SMap/Embed call.'''
    _real_node_init(self, args, Network, nodeName)
    if self.FunctionType == FunctionType.SMap:
        graph = Network.Graph
        is_external = nodeName in graph.graph.get('external_inputs', set())
        if is_external and graph.graph.get('external_input_mode', 'real_data') == 'real_data':
            return
        spec = _node_embedding_spec(self)
        offset = max(abs(delay) for _, delay in spec)
        self.libEnd_i = self.data.shape[0] - offset

def _generalized_node_forecast(self):
    '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
    pre-existing bugs (below) and replacing its SMap branch's
    Embed(E, tau, columns) call with _generalized_embed() -- the
    per-driver-delay embedding built from the network graph instead of one
    uniform (E, tau) applied to every predictor. This is the only branch
    this pipeline exercises, since every node uses SMap.

    (1) gmn/Forecast.py calls Embed() without importing pyEDM.Embed --
    raises NameError on the first Forecast() call against any SMap node.
    Moot now that this branch no longer calls pyEDM's Embed() at all.

    (2) `val` is a 2-tuple (time, predictions), then checked for divergence
    via `val.max() > 1E15` -- tuples have no .max(), so this raised
    AttributeError unconditionally, not just on genuine divergence. Fixed to
    check the predictions Series directly (val[1]).

    The time column is inserted explicitly from data's own offset-sliced,
    reset-index time values rather than gmn's own implicit index-aligned
    insert, since _generalized_embed() already resets its own index to
    0-based. Every other branch is copied verbatim from gmn/Forecast.py
    (unused here, since every node is SMap, but kept intact so this patch
    can't silently change behavior for a future non-SMap node).

    R14: a real-data-mode external input (see partition_network()'s
    docstring) short-circuits here too, before any SMap/embedding logic.
    Forecast()'s own one-step-ahead skill metric is meaningless for a
    variable this network doesn't predict at all (self.data is already
    fully real throughout Forecast mode -- no iterative lastDataOut
    feeding, unlike Generate() -- so there is nothing to "forecast" here
    beyond the real series itself). Returns (time, real values) as a
    trivial passthrough, matching Forecast()'s own (time, predictions)
    return shape, so gmn's per-node Forecast loop (which iterates every
    graph node structurally, with no partition awareness of its own)
    doesn't crash on an empty embedding spec.

    This passthrough is now sliced to Parameters.pred's own row range
    (found via direct investigation of a real run's diagnostic figure --
    docs/review-findings.md's R22 for the full trace) rather than
    returning self.data in full. Network.py sets
    dataLib_i = range(self.data.shape[0]) in Forecast mode ("all data
    since lib & pred are specified"), so self.data always holds the ENTIRE
    dataset here, not a pred-restricted slice -- returning it whole used to
    make this node's own Forecast() output thousands of rows longer than
    every SMap node's own short pred-range Predictions Series. Since
    gmn.GMN.Forecast() (gmn/GMN.py) assembles DataOut via plain per-node
    column assignment and takes the shared 'Time' column only from
    Network.TopologicalSorted[0] -- almost always a source/leaf node like
    this one, since it has no in-edges -- that full-length passthrough
    silently became DataOut's own row index and its 'Time' column's real
    values (0..N-1). Every SMap node's own short Predictions Series (a
    fresh 0-based pandas index, confirmed empirically against a live
    pyEDM.SMap call: its 'Predictions' column's index is always 0-based
    regardless of what its own 'Time' column reports) then pandas-aligned
    onto only the first `pred_end - pred_start + 1` of those row labels --
    mislabeling genuinely-near-generate_start predictions as though they
    occurred at the very start of the recording, and forcing every
    downstream .dropna() (run_forecast_diagnostic()'s own forecast_out)
    down to just that falsely-early sliver. This corrupted both the
    Forecast() trace's plotted x-position (the "small trace at the
    beginning" regardless of FORECAST_PLOT_RANGE) and the Forecast rho/f1
    scores printed in its title (scored against actual[0:few-hundred] --
    real data from the start of the recording -- instead of the actual
    generate_start window the prediction was computed from). Slicing to
    the same pred range every SMap node uses gives this node's own output
    the same length and 0-based index as theirs, so gmn's naive assembly
    aligns everything correctly instead of accidentally.

    (3) `columns=df.columns[1:-1]` used to drop BOTH `time` (first) AND
    the spec's own last entry -- since spec is self-lags-then-in-edges,
    that last entry is always the node's last discovered driver whenever
    it has any in-edges (confirmed empirically: spec=[T(t-0), T(t-1),
    A(t-3), B(t-0)] -> the old slice silently excluded B(t-0) from
    `columns`, though it stayed in `df` itself). Pre-R14 this was
    harmless -- self_delay_set padded every node with several self-lags,
    so losing one entry barely mattered. Post-R14 a node's spec IS its
    discovered drivers: a 1-driver interior node had zero columns left at
    all (`columns[1:-1]` on a 2-column df is empty -- pyEDM's "columns
    required" RuntimeError, found via this pipeline's own R14 test), and
    a 2+-driver node silently lost its last one. Fixed here to
    `df.columns[1:]` (drop only `time`) -- found and fixed together with
    R14 since R14 is what turns this from a latent, mostly-harmless
    slicing quirk into a real crash/data-loss risk, but the fix itself
    is general (keeps every discovered driver in the embedding
    regardless of EXTERNAL_INPUT_MODE), not R14-specific.'''
    if self.args.DEBUG :
        print( '-----> Node:Forecast() : ',self.FunctionType.name, self.name)
        print( self.data.tail( 2 ), flush = True );

    graph = self.Network.Graph
    is_external = self.name in graph.graph.get('external_inputs', set())
    if is_external and graph.graph.get('external_input_mode', 'real_data') == 'real_data':
        time_col = self.data.columns[0]
        pred = self.Parameters.pred
        if len(pred) and not pred.isspace():
            pred_start, pred_end = (int(x) for x in pred.split())
            sliced = self.data.iloc[pred_start - 1:pred_end].reset_index(drop=True)
            return (sliced[time_col], sliced[self.name])
        return (self.data[time_col], self.data[self.name])

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
                           columns         = df.columns[1:],
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

def _generalized_node_generate(self, lastDataOut):
    '''Corrected copy of gmn.Node.Generate (gmn/Generate.py), replacing its
    SMap branch's Embed(E, tau, columns) call with _generalized_embed() --
    see _generalized_node_forecast()'s and _node_embedding_spec()'s
    docstrings for why. Unlike Forecast()'s SMap branch, gmn's own
    Generate() code never tried to reuse real time values for its inserted
    time column (just a fresh synthetic 0-based arange()), so there's no
    index-alignment subtlety to correct here. Every other branch is copied
    verbatim (unused here, since every node is SMap).

    R14: a real-data-mode external input (see partition_network()'s
    docstring) short-circuits here, before any SMap/embedding logic --
    genuinely not simulated, its "generated" value each step is just the
    real recorded value at that row, read straight from
    self.Network.data (loaded once, in full, from DATA_AFTER_MDE -- see
    build_gmn_config()'s own docstring). self.data has already had
    lastDataOut appended above by the time this check runs, so its length
    IS the absolute row index of the next real row to hand back (both
    start in lockstep at generate_start real rows, then grow by exactly
    one row per step, for every node, via the same shared lastDataOut).

    Also fixes the same `columns=df.columns[1:-1]` -> `df.columns[1:]`
    driver-drop bug as _generalized_node_forecast()'s own docstring
    describes (found via this pipeline's own R14 test) -- the old slice
    silently excluded a node's LAST discovered driver from its SMap
    embedding, harmless pre-R14 (self_delay_set padding masked it) but a
    hard crash post-R14 for a 1-driver interior node.'''
    if self.args.DEBUG :
        print( '-----> Node:Generate() : ', self.FunctionType.name, self.name )
        print( self.data.tail( 2 ) );
        print( 'lastDataOut:' ); print( lastDataOut, flush = True )

    if not ( lastDataOut is None ):
        self.data = pd.concat( [ self.data, lastDataOut ] )

    graph = self.Network.Graph
    is_external = self.name in graph.graph.get('external_inputs', set())
    if is_external and graph.graph.get('external_input_mode', 'real_data') == 'real_data':
        row_i = len(self.data)
        real_data = self.Network.data
        if row_i >= len(real_data):
            raise RuntimeError(
                f"Generate: external input '{self.name}' ran out of real data at row {row_i} "
                f"(only {len(real_data)} rows available) -- shrink GENERATE_LENGTH or switch "
                f"EXTERNAL_INPUT_MODE to 'self_simulate'."
            )
        return float(real_data[self.name].iloc[row_i])

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
                           columns         = df.columns[1:],
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


# =============================================================================
# The patches themselves. Applied on import -- see this module's docstring.
# _real_node_init above already holds the original __init__, captured before
# this rebinds it, and the replacement calls it.
# =============================================================================
gmn.Node.__init__ = _generalized_node_init
gmn.Node.Forecast = _generalized_node_forecast
gmn.Node.Generate = _generalized_node_generate
