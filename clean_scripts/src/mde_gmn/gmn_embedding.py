"""Per-node embeddings, and the gmn patches that make GMN honour them.

GMN (the simulator this pipeline hands its network to) assumes every node uses
the same embedding: E coordinates spaced tau apart. This pipeline does not. A
node's coordinates come from its own in-edges, each with its own delay, so two
nodes can have different numbers of coordinates and different spacings. This
module builds those embeddings and makes GMN use them.

    embedding_spec_for()    a node's (column, delay) coordinates, read straight
                            off its in-edges. Lagged leaves contribute their
                            underlying observable plus their lag.
    generalized_embed()     materialises those coordinates as shifted columns.
    network_max_embedding_offset()
                            the largest delay anywhere in the network, i.e. how
                            many leading rows no node can predict yet.

!!! IMPORTING THIS MODULE PATCHES gmn !!!
    At the bottom, three gmn.Node methods are replaced: __init__ (to fix the
    library-end index, which the installed package derives from E * tau),
    Forecast and Generate (to embed per node instead of uniformly). The
    patches are applied at import time, so:
      * anything that constructs or runs gmn objects must import this module
        FIRST -- importing it late means some nodes were built unpatched;
      * importing it is not side-effect-free, which is why it is a module of
        its own rather than part of gmn_build.py.

HOW THE PATCHED METHODS ARE ORGANISED. gmn's own Forecast() and Generate() are
each one long if/elif chain over the node's function type. Those chains are
kept here branch for branch, but as one small function per type plus a dispatch
table, because only ONE of them is ever reached: build_gmn_config() fixes every
node in this pipeline to SMap, and SMap is the only branch this module changes
(gmn's Simplex branch never calls Embed() at all, so it cannot do real Takens
embedding). The other branches are carried unmodified so this patch cannot
silently change behaviour for a node type someone configures later.

The functions themselves are pure graph/pandas mechanics: they care only about
edge delays and column names, never about what the signals mean, so nothing
here is specific to TSLS-converted data.

Depends on: columns; gmn, networkx, numpy, pandas.
Imported by: gmn_build, theta, generate, pipeline.
"""

import gmn
import networkx as nx
import numpy as np
import pandas as pd
from gmn.Common import FunctionType

from mde_gmn.columns import _delayed_column_name


# =====================================================================
# What a node's embedding is
# =====================================================================

def embedding_spec_for(graph, node_name):
    '''One node's (column, delay) embedding coordinates, from the graph.

    A pure function, used both by the cached wrapper below (for gmn's own
    patched Forecast/Generate) and by the theta search, which builds
    embeddings with pandas/pyEDM directly, outside gmn's machinery, before
    any gmn.GMN object exists.

    MODE='MDE-GMN' (a MultiDiGraph from the network builders): a node's
    coordinates are its DISCOVERED drivers -- one coordinate per in-edge, at
    that edge's own delay -- plus, unconditionally, the node's own delay-0
    value. That self-anchor is load-bearing twice over:

      1. gmn's SMap call sets `target = node_name + '(t-0)'`, and pyEDM
         requires that column to exist in the embedded frame whatever the
         predictor columns are. Self-loops are excluded from selection, so no
         in-edge can ever supply it -- without the anchor every interior node
         would have no target column and pyEDM would raise outright.
      2. It is also just standard generalized-embedding practice. A classic
         Takens embedding is E lags of a variable's own history; substituting
         some of those coordinates for other variables' lags (Deyle &
         Sugihara) does not mean dropping the variable's own present value.

    So: self(t-0) always, doing double duty as the SMap target and a real
    predictor coordinate, plus one coordinate per in-edge. Deeper SELF lags
    appear only when nested-MDE genuinely selected them, never by default.

    A zero-driver node under EXTERNAL_INPUT_MODE='self_simulate' reaches this
    and, having no in-edges, gets the anchor-only spec [(node_name, 0)] --
    no special case needed, the same formula produces it. Under the
    'real_data' default such a node never reaches here at all: its
    Forecast/Generate short-circuit to a real-data passthrough.

    An in-edge from a lagged LEAF carries that leaf's own node name (e.g.
    'CA1_3_tsls::delay-9'), but an embedding coordinate has to name a data
    column, so it is resolved back to (lagged_of, edge delay). Real-observable
    in-edges pass through unchanged.

    MODE='GMN' (a plain DiGraph from apps.CreateNetwork, which carries no
    per-edge delays at all): falls back to the uniform scheme -- E lagged
    copies of every predecessor and of the node itself -- using
    graph.graph['fallback_E']/['fallback_tau'], set by build_network() from
    that mode's own E/tau grid search. That mode never runs the nested-MDE
    builder, so it has no "discovered driver" concept to work from.
    '''
    if isinstance(graph, nx.MultiDiGraph):
        spec = [(node_name, 0)]
        for driver, _node, data in graph.in_edges(node_name, data=True):
            spec.append((graph.nodes[driver].get('lagged_of', driver), data['delay']))
        return spec

    E = graph.graph.get('fallback_E', 3)
    tau = graph.graph.get('fallback_tau', -1)
    columns = list(graph.predecessors(node_name)) + [node_name]
    return [(c, k * tau) for c in columns for k in range(E)]


def network_max_embedding_offset(graph):
    '''Largest per-node embedding offset across the whole network.

    Nodes each have their own embedding now, so build_gmn_config()'s Forecast
    mode uses this to size a single network-wide lib/pred window that stays
    valid for every one of them. An in-edge delay is structurally never
    deeper than the span LAG_SET covers, so taking the self_delay_set's own
    maximum as an upper bound can only ever OVER-count -- which keeps the
    window valid rather than invalidating it.
    '''
    if isinstance(graph, nx.MultiDiGraph):
        self_max = max(abs(d) for d in graph.graph.get('self_delay_set', [0]))
        edge_max = max((abs(data['delay']) for _u, _v, data in graph.edges(data=True)),
                       default=0)
        return max(self_max, edge_max)
    E = graph.graph.get('fallback_E', 3)
    return (E - 1) * abs(graph.graph.get('fallback_tau', -1))


def _node_embedding_spec(node):
    '''embedding_spec_for() for a live gmn.Node, cached on the node.'''
    if not hasattr(node, '_embedding_spec'):
        node._embedding_spec = embedding_spec_for(node.Network.Graph, node.name)
    return node._embedding_spec


def generalized_embed(data, embedding_spec):
    '''Materialise an embedding spec as a frame of shifted columns.

    One column per (col, delay) pair, each holding that column's value
    abs(delay) rows in the past (every delay is <= 0). Named to match
    pyEDM.Embed()'s own "col(t-k)" convention, because gmn's SMap finds its
    prediction target by looking up Parameters.target + '(t-0)' BY NAME.

    Returns (embedded_df, offset) on a fresh 0-based index. `offset` is the
    largest lag magnitude in the spec, i.e. how many leading rows are dropped
    once every coordinate has been shifted into place.

    TWO WAYS A FRAME CAN CARRY A LAGGED VALUE, and this accepts both, because
    the frames it is called with disagree:
      * UNSHIFTED -- a raw observable column, shifted here. The (col, delay)
        pair taken literally.
      * PRE-SHIFTED -- a column already holding the value abs(delay) rows in
        the past, stored under the delayed name. data_after_mde (gmn_build)
        is built this way: it is keyed on GRAPH NODE NAMES, and a lagged
        leaf's node name IS the delayed name, so the shift happened once at
        build time and the raw observable is absent unless it is
        independently a node.

    PRE-SHIFTED WINS, and the order is not arbitrary. Raw-name-first is what
    job 4744294 died of after 19h. A node's Generate() frame carries EVERY
    column but populates only that node's own coordinates; the rest are NaN
    placeholders that gain one simulated row per step. So a raw column being
    PRESENT says nothing about whether it holds data: one coordinate found a
    raw column with 1 valid row out of 37783, shifted it to all-NaN, and
    pyEDM then dropped every row carrying a NaN -- all of them. The emptied
    library made SMap's knn negative, which reached scipy as a reduction over
    an empty array. The pre-shifted column it ignored held 37478 real values.
    Pre-shifted is also the semantically right source: a lagged driver is a
    leaf, hence an external input fed from REAL recorded data, which is
    exactly what gmn_build shifted into that column.

    delay == 0 stays on the raw branch and has to: there is no '::delay0'
    alias, and a delay-0 coordinate is the node's own current value or a
    contemporaneous driver -- under Generate() that must come from the
    GROWING SIMULATED column, never from a frozen recording, or the
    simulator reads the very values it is being scored against.

    WHY THE SPEC IS NOT REWRITTEN UPSTREAM INSTEAD. The spec's delay is the
    only thing that tells this function how many leading rows are unusable.
    Rewriting a pre-shifted coordinate to (leaf_name, 0) -- the obvious
    alternative -- would erase that number from the only place `offset` is
    computed from, so nothing would be dropped and the NaNs physically
    present at the top of the pre-shifted column would flow into the SMap
    fit. That is a silent wrong answer where this is a loud KeyError. So the
    delay stays in the spec and only the LOOKUP branches.

    Raises KeyError naming both spellings it tried when a coordinate resolves
    to neither, and ValueError when one resolves to a column that exists but
    is all-NaN over the usable rows -- see the guard at the end for why that
    is worth its own error.
    '''
    columns = {}
    for col, delay in embedding_spec:
        delayed = _delayed_column_name(col, delay) if delay else None
        if delayed is not None and delayed in data.columns:
            series = data[delayed]          # already shifted; taking it as-is IS the shift
        elif col in data.columns:
            series = data[col].shift(-delay)
        else:
            tried = f'the pre-shifted name {delayed!r} and ' if delayed else ''
            raise KeyError(
                f'generalized_embed: embedding coordinate ({col!r}, {delay}) '
                f'matches no column in the frame it was given -- tried '
                f'{tried}the raw name {col!r} (to shift here). The frame has '
                f'{len(data.columns)} columns. A graph node and its data column '
                f'have drifted apart; see embedding_spec_for() and '
                f"gmn_build.py's data_after_mde block.")
        columns[f'{col}(t-{abs(delay)})'] = series.to_numpy()

    embedded = pd.DataFrame(columns, index=data.index)
    offset = max(abs(delay) for _col, delay in embedding_spec)
    embedded = embedded.iloc[offset:].reset_index(drop=True)

    # An all-NaN coordinate is never a usable embedding, and left alone it
    # does NOT fail where it happened: pyEDM drops every row carrying a NaN,
    # so one such column empties the whole library and the failure surfaces
    # much later, and unrecognisably, as scipy reducing over an empty array.
    # Job 4744294 spent 19h reaching exactly that, and the run then recorded
    # the network as simply having produced no output. Naming the coordinate
    # here turns the next occurrence into a diagnosis.
    empty = [name for name in embedded.columns if embedded[name].isna().all()]
    if empty:
        shown = ', '.join(empty[:6]) + (' ...' if len(empty) > 6 else '')
        raise ValueError(
            f'generalized_embed: {len(empty)} of {len(embedded.columns)} embedding '
            f'coordinate(s) resolved to an all-NaN column over the {len(embedded)} '
            f'usable rows (offset {offset}): {shown}. The frame supplied a column '
            f'under that name but it carries no data in this window -- see this '
            f"function's PRE-SHIFTED WINS note: a node's Generate() frame holds "
            f'placeholder columns for every coordinate it does not own, so a name '
            f'being present does not mean it has values.')
    return embedded, offset


# =====================================================================
# Is this node fed from real data rather than simulated?
# =====================================================================

def _is_real_data_external_input(node):
    '''True for a zero-driver node under EXTERNAL_INPUT_MODE='real_data'.

    Such a node is genuinely exogenous: nested-MDE found nothing that drives
    it, so there is nothing to simulate and no embedding to build. Its
    Forecast/Generate short-circuit to the recorded values instead. See
    network.partition_network() and config.py's EXTERNAL_INPUT_MODE.
    '''
    graph = node.Network.Graph
    return (node.name in graph.graph.get('external_inputs', set())
            and graph.graph.get('external_input_mode', 'real_data') == 'real_data')


# =====================================================================
# Patch 1 -- Node.__init__
# =====================================================================

_real_node_init = gmn.Node.__init__


def _generalized_node_init(self, args, Network, nodeName):
    '''gmn.Node.__init__, with self.libEnd_i corrected for SMap nodes.

    The installed package computes that library-end index as
    `self.data.shape[0] - Parameters.E * abs(Parameters.tau)`, which assumes
    one shared uniform embedding. The generalized embedding this pipeline
    builds has its own, generally different, per-node offset. Recomputed
    exactly once here, right after __init__ sets the node's pristine
    self.data, since libEnd_i is meant to reflect that original library size
    before Generate() starts appending values.

    Skipped for a real-data external input: it has no embedding at all, so
    there is no spec to size libEnd_i from. The uncorrected value is left in
    place as inert data for a node that never runs an SMap/Embed call.
    '''
    _real_node_init(self, args, Network, nodeName)
    if self.FunctionType != FunctionType.SMap:
        return
    if _is_real_data_external_input(self):
        return
    spec = _node_embedding_spec(self)
    self.libEnd_i = self.data.shape[0] - max(abs(delay) for _col, delay in spec)


# =====================================================================
# Patch 2 -- Node.Forecast
# =====================================================================

def _lib_pred_ranges(Parameters, n_rows, lib_end_i):
    '''gmn's own lib/pred defaulting, used by both patched methods.'''
    lib = (Parameters.lib if len(Parameters.lib) and not Parameters.lib.isspace()
           else '1 %d' % lib_end_i)
    pred = (Parameters.pred if len(Parameters.pred) and not Parameters.pred.isspace()
            else '%d %d' % (n_rows - 1, n_rows))
    return lib, pred


def _forecast_external_input(self):
    '''A real-data external input's Forecast(): the recorded values.

    A one-step-ahead skill metric is meaningless for a variable this network
    does not predict at all -- self.data is already fully real throughout
    Forecast mode, so there is nothing to forecast beyond the real series.
    Returns (time, real values), matching Forecast()'s own (time,
    predictions) shape, so gmn's per-node loop (which iterates every graph
    node with no partition awareness of its own) does not crash on an empty
    embedding spec.

    SLICED TO Parameters.pred, not returned whole, and that matters. gmn's
    Network sets dataLib_i to the whole dataset in Forecast mode, so
    self.data holds everything here. Returning it in full made this node's
    output thousands of rows longer than every SMap node's short pred-range
    Series -- and because gmn.GMN.Forecast() assembles DataOut by plain
    per-node column assignment, taking the shared 'Time' column from
    TopologicalSorted[0] (almost always a leaf like this one, having no
    in-edges), that full-length passthrough silently became DataOut's row
    index and its 'Time' values. Every SMap node's own 0-based Series then
    pandas-aligned onto just the first few hundred of those labels,
    mislabelling predictions from near generate_start as though they had
    occurred at the very start of the recording -- which corrupted both the
    plotted x-position and the Forecast scores in the figure titles.
    '''
    time_col = self.data.columns[0]
    pred = self.Parameters.pred
    if len(pred) and not pred.isspace():
        pred_start, pred_end = (int(x) for x in pred.split())
        sliced = self.data.iloc[pred_start - 1:pred_end].reset_index(drop=True)
        return sliced[time_col], sliced[self.name]
    return self.data[time_col], self.data[self.name]


def _forecast_simplex(self):
    '''gmn's Simplex Forecast branch, carried unchanged.

    Never reached in this pipeline -- every node is SMap. Note that gmn's own
    copy of this method calls Embed() without importing it, so a Simplex
    Forecast would have raised NameError there; that is moot here only
    because this branch does not call Embed at all.
    '''
    Parameters, data = self.Parameters, self.data
    lib, pred = _lib_pred_ranges(Parameters, data.shape[0], self.libEnd_i)
    S = self.Function(
        dataFrame=data, lib=lib, pred=pred, E=Parameters.E, Tp=Parameters.Tp,
        knn=Parameters.knn, tau=Parameters.tau,
        exclusionRadius=Parameters.exclusionRadius, columns=Parameters.columns,
        target=Parameters.target, embedded=Parameters.embedded,
        validLib=Parameters.validLib, generateSteps=Parameters.generateSteps)
    return S.iloc[:, 0], S['Predictions']


def _forecast_smap(self):
    '''The SMap Forecast branch -- the only one this pipeline exercises.

    Replaces gmn's `Embed(E, tau, columns)` call with generalized_embed(),
    i.e. the per-driver-delay embedding read off the network graph instead of
    one uniform (E, tau) applied to every predictor. Two fixes to gmn's own
    version come with it:

      * `columns=df.columns[1:-1]` dropped BOTH the time column and the
        spec's last entry -- which, since the spec is self-then-in-edges, is
        always the node's last discovered driver. Harmless while every node
        carried several self-lags; a real failure once a node's spec became
        exactly its drivers (a 1-driver node had NO columns left, so pyEDM
        raised "columns required", and a 2+-driver node silently lost one).
        Fixed to `df.columns[1:]`, dropping only time.
      * the divergence check read `val.max()` on a 2-tuple, which raised
        AttributeError unconditionally rather than on actual divergence.
        Fixed to check the predictions Series.

    The time column is inserted from data's own offset-sliced, reset-index
    time values rather than gmn's implicit index-aligned insert, because
    generalized_embed() already resets its index to 0-based.
    '''
    Parameters, data = self.Parameters, self.data
    spec = _node_embedding_spec(self)
    df, offset = generalized_embed(data, spec)
    df.insert(0, data.columns[0],
              data[data.columns[0]].iloc[offset:].reset_index(drop=True))

    lib, pred = _lib_pred_ranges(Parameters, df.shape[0], self.libEnd_i)
    S = self.Function(
        dataFrame=df, lib=lib, pred=pred,
        E=len(spec), Tp=Parameters.Tp, knn=Parameters.knn, tau=1,
        theta=Parameters.theta, exclusionRadius=Parameters.exclusionRadius,
        columns=df.columns[1:], target=Parameters.target + '(t-0)', solver=None,
        embedded=Parameters.embedded, validLib=Parameters.validLib,
        generateSteps=Parameters.generateSteps)

    predictions = S['predictions']
    val = (predictions.iloc[:, 0], predictions['Predictions'])
    if val[1].max() > 1E15:
        raise RuntimeError(f'Forecast:SMap() Node: {self.name}   '
                           f'Divergence detected (>1E15)')
    return val


# Forecast: the two branches gmn implements, and the five it does not.
_FORECAST = {
    FunctionType.Simplex.value: _forecast_simplex,
    FunctionType.SMap.value: _forecast_smap,
}
_FORECAST_UNAVAILABLE = ('kedmSimplex', 'kedmSMap', 'Linear', 'SVR', 'knn')


def _generalized_node_forecast(self):
    '''gmn.Node.Forecast, with the SMap branch replaced.

    See _forecast_smap() for what changed and why, and _FORECAST above for
    the branch table. gmn's own method raises for every type outside
    Simplex/SMap, and so does this.
    '''
    if self.args.DEBUG:
        print('-----> Node:Forecast() : ', self.FunctionType.name, self.name)
        print(self.data.tail(2), flush=True)

    if _is_real_data_external_input(self):
        return _forecast_external_input(self)

    branch = _FORECAST.get(self.FunctionType.value)
    if branch is None:
        name = self.FunctionType.name
        if name in _FORECAST_UNAVAILABLE:
            raise RuntimeError(f'Forecast {name} function not available')
        raise RuntimeError(f'Forecast: unknown function type {name}')
    val = branch(self)

    if self.args.DEBUG:
        print(self.name, 'val:', val)
        print('<----- Node:Forecast() : ', self.FunctionType.name, self.name)
        print('', flush=True)
    return val


# =====================================================================
# Patch 3 -- Node.Generate
# =====================================================================

def _generate_external_input(self):
    '''A real-data external input's Generate(): the next recorded value.

    Genuinely not simulated. self.data has already had the previous step's
    output appended by the time this runs, so its length IS the absolute row
    index of the next real row to hand back: self.data and the real frame
    start in lockstep at generate_start rows and grow by exactly one row per
    step, for every node, via the same shared lastDataOut.
    '''
    row_i = len(self.data)
    real_data = self.Network.data
    if row_i >= len(real_data):
        raise RuntimeError(
            f"Generate: external input '{self.name}' ran out of real data at row "
            f"{row_i} (only {len(real_data)} rows available) -- shrink GENERATE_LENGTH "
            f"or switch EXTERNAL_INPUT_MODE to 'self_simulate'.")
    return float(real_data[self.name].iloc[row_i])


def _generate_simplex(self):
    '''gmn's Simplex Generate branch, carried unchanged (never reached here).'''
    Parameters, data = self.Parameters, self.data
    lib, pred = _lib_pred_ranges(Parameters, data.shape[0], self.libEnd_i)
    S = self.Function(
        dataFrame=data, lib=lib, pred=pred, E=Parameters.E, Tp=Parameters.Tp,
        knn=Parameters.knn, tau=Parameters.tau,
        exclusionRadius=Parameters.exclusionRadius, columns=Parameters.columns,
        target=Parameters.target, embedded=Parameters.embedded,
        validLib=Parameters.validLib, generateSteps=Parameters.generateSteps)
    return S['Predictions'].iloc[-1]


def _generate_smap(self):
    '''The SMap Generate branch -- the only one this pipeline exercises.

    Same substitution and the same `columns[1:-1]` -> `columns[1:]` fix as
    _forecast_smap(); see that function. Unlike Forecast's SMap branch,
    gmn's own Generate never tried to reuse real time values for its
    inserted time column (just a synthetic 0-based arange), so there is no
    index-alignment subtlety to correct here.
    '''
    Parameters, data = self.Parameters, self.data
    spec = _node_embedding_spec(self)
    df, _offset = generalized_embed(data, spec)
    df.insert(0, 'time', np.arange(df.shape[0]), allow_duplicates=True)

    lib, pred = _lib_pred_ranges(Parameters, df.shape[0], self.libEnd_i)
    if not (len(Parameters.pred) and not Parameters.pred.isspace()):
        # The new row is N+1; a Tp-step map predicts it from the state at row
        # N+1-Tp, so that is the last pred row. With Tp=1 this is gmn's own
        # "N-1 N". gmn predicted from row N whatever Tp was and appended the
        # result as row N+1, compressing time Tp-fold. This way a Tp-step
        # model generates as Tp interleaved chains sharing their inputs.
        tp, n = int(Parameters.Tp), df.shape[0]
        pred = '%d %d' % (n - tp, n + 1 - tp)
    S = self.Function(
        dataFrame=df, lib=lib, pred=pred,
        E=len(spec), Tp=Parameters.Tp, knn=Parameters.knn, tau=1,
        theta=Parameters.theta, exclusionRadius=Parameters.exclusionRadius,
        columns=df.columns[1:], target=Parameters.target + '(t-0)', solver=None,
        embedded=Parameters.embedded, validLib=Parameters.validLib,
        generateSteps=Parameters.generateSteps)

    val = S['predictions']['Predictions'].iloc[-1]
    if abs(val) > 1E15:
        raise RuntimeError(f'Generate:SMap() Node: {self.name}   '
                           f'Divergence detected (>1E15)')
    noise = self.Network.Graph.graph.get('generate_noise') or {}
    if noise.get('mode', 'none') == 'local_residual':
        val = local_residual_draw(self, df, lib, val, noise.get('seed', 0))
    return val


def local_residual_draw(node, df, lib, val, seed):
    '''One stochastic Generate() step: the S-map prediction `val` plus a
    residual drawn from the node's own library analogues.

    A deterministic free run iterates the CONDITIONAL MEAN of the next value.
    For a TSLS series the next value is bimodal -- x + dt (no spike) or ~0
    (spike) -- and its mean, (x + dt)(1 - p), is neither: it ramps too slowly
    and never resets, so the run drifts smoothly instead of firing. Adding the
    same small isotropic noise to every node, the way spiking-network models
    keep activity alive, does not help here: a spiking model turns noise into
    events through its threshold, and S-map has none, so isotropic noise
    only jitters the mean path -- and every downward wobble breaks TSLS's
    "only ever step down to reset" constraint.

    So this samples the conditional distribution instead, the analogue
    (k-nearest-neighbour) bootstrap way: take the E+1 library states nearest
    the current one (Simplex's neighbourhood, Euclidean on pyEDM's raw
    coordinates), pick ONE with Simplex's weights exp(-d/d_min), and add its
    deviation from the weighted neighbourhood mean:

        x(t+Tp) = S-map(t) + [ y_j - sum_i w_i y_i / sum_i w_i ]

    If one neighbour in 20 is followed by a reset, a reset is drawn with
    probability ~1/20 from that state. The noise is state-dependent and has
    the data's own shape; there is no amplitude to tune.

    `lib` is pyEDM's 1-based "start end" library; a neighbour's future must
    lie inside it. The KD-tree is built once per node per run and cached on
    the node (the library never changes during a run). The RNG is seeded
    from (`seed`, node name), so a seed reproduces the run exactly.'''
    state = getattr(node, '_analog_state', None)
    if state is None:
        import zlib
        from scipy.spatial import cKDTree
        cols = list(df.columns[1:])
        tp = int(node.Parameters.Tp)
        lib_start, lib_end = (int(x) for x in str(lib).split())
        X = df[cols].to_numpy(dtype=float)[lib_start - 1:lib_end - tp]
        Y = df[node.Parameters.target + '(t-0)'].to_numpy(dtype=float)[lib_start - 1 + tp:lib_end]
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
        X, Y = X[ok], Y[ok]
        state = {
            'cols': cols,
            'tree': cKDTree(X) if len(X) else None,
            'Y': Y,
            'k': min(len(cols) + 1, len(X)),
            'rng': np.random.default_rng([int(seed), zlib.crc32(node.name.encode())]),
        }
        node._analog_state = state

    if state['tree'] is None or state['k'] < 1:
        return val
    # The state the prediction was made from: row N+1-Tp (see _generate_smap).
    query = df[state['cols']].to_numpy(dtype=float)[-int(node.Parameters.Tp)]
    if not np.all(np.isfinite(query)):
        return val
    dist, idx = state['tree'].query(query, k=state['k'])
    dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
    w = np.exp(-dist / max(float(dist[0]), 1e-6))
    w = w / w.sum()
    y = state['Y'][idx]
    j = state['rng'].choice(len(idx), p=w)
    return float(val + (y[j] - float(np.dot(w, y))))


def _generate_kedm_simplex(self):
    '''gmn's kedmSimplex Generate branch, carried unchanged.'''
    Parameters = self.Parameters
    data_ = self.data[Parameters.columns]          # DataFrame; includes target
    targetLen = Parameters.E * abs(Parameters.tau)
    library = data_.iloc[:self.libEnd_i]
    S = self.Function(
        lib=library.to_numpy(), pred=data_.iloc[-targetLen:].to_numpy(),
        target=library[Parameters.target].to_numpy(),
        E=Parameters.E, tau=abs(Parameters.tau), Tp=Parameters.Tp)
    return S[-1]


def _generate_kedm_smap(self):
    '''gmn's kedmSMap Generate branch, carried unchanged.'''
    Parameters = self.Parameters
    data_ = self.data[self.name]                   # Series
    targetLen = Parameters.E * abs(Parameters.tau)
    library = data_.iloc[:self.libEnd_i]
    S = self.Function(
        lib=library.to_numpy(), pred=data_.iloc[-targetLen:].to_numpy(),
        target=library.to_numpy(),
        E=Parameters.E, tau=abs(Parameters.tau), Tp=Parameters.Tp,
        theta=Parameters.theta)
    return S[-1]


def _generate_regression(self, fit_model):
    '''Shared shape of gmn's Linear / SVR / knn Generate branches.

    All three fit a model on the node's own columns and predict one step
    from FindNextX(); only the estimator differs, so `fit_model(X, y)`
    is the one thing each passes in.
    '''
    Parameters = self.Parameters
    X = self.data[Parameters.columns].to_numpy()
    y = self.data[Parameters.target].to_numpy()
    model = fit_model(X, y)
    return model.predict(self.FindNextX(X))[0]


def _generate_linear(self):
    '''gmn's Linear Generate branch, carried unchanged.'''
    def fit(X, y):
        lr = self.Function()
        lr.fit(X, y)
        return lr
    return _generate_regression(self, fit)


def _generate_svr(self):
    '''gmn's SVR Generate branch, carried unchanged.'''
    def fit(X, y):
        svr = self.Function(kernel='rbf', gamma='scale', tol=0.001, epsilon=0.1,
                            shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        svr.fit(X, y)
        return svr
    return _generate_regression(self, fit)


def _generate_knn(self):
    '''gmn's knn Generate branch, carried unchanged.

    Differs from the other two regressions in one respect, kept as gmn has
    it: with more than one column, the node's own column is dropped from X.
    '''
    Parameters = self.Parameters
    X = self.data[Parameters.columns]
    if len(Parameters.columns) > 1:
        X = X.drop(self.name, axis='columns')
    knn = self.Function(n_neighbors=10, weights='distance')
    knn.fit(X.to_numpy(), self.data[Parameters.target].to_numpy())
    return knn.predict(self.FindNextX(X.to_numpy()))[0]


# Generate: gmn implements every function type.
_GENERATE = {
    FunctionType.Simplex.value: _generate_simplex,
    FunctionType.SMap.value: _generate_smap,
    FunctionType.kedmSimplex.value: _generate_kedm_simplex,
    FunctionType.kedmSMap.value: _generate_kedm_smap,
    FunctionType.Linear.value: _generate_linear,
    FunctionType.SVR.value: _generate_svr,
    FunctionType.knn.value: _generate_knn,
}


def _generalized_node_generate(self, lastDataOut):
    '''gmn.Node.Generate, with the SMap branch replaced.

    See _generate_smap() for what changed and _GENERATE above for the branch
    table.
    '''
    if self.args.DEBUG:
        print('-----> Node:Generate() : ', self.FunctionType.name, self.name)
        print(self.data.tail(2))
        print('lastDataOut:')
        print(lastDataOut, flush=True)

    if lastDataOut is not None:
        # Only this node's own columns. gmn appends the whole network row, so
        # every node's frame would grow network-wide -- memory and copying in
        # nodes^2 * rows -- while its embedding reads only time, its
        # predecessors and itself, which are exactly self.data's columns.
        self.data = pd.concat([self.data, lastDataOut.reindex(columns=self.data.columns)])

    # After the append, so len(self.data) is the next real row index.
    if _is_real_data_external_input(self):
        return _generate_external_input(self)

    if self.args.DEBUG:
        print('  Appended data : shape: ', self.data.shape)
        print(self.data.tail(3))
        print('', flush=True)

    branch = _GENERATE.get(self.FunctionType.value)
    if branch is None:
        raise RuntimeError(f'Generate: unknown function type {self.FunctionType.name}')
    val = branch(self)

    if self.args.DEBUG:
        print(self.name, 'val:', val)
        print('<----- Node:Generate() : ', self.FunctionType.name, self.name)
        print('', flush=True)
    return val


# =====================================================================
# The patches themselves. Applied on import -- see this module's docstring.
# _real_node_init above already holds the original __init__, captured before
# this rebinds it, and the replacement calls it.
# =====================================================================

gmn.Node.__init__ = _generalized_node_init
gmn.Node.Forecast = _generalized_node_forecast
gmn.Node.Generate = _generalized_node_generate
