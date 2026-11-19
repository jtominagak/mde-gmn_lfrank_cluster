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
from mde_gmn.columns import _delayed_column_name


def _embedding_spec_for(graph, node_name):
    '''Builds one node's (column, delay) embedding coordinates directly
    from the network graph -- a pure function, reusable both by
    _node_embedding_spec() (which caches this on a live gmn.Node object,
    for gmn's own monkeypatched Forecast()/Generate()) and by
    retune_theta_for_all_nodes() (which builds embeddings directly with
    pandas/pyEDM outside gmn's Node/Network machinery entirely, before any
    gmn.GMN object exists).

    MultiDiGraph (MODE='MDE-GMN', built by build_nested_mde_network()): a
    node's coordinates are its DISCOVERED drivers, not a fixed set. The deep
    self-lags (t-1, t-3, t-5, ...) that used to ride along on every node
    regardless of what MDE found are gone. But self ALWAYS keeps its own
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

    Two kinds of node, and this is where the naming diverges (network.py's
    module docstring is the authority):
      * a REAL observable at delay 0 is a graph node under its own name,
        e.g. 'CA1_3_tsls', and may acquire drivers of its own;
      * a LAGGED observable is its OWN LEAF NODE under a delayed name,
        e.g. 'CA1_3_tsls::delay-49', carrying lagged_of='CA1_3_tsls' and
        lag=-49. A leaf never has in-edges, so it can never close a
        cycle, and it has zero drivers by construction -- which is
        exactly why partition_network() classifies it as an external
        input that the simulation feeds from recorded data.
    So a lagged copy of B IS a second node, distinct from B's own node,
    and B's own node need not be in the graph at all for 'B::delay-49' to
    be in it. (An earlier version of this docstring claimed the opposite
    -- "exactly one graph node per raw observable" -- and that was simply
    wrong; _accept_mde_result_edges() calls add_node() with the RAW name
    only on the delay == 0 branch, and with the delayed name on the
    other.) The loop below therefore resolves a leaf in-edge back to
    (lagged_of, edge delay), so that the coordinate names a raw data
    column rather than a node name.

    CAVEAT -- that resolution assumes the caller's frame carries the RAW
    observable column, and data_after_mde (gmn_build.py) does not: it is
    keyed on graph node names, storing each leaf as a PRE-SHIFTED column
    under the leaf's own name, and omitting the raw observable unless
    that observable is independently a node. Calling this with that frame
    raises KeyError on the first leaf in-edge -- job 4741044 died exactly
    here, on 'CA1_12_tsls'. Unresolved: either this function emits the
    leaf's own name at delay 0 (and the caller accounts for the lag the
    column already carries when sizing its offset), or the frame also
    carries raw observable columns.

    Partitioning zero-driver nodes out stands on its own regardless of
    the above: such a node isn't something nested-MDE discovered as a
    network member at all, so counting it as an interior (simulated) node
    overstates what was actually found.

    Plain DiGraph (MODE='GMN', from apps.CreateNetwork -- no per-edge delay
    data exists in that mode at all): falls back to the original uniform
    E-lagged-copies-of-every-column scheme, using
    graph.graph['fallback_E']/['fallback_tau'] (set in build_network() from
    that mode's own CCM E/tau grid search) in place of a per-node nested-MDE
    result it does not have. MODE='GMN' never runs the nested-MDE builder at
    all, so it has no "discovered driver" concept to work from.

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
    implies for lower-offset nodes). self_max remains an upper bound on any
    node's real offset even though interior nodes no longer carry the full
    self_delay_set: an in-edge delay is structurally never deeper than the
    delays LAG_SET spans. So this can only over-count, never
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
    shifted into place.

    TWO WAYS A FRAME CAN CARRY A LAGGED VALUE, and this function accepts
    both, because the frames it is called with disagree:

      * UNSHIFTED -- a raw observable column, which this function shifts
        itself. That is the (col, delay) pair taken literally.
      * PRE-SHIFTED -- a column already holding the value abs(delay) rows
        in the past, stored under the delayed name '<col>::delay<delay>'.
        data_after_mde (gmn_build.py) is built this way: it is keyed on
        GRAPH NODE NAMES, and a lagged leaf's node name IS the delayed
        name, so the shift happened once at build time and the raw
        observable is absent unless it is independently a node.

    WHICH BRANCH WINS, and why the order is not arbitrary: a nonzero delay
    tries the PRE-SHIFTED name first, and the raw name only as a fallback.
    Raw-name-first is what job 4744294 died of, after 19h. A node's
    Generate() frame carries EVERY column but populates only that node's own
    coordinates; the rest are NaN placeholders that gain one simulated row
    per step. So a raw column being PRESENT says nothing about whether it
    holds data -- OFC_10_tsls's coordinate (CA1_55_tsls, -305) found a raw
    column (that observable is independently a node elsewhere in the same
    network) holding 1 valid row out of 37783, shifted it to all-NaN, and
    pyEDM then dropped every row carrying a NaN -- all 37478 of them. The
    emptied library made SMap's knn = len(lib_i) - 1 = -1, which reached
    scipy as a reduction over an empty array ('zero-size array to reduction
    operation maximum which has no identity'). The pre-shifted column it
    ignored held 37478 real values. Pre-shifted is also the semantically
    right source: a lagged driver is a leaf node, hence an external input
    fed from REAL recorded data, which is exactly what gmn_build.py shifted
    into that column at build time.

    delay == 0 stays on the raw branch, and has to: there is no '::delay0'
    alias (columns.py gives a delay-0 candidate the bare observable name),
    and a delay-0 coordinate is the node's own current value or a
    contemporaneous driver -- under Generate() that must come from the
    GROWING SIMULATED column, never from a frozen recording, or the
    simulator reads the very values it is being scored against. The raw
    branch also still serves MODE='GMN', whose uniform E/tau spec invents
    delays that no leaf column was ever built for.

    Resolving that here, rather than by rewriting the spec upstream, is
    deliberate, and the reason is the `offset` line below. The spec's delay
    is the only thing that tells this function how many leading rows are
    unusable: for head_position_x a deepest input of delay -393 is what
    makes offset 393 and drops those 393 NaN rows before pyEDM sees them.
    Rewriting a pre-shifted coordinate to (leaf_name, 0) -- the obvious
    alternative -- would erase that 393 from the only place offset is
    computed from, so max() would return 0, nothing would be dropped, and
    the NaNs physically present at the top of the pre-shifted column (put
    there by gmn_build.py's own shift) would flow into the SMap fit. That
    is a silent wrong answer where this is a loud KeyError. So the delay
    stays in the spec and ONLY the lookup branches; offset is computed
    from the spec exactly as before, untouched by any of this.

    Keeping the spec intact also keeps the output column names
    ('CA1_12_tsls(t-49)') identical under either branch, which matters
    because gmn's SMap finds its prediction target by looking up
    Parameters.target + '(t-0)' by name.

    Raises KeyError naming both spellings it tried when a coordinate
    resolves to neither -- see _embedding_spec_for()'s CAVEAT for the
    drift this guards (job 4741044 died on exactly that, after 40h).'''
    columns = {}
    for col, delay in embedding_spec:
        # Pre-shifted FIRST, raw only as a fallback -- see the docstring's
        # WHICH BRANCH WINS note. delay == 0 has no pre-shifted spelling and
        # must read the raw (growing, simulated) column.
        delayed = _delayed_column_name(col, delay) if delay else None
        if delayed is not None and delayed in data.columns:
            # Already shifted at build time -- taking it as-is IS the shift.
            series = data[delayed]
        elif col in data.columns:
            series = data[col].shift(-delay)
        else:
            tried = f'the pre-shifted name {delayed!r} and ' if delayed else ''
            raise KeyError(
                f'_generalized_embed: embedding coordinate ({col!r}, {delay}) '
                f'matches no column in the frame it was given -- tried '
                f'{tried}the raw name {col!r} (to shift here). The frame has '
                f'{len(data.columns)} columns. A graph node and its data column '
                f'have drifted apart; see _embedding_spec_for() and '
                f'gmn_build.py\'s data_after_mde block.')
        columns[f'{col}(t-{abs(delay)})'] = series.to_numpy()

    embedded = pd.DataFrame(columns, index=data.index)
    offset = max(abs(delay) for _, delay in embedding_spec)
    embedded = embedded.iloc[offset:].reset_index(drop=True)

    # An all-NaN coordinate is never a usable embedding, and left alone it does
    # NOT fail where it happened. pyEDM drops every row carrying a NaN, so one
    # such column empties the entire library and the failure surfaces much
    # later, and unrecognisably, as scipy reducing over an empty array
    # ('zero-size array to reduction operation maximum which has no identity').
    # Job 4744294 spent 19h reaching exactly that, and the pipeline then
    # recorded the whole network as simply having produced no output. Naming
    # the coordinate here turns the next occurrence into a diagnosis.
    empty = [name for name in embedded.columns if embedded[name].isna().all()]
    if empty:
        shown = ', '.join(empty[:6]) + (' ...' if len(empty) > 6 else '')
        raise ValueError(
            f'_generalized_embed: {len(empty)} of {len(embedded.columns)} embedding '
            f'coordinate(s) resolved to an all-NaN column over the '
            f'{len(embedded)} usable rows (offset {offset}): {shown}. The frame '
            f'supplied a column under that name but it carries no data in this '
            f'window -- see this function\'s WHICH BRANCH WINS note: a node\'s '
            f'Generate() frame holds placeholder columns for every coordinate it '
            f'does not own, so a name being present does not mean it has values.')
    return embedded, offset

def _local_residual_draw(node, df, lib, val, seed):
    '''One stochastic Generate() step: the S-map prediction `val` plus a
    residual drawn from this node's own library analogues.

    WHY. A deterministic Generate() iterates the CONDITIONAL MEAN of the next
    value. For a TSLS series the next value is bimodal -- x + dt (no spike) or
    ~0 (spike) -- and its mean, (x + dt)(1 - p), is neither: it ramps too
    slowly and never resets. Iterated, it drifts smoothly instead of firing.
    That is what job 4799404's free run shows: across all 116 simulated
    nodes, 0 resets in 400 steps, ramps at 25-60% of the true 0.025/row
    slope, some ramping DOWNWARD (OFC_201_tsls 17.04 -> 16.47), which TSLS
    cannot do, and a standard deviation of ~0.3 against ~2.9 in the
    recording. Adding the same small isotropic noise to every node (the
    inhibitory-network trick) does not fix this. That trick works in a
    spiking model because a threshold turns noise into spikes; S-map has no
    threshold, so isotropic noise only jitters the mean path. It also breaks
    TSLS's own constraints, since TSLS never steps down except to reset, and
    every downward wobble scores as a spurious reset in the f1 decoder.

    WHAT. Sample from the conditional distribution instead of its mean, in
    the classic analogue / k-nearest-neighbour bootstrap way (Lall & Sharma
    1996). Take the E+1 library states nearest the current state (Simplex's
    own neighbourhood, Euclidean on the same raw coordinates pyEDM uses).
    Pick ONE with Simplex's weights exp(-d/d_min), and add its deviation from
    the weighted neighbourhood mean:

        x(t+Tp) = S-map(t) + [ y_j - sum_i w_i y_i / sum_i w_i ]

    If 1 neighbour in 20 is followed by a reset, a reset is drawn with
    probability ~1/20 at that state, and lands near 0, because y_j is ~0 and
    S-map ~ the neighbourhood mean. Off-reset draws continue the ramp at the
    library's own slope. The noise is therefore state-dependent and has the
    data's own shape, with no amplitude to tune. With an empty neighbourhood
    (shouldn't happen) the deterministic value is returned.

    `lib` is the same 1-based "start end" library pyEDM was given; a
    neighbour's future must lie inside it. The KD-tree is built once per
    node per Generate() and cached on the node, because the library never
    changes during a run (only rows past it are appended). The RNG is seeded
    from (`seed`, node name), so a given seed reproduces the run exactly.'''
    state = getattr(node, '_analog_state', None)
    if state is None:
        from scipy.spatial import cKDTree
        import zlib
        cols = list(df.columns[1:])
        target = node.Parameters.target + '(t-0)'
        tp = int(node.Parameters.Tp)
        lib_start, lib_end = (int(x) for x in str(lib).split())
        X = df[cols].to_numpy(dtype=float)[lib_start - 1:lib_end - tp]
        Y = df[target].to_numpy(dtype=float)[lib_start - 1 + tp:lib_end]
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
    # The state the prediction was made from: row N+1-Tp (see the pred range).
    query = df[state['cols']].to_numpy(dtype=float)[-int(node.Parameters.Tp)]
    if not np.all(np.isfinite(query)):
        return val
    dist, idx = state['tree'].query(query, k=state['k'])
    dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
    d_min = max(float(dist[0]), 1e-6)
    w = np.exp(-dist / d_min)
    w = w / w.sum()
    y = state['Y'][idx]
    j = state['rng'].choice(len(idx), p=w)
    return float(val + (y[j] - float(np.dot(w, y))))


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

    Skipped entirely for an external input fed from real data (see
    partition_network()'s docstring) -- such a node has no embedding at all (_embedding_spec_for() is never reached for it; its
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

def _data_rows_to_embedded(rows, offset, node_name, clamp):
    '''Maps a pyEDM 1-based "start end" row range given in data-row
    coordinates onto the offset-trimmed frame _generalized_embed() returns
    (its row i is data row i + offset). A lib range may lose its first rows
    (clamp=True); a pred range may not, since that would silently change
    which rows are predicted.'''
    if not offset:
        return rows
    start, end = (int(x) for x in rows.split())
    start, end = start - offset, end - offset
    if start < 1:
        if not clamp:
            raise ValueError(
                f'Forecast(): node {node_name!r} has a {offset}-row embedding offset, '
                f'so pred range {rows!r} starts before its first embedded row -- '
                f'start the Forecast() window later.')
        start = 1
    return f'{start} {end}'


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

    An external input fed from real data (see partition_network()'s docstring)
    short-circuits here too, before any SMap or embedding logic. A one-step-ahead
    skill metric is meaningless for a
    variable this network doesn't predict at all (self.data is already
    fully real throughout Forecast mode -- no iterative lastDataOut
    feeding, unlike Generate() -- so there is nothing to "forecast" here
    beyond the real series itself). Returns (time, real values) as a
    trivial passthrough, matching Forecast()'s own (time, predictions)
    return shape, so gmn's per-node Forecast loop (which iterates every
    graph node structurally, with no partition awareness of its own)
    doesn't crash on an empty embedding spec.

    This passthrough is sliced to Parameters.pred's own row range rather than
    returning self.data in full. Without that, the diagnostic figure
    mislabelled the window -- see below. Network.py sets
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
    `columns`, though it stayed in `df` itself). That was harmless while every
    node was padded with several self-lags: losing one entry of many barely
    mattered. Once a node's spec became exactly its discovered drivers it
    turned into a real failure -- a 1-driver interior node had NO columns left
    (`columns[1:-1]` on a 2-column frame is empty, so pyEDM raises "columns
    required"), and a node with 2+ drivers silently lost its last one. Fixed to
    `df.columns[1:]`, dropping only `time`.'''
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

        # exp/spike-conversion-general fix: a configured lib/pred range is in
        # DATA-row coordinates (that is how the real-data passthrough above and
        # every offset-0 node read it, and gmn takes DataOut's shared Time
        # column from TopologicalSorted[0], an offset-0 source node). `df`
        # starts `offset` rows into the data, so the same range has to be
        # moved back by this node's own offset, or a node with an offset-k
        # embedding predicts rows k later than its Time label says (it showed
        # up as Forecast() "leading" the data by the deepest edge delay).
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = _data_rows_to_embedded(Parameters.lib, offset, self.name, clamp=True)
        else :
            lib  = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = _data_rows_to_embedded(Parameters.pred, offset, self.name, clamp=False)
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

    An external input fed from real data (see partition_network()'s docstring)
    short-circuits here, before any SMap or embedding logic. It is genuinely not
    simulated: its "generated" value each step is just the
    real recorded value at that row, read straight from
    self.Network.data (loaded once, in full, from DATA_AFTER_MDE -- see
    build_gmn_config()'s own docstring). self.data has already had
    lastDataOut appended above by the time this check runs, so its length
    IS the absolute row index of the next real row to hand back (both
    start in lockstep at generate_start real rows, then grow by exactly
    one row per step, for every node, via the same shared lastDataOut).

    Also applies the same `columns=df.columns[1:-1]` -> `df.columns[1:]` fix
    that _generalized_node_forecast()'s docstring describes: the old slice
    silently dropped a node's LAST discovered driver from its SMap embedding,
    which the old self-lag padding masked and a 1-driver interior node does
    not.'''
    if self.args.DEBUG :
        print( '-----> Node:Generate() : ', self.FunctionType.name, self.name )
        print( self.data.tail( 2 ) );
        print( 'lastDataOut:' ); print( lastDataOut, flush = True )

    if not ( lastDataOut is None ):
        # Append only THIS node's own columns. gmn's Generate.py appends the
        # whole step row, which carries every node in the network, so after
        # the first step every node's frame was as wide as the network:
        # memory and per-step copying both grew as nodes^2 * rows. That is
        # what killed job 4799407 (800 nodes x 800 columns x ~33k rows x 8 B
        # is ~170 GB per Generate(), against 240 GB requested) and most of
        # the 65 min each Generate() took in job 4799404. A node's embedding
        # only ever reads its own columns (time, its predecessors, itself),
        # and those are exactly self.data's columns from __init__, so the
        # values every SMap call sees are unchanged. The extra columns were
        # also the hazard _generalized_embed()'s WHICH BRANCH WINS note
        # describes: NaN placeholders that a raw-name lookup could hit.
        self.data = pd.concat(
            [self.data, lastDataOut.reindex(columns=self.data.columns)])

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
            # The new row is N+1. A Tp-step map predicts it from the state at
            # row N+1-Tp, so that is the last pred row and iloc[-1] below is
            # its prediction for row N+1. With Tp=1 this is gmn's own "N-1 N".
            # gmn predicted from row N whatever Tp was and appended the
            # result as row N+1, which for Tp>1 compresses time by a factor
            # of Tp: every step jumped Tp rows ahead but advanced one. With
            # this, a Tp-step model generates as Tp interleaved chains, each
            # advancing Tp rows per prediction and all sharing their inputs.
            tp = int(Parameters.Tp)
            pred = "%d %d" % ( df.shape[0] - tp, df.shape[0] + 1 - tp )

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

        noise = graph.graph.get('generate_noise') or {}
        if noise.get('mode', 'none') == 'local_residual':
            val = _local_residual_draw(self, df, lib, val, noise.get('seed', 0))

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
