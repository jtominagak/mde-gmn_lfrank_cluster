'''Batch/cluster port of notebooks/tsls-mde-gmn.ipynb, executed by
scripts/batch/tsls-mde-gmn.sbatch. Every parameter is read from a
TSLS_MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
PARAMETERS block) rather than edited here directly.

This is scripts/src/simple-mde-gmn.py's own two-stage nested-MDE + GMN
pipeline, ported wholesale (not imported -- see that file's own module
docstring for why duplication over a shared module was chosen; same
reasoning applies here), fed by a TSLS conversion stage instead of
simple-mde-gmn.py's own plain CSV load, and scored throughout via
cleaned_helper_functions.py's TSLS-aware metrics instead of dimx/pyEDM/
apps.InteractionMatrix directly (see the METRIC/TARGET_METRIC docstring in
load_config() below for why TSLS-encoded spike-timing series need a
different scoring function than plain Pearson rho).

Pipeline steps (see simple-mde-gmn.py's own module docstring for the
narrative behind each one -- this docstring only covers what's specific to
the TSLS front end and scoring):

  0. Load the raw 40 Hz spike-train file, TSLS-convert every candidate
     CA1/mPFC/OFC channel (external/TSLS/cleaned_helper_functions.py's
     compute_time_since_previous_spike), then burn-in trim: TSLS is
     undefined (NaN) before a channel's own first spike, and gmn's own
     Generate()/Forecast() SMap calls (unlike cleaned_helper_functions.py's
     own proper_simplex/proper_mde, which mask NaN per-embedding internally)
     need one fully non-NaN table for the whole lib/pred range, so the
     window can only start once every included channel has fired at least
     once.
  1. MDE forward-selection: two-stage nested-MDE network construction
     (skipped if MODE='GMN') via cleaned_helper_functions.py's proper_mde
     in place of dimx.MDE -- see run_mde_for_target()'s docstring.
  2. Network construction -- two unrelated methods depending on MODE,
     identical branching to simple-mde-gmn.py's own build_network():
       - MODE='MDE-GMN': build_nested_mde_network(), two serial phases --
         Stage 1 (real-observable-only BFS, run to full completion) then
         Stage 2 (per-node lagged-embedding attachment from the CCM-pre-
         filtered observable universe, seeded with every Stage 1 node) --
         so a Stage 2 real-observable selection recurses into its own
         further Stage 2 round exactly like a Stage 1 one recurses into
         further Stage 1 rounds, instead of terminating as a leaf -- both
         stages call the TSLS-flavored run_mde_for_target() below instead
         of dimx.MDE. No CCM interaction matrix, no apps.InteractionMatrix,
         no apps.CreateNetwork -- the resulting MultiDiGraph IS the network
         handed to gmn.GMN directly.
       - MODE='GMN': the original CCM interaction matrix, now via
         cleaned_helper_functions.proper_ccm (build_interaction_matrix())
         instead of apps.InteractionMatrix's CLI subprocess -- see that
         function's own docstring for the empty-network bug this already
         fixed, and why proper_ccm has no multiprocessing-pool-nesting
         concern at all (no Pool usage in cleaned_helper_functions.py).
  3. Node generation config: SMap uniformly for every node, via this
     pipeline's own generalized per-driver-delay embedding -- ported
     unchanged from simple-mde-gmn.py (see the gmn.Node monkeypatches right
     below this docstring), since it operates purely on the network graph
     and gmn's own data structures, with no dependency on what the columns
     represent.
  4. Theta search: retunes each node's own S-map theta once per network,
     scored via cleaned_helper_functions.tsls._score(metric=TARGET_METRIC)
     instead of pyEDM's plain ComputeError rho -- E and every driver's delay
     come directly from step 2's network, never searched.
  5. Start-point sweep: at the winning theta, tries several rows for
     Generate()/Forecast() to launch from and keeps whichever matches the
     real data best.
  6. Final rebuild at the winning theta and start row, plus the Forecast()
     diagnostic and a Fig-4e-style generative fidelity metric.
  7. MDE-only diagnostics (Tp-vs-rho sweep, univariate E/tau grid) --
     skipped if MODE='GMN'. The PCA/diffusion-map Evaluate comparison
     (dx.Evaluate) simple-mde-gmn.py still has here has NO TSLS port and
     stays dropped, exactly as this file's own architecture has always
     established: dx.Evaluate depends on a live dimx.MDE result object this
     pipeline doesn't build (proper_mde replaces dimx.MDE for selection
     itself, but is not a drop-in for dx.Evaluate specifically).

Differences from simple-mde-gmn.py, all specific to the TSLS front end and
scoring (see load_config()'s own docstrings for the full rationale on each):
  - cleaned_helper_functions.py is imported from the same directory as this
    script (the sbatch script copies it there alongside the data file).
  - lib_ratio (LIB_RATIO/MDE_VALIDATION_RATIO's three-way split) is computed
    against the row count AFTER TSLS conversion + burn-in trim, not the raw
    file's row count -- the usable window is itself data-driven (depends on
    which channel fires slowest), not a constant.
  - REGION_CHANNEL_CAP here caps how many raw per-region channels get
    TSLS-converted and offered to MDE/GMN as candidates (before the _tsls
    suffix), the same semantics as simple-mde-gmn.py's own REGION_CHANNEL_CAP
    just applied one step earlier in the pipeline.
  - Every MDE/CCM/embedding-search call goes through cleaned_helper_
    functions.py's proper_mde/proper_ccm/proper_simplex/proper_embed_
    dimension/proper_predict_interval instead of dimx.MDE/apps.
    InteractionMatrix/pyEDM.EmbedDimension/ComputeError/PredictInterval --
    these accept TSLS-appropriate metrics ('f1'/'gamma' event-timing scores,
    not just Pearson rho) and mask NaN per-embedding internally.
    Consequently: MDE_GMN_EXCLUSION_RADIUS's dynamic 'auto' mode (simple-
    mde-gmn.py's per-dimx.Run()-internal-call recomputation) has no
    equivalent -- proper_mde and friends are single opaque calls taking one
    static exclusionRadius, not a staged sequence of internal calls to
    intercept. TSLS_MDE_GMN_EXCLUSION_RADIUS is a plain integer here, no
    'auto' choice. Likewise, proper_mde has no multiprocessing of its own
    (confirmed: no Pool usage anywhere in cleaned_helper_functions.py), so
    Stage 1/Stage 2's wave pools use *only* outer (per-node)
    ProcessPoolExecutor parallelism -- the inner cores-per-worker split that
    fed dimx.MDE's own `cores=` argument has nothing to feed into here, and
    is dropped. The same applies to MODE='GMN's interaction-matrix
    construction: proper_ccm has no pool either, so
    MDE_GMN_INTERACTION_MATRIX_OUTER_CORES has no TSLS equivalent and isn't
    exposed.
  - METRIC/TARGET_METRIC/GAMMA_DELTA (see load_config()) pick which of
    cleaned_helper_functions.py's scoring functions every rho-like value in
    this pipeline is computed with -- TARGETS (continuous behavioral
    traces) always via TARGET_METRIC ('correlation'), every TSLS-converted
    neuron column via METRIC ('f1' by default, decoding both series back to
    discrete spike/reset events and matching them within an adaptive
    per-event tolerance window, since plain correlation over the whole TSLS
    sawtooth is dominated by the uninformative ramp segments between
    spikes).
  - CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS/CCM_SAMPLE/CCM_SLOPE calibrate every
    CCM convergence gate in this pipeline (proper_mde's own gate,
    build_interaction_matrix()'s) to this pipeline's much larger row counts
    -- see load_config()'s own docstring for the two production failures
    (empty networks / zero selected drivers) this calibration fixed.
  - GMN_MODE='combined' (MODE='MDE-GMN' only): the top-level, per-TARGETS
    MDE selection itself (main()'s own MDE block) runs ONCE via
    cleaned_helper_functions.multi_proper_mde -- forward-selecting drivers
    that jointly predict every TARGETS entry at once, via
    run_multi_mde_for_targets() -- instead of GMN_MODE='separate's one
    independent proper_mde call per target (run_mde_for_target(), unchanged
    from before multi_proper_mde was wired in). MULTI_MDE_E/MULTI_MDE_
    TARGET_AGGREGATE (see load_config()) configure that joint call; both
    are no-ops under GMN_MODE='separate'. GMN_MODE already independently
    controlled whether network construction spans one shared graph or one
    graph per target (target_groups_for_mode()) -- this extends that same
    "combined" meaning to the MDE stage feeding it.

Everything else (differences from the notebook it mirrors) matches
simple-mde-gmn.py's own module docstring: subprocess.run instead of
get_ipython().system() (MODE='GMN' no longer needs this at all here --
proper_ccm has no CLI to shell out to), plt.close(fig) instead of
plt.show(), one flat RESULTS_DIR, and CORES threaded through every
*_cores/numProcess parameter that actually has one to feed.

SEARCH REUSE / caching architecture (TSLS_MDE_GMN_REUSE_BEST_PARAMS,
default off): ported unchanged from simple-mde-gmn.py -- see that file's own
module docstring's SEARCH REUSE section for the full rationale. Every
genuinely expensive stage (MDE forward-selection, network construction,
theta search, the generate-start sweep) is deterministic given the same
inputs, so a fingerprint match against a past job's cache skips that stage
entirely. TSLS_MDE_GMN_PREV_RUNS_DIR/TSLS_MDE_GMN_RUN_CACHE_DIR point at a
durable bucket-storage pool (see tsls-mde-gmn.sbatch's CACHE_ROOT/
RUN_CACHE_DIR comment), not RESULTS_DIR -- every save_json_to_cache()/
save_pickle_to_cache() call writes to RUN_CACHE_DIR *immediately*, right
after its stage finishes, so a stage's result survives even if this job
later crashes or times out.
'''

import ast
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import types
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from configparser import ConfigParser
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict

import gmn
from gmn.Common import FunctionType
from apps.CreateNetwork import CreateNetwork
# NOTE: dimx (dx.MDE/dx.Evaluate), apps.InteractionMatrix, and pyEDM's
# Simplex/CCM/ComputeError/EmbedDimension/PredictInterval are not imported
# directly -- this pipeline uses cleaned_helper_functions.py's
# proper_mde/proper_ccm/proper_simplex/proper_embed_dimension/
# proper_predict_interval throughout instead (see this module's docstring
# and load_config()'s METRIC docstring for why). Embed remains for the
# same reason it always did (no TSLS-aware equivalent needed -- gmn's own
# node embedding never routes through _score()). SMap is imported directly
# (R20, docs/review-actions-v1.md): R10 found no native S-map in
# cleaned_helper_functions.py, but confirmed the compiled pyEDM.SMap gmn's
# own Forecast()/Generate() already calls at runtime (unaffected by this)
# is directly reusable for retune_theta_for_all_nodes()'s own theta
# *selection* sweep too -- see proper_predict_nonlinear()'s own docstring.
# PredictNonlinear stays imported only as the still-unused stdlib default
# this module never calls anymore (retune_theta_for_all_nodes() now calls
# proper_predict_nonlinear() instead) -- kept importable for
# tests/test_r20_predict_nonlinear.py's own equivalence check against it.
from pyEDM import Embed, PredictNonlinear, SMap

# R19 (docs/review-actions-v1.md): run_mde_evaluate()'s PCA/diffusion-map
# baselines. Not a dimx dependency -- these are the same underlying
# libraries dimx.Evaluate itself wraps (confirmed by reading
# dimx/Evaluate.py in full), imported directly here since this pipeline
# has no dimx dependency at all to route through.
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pydiffmap import diffusion_map as dmap

# cleaned_helper_functions.py isn't installed as a package -- the sbatch
# script copies it into this script's own directory (same convention as
# scripts/src/tsls_example.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaned_helper_functions as tsls  # noqa: E402 -- import after sys.path edit above


# =============================================================================
# Generalized per-driver-delay embedding -- ported UNCHANGED from
# simple-mde-gmn.py. Pure graph/pandas mechanics (networkx MultiDiGraph
# edges' own 'delay' attribute, gmn.Node's own data/Parameters), with no
# dependency on what the embedded columns represent -- applies identically
# whether those columns are TSLS-converted or not. Independently fixes both
# bugs this pipeline's own previous, narrower gmn.Node.Forecast patch
# addressed (see _generalized_node_forecast()'s own docstring for the
# specifics), so it's a straight replacement, not a merge.
# =============================================================================

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
        spec += [(u, data['delay']) for u, v, data in graph.in_edges(node_name, data=True)]
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


gmn.Node.__init__ = _generalized_node_init


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


gmn.Node.Forecast = _generalized_node_forecast


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


gmn.Node.Generate = _generalized_node_generate


# =============================================================================
# Configuration -- every value overridable via TSLS_MDE_GMN_* environment
# variables; see scripts/batch/tsls-mde-gmn.sbatch's MODIFIABLE PARAMETERS
# block for the canonical list with descriptions.
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
    # NOTE: as with simple-mde-gmn.py, the cluster deployment expects data
    # files copied flat into DATA_DIR by the sbatch script -- no nested
    # extraction-folder name.
    cfg['DATA_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('TSLS_MDE_GMN_DATA_FILE', 'J1620210605_epoch_8_freq_40_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- Overall pipeline shape ---------------------------------------
    cfg['MODE'] = os.environ.get('TSLS_MDE_GMN_MODE', 'MDE-GMN')  # 'MDE-GMN' | 'GMN'
    if cfg['MODE'] not in ('MDE-GMN', 'GMN'):
        raise ValueError(f"Unknown TSLS_MDE_GMN_MODE: {cfg['MODE']!r}")
    cfg['GMN_MODE'] = os.environ.get('TSLS_MDE_GMN_GMN_MODE', 'combined')  # 'combined' | 'separate'

    cfg['TARGETS'] = _env_list('TSLS_MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
    cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None

    # Three-way temporal split of the full (TSLS-converted, burn-in-trimmed)
    # dataset -- ported from simple-mde-gmn.py's own load_config() docstring,
    # resolved by main() once the trimmed row count is known:
    #   [0, A)      MDE library    -- LIB_RATIO fraction of the trimmed dataset
    #   [A, B)      MDE validation -- proper_mde's own held-out `pred` window
    #               (CCM/embedDimRhoMin/crossMapRhoMin gates, both nested-MDE
    #               stages) -- MDE_VALIDATION_RATIO fraction of the
    #               *remainder* after A
    #   [B, end)    Generate/Forecast validation -- generate_start floats
    #               only within this segment (GENERATE_START_MIN/MAX/STEP_
    #               OFFSET below), for GENERATE_LENGTH steps
    # This exists so network construction (both nested-MDE stages) and the
    # reported Generate()/Forecast() fidelity never touch the same rows --
    # see run_mde_for_target()'s docstring for the mechanics and main()'s
    # window-resolution block for the actual arithmetic. Unlike
    # simple-mde-gmn.py, computed against the row count AFTER TSLS
    # conversion + burn-in trim (main()), not the raw file's row count --
    # see this module's own docstring.
    cfg['LIB_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_LIB_RATIO', 0.7))
    cfg['MDE_VALIDATION_RATIO'] = float(os.environ.get('TSLS_MDE_GMN_MDE_VALIDATION_RATIO', 0.5))
    if not (0.0 < cfg['MDE_VALIDATION_RATIO'] < 1.0):
        raise ValueError(
            f"TSLS_MDE_GMN_MDE_VALIDATION_RATIO must be in (0, 1): {cfg['MDE_VALIDATION_RATIO']!r}"
        )

    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH
    # -- see simple-mde-gmn.py's own load_config() docstring for the full
    # rationale (identical here). 0 (default) means "use the full held-out
    # window"; a larger value is clamped down to it in main().
    cfg['GENERATE_LENGTH'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_LENGTH', 0))

    # Theiler-window / neighbor-exclusion radius for gmn's own recursive
    # Generate()/Forecast() nearest-neighbor search (SMap), passed through
    # as Parameters.exclusionRadius by build_gmn_config() -- a plain static
    # knob (confirmed against build_gmn_config()'s own docstring and every
    # consumption site in simple-mde-gmn.py), NOT the dynamic/auto
    # mechanism TSLS_MDE_GMN_EXCLUSION_RADIUS below uses -- these are two
    # independent EDM contexts (MDE's forward selection vs. GMN's own
    # Generate()/Forecast()) with their own separate embeddings, so there's
    # no single shared "auto" value that would make sense for both.
    # Deliberately not TSLS_MDE_GMN_-prefixed, matching simple-mde-gmn.py's
    # own GMN_EXCLUSION_RADIUS naming (kept visually distinct from the
    # unrelated MDE-side parameter).
    cfg['GMN_EXCLUSION_RADIUS'] = int(os.environ.get('GMN_EXCLUSION_RADIUS', 0))

    # Plotted x-axis range for the final Generate()-vs-Forecast()-vs-actual
    # diagnostic figure -- see run_forecast_diagnostic()'s plot_range_mode
    # docstring. Ported unchanged from simple-mde-gmn.py.
    cfg['FORECAST_PLOT_RANGE'] = os.environ.get('TSLS_MDE_GMN_FORECAST_PLOT_RANGE', 'full').strip().lower()
    if cfg['FORECAST_PLOT_RANGE'] not in ('full', 'windowed'):
        raise ValueError(
            f"TSLS_MDE_GMN_FORECAST_PLOT_RANGE must be 'full' or 'windowed', "
            f"got {cfg['FORECAST_PLOT_RANGE']!r}"
        )

    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
    # as offsets (rows) from GENERATE_VALIDATION_START -- see
    # simple-mde-gmn.py's own load_config() docstring for the full rationale
    # (identical here). Default MIN=0, MAX=1 tries only offset 0.
    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
    cfg['GENERATE_START_STEP'] = int(os.environ.get('TSLS_MDE_GMN_GENERATE_START_STEP', 50))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    # MDE_D counts distinct *real* observables, not delay-expanded candidate
    # columns -- reused as the per-round cap for both nested-MDE stages
    # (Stage 1: real-observable-only, so the multiplier is always 1;
    # Stage 2: lagged-only, multiplied by however many lag values
    # LAG_MODE/LAG_DENSITY/LAG_STEP below resolve to) -- see
    # effective_mde_dimension()'s docstring. Ported unchanged from
    # simple-mde-gmn.py.
    cfg['MDE_D'] = int(os.environ.get('TSLS_MDE_GMN_D', 20))
    # Prediction interval every proper_mde forward-selection call (both the
    # top-level per-target search and every nested-MDE per-node search)
    # ranks/validates every candidate driver against -- ported from
    # simple-mde-gmn.py's own TP parameter (proper_mde accepts Tp as a real
    # argument, confirmed), replacing this pipeline's own previous hardcoded
    # Tp=1.
    cfg['TP'] = int(os.environ.get('TSLS_MDE_GMN_TP', 1))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # CCM pre-filter (see ccm_prefilter_observables()'s docstring): a cheap
    # CCM(observable -> target) skill check run once per target group,
    # right before either nested-MDE stage, to shrink REGION_CHANNEL_CAP's
    # candidate pool down to only observables that cross-map at least one
    # target above this threshold -- cuts cost on both Stage 1 (real
    # observables) and Stage 2 (their lags, since Stage 2 draws from the
    # same filtered pool) for free. 0 disables the filter entirely (pool
    # passed through unchanged); >0 enables it as the rho cutoff.
    cfg['CCM_PREFILTER_RHO_MIN'] = float(os.environ.get('TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN', 0.2))
    # Static embedding dimension for the CCM pre-filter's own CCM calls --
    # its OWN parameter, deliberately NOT a reuse of MULTI_MDE_E below: this
    # pipeline's data is TSLS-converted from a 40 Hz recording vs.
    # simple-mde-gmn.py's 1 Hz input (40x more rows per unit of real time),
    # and both pipelines fix tau=-1 (one sample step), so one embedding
    # step here covers only 1/40th the real time one step covers in
    # simple-mde-gmn.py -- matching that in raw sample-count terms would
    # need an impractical E. See simple-mde-gmn.sbatch's own
    # CCM_PREFILTER_E comment for the full simple-mde-gmn-side reasoning.
    # 7 here is a deliberately modest compromise: the middle value of this
    # pipeline's own already-team-vetted E_RANGE search below (E_MIN=3/
    # E_MAX=13/E_STEP=4 -> 3,7,11), not an untested extrapolation, and
    # still leaves large margin against the CCM_WINDOW_ROWS/LIBSIZE_RATIOS
    # -driven smallest-libSize-vs-E feasibility ceiling.
    cfg['CCM_PREFILTER_E'] = int(os.environ.get('TSLS_MDE_GMN_CCM_PREFILTER_E', 7))
    # Library sizes for the CCM pre-filter, as fractions of the same
    # CCM_WINDOW_ROWS-bounded window (see that parameter's own docstring
    # below) every other CCM call in this pipeline already uses, rather
    # than absolute row counts -- that window's actual row count depends on
    # CCM_WINDOW_ROWS/the data itself. Converted to actual libSizes (deduped,
    # floored at 10 rows, same floor convention build_interaction_matrix()'s
    # own CCM_LIB_SIZE_PCTS conversion uses) once the window is known -- see
    # ccm_prefilter_observables()'s own docstring.
    cfg['LIBSIZE_RATIOS'] = _env_float_list(
        'TSLS_MDE_GMN_LIBSIZE_RATIOS', (0.05, 0.10, 0.20, 0.40, 0.6, 0.8, 0.9),
    )
    # Only consulted when GMN_MODE='combined' (see target_groups_for_mode()
    # and main()'s own MDE block): in that mode, the depth-0 (top-level,
    # per-TARGETS) MDE selection runs once via
    # cleaned_helper_functions.multi_proper_mde -- forward-selecting drivers
    # that jointly predict every TARGETS column at once -- instead of the
    # GMN_MODE='separate' behaviour of one independent
    # cleaned_helper_functions.proper_mde call per target.
    #
    # UPDATED: multi_proper_mde now DOES have a per-candidate auto-select
    # path (multi_proper_embed_dimension() -- a joint rho-vs-E sweep scored
    # via _simplex_multi_metric, the same per-target-then-aggregate
    # primitive the ranking stage already used), mirroring single-target
    # proper_mde's own E<=0 sentinel convention. main()'s own depth-0 seed
    # call always passes E=0 (auto-select unconditionally -- same
    # "hardcoded at the one call site" precedent as the single-target path,
    # see run_mde_for_target()'s own max_e docstring) and reuses
    # MULTI_MDE_E as the sweep's own ceiling (max_e) rather than a fixed
    # value -- the same "reuse an already-exposed budget rather than
    # introduce a second one" convention MDE_D already established for the
    # single-target path. MULTI_MDE_TARGET_AGGREGATE picks how per-target
    # scores are combined at all three stages now (ranking, the E-select
    # sweep, and the CCM gate): 'mean' avoids penalizing a driver that's
    # strong for one target and weak for another; 'min' only accepts
    # drivers that are simultaneously good for every target.
    cfg['MULTI_MDE_E'] = int(os.environ.get('TSLS_MDE_GMN_MULTI_MDE_E', 3))
    cfg['MULTI_MDE_TARGET_AGGREGATE'] = os.environ.get('TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE', 'mean')
    if cfg['MULTI_MDE_TARGET_AGGREGATE'] not in ('mean', 'min'):
        raise ValueError(
            f"TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE must be 'mean' or 'min': "
            f"{cfg['MULTI_MDE_TARGET_AGGREGATE']!r}"
        )
    # Theiler-window / neighbor-exclusion radius for every proper_mde/
    # proper_ccm/proper_simplex/proper_embed_dimension call this pipeline
    # makes for MDE forward-selection. UNLIKE simple-mde-gmn.py, this is a
    # PLAIN STATIC INTEGER ONLY -- no 'auto' dynamic mode. simple-mde-gmn.py's
    # 'auto' mode recomputes exclusionRadius fresh at every one of dimx.Run's
    # own internal CrossMapColumns/EmbedDimension/CCM calls (a staged
    # sequence of internal calls this pipeline's own forward-selection makes
    # as it grows a candidate embedding one column at a time); proper_mde
    # and friends are each a single opaque call taking one exclusionRadius
    # value for their own whole internal search, with no staged internal
    # calls to intercept the same way. See this module's own docstring for
    # the full explanation.
    cfg['EXCLUSION_RADIUS_VALUE'] = int(os.environ.get('TSLS_MDE_GMN_EXCLUSION_RADIUS', 0))
    # Nested-MDE network construction (MODE='MDE-GMN' only). Reuses MDE_D as
    # the per-node driver cap N at every nesting level (deliberately not a
    # separate parameter). NESTED_M is the number of *additional* nested-MDE
    # rounds beyond each target's own initial selection -- total graph depth
    # from each target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for
    # "uncapped". R13 (docs/review-actions-v1.md): confirmed this is a
    # WHOLE-NETWORK budget, shared jointly by Stage 1 and Stage 2 -- a
    # node's depth is its distance from the target regardless of which
    # stage discovered it (build_nested_mde_network()'s own `node_depth`),
    # not two independent per-stage counters. This was already true before
    # R13 (and before R12); R13 only confirmed and documented it, no
    # arithmetic changed. Ported unchanged from simple-mde-gmn.py -- see
    # build_nested_mde_network()'s docstring for the full algorithm.
    cfg['NESTED_M'] = int(os.environ.get('TSLS_MDE_GMN_NESTED_M', 1))
    # First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted
    # and offered as MDE candidates (or, when MODE='GMN', used directly as
    # the interaction-matrix's full node set). 999 is effectively uncapped
    # (matches notebooks/tsls-mde-gmn.ipynb's own full-channel-set decision).
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('TSLS_MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- Two-stage nested-MDE network construction (MODE='MDE-GMN' only) --
    # Ported from simple-mde-gmn.py -- see build_nested_mde_network() for the
    # full serial two-stage algorithm (Stage 1 real-observable-only BFS run
    # to completion, then Stage 2 lagged-embedding attachment, itself
    # recursive), and that file's own load_config() docstrings for STAGE2_ALLOW_REAL_
    # OBSERVABLES/DELAY_SOURCE/LAG_MODE/LAG_DENSITY/LAG_STEP/DELAY_SET's full
    # rationale (identical here -- this is all pure column-name/DataFrame
    # machinery with no TSLS-specific behavior). R15 (docs/review-actions-v1.md):
    # default flipped to True -- confirmed end-to-end (not inferred) that Stage 2's
    # own real-observable picks genuinely recurse into further Stage 2 rounds
    # (tests/test_r12_stage_serialization.py), bounded by the R13 shared depth
    # budget (tests/test_r13_shared_depth_budget.py), so the step 1 condition for
    # flipping the default holds.
    cfg['STAGE2_ALLOW_REAL_OBSERVABLES'] = _env_bool('TSLS_MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES', True)

    # R14 (docs/review-actions-v1.md): a zero-driver leaf is partitioned out
    # of the simulated network as an "external input" (partition_network())
    # rather than getting the old unconditional self-history embedding.
    # EXTERNAL_INPUT_MODE picks what Generate()/Forecast() actually do with
    # such a node -- see _generalized_node_generate()'s own docstring for
    # the full rationale (this was an explicit ask-before-implementing
    # decision, not a default guessed at):
    #   'real_data' (default): the node is genuinely exogenous -- its
    #     Generate() call short-circuits to the real recorded value at each
    #     step (self.Network.data, already loaded in full), never running
    #     an SMap self-regression. The INTERIOR network still free-runs
    #     autonomously; only external inputs are fed from real data. This
    #     is what "external input" means in the standard control-theory
    #     sense, and is the reading docs/tsls-mde-gmn-pipeline.pdf's own
    #     future doc-rewrite pass should adopt for its Sec 1.1 claim.
    #   'self_simulate': keeps today's exact pre-R14 behavior for these
    #     nodes (unconditional self_delay_set embedding, genuinely self-
    #     predicted every step, never touching real data past the initial
    #     seed) -- R14's partition becomes reporting/visualization-only in
    #     this mode, with no behavior change from before R14.
    # Either way, a TARGET that itself selects zero drivers is NEVER
    # classified as an external input regardless of this setting --
    # partition_network() raises instead (feeding a target's own real
    # values back in as its "simulation" and then scoring fidelity against
    # those same real values would be circular, producing a spuriously
    # perfect score for a network that is actually a failure).
    cfg['EXTERNAL_INPUT_MODE'] = os.environ.get('TSLS_MDE_GMN_EXTERNAL_INPUT_MODE', 'real_data')
    if cfg['EXTERNAL_INPUT_MODE'] not in ('real_data', 'self_simulate'):
        raise ValueError(
            f"TSLS_MDE_GMN_EXTERNAL_INPUT_MODE must be 'real_data' or 'self_simulate': "
            f"{cfg['EXTERNAL_INPUT_MODE']!r}"
        )

    cfg['DELAY_SOURCE'] = os.environ.get('TSLS_MDE_GMN_DELAY_SOURCE', 'lag_mode').strip().lower()
    if cfg['DELAY_SOURCE'] not in ('explicit', 'lag_mode'):
        raise ValueError(f"TSLS_MDE_GMN_DELAY_SOURCE must be 'explicit' or 'lag_mode': {cfg['DELAY_SOURCE']!r}")

    cfg['LAG_MODE'] = os.environ.get('TSLS_MDE_GMN_LAG_MODE', 'generation_length_adjusted').strip().lower()
    if cfg['LAG_MODE'] not in ('generation_length_adjusted', 'every_lag_possible'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_MODE must be 'generation_length_adjusted' or "
                          f"'every_lag_possible': {cfg['LAG_MODE']!r}")

    cfg['LAG_DENSITY'] = os.environ.get('TSLS_MDE_GMN_LAG_DENSITY', 'strided').strip().lower()
    if cfg['LAG_DENSITY'] not in ('strided', 'dense'):
        raise ValueError(f"TSLS_MDE_GMN_LAG_DENSITY must be 'strided' or 'dense': {cfg['LAG_DENSITY']!r}")
    cfg['LAG_STEP'] = int(os.environ.get('TSLS_MDE_GMN_LAG_STEP', 1))
    if cfg['LAG_STEP'] < 1:
        raise ValueError(f"TSLS_MDE_GMN_LAG_STEP must be >= 1: {cfg['LAG_STEP']!r}")

    cfg['DELAY_SET'] = _env_int_list('TSLS_MDE_GMN_DELAY_SET', [0, -1, -3, -5, -10])
    if any(d > 0 for d in cfg['DELAY_SET']):
        raise ValueError(f"TSLS_MDE_GMN_DELAY_SET must be <= 0 (no future data): {cfg['DELAY_SET']!r}")

    # ---- E/tau search range (MODE='GMN' only) --------------------------
    # Used exclusively by build_network()'s mode == 'GMN' branch (the
    # proper_ccm-based interaction-matrix E/tau grid search, via
    # target_E_tau_rho_grid()) -- MODE='MDE-GMN' no longer has a
    # network-wide E/tau at all. Also feeds run_theta_search()'s own theta
    # retuning range regardless of mode (THETA_RANGE below).
    cfg['E_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_E_MIN', 3)),
        int(os.environ.get('TSLS_MDE_GMN_E_MAX', 13)),
        int(os.environ.get('TSLS_MDE_GMN_E_STEP', 4)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('TSLS_MDE_GMN_TAU_MIN', -9)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('TSLS_MDE_GMN_TAU_STEP', 4)),
    )
    # Theta (SMap's nonlinearity/localization parameter) search range, used
    # by run_theta_search() for both MODE='MDE-GMN' and MODE='GMN' -- ported
    # unchanged from simple-mde-gmn.py. Replaces this pipeline's own
    # previous SMAP_SEARCH_MODE='coarse_retune'/'joint' combined (E, tau,
    # theta) grid search entirely (see run_theta_search()'s own docstring):
    # E and every driver's delay now come from the network itself (MODE=
    # 'MDE-GMN': nested-MDE; MODE='GMN': the E_RANGE/TAU_RANGE grid search
    # above), leaving theta as the only thing still searched per node.
    cfg['THETA_RANGE'] = _env_float_list(
        'TSLS_MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('TSLS_MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('TSLS_MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS', True)
    # RUN_MDE_EVALUATE: gates ONLY run_mde_evaluate() (R19's PCA/diffusion-
    # map baseline comparison) -- deliberately separate from
    # RUN_MDE_DIAGNOSTICS above, which still gates run_tp_vs_rho_sweep()/
    # run_univariate_e_tau_grid()/plot_3d_mde_projection() (unaffected;
    # verified under R7, not implicated in the finding below). Default
    # False: docs/review-findings.md's "diffusion-map crash and hang"
    # finding confirmed pydiffmap's own diffusion-map fit, on real TSLS
    # data, can both raise (ValueError: Input X contains infinity...,
    # reproduced in 1.8s at an 800-row scale) and hang indefinitely
    # (reproduced past 74 CPU-minutes with no completion, at a real
    # ~31,000-row scale) -- the mechanism for the hang specifically is NOT
    # diagnosed (see that finding's own "flag for the documentation pass"
    # note). run_mde_evaluate() itself still has its own exception guard
    # AND a wall-clock bound around each baseline independently (see that
    # function's own docstring) for whoever turns this back on, but the
    # toggle is what actually protects a run while it's off -- a guard is
    # only as good as remembering every place that could need one, so
    # skipping this by default until the mechanism is understood is the
    # deliberate choice here, not a workaround. R19 is optional tooling,
    # not part of the causal claim (docs/tsls-mde-gmn-pipeline.tex's own
    # sec:diagnostics linknote) -- skipping it is a legitimate
    # configuration. Logged once per run when off, so its absence is
    # never silent (see docs/review-findings.md's own §18.6 cross-
    # reference for why a swallowed diagnostic staying invisible is
    # exactly the failure mode to avoid repeating).
    cfg['RUN_MDE_EVALUATE'] = _env_bool('TSLS_MDE_GMN_RUN_MDE_EVALUATE', False)

    # ---- Parallelism ----------------------------------------------------
    # NOTE: no TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES here -- unlike
    # simple-mde-gmn.py's own MODE='GMN' branch (still shells out to
    # apps.InteractionMatrix, whose own outer pool that knob sizes), this
    # pipeline's MODE='GMN' path uses proper_ccm (build_interaction_matrix()
    # below), which -- like proper_mde -- has no multiprocessing pool of its
    # own at all (no Pool usage anywhere in cleaned_helper_functions.py), so
    # there's nothing for a knob like this to size. See this module's own
    # docstring.
    cfg['CORES'] = int(os.environ.get('TSLS_MDE_GMN_CORES', os.cpu_count() or 1))

    # ---- TSLS-aware scoring (cleaned_helper_functions.py's proper_mde/
    # proper_ccm/proper_simplex/adaptive_event_metrics_from_tsls, replacing
    # dimx.MDE / apps.InteractionMatrix / pyEDM.EmbedDimension / plain
    # ComputeError rho everywhere in this pipeline that scores a TSLS-
    # converted column) -----------------------------------------------------
    # 'correlation' (plain Pearson rho, pyEDM's own default) | 'mae' |
    # 'f1' | 'gamma' (Kistler coincidence factor) -- see
    # cleaned_helper_functions.py's _score()/adaptive_event_metrics_from_tsls
    # docstrings. 'f1'/'gamma' score TSLS series by decoding both true and
    # predicted series back to discrete reset/spike events (via a near-zero
    # threshold calibrated from the true series itself) and matching them
    # within an adaptive per-event tolerance window -- this is what actually
    # answers "did we get the spike timing right" instead of Pearson
    # correlation over the whole sawtooth, which is dominated by the smooth
    # (uninformative) ramp segments between spikes.
    cfg['METRIC'] = os.environ.get('TSLS_MDE_GMN_METRIC', 'f1')
    # TARGETS (head_position_x/y) are continuous behavioral traces, not
    # TSLS-encoded spike series -- they never "reset" near zero at discrete
    # events, so 'f1'/'gamma' event-decoding is meaningless for them. Always
    # 'correlation' (or 'mae') for anything scored against a TARGETS column;
    # METRIC above is only for scoring predictions of the TSLS-converted
    # neuron columns themselves.
    cfg['TARGET_METRIC'] = os.environ.get('TSLS_MDE_GMN_TARGET_METRIC', 'correlation')
    cfg['GAMMA_DELTA'] = int(os.environ.get('TSLS_MDE_GMN_GAMMA_DELTA', 2))
    # MODE='GMN' only (build_interaction_matrix() below) -- sweeps library
    # sizes as PERCENTAGES of however many rows it's handed. CCM_WINDOW_ROWS
    # bounds that row count -- CRITICAL for calibration, not just a speed
    # knob: too large a window makes every tested library size land in the
    # already-converged tail of the real curve, with the gate then
    # (correctly) reporting "no rise" even though the underlying
    # relationship is real. Fixed two past production failures this way
    # (apps.InteractionMatrix reporting ~all-zero interactions; proper_mde
    # rejecting every candidate for both targets) -- recalibrated against
    # real small-libSize data from this dataset. Keep CCM_WINDOW_ROWS's
    # smallest resulting libSize >= (max E in E_RANGE)+2.
    #
    # MODE='MDE-GMN' (run_mde_for_target()/run_multi_mde_for_targets(),
    # the pipeline's own default) no longer uses this knob at all -- see
    # TARGET_RANKING_WINDOW_ROWS/MIN_PRED_EVENTS/MAX_WINDOW_ROWS/
    # CCM_LIB_SIZES_ROWS below, and docs/review-findings.md's own
    # window-sizing section for why: a single fixed window/percentage-based
    # library grid made f1-scored ranking degenerate for real neuron
    # targets whose own reset events are too sparse within any window small
    # enough to keep the CCM gate's own convergence curve meaningful.
    cfg['CCM_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_CCM_WINDOW_ROWS', 800))
    # R17 (docs/review-actions-v1.md), gated on R9's own calibration: R9's
    # window sweep measured, at this pipeline's CCM_WINDOW_ROWS=800 default
    # (whose own smallest percentage below, 2%, is 16 rows), that 35% of a
    # 54-pair stratified sample were still rising at 40 rows (the 5%
    # point) -- a real but NOT majority fraction (R9's own decision rule:
    # "most sampled pairs" would mean adopting the reviewer's coarser
    # [5,20,40,60,80] grid outright, which drops everything below 40 rows
    # entirely; 35% doesn't clear that bar). So: coarsen only the upper
    # end, keeping the small (16/24/40/64-row) percentages that still
    # carry real rising-curve signal for a meaningful minority of pairs,
    # and only dropping percentages from the already-converged tail
    # (12%, 30% removed; 20%/50% kept) -- see docs/review-findings.md's
    # R17 section for the measured compute/selection effect on real data.
    cfg['CCM_LIB_SIZE_PCTS'] = tuple(
        _env_float_list('TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS', (2, 3, 5, 8, 20, 50))
    )
    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_MDE_GMN_CCM_SAMPLE', 20))
    cfg['CCM_SLOPE'] = float(os.environ.get('TSLS_MDE_GMN_CCM_SLOPE', 0.01))

    # Window-sizing fix (docs/review-findings.md) -- MODE='MDE-GMN' only.
    # Replaces CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS for run_mde_for_target()/
    # run_multi_mde_for_targets()'s own ranking-stage/E-select window and
    # the CCM gate's own library sweep, decoupled from each other (see
    # _resolve_target_ranking_window()'s and proper_mde()'s own `libSizes`
    # docstrings for the full reasoning -- a single fixed window/percentage
    # grid made f1-scored ranking degenerate for real sparse-firing
    # neurons).
    #
    # TARGET_RANKING_WINDOW_ROWS: the depth-0 seed's own fixed
    # ranking/E-select window (behavioral targets have no reset-event
    # concept, so there's nothing to extend against) -- arbitrary, set to
    # 1000 for now; larger than the old CCM_WINDOW_ROWS=800 default, no
    # other rationale yet.
    cfg['TARGET_RANKING_WINDOW_ROWS'] = int(
        os.environ.get('TSLS_MDE_GMN_TARGET_RANKING_WINDOW_ROWS', 1000))
    # MIN_PRED_EVENTS: for every Stage 1/Stage 2 (neuron) node, its own
    # ranking window is extended -- never past the leakage boundary B --
    # until its own PRED portion (the last (1 - lib_ratio) fraction of ROWS
    # of the window, the part `f1`-scoring actually touches) contains at
    # least this many of that neuron's real reset events, counted strictly
    # within [0, B). A plain fixed integer, not a fraction of anything
    # neuron-specific (see docs/review-findings.md for why an earlier
    # "half of this neuron's own total events" design was abandoned: it
    # made a busy neuron face a stricter test than a quiet one for no
    # stated reason, and -- checked against the whole window rather than
    # its pred portion -- was found on real data to pass while the pred
    # portion itself still had too few events to score anything on).
    #
    # Calibrated (not guessed) via r9_ccm_window_e_calibration.py's own
    # MIN_PRED_EVENTS section: f1 between a real TSLS slice and a lightly
    # perturbed copy of itself, >=200 draws per point, stratified by
    # firing-rate tercile. Result: a smooth ~1/sqrt(N) std decay with NO
    # natural elbow (5->0.146, 10->0.104, 20->0.070, 40->0.050, 80->0.036,
    # 160->0.025, pooled) -- so 20 is a threshold choice justified by the
    # cost/reachability tradeoff below, not a discovered stabilisation
    # point. At 20, std~0.07 is still ~3x smaller than CROSS_MAP_RHO_MIN's
    # own 0.2 default, and real data (REGION_CHANNEL_CAP=15, this session)
    # showed only 3/45 channels unreachable within the leakage-safe range
    # (median 3,593 rows needed) -- 40 and 80 buy modestly less noise at a
    # steep cost in both unreachable channels (8/45, 20/45) and window
    # length (median ~10,400 rows either way).
    cfg['MIN_PRED_EVENTS'] = int(os.environ.get('TSLS_MDE_GMN_MIN_PRED_EVENTS', 20))
    # MAX_WINDOW_ROWS: a plain compute ceiling on the search above -- NOT
    # calibrated (unlike MIN_PRED_EVENTS): if a node can't reach
    # MIN_PRED_EVENTS within min(MAX_WINDOW_ROWS, B), it's skipped (logged,
    # with the shortfall) rather than scored on too few events.
    # MIN_PRED_EVENTS is the statistical floor; this is the compute
    # ceiling -- the two express one thing each. 30000 is comfortably
    # above the largest real row count MIN_PRED_EVENTS=20 was observed to
    # need (26,748) and safely below this dataset's own leakage boundary
    # (~38,067) -- a backstop, not a tuned value.
    cfg['MAX_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_MAX_WINDOW_ROWS', 30000))
    # CCM_LIB_SIZES_ROWS: explicit, absolute row counts (never a percentage
    # of the -- now node-specific, potentially much larger -- ranking
    # window above) for the CCM gate's own convergence-curve library sweep.
    # Starts from R9/R17's own empirically-calibrated percentage grid
    # (2,3,5,8,20,50% of the old 800-row CCM_WINDOW_ROWS = 16,24,40,64,160,
    # 400 rows), plus one additional point (800) for better plateau
    # visibility now that the surrounding frame can be much larger --
    # re-validated at both an 800-row and an 8000-row frame on real data
    # (docs/review-findings.md): the same absolute values showed the same
    # qualitative "rising, then plateauing" shape at both scales, since
    # that shape is a property of the absolute library size, not its ratio
    # to the frame.
    cfg['CCM_LIB_SIZES_ROWS'] = tuple(
        _env_int_list('TSLS_MDE_GMN_CCM_LIB_SIZES_ROWS', (16, 24, 40, 64, 160, 400, 800))
    )

    # ---- Search reuse ---------------------------------------------------
    # See simple-mde-gmn.py's own module docstring's SEARCH REUSE section
    # for the full rationale; ported unchanged. Short version: reuse a past
    # job's E/tau/theta search result instead of redoing it, when settings
    # match exactly.
    cfg['REUSE_BEST_PARAMS'] = _env_bool('TSLS_MDE_GMN_REUSE_BEST_PARAMS', False)
    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_PREV_RUNS_DIR', '../../results'))
    cfg['RUN_CACHE_DIR'] = Path(os.environ.get('TSLS_MDE_GMN_RUN_CACHE_DIR', cfg['RESULTS_DIR']))
    cfg['RUN_CACHE_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- GPU (torch) engine ---------------------------------------------
    # GPU_ENABLED gates the whole torch/CUDA path. OFF by default, and when
    # off this pipeline behaves byte-for-byte as it did before the flag
    # existed (fork-based worker pools, pyEDM/scipy engines throughout).
    #
    # Why this flag also controls the multiprocessing start method: CUDA does
    # NOT support fork() -- forking a process that holds a CUDA context gives
    # undefined behaviour in the child, typically surfacing as
    # "RuntimeError: Cannot re-initialize CUDA in forked subprocess". PyTorch's
    # own multiprocessing guidance requires 'spawn' (or 'forkserver') for CUDA
    # in subprocesses. See docs/no-kEDM-gpu-optimization-plan.md section 7a.
    #
    # Spawn is NOT free here, which is exactly why this is conditional rather
    # than a blanket switch: a spawn worker re-imports this module from
    # scratch, measured at ~18s on Deigo (pandas/numpy/matplotlib/pyEDM/gmn
    # all get re-imported), and the outer pools below are recreated once per
    # BFS wave -- so the cost is paid per worker per wave. CPU-only runs get
    # no benefit from it, so they keep fork.
    #
    # Structural preconditions for spawn are already satisfied by this module
    # and were verified before adding this (see the plan doc): the gmn
    # monkeypatching is applied at module scope (re-executes correctly on
    # re-import in a spawn worker), there is a proper `if __name__ ==
    # "__main__"` guard, and there is no runtime-mutated module-level global
    # state a spawn worker would silently miss.
    cfg['GPU_ENABLED'] = _env_bool('TSLS_MDE_GMN_GPU_ENABLED', False)
    cfg['MP_START_METHOD'] = 'spawn' if cfg['GPU_ENABLED'] else 'fork'

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg


# =============================================================================
# TSLS conversion + burn-in trim (kept unchanged -- TSLS-specific front end;
# mirrors notebooks/tsls-mde-gmn.ipynb cells 3-4; see that notebook's intro
# cell for the full rationale)
# =============================================================================

def region_neuron_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]


def raw_neuron_columns(df, cap):
    return (
        region_neuron_columns(df, 'CA1_', cap)
        + region_neuron_columns(df, 'mPFC_', cap)
        + region_neuron_columns(df, 'OFC_', cap)
    )


def convert_to_tsls(df_raw, neuron_columns, targets, results_dir):
    '''Spike counts -> TSLS for every candidate channel; behavioral TARGETS
    columns pass through untouched (they are already continuous, not spike
    counts). Returns the full (not yet burn-in-trimmed) TSLS DataFrame and
    the list of its _tsls-suffixed observable columns.'''
    time_grid = df_raw['time_bin_center'].to_numpy()

    data = {'time_bin_center': time_grid}
    for t in targets:
        data[t] = df_raw[t].to_numpy()

    for i, neuron in enumerate(neuron_columns):
        spike_times = df_raw.loc[df_raw[neuron] > 0, 'time_bin_center'].to_numpy()
        data[f'{neuron}_tsls'] = tsls.compute_time_since_previous_spike(spike_times, time_grid)
        if (i + 1) % 100 == 0 or i + 1 == len(neuron_columns):
            print(f'  TSLS-converted {i + 1}/{len(neuron_columns)} channels', flush=True)

    tsls_df_full = pd.DataFrame(data)
    tsls_df_full.to_csv(results_dir / 'tsls_data_full.csv', index=False)
    observable_columns = [f'{n}_tsls' for n in neuron_columns]
    print(f'-> tsls_data_full.csv  shape={tsls_df_full.shape}', flush=True)
    return tsls_df_full, observable_columns


def burn_in_trim(tsls_df_full, observable_columns, results_dir):
    '''Drop leading rows until every observable_columns entry is non-NaN
    (see module docstring for why this is a correctness requirement for the
    gmn.Node Generate()/Forecast() handoff, not just cleanliness). Raises
    if any candidate channel never fires at all in this window.'''
    first_valid_idx = tsls_df_full[observable_columns].notna().idxmax()
    never_valid = [c for c in observable_columns if not tsls_df_full[c].notna().any()]
    if never_valid:
        raise ValueError(
            f'{len(never_valid)} channel(s) never fire in this window, TSLS is '
            f'all-NaN: {never_valid} -- widen the window or drop these channels.'
        )

    burn_in_rows = int(first_valid_idx.max())
    slowest_channel = first_valid_idx.idxmax()
    df = tsls_df_full.iloc[burn_in_rows:].reset_index(drop=True)

    df.to_csv(results_dir / 'tsls_data_trimmed.csv', index=False)
    print(f'Burn-in: dropped first {burn_in_rows}/{len(tsls_df_full)} rows '
          f'({burn_in_rows / len(tsls_df_full):.1%}) -- slowest channel to fire: {slowest_channel}',
          flush=True)
    print(f'-> tsls_data_trimmed.csv  shape={df.shape}', flush=True)
    return df


def _ccm_window(df, lib_ratio, max_rows):
    '''Bound the row count fed into any percentage-based CCM libSize sweep
    to max_rows, preserving lib_ratio's proportion within that smaller
    window. See load_config()'s CCM_WINDOW_ROWS docstring for why this is
    necessary (not optional) for CCM's convergence gate to mean anything at
    this pipeline's row counts.'''
    n = min(len(df), max_rows)
    windowed = df.iloc[:n].reset_index(drop=True)
    lib_end = max(1, int(round(n * lib_ratio)))
    return windowed, lib_end


def _resolve_target_ranking_window(target, full_df, usable_df, lib_ratio,
                                    is_behavioral_target, target_ranking_window_rows,
                                    min_pred_events, max_window_rows):
    '''Window-sizing fix (found while verifying R22, docs/review-actions-v1.md
    -- see docs/review-findings.md's own window-sizing section for the full
    real-data investigation, including the two abandoned designs this
    replaces): resolves the ranking-stage/E-auto-select window for one
    node's own run_mde_for_target() call, decoupled from CCM_WINDOW_ROWS
    (which stays governing the CCM gate's own convergence-curve library
    sizing, now via absolute row counts -- see `ccm_lib_sizes_rows` in
    run_mde_for_target()'s own docstring -- not this function's own frame
    size at all).

    BEHAVIORAL TARGETS (`is_behavioral_target=True`, always true at the
    depth-0 seed): no TSLS reset-event concept applies (the target is a
    genuine continuous trace, correlation-scored) -- window is simply
    `min(target_ranking_window_rows, len(usable_df))`, a fixed constant
    (default 1000), unrelated to firing rate. `min_pred_events` is not
    used, `max_window_rows` is not used, and the returned event count is
    always `None`.

    NEURON TARGETS (`is_behavioral_target=False`, virtually every Stage 1/
    Stage 2 node): grow the window W (in rows); within W, the PRED portion
    is its own last `1 - lib_ratio` fraction of ROWS (the same split
    `_ccm_window()` will later carve out of this exact W for the CCM
    gate -- see that function). Count `target`'s real reset events that
    fall in that row range -- never by events, only ever by rows -- and
    stop at the smallest W where that count is >= `min_pred_events`.
    `min_pred_events` is a plain, fixed integer (not a fraction of
    anything neuron-specific): it answers "how many matched events does
    f1 need before it is signal rather than noise", which is a property
    of the metric, not of the neuron (see docs/review-findings.md for the
    full reasoning behind abandoning the earlier
    fraction-of-a-neuron's-own-total-events design, which made a busy
    neuron face a stricter test than a quiet one for no stated reason, and
    which -- checked against the WHOLE window rather than its pred
    portion -- was later found on real data to pass while the pred
    portion itself still had too few events to score anything on).

    The search never grows W past `min(max_window_rows, len(usable_df))`
    -- `max_window_rows` is a plain compute ceiling (not calibrated; see
    load_config()'s own docstring), `len(usable_df)` is the leakage-safe
    boundary this function's own caller has already trimmed to (see
    run_mde_for_target()'s own LEAKAGE docstring). ALL event counting here
    happens strictly within `usable_df` (i.e. `[0, B)`) -- never using
    `full_df` rows at or past B, since sizing a search parameter from
    held-out data is the same class of leakage the B boundary exists to
    prevent (`full_df` is accepted only for signature symmetry with the
    behavioral branch and is not read in this branch).

    Returns (window_rows, pred_event_count). On success, `pred_event_count`
    is the achieved count in the resolved window's own pred portion
    (>= min_pred_events). On failure (`min_pred_events` unreachable
    anywhere within the cap), returns (None, pred_event_count) where
    pred_event_count is the best (largest) count achieved at ANY window
    size tried, not necessarily at the cap itself -- the pred portion's
    own event count is not guaranteed monotonic in W (both its edges only
    ever move forward as W grows, but an event row can age out of the
    portion's left edge in the same step before any new event row enters
    at the right, so the count can dip locally) -- the caller can and
    should report the shortfall as `min_pred_events - pred_event_count`
    rather than silently scoring the node on too few events.'''
    n_usable = len(usable_df)
    if is_behavioral_target:
        return min(target_ranking_window_rows, n_usable), None

    cap = min(max_window_rows, n_usable)
    if cap < 1:
        return None, 0

    usable_vals = usable_df[target].to_numpy()[:cap]
    near_zero_threshold = tsls.infer_near_zero_threshold_from_true_tsls(
        full_df[target].to_numpy())
    # detect_reset_events_from_tsls classifies each row from only that row
    # and the one before it (np.diff(..., prepend=x[0])), so truncating the
    # input to `cap` rows first cannot change any detection within [0, cap).
    event_indices = tsls.detect_reset_events_from_tsls(
        usable_vals, near_zero_threshold=near_zero_threshold, drop_threshold=0.0)

    # event_indices is sorted, so searchsorted gives the exact cumulative
    # count up to any row bound with no explicit loop. pred_start[i] is the
    # first row of the pred portion for window size w_candidates[i] -- the
    # same `round(w * lib_ratio)` formula _ccm_window() uses for its own
    # lib_end, so the final resolved window's pred portion here is exactly
    # what the CCM gate will later carve out of it too.
    w_candidates = np.arange(1, cap + 1)
    pred_start = np.round(w_candidates * lib_ratio).astype(int)
    lo = np.searchsorted(event_indices, pred_start, side='left')
    hi = np.searchsorted(event_indices, w_candidates, side='left')
    pred_events_up_to_w = hi - lo

    passing = pred_events_up_to_w >= min_pred_events
    if not passing.any():
        # Best count seen at ANY window size tried, not just at the cap --
        # see this function's own docstring for why the two can differ.
        return None, int(pred_events_up_to_w.max())
    first = int(np.argmax(passing))
    return int(w_candidates[first]), int(pred_events_up_to_w[first])


def _preview_window_rows(node, full_df, delay_set, prediction_start, prediction_length, tp,
                          is_behavioral_target, target_ranking_window_rows,
                          min_pred_events, max_window_rows):
    '''Speculative, read-only resolve of one node's own expected ranking-
    window size, called from the progress-monitoring log's own ETA
    estimate (see _format_eta()'s per-window-row mode) BEFORE that node's
    real run_mde_for_target() call -- independent of, and without any
    effect on, the real resolve that call performs later. Node cost scales
    roughly with resolved window size (see _resolve_target_ranking_window()
    itself), so this gives the monitoring log a real cost signal per
    pending node instead of treating every node as equally expensive.

    Mirrors run_mde_for_target()'s own usable_df/lib_ratio derivation
    exactly (see that function's own LEAKAGE docstring), narrowed to just
    `node`'s own column -- cheap relative to the real MDE call it is
    estimating (a single-column slice plus one vectorized event-count
    search, not the full candidate-pool ranking sweep). Returns 0 (not a
    real window size) if the node turns out unreachable -- an unreachable
    node fails fast in the real call regardless, so weighting it as
    negligible cost for the ETA is reasonable, not misleading.

    Behavioral targets short-circuit before touching `full_df` at all:
    `_resolve_target_ranking_window()`'s own behavioral branch never reads
    the target's column either (the window is the fixed
    `target_ranking_window_rows`, capped by available rows) -- mirroring
    that here avoids requiring `full_df` to even contain this node's own
    column for a case where the real resolve never looks at it.'''
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp
    n_usable = usable_end - warmup
    if is_behavioral_target:
        return min(target_ranking_window_rows, n_usable)
    usable_df = full_df[[node]].iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, n_usable)
    window_rows, _ = _resolve_target_ranking_window(
        node, full_df, usable_df, lib_ratio,
        is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
    )
    return window_rows or 0


def _metric_for(column, targets, metric, target_metric):
    '''TARGETS (continuous behavioral traces) always score via
    target_metric ('correlation'); everything else (TSLS-converted neuron
    columns) uses metric ('f1' by default) -- see load_config()'s
    TARGET_METRIC docstring for why this split is required, not cosmetic.'''
    return target_metric if column in targets else metric


# =============================================================================
# Delay/column-naming infrastructure -- ported UNCHANGED from
# simple-mde-gmn.py. Pure string/DataFrame manipulation with no dependency
# on what the columns represent; reused as-is against TSLS's own
# `_tsls`-suffixed column names.
# =============================================================================

DELAY_MARKER = '::delay'


def _delayed_column_name(observable, delay):
    return f'{observable}{DELAY_MARKER}{delay}'


def _parse_delayed_column_name(name):
    observable, _, delay_str = name.rpartition(DELAY_MARKER)
    return observable, int(delay_str)


def observable_and_delay(column_name, raw_columns):
    '''Maps a (possibly delay-expanded) candidate column name back to its
    underlying (observable, delay) pair. Delay=0 candidates are just the
    raw observable name itself (see expand_delay_columns()'s docstring for
    why that delay isn't also given a suffixed alias) -- checked against
    `raw_columns` FIRST rather than by string shape alone, in case a real
    channel name were ever to collide with the delay-suffix marker
    (membership is the authoritative signal whenever available).

    Falls back to a marker-presence check (bug found 2026-08-25, job
    39605472/39605473 -- ported from simple-mde-gmn.py's own identical
    fix) when `column_name` isn't in `raw_columns` at all. Originally this
    triggered on every run, because depth-0's mde_results entry used to be
    computed in main() against the *full*, unfiltered candidate pool,
    before ccm_prefilter_observables() ever ran -- main() now runs the
    pre-filter first and seeds depth-0 from the *same* filtered pool
    build_nested_mde_network() itself uses, so that original mismatch no
    longer happens on a normal run. The fallback stays as a cache-reuse
    safety net: with REUSE_BEST_PARAMS, a reused mde_results entry can come
    from an earlier job that used a different CCM_PREFILTER_RHO_MIN/E (or
    predates the pre-filter feature entirely), so its SELECTED_COLUMNS can
    still legitimately reference an observable absent from *this* run's
    (filtered) `raw_columns`. Without this fallback,
    _parse_delayed_column_name() crashes on any such name (confirmed:
    `ValueError: invalid literal for int() with base 10: 'CA1_12_tsls'`)
    since it has no delay marker to parse. Residual risk is unchanged from
    before this fallback existed: only a real channel name that is BOTH
    missing from `raw_columns` AND happens to contain the literal
    `DELAY_MARKER` substring would still be misclassified -- this
    codebase's actual naming scheme never produces that.'''
    if column_name in raw_columns:
        return column_name, 0
    if DELAY_MARKER not in column_name:
        return column_name, 0
    return _parse_delayed_column_name(column_name)


def expand_delay_columns(df, columns, delay_set):
    '''Adds one new column per (observable, delay) pair in `columns` x
    `delay_set` (delay=0 excluded -- the observable's own raw column
    already IS that pair, no separate alias needed) to a copy of `df`, each
    holding that observable's value abs(delay) rows in the past
    (pandas .shift() -- every delay is <= 0). This is what lets forward
    selection treat each (observable, delay) combination as its own
    independent candidate driver, instead of being stuck with one fixed lag
    applied uniformly to every column -- see build_nested_mde_network()'s
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


def derive_lag_set(delay_source, explicit_delay_set, lag_mode, lag_density, lag_step,
                    generate_length, mde_lib_size, min_lib_rows=50):
    '''Resolves the nonzero-delay set used for Stage 2's lagged-driver
    candidate pool (build_nested_mde_network()'s own Stage-2 pass) and,
    unioned with {0}, every node's unconditional self_delay_set -- see
    load_config()'s DELAY_SOURCE/LAG_MODE/LAG_DENSITY/LAG_STEP docstrings
    for the parameter semantics. Ported unchanged from simple-mde-gmn.py.

    delay_source='explicit': just the hand-picked delay_set list, stripped
    of any 0 entry.

    delay_source='lag_mode': auto-derived --
      lag_mode='generation_length_adjusted': bound = generate_length.
      lag_mode='every_lag_possible': bound = mde_lib_size - min_lib_rows
        (clamped to >= 1) -- as deep as the MDE library segment [0, A)
        structurally allows while reserving min_lib_rows real rows so MDE's
        own forward-selection still has a meaningful library left after the
        deepest lag's warm-up NaN rows are skipped.
    Either bound is then sampled every lag_step rows (lag_density='strided')
    or every single integer lag (lag_density='dense').

    Returns a sorted-descending list of negative ints -- no 0 entry, same
    convention as the explicit-source path above.'''
    if delay_source == 'explicit':
        return sorted((d for d in explicit_delay_set if d != 0), reverse=True)

    if lag_mode == 'generation_length_adjusted':
        bound = max(1, generate_length)
    elif lag_mode == 'every_lag_possible':
        bound = max(1, mde_lib_size - min_lib_rows)
    else:
        raise ValueError(f"Unknown lag_mode: {lag_mode!r}")

    step = lag_step if lag_density == 'strided' else 1
    if lag_density not in ('strided', 'dense'):
        raise ValueError(f"Unknown lag_density: {lag_density!r}")

    return list(range(-1, -bound - 1, -step))


def effective_mde_dimension(mde_d, candidate_columns, num_delays):
    '''MDE_D names the desired count of distinct *real* observables MDE
    should be able to select -- not delay-expanded columns. Each real
    observable occupies `num_delays` separate candidate columns (one per
    delay), so the D actually handed to proper_mde is scaled up by
    num_delays -- see simple-mde-gmn.py's own docstring for the full
    rationale (identical here). Capped at the number of real observables
    actually present in this call's own candidate pool.'''
    num_real_observables_available = len(candidate_columns) // num_delays
    return min(mde_d, num_real_observables_available) * num_delays


# =============================================================================
# MDE (mirrors notebook cells 2-5)
# =============================================================================

def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                        mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                        results_dir, delay_set, tp=1,
                        metric='f1', target_metric='correlation', gamma_delta=2,
                        ccm_sample=20, ccm_slope=0.01,
                        is_behavioral_target=True, target_ranking_window_rows=1000,
                        min_pred_events=20, max_window_rows=30000,
                        ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                        reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                        data_file=None, data_file_sha256=None, code_fingerprint=None,
                        cache_stage='stage1', max_e=None, inner_workers=1, raw_columns=None,
                        use_gpu=False):
    '''Run cleaned_helper_functions.proper_mde's forward selection for a
    single target variable -- TSLS-flavored equivalent of
    simple-mde-gmn.py's own run_mde_for_target() (dimx.MDE), matching that
    function's return shape (a 'm' namespace exposing MDEcolumns/MDErho/
    MDEOut/args.lib/args.pred/args.tau/args.Tp) exactly, so every downstream
    consumer ported from simple-mde-gmn.py (_accept_mde_result_edges(),
    _failed_mde_result(), plot_mde_rho_vs_dimension(),
    run_tp_vs_rho_sweep(), run_univariate_e_tau_grid()) works completely
    unchanged. `mde_d` here is already the *effective* dimension count (see
    effective_mde_dimension()) -- NOT what `max_e` should be capped at (see
    below), since Stage 2 inflates `mde_d` by num_delays for a completely
    unrelated reason (candidate-pool breadth, not per-candidate embedding
    depth).

    E/tau are no longer hardcoded at this call site (previously E=3,
    tau=-1, a workaround for proper_mde's E<=0 auto-select path being
    blocked for non-'correlation' metrics -- see
    cleaned_helper_functions.proper_mde's own E docstring for the fix:
    proper_embed_dimension, a metric-aware EmbedDimension replacement,
    already existed in that file but proper_mde's own E<=0 branch never
    called it). Now: E=0 (auto-select per candidate, per dimension, scored
    with whichever metric that call already uses -- matching
    simple-mde-gmn.py's own dx.MDE call, which never overrides E/tau
    either and lets dimx.MDE auto-select natively) and maxE=`max_e`, the
    RAW (un-scaled-by-num_delays) TSLS_MDE_GMN_MDE_D value -- the per-node
    driver-count budget this pipeline already exposes, reused here rather
    than inventing a separate hardcoded ceiling, per explicit direction.
    tau is simply no longer passed -- proper_mde's own default (-1) applies,
    same numeric value as before, just not redundantly re-pinned here.

    metric vs. ccmMetric passed to proper_mde: `target_metric` scores the
    RANKING stage (predictions of `target` -- a continuous TARGETS column
    at the depth-0 seed, a neuron at every Stage 1/Stage 2 node -- never
    `metric`) AND the E-auto-select sweep above (same reasoning: it also
    predicts `target`); `metric` scores the CCM causality gate (predictions
    of each CANDIDATE column, always a TSLS-converted neuron series) -- see
    load_config()'s TARGET_METRIC docstring and this file's _metric_for().
    Callers are responsible for resolving `target_metric` correctly for
    whatever `target` actually is here (`_metric_for(target, targets,
    metric, target_metric)` at the depth>=1 call sites -- see the
    metric-routing fix in docs/review-findings.md for why this call site
    previously got it wrong for every neuron node).

    WINDOW SIZING (docs/review-findings.md's own window-sizing section --
    found and fixed alongside the metric-routing bug above, since fixing
    that bug alone made `f1`-scored ranking degenerate under the previous
    fixed CCM_WINDOW_ROWS=800 window): the ranking-stage/E-auto-select
    window is no longer CCM_WINDOW_ROWS -- it is resolved per node by
    `_resolve_target_ranking_window()` (see that function's own docstring),
    decoupled entirely from the CCM gate's own convergence-curve library
    sizing (`ccm_lib_sizes_rows`, absolute row counts, never a percentage
    of this window -- see that parameter's own docstring below for why).
    `is_behavioral_target=True` (the depth-0 seed's own case): a fixed
    `target_ranking_window_rows` (default 1000), no event-count concept.
    `is_behavioral_target=False` (virtually every Stage 1/Stage 2 node):
    extended until `target`'s own PRED portion of the window (the part
    `f1`-scoring actually touches, not the window as a whole) contains at
    least `min_pred_events` (default 20, empirically calibrated -- see
    load_config()'s own docstring) real reset events, capped at
    `max_window_rows` (default 30000, a plain compute ceiling, not
    calibrated) -- see `_resolve_target_ranking_window()`'s own docstring
    for the full reasoning and the two earlier designs this replaces.

    LEAKAGE (see load_config()'s three-way A/B/end window docstring,
    identical rationale to simple-mde-gmn.py's own run_mde_for_target()):
    `prediction_start`/`prediction_length` here are always A and B-A. The
    row range handed to proper_mde is trimmed twice before being resolved
    to this node's own ranking window (see _resolve_target_ranking_window()
    and _ccm_window()): `warmup` rows are skipped at the front (delay-
    expansion NaN warm-up, same as simple-mde-gmn.py) and `tp` rows are
    trimmed from the back (B - tp, not B, so the last validated row's own
    t+tp target label can never land at or past B) -- both applied BEFORE
    the window is resolved, so neither the fixed depth-0 window nor the
    extended per-node neuron window can ever reach into the Generate/
    Forecast validation segment [B, end), regardless of how large either
    grows.

    CACHE (reuse_cache, gated by TSLS_MDE_GMN_REUSE_BEST_PARAMS): ported
    unchanged from simple-mde-gmn.py -- a cache hit reconstructs the same
    'm' namespace shape from a pickled dict instead of calling proper_mde
    at all. `cache_stage` ('stage1' default, 'stage2' from
    build_nested_mde_network()'s own Stage 2 pass) keeps the two stages'
    cache files distinct, same as simple-mde-gmn.py.

    INNER POOL (inner_workers, default 1 -- no pool at all): this call's
    own proper_mde() forward-selection is otherwise entirely single-core
    (see this module's own docstring -- proper_mde has no multiprocessing
    of its own). `inner_workers` opens exactly ONE ThreadPoolExecutor here,
    for this node's whole proper_mde() call -- reused across every
    dimension's ranking loop and every CCM-gate candidate inside that one
    call, then closed when this function returns -- and passes it down as
    proper_mde's own `pool` argument (see that function's docstring for
    why threads, not another process pool: every hot loop it reaches is a
    compiled pyEDM/scipy call that releases the GIL). Callers
    (build_nested_mde_network()'s own Stage 1 and Stage 2 passes) size
    this as `max(1, cores // outer_workers)`, recomputed fresh per wave
    from that wave's own actual size -- see that function's own docstring
    -- which is what keeps outer_workers * inner_workers <= cores always: the same
    invariant that kept simple-mde-gmn.py's own outer x inner CCM
    parallelism (see that file's own module docstring's FIXED section)
    from repeating the pool-squaring blowup apps.InteractionMatrix x
    pyEDM.CCM's nested pools caused once, before that fix. No TSLS-side
    equivalent of that bug ever existed here (proper_mde never had any
    multiprocessing at all until this parameter) -- this inner pool is
    new, sized the same careful way from the start.

    `raw_columns` (R21, docs/review-actions-v1.md): when given, builds a
    `{column: delay}` lookup via `observable_and_delay()` for every column
    in `observable_columns` and passes it to `proper_mde` as
    `column_delay`, enabling that function's own dynamic per-dimension
    exclusion radius (see its docstring). `None` (default, and what
    main()'s own depth-0 seed call passes -- that pool is always
    real-only, so there is no delay to look up) -> `column_delay=None` ->
    proper_mde's own no-op default. `build_nested_mde_network()`'s Stage 1
    and Stage 2 calls both pass `raw_columns=raw_columns` (their
    `observable_columns`/`candidate_pool` can genuinely contain
    delay-suffixed synthetic names, confirmed on real data this session).

    `use_gpu` (docs/no-kEDM-gpu-optimization-plan.md section 8): threaded
    straight through to `proper_mde`'s own `use_gpu` -- see that
    function's docstring. `False` (default) is byte-for-byte today's
    behaviour; `True` only actually changes anything on a machine with a
    visible CUDA device (transparent CPU fallback otherwise). Callers
    pass `cfg['GPU_ENABLED']` down this same chain that carries
    `mp_start_method` (main() -> build_network() ->
    build_nested_mde_network() -> here), not a separate flag of its own.

    `ccm_lib_sizes_rows` (window-sizing fix, docs/review-findings.md):
    explicit, absolute row counts for the CCM gate's own convergence-curve
    library sweep, passed to `proper_mde` as `libSizes` (bypassing its own
    `pLibSizes`-as-percentage-of-window computation entirely). Deliberately
    NOT a percentage of this node's own (now node-specific, potentially
    much larger) ranking window: R9's own empirical calibration
    (docs/review-findings.md) found the "rising, then plateauing"
    convergence shape is a property of the *absolute* library size, not
    its ratio to the surrounding frame -- confirmed by re-running that same
    calibration at both an 800-row frame (R9's own original scale) and an
    8000-row frame (a realistic neuron-target scale under this fix): the
    same absolute values showed the same qualitative shape at both scales.
    See load_config()'s own CCM_LIB_SIZES_ROWS docstring for the exact
    calibrated default and the reasoning behind extending R9/R17's own
    6-value grid with one more point.'''
    # Fallback only -- every real call site (main()'s depth-0 seed, and
    # both of build_nested_mde_network()'s per-wave passes) passes this
    # explicitly as the raw TSLS_MDE_GMN_MDE_D value. Falls back to mde_d
    # (the effective, possibly num_delays-scaled dimension) only if a
    # caller genuinely omits it, matching this function's own pre-max_e
    # behavior for that edge case rather than raising.
    if max_e is None:
        max_e = mde_d

    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]

    # See this function's own LEAKAGE docstring above for why both trims
    # happen before windowing, not after.
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp

    usable_df = mde_df.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)

    # Window-sizing fix (docs/review-findings.md) -- resolved BEFORE the
    # cache fingerprint below, since both the resolved window size and (for
    # a neuron target) pred_event_count are real, data-dependent inputs to
    # this call's own outcome and must be part of it (A5).
    window_rows, pred_event_count = _resolve_target_ranking_window(
        target, df, usable_df, lib_ratio,
        is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
    )
    if window_rows is None:
        shortfall = min_pred_events - pred_event_count
        cap = min(max_window_rows, len(usable_df))
        print(f'{target}: cannot reach MIN_PRED_EVENTS={min_pred_events} in the pred portion '
              f'of any window up to min(MAX_WINDOW_ROWS, leakage-safe range)={cap} rows -- '
              f'best achievable was {pred_event_count} events (short by {shortfall}). '
              f'Treating as a dead-end (zero drivers selected, no further nesting from this '
              f'node) rather than scoring it on too few events.', flush=True)
        return _failed_mde_result(target)
    if not is_behavioral_target:
        print(f'{target}: resolved ranking window={window_rows} rows, '
              f'pred_event_count={pred_event_count} (MIN_PRED_EVENTS={min_pred_events})',
              flush=True)

    fingerprint = None
    cache_filename = f'mde_selection_{cache_stage}_{target}.pkl'
    if reuse_cache:
        fingerprint = build_mde_selection_fingerprint(
            target, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
            window_rows, pred_event_count, ccm_lib_sizes_rows,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, max_e,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
                           load_one=_load_pickle_cache_entry)
        if hit is not None:
            entry, source = hit
            print(f'{target}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached MDE selection '
                  f'{source} -- skipping forward-selection entirely', flush=True)
            m = types.SimpleNamespace(
                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
                                            tau=entry['tau'], Tp=entry['Tp']),
            )
            m.MDEOut.to_csv(results_dir / f'mde_output_{cache_stage}_{target}.csv', index=False)
            selected_columns = list(m.MDEcolumns)
            print(f'MDE selected columns for {target}:', selected_columns, flush=True)
            return {
                'target': target,
                'mde_df': mde_df,
                'm': m,
                'SELECTED_COLUMNS': selected_columns,
            }
        print(f'{target}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached MDE selection '
              f'found under {prev_runs_dir} -- running forward-selection.', flush=True)

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    # One inner pool for this node's WHOLE proper_mde() call -- opened
    # once here, reused across every dimension's ranking loop and every
    # CCM-gate candidate inside that one call, closed on exit. See this
    # function's own INNER POOL docstring above for the outer_workers *
    # inner_workers <= cores invariant this relies on, and why threads
    # (nullcontext()/None when inner_workers <= 1, so this is a no-op
    # for any caller that doesn't pass it -- identical to before this
    # parameter existed).
    inner_pool_ctx = (
        ThreadPoolExecutor(max_workers=inner_workers) if inner_workers > 1
        else nullcontext()
    )
    # R21 (docs/review-actions-v1.md): column_delay lookup for proper_mde's
    # own dynamic per-dimension exclusion radius -- only meaningful when
    # `observable_columns` can contain delay-suffixed synthetic names
    # (build_nested_mde_network()'s Stage 1/Stage 2 candidate pools; see
    # that function's own call sites, which pass raw_columns=raw_columns).
    # `raw_columns=None` (main()'s own depth-0 seed call, always a
    # real-only pool) -> column_delay=None -> proper_mde's own no-op
    # default, identical to before this parameter existed.
    column_delay = (
        {c: observable_and_delay(c, raw_columns)[1] for c in observable_columns}
        if raw_columns is not None else None
    )
    # Diagnostic only (found while investigating a real, very slow Stage 2
    # call, docs/review-findings.md) -- same raw_columns-gated pattern as
    # column_delay above, just the observable half of observable_and_delay()'s
    # own return value instead of the delay half. Lets proper_mde's own
    # verbose log show how many *distinct* real observables a long run has
    # actually found, alongside how many dimensions it took to find them.
    column_observable = (
        {c: observable_and_delay(c, raw_columns)[0] for c in observable_columns}
        if raw_columns is not None else None
    )

    with inner_pool_ctx as inner_pool:
        result = tsls.proper_mde(
            ccm_df, target=target, columns=observable_columns, D=mde_d,
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
            Tp=tp, E=0, maxE=max_e, exclusionRadius=exclusion_radius,
            libSizes=ccm_lib_sizes_rows, sample=ccm_sample, ccmSlope=ccm_slope,
            crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
            metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
            verbose=True, pool=inner_pool, column_delay=column_delay,
            column_observable=column_observable, use_gpu=use_gpu,
        )

    mde_out = pd.DataFrame({'rho': result['MDErho']})
    mde_out.to_csv(results_dir / f'mde_output_{cache_stage}_{target}.csv', index=False)

    selected_columns = list(result['MDEcolumns'])
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    m = types.SimpleNamespace(
        MDEcolumns=selected_columns,
        MDErho=np.asarray(result['MDErho']),
        MDEOut=mde_out,
        args=types.SimpleNamespace(
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)], tau=-1, Tp=tp,
        ),
    )

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
        'mde_df': ccm_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }


def run_multi_mde_for_targets(targets, df, observable_columns, prediction_start, prediction_length,
                               mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                               results_dir, delay_set, tp=1,
                               metric='f1', target_metric='correlation', gamma_delta=2,
                               ccm_sample=20, ccm_slope=0.01,
                               target_ranking_window_rows=1000,
                               ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                               reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                               data_file=None, data_file_sha256=None, code_fingerprint=None,
                               cache_stage='stage1', E=3, max_e=10, target_aggregate='mean'):
    '''Multi-target counterpart of run_mde_for_target() -- GMN_MODE='combined'
    only (see load_config()'s MULTI_MDE_E docstring and main()'s own MDE
    block). Runs cleaned_helper_functions.multi_proper_mde's forward
    selection ONCE for the whole `targets` group, picking drivers that
    jointly predict every target at once, instead of one independent
    proper_mde call per target. Every other argument has the identical
    meaning to run_mde_for_target()'s own; `E`/`max_e` (E<=0 auto-selects
    per candidate via multi_proper_embed_dimension(), bounded by max_e --
    see multi_proper_mde's own docstring; E>0 is still the static,
    every-candidate-identical original behavior) and `target_aggregate`
    are the genuinely new knobs.

    Returns the same dict shape as run_mde_for_target() except keyed by
    `'targets'` (list) instead of a single `'target'` -- main() builds one
    per-target *view* of this single shared result (same 'm'/'mde_df'/
    'SELECTED_COLUMNS', just re-keyed with a 'target' entry) so every
    downstream consumer that indexes mde_results[target] (network
    construction's depth-0 seed, the MDE diagnostics plots/sweeps) keeps
    working completely unchanged -- see main()'s own MDE block. This is
    also why `mde_df`/`SELECTED_COLUMNS`/edge weights end up identical
    (shared) across every target in the group: the joint selection doesn't
    produce a per-target breakdown, by design (it's the same drivers,
    accepted because they predict the *group*, not any one target alone).'''
    mde_df = df[list(targets) + list(observable_columns)].iloc[:prediction_start + prediction_length]

    # See run_mde_for_target()'s own LEAKAGE docstring -- identical windowing
    # rationale, just against a dataFrame carrying every target column
    # instead of one.
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp
    usable_df = mde_df.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)

    # Window-sizing fix (docs/review-findings.md): every target here is
    # always behavioral (GMN_MODE='combined''s depth-0 seed only, per this
    # function's own docstring) -- no per-node event-count extension, just
    # the same fixed constant _resolve_target_ranking_window() uses for a
    # single behavioral target, applied once for the whole joint call.
    window_rows = min(target_ranking_window_rows, len(usable_df))

    targets_tag = '_'.join(targets)
    fingerprint = None
    cache_filename = f'mde_selection_{cache_stage}_multi_{targets_tag}.pkl'
    if reuse_cache:
        fingerprint = build_multi_mde_selection_fingerprint(
            targets, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            target_ranking_window_rows, window_rows, ccm_lib_sizes_rows,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, E, target_aggregate,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
                           load_one=_load_pickle_cache_entry)
        if hit is not None:
            entry, source = hit
            print(f'{targets_tag}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached joint MDE '
                  f'selection {source} -- skipping forward-selection entirely', flush=True)
            m = types.SimpleNamespace(
                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
                                            tau=entry['tau'], Tp=entry['Tp']),
            )
            m.MDEOut.to_csv(results_dir / f'mde_output_{cache_stage}_multi_{targets_tag}.csv', index=False)
            selected_columns = list(m.MDEcolumns)
            print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)
            return {
                'targets': list(targets),
                'mde_df': mde_df,
                'm': m,
                'SELECTED_COLUMNS': selected_columns,
            }
        print(f'{targets_tag}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached joint '
              f'MDE selection found under {prev_runs_dir} -- running forward-selection.', flush=True)

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    result = tsls.multi_proper_mde(
        ccm_df, targets=list(targets), columns=observable_columns, D=mde_d,
        lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
        Tp=tp, E=E, maxE=max_e, exclusionRadius=exclusion_radius,
        libSizes=ccm_lib_sizes_rows, sample=ccm_sample, ccmSlope=ccm_slope,
        crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
        metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
        target_aggregate=target_aggregate,
        verbose=True,
    )

    mde_out = pd.DataFrame({'rho': result['MDErho']})
    mde_out.to_csv(results_dir / f'mde_output_{cache_stage}_multi_{targets_tag}.csv', index=False)

    selected_columns = list(result['MDEcolumns'])
    print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)

    m = types.SimpleNamespace(
        MDEcolumns=selected_columns,
        MDErho=np.asarray(result['MDErho']),
        MDEOut=mde_out,
        args=types.SimpleNamespace(
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)], tau=-1, Tp=tp,
        ),
    )

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
        'targets': list(targets),
        'mde_df': ccm_df,
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
    # expand_delay_columns()'s docstring) that don't exist in the raw `df`
    # this function indexes into -- mapped back to their underlying raw
    # observable (the delay itself isn't meaningful for this plot, a visual
    # trajectory of the real, undelayed driver values).
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
    # [start, stop] list form (1-offset, inclusive) -- proper_simplex/
    # proper_embed_dimension expect this form, not pyEDM's own "start stop"
    # string form.
    split = int(n_rows * train_frac)
    return [1, split], [split + 1, n_rows]


def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals, metric, gamma_delta):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                out = tsls.proper_simplex(
                    lib_df, columns=target, target=target,
                    E=E, tau=tau, lib=lib_range, pred=pred_range,
                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][metric]
            except Exception:
                pass
    return rho_grid


def estimate_node_dimension(lib_df, node, metric, gamma_delta, max_e=15, fallback=4):
    '''EmbedDimension-equivalent for TSLS-aware metrics via
    proper_embed_dimension (pyEDM's own EmbedDimension is Pearson-rho only,
    so it can't score 'f1'/'gamma').'''
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = tsls.proper_embed_dimension(
            lib_df, columns=node, target=node,
            maxE=max_e, lib=lib_range, pred=pred_range,
            metric=metric, gammaDelta=gamma_delta,
        )
        return int(ed.loc[ed[metric].idxmax(), 'E'])
    except Exception:
        return fallback


def _ccm_converged(mean_vals, direction, ccm_slope):
    '''Two-part convergence test mirroring proper_mde's own CCM gate
    (cleaned_helper_functions.py: endpoint improved AND fitted slope vs.
    normalized libSize position exceeds ccm_slope). Returns (passed, slope).'''
    if np.all(np.isnan(mean_vals)):
        return False, 0.0
    x = np.arange(len(mean_vals), dtype=float)
    x /= x[-1] if x[-1] > 0 else 1.0
    slope = float(np.polyfit(x, mean_vals, 1)[0]) if len(mean_vals) > 1 else 0.0
    endpoint_improved = (direction * (mean_vals[-1] - mean_vals[0])) > 0
    return (endpoint_improved and (direction * slope) > ccm_slope), slope


def build_interaction_matrix(data_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
                              gamma_delta, ccm_window_rows, ccm_lib_size_pcts, ccm_sample,
                              ccm_slope):
    '''NxN interaction matrix via cleaned_helper_functions.proper_ccm,
    replacing apps.InteractionMatrix's CLI subprocess -- MODE='GMN' only.
    No multiprocessing-pool-nesting concern at all here (unlike
    simple-mde-gmn.py's own MODE='GMN' branch): proper_ccm has no pool of
    its own to nest inside an outer one.

    Per-pair metric selection via _metric_for(): `nodes` mixes TSLS-
    converted neuron columns with the (continuous, non-TSLS) TARGETS
    columns, so each proper_ccm call's target_measurement/
    columns_measurement must independently match whichever side (a or b) is
    actually being predicted in that direction.

    Matches apps.CreateNetwork()'s documented convention:
    `iMatrix.loc[driven_node, driver_node]` = skill of predicting
    driven_node FROM driver_node's history. proper_ccm's c2t direction
    (`target` predicted FROM `columns`) maps directly onto that; calling it
    once per *unordered* pair yields both c2t and t2c. Non-converged
    directions are left at 0 (the pre-filled default), which is what
    build_network()'s downstream threshold + per-node top-N-drivers
    selection already expects.'''
    windowed, _ = _ccm_window(data_lib, 0.5, ccm_window_rows)
    libSizes = sorted({max(10, int(len(windowed) * p / 100.0)) for p in ccm_lib_size_pcts})

    iMatrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            a_metric = _metric_for(a, targets, metric, target_metric)
            b_metric = _metric_for(b, targets, metric, target_metric)
            try:
                c2t, t2c, _ = tsls.proper_ccm(
                    windowed, libSizes=libSizes, sample=ccm_sample,
                    tau=ccm_tau, Tp=1, E=ccm_E,
                    target=a, columns=b,
                    target_measurement=a_metric, columns_measurement=b_metric,
                    gammaDelta=gamma_delta,
                )
            except Exception:
                continue

            for series, driven, driver, node_metric in (
                (c2t, a, b, a_metric), (t2c, b, a, b_metric),
            ):
                mean_vals = np.nanmean(series, axis=0)
                direction = 1 if node_metric in ('correlation', 'f1', 'gamma') else -1
                passed, slope = _ccm_converged(mean_vals, direction, ccm_slope)
                if passed:
                    iMatrix.loc[driven, driver] = float(mean_vals[-1])

    for n in nodes:
        iMatrix.loc[n, n] = np.nan
    return iMatrix


# Progress-monitoring log (found necessary this session: this pipeline has
# a real history of runs that never finish -- job 38976891 timed out after
# 4+ hours on a single depth-1 Stage 1 wave of 3 nodes, still doing real
# work throughout but producing no new output -- with no timing
# instrumentation anywhere to say which node was slow or how close to done
# it was). HEARTBEAT_SECONDS: how often the Stage 1/Stage 2 wave loops
# below re-check and log "still working" while a wave's own nodes are
# still running, even if none of them have finished yet -- frequent enough
# to show real liveness on a run that used to go silent for hours, not so
# frequent it adds meaningful I/O overhead or log noise.
HEARTBEAT_SECONDS = 300


def _log_progress(log_path, message):
    '''Appends one timestamped line to the progress-monitoring log (see
    load_config()'s own PROGRESS_LOG_PATH -- flash, not bucket, so it's
    readable while a job is still running, and survives independently of
    the bucket-archival step). Every write is flushed and fsynced
    immediately -- this is a diagnostic log for a pipeline with a real
    history of runs that never finish (see JOB_LOG.md), so a line must
    survive a crash/OOM/timeout the instant after it's written, not just
    on a clean close. `log_path=None` is a silent no-op (callers that
    don't have one, e.g. tests, never need to guard every call site with
    an `if`).

    Only ever called from the main process -- Stage 1/Stage 2 worker
    processes (run inside ProcessPoolExecutor) are never given this path,
    so there is no concurrent-writer question to solve here (see this
    module's own docstring section on the monitoring log for why timing
    is measured from the orchestrating process's own submit/as_completed
    calls instead of from inside proper_mde itself).'''
    if log_path is None:
        return
    with open(log_path, 'a') as f:
        f.write(f'{datetime.now().isoformat()} {message}\n')
        f.flush()
        os.fsync(f.fileno())


def _format_eta(durations, n_remaining, per_unit=None, unit_remaining=None):
    '''Rough remaining-time estimate for a monitoring-log heartbeat/summary
    line, from durations already observed *in this same run* (never across
    jobs -- see load_config()'s own PROGRESS_LOG_PATH docstring for why).

    Two modes:
    - Flat per-task average: pass `durations` (list of completed task
      durations, seconds) and `n_remaining` (how many more of that same
      task type are left) -- estimate is `mean(durations) * n_remaining`.
    - Per-window-row rate (used for Stage 1/Stage 2 nodes, whose own cost
      scales roughly with resolved window size -- see
      _resolve_target_ranking_window()): pass `per_unit` (list of
      observed seconds-per-window-row, one entry per completed node) and
      `unit_remaining` (total window rows still queued across pending
      nodes) instead of/alongside durations -- estimate is
      `mean(per_unit) * unit_remaining`. Falls back to the flat estimate
      if `per_unit` has no observations yet (e.g. the very first node in
      a wave).

    Returns '' (not an estimate) when there is nothing completed yet to
    base one on -- an ETA fabricated from zero observations would be
    actively misleading, not just uninformative.'''
    if per_unit and unit_remaining is not None and unit_remaining > 0:
        rate = sum(per_unit) / len(per_unit)
        return (f' (~{rate * unit_remaining:.0f}s remaining, based on {len(per_unit)} '
                f'completed node(s) averaging {rate * 1000:.2f}ms/window-row)')
    if durations and n_remaining and n_remaining > 0:
        mean_s = sum(durations) / len(durations)
        return (f' (~{mean_s * n_remaining:.0f}s remaining, based on {len(durations)} '
                f'completed averaging {mean_s:.1f}s each)')
    return ''


def _failed_mde_result(target):
    '''Fallback run_mde_for_target()-shaped result for a node/target whose
    MDE call raised, instead of returning normally -- ported unchanged from
    simple-mde-gmn.py (job 37946756: dimx's own internal EmbedDimension hit
    an all-NaN rho column; proper_mde has its own analogous internal
    failure modes against degenerate candidate data, not something this
    pipeline's own gates catch or control). Shaped to be indistinguishable,
    to every downstream consumer, from a node that legitimately selected
    zero drivers -- already one of build_nested_mde_network()'s documented
    ordinary BFS termination conditions -- so a single degenerate node
    dead-ends cleanly instead of the exception propagating out of a
    ProcessPoolExecutor wave and crashing everything already computed
    alongside it.'''
    return {
        'target': target,
        'mde_df': None,
        'm': types.SimpleNamespace(
            MDEcolumns=[], MDErho=np.array([]), MDEOut=None,
            args=types.SimpleNamespace(lib=None, pred=None, tau=None, Tp=None),
        ),
        'SELECTED_COLUMNS': [],
    }


def _failed_multi_mde_result(targets):
    '''Multi-target counterpart of _failed_mde_result() -- same shape as
    run_multi_mde_for_targets()'s own return (``'targets'`` list instead of
    a single ``'target'``), used when GMN_MODE='combined''s joint
    multi_proper_mde call raises instead of returning normally.'''
    return {
        'targets': list(targets),
        'mde_df': None,
        'm': types.SimpleNamespace(
            MDEcolumns=[], MDErho=np.array([]), MDEOut=None,
            args=types.SimpleNamespace(lib=None, pred=None, tau=None, Tp=None),
        ),
        'SELECTED_COLUMNS': [],
    }


def _accept_mde_result_edges(graph, edges, node, result, raw_columns):
    '''Shared MDE-edge-acceptance rules -- self-loop guard, NaN
    marginal-gain rejection, eager cycle rejection -- ported unchanged from
    simple-mde-gmn.py (see that file's build_nested_mde_network() docstring
    for the full rationale on each). Applied to every column
    run_mde_for_target() selected for `node`. Mutates `graph` in place and
    appends accepted rows to `edges`.

    Returns the list of (driver, delay) pairs actually accepted as edges
    into `node` (in selection order).'''
    accepted = []
    cols = result['SELECTED_COLUMNS']
    gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))

    for col, gain in zip(cols, gains):
        driver, delay = observable_and_delay(col, raw_columns)

        if driver == node:
            continue  # explicit self-loop guard (belt-and-suspenders)

        if np.isnan(gain):
            print(f'MDE edge accept: rejected {driver} (delay {delay}) -> {node} '
                  f'(marginal MDE rho gain is NaN -- treating as a dead-end edge)', flush=True)
            continue

        graph.add_node(driver)
        key = graph.add_edge(driver, node, delay=delay, weight=gain)

        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(driver, node, key=key)
            if graph.degree(driver) == 0:
                graph.remove_node(driver)
            print(f'MDE edge accept: rejected {driver} (delay {delay}) -> {node} '
                  f'(would create a cycle)', flush=True)
            continue

        edges.append({'driver': driver, 'node': node, 'delay': delay, 'weight': gain})
        accepted.append((driver, delay))

    return accepted


def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
                              prediction_start, prediction_length, nested_n, nested_m,
                              embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                              results_dir, mde_results, delay_set, tp,
                              metric, target_metric, gamma_delta,
                              target_ranking_window_rows, min_pred_events, max_window_rows,
                              ccm_lib_sizes_rows, ccm_sample, ccm_slope, cores,
                              lag_set, stage2_allow_real_observables,
                              reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                              data_file=None, data_file_sha256=None, code_fingerprint=None,
                              progress_log_path=None, mp_start_method='fork', use_gpu=False):
    '''Two-stage nested-MDE network construction, MODE='MDE-GMN' only --
    FULLY SERIAL between stages (R12, 2026-08-27: reverted from the
    2026-08-25 unified-per-wave-BFS design -- see JOB_LOG.md for that
    design and why it's being replaced here). Stage 1's own real-
    observable BFS runs to completion, over every depth the shared
    nested_m budget allows, before Stage 2 ever starts. Stage 2 then runs
    its own separate BFS -- seeded with EVERY node Stage 1 discovered (so
    every backbone node still gets a lagged/real embedding attached, the
    same guarantee the old unified design gave one wave at a time) --
    and Stage 2 selections are themselves recursive: a delay=0 (real-
    observable) driver Stage 2 selects (only reachable when
    stage2_allow_real_observables=True) is enqueued for its own further
    Stage 2 round, exactly like Stage 1 nodes recurse into further Stage
    1 rounds, NOT a Stage 1 round -- Stage 2 never triggers a fresh
    Stage-1-style (real-observable-only) call on anything.

    This is a deliberate, explicit un-doing of the 2026-08-25 unification,
    per review action R12: that revision fixed a real bug (the historical
    dead-end-leaf problem, Sec 5.3 -- attach_lag_embeddings() ran ONCE
    over Stage 1's final node set, non-recursively, so a Stage-2-selected
    real observable could never itself become a nested-MDE target) but
    serialization and recursion are orthogonal properties, and the fix
    conflated them. What had to be restored is the sequencing (Stage 1
    complete, THEN Stage 2); what had to be kept is Stage 2's own
    recursion (added 2026-08-25, not reverted here). Neither this
    function's Stage 1 phase nor its Stage 2 phase flattens into a single
    non-recursive sweep.

    Depth accounting is joint across both phases -- a node's depth is its
    distance from the target regardless of which phase discovered it (see
    `node_depth` below), and the same nested_m budget check
    (`nested_m == 0 or depth < nested_m`, depth = the discovering node's
    OWN depth, unchanged arithmetic from the prior revision -- R13 does
    not touch this) gates further recursion in EITHER phase. A
    consequence documented in the review (R12/R15's own interaction
    note, R15 itself out of scope here): because Stage 1 now runs to full
    depth before Stage 2 ever sees a node, a real-observable Stage 2
    selects on a branch Stage 1 already carried out to the depth cap is
    beyond budget and terminates immediately -- correct budget-bound
    termination, not a broken recursion. Distinguishing that from Stage 2
    simply finding zero real-observable drivers on a branch (driver-
    exhausted termination) in the run log is R15 step 4's own job, not
    done by this revision.

    Stage 2's initial wave seeds every node Stage 1 visited, at whatever
    depth Stage 1 discovered it at -- these depths are NOT uniform within
    that first Stage 2 wave (Stage 1's own BFS produced them one wave per
    depth; Stage 2 processes them all at once for maximum parallelism, the
    same "single sweep over Stage 1's final node set" shape the historical
    §5.3 bug had -- the difference is this sweep recurses further whereas
    that one didn't). Because a wave can therefore mix depths, a brand-new
    driver proposed by more than one same-wave parent takes the SHALLOWEST
    proposed depth (see the next_wave dict-of-mins pattern below), not an
    arbitrary iteration-order-dependent one -- this is a strict
    correctness improvement over the prior revision's plain-list next_wave
    (harmless there, since every wave was single-depth by construction, so
    plain-list and min-tracked next_wave picked the same result).

    Differences from simple-mde-gmn.py, both pre-existing (not new to
    this revision): (1) neither pool passes `initializer=` -- simple-
    mde-gmn.py's own _mde_wave_worker_init() reapplies dimx.Run
    monkeypatches inside every worker, which has no TSLS-side equivalent
    (no dimx.Run to monkeypatch at all); (2) the outer/inner split is
    computed the same way (outer_workers = min(node count, cores),
    inner_workers = cores // outer_workers, recomputed fresh per phase
    per wave from that phase's own real node count) but fed to a
    different layer than simple-mde-gmn.py's dx.MDE(cores=...) -- here
    inner_workers becomes a ThreadPoolExecutor opened once per node
    inside run_mde_for_target() itself (see that function's own INNER
    POOL docstring), since proper_mde has no multiprocessing of its own
    to hand a bare core count to.'''
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    stage1_num_delays = len(expanded_columns) // len(raw_columns)

    stage2_df, stage2_expanded_columns = expand_delay_columns(expanded_df, raw_columns, lag_set)
    stage2_lag_only_columns = stage2_expanded_columns[len(raw_columns):]
    stage2_candidate_universe = (
        stage2_expanded_columns if stage2_allow_real_observables else stage2_lag_only_columns
    )
    stage2_num_delays = len(lag_set) + (1 if stage2_allow_real_observables else 0)

    edges = []
    nested_mde_results = {}
    node_depth = {t: 0 for t in targets}

    # =====================================================================
    # PHASE 1 -- Stage 1: real-observable-only BFS, run to full completion
    # (every depth the nested_m budget allows) before Stage 2 starts at all.
    # =====================================================================
    visited1 = set()
    current_wave = [(t, 0) for t in targets]
    stage1_start = time.perf_counter()
    stage1_durations = []   # seconds per completed node, this run only
    stage1_rates = []       # seconds per resolved window-row, this run only -- see _format_eta()
    _log_progress(progress_log_path, 'STAGE 1 START (real-observable BFS)')

    while current_wave:
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited1 or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        to_compute = [(node, depth) for node, depth in wave_nodes
                      if not (depth == 0 and node in mde_results)]
        stage1_results = {
            node: mde_results[node] for node, depth in wave_nodes
            if depth == 0 and node in mde_results
        }

        if to_compute:
            outer_workers = min(len(to_compute), max(1, cores))
            inner_workers = max(1, cores // outer_workers)
            # R5 (docs/review-actions-v1.md): product logged explicitly so
            # a run's own log can be checked for outer*inner <= cores
            # without re-deriving it by hand.
            print(f'nested MDE (Stage 1): wave of {len(to_compute)} node(s) at depth '
                  f'{to_compute[0][1]} -- {outer_workers} parallel worker(s) '
                  f'x {inner_workers} inner thread(s) each '
                  f'(product={outer_workers * inner_workers}, cores={cores})', flush=True)
            wave_start = time.perf_counter()
            _log_progress(progress_log_path,
                          f'stage1 wave START -- {len(to_compute)} node(s) at depth '
                          f'{to_compute[0][1]}: {[n for n, _ in to_compute]}')

            # mp_start_method: 'fork' (default, CPU-only runs -- unchanged
            # behaviour) or 'spawn' when the GPU/torch path is enabled, since
            # CUDA does not support fork(). See load_config()'s GPU_ENABLED
            # docstring and docs/no-kEDM-gpu-optimization-plan.md section 7a.
            with ProcessPoolExecutor(max_workers=outer_workers,
                                      mp_context=mp.get_context(mp_start_method)) as executor:
                futures = {}
                node_window_estimate = {}
                for node, depth in to_compute:
                    candidate_pool = [c for c in expanded_columns
                                       if observable_and_delay(c, raw_columns)[0] != node]
                    effective_d = effective_mde_dimension(nested_n, candidate_pool, stage1_num_delays)
                    is_behav = (node in targets)
                    # Monitoring-log ETA input only -- see _preview_window_rows()'s
                    # own docstring for why this speculative resolve is safe and
                    # independent of the real one run_mde_for_target() performs.
                    node_window_estimate[node] = _preview_window_rows(
                        node, expanded_df, delay_set, prediction_start, prediction_length, tp,
                        is_behav, target_ranking_window_rows, min_pred_events, max_window_rows,
                    )
                    # Metric-routing fix (found while verifying R22's own
                    # E-engine-swap effect): `node` here is virtually never
                    # one of `targets` (Stage 1 discovers real-observable
                    # DRIVERS) -- passing the closed-over `target_metric`
                    # unconditionally would score this node's own ranking
                    # stage/E-auto-select via TARGET_METRIC ('correlation')
                    # instead of _metric_for()'s own rule ('f1' for any
                    # non-TARGETS column). See run_mde_for_target()'s own
                    # docstring for why target_metric drives that stage
                    # (predictions of `target` == `node` here) -- the CCM
                    # gate's own `metric` argument is untouched, already
                    # correct (it always scores a candidate, never `node`
                    # itself, and no candidate here is ever a target).
                    # All arguments from `tp` onward passed by keyword --
                    # this call site inserts new parameters (window-sizing
                    # fix) ahead of several trailing ones (reuse_cache
                    # onward); relying on positional order here would be a
                    # silent-miswiring trap the moment either signature
                    # changes again.
                    fut = executor.submit(
                        run_mde_for_target, node, expanded_df, candidate_pool,
                        prediction_start, prediction_length, effective_d,
                        embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                        results_dir, delay_set,
                        tp=tp, metric=metric,
                        target_metric=_metric_for(node, targets, metric, target_metric),
                        gamma_delta=gamma_delta, ccm_sample=ccm_sample, ccm_slope=ccm_slope,
                        is_behavioral_target=is_behav,
                        target_ranking_window_rows=target_ranking_window_rows,
                        min_pred_events=min_pred_events, max_window_rows=max_window_rows,
                        ccm_lib_sizes_rows=ccm_lib_sizes_rows,
                        reuse_cache=reuse_cache, prev_runs_dir=prev_runs_dir,
                        run_cache_dir=run_cache_dir, data_file=data_file,
                        data_file_sha256=data_file_sha256, code_fingerprint=code_fingerprint,
                        max_e=nested_n, inner_workers=inner_workers, raw_columns=raw_columns,
                        use_gpu=use_gpu,
                    )
                    futures[fut] = node

                # Heartbeat-polling replacement for a plain `for fut in
                # as_completed(futures):` -- see this module's own
                # HEARTBEAT_SECONDS docstring for why: a wave with no
                # heartbeat can go silent for hours (job 38976891) with no
                # way to tell "still working" from "stuck". Each timeout
                # with nothing newly done logs one heartbeat line; each
                # newly completed future is logged (and timed) immediately.
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=HEARTBEAT_SECONDS,
                                          return_when=FIRST_COMPLETED)
                    if not done:
                        elapsed = time.perf_counter() - wave_start
                        still_waiting = sorted(futures[f] for f in pending)
                        remaining_rows = sum(node_window_estimate.get(futures[f], 0)
                                              for f in pending)
                        eta = _format_eta(stage1_durations, len(pending),
                                          per_unit=stage1_rates, unit_remaining=remaining_rows)
                        _log_progress(
                            progress_log_path,
                            f'stage1 wave HEARTBEAT -- {len(futures) - len(pending)}/{len(futures)} '
                            f'done, elapsed={elapsed:.0f}s, still waiting on: {still_waiting}{eta}')
                        continue
                    for fut in done:
                        node = futures[fut]
                        node_duration = time.perf_counter() - wave_start
                        try:
                            stage1_results[node] = fut.result()
                            _log_progress(
                                progress_log_path,
                                f'stage1 node END -- {node} duration={node_duration:.1f}s '
                                f'selected={stage1_results[node]["SELECTED_COLUMNS"]}')
                        except Exception as exc:
                            # See _failed_mde_result()'s own docstring -- a
                            # single node's MDE call failing must not crash the
                            # rest of this wave or every already-completed node
                            # in prior waves.
                            print(f'nested MDE (Stage 1): {node} MDE call FAILED: {exc!r} -- treating '
                                  f'as a dead-end (zero drivers selected, no further nesting from this '
                                  f'node)', flush=True)
                            _log_progress(
                                progress_log_path,
                                f'stage1 node END -- {node} duration={node_duration:.1f}s FAILED: {exc!r}')
                            stage1_results[node] = _failed_mde_result(node)
                        stage1_durations.append(node_duration)
                        window_est = node_window_estimate.get(node, 0)
                        if window_est > 0:
                            stage1_rates.append(node_duration / window_est)

            _log_progress(progress_log_path,
                          f'stage1 wave END -- {len(to_compute)} node(s), '
                          f'{time.perf_counter() - wave_start:.1f}s total')

        next_wave = {}
        for node, depth in wave_nodes:
            visited1.add(node)
            node_depth[node] = depth
            result1 = stage1_results[node]
            nested_mde_results[node] = result1

            accepted1 = _accept_mde_result_edges(graph, edges, node, result1, raw_columns)
            for driver, delay in accepted1:
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
                    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)

        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 1 END -- {len(visited1)} node(s), '
                  f'{time.perf_counter() - stage1_start:.1f}s total')
    print(f'nested MDE (Stage 1): completed -- {len(visited1)} node(s) in the real-observable '
          f'backbone, Stage 2 starts now', flush=True)

    # =====================================================================
    # PHASE 2 -- Stage 2: lagged (+ optionally real) BFS. Seeded with EVERY
    # node Stage 1 discovered (so every backbone node still gets a lagged
    # embedding attached), then its own delay=0 (real-observable)
    # selections recurse into further Stage 2 rounds -- never a Stage 1
    # round -- subject to the same shared nested_m budget/depth.
    # =====================================================================
    visited2 = set()
    current_wave = [(node, node_depth[node]) for node in visited1]
    # R15: distinguishable termination causes for Stage 2's own real-
    # observable recursion -- budget_bound counts an accepted delay=0 pick
    # that would have recursed except the shared nested_m budget already
    # stopped at this depth; driver_exhausted counts a node whose own
    # depth was still within budget but whose Stage 2 call simply
    # selected zero real-observable drivers, so there was nothing to
    # recurse into regardless of budget. See build_nested_mde_network()'s
    # own final summary print for where these get reported.
    stage2_budget_bound = 0
    stage2_driver_exhausted = 0
    stage2_start = time.perf_counter()
    stage2_durations = []
    stage2_rates = []
    _log_progress(progress_log_path, 'STAGE 2 START (lagged/real BFS)')

    while current_wave:
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited2 or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        wave_depths = sorted({d for _, d in wave_nodes})
        outer_workers2 = min(len(wave_nodes), max(1, cores))
        inner_workers2 = max(1, cores // outer_workers2)
        # R5: product logged explicitly, same reasoning as Stage 1's own print above.
        print(f'nested MDE (Stage 2): wave of {len(wave_nodes)} node(s) at depth(s) '
              f'{wave_depths} -- {outer_workers2} parallel worker(s) '
              f'x {inner_workers2} inner thread(s) each '
              f'(product={outer_workers2 * inner_workers2}, cores={cores}), candidate pool size '
              f'{len(stage2_candidate_universe)} (lag_set={lag_set}, '
              f'stage2_allow_real_observables={stage2_allow_real_observables})', flush=True)
        wave_start = time.perf_counter()
        _log_progress(progress_log_path,
                      f'stage2 wave START -- {len(wave_nodes)} node(s) at depth(s) '
                      f'{wave_depths}: {[n for n, _ in wave_nodes]}')

        stage2_results = {}
        # Same fork-vs-spawn rationale as Stage 1's pool above.
        with ProcessPoolExecutor(max_workers=outer_workers2,
                                  mp_context=mp.get_context(mp_start_method)) as executor:
            futures = {}
            node_window_estimate = {}
            for node, depth in wave_nodes:
                candidate_pool = [c for c in stage2_candidate_universe
                                   if observable_and_delay(c, raw_columns)[0] != node]
                effective_d = effective_mde_dimension(nested_n, candidate_pool, stage2_num_delays)
                is_behav = (node in targets)
                # Monitoring-log ETA input only -- see Stage 1's own
                # identical comment above / _preview_window_rows()'s
                # docstring. Uses stage2_df/lag_set (Stage 2's own frame
                # and delay set), matching the real call just below.
                node_window_estimate[node] = _preview_window_rows(
                    node, stage2_df, lag_set, prediction_start, prediction_length, tp,
                    is_behav, target_ranking_window_rows, min_pred_events, max_window_rows,
                )
                # Metric-routing fix -- see Stage 1's own identical comment
                # above. `node` here is a Stage 1 backbone node (real
                # observable) or a Stage-2-recursed real observable, never
                # one of `targets` at this depth. All arguments from `tp`
                # onward passed by keyword -- see Stage 1's own identical
                # comment for why.
                fut = executor.submit(
                    run_mde_for_target, node, stage2_df, candidate_pool,
                    prediction_start, prediction_length, effective_d,
                    embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                    results_dir, lag_set,
                    tp=tp, metric=metric,
                    target_metric=_metric_for(node, targets, metric, target_metric),
                    gamma_delta=gamma_delta, ccm_sample=ccm_sample, ccm_slope=ccm_slope,
                    is_behavioral_target=is_behav,
                    target_ranking_window_rows=target_ranking_window_rows,
                    min_pred_events=min_pred_events, max_window_rows=max_window_rows,
                    ccm_lib_sizes_rows=ccm_lib_sizes_rows,
                    reuse_cache=reuse_cache, prev_runs_dir=prev_runs_dir,
                    run_cache_dir=run_cache_dir, data_file=data_file,
                    data_file_sha256=data_file_sha256, code_fingerprint=code_fingerprint,
                    cache_stage='stage2',
                    max_e=nested_n, inner_workers=inner_workers2, raw_columns=raw_columns,
                    use_gpu=use_gpu,
                )
                futures[fut] = node

            # Heartbeat-polling -- see Stage 1's own identical comment above.
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=HEARTBEAT_SECONDS,
                                      return_when=FIRST_COMPLETED)
                if not done:
                    elapsed = time.perf_counter() - wave_start
                    still_waiting = sorted(futures[f] for f in pending)
                    remaining_rows = sum(node_window_estimate.get(futures[f], 0) for f in pending)
                    eta = _format_eta(stage2_durations, len(pending),
                                      per_unit=stage2_rates, unit_remaining=remaining_rows)
                    _log_progress(
                        progress_log_path,
                        f'stage2 wave HEARTBEAT -- {len(futures) - len(pending)}/{len(futures)} '
                        f'done, elapsed={elapsed:.0f}s, still waiting on: {still_waiting}{eta}')
                    continue
                for fut in done:
                    node = futures[fut]
                    node_duration = time.perf_counter() - wave_start
                    try:
                        stage2_results[node] = fut.result()
                        _log_progress(
                            progress_log_path,
                            f'stage2 node END -- {node} duration={node_duration:.1f}s '
                            f'selected={stage2_results[node]["SELECTED_COLUMNS"]}')
                    except Exception as exc:
                        print(f'nested MDE (Stage 2): {node} MDE call FAILED: {exc!r} -- treating as '
                              f'a dead-end (zero lagged drivers attached for this node)', flush=True)
                        _log_progress(
                            progress_log_path,
                            f'stage2 node END -- {node} duration={node_duration:.1f}s FAILED: {exc!r}')
                        stage2_results[node] = _failed_mde_result(node)
                    stage2_durations.append(node_duration)
                    window_est = node_window_estimate.get(node, 0)
                    if window_est > 0:
                        stage2_rates.append(node_duration / window_est)

        _log_progress(progress_log_path,
                      f'stage2 wave END -- {len(wave_nodes)} node(s), '
                      f'{time.perf_counter() - wave_start:.1f}s total')

        next_wave = {}
        for node, depth in wave_nodes:
            visited2.add(node)
            node_depth.setdefault(node, depth)
            result2 = stage2_results[node]
            accepted2 = _accept_mde_result_edges(graph, edges, node, result2, raw_columns)
            under_budget = nested_m == 0 or depth < nested_m

            real_picks = [(driver, delay) for driver, delay in accepted2 if delay == 0]
            if under_budget and not real_picks:
                # Budget wasn't the constraint here -- Stage 2 genuinely
                # found no real-observable driver for this node at all.
                stage2_driver_exhausted += 1

            for driver, delay in real_picks:
                # A delay=0 (real-observable) pick here recurses into its
                # own further Stage 2 round -- NOT a Stage 1 round -- same
                # nested_m budget as Stage 1's own recursion above.
                if under_budget:
                    if driver not in visited2:
                        next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)
                else:
                    stage2_budget_bound += 1
                # delay != 0: leaf -- edge already added above.

        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 2 END -- {len(visited2)} node(s), '
                  f'{time.perf_counter() - stage2_start:.1f}s total')

    if stage2_allow_real_observables:
        print(f'nested MDE (Stage 2): real-observable recursion terminations -- '
              f'{stage2_budget_bound} budget-bound, {stage2_driver_exhausted} driver-exhausted',
              flush=True)

    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])

    # R13: nested_m is a whole-network depth budget shared by both stages
    # (node_depth is one dict populated by whichever stage discovers a
    # node first, checked against the same nested_m in both phases' own
    # recursion above) -- realized_max_depth reports how deep that joint
    # budget actually got used this run, distinct from the configured cap
    # itself (nested_m_display below).
    realized_max_depth = max(node_depth.values()) if node_depth else 0
    nested_m_display = 'unlimited' if nested_m == 0 else nested_m
    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
          f'(N={nested_n}, M={nested_m_display}, realized max depth={realized_max_depth})',
          flush=True)
    _log_progress(progress_log_path,
                  f'NESTED MDE NETWORK COMPLETE -- {len(graph.nodes)} nodes, {len(edges_df)} edges')

    return graph, edges_df, nested_mde_results


def _ccm_prefilter_pair(candidate, target, targets, windowed, e, tp, metric, target_metric,
                         gamma_delta, exclusion_radius, ccm_sample, lib_sizes):
    '''One (candidate, target) pair's worth of ccm_prefilter_observables()
    work -- run in an outer ThreadPoolExecutor worker (see that function's
    docstring for why threads: proper_ccm's compiled pyEDM/KDTree calls
    release the GIL, already established for run_mde_for_target()'s own
    inner_workers pool). `targets` (the full group, not just `target`) is
    passed straight through to _metric_for() so it correctly classifies any
    column that happens to be one of the group's own targets, matching
    build_interaction_matrix()'s own usage. Returns (candidate, target,
    rho) -- rho is NaN on any failure (degenerate column, etc.), same
    spirit as _failed_mde_result() elsewhere in this file.'''
    try:
        c2t, _, _ = tsls.proper_ccm(
            windowed, libSizes=lib_sizes, sample=ccm_sample, tau=-1, Tp=tp, E=e,
            target=target, columns=candidate,
            target_measurement=_metric_for(target, targets, metric, target_metric),
            columns_measurement=_metric_for(candidate, targets, metric, target_metric),
            exclusionRadius=exclusion_radius, gammaDelta=gamma_delta,
        )
        rho = float(np.nanmean(c2t, axis=0)[-1])
    except Exception:
        rho = float('nan')
    return candidate, target, rho


def ccm_prefilter_observables(df, candidate_columns, targets, prediction_start,
                               rho_min, e, tp, metric, target_metric, gamma_delta,
                               exclusion_radius, ccm_window_rows, ccm_sample,
                               libsize_ratios, cores):
    '''TSLS-flavored counterpart of simple-mde-gmn.py's own
    ccm_prefilter_observables() -- see that function's docstring for the
    full rationale (identical here: cheap CCM(observable -> target)
    pre-filter, run once per target_groups_for_mode() group right before
    either nested-MDE stage, covering both Stage 1's real observables and
    Stage 2's lags in one pass since Stage 2 draws from this same filtered
    list).

    Every CCM-call parameter here is either reused from an existing config
    value used elsewhere in this pipeline (Tp, metric/target_metric/
    gamma_delta via the existing _metric_for() convention, exclusionRadius,
    sample via CCM_SAMPLE) or, where nothing existing fits, either matches
    proper_mde's own unoverridden default (`tau`: confirmed "tau is simply
    no longer passed -- proper_mde's own default (-1) applies",
    run_mde_for_target()'s own docstring) or is its own new, deliberately
    pipeline-specific parameter (`e`, from CCM_PREFILTER_E -- NOT a reuse of
    MULTI_MDE_E: this pipeline's 40 Hz-sampled data needs a genuinely
    different static E than simple-mde-gmn.py's 1 Hz data for a comparable
    real-time embedding window at the same tau=-1 -- see
    load_config()'s own CCM_PREFILTER_E docstring for the full reasoning).

    `windowed` mirrors build_interaction_matrix()'s own windowing exactly:
    `df.iloc[:prediction_start]` (the same leakage-safe MDE library slice
    build_network()'s own `data_after_mde_lib` uses, just computed directly
    from the raw candidate pool here since `selected_union` doesn't exist
    yet at this point in build_network()) through `_ccm_window(..., 0.5,
    ccm_window_rows)` -- the *same* CCM_WINDOW_ROWS-bounded sub-window every
    other CCM call in this pipeline already goes through. `libsize_ratios`
    (fractions of `len(windowed)`, i.e. of this same bounded window) are
    converted to actual row-count libSizes here, deduped and floored at 10
    rows -- same floor convention build_interaction_matrix()'s own
    CCM_LIB_SIZE_PCTS conversion uses.

    Parallelism: one outer ThreadPoolExecutor (sized by `cores`) over
    (candidate, target) pairs -- see _ccm_prefilter_pair()'s own docstring
    for why threads are safe/effective here.

    Returns (kept_columns, scores) -- `kept_columns` is `candidate_columns`
    (with `targets` excluded) filtered down to those scoring >= `rho_min`,
    in original order; `scores` is {candidate: mean_rho} for every
    candidate considered (including dropped ones), for diagnostics.

    R2 (docs/review-actions-v1.md): audited main()'s call site and traced
    every column this function ever sees back to convert_to_tsls()'s own
    `f'{n}_tsls'` construction -- real channel names, never containing
    DELAY_MARKER, and expand_delay_columns() (the only place a
    DELAY_MARKER-suffixed name is ever created) is never called before
    this point in main() (it only runs inside build_nested_mde_network(),
    itself called from build_network(), which runs strictly after every
    call to this function). No cached mde_results entry feeds this
    function's candidate list either -- it always comes straight from
    main()'s own `obs_cols`. Confirmed clean; this assertion is kept as a
    cheap, permanent guard against a future change reopening that path,
    not a fix for a live issue.'''
    bad = [c for c in candidate_columns if DELAY_MARKER in c]
    assert not bad, (
        f'ccm_prefilter_observables() received DELAY_MARKER-suffixed column(s): {bad} -- '
        f'candidate_columns must be real (delay-0) observables only'
    )
    candidates = [c for c in candidate_columns if c not in targets]
    if not candidates:
        return [], {}

    usable_df = df.iloc[:prediction_start].reset_index(drop=True)
    windowed, _ = _ccm_window(usable_df, 0.5, ccm_window_rows)
    lib_sizes = sorted({max(10, round(r * len(windowed))) for r in libsize_ratios})

    per_candidate_scores = {c: [] for c in candidates}
    with ThreadPoolExecutor(max_workers=max(1, cores)) as executor:
        futures = [
            executor.submit(_ccm_prefilter_pair, c, t, targets, windowed, e, tp, metric,
                             target_metric, gamma_delta, exclusion_radius, ccm_sample, lib_sizes)
            for c in candidates for t in targets
        ]
        for fut in as_completed(futures):
            candidate, _target, rho = fut.result()
            per_candidate_scores[candidate].append(rho)

    scores = {c: float(np.nanmean(v)) if v else float('nan') for c, v in per_candidate_scores.items()}
    kept = [c for c in candidates if scores.get(c, float('nan')) >= rho_min]
    return kept, scores


def partition_network(graph, targets, external_input_mode):
    '''R14 (docs/review-actions-v1.md): partitions a MultiDiGraph built by
    build_nested_mde_network() into interior nodes (>=1 discovered driver
    -- genuinely simulated, their embedding built from those in-edges only,
    see _embedding_spec_for()'s docstring) and external inputs (0 discovered
    drivers -- what EXTERNAL_INPUT_MODE does with them is that function's
    own concern, not this one's). Stored explicitly on the graph object
    (`graph.graph['interior_nodes']`/`['external_inputs']`/
    `['external_input_mode']`) rather than left implicit in edge counts, so
    every consumer (embedding-spec construction, theta search, the 3D
    figure, run-log reporting) reads the same authoritative classification
    instead of re-deriving it.

    A TARGET with zero discovered drivers is NEVER classified as an
    external input, regardless of `external_input_mode` -- raises instead.
    This is a failed network, not a valid one: feeding a target's own real
    values back in as its "simulation" (EXTERNAL_INPUT_MODE='real_data')
    and then scoring Generate() fidelity against those same real values
    would be circular, producing a spuriously perfect score for exactly the
    case that most needs to be flagged as a failure (confirmed against Run
    A: head_position_y selected zero real drivers at depth 0 in that run --
    this exemption exists because that is a real, not hypothetical, case).

    Mutates `graph` in place; returns (interior_nodes, external_inputs) as
    plain sets for convenience.'''
    interior_nodes = set()
    external_inputs = set()
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            if node in targets:
                raise ValueError(
                    f"{node}: target selected zero drivers after nested-MDE (both stages) -- "
                    f"this is a failed network, not a valid external input. A target can never "
                    f"be exogenous: EXTERNAL_INPUT_MODE='real_data' would feed its own real "
                    f"values back in as its \"simulation\" and then score fidelity against those "
                    f"same real values, which is circular. Investigate why {node} found no "
                    f"drivers (EMBED_DIM_RHO_MIN/CROSS_MAP_RHO_MIN/CCM_PREFILTER_RHO_MIN too "
                    f"strict? too few candidate observables?) rather than working around it here."
                )
            external_inputs.add(node)
        else:
            interior_nodes.add(node)

    graph.graph['interior_nodes'] = interior_nodes
    graph.graph['external_inputs'] = external_inputs
    graph.graph['external_input_mode'] = external_input_mode
    return interior_nodes, external_inputs


def _simulated_nodes(graph):
    '''Nodes that actually go through gmn's own SMap/embedding machinery --
    interior nodes always, plus external inputs too under
    EXTERNAL_INPUT_MODE='self_simulate' (where they keep the old
    unconditional self_delay_set embedding, see _embedding_spec_for()'s
    docstring). Excludes real-data-mode external inputs: their Generate()/
    Forecast() short-circuit to a real-data passthrough with no embedding
    at all, so retuning theta for them (retune_theta_for_all_nodes(), which
    calls _embedding_spec_for() directly) would either waste effort on an
    inert result or crash on an empty spec.'''
    external_inputs = graph.graph.get('external_inputs', set())
    mode = graph.graph.get('external_input_mode', 'real_data')
    if mode == 'self_simulate':
        return list(graph.nodes)
    return [n for n in graph.nodes if n not in external_inputs]


def report_partition(graph, network_id, results_dir):
    '''R14's own explicit reporting requirement: how many external inputs,
    which observables and delays, and which interior nodes each feeds --
    printed to the run log and written to a per-network CSV in
    `results_dir` (partition_{network_id}.csv). Reads the classification
    partition_network() already stored on `graph.graph` rather than
    recomputing it.'''
    external_inputs = graph.graph.get('external_inputs', set())
    interior_nodes = graph.graph.get('interior_nodes', set())
    mode = graph.graph.get('external_input_mode', 'real_data')

    print(f'{network_id}: partition -- {len(interior_nodes)} interior node(s), '
          f'{len(external_inputs)} external input(s) (EXTERNAL_INPUT_MODE={mode!r})', flush=True)

    rows = []
    for node in sorted(external_inputs):
        out_edges = list(graph.out_edges(node, data=True))
        if not out_edges:
            print(f'  external input {node}: feeds nothing (dead-end leaf)', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': '', 'delay': ''})
            continue
        for _, fed_node, data in out_edges:
            print(f'  external input {node} -> {fed_node} (delay {data["delay"]})', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': fed_node,
                         'delay': data['delay']})
    for node in sorted(interior_nodes):
        rows.append({'node': node, 'role': 'interior', 'feeds': '', 'delay': ''})

    partition_file = results_dir / f'partition_{network_id}.csv'
    pd.DataFrame(rows, columns=['node', 'role', 'feeds', 'delay']).to_csv(partition_file, index=False)


def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, cores,
                   metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
                   ccm_sample, ccm_slope,
                   target_ranking_window_rows=1000, min_pred_events=20, max_window_rows=30000,
                   ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                   observable_columns=None, nested_n=None, nested_m=None,
                   embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
                   lag_set=None, stage2_allow_real_observables=False, tp=1,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15,
                   reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                   data_file=None, data_file_sha256=None, code_fingerprint=None,
                   gmn_mode=None, multi_mde_e=None, multi_mde_target_aggregate=None,
                   external_input_mode='real_data', progress_log_path=None,
                   mp_start_method='fork', use_gpu=False):
    '''Build one network spanning `targets` and their driver candidates, via
    one of two unrelated constructions depending on `mode` -- same
    mode-branch restructuring as simple-mde-gmn.py's own build_network():
      - mode == 'MDE-GMN': two-stage nested-MDE, both stages unified into
        build_nested_mde_network()'s own BFS, no apps.CreateNetwork() call
        -- see simple-mde-gmn.py's own build_network() docstring for the
        full rationale (identical here). `observable_columns` is already
        the CCM-pre-filtered pool -- main() runs the pre-filter once per
        target_groups_for_mode() group before this function is ever
        called, so this function no longer runs it itself.
        observable_columns/nested_n/nested_m/embed_dim_rho_min/
        cross_map_rho_min/lag_set are required in this mode.
      - mode == 'GMN': the interaction matrix via
        cleaned_helper_functions.proper_ccm (build_interaction_matrix()) --
        TSLS's own existing replacement for apps.InteractionMatrix's CLI
        subprocess, unaffected by any of the MDE-GMN-side two-stage rework
        -- then a node-dimension estimate and apps.CreateNetwork with its
        driver-ranking-bug workaround, same as before.
    '''
    network_id = '_'.join(targets)
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    ccm_E = ccm_tau = None
    node_dimensions = {}

    fingerprint = None
    cache_filename = f'network_cache_{network_id}.pkl'
    cache_hit = None
    if reuse_cache:
        fingerprint = build_network_fingerprint(
            mode, targets, observable_columns, nested_n, nested_m, tp,
            embed_dim_rho_min, cross_map_rho_min, exclusion_radius, lag_set,
            e_range, tau_range, prediction_start, prediction_length,
            metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint,
            stage2_allow_real_observables=stage2_allow_real_observables,
            gmn_mode=gmn_mode, multi_mde_e=multi_mde_e,
            multi_mde_target_aggregate=multi_mde_target_aggregate,
            target_ranking_window_rows=target_ranking_window_rows,
            min_pred_events=min_pred_events, max_window_rows=max_window_rows,
            ccm_lib_sizes_rows=ccm_lib_sizes_rows,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint, load_one=_load_pickle_cache_entry)
        if hit is not None:
            cache_hit, source = hit
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached network {source} '
                  f'-- skipping network construction entirely', flush=True)
        else:
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached network '
                  f'found under {prev_runs_dir} -- building from scratch.', flush=True)

    if mode == 'MDE-GMN':
        if cache_hit is not None:
            graph = cache_hit['graph']
        else:
            # `observable_columns` is already the CCM-pre-filtered pool --
            # main() runs ccm_prefilter_observables() once per
            # target_groups_for_mode() group before this function is ever
            # called, so depth-0 itself only ever searches already-
            # filtered observables too (see main()'s own depth-0
            # docstring for why the pre-filter had to move there rather
            # than staying here). Both stages, fully serial (Stage 1 runs
            # to completion, then Stage 2), in one call -- see
            # build_nested_mde_network()'s own docstring (revised
            # 2026-08-27, R12: Stage 2's own real-observable selections
            # still recurse exactly like Stage 1's, just after Stage 1
            # completes rather than interleaved with it wave-by-wave).
            graph, _, _ = build_nested_mde_network(
                targets, df, observable_columns, observable_columns,
                prediction_start, prediction_length,
                nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                results_dir, mde_results, [], tp,
                metric, target_metric, gamma_delta,
                target_ranking_window_rows, min_pred_events, max_window_rows,
                ccm_lib_sizes_rows, ccm_sample, ccm_slope, cores,
                lag_set, stage2_allow_real_observables,
                reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                code_fingerprint,
                progress_log_path=progress_log_path,
                mp_start_method=mp_start_method,
                use_gpu=use_gpu,
            )
        graph.graph['self_delay_set'] = sorted(set(lag_set) | {0}, reverse=True)
        partition_network(graph, targets, external_input_mode)
        report_partition(graph, network_id, results_dir)
        selected_union = [n for n in graph.nodes if n not in targets]
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
        .iloc[:len(df)]
        .reset_index(drop=True)
    )
    data_after_mde.insert(0, 'Time', np.arange(len(data_after_mde)))
    data_after_mde_file = results_dir / f'data_after_mde_{network_id}.csv'
    data_after_mde.to_csv(data_after_mde_file, index=False)

    data_after_mde_lib = data_after_mde.iloc[:prediction_start]
    data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
    data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)

    if mode == 'MDE-GMN':
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
            gmn_construction_start = time.perf_counter()
            _log_progress(progress_log_path, f'{network_id}: GMN network construction START')
            e_vals, tau_vals = list(e_range), list(tau_range)
            # target_E_tau_rho_grid always scores a TARGETS column's own
            # self-predictability -- always target_metric (continuous),
            # never metric.
            target_grids = [
                target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, target_metric, gamma_delta)
                for t in targets
            ]
            mean_grid = np.nanmean(np.stack(target_grids), axis=0)
            best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
            ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
            print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
                  f'(grid-search {target_metric}={mean_grid[best_i, best_j]:.3f})', flush=True)

            nodes = [c for c in data_after_mde_lib.columns if c != 'Time']
            iMatrix = build_interaction_matrix(
                data_after_mde_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
                gamma_delta, ccm_window_rows, ccm_lib_size_pcts, ccm_sample, ccm_slope,
            )
            interaction_csv = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}_CCM.csv'
            iMatrix.to_csv(interaction_csv)

            node_dimensions = {
                node: estimate_node_dimension(
                    data_after_mde_lib, node, _metric_for(node, targets, metric, target_metric),
                    gamma_delta, max_e=max_node_dimension, fallback=fallback_node_dimension,
                )
                for node in iMatrix.index
            }
            print(f'{network_id}: node N_D =', node_dimensions, flush=True)

            # apps.CreateNetwork() keeps each node's *weakest* above-
            # threshold drivers instead of the strongest -- worked around
            # by pre-selecting the strongest N_D drivers ourselves, then
            # calling CreateNetwork with numDrivers set to the full column
            # count so its buggy truncation never has anything left to
            # remove.
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
            _log_progress(
                progress_log_path,
                f'{network_id}: GMN network construction END -- '
                f'{time.perf_counter() - gmn_construction_start:.1f}s total')
        else:
            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'

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
    depends on them. Edges are colored by delay, with a legend mapping
    color -> delay, whenever the graph carries per-edge delay data (a
    MultiDiGraph built by build_nested_mde_network() for MODE='MDE-GMN').
    MODE='GMN' networks (a plain DiGraph from apps.CreateNetwork, no delay
    concept at all) fall back to the original by-target edge coloring.
    Ported unchanged from simple-mde-gmn.py -- purely graph-structural, no
    dependency on what the columns represent.'''
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


def plot_3d_network(net_result, target_colors, save_path):
    '''Interactive HTML redesign of the 3D network figure (superseding the
    earlier static-PNG version this docstring used to describe). MODE=
    'MDE-GMN' only (needs the per-edge delay data and interior/external
    partition only that mode's MultiDiGraph has, both absent from MODE=
    'GMN''s plain DiGraph).

    x-y plane (z=0): only the INTERIOR nodes (>=1 discovered driver --
    genuine Stage 1/2 network members, partition_network()'s own
    classification read from graph.graph, not re-derived here), laid out
    via a spring layout of the real-observable (delay=0) edges between two
    interior nodes -- the same structural skeleton draw_colored_network()
    draws in 2D, just restricted to interior nodes since external inputs
    are handled entirely separately below.

    Everything that drives an interior node but has no drivers of its own
    is lifted onto the z-axis as its own distinct "entity" -- never stacked
    directly above whichever node it happens to feed (the old design's
    approach, which duplicated a driver once per consumer). One entity per
    distinct (driver, delay!=0) pair (a lagged copy of an observable -- this
    covers both a genuine external input's lagged edge and an interior
    node's own AR self-loop, since a lagged copy is conceptually distinct
    from that observable's own real-time position either way), labeled
    f"{driver}_delay_{delay}"; plus one entity per external-input driver
    whose edge has delay==0 (a real-observable leaf), labeled with its own
    plain name. Each entity gets a single (x, y) position solved by a second
    spring-layout pass with the interior positions held fixed (see
    _entity_layout()), so a single driver feeding several nodes appears once
    with arrows fanning out to each of them, rather than duplicated per
    consumer. z = |delay| for a lagged-copy entity, or a small fixed
    elevation for a delay==0 leaf entity (enough to read as "off the plane"
    without implying a false delay magnitude).

    Every graph edge becomes one directed arrow (a plotly Cone trace,
    plotly's own 3D substitute for a directed arrowhead -- there is no 3D
    analog of draw_networkx_edges()'s 2D arrows) between its resolved
    driver position and its driven node's plane position, colored by delay
    using the same viridis mapping draw_colored_network() uses so the two
    figures read consistently. Nodes are drawn as three marker shapes --
    circle (interior), diamond (real-observable leaf entity), square
    (lagged-copy entity, colored by delay rather than by target) -- the
    same 3-way distinction the old design drew via circle/triangle/square
    (plotly's 3D markers have no triangle symbol; diamond substitutes).
    Rendered via plotly rather than matplotlib specifically for the
    interactivity (rotate/zoom/hover) a static image can't offer.'''
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    def _color(c):
        '''Normalizes any color target_colors/our own fallbacks/viridis's
        cmap() might hand us -- a matplotlib named string ('tab:red'), a raw
        RGBA float tuple (what cfg['TARGET_COLORS'] actually contains in
        production, via plt.get_cmap('tab10')(i % 10)), or an already-hex
        string -- into a plotly-safe '#rrggbb' hex string. Plotly's color
        validators reject bare RGBA tuples (mistaken for numeric colorscale
        data) and non-CSS names like 'tab:red'.'''
        return mcolors.to_hex(mcolors.to_rgba(c))

    graph = net_result['network']['Graph']
    interior_nodes = graph.graph.get('interior_nodes', set())

    all_delays = sorted({d['delay'] for _, _, d in graph.edges(data=True)})
    cmap = plt.get_cmap('viridis')
    delay_color = {
        d: _color(cmap(i / max(1, len(all_delays) - 1)))
        for i, d in enumerate(all_delays)
    }
    nonzero_abs_delays = sorted({abs(d) for d in all_delays if d != 0})
    leaf_z = nonzero_abs_delays[0] * 0.5 if nonzero_abs_delays else 1.0

    # ---- pass 1: interior-only real-observable backbone layout (z=0) ----
    interior_real_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d['delay'] == 0 and u in interior_nodes and v in interior_nodes
    ]
    backbone = nx.DiGraph()
    backbone.add_nodes_from(interior_nodes)
    backbone.add_edges_from(interior_real_edges)
    interior_pos = nx.spring_layout(backbone, seed=0) if interior_nodes else {}

    # ---- classify every edge into "already on the plane" (interior ->
    # interior, delay==0) or "driven by an elevated entity" ----
    entity_meta = {}     # label -> {'driver':.., 'delay':.., 'kind': 'leaf'|'delay_copy'}
    entity_out_edges = []  # (label, driven_node) -- for the pass-2 layout graph
    for u, v, data in graph.edges(data=True):
        delay = data['delay']
        if delay == 0 and u in interior_nodes:
            continue  # part of the pass-1 backbone
        if delay == 0:
            # u is necessarily an external input here -- external inputs
            # have in_degree 0 by partition_network()'s own definition, so
            # they can never be a driven node v, and any interior u with
            # delay==0 was already excluded just above.
            label, kind = u, 'leaf'
        else:
            label, kind = f'{u}_delay_{delay}', 'delay_copy'
        entity_meta[label] = {'driver': u, 'delay': delay, 'kind': kind}
        entity_out_edges.append((label, v))

    # ---- pass 2: constrained layout -- interior positions held fixed,
    # solve (x, y) only for the new elevated entities. spring_layout
    # auto-seeds any node present in `combined` but absent from `pos=` at a
    # random position scaled to the *given* positions' own magnitude, so no
    # manual initial guess is needed for the new entity nodes. ----
    combined = nx.DiGraph()
    combined.add_nodes_from(interior_nodes)
    combined.add_edges_from(entity_out_edges)
    if interior_nodes:
        solved_pos = nx.spring_layout(
            combined, pos=dict(interior_pos), fixed=list(interior_nodes), seed=0,
        )
    else:
        solved_pos = nx.spring_layout(combined, seed=0) if combined.nodes else {}

    pos2d = dict(interior_pos)
    for label in entity_meta:
        pos2d[label] = solved_pos.get(label, (0.0, 0.0))

    def _xyz(name):
        x, y = pos2d[name]
        if name in entity_meta:
            m = entity_meta[name]
            z = abs(m['delay']) if m['kind'] == 'delay_copy' else leaf_z
        else:
            z = 0.0
        return float(x), float(y), float(z)

    def _src_label(u, delay):
        if delay == 0 and u in interior_nodes:
            return u
        if delay == 0:
            return u
        return f'{u}_delay_{delay}'

    # ---- edge traces, grouped by delay (line + arrowhead cone per group) ----
    edges_by_delay = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        delay = data['delay']
        edges_by_delay[delay].append((_xyz(_src_label(u, delay)), _xyz(v)))

    fig = go.Figure()
    for delay, segs in sorted(edges_by_delay.items()):
        color = delay_color[delay]
        xs, ys, zs = [], [], []
        for (x0, y0, z0), (x1, y1, z1) in segs:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=color, width=3),
            name=f'delay={delay}', legendgroup=f'delay_{delay}',
            hoverinfo='none', opacity=0.7,
        ))

        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
        for (x0, y0, z0), (x1, y1, z1) in segs:
            tx, ty, tz = x1 - x0, y1 - y0, z1 - z0
            cx.append(x0 + 0.88 * tx); cy.append(y0 + 0.88 * ty); cz.append(z0 + 0.88 * tz)
            cu.append(tx); cv.append(ty); cw.append(tz)
        fig.add_trace(go.Cone(
            x=cx, y=cy, z=cz, u=cu, v=cv, w=cw,
            anchor='tail', sizemode='absolute', sizeref=0.15,
            colorscale=[[0, color], [1, color]], showscale=False,
            legendgroup=f'delay_{delay}', showlegend=False, hoverinfo='none',
        ))

    # ---- node traces: interior (circle), leaf (diamond), delay-copy (square) ----
    def _node_trace(names, symbol, colors, size, label):
        if not names:
            return None
        xs, ys, zs = zip(*(_xyz(n) for n in names))
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers+text', text=list(names),
            textposition='top center', textfont=dict(size=9),
            marker=dict(symbol=symbol, size=size, color=list(colors),
                        line=dict(color='black', width=1)),
            name=label,
        )

    interior_list = sorted(interior_nodes)
    leaf_list = sorted(n for n, m in entity_meta.items() if m['kind'] == 'leaf')
    delay_copy_list = sorted(n for n, m in entity_meta.items() if m['kind'] == 'delay_copy')

    for t in (
        _node_trace(interior_list, 'circle',
                    [_color(target_colors.get(n, 'tab:blue')) for n in interior_list],
                    8, f'interior (n={len(interior_list)})'),
        _node_trace(leaf_list, 'diamond',
                    [_color(target_colors.get(entity_meta[n]['driver'], 'tab:orange'))
                     for n in leaf_list],
                    7, f'external input, real-time (n={len(leaf_list)})'),
        _node_trace(delay_copy_list, 'square',
                    [delay_color[entity_meta[n]['delay']] for n in delay_copy_list],
                    6, f'lagged driver copy (n={len(delay_copy_list)})'),
    ):
        if t is not None:
            fig.add_trace(t)

    fig.update_layout(
        scene=dict(
            xaxis_title='X (interior network layout)',
            yaxis_title='Y (interior network layout)',
            zaxis_title='|delay| (0 = real-time leaf elevation)',
        ),
        title=f"3D network projection ({net_result['network_id']})",
        legend_title=f"n_interior={len(interior_list)}, "
                     f"n_external_leaf={len(leaf_list)}, n_lagged_copy={len(delay_copy_list)}",
    )
    fig.write_html(save_path)


# =============================================================================
# GMN node generation config (mirrors notebook cell 8) -- ported UNCHANGED
# from simple-mde-gmn.py. Pure gmn/.cfg-file mechanics, no dependency on
# what the embedded columns represent.
# =============================================================================

def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                      mode='Generate', function='SMap', theta=3,
                      node_theta_overrides=None, variant='', generate_length=None,
                      generate_start=None, max_offset=None, exclusion_radius=0):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    exclusion_radius: written into the [EDM] section below and read
    straight through by gmn's own (monkeypatched) Node.Forecast/Generate as
    `Parameters.exclusionRadius`, passed to every pyEDM Simplex/SMap call
    gmn itself makes -- the same Theiler-window neighbor-exclusion
    mechanism as TSLS_MDE_GMN_EXCLUSION_RADIUS (this pipeline's MDE-side
    parameter), just applied to GMN's own recursive Generate()/Forecast()
    nearest-neighbor search rather than MDE's forward selection -- two
    independent EDM contexts with their own separate embeddings, so there's
    no single shared value that would make sense for both. This one is a
    plain static knob, not dynamically recomputed the way MDE's 'auto' mode
    would be (that mode has no equivalent here regardless -- see this
    module's own docstring).

    E/tau are no longer parameters here -- every node's real embedding now
    comes from _node_embedding_spec()/_embedding_spec_for() (per-driver
    delays off the network graph, plus self_delay_set or
    fallback_E/fallback_tau), not a shared network-wide (E, tau) written
    into this .cfg. Placeholder 'E'/'tau' values are still written into the
    [EDM] section purely because gmn.Node.__init__'s own (unpatched)
    arithmetic reads them before _generalized_node_init immediately
    corrects self.libEnd_i; their actual values are otherwise inert.

    generate_start (both modes): row index where Generate() starts
    free-running / where Forecast()'s pred window starts, independent of
    prediction_start -- the latter is the fixed MDE lib/pred boundary that
    MDE variable selection and the CCM/embedDimRhoMin/crossMapRhoMin gates
    validate against, and stays fixed regardless of what row a diagnostic
    Generate()/Forecast() call happens to start at. Defaults to
    prediction_start when None.

    generate_length (both modes): number of steps simulated/predicted from
    generate_start, independent of prediction_length. A full Generate()
    call over the whole held-out window can be the single most expensive
    step in the search loop, so decoupling this lets you shrink it without
    also shrinking the held-out window MDE/CCM validate against. Defaults
    to prediction_length when None. mode='Forecast' honors this same cap so
    its window matches Generate()'s -- run_forecast_diagnostic() plots both
    on the same axes.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting, so it can't
    do real Takens delay embedding. gmn's SMap branch does call
    Embed(E, tau, columns) first (now replaced by this pipeline's own
    generalized embedding).

    node_theta_overrides: optional {nodeName: theta} dict -- written as
    per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
    lookup); every other node uses the network-wide `theta`. The trailing
    '/' on configPath below works around gmn/Node.py:78's naive string
    concatenation path lookup.

    max_offset (mode='Forecast' only): the network's largest per-node
    embedding offset (network_max_embedding_offset()), used to size a
    single, network-wide lib/pred window valid for every node's own
    embedding.'''
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
        forecast_length = generate_length if generate_length is not None else prediction_length
        if offset >= generate_start or offset >= forecast_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'leaves too few rows in generate_start={generate_start}/'
                f'forecast_length={forecast_length} for a valid lib/pred split '
                f'-- shrink TSLS_MDE_GMN_DELAY_SET or grow the prediction window.'
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
# Cross-job caching / fingerprint architecture -- ported from
# simple-mde-gmn.py; see that file's own module docstring's SEARCH REUSE
# section for the full rationale. Fingerprint functions fold in this
# pipeline's own TSLS-specific cfg knobs (METRIC/TARGET_METRIC/GAMMA_DELTA/
# CCM_WINDOW_ROWS/CCM_LIB_SIZE_PCTS/CCM_SAMPLE/CCM_SLOPE) alongside the
# fields simple-mde-gmn.py already fingerprints.
# =============================================================================

def _hash_file(path, chunk_size=1 << 20):
    '''sha256 of a file's actual bytes -- ported unchanged from
    simple-mde-gmn.py.'''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def _strip_docstrings(tree):
    '''Removes every module/class/function docstring from an AST in place
    -- ported unchanged from simple-mde-gmn.py.'''
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return tree


def _pipeline_code_fingerprint():
    '''SHA-256 of this file's own normalized AST (docstrings stripped) --
    ported unchanged from simple-mde-gmn.py, applied to this file.'''
    tree = ast.parse(Path(__file__).read_text())
    _strip_docstrings(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _environment_fingerprint():
    '''Captures the installed numerical/algorithmic dependencies this
    pipeline's cached stages actually run through. Unlike simple-mde-gmn.py:
    no `dimx` (not imported at all in this pipeline) and no
    `apps.InteractionMatrix` (MODE='GMN' here uses proper_ccm instead, no
    subprocess); `cleaned_helper_functions` (this pipeline's own TSLS-aware
    scoring engine, not an installed package) is content-hashed directly,
    same approach as `apps.CreateNetwork` (still used for MODE='GMN''s own
    network object).'''
    versions = {}
    for pkg in ('gmn', 'pyEDM', 'networkx', 'numpy', 'pandas'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    for mod_name in ('apps.CreateNetwork',):
        mod = importlib.import_module(mod_name)
        versions[mod_name] = _hash_file(Path(mod.__file__))
    versions['cleaned_helper_functions'] = _hash_file(Path(tsls.__file__))
    return versions


def build_code_fingerprint():
    '''Combines _pipeline_code_fingerprint() and _environment_fingerprint()
    into the single value every cache fingerprint embeds -- ported
    unchanged from simple-mde-gmn.py. Computed once per job in main()
    (gated by TSLS_MDE_GMN_REUSE_BEST_PARAMS) and threaded down as a plain
    argument rather than recomputed per call.'''
    return {'source': _pipeline_code_fingerprint(), 'environment': _environment_fingerprint()}


def build_search_fingerprint(cfg, net_result, data_file_sha256, code_fingerprint):
    '''Everything that determines one network's theta search outcome
    (run_theta_search()) -- ported from simple-mde-gmn.py, with TSLS's own
    scoring/CCM-calibration knobs folded in (they shape the network's own
    topology/selection outcome just as much as MDE_D/NESTED_M/etc. do) and
    `exclusion_radius` simplified to the plain value directly (no 'auto'
    mode to normalize against here, unlike simple-mde-gmn.py).'''
    graph = net_result['network']['Graph']
    return {
        'mode': cfg['MODE'],
        'gmn_mode': cfg['GMN_MODE'],
        'targets': sorted(net_result['targets']),
        'mde_d': cfg['MDE_D'],
        'multi_mde_e': cfg['MULTI_MDE_E'],
        'multi_mde_target_aggregate': cfg['MULTI_MDE_TARGET_AGGREGATE'],
        'tp': cfg['TP'],
        'embed_dim_rho_min': cfg['EMBED_DIM_RHO_MIN'],
        'cross_map_rho_min': cfg['CROSS_MAP_RHO_MIN'],
        'exclusion_radius': cfg['EXCLUSION_RADIUS_VALUE'],
        'gmn_exclusion_radius': cfg['GMN_EXCLUSION_RADIUS'],
        'stage2_allow_real_observables': cfg['STAGE2_ALLOW_REAL_OBSERVABLES'],
        'nested_m': cfg['NESTED_M'],
        'region_channel_cap': cfg['REGION_CHANNEL_CAP'],
        'lib_ratio': cfg['LIB_RATIO'],
        'mde_validation_ratio': cfg['MDE_VALIDATION_RATIO'],
        'generate_length': cfg['GENERATE_LENGTH'],
        'lag_set': list(cfg['LAG_SET']),
        'e_vals': list(cfg['E_RANGE']),
        'tau_vals': list(cfg['TAU_RANGE']),
        'theta_vals': list(cfg['THETA_RANGE']),
        'metric': cfg['METRIC'],
        'target_metric': cfg['TARGET_METRIC'],
        'gamma_delta': cfg['GAMMA_DELTA'],
        'ccm_window_rows': cfg['CCM_WINDOW_ROWS'],
        'ccm_lib_size_pcts': list(cfg['CCM_LIB_SIZE_PCTS']),
        'ccm_sample': cfg['CCM_SAMPLE'],
        'ccm_slope': cfg['CCM_SLOPE'],
        'data_file': cfg['DATA_FILE'],
        'data_file_sha256': data_file_sha256,
        'prediction_start': cfg['PREDICTION_START'],
        'prediction_length': cfg['PREDICTION_LENGTH'],
        'network_nodes': sorted(graph.nodes),
        # R14: doesn't change topology (build_network_fingerprint()'s own
        # concern), but does change what Generate()/theta search actually
        # do with a zero-driver node -- belongs here, not there.
        'external_input_mode': cfg['EXTERNAL_INPUT_MODE'],
        'code_fingerprint': code_fingerprint,
        'stage': 'theta_search',
    }


def build_network_fingerprint(mode, targets, observable_columns, nested_n, nested_m, tp,
                               embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                               delay_set, e_range, tau_range, prediction_start,
                               prediction_length, metric, target_metric, gamma_delta,
                               ccm_window_rows, ccm_lib_size_pcts, ccm_sample, ccm_slope,
                               data_file, data_file_sha256, code_fingerprint,
                               stage2_allow_real_observables=False,
                               gmn_mode=None, multi_mde_e=None, multi_mde_target_aggregate=None,
                               target_ranking_window_rows=1000, min_pred_events=20,
                               max_window_rows=30000,
                               ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800)):
    '''Everything that determines one group's network-construction outcome
    -- ported from simple-mde-gmn.py, TSLS scoring/CCM-calibration knobs
    folded in. Computable *before* that construction runs, since the whole
    point is deciding whether to skip it. `observable_columns` here is
    already the CCM-pre-filtered pool (main() runs the pre-filter, and its
    own threshold/E/libSize-ratio knobs, before this fingerprint is ever
    computed -- see build_network()'s own docstring), so -- matching this
    function's own original design philosophy, predating the pre-filter --
    the resolved list alone captures everything a cache needs to
    distinguish, the same way it already does for REGION_CHANNEL_CAP; no
    separate pre-filter fields needed here.

    gmn_mode/multi_mde_e/multi_mde_target_aggregate: this function itself
    doesn't consume these (the depth-0 `mde_results` it's handed is already
    computed upstream in main() -- see run_multi_mde_for_targets()'s
    docstring), but they determine that upstream mde_results just as much
    as mde_d/tp/embed_dim_rho_min/etc. below do, so they must be part of
    this fingerprint too -- otherwise switching GMN_MODE (or its
    multi_proper_mde knobs) between two REUSE_BEST_PARAMS runs wouldn't
    invalidate a stale cached network built from the other mode's
    mde_results.

    `target_ranking_window_rows`/`min_pred_events`/`max_window_rows`/
    `ccm_lib_sizes_rows` (window-sizing fix, docs/review-findings.md):
    MODE='MDE-GMN' only -- `build_nested_mde_network()`'s own per-node
    calls now resolve their own ranking window from these, decoupled from
    `ccm_window_rows`/`ccm_lib_size_pcts` (which stay exactly what they
    always were, MODE='GMN'-only, via `build_interaction_matrix()`). Real
    topology inputs for MDE-GMN networks now, same reasoning as every
    other knob here.'''
    return {
        'mode': mode,
        'gmn_mode': gmn_mode,
        'multi_mde_e': multi_mde_e,
        'multi_mde_target_aggregate': multi_mde_target_aggregate,
        'targets': sorted(targets),
        'observable_columns': sorted(observable_columns),
        'mde_d': nested_n,
        'nested_m': nested_m,
        'tp': tp,
        'embed_dim_rho_min': embed_dim_rho_min,
        'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius,
        'delay_set': list(delay_set) if delay_set is not None else None,
        'stage2_allow_real_observables': stage2_allow_real_observables,
        'e_vals': list(e_range),
        'tau_vals': list(tau_range),
        'metric': metric,
        'target_metric': target_metric,
        'gamma_delta': gamma_delta,
        'ccm_window_rows': ccm_window_rows,
        'ccm_lib_size_pcts': list(ccm_lib_size_pcts),
        'target_ranking_window_rows': target_ranking_window_rows,
        'min_pred_events': min_pred_events,
        'max_window_rows': max_window_rows,
        'ccm_lib_sizes_rows': list(ccm_lib_sizes_rows),
        'ccm_sample': ccm_sample,
        'ccm_slope': ccm_slope,
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
                                     metric, target_metric, gamma_delta,
                                     is_behavioral_target, target_ranking_window_rows,
                                     min_pred_events, max_window_rows, window_rows, pred_event_count,
                                     ccm_lib_sizes_rows, ccm_sample, ccm_slope,
                                     data_file, data_file_sha256, code_fingerprint,
                                     cache_stage='stage1', max_e=None):
    '''Everything that determines one run_mde_for_target() call's outcome,
    computable before that call runs -- ported from simple-mde-gmn.py, TSLS
    scoring/CCM-calibration knobs folded in. `candidate_pool` is kept in
    order rather than sorted -- forward selection can tie-break on column
    order, so the same columns in a different order aren't guaranteed to
    select the same result. `max_e` (the raw MDE_D-derived auto-select
    ceiling proper_embed_dimension's own sweep is capped at, distinct from
    `mde_d`'s possibly num_delays-scaled value -- see run_mde_for_target's
    own docstring) is a real input to the outcome now that E is
    auto-selected rather than a hardcoded constant, so it must be part of
    the fingerprint too: a run with a different max_e can select a
    different E, hence different candidates, for the same everything else.

    Window-sizing fix (docs/review-findings.md): `window_rows`/
    `pred_event_count` are the RESOLVED, data-dependent outputs of
    _resolve_target_ranking_window() for this specific node -- not just the
    config-level knobs (`target_ranking_window_rows`/`min_pred_events`/
    `max_window_rows`) that produced them. All must be in the
    fingerprint (A5): the resolved window is itself a real input to this
    call's outcome, and a cache built before this fix (a different window,
    a different -- previously wrong -- target_metric for every neuron
    node) must not be silently reused.'''
    return {
        'node': node,
        'candidate_pool': list(candidate_pool),
        'mde_d': mde_d,
        'max_e': max_e,
        'tp': tp,
        'embed_dim_rho_min': embed_dim_rho_min,
        'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius,
        'delay_set': list(delay_set),
        'metric': metric,
        'target_metric': target_metric,
        'gamma_delta': gamma_delta,
        'is_behavioral_target': is_behavioral_target,
        'target_ranking_window_rows': target_ranking_window_rows,
        'min_pred_events': min_pred_events,
        'max_window_rows': max_window_rows,
        'window_rows': window_rows,
        'pred_event_count': pred_event_count,
        'ccm_lib_sizes_rows': list(ccm_lib_sizes_rows),
        'ccm_sample': ccm_sample,
        'ccm_slope': ccm_slope,
        'prediction_start': prediction_start,
        'prediction_length': prediction_length,
        'data_file': data_file,
        'data_file_sha256': data_file_sha256,
        'code_fingerprint': code_fingerprint,
        'stage': f'mde_selection_{cache_stage}',
    }


def build_multi_mde_selection_fingerprint(targets, candidate_pool, mde_d, tp, embed_dim_rho_min,
                                           cross_map_rho_min, exclusion_radius, delay_set,
                                           prediction_start, prediction_length,
                                           metric, target_metric, gamma_delta,
                                           target_ranking_window_rows, window_rows,
                                           ccm_lib_sizes_rows, ccm_sample, ccm_slope,
                                           data_file, data_file_sha256, code_fingerprint,
                                           cache_stage='stage1', E=3, target_aggregate='mean'):
    '''Multi-target counterpart of build_mde_selection_fingerprint() -- same
    fields, `targets` (list, kept in the group's own order -- see that
    function's own candidate_pool-ordering rationale, identical here) in
    place of a single `node`, plus `E`/`target_aggregate`
    (multi_proper_mde's own extra required/optional arguments, with no
    proper_mde equivalent -- see run_multi_mde_for_targets()'s docstring).
    `window_rows` (window-sizing fix, docs/review-findings.md): the
    resolved window itself, not just the `target_ranking_window_rows`
    constant that produced it -- kept as its own field for the same A5
    reason build_mde_selection_fingerprint() records both.'''
    return {
        'targets': list(targets),
        'candidate_pool': list(candidate_pool),
        'mde_d': mde_d,
        'E': E,
        'target_aggregate': target_aggregate,
        'tp': tp,
        'embed_dim_rho_min': embed_dim_rho_min,
        'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius,
        'delay_set': list(delay_set),
        'metric': metric,
        'target_metric': target_metric,
        'gamma_delta': gamma_delta,
        'target_ranking_window_rows': target_ranking_window_rows,
        'window_rows': window_rows,
        'ccm_lib_sizes_rows': list(ccm_lib_sizes_rows),
        'ccm_sample': ccm_sample,
        'ccm_slope': ccm_slope,
        'prediction_start': prediction_start,
        'prediction_length': prediction_length,
        'data_file': data_file,
        'data_file_sha256': data_file_sha256,
        'code_fingerprint': code_fingerprint,
        'stage': f'mde_selection_multi_{cache_stage}',
    }


def _load_json_cache_entry(path):
    with open(path) as f:
        return json.load(f)


def _load_pickle_cache_entry(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_cached(prev_runs_dir, filename, fingerprint, load_one=_load_json_cache_entry):
    '''Generic cross-job cache lookup -- ported unchanged from
    simple-mde-gmn.py, shared by every cacheable stage.'''
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
    '''Ported unchanged from simple-mde-gmn.py -- writes once per unique
    directory in {results_dir, run_cache_dir}, deduplicated.'''
    written = []
    for directory in dict.fromkeys([results_dir, run_cache_dir]):
        path = directory / filename
        write_one(path)
        written.append(path)
    return written[0]


def save_json_to_cache(filename, payload, results_dir, run_cache_dir):
    def write_one(path):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)


def save_pickle_to_cache(filename, payload, results_dir, run_cache_dir):
    def write_one(path):
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)


def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
    hit = find_cached(prev_runs_dir, f'best_params_{network_id}.json', fingerprint)
    if hit is None:
        return None
    entry, path = hit
    return entry['best_node_theta'], path


def save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta, source):
    return save_json_to_cache(
        f'best_params_{network_id}.json',
        {'fingerprint': fingerprint, 'best_node_theta': best_node_theta, 'source': source},
        results_dir, run_cache_dir,
    )


# =============================================================================
# Theta search (mirrors notebook cell 10) -- replaces this pipeline's own
# previous SMAP_SEARCH_MODE='coarse_retune'/'joint' combined (E, tau, theta)
# grid search entirely. E and every driver's delay now come from the
# network itself (see build_network()'s docstring), leaving theta as the
# only thing still searched per node -- same reduction in search cost as
# simple-mde-gmn.py's own equivalent change.
# =============================================================================

def proper_predict_nonlinear(dataFrame, columns, target, theta, lib, pred,
                              E=1, Tp=1, knn=0, tau=-1, exclusionRadius=0,
                              embedded=False, metric='correlation', gamma_delta=2):
    '''R20 (docs/review-actions-v1.md): TSLS-aware replacement for
    pyEDM.PredictNonlinear's own theta-selection sweep -- NOT a new S-map
    engine. R10 found no native S-map in cleaned_helper_functions.py, but
    reading pyEDM's own source (pyEDM/API.py's PredictNonlinear;
    pyEDM/PoolFunc.py's PredictNLSMapFunc worker) shows PredictNonlinear
    already does nothing more than call pyEDM.SMap(...) once per
    candidate theta and score it via `ComputeError(obs, pred)['rho']` --
    the SAME compiled S-map gmn's own Forecast()/Generate() already calls
    directly at runtime for every node (unaffected by this function
    either way -- theta is just a number once chosen, regardless of which
    sweep chose it). `ComputeError`'s own `rho = corrcoef(obs, pred)[0,1]`
    is the identical formula `cleaned_helper_functions._score(kind=
    'correlation')` uses, so this function reproduces PredictNonlinear's
    own selection exactly at metric='correlation' (see
    tests/test_r20_predict_nonlinear.py's own equivalence check) while
    also supporting 'f1'/'gamma'/'mae' for non-target (neuron) nodes,
    which PredictNonlinear itself can never score.

    Deliberately sequential -- one pyEDM.SMap call per theta, in-process,
    no multiprocessing.Pool. See sitecustomize.py's own module docstring
    (this venv) for why: pyEDM's own default PredictNonlinear/
    EmbedDimension/PredictInterval each spawn a *fresh* Pool per call,
    which is what turned this pipeline's jobs into a ~2-day-regardless-
    of-config bug before that file patched all three to a sequential
    drop-in. This function must not reintroduce that mistake -- theta
    grids here are ~10-20 values, cheap enough sequentially that adding
    pooling machinery back in isn't warranted.

    Returns a DataFrame with columns 'theta'/'score' (one row per
    candidate) -- callers pick `df.loc[df['score'].idxmax(), 'theta']`,
    mirroring PredictNonlinear's own `df.loc[df['rho'].idxmax(), 'theta']`
    idiom exactly, just against a metric-agnostic column name instead of
    a hardcoded 'rho'.'''
    rows = []
    for th in theta:
        S = SMap(
            dataFrame=dataFrame, columns=columns, target=target,
            lib=lib, pred=pred, E=E, Tp=Tp, knn=knn, tau=tau,
            theta=th, exclusionRadius=exclusionRadius,
            embedded=embedded, showPlot=False,
        )
        predictions = S['predictions']
        obs = predictions['Observations'].to_numpy()
        prd = predictions['Predictions'].to_numpy()
        rows.append({'theta': th, 'score': tsls._score(obs, prd, metric, gammaDelta=gamma_delta)})
    return pd.DataFrame(rows)


def retune_theta_for_all_nodes(data_lib, graph, nodes, targets, theta_range, cores,
                                metric, target_metric, gamma_delta, min_pred_rows=20,
                                progress_log_path=None):
    '''Finds each node's own best theta via proper_predict_nonlinear() (R20,
    docs/review-actions-v1.md) on a 70/30 lib/pred split of that node's own
    generalized embedding. Previously scored via pyEDM.PredictNonlinear's
    own internal Pearson rho regardless of METRIC -- the one stage in this
    pipeline that didn't route through _metric_for() (confirmed against
    source before this change, per A1). Now scores each node via
    _metric_for(node, targets, metric, target_metric) -- 'f1' for neuron
    nodes, 'correlation' (target_metric) for TARGETS, matching every
    other scoring call site in this file. `cores` is no longer used here
    (proper_predict_nonlinear() is deliberately sequential -- see its own
    docstring) but is kept as a parameter for call-site compatibility with
    run_theta_search(), which still needs it for evaluate_smap_candidate()'s
    own Generate() call right after this one. Falls back to theta=3
    whenever there's too little data for the split or the sweep itself
    fails (unchanged from before).'''
    best_theta_by_node = {}
    theta_durations = []
    _log_progress(progress_log_path, f'theta retuning START -- {len(nodes)} node(s)')
    theta_stage_start = time.perf_counter()

    for node in nodes:
        node_start = time.perf_counter()
        spec = _embedding_spec_for(graph, node)
        embedded, offset = _generalized_embed(data_lib, spec)
        embedded.insert(0, 'Time', np.arange(len(embedded)))
        embedded[node] = data_lib[node].to_numpy()[offset:]

        n_emb = len(embedded)
        split = int(n_emb * 0.7)
        n_pred = n_emb - split
        if n_pred < min_pred_rows:
            best_theta_by_node[node] = 3
            theta_durations.append(time.perf_counter() - node_start)
            continue

        lib_range = f'1 {split}'
        pred_range = f'{split + 1} {n_emb}'
        node_metric = _metric_for(node, targets, metric, target_metric)

        try:
            pn = proper_predict_nonlinear(
                embedded, columns=list(embedded.columns[1:-1]), target=node,
                theta=list(theta_range), lib=lib_range, pred=pred_range,
                Tp=1, embedded=True, metric=node_metric, gamma_delta=gamma_delta,
            )
            best_theta_by_node[node] = float(pn.loc[pn['score'].idxmax(), 'theta'])
        except Exception:
            best_theta_by_node[node] = 3
        node_duration = time.perf_counter() - node_start
        theta_durations.append(node_duration)
        eta = _format_eta(theta_durations, len(nodes) - len(theta_durations))
        _log_progress(progress_log_path,
                      f'theta node END -- {node} duration={node_duration:.1f}s '
                      f'theta={best_theta_by_node[node]}{eta}')

    _log_progress(progress_log_path,
                  f'theta retuning END -- {len(nodes)} node(s), '
                  f'{time.perf_counter() - theta_stage_start:.1f}s total')
    return best_theta_by_node


def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
                             node_theta, variant, results_dir, cores,
                             target_metric, gamma_delta,
                             generate_length=None, generate_start=None, exclusion_radius=0,
                             generate_validation_start=None):
    '''Runs one Generate() call and scores it as mean rho, across
    `net_result['targets']` only, between the simulated and real held-out
    series -- ported from simple-mde-gmn.py, scored via
    tsls._score(metric=target_metric) instead of pyEDM's plain
    ComputeError rho (TARGETS are continuous, so target_metric is always
    'correlation' by default -- see load_config()'s TARGET_METRIC
    docstring). This is "generative skill" as this pipeline defines it
    everywhere -- run_theta_search() and sweep_generate_start() both call
    this same function, at the same objective, so a search over either axis
    is comparable to the other.

    generate_start defaults to generate_validation_start (== cfg's B, the
    start of the Generate/Forecast validation segment), NOT prediction_start
    (== A) -- a Generate() call scored here must never launch from inside
    the segment MDE's own driver selection already validated against.'''
    generate_start = generate_start if generate_start is not None else generate_validation_start
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
        target_rhos[target] = tsls._score(actual_vals, gen_vals, target_metric, gammaDelta=gamma_delta)

    score = float(np.nanmean(list(target_rhos.values())))
    return score, target_rhos, G


def run_theta_search(gmn_runs, prediction_start, prediction_length, theta_range, results_dir,
                      cores, metric, target_metric, gamma_delta,
                      generate_length=None, reuse_best_params=False, prev_runs_dir=None,
                      run_cache_dir=None, fingerprints=None, exclusion_radius=0,
                      generate_validation_start=None, progress_log_path=None):
    '''Retunes each network's per-node theta once, then scores the result
    with a single Generate() call -- ported from simple-mde-gmn.py. E and
    each driver's delay are no longer search parameters at all: for
    MODE='MDE-GMN' they come directly from nested-MDE's own discovered
    network; for MODE='GMN' they still come from that mode's own CCM E/tau
    grid search (fallback_E/fallback_tau, set in build_network()).'''
    theta_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        # R14: exclude real-data-mode external inputs -- they have no
        # embedding to retune theta for (see _simulated_nodes()'s docstring).
        all_nodes = _simulated_nodes(graph)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        fingerprint = fingerprints[network_id] if fingerprints else None
        if reuse_best_params and fingerprint is not None:
            cached = find_cached_best_params(prev_runs_dir, network_id, fingerprint)
            if cached is not None:
                best_node_theta, source = cached
                print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched {source} -- '
                      f'skipping theta retuning entirely', flush=True)
                theta_search_results[network_id] = {
                    'best_node_theta': best_node_theta, 'reused_from': str(source),
                }
                save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                                  source=f'reused_from:{source}')
                continue
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
                  f'found under {prev_runs_dir} -- retuning theta.', flush=True)

        print(f'{network_id}: retuning theta for {len(all_nodes)} node(s)', flush=True)
        best_node_theta = retune_theta_for_all_nodes(
            data_after_mde_lib, graph, all_nodes, net_result['targets'], theta_range, cores,
            metric, target_metric, gamma_delta,
            progress_log_path=progress_log_path,
        )

        try:
            score, _, _ = evaluate_smap_candidate(
                net_result, prediction_start, prediction_length, best_node_theta,
                'theta_search', results_dir, cores, target_metric, gamma_delta,
                generate_length=generate_length,
                exclusion_radius=exclusion_radius,
                generate_validation_start=generate_validation_start,
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
# Generate()/Forecast() start-point sweep -- ported UNCHANGED (structurally)
# from simple-mde-gmn.py. Doesn't exist in this pipeline's own previous
# architecture at all.
# =============================================================================

def build_start_sweep_fingerprint(network_fingerprint, node_theta, start_offsets,
                                   generate_length, exclusion_radius):
    '''Everything that determines one network's start-point sweep outcome
    -- ported unchanged from simple-mde-gmn.py. network_fingerprint already
    embeds every TSLS-specific scoring/CCM-calibration knob (via
    build_search_fingerprint()), so nothing TSLS-specific needs adding
    here.'''
    return {
        'network_fingerprint': network_fingerprint,
        'node_theta': {k: node_theta[k] for k in sorted(node_theta)},
        'start_offsets': list(start_offsets),
        'generate_length': generate_length,
        'exclusion_radius': exclusion_radius,
        'stage': 'start_sweep',
    }


def sweep_generate_start(gmn_runs, theta_search_results, prediction_start, prediction_length,
                          results_dir, cores, generate_length, start_offsets,
                          target_metric, gamma_delta, exclusion_radius=0,
                          reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                          fingerprints=None, generate_validation_start=None,
                          progress_log_path=None):
    '''Finds, per network, which row Generate() should start free-running
    from to best match the real held-out data -- ported from
    simple-mde-gmn.py, scored via the same TSLS-aware evaluate_smap_candidate()
    the theta search uses. Returns {network_id: best_generate_start_row}.'''
    best_starts = {}
    for network_id, run in gmn_runs.items():
        if len(start_offsets) == 1:
            best_starts[network_id] = generate_validation_start + start_offsets[0]
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
                best_starts[network_id] = generate_validation_start + best_offset
                print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached start sweep '
                      f'{source} -- skipping sweep entirely (best offset {best_offset}, '
                      f'score={scores.get(best_offset, float("nan")):.4f})', flush=True)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
                ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
                ax.set_xlabel('Start offset from GENERATE_VALIDATION_START (rows)')
                ax.set_ylabel('Generate() rho (mean over targets)')
                ax.set_title(f'Generate() start-point sweep -- {network_id} (cached)')
                fig.tight_layout()
                fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
                plt.close(fig)
                continue
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached start '
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

            generate_start = generate_validation_start + offset
            try:
                score, _, _ = evaluate_smap_candidate(
                    net_result, prediction_start, prediction_length,
                    node_theta, f'startsweep_offset{offset}', results_dir, cores,
                    target_metric, gamma_delta,
                    generate_length=generate_length, generate_start=generate_start,
                    exclusion_radius=exclusion_radius,
                )
            except Exception as exc:
                print(f'{network_id}: start offset {offset} FAILED: {exc!r} -- scoring as NaN', flush=True)
                score = np.nan

            scores[offset] = score
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'start_offsets': start_offsets,
                             'generate_length': generate_length, 'scores': scores}, f)
            print(f'{network_id}: start offset {offset:5d} (row {generate_start}) score={score:.4f}', flush=True)
            _log_progress(progress_log_path,
                          f'{network_id}: generate-start offset {offset} done '
                          f'({len(scores)}/{len(start_offsets)}), score={score:.4f}')

        valid_offsets = [o for o in start_offsets if not np.isnan(scores.get(o, np.nan))]
        best_offset = max(valid_offsets, key=lambda o: scores[o]) if valid_offsets else start_offsets[0]
        best_starts[network_id] = generate_validation_start + best_offset
        print(f'{network_id}: best generate start = row {best_starts[network_id]} '
              f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)

        if sweep_fingerprint is not None:
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
        ax.set_xlabel('Start offset from GENERATE_VALIDATION_START (rows)')
        ax.set_ylabel('Generate() rho (mean over targets)')
        ax.set_title(f'Generate() start-point sweep -- {network_id}')
        fig.tight_layout()
        fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
        plt.close(fig)

    return best_starts


# =============================================================================
# Final rebuild + Forecast() diagnostic + fidelity metric
# (mirrors notebook cells 11-13) -- ported from simple-mde-gmn.py, scored
# throughout via _metric_for()/tsls._score() instead of plain ComputeError.
# =============================================================================

def rebuild_at_best_params(gmn_runs, theta_search_results, prediction_start, prediction_length,
                            results_dir, cores, metric, target_metric, gamma_delta,
                            generate_length=None, generate_starts=None,
                            exclusion_radius=0, generate_validation_start=None,
                            progress_log_path=None):
    '''Reruns Generate() at the winning per-node theta found by
    run_theta_search(), and the winning start row found by
    sweep_generate_start(). Ported from simple-mde-gmn.py -- try/except-
    guarded per network: a failure here is logged and that network is
    skipped (run['G'] stays unset) rather than crashing the whole job.

    R11 (docs/review-actions-v1.md): the search OBJECTIVE stays
    target-only -- confirmed evaluate_smap_candidate()'s own `score` (what
    both run_theta_search() and sweep_generate_start() actually optimize)
    is already a mean over `net_result['targets']` only, via
    `target_metric`, never scoring a driver node or mixing metrics into
    that mean. This function's own per-node print loop below (every node,
    its own `_metric_for`-resolved metric, tagged TARGET/driver) already
    existed and is PURELY REPORTING on the one already-decided winning
    configuration -- it does not feed back into any search. The only
    thing this task adds is writing that same data to a CSV in
    `results_dir` alongside the existing print, so it survives past the
    log (metrics are not mutually comparable across rows -- correlation
    for targets, f1 for neurons -- so the CSV labels each row's own metric
    and is not meant to be summed down the score column).'''
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        node_theta = theta_search_results[network_id]['best_node_theta']
        generate_start = generate_starts[network_id] if generate_starts else generate_validation_start

        generate_rebuild_start = time.perf_counter()
        _log_progress(progress_log_path, f'{network_id}: FINAL Generate() rebuild START')
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
            _log_progress(
                progress_log_path,
                f'{network_id}: FINAL Generate() rebuild END -- FAILED '
                f'{time.perf_counter() - generate_rebuild_start:.1f}s: {exc!r}')
            continue
        _log_progress(progress_log_path,
                      f'{network_id}: FINAL Generate() rebuild END -- '
                      f'{time.perf_counter() - generate_rebuild_start:.1f}s')

        run['G'] = G
        run['best_node_theta'] = node_theta
        run['generate_start'] = generate_start

        actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
        generated = G.DataOut.reset_index(drop=True)
        nodes = [c for c in generated.columns if c != 'Time']

        print(f"{network_id} Generate() score, held-out window (start row {generate_start}):", flush=True)
        per_node_rows = []
        for node in nodes:
            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
            a = actual_df[node].to_numpy()[generate_start:generate_start + len(generated)]
            r = tsls._score(a, generated[node].to_numpy(), node_metric, gammaDelta=gamma_delta)
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} {node_metric}={r:7.3f}", flush=True)
            per_node_rows.append({'node': node, 'role': tag, 'metric': node_metric, 'score': r})

        # R11: purely additive -- same data the print loop above already
        # reports, written to a file so it survives past the log. Metrics
        # differ per row (correlation for targets, f1 for neurons) -- do
        # not sum/average the score column, it mixes incompatible scales.
        pd.DataFrame(per_node_rows, columns=['node', 'role', 'metric', 'score']).to_csv(
            results_dir / f'per_node_scores_{network_id}.csv', index=False)


def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             node_theta, results_dir, cores, save_path,
                             metric, target_metric, gamma_delta,
                             generate_length=None, generate_start=None, nodes_per_figure=10,
                             exclusion_radius=0, plot_range_mode='full',
                             generate_validation_start=None, total_rows=None):
    '''Runs the Forecast() diagnostic and draws the one actual-vs-
    Generate()-vs-Forecast() figure for this network -- ported from
    simple-mde-gmn.py, including the 'full'/'windowed' plot_range_mode
    toggle (see that file's own docstring for the full behavior of each
    option -- identical here). Returns (None, {}) instead of raising if the
    Forecast() build/call itself fails.'''
    generate_length = generate_length if generate_length is not None else prediction_length
    generate_start = generate_start if generate_start is not None else generate_validation_start

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
            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
            actual = actual_df[node].to_numpy()
            forecast_pred = forecast_out[node].to_numpy()
            rhos[node] = tsls._score(actual[time_idx], forecast_pred, node_metric, gammaDelta=gamma_delta)

            ax.plot(np.arange(plot_start, plot_end), actual[plot_start:plot_end],
                     color='tab:blue', lw=1, alpha=0.6, label='Lib/actual')
            if node in generate_out.columns:
                ax.plot(generate_time, generate_out[node].to_numpy(),
                        color='tab:red', lw=1, label='Generate()')
            ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
            ax.axvline(generate_start, color='gray', ls='--', lw=0.8)
            # R22 (docs/review-findings.md): defensive second layer -- clips
            # the view to the intended window regardless of what any stray
            # out-of-window data (a future variant of the leaf-passthrough
            # mislabeling this section's own fix addresses) might otherwise
            # stretch matplotlib's autoscale to.
            ax.set_xlim(plot_start, plot_end)
            ax.set_title(f"{node}   Forecast {node_metric}={rhos[node]:.3f}", fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"Generate() vs Forecast() vs actual -- {net_result['network_id']} "
                     f"(nodes {part * nodes_per_figure + 1}-"
                     f"{part * nodes_per_figure + len(part_nodes)} of {len(nodes)})", fontsize=11)
        fig.tight_layout()
        part_path = save_path.with_name(f'{save_path.stem}_part{part + 1:02d}{save_path.suffix}')
        fig.savefig(part_path, dpi=300)
        plt.close(fig)

    return Gf, rhos


def gmn_fidelity_rho(net_result, G, generate_start, metric, target_metric, gamma_delta,
                      default_E=3, tau=-1):
    '''Fig-4e-style generative fidelity: per node, a fresh proper_simplex
    predicts the real actual series *from* the generated series on a 50/50
    split of the generated window -- ported from simple-mde-gmn.py, scored
    via _metric_for()/proper_simplex instead of pyEDM's plain Simplex/
    ComputeError (proper_simplex is TSLS's own equivalent -- see
    load_config()'s METRIC docstring). E defaults to node_dimensions (an
    EmbedDimension-based estimate from build_network(), when computed).'''
    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[generate_start: generate_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            out = tsls.proper_simplex(
                pair_df, columns=f'{node}_gen', target=f'{node}_actual',
                E=E, tau=tau, Tp=0,
                lib=[1, split], pred=[split + 1, n_gen],
                embedded=False, metric=node_metric, gammaDelta=gamma_delta,
            )
            fidelity_rho[node] = out['metrics'][node_metric]
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
# NOTE (superseded by R19, docs/review-actions-v1.md): the notebook cell 15
# diagnostic (dx.Evaluate: MDE selection vs. PCA/diffusion-map baselines)
# used to have no cleaned_helper_functions.py equivalent here -- dx.Evaluate
# is a standalone dimx tool with no TSLS-aware port, and it depended on a
# dx.MDE result object this pipeline no longer constructs. run_mde_evaluate()
# below reimplements the same MDE/PCA/diffusion-map comparison directly
# against this pipeline's own building blocks (proper_simplex, sklearn,
# pydiffmap -- the same underlying libraries dx.Evaluate itself wraps)
# instead of routing through dimx at all -- see that function's own
# docstring and docs/review-findings.md's R19 section for the full design.

def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir,
                         target_metric, gamma_delta):
    '''Both TARGETS' own self-predictability -- always target_metric
    (continuous), never metric (see load_config()'s TARGET_METRIC
    docstring). Reads r['m'].args.lib/pred/tau -- the 'm' namespace shape
    every run_mde_for_target() result carries (see that function's own
    docstring).

    R7 (docs/review-actions-v1.md): previously called fig.savefig()
    unconditionally, even when every target hit the `if not cols`
    skip below (all zero-driver, or all _failed_mde_result()'s own
    mde_df=None case, which also carries SELECTED_COLUMNS=[] -- see
    _failed_mde_result()'s own docstring) -- an empty axes with no data
    and a legend with nothing to show, written to disk as if it were a
    real result. Now skips the write entirely in that case (figure is
    still closed either way, so no resource leak).'''
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        if not cols:
            print(f'{target}: no MDE-selected columns, skipping Tp-vs-rho sweep.', flush=True)
            continue
        pi = tsls.proper_predict_interval(
            r['mde_df'], columns=cols, target=target,
            lib=r['m'].args.lib, pred=r['m'].args.pred,
            E=len(cols), tau=r['m'].args.tau,
            maxTp=max_tp, metric=target_metric, gammaDelta=gamma_delta,
        )
        ax.plot(pi['Tp'], pi[target_metric], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    if not annotation_lines:
        print('Tp-vs-rho sweep: every target had zero MDE-selected columns -- '
              'skipping the figure entirely (no data to plot).', flush=True)
        plt.close(fig)
        return

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel(f'Prediction skill ({target_metric})')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir,
                               target_metric, gamma_delta):
    '''R7 (docs/review-actions-v1.md): unlike run_tp_vs_rho_sweep() above,
    this is a UNIVARIATE self-predictability grid (columns=target,
    target=target) -- it needs no MDE-selected drivers at all, so it
    deliberately does NOT skip on SELECTED_COLUMNS being empty (a
    legitimate zero-driver target, e.g. Run A's own head_position_y, still
    gets a real, informative grid here). It DOES need `mde_df` to be a
    real DataFrame, though: a target whose depth-0 MDE call itself raised
    (not "selected zero drivers", a different case -- see
    _failed_mde_result()'s own docstring) gets mde_df=None, which would
    otherwise crash proper_simplex() below outright, and take down every
    remaining target's own (unrelated, otherwise-computable) figure with
    it, since the exception would propagate out of this whole function.
    Guarded per-target instead, matching this codebase's own established
    "one degenerate item dead-ends cleanly, it doesn't crash the batch"
    convention (_failed_mde_result()'s own docstring).'''
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        if r['mde_df'] is None:
            print(f'{target}: MDE call failed (mde_df is None), skipping E/tau grid.', flush=True)
            continue
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                out = tsls.proper_simplex(
                    r['mde_df'], columns=target, target=target,
                    E=E, tau=tau, lib=r['m'].args.lib, pred=r['m'].args.pred,
                    embedded=False, metric=target_metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][target_metric]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label=target_metric)

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)


def _cae(x, y):
    '''Cumulative absolute error -- recovered verbatim from dimx.Evaluate's
    own CAE() (.venv/.../dimx/Evaluate.py:503-506: `sum(abs(x - y))`), per
    R19 (docs/review-actions-v1.md)'s explicit instruction to read the
    source and replicate the exact definition rather than reimplementing
    from the name. Unnormalized -- not a mean, not a fraction -- and, for
    a 2-column (x, y) array (GMN_MODE='combined'), sums the absolute error
    over BOTH columns at once (numpy's own `abs`/`sum` broadcast this way
    with no axis given), which is the natural generalization used for the
    combined-mode spatial panel's own CAE.'''
    return float(np.sum(np.abs(np.asarray(x) - np.asarray(y))))


def _score_baseline(target_pred, predicted, targets, combined, target_metric, gamma_delta):
    '''Shared scoring step for run_mde_evaluate()'s PCA/diffusion-map
    baselines -- factored out so both baselines' own try/except blocks
    (see that function's own docstring on why they're independently
    guarded) call one identical implementation rather than two copies of
    the combined-vs-single-target branch drifting apart.'''
    if combined:
        return float(np.nanmean([
            tsls._score(target_pred[:, i], predicted[:, i], target_metric, gammaDelta=gamma_delta)
            for i in range(len(targets))
        ]))
    return tsls._score(target_pred, predicted, target_metric, gammaDelta=gamma_delta)


# Wall-clock bound for run_mde_evaluate()'s own PCA/diffusion-map baselines
# (docs/review-findings.md's "diffusion-map crash and hang" finding): a
# real diffusion-map fit on real TSLS data, at real (~31,000-row) scale,
# has been observed to run past 74 CPU-minutes with no completion, crash,
# or any output at all -- a non-raising hang a try/except cannot catch,
# since there's nothing to except until the call returns control. Not
# calibrated against a known-good completion time (no successful fit at
# this scale has yet been observed to time); chosen as generous headroom
# over the 1.8s a genuine crash takes to surface, while still bounding the
# worst case to a tolerable, one-time-per-network cost.
MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS = 300


def _pca_baseline_worker(queue, pool_lib, pool_pred, target_lib, D):
    '''Runs in a genuinely separate process -- see
    _run_baseline_with_timeout()'s own docstring for why. Catches its own
    exception and puts it on the queue rather than letting the child
    process's own traceback (still visible in the job's .err) be the only
    record -- so the parent can log it in run_mde_evaluate()'s own
    existing "FAILED for {targets}: {exc!r}" style regardless of which
    process actually raised it.'''
    try:
        pca = PCA(n_components=D)
        pca.fit(pool_lib)
        pca_lib_proj, pca_pred_proj = pca.transform(pool_lib), pca.transform(pool_pred)
        pca_predicted = LinearRegression().fit(pca_lib_proj, target_lib).predict(pca_pred_proj)
        queue.put(('ok', pca_predicted, pca_pred_proj))
    except Exception as exc:
        queue.put(('error', repr(exc), None))


def _dmap_baseline_worker(queue, pool_lib, pool_pred, target_lib, D, dmap_k, dmap_epsilon):
    '''Same contract as _pca_baseline_worker() above.'''
    try:
        dmap_model = dmap.DiffusionMap.from_sklearn(
            n_evecs=D, k=dmap_k, epsilon=dmap_epsilon,
            neighbor_params={'n_jobs': -1, 'algorithm': 'kd_tree'},
        )
        dmap_model.fit(pool_lib)
        dmap_lib_proj, dmap_pred_proj = dmap_model.transform(pool_lib), dmap_model.transform(pool_pred)
        dmap_predicted = LinearRegression().fit(dmap_lib_proj, target_lib).predict(dmap_pred_proj)
        queue.put(('ok', dmap_predicted, dmap_pred_proj))
    except Exception as exc:
        queue.put(('error', repr(exc), None))


def _run_baseline_with_timeout(worker_fn, args, timeout_seconds=MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS):
    '''Runs worker_fn(queue, *args) in a genuinely separate process,
    force-killed if it exceeds timeout_seconds -- the wall-clock bound
    run_mde_evaluate()'s own PCA/diffusion-map guards need underneath
    their existing exception handling (docs/review-findings.md's
    "diffusion-map crash and hang" finding: a try/except plus a
    signal-based "soft" timeout is not enough here, since pydiffmap's own
    KD-tree/BLAS calls can run for a long time without ever returning
    control to the Python interpreter -- a signal fires but its handler
    can't actually run until the call returns on its own, which is
    exactly the case that needs covering. Only an OS-level SIGKILL of a
    genuinely separate process reclaims control regardless of what the
    computation is doing internally). Cost: one process fork per baseline
    call (cheap under `fork` -- no re-import, copy-on-write memory, real
    overhead is milliseconds) plus pickling the (modest-sized, D-column)
    result back through a Queue; the real cost is that a hard kill on
    timeout discards whatever progress a slow-but-would-eventually-
    succeed computation had made, converting it into a skip -- the
    deliberate trade for a bounded worst case instead of an unbounded one.

    Returns (status, value): ('ok', predicted_and_proj_tuple) on success;
    ('timeout', None) if force-killed; ('error', repr(exc)) if the worker
    itself raised (only ever read in the success path below -- the
    process has already finished and flushed its queue write by the time
    proc.join() returns non-alive, so there is no kill-mid-write race to
    guard against).'''
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(queue, *args))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return 'timeout', None
    try:
        status, a, b = queue.get_nowait()
    except Exception:
        return 'error', f'child process exited without a result (exitcode={proc.exitcode})'
    if status == 'ok':
        return 'ok', (a, b)
    return 'error', a


def run_mde_evaluate(net_result, prediction_start, prediction_length, results_dir,
                      target_metric, gamma_delta, exclusion_radius=0,
                      dmap_k=15, dmap_epsilon='bgh_generous', max_traces_shown=7,
                      figsize=(8, 8)):
    '''R19 (docs/review-actions-v1.md): TSLS-side counterpart of
    simple-mde-gmn.py's dx.Evaluate()-based run_mde_evaluate() -- see the
    NOTE above this function for why this pipeline had no such comparison
    until now. Reimplements dx.Evaluate's own MDE-vs-PCA-vs-diffusion-map
    comparison directly against proper_simplex (MDE) and sklearn.PCA /
    pydiffmap.diffusion_map (the same two libraries dx.Evaluate itself
    wraps, confirmed by reading dimx/Evaluate.py in full) rather than
    reusing dx.Evaluate, which has no _metric_for()/_score() awareness and
    depends on a dimx.MDE result object this pipeline no longer
    constructs.

    D = the number of *direct* drivers of the target in the COMPLETE
    nested-MDE network (both stages) -- graph.predecessors(target), read
    from net_result['network']['Graph'] (built by build_network() /
    build_nested_mde_network()), NOT a depth-0-only SELECTED_COLUMNS list.
    PCA and diffusion-map each fit their first D components on the same
    pool, so all three methods compare at equal dimensionality (R19's own
    requirement). That pool is every non-target column in
    net_result['DATA_AFTER_MDE'] -- i.e. every observable nested-MDE kept
    anywhere in the whole network (a superset of the target's own direct
    drivers whenever Stage 2 chains through an intermediate node), the
    TSLS-side analog of dx.Evaluate's own FilterData()-selected column
    subset (confirmed: simple-mde-gmn.py's own run_mde_evaluate() passes
    `columns_range=[1, mde_df.shape[1]]` against `mde_df = df[[target] +
    observable_columns]` -- i.e. every candidate column except the target
    itself, never the full pre-CCM-prefilter universe, which isn't
    serialized anywhere after build_network() returns).

    net_result['targets'] has one entry (GMN_MODE='separate') -> a
    standard single-target figure, panel 1 an actual-vs-predicted time
    series, matching docs/reference/mde-evaluate-reference-plot.png's
    layout exactly. Two entries (GMN_MODE='combined') -> ONE joint figure
    for the pair: panel 1 is a spatial (x, y) trajectory instead (explicit
    user direction -- R19's own "ask before implementing" gate: "the
    instruction reads as [a spatial plot]" -- confirmed against the
    reference plot's own single-target case, which does not settle it).
    D for this joint figure is the union of each target's own
    direct-driver set (this pipeline's own reading, documented here since
    R19's text only defines D for a single target -- "how many distinct
    driver nodes feed the target(s) this figure is about," the natural
    generalization once there is more than one target to be a direct
    driver of). PCA/diffusion-map's own linear-regression step predicts
    both target columns at once (sklearn's LinearRegression natively
    supports a 2-column y) -- R19's own "PCA and diffusion-map extract
    components of the multivariate target" -- rather than two independent
    single-target Evaluate runs; MDE still reconstructs each target
    separately (proper_simplex predicts one column at a time) and the two
    reconstructions are combined into one (x, y) trajectory for the
    spatial panel, since there is no multivariate Simplex call to make
    instead.

    Held-out window: [prediction_start, prediction_start+prediction_length)
    -- the same MDE validation segment (B) evaluate_smap_candidate() and
    every other MDE-side diagnostic in this file already uses; fit window
    is [0, prediction_start). Never touches the Generate/Forecast
    validation segment [B, end) -- same leakage boundary as everywhere
    else in this pipeline.

    Every target in net_result['targets'] is guaranteed to have D >= 1 by
    the time this runs: build_network() already calls partition_network()
    on this same graph, which raises if any target has zero drivers (R14)
    -- so there is no zero-driver-target case to guard here.

    Scoring (R19's own "scoring routes through _metric_for" acceptance
    criterion): TARGETS are always continuous, so _metric_for() always
    resolves to target_metric here regardless of which node is being
    scored -- reported per method via tsls._score(), alongside CAE (which
    stays the reference figure's own display convention, not replaced by
    it). Writes results_dir/mde_evaluate_{network_id}.png; returns a dict
    of the computed CAE/score values for the caller to log.'''
    targets = net_result['targets']
    graph = net_result['network']['Graph']
    combined = len(targets) > 1

    data_after_mde = pd.read_csv(net_result['DATA_AFTER_MDE'])
    pool_columns = [c for c in data_after_mde.columns if c not in ('Time', *targets)]

    lib = [1, prediction_start]
    pred = [prediction_start + 1, prediction_start + prediction_length]
    pred_slice = slice(prediction_start, prediction_start + prediction_length)

    direct_drivers = {t: list(graph.predecessors(t)) for t in targets}
    driver_union = sorted(set().union(*direct_drivers.values()))
    D = len(driver_union)

    # ---- MDE: each target's own reconstruction from its own direct drivers ----
    mde_predicted, mde_cae, mde_score = {}, {}, {}
    for t in targets:
        out = tsls.proper_simplex(
            data_after_mde, columns=direct_drivers[t], target=t,
            lib=lib, pred=pred, Tp=0, embedded=True,
            exclusionRadius=exclusion_radius,
            metric=target_metric, gammaDelta=gamma_delta,
        )
        predicted = out['predictions']['Predictions'].to_numpy()
        actual = out['predictions']['Observations'].to_numpy()
        mde_predicted[t] = predicted
        mde_cae[t] = _cae(actual, predicted)
        mde_score[t] = out['metrics'][target_metric]

    # ---- PCA / diffusion-map: fit on the network-wide driver pool, regress
    # onto the target(s) (jointly, in combined mode) ----------------------
    pool_lib = data_after_mde[pool_columns].iloc[:prediction_start].to_numpy()
    pool_pred = data_after_mde[pool_columns].iloc[pred_slice].to_numpy()
    target_lib = data_after_mde[targets].iloc[:prediction_start].to_numpy()
    target_pred = data_after_mde[targets].iloc[pred_slice].to_numpy()
    if not combined:
        target_lib = target_lib.ravel()
        target_pred = target_pred.ravel()

    # PCA and diffusion-map are optional, unsupervised comparison
    # baselines, not part of MDE's own causal claim (docs/tsls-mde-gmn-
    # pipeline.tex sec:diagnostics's own linknote) -- guarded independently
    # per this codebase's own established "one degenerate item dead-ends
    # cleanly, it doesn't crash the batch" convention (already applied to
    # Stage 1/Stage 2's per-node exceptions, sec:stage1-parallelization;
    # rebuild_at_best_params()'s per-network Generate() divergence,
    # sec:validation; run_tp_vs_rho_sweep()'s per-target skip,
    # sec:diagnostics), which run_mde_evaluate() itself had never received.
    # A failure in either is logged with the exception text and the
    # target(s) it occurred on, and that baseline's own numbers/plot panel
    # are skipped -- the run continues to completion and archives per
    # sec:cluster-storage regardless of which (if either) baseline
    # actually produced a result.
    pca_predicted = pca_pred_proj = pca_cae = pca_score = None
    status, value = _run_baseline_with_timeout(_pca_baseline_worker, (pool_lib, pool_pred, target_lib, D))
    if status == 'ok':
        pca_predicted, pca_pred_proj = value
        pca_cae = _cae(target_pred, pca_predicted)
        pca_score = _score_baseline(target_pred, pca_predicted, targets, combined,
                                     target_metric, gamma_delta)
    elif status == 'timeout':
        print(f"{net_result['network_id']}: PCA baseline exceeded "
              f"{MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS}s wall-clock bound for {targets} -- "
              f"terminated and skipped (MDE and any other baseline continue unaffected)",
              flush=True)
    else:
        print(f"{net_result['network_id']}: PCA baseline FAILED for {targets}: {value} -- "
              f"skipping this baseline (MDE and any other baseline continue unaffected)",
              flush=True)

    dmap_predicted = dmap_pred_proj = dmap_cae = dmap_score = None
    status, value = _run_baseline_with_timeout(
        _dmap_baseline_worker, (pool_lib, pool_pred, target_lib, D, dmap_k, dmap_epsilon))
    if status == 'ok':
        dmap_predicted, dmap_pred_proj = value
        dmap_cae = _cae(target_pred, dmap_predicted)
        dmap_score = _score_baseline(target_pred, dmap_predicted, targets, combined,
                                      target_metric, gamma_delta)
    elif status == 'timeout':
        print(f"{net_result['network_id']}: Diffusion-map baseline exceeded "
              f"{MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS}s wall-clock bound for {targets} -- "
              f"terminated and skipped (MDE and any other baseline continue unaffected)",
              flush=True)
    else:
        print(f"{net_result['network_id']}: Diffusion-map baseline FAILED for {targets}: {value} -- "
              f"skipping this baseline (MDE and any other baseline continue unaffected)",
              flush=True)

    pca_summary = (f"PCA CAE={pca_cae:.2f} {target_metric}={pca_score:.3f}"
                   if pca_cae is not None else "PCA FAILED (skipped)")
    dmap_summary = (f"D-Map CAE={dmap_cae:.2f} {target_metric}={dmap_score:.3f}"
                    if dmap_cae is not None else "D-Map FAILED (skipped)")
    print(f"run_mde_evaluate ({net_result['network_id']}, D={D}): "
          f"MDE CAE={ {t: round(mde_cae[t], 2) for t in targets} } "
          f"{target_metric}={ {t: round(mde_score[t], 3) for t in targets} } | "
          f"{pca_summary} | {dmap_summary}", flush=True)

    # ---- Figure: 4 stacked panels, matching the reference -----------------
    maxN = min(max_traces_shown, D)
    x_pred = np.arange(prediction_start, prediction_start + prediction_length)

    fig = plt.figure(figsize=figsize)
    if combined:
        # Panel 1 is spatial (x-y position), not a time series -- cannot
        # share a time x-axis with panels 2-4, unlike the single-target case.
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)
        ax3 = fig.add_subplot(gs[3], sharex=ax1)
    else:
        (ax0, ax1, ax2, ax3) = fig.subplots(4, 1, sharex=True)
    axs = [ax0, ax1, ax2, ax3]

    axs[0].set_title(f"{net_result['network_id']}: train {lib}  predict {pred}")
    axs[1].set_ylabel('MDE', fontsize=12)
    axs[2].set_ylabel('Diffusion Map', fontsize=12)
    axs[3].set_ylabel('PCA', fontsize=12)
    lw = 2.0

    if combined:
        tx, ty = targets
        axs[0].plot(target_pred[:, 0], target_pred[:, 1], color='black', lw=lw,
                    label=f'{tx}/{ty}')
        mde_xy = np.column_stack([mde_predicted[tx], mde_predicted[ty]])
        axs[0].plot(mde_xy[:, 0], mde_xy[:, 1], lw=lw,
                     label=f'MDE   {mde_cae[tx] + mde_cae[ty]:.2f}')
        if pca_predicted is not None:
            axs[0].plot(pca_predicted[:, 0], pca_predicted[:, 1], lw=lw, label=f'PCA    {pca_cae:.2f}')
        if dmap_predicted is not None:
            axs[0].plot(dmap_predicted[:, 0], dmap_predicted[:, 1], lw=lw, label=f'D-Map {dmap_cae:.2f}')
        axs[0].set_xlabel(tx)
        axs[0].set_ylabel(ty)
        axs[0].set_ylabel('Predictions', fontsize=12)
    else:
        target = targets[0]
        axs[0].set_ylabel('Predictions', fontsize=12)
        if dmap_predicted is not None:
            axs[0].plot(x_pred, dmap_predicted, lw=lw, label=f'D-Map {dmap_cae:.2f}')
        if pca_predicted is not None:
            axs[0].plot(x_pred, pca_predicted, lw=lw, label=f'PCA    {pca_cae:.2f}')
        axs[0].plot(x_pred, mde_predicted[target], lw=lw, label=f'MDE   {mde_cae[target]:.2f}')
        axs[0].plot(x_pred, target_pred, color='black', lw=lw, label=target)
    axs[0].legend(title='CAE', bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # driver_union, not a per-target loop -- a driver shared by more than
    # one target in combined mode (e.g. a common upstream node) would
    # otherwise get plotted and labeled once per target it feeds, showing
    # the identical trace twice under the same name.
    for driver in driver_union[:maxN]:
        axs[1].plot(x_pred, data_after_mde[driver].iloc[pred_slice], lw=lw, label=driver)
    axs[1].legend(title='MDE', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # Panels 2/3: same skip-not-crash handling as panel 0 above -- a failed
    # baseline has no projection to plot, so the panel is left empty with
    # an explicit label rather than raising on a None subscript.
    if dmap_pred_proj is not None:
        for col in range(maxN):
            axs[2].plot(x_pred, dmap_pred_proj[:, col], lw=lw, label=str(col))
        axs[2].legend(title='Diffusion Map', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    else:
        axs[2].text(0.5, 0.5, 'Diffusion Map baseline FAILED (see log)', ha='center', va='center',
                    transform=axs[2].transAxes, fontsize=9, color='gray')

    if pca_pred_proj is not None:
        for col in range(maxN):
            axs[3].plot(x_pred, pca_pred_proj[:, col], lw=lw, label=str(col))
        axs[3].legend(title='PCA', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    else:
        axs[3].text(0.5, 0.5, 'PCA baseline FAILED (see log)', ha='center', va='center',
                    transform=axs[3].transAxes, fontsize=9, color='gray')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3 if combined else 0.04)
    fig.savefig(results_dir / f"mde_evaluate_{net_result['network_id']}.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'D': D, 'direct_drivers': direct_drivers,
        'mde_cae': mde_cae, 'mde_score': mde_score,
        'pca_cae': pca_cae, 'pca_score': pca_score,
        'dmap_cae': dmap_cae, 'dmap_score': dmap_score,
    }


def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")


# =============================================================================
# Orchestration
# =============================================================================

def main():
    cfg = load_config()

    # Progress-monitoring log (see HEARTBEAT_SECONDS/_log_progress()'s own
    # docstrings): cfg['RESULTS_DIR'] is already on /flash for the whole
    # run's duration (TSLS_MDE_GMN_RESULTS_DIR -> $OUT_DIR in the sbatch
    # script, archived to bucket only at job end), so this needs no new
    # path plumbing. Opened in truncate mode -- each job already gets its
    # own fresh RESULTS_DIR, so there is no earlier content to preserve.
    progress_log_path = cfg['RESULTS_DIR'] / 'progress.log'
    open(progress_log_path, 'w').close()
    pipeline_start = time.perf_counter()
    _log_progress(progress_log_path, 'PIPELINE START')

    df_raw = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Raw data shape:', df_raw.shape, flush=True)

    neuron_columns = raw_neuron_columns(df_raw, cfg['REGION_CHANNEL_CAP'])
    print(f'{len(neuron_columns)} raw spike-count channels to TSLS-convert', flush=True)
    tsls_df_full, obs_cols = convert_to_tsls(df_raw, neuron_columns, cfg['TARGETS'], cfg['RESULTS_DIR'])
    df = burn_in_trim(tsls_df_full, obs_cols, cfg['RESULTS_DIR'])

    # Three-way A/B/end split -- see load_config()'s docstring for the full
    # rationale (identical to simple-mde-gmn.py, ported here), computed
    # against the TSLS-converted, burn-in-trimmed row count (len(df)), not
    # the raw file's row count.
    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))                          # A
    post_lib_remainder = len(df) - cfg['PREDICTION_START']
    cfg['PREDICTION_LENGTH'] = int(round(post_lib_remainder * cfg['MDE_VALIDATION_RATIO']))   # B - A
    cfg['GENERATE_VALIDATION_START'] = cfg['PREDICTION_START'] + cfg['PREDICTION_LENGTH']     # B
    cfg['GENERATE_VALIDATION_LENGTH'] = len(df) - cfg['GENERATE_VALIDATION_START']            # end - B

    if cfg['GENERATE_LENGTH'] <= 0:
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']
    elif cfg['GENERATE_LENGTH'] > cfg['GENERATE_VALIDATION_LENGTH']:
        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the Generate/Forecast "
              f"validation segment ({cfg['GENERATE_VALIDATION_LENGTH']} rows); clamping "
              f"to it.", flush=True)
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']

    max_feasible_offset = cfg['GENERATE_VALIDATION_LENGTH'] - cfg['GENERATE_LENGTH']
    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
                                cfg['GENERATE_START_STEP']))
    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
    if feasible_offsets != start_offsets:
        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]

    cfg['LAG_SET'] = derive_lag_set(
        cfg['DELAY_SOURCE'], cfg['DELAY_SET'], cfg['LAG_MODE'], cfg['LAG_DENSITY'], cfg['LAG_STEP'],
        cfg['GENERATE_LENGTH'], cfg['PREDICTION_START'],
    )

    # VERIFY (see this pipeline's own design-decision record for why this
    # guard exists here and not in simple-mde-gmn.py): Stage 2's lag-
    # embedding depth must fit within PREDICTION_START (A), the MDE library
    # segment -- an implicit assumption in simple-mde-gmn.py too, but
    # TSLS's own PREDICTION_START is data-driven (burn_in_trim() shrinks
    # the usable row count depending on which channel fires slowest, unlike
    # simple-mde-gmn.py's fixed-length 1 Hz input), so this fails fast with
    # a clear message instead of a confusing downstream all-NaN/shape
    # error deep inside proper_mde or gmn's own SMap calls.
    max_lag_depth = max((abs(d) for d in cfg['LAG_SET']), default=0)
    if max_lag_depth >= cfg['PREDICTION_START']:
        raise ValueError(
            f"Stage 2's lag-embedding depth ({max_lag_depth} rows, from LAG_SET= "
            f"{cfg['LAG_SET']}) does not fit within PREDICTION_START "
            f"({cfg['PREDICTION_START']} rows, the MDE library segment) -- every "
            f"lagged candidate column would be all-NaN across the entire library. "
            f"This is data-driven for TSLS (burn-in trim shrinks the usable row "
            f"count depending on which channel fires slowest), unlike "
            f"simple-mde-gmn.py's fixed-length input. Narrow "
            f"TSLS_MDE_GMN_LAG_MODE/LAG_STEP, shrink TSLS_MDE_GMN_GENERATE_LENGTH, "
            f"or raise TSLS_MDE_GMN_LIB_RATIO to fix."
        )

    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE']) if cfg['REUSE_BEST_PARAMS'] else None
    code_fingerprint = build_code_fingerprint() if cfg['REUSE_BEST_PARAMS'] else None

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    filtered_pools = {}
    if cfg['MODE'] == 'MDE-GMN':
        # CCM pre-filter runs once per target_groups_for_mode() group, here,
        # BEFORE depth-0 -- so depth-0's own candidate pool is already the
        # filtered one, same as every later nested-MDE round. See this
        # pipeline's own build_network() docstring: that function no longer
        # runs the pre-filter itself, precisely because main() now always
        # does it first.
        for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
            if cfg['CCM_PREFILTER_RHO_MIN'] > 0:
                filtered_columns, _ = ccm_prefilter_observables(
                    df, obs_cols, group, cfg['PREDICTION_START'],
                    cfg['CCM_PREFILTER_RHO_MIN'], cfg['CCM_PREFILTER_E'], cfg['TP'],
                    cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
                    cfg['LIBSIZE_RATIOS'], cfg['CORES'],
                )
                print(f'CCM pre-filter: kept {len(filtered_columns)}/{len(obs_cols)} '
                      f'observables (rho >= {cfg["CCM_PREFILTER_RHO_MIN"]}) for {group}', flush=True)
            else:
                filtered_columns = obs_cols
            filtered_pools[tuple(group)] = filtered_columns

        # Stage 1 seed: real-observable-only MDE, no delay expansion --
        # matches build_nested_mde_network()'s own Stage-1 candidate pool
        # exactly, so its depth-0 reuse of this result stays valid.
        if cfg['GMN_MODE'] == 'combined':
            # One joint multi_proper_mde call across every target, instead
            # of GMN_MODE='separate's per-target proper_mde loop below --
            # see run_multi_mde_for_targets()'s and load_config()'s own
            # MULTI_MDE_E docstrings. The single shared result is fanned out
            # into one per-target *view* below (same 'm'/'mde_df'/
            # 'SELECTED_COLUMNS' object, just re-keyed with a 'target' entry
            # each) so every downstream consumer that indexes
            # mde_results[target] -- build_nested_mde_network()'s depth-0
            # reuse (adds the SAME joint drivers as edges into every target
            # node in the group, which is the intended "combined" MDE
            # behaviour), and the diagnostics plots/sweeps below -- keeps
            # working completely unchanged.
            candidate_pool = [c for c in filtered_pools[tuple(cfg['TARGETS'])] if c not in cfg['TARGETS']]
            effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
            depth0_start = time.perf_counter()
            _log_progress(progress_log_path, f"depth0 joint MDE START -- targets={cfg['TARGETS']}")
            try:
                shared_result = run_multi_mde_for_targets(
                    cfg['TARGETS'], df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                    effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                    tp=cfg['TP'],
                    metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                    gamma_delta=cfg['GAMMA_DELTA'],
                    target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                    ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                    ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                    reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                    run_cache_dir=cfg['RUN_CACHE_DIR'],
                    data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                    code_fingerprint=code_fingerprint,
                    E=0, max_e=cfg['MULTI_MDE_E'], target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
                )
            except Exception as exc:
                # Same failure mode _failed_mde_result() guards against below
                # -- this is the depth-0 seed for the Stage-1 BFS (shared
                # across every target in the group), so it needs the
                # identical guard or the exact crash just relocates one
                # level up.
                print(f"{cfg['TARGETS']}: joint (multi-target) MDE call FAILED: {exc!r} -- "
                      f'treating as a dead-end (zero drivers selected for every target)', flush=True)
                shared_result = _failed_multi_mde_result(cfg['TARGETS'])
            _log_progress(progress_log_path,
                          f'depth0 joint MDE END -- {time.perf_counter() - depth0_start:.1f}s '
                          f"selected={shared_result['SELECTED_COLUMNS']}")
            for target in cfg['TARGETS']:
                mde_results[target] = {**shared_result, 'target': target}
        else:
            for target in cfg['TARGETS']:
                candidate_pool = [c for c in filtered_pools[(target,)] if c != target]
                effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
                depth0_start = time.perf_counter()
                _log_progress(progress_log_path, f'depth0 MDE START -- {target}')
                try:
                    mde_results[target] = run_mde_for_target(
                        target, df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                        effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                        cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                        tp=cfg['TP'],
                        metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                        gamma_delta=cfg['GAMMA_DELTA'],
                        is_behavioral_target=True,
                        target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                        min_pred_events=cfg['MIN_PRED_EVENTS'],
                        max_window_rows=cfg['MAX_WINDOW_ROWS'],
                        ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                        ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                        run_cache_dir=cfg['RUN_CACHE_DIR'],
                        data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                        code_fingerprint=code_fingerprint,
                        max_e=cfg['MDE_D'],
                        # This loop is sequential over cfg['TARGETS'] (one
                        # target computed at a time, no outer pool sharing
                        # cores with it) -- so unlike build_nested_mde_
                        # network()'s own per-wave outer_workers-based
                        # split, this depth-0 seed call gets the FULL core
                        # budget for its own inner pool.
                        inner_workers=max(1, cfg['CORES']),
                        use_gpu=cfg['GPU_ENABLED'],
                    )
                except Exception as exc:
                    # Same failure mode _failed_mde_result() guards against
                    # in build_nested_mde_network() below -- this is the
                    # depth-0 seed for that same Stage-1 BFS, so it needs
                    # the identical guard or the exact crash just
                    # relocates one level up.
                    print(f'{target}: top-level MDE call FAILED: {exc!r} -- treating as a dead-end '
                          f'(zero drivers selected for this target)', flush=True)
                    mde_results[target] = _failed_mde_result(target)
                _log_progress(progress_log_path,
                              f'depth0 MDE END -- {target} {time.perf_counter() - depth0_start:.1f}s '
                              f"selected={mde_results[target]['SELECTED_COLUMNS']}")

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
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} TSLS-converted observable "
              f"columns as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
            cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
            target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
            min_pred_events=cfg['MIN_PRED_EVENTS'],
            max_window_rows=cfg['MAX_WINDOW_ROWS'],
            ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
            observable_columns=filtered_pools.get(tuple(group), obs_cols),
            nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
            exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
            lag_set=cfg['LAG_SET'], stage2_allow_real_observables=cfg['STAGE2_ALLOW_REAL_OBSERVABLES'],
            tp=cfg['TP'],
            reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
            run_cache_dir=cfg['RUN_CACHE_DIR'],
            data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
            code_fingerprint=code_fingerprint,
            gmn_mode=cfg['GMN_MODE'], multi_mde_e=cfg['MULTI_MDE_E'],
            multi_mde_target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
            external_input_mode=cfg['EXTERNAL_INPUT_MODE'],
            progress_log_path=progress_log_path,
            mp_start_method=cfg['MP_START_METHOD'],
            use_gpu=cfg['GPU_ENABLED'],
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        if cfg['MODE'] == 'MDE-GMN':
            plot_3d_network(
                net_result, cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / f"network_3d_{net_result['network_id']}.html",
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
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'],
        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'],
        fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    generate_starts = sweep_generate_start(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
        cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'], fingerprints=fingerprints,
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    rebuild_at_best_params(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    for network_id, run in gmn_runs.items():
        if 'G' not in run:
            # rebuild_at_best_params() already logged why and skipped this
            # network -- nothing downstream has anything to work with
            # without G.
            print(f'{network_id}: no successful Generate() rebuild -- skipping '
                  f'Forecast()/fidelity/output for this network', flush=True)
            continue

        forecast_start = time.perf_counter()
        _log_progress(progress_log_path, f'{network_id}: Forecast() diagnostic START')
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
            plot_range_mode=cfg['FORECAST_PLOT_RANGE'],
            generate_validation_start=cfg['GENERATE_VALIDATION_START'], total_rows=len(df),
        )
        _log_progress(progress_log_path,
                      f'{network_id}: Forecast() diagnostic END -- '
                      f'{time.perf_counter() - forecast_start:.1f}s')
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        if Gf is None:
            print(f'{network_id}: no Forecast() rho to report for this network', flush=True)
        else:
            print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
            for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
                print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'],
                                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics -------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        _log_progress(progress_log_path, 'R7 diagnostics START (tp-vs-rho sweep, E/tau grid)')
        r7_start = time.perf_counter()
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
                             cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
                                   cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        _log_progress(progress_log_path,
                      f'R7 diagnostics END -- {time.perf_counter() - r7_start:.1f}s')
        # R19 (docs/review-actions-v1.md): per constructed network -- needs
        # net_result['network']['Graph'] for D (direct-driver count), which
        # mde_results (depth-0-only) doesn't carry. Gated on its own
        # RUN_MDE_EVALUATE, deliberately separate from RUN_MDE_DIAGNOSTICS
        # above (see that config's own docstring) -- R7's two diagnostics
        # just above are unaffected and always run when RUN_MDE_DIAGNOSTICS
        # is set, regardless of this toggle.
        if cfg['RUN_MDE_EVALUATE']:
            for network_id, run in gmn_runs.items():
                r19_start = time.perf_counter()
                _log_progress(progress_log_path, f'{network_id}: R19 run_mde_evaluate START')
                run_mde_evaluate(run['net_result'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                                  cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                                  exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
                _log_progress(progress_log_path,
                              f'{network_id}: R19 run_mde_evaluate END -- '
                              f'{time.perf_counter() - r19_start:.1f}s')
        else:
            # Logged once per run, unconditionally -- a skipped diagnostic
            # that never appears anywhere is exactly how §18.6's swallowed
            # exception went unnoticed for so long (docs/review-findings.md).
            msg = ("R19 run_mde_evaluate SKIPPED by configuration "
                   "(RUN_MDE_EVALUATE=false) -- see docs/review-findings.md's "
                   "'diffusion-map crash and hang' finding for why this defaults off")
            print(msg, flush=True)
            _log_progress(progress_log_path, msg)
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    _log_progress(progress_log_path,
                  f'PIPELINE COMPLETE -- {time.perf_counter() - pipeline_start:.1f}s total')
    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
