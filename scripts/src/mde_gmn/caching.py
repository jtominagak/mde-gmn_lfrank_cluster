"""Cross-job caching: fingerprints, lookup, and writes.

Expensive deterministic stages (per-node MDE, network construction, theta
search, the generate-start sweep) are cached under a fingerprint so a
resubmitted job reuses them instead of recomputing. A fingerprint combines
the stage's own parameters with build_code_fingerprint(), which hashes THE
WHOLE PIPELINE PACKAGE plus the installed dependency versions -- so a real
code or dependency change misses the cache rather than silently reusing a
result computed by different code. Cosmetic edits (docstrings, comments) do
not, because the hash is over a docstring-stripped AST.

Depends on: cleaned_helper_functions (content-hashed, not called).
Imported by: mde, network, network_parallel, gmn_build, theta, generate,
    pipeline.
"""

import ast
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import cleaned_helper_functions as tsls


def _hash_file(path, chunk_size=1 << 20):
    '''sha256 of a file's actual bytes.'''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def _strip_docstrings(tree):
    '''Removes every module/class/function docstring from an AST in place
   .'''
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return tree

def _pipeline_code_fingerprint(entry_path=None):
    """SHA-256 over the whole pipeline's code, docstrings stripped.

    Covers every module in this package plus `entry_path` (the entry script
    that is running -- the serial and parallel pipelines therefore get
    different fingerprints, so they never share cache entries even where
    their per-node calls would agree).

    Docstrings are stripped before hashing, so comment/docstring edits reuse
    the cache while any real change to a function body invalidates it. The
    whole package is hashed rather than one file: with the code split across
    modules, hashing only the entry script would let a genuine change in, say,
    mde.py silently reuse results computed by the old version.
    """
    digest = hashlib.sha256()
    paths = sorted(Path(__file__).resolve().parent.glob('*.py'))
    if entry_path is not None:
        paths.append(Path(entry_path).resolve())
    for path in paths:
        tree = ast.parse(path.read_text())
        _strip_docstrings(tree)
        digest.update(path.name.encode())
        digest.update(ast.dump(tree).encode())
    return digest.hexdigest()


def _environment_fingerprint():
    '''The versions of everything numerical a cached result depended on.

    If a dependency that does the arithmetic changes, a cached result may no
    longer be what the current code would produce, so its version belongs in
    the fingerprint. `cleaned_helper_functions` is not an installed package, so
    it is hashed by content instead -- as is `apps.CreateNetwork`, used by
    MODE='GMN'.'''
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

def build_code_fingerprint(entry_path=None):
    '''The single code+environment value every cache fingerprint embeds.

    Computed once per job by the entry script (which passes its own
    __file__ as `entry_path`), gated by TSLS_MDE_GMN_REUSE_BEST_PARAMS, then
    threaded down as a plain argument rather than recomputed per call.'''
    return {'source': _pipeline_code_fingerprint(entry_path),
            'environment': _environment_fingerprint()}

def build_search_fingerprint(cfg, net_result, data_file_sha256, code_fingerprint):
    '''Everything that determines one network's theta search outcome
    (run_theta_search()). The TSLS scoring and CCM-calibration knobs are part
    of it: they shape which drivers get selected just as much as MDE_D or
    NESTED_M do.'''
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
        'embed_dim_rho_min_f1': cfg['EMBED_DIM_RHO_MIN_F1'],
        'cross_map_rho_min_f1': cfg['CROSS_MAP_RHO_MIN_F1'],
        'exclusion_radius': cfg['EXCLUSION_RADIUS_VALUE'],
        'gmn_exclusion_radius': cfg['GMN_EXCLUSION_RADIUS'],
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
        # Does not change the network's topology -- that is
        # build_network_fingerprint()'s concern -- but does change what
        # Generate() and the theta search do with a zero-driver node, so it
        # belongs in this fingerprint rather than that one.
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
                               gmn_mode=None, multi_mde_e=None, multi_mde_target_aggregate=None,
                               target_ranking_window_rows=1000, min_pred_events=20,
                               max_window_rows=30000,
                               ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                               builder_kwargs=None):
    '''Everything that determines one group's network-construction outcome.
    Computable BEFORE that construction runs, since the whole point is deciding
    whether it can be skipped. `observable_columns` here is
    already the CCM-pre-filtered pool (main() runs the pre-filter, and its
    own threshold/E/libSize-ratio knobs, before this fingerprint is ever
    computed -- see build_network()'s own docstring), so -- matching this
    function's own original design philosophy, predating the pre-filter --
    the resolved list alone captures everything a cache needs to
    distinguish, the same way it already does for REGION_CHANNEL_CAP; no
    separate pre-filter fields needed here.

    builder_kwargs: the options passed to the network builder itself, which
    build_network() does not otherwise know the meaning of -- today that is the
    parallel builder's `parallel_mde_scope`. These change WHICH NETWORK gets
    built, so they have to be here: 'staged' and 'union' agree on every other
    field in this fingerprint (same targets, same observables, same entry
    script, therefore the same code fingerprint), so without this a 'union'
    run would match a 'staged' run's cached network, print "matched cached
    network", skip the whole MDE phase and report the other algorithm's result.
    Omitted from the dict entirely when empty, so the serial pipeline's
    existing cache entries still match; the parallel fork's older entries do
    not, which is correct -- they were written without recording which scope
    produced them, so they cannot be attributed to either.

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
    `ccm_lib_sizes_rows`: MODE='MDE-GMN' only -- `build_nested_mde_network()`'s own per-node
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
        **({'builder_kwargs': {k: builder_kwargs[k] for k in sorted(builder_kwargs)}}
           if builder_kwargs else {}),
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
    computable before that call runs. `candidate_pool` is kept in
    order rather than sorted -- forward selection can tie-break on column
    order, so the same columns in a different order aren't guaranteed to
    select the same result. `max_e` (the raw MDE_D-derived auto-select
    ceiling proper_embed_dimension's own sweep is capped at, distinct from
    `mde_d`'s possibly num_delays-scaled value -- see run_mde_for_target's
    own docstring) is a real input to the outcome now that E is
    auto-selected rather than a hardcoded constant, so it must be part of
    the fingerprint too: a run with a different max_e can select a
    different E, hence different candidates, for the same everything else.

    `window_rows`/`pred_event_count` are the RESOLVED, data-dependent outputs
    of
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
    `window_rows`: the resolved window itself, not just the
    `target_ranking_window_rows` constant that produced it -- for the same
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
    '''Generic cross-job cache lookup, shared by every cacheable stage.'''
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
    '''writes once per unique
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

def build_start_sweep_fingerprint(network_fingerprint, node_theta, start_offsets,
                                   generate_length, exclusion_radius):
    '''Everything that determines one network's start-point sweep outcome
   . network_fingerprint already
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
