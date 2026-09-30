"""Column naming for delayed (lagged) observables.

A candidate driver is an (observable, delay) pair. Pairs live in the
DataFrame as columns named "<observable>::delay-<n>" (delay 0 keeps the raw
name), and this module is the ONLY place that spelling is constructed or
parsed -- everything else goes through observable_and_delay() /
_delayed_column_name(), so the format can change in one place.

Depends on: nothing in this package (stdlib + pandas via the caller's frame).
Imported by: mde, network, network_parallel, gmn_embedding, gmn_build,
    diagnostics, pipeline.
"""


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
