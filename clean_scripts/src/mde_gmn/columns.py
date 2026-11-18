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

    Falls back to a marker-presence check when `column_name` is not in
    `raw_columns` at all. That used to happen on every run, because depth-0's
    mde_results entry was computed against the full, unfiltered candidate pool
    before ccm_prefilter_observables() ran -- main() now pre-filters first and
    seeds depth-0 from the same filtered pool the builder uses, so the mismatch
    no
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
                    generate_length, mde_lib_size, lag_count=None, min_lib_rows=50):
    '''Resolves the nonzero-delay set used for Stage 2's lagged-driver
    candidate pool (build_nested_mde_network()'s own Stage-2 pass) and,
    unioned with {0}, every node's unconditional self_delay_set -- see
    config.py's DELAY_SOURCE/LAG_MODE/LAG_DENSITY/LAG_STEP/LAG_COUNT entries
    for the parameter semantics.

    THE THREE SOURCES DIFFER ONLY IN WHICH QUANTITY YOU PIN. A lag set is a
    stride and how far back it reaches; fix any two of {stride, count, depth}
    and the third follows. DELAY_SOURCE chooses which one you state:

      'lag_mode'  pins the DEPTH; the count follows from it. Auto-derived,
        so the count is whatever the data implies -- which is how it gets
        large without anyone asking (see WHY 'tau_count' EXISTS below).
          lag_mode='generation_length_adjusted': depth = generate_length.
          lag_mode='every_lag_possible': depth = mde_lib_size - min_lib_rows
            (clamped to >= 1) -- as deep as the MDE library segment [0, A)
            structurally allows while reserving min_lib_rows real rows so
            MDE's own forward-selection still has a meaningful library left
            after the deepest lag's warm-up NaN rows are skipped.
        That depth is then sampled every lag_step rows (lag_density=
        'strided') or at every single integer lag ('dense').

      'tau_count' pins the COUNT; the depth follows: exactly lag_count lags
        spaced lag_step apart (lag_step is the tau), so the deepest lag is
        lag_step * lag_count. lag_density does not apply here -- the stride
        is always lag_step, so lag_step=1 gives consecutive lags.

      'explicit'  pins EVERY VALUE: the hand-picked explicit_delay_set list,
        stripped of any 0 entry.

    WHY 'tau_count' EXISTS. Under 'lag_mode' the count is a side effect of a
    knob chosen for an unrelated reason: 'generation_length_adjusted' ties it
    to GENERATE_LENGTH, which is how many steps Generate() free-runs for. The
    two quantities have no reason to agree, and the coupling has already cost
    real jobs -- GENERATE_LENGTH=400 with lag_step=4 resolves to 100 lags per
    observable, and leaving GENERATE_LENGTH at its 0 (= full held-out window)
    default resolved to roughly 830. Stage 2's candidate pool is
    (observables x lags) and Stage 2 is where essentially all the runtime
    goes, so that count is a compute budget. 'tau_count' lets you state the
    budget directly instead of back-solving GENERATE_LENGTH for it.

    Returns a sorted-descending list of negative ints -- never a 0 entry, the
    same convention on all three paths.'''
    if delay_source == 'explicit':
        return sorted((d for d in explicit_delay_set if d != 0), reverse=True)

    if delay_source == 'tau_count':
        if lag_count is None or lag_count < 1:
            raise ValueError(
                f"delay_source='tau_count' needs lag_count >= 1, got {lag_count!r}")
        return list(range(-lag_step, -lag_step * lag_count - 1, -lag_step))

    if delay_source != 'lag_mode':
        raise ValueError(f"Unknown delay_source: {delay_source!r}")

    if lag_mode == 'generation_length_adjusted':
        depth = max(1, generate_length)
    elif lag_mode == 'every_lag_possible':
        depth = max(1, mde_lib_size - min_lib_rows)
    else:
        raise ValueError(f"Unknown lag_mode: {lag_mode!r}")

    if lag_density not in ('strided', 'dense'):
        raise ValueError(f"Unknown lag_density: {lag_density!r}")
    step = lag_step if lag_density == 'strided' else 1

    return list(range(-1, -depth - 1, -step))

def effective_mde_dimension(mde_d, candidate_columns):
    '''The D handed to proper_mde: MDE_D, capped at the size of the pool.

    D is the number of forward-selection rounds, so it caps how many columns
    a node selects, and each selected column becomes one coordinate of that
    node's S-map embedding. It is a cap on embedding dimension, and it means
    the same thing with or without lags: a lagged column counts against it
    like any other. (It used to be multiplied by the number of delays per
    observable; see docs/parameters.md, MDE_D, for why that was dropped.)'''
    return min(mde_d, len(candidate_columns))
