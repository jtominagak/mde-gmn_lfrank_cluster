"""mde_gmn -- the MDE-GMN pipeline, split into one module per job.

Two entry scripts run this package. They differ ONLY in which network builder
they hand to the orchestrator:

    scripts/src/tsls-mde-gmn.py                          -> network (serial)
    scripts/src/no-kedm-optimization-...-tsls-mde-gmn.py -> network_parallel
                                                            (GPU branch only)

WHAT THE PIPELINE DOES, in order. Each module's own docstring says which step
it is, and explains the idea before the mechanics.

    1. tsls_data      read the recording, convert spike trains to
                      time-since-last-spike signals, trim the burn-in
    2. mde            per node, pick the drivers that best predict it
                      (forward selection + a CCM causality gate)
    3. network        assemble those per-node results into one acyclic network:
                      real observables form the backbone, lagged observables
                      attach as leaf nodes
    4. gmn_build      hand the network to GMN: per-node dimensions, the .cfg
    5. theta          tune each node's S-map theta
    6. generate       free-run the network, score it against held-out data
    7. diagnostics    optional figures and baseline comparisons

network_parallel (GPU branch only) is an alternative to step 3: the same
network, built by computing every MDE call up front instead of one wave at a
time. Its docstring explains the trade-off.

Supporting modules: config (every environment knob), pipeline (main(), the run
step by step), columns (delay-suffixed column names), caching (the cross-job
result cache and its fingerprints), progress (the heartbeat log), plots
(network figures).

DEPENDENCY DIRECTION -- a module only imports from the ones above it, so this
list is also a safe reading order. Verified against the imports, not aspirational.

    columns, progress, tsls_data, caching, gmn_embedding, plots, diagnostics
                                 no dependency on any other module here
    mde                          -> caching, columns, tsls_data
    network                      -> mde, columns, progress, tsls_data
    network_parallel             -> network, mde, caching, progress
    gmn_build                    -> network, caching, progress, tsls_data
    theta                        -> gmn_build, gmn_embedding, network, caching,
                                    progress, tsls_data
    generate                     -> theta, gmn_build, gmn_embedding, caching,
                                    progress, tsls_data

There are no import cycles: no module imports an entry script, and the network
builder is passed in as an argument rather than imported.

TWO THINGS THAT WILL SURPRISE YOU
  * Importing gmn_embedding PATCHES gmn (three gmn.Node methods, at import
    time). Anything that builds gmn objects must import it first. Its docstring
    explains why.
  * network reaches MDE as `mde.run_mde_for_target(...)`, through the module
    object, because the tests install a fake by assigning to the mde module.
    Importing the name directly would bypass the fake silently.

`cleaned_helper_functions` (the TSLS-aware EDM engine: proper_mde, proper_ccm,
proper_simplex) and `gpu_edm` (the optional CUDA CCM primitive) sit OUTSIDE
this package, next to the entry scripts, because the sbatch scripts stage them
as individual files and the environment fingerprint hashes them by content.
"""
