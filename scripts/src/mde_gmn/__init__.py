"""mde_gmn -- the MDE-GMN pipeline, split into one module per job.

Two entry scripts run this package. They differ ONLY in which network builder
they hand to the orchestrator:

    scripts/src/tsls-mde-gmn.py                       -> network (serial)
    scripts/src/no-kedm-optimization-...-tsls-mde-gmn.py -> network_parallel

WHAT THE PIPELINE DOES, in order
    1. tsls_data    read the recording, convert spike trains to
                    time-since-last-spike signals, trim the burn-in
    2. mde          per node, pick the drivers that best predict it
                    (forward selection + a CCM causality gate)
    3. network      assemble those per-node results into one acyclic network:
                    real observables form the backbone, lagged observables
                    attach as leaf nodes
    4. gmn_build    hand the network to GMN: per-node embeddings, config
    5. theta        tune each node's SMap theta
    6. generate     free-run the network, score it against held-out data
    7. diagnostics  optional figures and baseline comparisons

DEPENDENCY DIRECTION (a module only imports from the ones above it)
    columns, progress            no package dependencies
    tsls_data, caching           -> cleaned_helper_functions
    mde                          -> columns, tsls_data, caching, progress
    network, network_parallel    -> mde, columns, caching, progress
    gmn_embedding                -> columns          (patches gmn on import)
    gmn_build                    -> columns, caching, gmn_embedding
    theta, generate              -> gmn_embedding, gmn_build, caching, progress
    diagnostics                  -> columns, tsls_data
    config                       no package dependencies
    pipeline                     -> everything above

Nothing imports `pipeline`, and no module imports an entry script: the
builders are passed in as arguments, so there are no import cycles.

`cleaned_helper_functions` (the TSLS-aware EDM engine: proper_mde, proper_ccm,
proper_simplex) and `gpu_edm` (the optional CUDA CCM primitive) sit OUTSIDE
this package, next to the entry scripts, because the sbatch scripts stage
them as individual files.
"""
