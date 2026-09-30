# mde_gmn — what the pipeline is made of

Written for someone reading this codebase for the first time. It says where the
code lives and how the pieces depend on each other; the algorithm itself is
explained in each module's own docstring, starting with `network.py`.

## The two entry scripts

Both are about 45 lines. Each sets `sys.path`, imports the package, and calls
`main()`. They differ in exactly one argument.

| | branch | network builder |
|---|---|---|
| `scripts/src/tsls-mde-gmn.py` | `main` | `network.py` — one MDE call at a time, in breadth-first waves |
| `scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py` | `no-kEDM-gpu-optimization` | `network_parallel.py` — every MDE call up front, then replay |

Everything else — configuration, data, caching, GMN, theta, scoring, output —
is shared.

## The stages, in the order a run executes them

```
       config.py ─── every TSLS_MDE_GMN_* knob, read once
           │
           ▼
   ┌── pipeline.py ── main(): the run, step by step ───────────────────┐
   │                                                                   │
   │   1  tsls_data.py    recording ─► time-since-last-spike, trimmed   │
   │         │                                                          │
   │   2  mde.py          per node: which signals predict it?           │
   │         │            forward selection + a CCM causality gate      │
   │         ▼                                                          │
   │   3  network.py      those answers ─► one acyclic graph            │
   │      (or network_parallel.py, same graph, different schedule)      │
   │         │                                                          │
   │   4  gmn_build.py    graph ─► GMN: node dimensions, the .cfg       │
   │         │                                                          │
   │   5  theta.py        per node: how sharply to weight neighbours    │
   │         │                                                          │
   │   6  generate.py     free-run it; score against held-out data      │
   │         │                                                          │
   │   7  diagnostics.py  optional: baselines, sanity sweeps            │
   │      plots.py        optional: network figures                     │
   └───────────────────────────────────────────────────────────────────┘

   used throughout:  columns.py   delay-suffixed column names
                     caching.py   cross-job result cache + fingerprints
                     progress.py  the heartbeat log
                     gmn_embedding.py  per-node embeddings — AND it patches
                                       gmn when imported (see below)
```

## Who imports whom

A module only imports from modules above it, so this is also a safe reading
order. Derived from the actual imports.

```
  columns    progress    tsls_data    caching    gmn_embedding    plots    diagnostics
     └───────────┴────────────┴───────────┴──────────────┴──────────┴──────────┘
                                   no package dependencies
                                          │
                      mde ────────────────┘   (caching, columns, tsls_data)
                       │
          ┌────────────┴────────────┐
       network                network_parallel      (network_parallel also
          │                        │                 imports network)
          ▼                        │
      gmn_build ◄──────────────────┘
          │
        theta
          │
       generate
          │
       pipeline  ── imports everything above; nothing imports pipeline
```

Outside the package, next to the entry scripts, because the sbatch scripts
stage them as individual files and the environment fingerprint hashes them by
content:

- `cleaned_helper_functions.py` — the TSLS-aware EDM engine (`proper_mde`,
  `proper_ccm`, `proper_simplex`).
- `gpu_edm.py` — the optional CUDA cross-map primitive (GPU branch only).

## Three things that will surprise you

1. **Importing `gmn_embedding` patches `gmn`.** Three `gmn.Node` methods are
   replaced at import time, so that GMN honours this pipeline's per-node
   embeddings instead of one global `(E, tau)`. Anything that constructs `gmn`
   objects must import it first.

2. **`network.py` calls MDE through the module object** — `from mde_gmn import
   mde`, then `mde.run_mde_for_target(...)`. The tests install a fake by
   assigning to `mde_gmn.mde`; importing the name directly would bind the real
   function and the fake would be silently bypassed.
   `tests/_pipeline_test_utils.install_fake_mde()` handles this and raises if a
   module ever has both bindings.

3. **The cache key includes the entry script's path.** `main()` takes
   `entry_path` for that reason: without it, the serial and parallel pipelines
   would share one cache namespace, because every other input to the
   fingerprint is identical.

## Running the tests

There is no pytest in this venv. Each test is a script, and never runs on a
login node:

```bash
srun -p short -t 0:30:00 -c 4 --mem=8G bash -c 'cd /home/j/j-tominaga/pao_rotation && \
  fail=0; for t in tests/test_*.py; do .venv/bin/python3 "$t" >/dev/null 2>&1 || \
  { fail=1; echo "FAIL $t"; }; done; echo "SUITE-EXIT=$fail"'
```

17 files on `main`, 21 on `no-kEDM-gpu-optimization`.
`tests/test_lagged_leaf_nodes.py` is the executable spec for the network
algorithm; `tests/test_parallel_mde_equivalence.py` (GPU branch) asserts that
the parallel builder's `staged` scope produces the serial builder's network,
edge for edge in order.
