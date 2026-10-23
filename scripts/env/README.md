# `scripts/env/` — the parts of the cluster venvs that are not reproducible

The cluster venvs are 3.2 GB (`.venv`) and 8.6 GB (`.venv-gpu`), almost
entirely binaries: 4.5 GB of NVIDIA CUDA libraries, 2.5 GB of torch and 1.2 GB
of triton in the GPU venv alone. None of that belongs in git and none of it
runs on a laptop.

But two things inside those venvs **are** substance, and until 2026-10-23 they
existed in no other place. If a venv had been deleted or rebuilt, they were
gone, and nothing in git could restore them. They are now here.

## `sitecustomize.py` — required, not optional

202 lines that Python's `site` module auto-imports at interpreter startup for
every process using the venv. It is **not** importable from the repo: it only
takes effect when installed into the venv's `site-packages`. It fixes two
things, both of which cost real money when missing:

1. **The pyEDM per-call `Pool` bug.** `CCM`, `EmbedDimension`,
   `PredictInterval` and `PredictNonlinear` each build a brand-new
   multiprocessing Pool — with a full re-import of numpy/scipy/pandas/sklearn/
   pyEDM/gmn per worker — on *every single call*. This pipeline's call sites
   only ever dispatch a handful of tiny per-value tasks, so spawn and teardown
   dwarf the work. Symptom: jobs running ~2 days regardless of dataset size,
   `MODE` or `MDE_D`, and `"No module named 'kedm'"` printed hundreds of
   thousands of times in one `.out` log. The file forces all four onto their
   sequential in-process paths.
2. **The `FindNeighbors()` warning flood.** pyEDM emits a `UserWarning` whenever
   a prediction finds fewer valid neighbors than requested. This pipeline varies
   `exclusionRadius` on nearly every call, so the interpolated values in the
   message text defeat Python's exact-text warning dedup and every occurrence
   prints. Symptom: **68–71 GB `.err` files** and a plausible contribution to an
   observed `oom_kill`. The file filters that one warning.

It must patch at interpreter startup rather than inside pipeline code because
`simple-mde-gmn.py` launches `python -m apps.InteractionMatrix` as a
**subprocess**, which re-imports pyEDM from disk unpatched no matter what the
parent did in its own memory. `sitecustomize.py` is the only location that
reaches both.

### Installing it

```bash
cp scripts/env/sitecustomize.py \
   "$(.venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')/"
# verify
.venv/bin/python3 -c "import sitecustomize; print(sitecustomize.__file__)"
```

`scripts/saion/setup_gpu_venv.sh` (on the `no-kEDM-gpu-optimization` branch,
along with the rest of `scripts/saion/`) copies it from the CPU venv into the
GPU venv and **aborts** if the CPU venv does not have it:

```
no sitecustomize.py in <site> -- the pyEDM Pool fix is missing from the CPU venv too
```

So the CPU venv is the source of truth for the GPU venv, and this directory is
now the source of truth for the CPU venv. `scripts/batch/saion-preflight.sbatch`
checks it is importable before any real job depends on it.

## The frozen manifests

`requirements-cpu-frozen.txt` (38 packages) is what was actually installed in
`.venv` on 2026-10-23, read from `importlib.metadata` — the uv venvs have no
`pip`, so `pip freeze` returns nothing. The GPU counterpart,
`requirements-gpu-frozen.txt` (58 packages, including `torch==2.14.0+cu126`
and `triton==3.8.0`), is on the `no-kEDM-gpu-optimization` branch, where the
GPU venv it describes is built.

These are **not** a substitute for `uv.lock`, which is tracked and remains
authoritative for resolution. They exist because **five of these versions are
part of the cache fingerprint**:

| | version |
|---|---|
| `gmn` | 1.5.1 |
| `pyEDM` | 2.5.0 |
| `networkx` | 3.6.1 |
| `numpy` | 2.4.6 |
| `pandas` | 3.0.3 |
| `torch` | 2.14.0+cpu / 2.14.0+cu126 |

`_environment_fingerprint()` in `mde_gmn/caching.py` reads those five with
`importlib.metadata.version()` and embeds them in every cache key. Bumping any
one of them **invalidates the entire `/bucket` cache** — every cached MDE
selection, network, theta search and start sweep misses, and the next job
recomputes from scratch. That is a days-of-compute decision, which is why the
numbers are written down where they can be read without a cluster login.

`torch` is listed for completeness; it is not in the fingerprint.
