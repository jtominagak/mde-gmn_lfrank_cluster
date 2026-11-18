# `env/` — the one part of the environment that is code

`sitecustomize.py` must be installed into **every** venv that runs this
pipeline. Without it, nothing fails; the run just takes days instead of hours
(see the file's own docstring for why).

It only takes effect from the venv's `site-packages`. Importing it from here
does nothing.

```bash
# CPU venv
cp clean_scripts/env/sitecustomize.py \
   "$(.venv/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"
.venv/bin/python3 -c "import sitecustomize; print(sitecustomize.__file__)"
```

You do not need to install it into the GPU venv by hand:
`saion/setup_gpu_venv.sh` installs it from this directory.
