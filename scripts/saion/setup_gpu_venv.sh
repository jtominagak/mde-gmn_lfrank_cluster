#!/bin/bash
# =============================================================================
# setup_gpu_venv.sh -- complete /home/j/j-tominaga/pao_rotation/.venv-gpu for the Saion GPU jobs
# =============================================================================
# Run on a gpu-p100 COMPUTE node, per the guide ("You should build your
# application on a GPU compute node"; "build on a gpu-p100 node using CUDA no
# newer than 12.8" so it also works on gpu-v100). No GPU needed to install:
#
#     srun -p gpu-p100 -t 0-1 -c 4 --mem=16G --pty bash -l
#     bash /home/j/j-tominaga/pao_rotation/scripts/saion/setup_gpu_venv.sh
#
# If that node cannot reach the internet (downloads fail), exit the srun
# shell and run the same command on the login node instead: installing
# pre-built wheels is data movement, not compilation.
#
# Why this exists: .venv-gpu was built for the standalone GPU prototypes and
# only ever got numpy/pandas/pyEDM/torch. The real pipeline also imports gmn,
# dimx (apps.*), pydiffmap, sklearn, networkx, matplotlib, ... and -- just as
# important -- relies on .venv's sitecustomize.py (the pyEDM per-call Pool
# spawn fix; without it jobs slow down by an order of magnitude).
#
# What it does:
#   1. Pins every package to the exact version in the CPU .venv (uv pip
#      freeze), EXCEPT torch/triton/nvidia-*/cuda-* which must be CUDA builds.
#   2. Installs torch from PyTorch's cu126 index. cu126 is deliberate: P100
#      nodes top out at CUDA 12.8, and cu128+ wheels dropped Pascal (sm_60).
#      The same wheel runs on the V100 nodes, which is what lets
#      scripts/saion/submit_saion.sh send a job to either partition.
#   3. Copies sitecustomize.py across.
#
# Never touches pyproject.toml / uv.lock (shared with Deigo via /home/j/j-tominaga) -- only
# `uv pip ... --python .venv-gpu/bin/python`, which bypasses project mode.
# Safe to re-run: already-satisfied pins are no-ops.
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/j/j-tominaga/pao_rotation"
UV="/home/j/j-tominaga/.local/bin/uv"
CPU_VENV="$PROJ_DIR/.venv"
GPU_VENV="$PROJ_DIR/.venv-gpu"
TORCH_SPEC="torch==2.14.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$UV" ] || die "uv not found at $UV"
[ -x "$CPU_VENV/bin/python" ] || die "CPU venv missing at $CPU_VENV -- it is the version source of truth"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "NOTE: running inside Slurm job $SLURM_JOB_ID. If downloads fail, this node has no internet -- rerun on the login node." >&2
fi

if [ ! -x "$GPU_VENV/bin/python" ]; then
    echo "== creating $GPU_VENV"
    "$UV" venv "$GPU_VENV" --python 3.11
fi

# Inside the venv dir, not the login node's shared /tmp.
reqs="$GPU_VENV/.saion-pinned-requirements.txt"
trap 'rm -f "$reqs"' EXIT
"$UV" pip freeze --python "$CPU_VENV/bin/python" \
    | grep -v -E '^(torch|torchvision|torchaudio|triton|nvidia-|cuda-)' \
    | grep -v -E '^(-e |pao-rotation)' \
    > "$reqs"
echo "== pinning $(wc -l < "$reqs") packages from $CPU_VENV"
"$UV" pip install --python "$GPU_VENV/bin/python" -r "$reqs"

echo "== installing $TORCH_SPEC from $TORCH_INDEX"
"$UV" pip install --python "$GPU_VENV/bin/python" "$TORCH_SPEC" --index-url "$TORCH_INDEX"

site_cpu="$("$CPU_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
site_gpu="$("$GPU_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -f "$site_cpu/sitecustomize.py" ] || die "no sitecustomize.py in $site_cpu -- the pyEDM Pool fix is missing from the CPU venv too"
cp -v "$site_cpu/sitecustomize.py" "$site_gpu/sitecustomize.py"

# Static checks only (no imports of torch/CUDA here -- login nodes have no
# CUDA libraries; the real import/CUDA checks are in saion-preflight.sbatch).
echo "== verifying"
"$GPU_VENV/bin/python" - <<'EOF'
import importlib.metadata as md, importlib.util, sys
torch_v = md.version('torch')
missing = [m for m in ('gmn', 'dimx', 'apps', 'pydiffmap', 'pyEDM', 'sklearn', 'networkx',
                       'matplotlib', 'pandas', 'numpy', 'scipy', 'statsmodels', 'torch')
           if importlib.util.find_spec(m) is None]
problems = []
if missing:
    problems.append(f'missing modules: {missing}')
if not torch_v.endswith('+cu126'):
    problems.append(f'torch is {torch_v}, expected a +cu126 build')
if importlib.util.find_spec('sitecustomize') is None:
    problems.append('sitecustomize.py not importable')
print('torch', torch_v)
if problems:
    print('FAILED:', *problems, sep='\n  ')
    sys.exit(1)
print('OK -- .venv-gpu is complete. Next: sbatch scripts/batch/saion-preflight.sbatch')
EOF
