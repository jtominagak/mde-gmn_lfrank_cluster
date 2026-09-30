'''TSLS nested-MDE + GMN pipeline -- the serial entry point.

Run by scripts/batch/tsls-mde-gmn.sbatch. Every parameter comes from a
TSLS_MDE_GMN_* environment variable (the sbatch script's MODIFIABLE PARAMETERS
block is the list), so nothing is configured by editing this file.

WHERE THE CODE IS
    All of it is in the mde_gmn package next to this file. Start with
    mde_gmn/__init__.py: it names every module, says what each stage does, and
    gives a safe reading order. mde_gmn/pipeline.py's main() is the run itself,
    step by step.

WHAT THIS SCRIPT ADDS
    Nothing but a choice. It runs the pipeline with the SERIAL network builder,
    which issues one MDE call at a time in breadth-first waves. Its sibling on
    the no-kEDM-gpu-optimization branch,
    no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py, passes the parallel
    builder instead and is otherwise the same few lines.

WHAT 'TSLS' MEANS HERE
    The input is a 40 Hz spike-train recording. Each channel is converted to
    time-since-last-spike (TSLS): at every sample, how long since that channel
    last fired. That turns a sparse train of events into a continuous signal
    delay embedding can work on. Because scores on such signals cannot be
    judged by plain Pearson correlation, every scoring call goes through
    cleaned_helper_functions.py's TSLS-aware metrics instead of pyEDM's
    defaults -- see mde_gmn/config.py's METRIC entry for why.
'''

import sys
from pathlib import Path

# cleaned_helper_functions.py and the mde_gmn package are not installed; the
# sbatch script copies them next to this file, so this directory has to be
# importable before the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mde_gmn.pipeline import main  # noqa: E402 -- import after the sys.path edit above


if __name__ == "__main__":
    # __file__ goes into the code fingerprint, which is what keeps this
    # pipeline's cached results separate from the parallel one's.
    main(__file__)
