'''TSLS nested-MDE + GMN pipeline -- the parallel-MDE entry point.

Run by scripts/batch/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch
(and its Saion sibling). Every parameter comes from a TSLS_MDE_GMN_*
environment variable, so nothing is configured by editing this file.

WHERE THE CODE IS
    All of it is in the mde_gmn package next to this file. Start with
    mde_gmn/__init__.py for the module map, and mde_gmn/pipeline.py's main()
    for the run itself.

HOW THIS DIFFERS FROM tsls-mde-gmn.py
    In exactly one respect: the network builder. The serial pipeline runs MDE
    one breadth-first wave at a time, so every wave ends in a barrier that
    leaves most of the machine idle. This one uses
    mde_gmn/network_parallel.py, which computes MDE for every node that could
    be needed up front with no barriers, then replays the graph walk against
    those results. Read that module's docstring for the trade-off (wasted
    compute bought with wall-clock) and for PARALLEL_MDE_SCOPE, the one
    configuration knob the serial pipeline does not have.

    Everything else -- config, data, caching, GMN, theta, scoring, output --
    is the same shared code, which is why this file is a dozen lines.

WHY IT IS A SEPARATE FILE AND NOT A FLAG
    The two pipelines keep separate result caches, and the cache key includes
    this file's path. Keeping them apart means a run of one can never consume
    the other's cached MDE results.
'''

import sys
from pathlib import Path

# cleaned_helper_functions.py, gpu_edm.py and the mde_gmn package are not
# installed; the sbatch script copies them next to this file, so this directory
# has to be importable before the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mde_gmn import network_parallel  # noqa: E402 -- import after the sys.path edit
from mde_gmn.pipeline import main  # noqa: E402


if __name__ == "__main__":
    # __file__ goes into the code fingerprint, which is what keeps this
    # pipeline's cached results separate from the serial one's.
    main(
        __file__,
        network_builder=network_parallel.build_nested_mde_network,
        builder_config_keys={'parallel_mde_scope': 'PARALLEL_MDE_SCOPE'},
    )
