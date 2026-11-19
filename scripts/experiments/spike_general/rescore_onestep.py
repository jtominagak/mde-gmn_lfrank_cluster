"""Re-run only the one-step Forecast() of finished Stage II jobs.

Jobs staged before commit 773e99d scored one-step with Forecast() rows
mislabelled by each node's embedding offset (see
tests/test_forecast_offset_alignment.py). Their network, theta, Generate()
rollouts and topology are unaffected, so instead of rerunning a job this
loads its archived network, data and tuned theta from /bucket and reruns
autonomous_eval.run_one_step() with the fixed code.

Usage (compute node -- this runs S-map):
  python rescore_onestep.py <job_id> [<job_id> ...]
Writes docs/experiments/spike-conversion-general/rescored/onestep_<job_id>.csv,
which aggregate.py prefers over the job's own onestep file.
"""
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path, PosixPath  # noqa: F401  (PosixPath: eval of the logged Config)

import pandas as pd

HERE = Path(__file__).resolve()
P = HERE.parents[3]
sys.path.insert(0, str(P / 'scripts' / 'src'))
from mde_gmn.autonomous_eval import run_one_step  # noqa: E402

OUTPUTS = Path('/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs')
NET = 'head_position_x_head_position_y'
OUT = P / 'docs/experiments/spike-conversion-general/rescored'


def rescore(job, cores):
    d = OUTPUTS / f'job_{job}'
    log = next(d.glob('spk_gen-*.out'))
    line = next(l for l in log.read_text(errors='replace').splitlines() if l.startswith('Config: {'))
    cfg = eval(line[len('Config: '):], {'PosixPath': PosixPath, 'range': range})
    with open(next(d.glob(f'Network_nestedMDE_*_{NET}.pkl')), 'rb') as f:
        network = pickle.load(f)
    theta = json.loads((d / f'best_params_{NET}.json').read_text())['best_node_theta']
    net_result = {'targets': cfg['TARGETS'], 'network_id': NET, 'gmn_tp': cfg['GMN_TP'],
                  'DATA_AFTER_MDE': d / f'data_after_mde_{NET}.csv',
                  'NETWORK_FILE': next(d.glob(f'Network_nestedMDE_*_{NET}.pkl')),
                  'network': network}
    actual = pd.read_csv(net_result['DATA_AFTER_MDE'])
    with tempfile.TemporaryDirectory() as tmp:
        one = run_one_step(net_result, theta, cfg, cfg['GENERATE_VALIDATION_START'], actual,
                           network['Graph'], Path(tmp), cores)
    one['external_input_mode'] = network['Graph'].graph.get('external_input_mode', 'real_data')
    one['rescored'] = True
    OUT.mkdir(parents=True, exist_ok=True)
    one.to_csv(OUT / f'onestep_{job}.csv', index=False)
    driven = one[one['n_drivers'] > 0]
    print(f'job {job}: driven nodes r median {driven["r"].median():.3f}, '
          f'skill median {driven["skill_vs_persistence"].median():.3f}; '
          f'targets r {one.loc[one.role == "target", "r"].round(3).tolist()}', flush=True)


if __name__ == '__main__':
    import os
    cores = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    for j in sys.argv[1:]:
        rescore(j, cores)
