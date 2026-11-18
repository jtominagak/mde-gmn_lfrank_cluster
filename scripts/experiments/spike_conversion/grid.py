"""The ablation grid for exp/spike-conversion-neuro: one entry per job.

Stage I  -- every representation at EMBED_TAU=4 (100 ms):
            sawtooth under the new evaluation, sawtooth under the parent's
            event evaluation (to separate the two changes), Model A's 6-point
            grid, Model B's 8-point grid (at Model A's centre time constants),
            Model C's 2-point grid.
Stage II -- the Takens-lag sweep EMBED_TAU in {2, 6, 8} for the sawtooth and
            for the best Stage I configuration of each model family. Added to
            GRID by name once Stage I has picked them (STAGE2_BASES).

`python3 grid.py` prints one line per job: label, then the env assignments
submit_grid.sh exports.
"""

GRID = []


def _add(label, family, stage, **env):
    GRID.append({'label': label, 'family': family, 'stage': stage,
                 'env': {k: str(v) for k, v in env.items()}})


_add('tsls', 'sawtooth', 'I', SPIKE_TRANSFORM='tsls', EMBED_TAU=4)
_add('tsls_events', 'sawtooth', 'I', SPIKE_TRANSFORM='tsls', EMBED_TAU=4, NEURON_EVAL='events')
for ts in (50, 75):
    for tc in (300, 500, 800):
        _add(f'A_s{ts}_c{tc}', 'A_calcium', 'I', SPIKE_TRANSFORM='calcium',
             TAU_SYN_MS=ts, TAU_CA_MS=tc, EMBED_TAU=4)
for tw in (1500, 2500):
    for n in (1.5, 2.0):
        for sig in ('median', 'p75'):
            _add(f'B_w{tw}_n{n}_{sig}', 'B_adapt_nr', 'I', SPIKE_TRANSFORM='adapt_nr',
                 TAU_SYN_MS=50, TAU_CA_MS=500, TAU_W_MS=tw, NR_N=n, NR_SIGMA=sig, EMBED_TAU=4)
for z in (0.7, 0.9):
    _add(f'C_z{z}', 'C_theta_osc', 'I', SPIKE_TRANSFORM='theta_osc', OSC_F0_HZ=8, OSC_ZETA=z,
         EMBED_TAU=4)

# Filled in after Stage I (see REPORT.md for how each was chosen).
STAGE2_BASES = []
for base in STAGE2_BASES:
    src = next(g for g in GRID if g['label'] == base)
    for tau in (2, 6, 8):
        _add(f'{base}_tau{tau}', src['family'], 'II', **{**src['env'], 'EMBED_TAU': tau})


def spike_params(g):
    '''The SPIKE_PARAMS dict config.load_config() would build for entry g.'''
    e = g['env']
    ts = float(e.get('TAU_SYN_MS', 50)) / 1000
    tc = float(e.get('TAU_CA_MS', 500)) / 1000
    return {'TAU_SYN': ts, 'TAU_CA': tc, 'TAU_W': float(e.get('TAU_W_MS', 2500)) / 1000,
            'ADAPT_A': 1.0, 'ADAPT_B': ts * tc, 'NR_LAMBDA': 1.0,
            'NR_N': float(e.get('NR_N', 2.0)), 'NR_SIGMA': e.get('NR_SIGMA', 'median'),
            'OSC_F0': float(e.get('OSC_F0_HZ', 8)), 'OSC_ZETA': float(e.get('OSC_ZETA', 0.7))}


if __name__ == '__main__':
    import sys
    stages = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else {'I', 'II'}
    for g in GRID:
        if g['stage'] in stages:
            print(g['label'], ' '.join(f'{k}={v}' for k, v in g['env'].items()))
