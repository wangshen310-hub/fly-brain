"""Native causal sign audit of the engineered two-hop eligibility surrogate.

The task is motor controllability, not choosing a correct Snake action. No
reward labels are used, no diagnostic vector controls a game, and no training
directory is written. Finite differences are actual full-network spikes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.centered_learning import CenteredCreditBrain as CoveredNodeBrain
from flysnake.research.paired import digest, writer_lock
from flysnake.research.projection import CoveredGame


def observe(brain, rgb):
    brain.reset_trial()
    receipt, _ = brain.observe(rgb)
    counts = [receipt['counts'][a][0] for a in ('left', 'right')]
    return receipt, np.array(counts, dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scenes', type=int, default=6)
    p.add_argument('--amplitude', type=float, default=.1)
    args = p.parse_args()
    if args.scenes<1 or not 0<args.amplitude<=1:
        raise ValueError('Positive scenes and bounded intervention required')
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'
        if path.exists():
            raise ValueError('Preserve existing audit')
        report = {'protocol': 'native-centered-credit-causal-sign-audit-v1',
                  'status': 'loading', 'amplitude': args.amplitude, 'rows': [],
                  'interpretation': 'Causal motor sign audit only. No learned behavior or exact gradient claim.'}
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        atomic_json(path, report)
        try:
            brain = CoveredNodeBrain()
            training = json.loads((args.run/'status.json').read_text(encoding='utf-8'))
            activity = training['activity_baseline']
            brain.restore_activity_baseline(args.run/activity['file'], activity['sha256'])
            report['brain'] = brain.identity
            zeros = np.zeros_like(brain.theta)
            for seed in range(930201, 930201+args.scenes):
                game = CoveredGame(seed=seed, size=4, geometry=brain.geometry)
                rgb = game.render_observation()
                brain.set_parameters(zeros)
                baseline, baseline_counts = observe(brain, rgb)
                native_counts = brain.brain.last['counts'].copy()
                # A fixed left-current/right-current contrast independent of
                # pixels or game correctness; only tests the declared slope.
                direction = brain.credit_from_counts(native_counts, np.array([1., -1.]))
                row = {'seed': seed, 'image_sha256': baseline['observation_sha256'],
                       'baseline_counts': baseline_counts.tolist(), 'interventions': {}}
                for locus, mask in (('direct', brain.direct), ('upstream', ~brain.direct),
                                    ('combined', np.ones(len(zeros), dtype=bool))):
                    vector = np.where(mask, direction, 0.)
                    scale = float(np.max(abs(vector)))
                    if scale==0:
                        row['interventions'][locus] = {'status': 'no_eligibility'}
                        continue
                    vector *= args.amplitude/scale
                    actual = []
                    for sign in (1, -1):
                        brain.set_parameters(sign*vector)
                        receipt, counts = observe(brain, rgb)
                        if receipt['observation_sha256']!=row['image_sha256']:
                            raise RuntimeError('Intervention image changed')
                        actual.append(counts)
                    contrast = float((actual[0][0]-actual[0][1])-(actual[1][0]-actual[1][1]))
                    row['interventions'][locus] = {'positive_counts': actual[0].tolist(),
                                                 'negative_counts': actual[1].tolist(),
                                                 'left_minus_right_spike_change': contrast,
                                                 'predicted_sign': 1,
                                                 'direction_sha256': digest(vector),
                                                 'nonzero_edges': int(np.count_nonzero(vector))}
                    report.update(status='running', current_seed=seed, current_locus=locus)
                    atomic_json(path, report)
                    print(json.dumps({'seed': seed, 'locus': locus, 'spike_contrast_change': contrast}), flush=True)
                report['rows'].append(row)
                atomic_json(path, report)
            report['summary'] = {}
            for locus in ('direct', 'upstream', 'combined'):
                changes = [r['interventions'][locus]['left_minus_right_spike_change'] for r in report['rows']
                           if 'left_minus_right_spike_change' in r['interventions'][locus]]
                report['summary'][locus] = {'positive': sum(v>0 for v in changes),
                                            'negative': sum(v<0 for v in changes),
                                            'zero': sum(v==0 for v in changes), 'n': len(changes),
                                            'changes': changes}
            report.update(status='complete', completed_at=time.time())
            atomic_json(path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__=='__main__':
    main()
