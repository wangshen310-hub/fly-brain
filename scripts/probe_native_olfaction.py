"""Read-only full-brain sensory causality probe, no game learning or controller."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import NativeSearchBrain, writer_lock, digest
from flysnake.research.projection import coverage_geometry, CoveredGame
from flysnake.research.olfaction import native_orn_groups, receptor_currents, receptor_identity, OdorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classes', nargs='+', default=['ORN_DM1', 'ORN_DM2', 'ORN_DM3', 'ORN_DA1'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'
        if path.exists():
            raise ValueError('Preserve existing probe')
        report = {'protocol': 'native-olfaction-causal-probe-v1', 'status': 'loading', 'rows': [],
                  'interpretation': 'Sensory transmission only, not learning evidence or natural odor tuning.'}
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        atomic_json(path, report)
        try:
            native = NativeSearchBrain()
            brain = native.brain
            b = brain.b
            geometry = coverage_geometry(b.uv, 4)
            game = CoveredGame(seed=930301, size=4, geometry=geometry)
            game.food = None
            rgb = game.render_observation()
            base_tonic = b.tonic.copy()
            report.update(brain=native.identity, geometry=geometry)
            motors = np.r_[brain.native.groups['left'], brain.native.groups['right']]
            pre = np.unique(np.searchsorted(b.ptr, np.flatnonzero(np.isin(b.post, motors)), side='right')-1)
            report['motor_afferents'] = len(pre)
            for cell_class in args.classes:
                config = OdorConfig(receptor_class=cell_class)
                groups = native_orn_groups(brain, cell_class)
                identity = receptor_identity(brain, groups, config)
                class_rows = []
                counts_by_condition = {}
                for condition, concentrations in [('dark', [0.,0.]), ('left_low', [.25,0.]),
                         ('right_low', [0.,.25]), ('left', [1.,0.]), ('right', [0.,1.]), ('bilateral', [1.,1.])]:
                    b.tonic[:] = base_tonic
                    for indices, currents in receptor_currents(brain, groups, concentrations, config):
                        b.tonic[indices] += currents
                    brain.reset_trial()
                    receipt = brain.observe(rgb, task='snake', learning=False, reset=False)
                    counts = brain.last['counts'].copy()
                    counts_by_condition[condition] = counts
                    row = {'class': cell_class, 'condition': condition, 'concentration': concentrations,
                           'receptors': identity, 'receipt': {k:v for k,v in receipt.items() if k!='afferent_counts'},
                           'receptor_spikes': {side: int(counts[ix].sum()) for side,ix in groups.items()},
                           'native_counts_sha256': digest(counts), 'total_spikes': int(counts.sum())}
                    class_rows.append(row)
                    report.update(status='running', current_class=cell_class, current_condition=condition)
                    report['rows'].append(row)
                    atomic_json(path, report)
                    print(json.dumps({'class': cell_class, 'condition': condition,
                                      'motor': receipt['counts'], 'receptors': row['receptor_spikes']}), flush=True)
                left, right = counts_by_condition['left'], counts_by_condition['right']
                report.setdefault('summary', {})[cell_class] = {
                    'left_right_changed_cells': int(np.count_nonzero(left!=right)),
                    'left_right_changed_motor_afferents': int(np.count_nonzero(left[pre]!=right[pre])),
                    'motor_contrast': (left[motors]-right[motors]).tolist(),
                    'afferent_count_difference_rms': float(np.sqrt(np.mean((left[pre].astype(float)-right[pre])**2)))}
            b.tonic[:] = base_tonic
            report.update(status='complete', completed_at=time.time())
            atomic_json(path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__ == '__main__':
    main()
