"""Reward-only evolution of individual existing synapses (development assay).

RGB is the sole sensory input. Fixed native motor spikes remain the sole action
source. This engineering search is NOT a claim about biological learning rules.
No anatomical sign, edge, or decoder is changed during a training run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import cue_scene, neutral_scene
from flysnake.lab.experiment import atomic_json
from probe_motor_controllability import evaluate


class IndividualBrain:
    def __init__(self, readout='dnp20', locus='motor', basis='individual', selection_scenes=2, tonic=0.):
        self.brain = LabBrain()
        b = self.brain.b
        if readout in ('dnpe017', 'dna02', 'dna01'):
            # Fixed, explicitly declared alternative BCI before calibration.
            pair = {'dnpe017': (10527, 555871), 'dna02': (523769, 10360), 'dna01': (10442, 10760)}[readout]
            ids = {'left': [pair[0]], 'forward': [10162, 10059], 'right': [pair[1]]}
            self.brain.native.ids = ids
            self.brain.native.groups = {k: np.array([np.flatnonzero(b.ids == c)[0] for c in v]) for k, v in ids.items()}
        if not np.isfinite(tonic) or not 0 <= tonic <= 15:
            raise ValueError('Motor tonic current must be finite and between 0 and 15')
        b.tonic[np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]] = tonic
        self.brain.calibrate([neutral_scene(i + 901) for i in range(8)])
        targets = np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]
        if locus == 'projection':
            targets = np.r_[targets, [np.flatnonzero(b.ids == c)[0] for c in (12356, 12764)]]
        edges = np.flatnonzero(np.isin(b.post, targets) & (b.weight != 0))
        pre = np.searchsorted(b.ptr, edges, side='right') - 1
        keep = b.modulation_mask[pre] == 0
        edges, pre = edges[keep], pre[keep]
        # Select active synapses using only unlabelled native neural activity.
        samples = []
        selection_seeds = list(range(4081, 4081 + selection_scenes)) + [444, 555, 6121]
        for seed in selection_seeds:
            for cue in ('left', 'right'):
                self.brain.observe(cue_scene(seed, cue), learning=False)
                samples.append(self.brain.last['counts'][pre].copy())
        active = np.any(np.array(samples) > 0, axis=0)
        self.edges, self.pre = edges[active], pre[active]
        self.base = b.weight[self.edges].copy()
        self.group = np.arange(len(self.edges))
        self.theta = np.zeros(len(self.edges))
        if basis == 'activity':
            activity = np.array(samples, dtype=float)[:, active]
            normalized = (activity - activity.mean(axis=0)) / np.maximum(activity.std(axis=0), 1.)
            _, _, vh = np.linalg.svd(normalized, full_matrices=False)
            loading = vh[0] * np.sign(self.base)
            threshold = max(abs(loading).max() * .1, 1e-12)
            preference = np.where(loading < -threshold, 0, np.where(loading > threshold, 2, 1))
            side = np.isin(b.post[self.edges], self.brain.native.groups['right']).astype(int)
            self.group = side * 3 + preference
            self.theta = np.zeros(6)
        self.report = {'readout': readout, 'locus': locus, 'calibration': self.brain.calibration,
                       'selected_cells': self.brain.native.ids, 'mutable_edges': len(self.edges),
                       'candidate_fast_edges': len(edges), 'selection': 'nonzero presynaptic spikes, no reward labels',
                       'pre_ids': b.ids[self.pre].tolist(), 'edge_ids': self.edges.tolist(),
                       'basis': basis, 'groups': self.group.tolist(), 'selection_scenes': selection_scenes,
                       'unlabelled_selection_seeds': selection_seeds,
                       'symmetric_motor_tonic_current': tonic}

    def weight_hash(self):
        return hashlib.sha256(self.brain.b.weight[self.edges].tobytes()).hexdigest()

    def set_parameters(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta.shape or not np.isfinite(theta).all() or np.abs(theta).max() > 4.000001:
            raise ValueError('Invalid bounded synaptic gains')
        self.theta = theta.copy()
        self.brain.b.weight[self.edges] = self.base * np.exp(theta[self.group])
        assert np.array_equal(np.sign(self.brain.b.weight[self.edges]), np.sign(self.base))

    def observe(self, rgb):
        r = self.brain.observe(rgb, learning=False, task='choice')
        r['memory_sha256'] = self.weight_hash()
        return r

    def feedback(self, kind):
        before = self.weight_hash()
        r = self.brain.feedback(kind, learning=False)
        assert before == self.weight_hash()
        r.update(memory_before=before, memory_after=before, weight_update='reward-only evolutionary candidate selection')
        return r

    def save(self, path):
        np.savez_compressed(path, edges=self.edges, base=self.base, weight=self.brain.b.weight[self.edges],
                            theta=self.theta, group=self.group, calibration=json.dumps(self.brain.calibration), report=json.dumps(self.report))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=128)
    p.add_argument('--seed', type=int, default=92211)
    p.add_argument('--readout', choices=('dnp20', 'dnpe017', 'dna02', 'dna01'), default='dnp20')
    p.add_argument('--locus', choices=('motor', 'projection'), default='motor')
    p.add_argument('--basis', choices=('individual', 'activity'), default='individual')
    p.add_argument('--selection-scenes', type=int, default=2)
    p.add_argument('--tonic', type=float, default=0.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status.json').exists():
        raise ValueError('Use a fresh directory')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    rng = np.random.default_rng(args.seed)
    brain = IndividualBrain(args.readout, args.locus, args.basis, args.selection_scenes, args.tonic)
    report = {'protocol': 'individual-native-synapse-search-dev-v6', 'status': 'running',
              'seed': args.seed, **brain.report, 'bounds_log_gain': [-4, 4],
              'decoder_locked': True, 'signs_preserved': True, 'new_edges': 0, 'iterations': []}
    print(json.dumps({k: v for k, v in brain.report.items() if k not in ('pre_ids', 'edge_ids', 'groups')}), flush=True)
    start = time.time()
    best = -float('inf')
    elite = []
    dimension = len(brain.theta)
    for i in range(args.candidates):
        if i == 0:
            theta = np.zeros(dimension)
        elif i % 4 == 1 or not elite:
            theta = rng.uniform(-4, 4, dimension)
        else:
            theta = elite[int(rng.integers(len(elite)))].copy()
            mask = rng.random(dimension) < (0.2 if i % 4 == 2 else 0.7)
            theta[mask] += rng.normal(0, 1.5, int(mask.sum()))
            theta = np.clip(theta, -4, 4)
        brain.set_parameters(theta)
        fitness, rows = evaluate(brain, [6121], 'opposite', feedback=True)
        if fitness > best:
            best, elite = fitness, []
        if fitness == best:
            elite.append(theta.copy())
            if len(elite) > 16:
                elite.pop(int(rng.integers(len(elite))))
        record = {'iteration': i, 'fitness': fitness, 'best_fitness': best,
                  'theta': theta.tolist(), 'rows': rows, 'elapsed_seconds': time.time() - start}
        report['iterations'].append(record)
        report.update(best_fitness=best, best_theta=elite[-1].tolist())
        atomic_json(args.output / 'status.json', report)
        if i % 8 == 0 or fitness == 1:
            print(json.dumps({k: record[k] for k in ('iteration', 'fitness', 'best_fitness', 'elapsed_seconds')}), flush=True)
        if best == 1:
            break
    brain.set_parameters(elite[-1])
    fitness, rows = evaluate(brain, range(7310, 7318), 'opposite', feedback=False)
    brain.save(args.output / 'candidate.npz')
    report.update(status='complete', test_fitness=fitness, test=rows,
                  interpretation='Development screen only. Independent controls, reversal, and Snake remain required.')
    atomic_json(args.output / 'status.json', report)
    print(json.dumps({'status': 'complete', 'test_fitness': fitness, 'mutable_edges': len(brain.edges)}), flush=True)


if __name__ == '__main__':
    main()
