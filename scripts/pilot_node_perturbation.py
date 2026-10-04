"""Development: reward-modulated neural node perturbation on native synapses.

Noise is a declared injected motor-neuron current, never a sampled action. All
observed choices still come from native integration and a fixed decoder. Heldout
evaluation disables this noise. The local learning rule is an engineering model.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probe_individual_synapses import IndividualBrain
from probe_motor_controllability import evaluate
from flysnake.lab.environment import cue_scene
from flysnake.lab.experiment import atomic_json


class PerturbedBrain(IndividualBrain):
    def __init__(self, seed, tonic=6.):
        super().__init__('dna02', 'motor', 'individual', 6, tonic)
        b = self.brain.b
        targets = np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]
        edges = np.flatnonzero(np.isin(b.post, targets) & (b.weight != 0))
        pre = np.searchsorted(b.ptr, edges, side='right') - 1
        fast = b.modulation_mask[pre] == 0
        self.edges, self.pre = edges[fast], pre[fast]
        self.base = b.weight[self.edges].copy()
        self.group = np.arange(len(self.edges))
        self.theta = np.zeros(len(self.edges))
        self.side = np.isin(b.post[self.edges], self.brain.native.groups['right']).astype(int)
        self.mean_pre = np.zeros(len(self.edges))
        self.rng = np.random.default_rng(seed)
        self.tonic = tonic
        self.reward_baseline = 0.
        self.report.update(mutable_edges=len(self.edges), edge_ids=self.edges.tolist(), groups=self.group.tolist(),
                           pre_ids=b.ids[self.pre].tolist(), selection='all existing fast motor afferents',
                           noise='independent Normal(0, 3) motor currents; training only', learning='reward-modulated node perturbation')

    def train_observe(self, rgb):
        self.noise = self.rng.normal(0, 3, 2)
        b = self.brain.b
        for name, value in zip(('left', 'right'), self.noise):
            b.tonic[self.brain.native.groups[name]] = self.tonic + value
        r = self.observe(rgb)
        pre = self.brain.last['counts'][self.pre].astype(float)
        self.eligibility = np.clip(pre - self.mean_pre, -10, 10) * self.noise[self.side] / 9 * np.sign(self.base)
        self.mean_pre += .05 * (pre - self.mean_pre)
        return r

    def learn(self, kind, learning=True, eta=.05):
        # Stimulus is actually delivered; valence/magnitude are explicit choices.
        receipt = self.feedback(kind)
        before = self.weight_hash()
        reward = (1 if kind == 'reward' else -1) * abs(receipt['modulation'])
        if learning:
            self.set_parameters(np.clip(self.theta + eta * (reward - self.reward_baseline) * self.eligibility, -4, 4))
            self.reward_baseline += .05 * (reward - self.reward_baseline)
        self.brain.b.tonic[np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]] = self.tonic
        receipt.update(memory_before=before, memory_after=self.weight_hash(), eta=eta,
                       weight_update='local presynaptic activity x motor current perturbation x scalar reward')
        return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=160)
    p.add_argument('--eta', type=float, default=.05)
    p.add_argument('--seed', type=int, default=92301)
    p.add_argument('--tonic', type=float, default=6.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status.json').exists():
        raise ValueError('Use a fresh experiment directory')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    brain = PerturbedBrain(args.seed, args.tonic)
    report = {'protocol': 'native-node-perturbation-dev-v1', 'status': 'running', **brain.report,
              'seed': args.seed, 'eta': args.eta, 'rows': [], 'evaluations': []}
    fit, rows = evaluate(brain, range(8120, 8128), 'opposite')
    report['before'] = {'fitness': fit, 'rows': rows}
    start = time.time()
    for i in range(args.trials):
        side = 'left' if i % 2 == 0 else 'right'
        result = brain.train_observe(cue_scene(6200 + i // 2, side))
        correct = result['action'] == ('right' if side == 'left' else 'left')
        receipt = brain.learn('reward' if correct else 'aversive', eta=args.eta)
        report['rows'].append({'i': i, 'side': side, 'action': result['action'], 'correct': correct,
                               'counts': result['counts'], 'noise': brain.noise.tolist(), 'feedback': receipt})
        if (i + 1) % 40 == 0:
            fit, rows = evaluate(brain, [6811, 6812], 'opposite')
            report['evaluations'].append({'through': i + 1, 'fitness': fit, 'rows': rows})
            brain.save(args.output / f'candidate-{i+1}.npz')
            print(json.dumps({'through': i+1, 'development_fitness': fit, 'recent_correct': sum(r['correct'] for r in report['rows'][-40:]),
                              'theta_range': [float(brain.theta.min()), float(brain.theta.max())], 'elapsed': time.time()-start}), flush=True)
        atomic_json(args.output / 'status.json', report)
    fit, rows = evaluate(brain, range(9120, 9136), 'opposite')
    brain.save(args.output / 'candidate.npz')
    report.update(status='complete', test_fitness=fit, test=rows,
                  interpretation='Development only; no claim of stable associative learning or Snake competence')
    atomic_json(args.output / 'status.json', report)
    print(json.dumps({'status': 'complete', 'test_fitness': fit}), flush=True)


if __name__ == '__main__':
    main()
