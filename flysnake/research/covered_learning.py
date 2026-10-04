"""Engineered reward-modulated credit on retained two-hop synapses.

RGB and scalar game outcomes are the task interface. Exploration is injected
current in native DNa02 cells; every action still comes from integrated native
spikes. This is an approximate learning hypothesis, not validated fly biology
or an exact gradient of recurrent spiking dynamics.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .paired import NativeSearchBrain, digest
from .projection import CoveredGame, coverage_geometry

VERSION = 'covered-native-motor-node-reward-v1'


@dataclass(frozen=True)
class CoveredLearningConfig:
    noise_sigma: float = 3.
    learning_rate: float = .3
    gain_bound: float = 4.
    maximum_step: float = .08
    upstream_credit_scale: float = .05


def reward_update(theta, eligibility, returns, *, arm, config, shuffle_seed):
    theta = np.asarray(theta, dtype=float)
    eligibility = np.asarray(eligibility, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if (theta.ndim!=1 or eligibility.ndim!=2 or eligibility.shape[1]!=len(theta)
            or returns.shape!=(len(eligibility),) or len(returns)<2
            or not all(np.isfinite(a).all() for a in (theta, eligibility, returns))):
        raise ValueError('Finite matched eligibility and actual returns required')
    if arm not in ('learning', 'frozen', 'shuffled'):
        raise ValueError('Unknown feedback control')
    feedback = returns.copy()
    if arm=='shuffled':
        np.random.default_rng(shuffle_seed).shuffle(feedback)
    advantage = feedback-feedback.mean()
    gradient = np.mean(advantage[:,None]*eligibility, axis=0)
    step = np.clip(config.learning_rate*gradient, -config.maximum_step, config.maximum_step)
    if arm=='frozen':
        step[:] = 0
    updated = np.clip(theta+step, -config.gain_bound, config.gain_bound)
    return updated, {'gradient_rms': float(np.sqrt(np.mean(gradient**2))),
                     'step_rms': float(np.sqrt(np.mean((updated-theta)**2))),
                     'return_mean': float(returns.mean()),
                     'return_std': float(returns.std()),
                     'feedback_permuted': arm=='shuffled'}


class CoveredNodeBrain:
    def __init__(self, config=CoveredLearningConfig()):
        values = list(asdict(config).values())
        if not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Positive finite learning configuration required')
        native = NativeSearchBrain()
        self.brain = native.brain
        b = self.brain.b
        self.config = config
        self.geometry = coverage_geometry(b.uv, 4)
        # No training or reward has occurred. Final neutral calibration is
        # established for the new display before the first trial, then locked.
        neutral = []
        for seed in range(901, 909):
            game = CoveredGame(seed=seed, size=4, geometry=self.geometry)
            game.food = None; neutral.append(game.render_observation())
        self.brain.calibrated = False
        self.brain.calibrate(neutral)
        motors = np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]
        self.edges = native.edges.copy()
        self.pre = np.searchsorted(b.ptr, self.edges, side='right')-1
        self.direct = np.isin(b.post[self.edges], motors)
        self.side = np.isin(b.post[self.edges], self.brain.native.groups['right']).astype(int)
        self.base = b.weight[self.edges].copy()
        self.theta = np.zeros(len(self.edges))
        self.identity = {**native.identity, 'version': VERSION, 'geometry': self.geometry,
                         'calibration': self.brain.calibration,
                         'mutable_edges': len(self.edges), 'dimensions': len(self.edges),
                         'edges_sha256': digest(self.edges), 'base_sha256': digest(self.base),
                         'group_sha256': digest(np.arange(len(self.edges), dtype=np.int64)),
                         'config': asdict(config),
                         'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'projection_source_sha256': hashlib.sha256(Path(__file__).with_name('projection.py').read_bytes()).hexdigest(),
                         'eligibility': 'Presynaptic spikes times signed motor-current noise; one anatomical upstream hop receives a bounded activity-slope surrogate',
                         'upstream_surrogate': '0.1 + min(native spike count / 10, 1.5); engineering credit slope, not a measured neuronal derivative',
                         'exact_policy_gradient': False}

    def set_parameters(self, theta):
        theta = np.asarray(theta, dtype=float)
        if (theta.shape!=self.theta.shape or not np.isfinite(theta).all()
                or (abs(theta)>self.config.gain_bound+1e-9).any()):
            raise ValueError('Invalid bounded native synaptic gain')
        weight = (self.base*np.exp(theta)).astype(self.base.dtype)
        if not np.array_equal(np.sign(weight), np.sign(self.base)):
            raise RuntimeError('Existing connection sign changed')
        self.theta = theta.copy()
        self.brain.b.weight[self.edges] = weight

    def weight_hash(self):
        return digest(self.brain.b.weight[self.edges])

    def observe(self, rgb, noise=None):
        if noise is None:
            noise = np.zeros(2)
        noise = np.asarray(noise, dtype=float)
        if noise.shape!=(2,) or not np.isfinite(noise).all():
            raise ValueError('Two finite native current perturbations required')
        for name, value in zip(('left', 'right'), noise):
            self.brain.b.tonic[self.brain.native.groups[name]] = 12.+value
        receipt = self.brain.observe(rgb, task='snake', learning=False, reset=False)
        all_counts = self.brain.last['counts']
        counts = all_counts[self.pre].astype(float)
        # Bounded smooth scaling avoids letting a constantly high-rate cell
        # dominate all credit. No action or desired direction enters the trace.
        activation = counts/(counts+3.)
        delta = noise/self.config.noise_sigma**2
        eligibility = np.zeros(len(self.edges))
        eligibility[self.direct] = (activation[self.direct]*np.sign(self.base[self.direct])
                                    *delta[self.side[self.direct]])
        # Approximate credit propagation follows only retained afferent->motor
        # edges. It does not infer a desired action from the scene. Recurrent
        # dynamics are still integrated by the native kernel; this two-hop
        # surrogate is intentionally not called an exact recurrent gradient.
        adjoint = np.zeros(self.brain.b.n)
        weights = self.brain.b.weight[self.edges[self.direct]]
        np.add.at(adjoint, self.pre[self.direct],
                  weights*delta[self.side[self.direct]])
        post = self.brain.b.post[self.edges]
        slope = .1+np.minimum(all_counts[post]/10., 1.5)
        upstream = ~self.direct
        eligibility[upstream] = (self.config.upstream_credit_scale*activation[upstream]
                                 *np.sign(self.base[upstream])*slope[upstream]
                                 *np.clip(adjoint[post[upstream]], -3., 3.))
        receipt['memory_sha256'] = self.weight_hash()
        receipt.pop('afferent_counts', None)
        return receipt, eligibility

    def feedback(self, kind):
        before = self.weight_hash()
        receipt = self.brain.feedback(kind, learning=False)
        if receipt['delivered_spikes']<=0 or self.weight_hash()!=before:
            raise RuntimeError('Native feedback delivery or frozen rollout failed')
        return receipt

    def reset_trial(self):
        for name in ('left', 'right'):
            self.brain.b.tonic[self.brain.native.groups[name]] = 12.
        self.brain.reset_trial()
