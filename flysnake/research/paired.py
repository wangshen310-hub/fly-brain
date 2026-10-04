"""Reward-only paired search of *existing* native neural synaptic gains.

The optimizer is an engineering learning rule, not reconstructed fly biology.
It never emits an action. Neural inference accepts only RGB and uses the fixed
spike decoder. Shared parameter groups are chosen without images or labels.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from .vision import RateRetinaBrain, ReadableGame
from .operating import apply_operating_point

VERSION = 'native-two-hop-paired-event-reward-v2'


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    sigma: float = .3
    learning_rate: float = .08
    bound: float = 2.
    dimensions: int = 256


class PairedSearch:
    """Antithetic Gaussian finite differences of episodic return, with Adam.

    With bounds inactive this estimates a smoothed return gradient. Clipping
    near bounds introduces bias; no claim of an exact biological gradient.
    """
    def __init__(self, seed, config=SearchConfig()):
        if (config.dimensions < 1 or not np.isfinite(
                [config.sigma, config.learning_rate, config.bound]).all()
                or min(config.sigma, config.learning_rate, config.bound) <= 0):
            raise ValueError('Invalid search configuration')
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros(config.dimensions)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.updates = 0

    def directions(self, n):
        if n < 2:
            raise ValueError('At least two independent perturbation pairs required')
        return self.rng.normal(size=(n, self.config.dimensions))

    def candidate(self, direction, sign):
        if sign not in (-1, 1):
            raise ValueError('Invalid antithetic sign')
        return np.clip(self.theta + sign*self.config.sigma*direction,
                       -self.config.bound, self.config.bound)

    def update(self, directions, returns, arm='learning'):
        eps = np.asarray(directions, dtype=float)
        scores = np.asarray(returns, dtype=float)
        if (eps.ndim != 2 or eps.shape[1] != len(self.theta)
                or scores.shape != (len(eps), 2)
                or not np.isfinite(eps).all() or not np.isfinite(scores).all()):
            raise ValueError('Invalid paired returns')
        if arm not in ('learning', 'frozen', 'shuffled'):
            raise ValueError('Invalid control arm')
        # Shuffling preserves this generation's return distribution but breaks
        # its association with the candidate that actually earned it.
        if arm == 'shuffled':
            # Separate deterministic stream so exploration stays matched.
            np.random.default_rng(77191 + self.updates).shuffle(scores := scores.copy().reshape(-1))
            scores = scores.reshape(-1, 2)
        diff = scores[:, 0] - scores[:, 1]
        gradient = np.mean(diff[:, None]*eps, axis=0)/(2*self.config.sigma)
        self.updates += 1
        old = self.theta.copy()
        if arm != 'frozen':
            self.m = .9*self.m + .1*gradient
            self.v = .999*self.v + .001*gradient**2
            m = self.m/(1-.9**self.updates)
            v = self.v/(1-.999**self.updates)
            delta = self.config.learning_rate*m/(np.sqrt(v)+1e-8)
            self.theta = np.clip(self.theta+delta, -self.config.bound, self.config.bound)
        return {'gradient_rms': float(np.sqrt(np.mean(gradient**2))),
                'step_rms': float(np.sqrt(np.mean((self.theta-old)**2))),
                'nonzero_pairs': int(np.count_nonzero(diff)),
                'near_bound': int((np.abs(self.theta) > .99*self.config.bound).sum())}


class NativeSearchBrain:
    def __init__(self, dimensions=256):
        if dimensions < 8 or dimensions % 8:
            raise ValueError('Dimensions must be a positive multiple of eight')
        self.brain = RateRetinaBrain()
        b = self.brain.b
        self.operating = apply_operating_point(self.brain)
        ids = {'left': [523769], 'forward': [10162, 10059], 'right': [10360]}
        self.brain.native.ids = ids
        self.brain.native.groups = {k: np.array([np.flatnonzero(b.ids == c)[0] for c in v])
                                    for k, v in ids.items()}
        motors = np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]
        b.tonic[motors] = 12.
        direct = np.flatnonzero(np.isin(b.post, motors) & (b.weight != 0))
        direct_pre = np.searchsorted(b.ptr, direct, side='right')-1
        upstream = np.unique(direct_pre[b.modulation_mask[direct_pre] == 0])
        # Two anatomical hops, selected without task labels. All other edges
        # remain present and continue participating in full native dynamics.
        edges = np.flatnonzero(np.isin(b.post, np.r_[motors, upstream]) & (b.weight != 0))
        pre = np.searchsorted(b.ptr, edges, side='right')-1
        keep = (b.modulation_mask[pre] == 0) & ~np.isin(b.post[edges], np.r_[b.retina, b.r8])
        self.edges = edges[keep]
        pre = pre[keep]
        self.base = b.weight[self.edges].copy()
        # Half the parameters for direct inputs (side/sign/hash); half for
        # upstream processing (sign/hash). Hashing has no spatial/action label.
        h = (b.ids[pre].astype(np.uint64)*np.uint64(2654435761)
             + b.ids[b.post[self.edges]].astype(np.uint64)*np.uint64(2246822519))
        half = dimensions//2
        direct_mask = np.isin(b.post[self.edges], motors)
        side = np.isin(b.post[self.edges], self.brain.native.groups['right']).astype(int)
        sign = (self.base < 0).astype(int)
        self.group = np.where(direct_mask, (side*2+sign)*(half//4)+(h % (half//4)),
                              half+sign*(half//2)+(h % (half//2))).astype(int)
        self.theta = np.zeros(dimensions)
        neutral = []
        for seed in range(901, 909):
            game = ReadableGame(seed=seed, size=4)
            game.food = None
            neutral.append(game.render_observation())
        self.brain.calibrate(neutral)
        self.identity = {'version': VERSION, 'neurons': b.n, 'graph_edges': len(b.post),
                         'mutable_edges': len(self.edges), 'dimensions': dimensions,
                         'direct_edges': int(direct_mask.sum()), 'upstream_cells': len(upstream),
                         'edges_sha256': digest(self.edges), 'base_sha256': digest(self.base),
                         'group_sha256': digest(self.group), 'calibration': self.brain.calibration,
                         'graph_sha256': self.brain.native.prepared['graph_sha256'],
                         'operating': self.operating, 'retina': self.brain.metadata()['retina_config'],
                         'selected_cells': ids, 'motor_tonic': 12.,
                         'external_policy': False, 'topology_modified': False,
                         'source_signature': self.brain.source_signature}

    def set_parameters(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta.shape or not np.isfinite(theta).all() or (abs(theta)>2.000001).any():
            raise ValueError('Invalid native synaptic gain')
        self.theta = theta.copy()
        weight = self.base*np.exp(self.theta[self.group])
        if not np.array_equal(np.sign(weight), np.sign(self.base)):
            raise RuntimeError('Synaptic sign changed')
        self.brain.b.weight[self.edges] = weight

    def weight_hash(self):
        return digest(self.brain.b.weight[self.edges])

    def observe(self, rgb):
        receipt = self.brain.observe(rgb, task='snake', learning=False, reset=False)
        receipt['memory_sha256'] = self.weight_hash()
        # Old lab afferent diagnostics are irrelevant to this new plastic locus.
        receipt.pop('afferent_counts', None)
        return receipt

    def feedback(self, kind):
        return self.brain.feedback(kind, learning=False)


@contextmanager
def writer_lock(directory):
    """OS lock released on process death; an abandoned file is harmless."""
    path = Path(directory)/'writer.lock'
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another process owns this experiment directory') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def save_checkpoint(path, brain, search, provenance):
    if not np.array_equal(brain.theta, search.theta):
        raise ValueError('Cannot checkpoint an exploratory candidate as committed memory')
    metadata = {'identity': brain.identity, 'config': asdict(search.config),
                'rng': search.rng.bit_generator.state, 'updates': search.updates,
                'weights': brain.weight_hash(), 'provenance': provenance}
    with Path(path).open('xb') as f:
        np.savez_compressed(f, theta=search.theta, m=search.m, v=search.v,
                            metadata=json.dumps(metadata, sort_keys=True))
        f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_checkpoint(path, expected_sha, brain, search, provenance):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Checkpoint file hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        # JSON normalizes tuples and numpy scalars in identity metadata.
        expected = json.loads(json.dumps({'identity': brain.identity, 'config': asdict(search.config),
                                         'provenance': provenance}))
        if any(meta[k] != v for k, v in expected.items()):
            raise ValueError('Incompatible checkpoint source, graph, calibration or configuration')
        arrays = {k: z[k].copy() for k in ('theta', 'm', 'v')}
    if any(v.shape != search.theta.shape or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Invalid checkpoint arrays')
    if (abs(arrays['theta']) > search.config.bound).any() or (arrays['v'] < 0).any():
        raise ValueError('Invalid checkpoint parameter bounds')
    if type(meta['updates']) is not int or meta['updates'] < 0:
        raise ValueError('Invalid optimizer step')
    rng = np.random.default_rng()
    rng.bit_generator.state = meta['rng']
    # Validate the effective-weight hash before modifying the live model.
    weight = (brain.base*np.exp(arrays['theta'][brain.group])).astype(brain.base.dtype)
    if digest(weight) != meta['weights']:
        raise ValueError('Checkpoint effective weights mismatch')
    brain.set_parameters(arrays['theta'])
    search.theta, search.m, search.v = arrays['theta'], arrays['m'], arrays['v']
    search.updates = meta['updates']; search.rng = rng
    return meta
