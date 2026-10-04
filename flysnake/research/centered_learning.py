"""Context-centered native synaptic credit with mean-current homeostasis.

All inference uses the retained full spiking model. A label-free, frozen
activity baseline is measured before reward learning. Changing an existing
weight receives a scene-independent tonic correction to its mean current,
reducing the temptation to learn one constant turn. The bounded correction and
two-hop slope are engineering assumptions, not established fly physiology.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .covered_learning import CoveredNodeBrain, CoveredLearningConfig, reward_update
from .paired import digest

VERSION = 'native-centered-two-hop-homeostatic-reward-v1'


@dataclass(frozen=True)
class CenteredLearningConfig(CoveredLearningConfig):
    gain_bound: float = 2.
    activity_floor_spikes: float = 3.
    homeostatic_limit_mV: float = 3.
    activity_clip: float = 2.


def mean_current_correction(delta_weight, pre, post, mean_counts, neurons,
                            observation_ms, limit_mV):
    delta_weight = np.asarray(delta_weight)
    pre, post = np.asarray(pre), np.asarray(post)
    mean_counts = np.asarray(mean_counts)
    if (delta_weight.shape!=pre.shape or pre.shape!=post.shape or pre.ndim!=1
            or mean_counts.shape!=(neurons,) or observation_ms<=0 or limit_mV<=0
            or not np.isfinite(delta_weight).all() or not np.isfinite(mean_counts).all()
            or (mean_counts<0).any() or (pre<0).any() or (pre>=neurons).any()
            or (post<0).any() or (post>=neurons).any()):
        raise ValueError('Valid native edges and frozen activity baseline required')
    # The native EPSC decay constant is 5 ms. This is an average-current
    # compensation; refractory rejection, adaptation and recurrence make it
    # approximate rather than an exact cancellation of baseline firing.
    correction = np.zeros(neurons)
    np.add.at(correction, post, -delta_weight*mean_counts[pre]*5./observation_ms)
    return np.clip(correction, -limit_mV, limit_mV)


class CenteredCreditBrain(CoveredNodeBrain):
    def __init__(self, config=CenteredLearningConfig()):
        super().__init__(config)
        self.base_tonic = self.brain.b.tonic.copy()
        self.homeostatic_current = np.zeros(self.brain.b.n)
        self.mean_counts = None
        self.std_counts = None
        self.identity.update(version=VERSION,
            centered_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            eligibility='Centered native presynaptic spike activity x signed weight preconditioner x native motor-current noise; one anatomical upstream hop',
            homeostasis='Frozen mean EPSC current compensation through existing postsynaptic tonic currents; capped independently of scene and action',
            activity_baseline_fitted=False)

    def _accept_baseline(self, mean, std, hashes):
        mean = np.asarray(mean, dtype=float); std = np.asarray(std, dtype=float)
        if (mean.shape!=(self.brain.b.n,) or std.shape!=mean.shape or len(hashes)<16
                or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or (mean<0).any() or (std<0).any()):
            raise ValueError('Valid measured native activity baseline required')
        self.mean_counts = mean.copy(); self.std_counts = std.copy()
        self.identity.update(activity_baseline_fitted=True,
                             mean_counts_sha256=digest(mean), std_counts_sha256=digest(std),
                             baseline_image_hashes=hashes,
                             variable_presynaptic_cells=int(np.count_nonzero(std[np.unique(self.pre)])))

    def fit_activity_baseline(self, frames):
        if self.mean_counts is not None or np.any(self.theta):
            raise ValueError('Activity baseline is frozen before any rewarded update')
        samples = []; hashes = []
        for rgb in frames:
            self.brain.b.tonic[:] = self.base_tonic
            self.brain.reset_trial()
            receipt = self.brain.observe(rgb, task='snake', learning=False, reset=False)
            samples.append(self.brain.last['counts'].copy())
            hashes.append(receipt['observation_sha256'])
        if len(samples)<16:
            raise ValueError('At least 16 label-free calibration frames required')
        samples = np.array(samples, dtype=float)
        self._accept_baseline(samples.mean(0), samples.std(0), hashes)
        return {'frames': len(frames), 'unique_images': len(set(hashes)),
                'labels': False, 'reward': False, 'image_hashes': hashes}

    def save_activity_baseline(self, path):
        if self.mean_counts is None:
            raise ValueError('Activity baseline has not been measured')
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, mean=self.mean_counts, std=self.std_counts,
                                image_hashes=self.identity['baseline_image_hashes'],
                                source_sha256=self.identity['centered_source_sha256'],
                                graph_sha256=self.identity['graph_sha256'],
                                calibration=json.dumps(self.brain.calibration, sort_keys=True))
            stream.flush(); os.fsync(stream.fileno())
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def restore_activity_baseline(self, path, expected_sha):
        if self.mean_counts is not None:
            raise ValueError('Activity baseline already locked')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('Activity baseline file checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            if (str(z['source_sha256'])!=self.identity['centered_source_sha256']
                    or str(z['graph_sha256'])!=self.identity['graph_sha256']
                    or json.loads(str(z['calibration']))!=self.brain.calibration):
                raise ValueError('Incompatible activity baseline')
            self._accept_baseline(z['mean'], z['std'], z['image_hashes'].tolist())

    def set_parameters(self, theta):
        if self.mean_counts is None and np.any(theta):
            raise ValueError('Fit activity baseline before learning')
        super().set_parameters(theta)
        if self.mean_counts is None:
            return
        b = self.brain.b
        self.homeostatic_current = mean_current_correction(
            b.weight[self.edges]-self.base, self.pre, b.post[self.edges],
            self.mean_counts, b.n, self.brain.config.observation_ms,
            self.config.homeostatic_limit_mV)
        b.tonic[:] = self.base_tonic+self.homeostatic_current

    def reset_trial(self):
        self.brain.b.tonic[:] = self.base_tonic+self.homeostatic_current
        self.brain.reset_trial()

    def observe(self, rgb, noise=None):
        if self.mean_counts is None:
            raise ValueError('Only RGB calibration is permitted before baseline is locked')
        noise = np.zeros(2) if noise is None else np.asarray(noise, dtype=float)
        if noise.shape!=(2,) or not np.isfinite(noise).all():
            raise ValueError('Two finite native current perturbations required')
        b = self.brain.b
        for name, value in zip(('left', 'right'), noise):
            indices = self.brain.native.groups[name]
            b.tonic[indices] = self.base_tonic[indices]+self.homeostatic_current[indices]+value
        receipt = self.brain.observe(rgb, task='snake', learning=False, reset=False)
        counts = self.brain.last['counts']
        eligibility = self.credit_from_counts(counts, noise/self.config.noise_sigma**2)
        receipt['memory_sha256'] = self.weight_hash()
        receipt['homeostatic_current_sha256'] = digest(self.homeostatic_current)
        receipt.pop('afferent_counts', None)
        return receipt, eligibility

    def credit_from_counts(self, counts, delta):
        """Native activity/current sensitivity only, without outcome or labels."""
        b = self.brain.b
        counts = np.asarray(counts)
        delta = np.asarray(delta)
        if (counts.shape!=(b.n,) or delta.shape!=(2,) or not np.isfinite(counts).all()
                or not np.isfinite(delta).all() or (counts<0).any()):
            raise ValueError('Valid native counts and motor-current sensitivity required')
        activity = np.clip((counts[self.pre]-self.mean_counts[self.pre]) /
                           np.maximum(self.std_counts[self.pre], self.config.activity_floor_spikes),
                           -self.config.activity_clip, self.config.activity_clip)
        weight = b.weight[self.edges]
        factor = weight/(np.abs(self.base)+1.)
        eligibility = np.zeros(len(self.edges))
        eligibility[self.direct] = activity[self.direct]*factor[self.direct]*delta[self.side[self.direct]]
        adjoint = np.zeros(b.n)
        np.add.at(adjoint, self.pre[self.direct], weight[self.direct]*delta[self.side[self.direct]])
        post = b.post[self.edges]
        slope = .1+np.minimum(counts[post]/10., 1.5)
        upstream = ~self.direct
        eligibility[upstream] = (self.config.upstream_credit_scale*activity[upstream]*factor[upstream]
                                 *slope[upstream]*np.clip(adjoint[post[upstream]], -3., 3.))
        return eligibility
