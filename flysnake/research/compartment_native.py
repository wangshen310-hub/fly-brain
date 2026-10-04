"""Isolated native gamma1 memory assay, with a monotonic raw-pulse BCI.

The retained full-graph event kernel already contains a spike-timed, dopamine-
gated KC-to-MBON depression rule. This protocol calls that kernel directly for
the anatomically identified PPL101/MBON11 compartment. It does not call the old
centered rate rule or any game-return candidate optimizer. This is an explicitly
unvalidated physiological model, not a claim of biological brain reproduction.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from flysnake.neural import RealFlyBrain
from flysnake.lab.protocol import sample_linear
from .vision import RetinaConfig, light_to_current
from .olfaction import native_orn_groups
from .operating import OperatingPoint, apply_operating_point
from .projection import coverage_geometry
from .raw_turn import decode_raw_turn
from .paired import digest

VERSION = 'native-gamma1-event-memory-raw-steering-v1'
MOTOR_IDS = {'left': [523769], 'forward': [10162, 10059], 'right': [10360]}


@dataclass(frozen=True)
class CompartmentConfig:
    central_headroom_mV: float = 4.
    motor_current_mV: float = 12.
    depression_eta: float = .001
    receptor_classes: tuple = ('ORN_DM1', 'ORN_DM4')
    observation_ms: int = 300
    feedback_current_mV: float = 30.


class NativeCompartmentBrain:
    def __init__(self, config=CompartmentConfig()):
        if (not np.isfinite([config.central_headroom_mV, config.motor_current_mV,
                             config.depression_eta, config.feedback_current_mV]).all()
                or min(config.central_headroom_mV, config.motor_current_mV) < 0
                or config.depression_eta <= 0 or config.feedback_current_mV <= 0
                or type(config.observation_ms) is not int or config.observation_ms <= 0
                or len(config.receptor_classes) != 2):
            raise ValueError('Explicit finite sensory and neural assay configuration required')
        self.config = config
        self.native = RealFlyBrain(learning=False)
        self.b = self.native.brain
        b = self.b
        self.source_signature = 'independent-gamma1-assay'
        self.retina_config = RetinaConfig()
        self.retinal_light = np.zeros(len(b.retina))
        self.r8_light = np.zeros(len(b.r8))
        self.neural_ms = 0
        self.last_counts = np.zeros(b.n, dtype=np.int32)
        self.groups = {name: np.array([np.flatnonzero(b.ids == cell)[0] for cell in ids])
                       for name, ids in MOTOR_IDS.items()}
        self.orn = {name: native_orn_groups(self, name) for name in config.receptor_classes}
        original = b.circuit
        self.memory_readouts = {
            'gamma1_MBON11': original['mb'][4:].copy(),
            'alpha1_MBON07': original['mb'][:4].copy(),
            'PPL101': original['aversive'].copy(),
            'PAM11': original['reward'].copy(),
            'KC': original['kc'].copy(),
        }
        # Confirm by graph IDs, rather than assuming an undocumented array order.
        if (set(b.ids[self.memory_readouts['gamma1_MBON11']]) != {10704, 11402}
                or set(b.ids[self.memory_readouts['alpha1_MBON07']]) != {12859, 15626, 18603, 515338}):
            raise ValueError('Native memory readout identity differs')
        select = np.isin(b.post[original['edges']], self.memory_readouts['gamma1_MBON11'])
        dan_rows = np.flatnonzero(np.isin(original['dan'], original['aversive']))
        circuit = dict(original)
        circuit.update(edges=original['edges'][select].copy(),
                       pre=original['pre'][select].copy(),
                       dan=original['aversive'].copy(),
                       gain=original['gain'][dan_rows][:, select].copy())
        circuit['dan_index'] = np.full(b.n, -1, dtype=np.int8)
        circuit['dan_index'][circuit['dan']] = np.arange(len(circuit['dan']))
        b.circuit = circuit
        b.baseline_plastic = b.weight[circuit['edges']].copy()
        b.eta = config.depression_eta
        self.edges = circuit['edges']
        self.pre = circuit['pre']
        self.base = b.baseline_plastic.copy()
        self.initial_all_weights_sha256 = digest(b.weight)
        self.initial_tonic = b.tonic.copy()
        self.operating = None
        self.set_working_point(config.central_headroom_mV)
        self.geometry = coverage_geometry(b.uv, 4)
        self.identity = {
            'version': VERSION, 'config': asdict(config), 'neurons': b.n,
            'graph_edges': len(b.post), 'graph_sha256': self.native.prepared['graph_sha256'],
            'plastic_edges': len(self.edges), 'edges_sha256': digest(self.edges),
            'pre_ids_sha256': digest(b.ids[self.pre]), 'base_sha256': digest(self.base),
            'native_initial_all_weights_sha256': self.initial_all_weights_sha256,
            'compartment': 'PPL101 -> gamma1pedc / KC -> MBON11',
            'memory_readouts': {k: b.ids[ix].tolist() for k, ix in self.memory_readouts.items()},
            'orn_ids': {name: {side: b.ids[ix].tolist() for side, ix in groups.items()}
                        for name, groups in self.orn.items()},
            'motor_ids': MOTOR_IDS, 'geometry': self.geometry,
            'operating': self.operating, 'native_kernel': b.build,
            'original_rate_memory_rule_used': False,
            'kernel_event_plasticity_used': True, 'native_rule_parameters': b.rule_parameters,
            'external_action_policy': False, 'game_return_optimizer': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'assumptions': 'LIF receptors, intrinsic currents, dopamine contact gains and event LTD parameters '
                           'are declared engineering assumptions. Gamma1 assay only; alpha1 rule not inferred from it.',
        }

    def set_working_point(self, headroom_mV):
        if not np.isfinite(headroom_mV) or headroom_mV < 0:
            raise ValueError('Nonnegative fixed headroom required')
        self.b.tonic[:] = self.initial_tonic
        self.operating = (apply_operating_point(self, OperatingPoint(headroom_mV=headroom_mV))
                          if headroom_mV else {'version': 'no-added-central-current', 'cells': 0})
        self.b.tonic[np.r_[self.groups['left'], self.groups['right']]] = self.config.motor_current_mV

    def weight_hash(self):
        return digest(self.b.weight[self.edges])

    def reset_memory(self):
        self.b.weight[self.edges] = self.base

    def set_fraction(self, fraction):
        value = np.asarray(fraction)
        if (value.shape != self.base.shape or not np.isfinite(value).all()
                or (value < .1).any() or (value > 1).any()):
            raise ValueError('Gamma1 efficacy must preserve the positive native sign and LTD bounds')
        self.b.weight[self.edges] = self.base * value

    def reset_trial(self, rgb=None):
        # Reset dynamics explicitly while retaining the actual learned weights.
        weights = self.b.weight[self.edges].copy()
        self.b.reset(keep_memory=True)
        self.b.weight[self.edges] = weights
        self.retinal_light.fill(0)
        self.r8_light.fill(0)
        self.integrate(np.full((180, 320, 3), 85, np.uint8) if rgb is None else rgb, 100)

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        if type(duration_ms) is not int or duration_ms <= 0 or duration_ms % 10:
            raise ValueError('Explicit multiple-of-10ms neural duration required')
        if type(learning) is not bool:
            raise ValueError('Explicit native plasticity switch required')
        b = self.b
        linear = sample_linear(rgb, b.uv)
        target = linear @ np.asarray([.2126, .7152, .0722])
        r8 = sample_linear(rgb, b.r8_uv)
        target8 = r8[np.arange(len(b.r8)), b.r8_channel]
        concentrations = np.zeros((2, 2)) if concentrations is None else np.asarray(concentrations, float)
        if (concentrations.shape != (2, 2) or not np.isfinite(concentrations).all()
                or (concentrations < 0).any() or (concentrations > 1).any()):
            raise ValueError('Two physical receptor channels with L/R concentrations required')
        before = self.weight_hash()
        counts = np.zeros(b.n, dtype=np.int32)
        for _ in range(duration_ms // 10):
            alpha = 1-np.exp(-10/10.)
            self.retinal_light += alpha*(target-self.retinal_light)
            self.r8_light += alpha*(target8-self.r8_light)
            pulses = [(b.retina, light_to_current(self.retinal_light, -45-b.rest[b.retina], self.retina_config)),
                      (b.r8, light_to_current(self.r8_light, -45-b.rest[b.r8], self.retina_config))]
            for channel, groups in enumerate(self.orn.values()):
                for side, value in zip(('L', 'R'), concentrations[channel]):
                    ix = groups[side]
                    pulses.append((ix, light_to_current(value, -45-b.rest[ix], self.retina_config)))
            if stimulation is not None:
                pulses.extend(stimulation)
            # Direct native integration: actual KC spike eligibility and delayed
            # DAN spike deliveries perform LTD inside C++, only on selected edges.
            c, _ = b._neural_step(np.zeros(len(b.retina)), 10, learning=learning, stimulation=pulses)
            counts += c
        if not np.isfinite(b.v).all() or not np.isfinite(b.g).all():
            raise FloatingPointError('Nonfinite full-graph neural dynamics')
        if not learning and before != self.weight_hash():
            raise RuntimeError('Frozen assay changed native weights')
        self.neural_ms += duration_ms
        self.last_counts = counts
        turn_counts = {name: counts[ix].tolist() for name, ix in self.groups.items()}
        result = decode_raw_turn(turn_counts, duration_ms)
        result.update(protocol=VERSION, neural_time_end=self.neural_ms,
            observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            memory_before=before, memory_after=self.weight_hash(), learning=learning,
            actual_native_spikes=int(counts.sum()), full_counts_sha256=digest(counts),
            KC_active=int(np.count_nonzero(counts[self.memory_readouts['KC']])),
            KC_spikes=int(counts[self.memory_readouts['KC']].sum()),
            plastic_pre_active=int(np.count_nonzero(counts[self.pre])),
            plastic_pre_spikes=int(counts[self.pre].sum()),
            readouts={k: counts[ix].tolist() for k, ix in self.memory_readouts.items() if k != 'KC'},
            orn_spikes={name: {side: int(counts[ix].sum()) for side, ix in groups.items()}
                        for name, groups in self.orn.items()},
            concentrations=concentrations.tolist())
        return result

    def checkpoint(self, path, metadata):
        """Immutable trial-boundary weights, with caller-owned exact protocol state."""
        path = Path(path)
        full = {'identity': self.identity, 'memory_sha256': self.weight_hash(), 'state': metadata}
        with path.open('xb') as stream:
            np.savez_compressed(stream, weights=self.b.weight[self.edges], metadata=json.dumps(full, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def restore(self, path, sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError('Immutable gamma1 checkpoint hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            weight = z['weights'].copy()
        if metadata['identity'] != json.loads(json.dumps(self.identity)):
            raise ValueError('Checkpoint source, anatomy or configuration differs')
        if (weight.shape != self.base.shape or weight.dtype != self.base.dtype
                or not np.isfinite(weight).all() or (weight < self.base*.1).any()
                or (weight > self.base).any() or digest(weight) != metadata['memory_sha256']):
            raise ValueError('Invalid retained native gamma1 memory')
        self.b.weight[self.edges] = weight
        self.reset_trial()
        return metadata
