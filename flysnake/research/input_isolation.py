"""Causal sensory ablations for the native compartment assay.

Flags disable declared external currents, never identify game objects or choose
actions. They distinguish the old unconditional lamina drive, retinal input,
and inactive ORN dark floor. Used only as an explicit diagnostic configuration.
"""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
from flysnake.lab.protocol import sample_linear
from .compartment_native import NativeCompartmentBrain, CompartmentConfig
from .vision import light_to_current
from .raw_turn import decode_raw_turn
from .paired import digest

VERSION = 'native-compartment-sensory-isolation-v1'


@dataclass(frozen=True)
class InputConfig:
    vision_input: bool = True
    lamina_bias_mV: float = 12.
    zero_odor_is_silent: bool = False


class IsolatedInputBrain(NativeCompartmentBrain):
    def __init__(self, config=CompartmentConfig(central_headroom_mV=0.), inputs=InputConfig()):
        super().__init__(config)
        self.set_input_configuration(inputs)
        self.identity.update(version=VERSION, input_config=asdict(inputs),
                             input_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def set_input_configuration(self, inputs):
        if (type(inputs.vision_input) is not bool or type(inputs.zero_odor_is_silent) is not bool
                or not np.isfinite(inputs.lamina_bias_mV) or inputs.lamina_bias_mV<0):
            raise ValueError('Explicit external current ablation required')
        self.input_config = inputs

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        if type(duration_ms) is not int or duration_ms <= 0 or duration_ms%10:
            raise ValueError('Explicit multiple-of-10ms duration required')
        if type(learning) is not bool:
            raise ValueError('Explicit native plasticity switch required')
        b = self.b
        linear = sample_linear(rgb, b.uv)
        target = linear @ np.asarray([.2126,.7152,.0722])
        r8 = sample_linear(rgb, b.r8_uv)
        target8 = r8[np.arange(len(b.r8)), b.r8_channel]
        c = np.zeros((2,2)) if concentrations is None else np.asarray(concentrations, float)
        if c.shape!=(2,2) or not np.isfinite(c).all() or (c<0).any() or (c>1).any():
            raise ValueError('Two physical receptor channels with L/R concentrations required')
        before = self.weight_hash()
        counts = np.zeros(b.n, dtype=np.int32)
        for _ in range(duration_ms//10):
            alpha = 1-np.exp(-1.)
            self.retinal_light += alpha*(target-self.retinal_light)
            self.r8_light += alpha*(target8-self.r8_light)
            pulses = []
            if self.input_config.vision_input:
                pulses += [(b.retina, light_to_current(self.retinal_light, -45-b.rest[b.retina], self.retina_config)),
                           (b.r8, light_to_current(self.r8_light, -45-b.rest[b.r8], self.retina_config))]
            for channel, groups in enumerate(self.orn.values()):
                for side, value in zip(('L','R'), c[channel]):
                    ix = groups[side]
                    current = (0. if self.input_config.zero_odor_is_silent and value==0 else
                               light_to_current(value, -45-b.rest[ix], self.retina_config))
                    pulses.append((ix,current))
            if stimulation is not None:
                pulses.extend(stimulation)
            emitted, _ = b._neural_step(np.zeros(len(b.retina)),10, learning=learning,
                stimulation=pulses, lamina_bias=self.input_config.lamina_bias_mV)
            counts += emitted
        if not np.isfinite(b.v).all() or not np.isfinite(b.g).all():
            raise FloatingPointError('Nonfinite full-graph neural state')
        if not learning and before!=self.weight_hash():
            raise RuntimeError('Frozen isolation assay changed weights')
        self.neural_ms += duration_ms
        self.last_counts = counts
        self.native_last_cursor = b.cursor
        result = decode_raw_turn({name:counts[ix].tolist() for name,ix in self.groups.items()}, duration_ms)
        result.update(protocol=VERSION, neural_time_end=self.neural_ms,
            observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), input_config=asdict(self.input_config),
            memory_before=before, memory_after=self.weight_hash(), learning=learning,
            actual_native_spikes=int(counts.sum()), full_counts_sha256=digest(counts),
            KC_active=int(np.count_nonzero(counts[self.memory_readouts['KC']])),
            KC_spikes=int(counts[self.memory_readouts['KC']].sum()),
            plastic_pre_active=int(np.count_nonzero(counts[self.pre])),
            plastic_pre_spikes=int(counts[self.pre].sum()),
            readouts={k:counts[ix].tolist() for k,ix in self.memory_readouts.items() if k!='KC'},
            orn_spikes={name:{side:int(counts[ix].sum()) for side,ix in groups.items()}
                        for name,groups in self.orn.items()}, concentrations=c.tolist())
        return result
