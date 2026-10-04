"""Independent physical coupling calibration on every existing native edge.

The transferred 0.275mV/contact coefficient was fitted in Shiu's female FlyWire
sugar-to-MN9 model. Its transfer to MaleCNS and olfactory memory is unvalidated.
This declared uniform multiplier changes no endpoints or signs and never uses
game reward or an action target. It is an engineering sensitivity experiment.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
from pathlib import Path
import numpy as np
from .input_isolation import IsolatedInputBrain, InputConfig
from .compartment_native import CompartmentConfig
from .paired import digest

VERSION = 'native-uniform-physical-coupling-clearance-calibration-v1'


@dataclass(frozen=True)
class PhysicalConfig:
    synaptic_scale: float = 1.
    motor_current_mV: float = 12.


class PhysicalNativeBrain(IsolatedInputBrain):
    def __init__(self, physical=PhysicalConfig(),
                 inputs=InputConfig(vision_input=False, lamina_bias_mV=0., zero_odor_is_silent=True)):
        super().__init__(config=CompartmentConfig(central_headroom_mV=0.,
                         motor_current_mV=physical.motor_current_mV), inputs=inputs)
        self.retained_full_baseline = self.b.weight.copy()
        self.retained_full_baseline_sha256 = digest(self.retained_full_baseline)
        self.set_physical_configuration(physical)

    def set_physical_configuration(self, physical):
        if (not np.isfinite([physical.synaptic_scale, physical.motor_current_mV]).all()
                or not 0 < physical.synaptic_scale <= 2 or physical.motor_current_mV < 0):
            raise ValueError('Explicit positive whole-graph coupling and motor current required')
        self.physical_config = physical
        b = self.b
        # Always reference the locked retained weights, never compound gains or
        # optimize on trial scores. This operation resets memory for an assay.
        b.weight[:] = self.retained_full_baseline * np.float32(physical.synaptic_scale)
        if not np.array_equal(np.signbit(b.weight), np.signbit(self.retained_full_baseline)):
            raise AssertionError('Physical multiplier changed a retained sign')
        self.base = b.weight[self.edges].copy(); b.baseline_plastic = self.base.copy()
        self.initial_all_weights_sha256 = digest(b.weight)
        b.initial_weight_sha256 = self.initial_all_weights_sha256
        self.config = replace(self.config, motor_current_mV=physical.motor_current_mV)
        self.set_working_point(0.)
        self.identity.update(version=VERSION, config=asdict(self.config),
            physical_configuration=asdict(physical),
            transferred_retained_full_weights_sha256=self.retained_full_baseline_sha256,
            native_initial_all_weights_sha256=self.initial_all_weights_sha256,
            base_sha256=digest(self.base),
            physical_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physical_assumption='Uniform positive scalar on every retained edge. '
                'Not reward-trained or biologically fitted. No topology/sign changes or action policy.')

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        result = super().integrate(rgb, duration_ms, concentrations, stimulation=stimulation, learning=learning)
        result.update(protocol=VERSION, physical_configuration=asdict(self.physical_config))
        return result
