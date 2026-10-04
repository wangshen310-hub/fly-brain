"""Local subthreshold MBON working point with existing APL feedback.

This changes intrinsic excitability of two anatomically selected memory output
cells, never selects actions or uses reward to tune a candidate. The currents
are explicit engineering assumptions below the isolated LIF firing threshold.
"""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
from .apl_native_v2 import APLNativeBrain, APLConfig

VERSION = 'native-local-APL-gamma1-subthreshold-working-point-v1'


@dataclass(frozen=True)
class TonicConfig:
    mbon_current_mV: float = 3.


class TonicAPLBrain(APLNativeBrain):
    def __init__(self, tonic=TonicConfig(), apl=APLConfig(inhibitory_gain=2.)):
        super().__init__(apl=apl)
        self.set_mbon_current(tonic)

    def set_mbon_current(self, tonic):
        cells = self.memory_readouts['gamma1_MBON11']
        if (not np.isfinite(tonic.mbon_current_mV) or tonic.mbon_current_mV < 0
                or (tonic.mbon_current_mV >= -45 - self.b.rest[cells]).any()):
            raise ValueError('Explicit subthreshold MBON intrinsic current required')
        self.tonic_config = tonic
        self.b.tonic[cells] = tonic.mbon_current_mV
        self.identity.update(version=VERSION, tonic_configuration=asdict(tonic),
            intrinsic_target_ids=self.b.ids[cells].tolist(),
            tonic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            intrinsic_assumption='Two actual gamma1 MBON cells receive declared fixed subthreshold current. '
                'No game-dependent current, action teacher or biological fit.')

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        result = super().integrate(rgb, duration_ms, concentrations, stimulation=stimulation, learning=learning)
        result.update(protocol=VERSION, tonic_configuration=asdict(self.tonic_config))
        return result
