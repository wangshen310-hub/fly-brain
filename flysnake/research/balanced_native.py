"""Explicit fast excitation/inhibition sensitivity on the retained full graph.

The anatomical sign/endpoints are unchanged. Gains are declared physical model
parameters, not biologically measured values or game-reward training variables.
This is separate from the locked uniform-coupling and original native protocols.
"""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
from .physical_native import PhysicalNativeBrain, PhysicalConfig
from .paired import digest

VERSION = 'native-sign-preserving-excitation-inhibition-clearance-v1'


@dataclass(frozen=True)
class BalanceConfig:
    excitatory_gain: float = 1.
    inhibitory_gain: float = 2.


class BalancedNativeBrain(PhysicalNativeBrain):
    def __init__(self, balance=BalanceConfig(), physical=PhysicalConfig()):
        super().__init__(physical=physical)
        self.set_balance(balance)

    def set_balance(self, balance):
        if (not np.isfinite([balance.excitatory_gain, balance.inhibitory_gain]).all()
                or not 0 < balance.excitatory_gain <= 8 or not 0 < balance.inhibitory_gain <= 8):
            raise ValueError('Explicit positive fast-coupling gains required')
        self.balance_config = balance
        b = self.b
        gains = np.where(self.retained_full_baseline < 0, balance.inhibitory_gain,
                         balance.excitatory_gain).astype(np.float32)
        b.weight[:] = self.retained_full_baseline * gains * np.float32(self.physical_config.synaptic_scale)
        if not np.array_equal(np.signbit(b.weight), np.signbit(self.retained_full_baseline)):
            raise AssertionError('Balance configuration changed a retained sign')
        self.base = b.weight[self.edges].copy(); b.baseline_plastic = self.base.copy()
        self.initial_all_weights_sha256 = digest(b.weight)
        b.initial_weight_sha256 = self.initial_all_weights_sha256
        self.identity.update(version=VERSION, balance_configuration=asdict(balance),
            native_initial_all_weights_sha256=self.initial_all_weights_sha256,
            base_sha256=digest(self.base),
            balance_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physical_assumption='Positive gains on existing signed edge weights. '
                'Endpoints and signs retained. No biological calibration or reward/action fitting claimed.')

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        result = super().integrate(rgb, duration_ms, concentrations, stimulation=stimulation, learning=learning)
        result.update(protocol=VERSION, balance_configuration=asdict(self.balance_config))
        return result
