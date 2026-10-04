"""Anatomically local APL->KC feedback sensitivity, no new neural controller.

Lin et al. identifies this inhibitory feedback locus in sparse odor coding.
Its unitary gain in our MaleCNS LIF model remains an engineering hypothesis.
All unrelated existing weights, endpoints and signs retain their baseline.
"""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
from .physical_native import PhysicalNativeBrain
from .paired import digest

VERSION = 'native-local-APL-to-KC-feedback-clearance-v1'


@dataclass(frozen=True)
class APLConfig:
    inhibitory_gain: float = 2.


class APLNativeBrain(PhysicalNativeBrain):
    def __init__(self, apl=APLConfig()):
        super().__init__()
        b = self.b
        from stonkfly.neural.common import annotations
        annotation = annotations(b.ids)
        self.apl = np.flatnonzero(annotation.type.fillna('').eq('APL')).astype(np.int32)
        if set(b.ids[self.apl]) != {10540, 10977}:
            raise ValueError('Unexpected anatomical APL identities')
        if not annotation.neurotransmitter.iloc[self.apl].str.lower().eq('gaba').all():
            raise ValueError('APL inhibitory transmitter identity differs')
        edges = np.concatenate([np.arange(b.ptr[i], b.ptr[i+1], dtype=np.int64) for i in self.apl])
        self.apl_edges = edges[b.circuit['kc_mask'][b.post[edges]].astype(bool)]
        if not len(self.apl_edges) or (self.retained_full_baseline[self.apl_edges] >= 0).any():
            raise ValueError('No verified negative APL-to-KC existing weights')
        self.set_apl(apl)

    def set_apl(self, apl):
        if not np.isfinite(apl.inhibitory_gain) or not 0 < apl.inhibitory_gain <= 16:
            raise ValueError('Explicit positive APL feedback gain required')
        self.apl_config = apl
        b = self.b; b.weight[:] = self.retained_full_baseline
        b.weight[self.apl_edges] *= np.float32(apl.inhibitory_gain)
        self.base = b.weight[self.edges].copy(); b.baseline_plastic = self.base.copy()
        self.initial_all_weights_sha256 = digest(b.weight)
        b.initial_weight_sha256 = self.initial_all_weights_sha256
        self.identity.update(version=VERSION, apl_configuration=asdict(apl),
            apl_ids=b.ids[self.apl].tolist(), apl_feedback_edges=len(self.apl_edges),
            apl_feedback_edge_sha256=digest(self.apl_edges),
            native_initial_all_weights_sha256=self.initial_all_weights_sha256,
            base_sha256=digest(self.base),
            apl_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physical_assumption='Only actual inhibitory APL-to-KC existing weights have a declared gain. '
                'All other weights retained, no biological gain fit, game reward or action target.')

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        result = super().integrate(rgb, duration_ms, concentrations, stimulation=stimulation, learning=learning)
        result.update(protocol=VERSION, apl_configuration=asdict(self.apl_config))
        return result
