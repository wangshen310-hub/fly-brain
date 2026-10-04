"""RGB-only adapter to the full native graph with the locked local APL cable.

No scene objects, coordinates, olfactory concentrations, desired actions, or
arbitrary stimulation enter this interface. Plasticity is frozen for the first
visual propagation audit. The phototransduction and fixed motor working point
remain explicitly unvalidated engineering assumptions.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
from .apl_local import LocalAPLBrain, LocalAPLConfig, file_hash
from .apl_local_state_v2 import save, load
from .compartment_native import CompartmentConfig
from .input_isolation import InputConfig
from .optical_sampling import ApertureConfig, sample_aperture_linear, identity
from .paired import digest
from .raw_turn import decode_raw_turn
from .vision import light_to_current

VERSION = 'native-local-APL-RGB-only-frozen-input-v1'


class VisualAPLBrain(LocalAPLBrain):
    def __init__(self, directory, *, attachment, apl=LocalAPLConfig(),
                 aperture=ApertureConfig(radius_pixels=2.25, subdivisions=3)):
        # Validate optics before loading the full graph.
        identity(np.array([[.5, .5]]), aperture)
        sample_aperture_linear(np.zeros((180, 320, 3), np.uint8),
                               np.array([[.5, .5]]), aperture)
        super().__init__(directory, attachment=attachment, apl=apl,
            config=CompartmentConfig(central_headroom_mV=0.),
            inputs=InputConfig(vision_input=True, lamina_bias_mV=0.,
                               zero_odor_is_silent=True))
        b = self.b
        self.aperture = aperture
        self.visual_input_enabled = True
        self.optical_uv_hashes = (digest(b.uv), digest(b.r8_uv))
        self.external_current_mask = np.zeros(b.n, bool)
        self.external_current_mask[np.r_[b.retina, b.r8,
            self.groups['left'], self.groups['right']]] = True
        self.motor_tonic = np.zeros(b.n, np.float32)
        self.motor_tonic[np.r_[self.groups['left'], self.groups['right']]] = self.config.motor_current_mV
        if not np.array_equal(b.tonic, self.motor_tonic):
            raise ValueError('RGB audit requires only the declared fixed motor tonic')
        self.identity.update(version=VERSION, visual_source_sha256=file_hash(__file__),
            accepted_sensory_inputs=['RGB uint8 180x320x3'],
            olfactory_external_input=False, arbitrary_stimulation_allowed=False,
            scene_objects_or_coordinates_used=False, learning_enabled=False,
            kernel_event_plasticity_used=False, kernel_event_plasticity_available=True,
            original_rate_memory_rule_used=False, optics={
                'R1_R6': identity(b.uv, aperture), 'R8': identity(b.r8_uv, aperture)},
            retina_transfer=asdict(self.retina_config),
            optical_lowpass_ms=10., unconditional_lamina_current_mV=0.,
            fixed_turn_cell_current_mV=self.config.motor_current_mV,
            visual_input_enabled=True,
            assumptions='Unfitted LIF photoreceptors and static optical aperture; unchanged anatomical UVs. '
                'Full graph including unstimulated ORNs retained. No game labels, olfactory field, '
                'central background current or external action policy. Legacy gamma1 cells are readouts '
                'for auditing, not an assumed correct visual learning circuit.')

    def set_visual_input(self, enabled):
        """A declared sensory-off intervention, never a controller."""
        if type(enabled) is not bool:
            raise ValueError('Explicit retinal input switch required')
        self.visual_input_enabled = enabled
        self.identity['visual_input_enabled'] = enabled

    def integrate(self, rgb, duration_ms):
        if type(duration_ms) is not int or duration_ms <= 0 or duration_ms % 10:
            raise ValueError('Positive integral multiple of 10ms required')
        b = self.b
        if not np.array_equal(b.tonic, self.motor_tonic):
            raise ValueError('Undeclared tonic current at RGB-only boundary')
        if (digest(b.uv), digest(b.r8_uv)) != self.optical_uv_hashes:
            raise ValueError('Original anatomical receptor coordinates changed')
        # This validation precedes any neural mutation.
        linear = sample_aperture_linear(rgb, b.uv, self.aperture)
        color = sample_aperture_linear(rgb, b.r8_uv, self.aperture)
        target = linear @ np.asarray([.2126, .7152, .0722])
        target8 = color[np.arange(len(b.r8)), b.r8_channel]
        before = self.weight_hash()
        start = self.neural_ms
        counts = np.zeros(b.n, np.int32)
        outside_nonzero = 0
        for _ in range(duration_ms // 10):
            alpha = 1 - np.exp(-1.)
            self.retinal_light += alpha * (target - self.retinal_light)
            self.r8_light += alpha * (target8 - self.r8_light)
            pulses = []
            if self.visual_input_enabled:
                pulses = [
                    (b.retina, light_to_current(self.retinal_light,
                        -45 - b.rest[b.retina], self.retina_config)),
                    (b.r8, light_to_current(self.r8_light,
                        -45 - b.rest[b.r8], self.retina_config)),
                ]
            emitted, _ = b._neural_step(np.zeros(len(b.retina)), 10,
                learning=False, stimulation=pulses, lamina_bias=0.)
            outside_nonzero = max(outside_nonzero,
                int(np.count_nonzero(b.drive[~self.external_current_mask])))
            if outside_nonzero:
                raise AssertionError('Nonvisual external drive entered the complete native graph')
            counts += emitted
        k = self.kernel
        if (not np.isfinite(b.v).all() or not np.isfinite(b.g).all()
                or not np.isfinite(k.u).all() or not np.isfinite(k.inhibition).all()):
            raise FloatingPointError('Nonfinite RGB/local-APL native state')
        if np.any(counts[k.mask.astype(bool)]):
            raise AssertionError('Graded APL emitted a point spike')
        if before != self.weight_hash():
            raise AssertionError('Frozen visual audit changed memory weights')
        self.last_counts = counts
        self.neural_ms += duration_ms
        self.native_last_cursor = b.cursor
        result = decode_raw_turn({name: counts[ix].tolist()
                                  for name, ix in self.groups.items()}, duration_ms)
        result.update(protocol=VERSION, neural_time_start=start,
            neural_time_end=self.neural_ms,
            observation_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            full_counts_sha256=digest(counts), actual_native_spikes=int(counts.sum()),
            sensory_input='RGB only', retinal_input_enabled=self.visual_input_enabled,
            retinal_spikes=int(counts[b.retina].sum()), r8_spikes=int(counts[b.r8].sum()),
            nonvisual_external_drive_nonzero_cells=outside_nonzero,
            external_drive_sha256=digest(b.drive),
            memory_before=before, memory_after=self.weight_hash(), learning=False,
            readouts={name: counts[ix].tolist() for name, ix in self.memory_readouts.items()
                      if name != 'KC'},
            KC_active=int(np.count_nonzero(counts[self.memory_readouts['KC']])),
            KC_spikes=int(counts[self.memory_readouts['KC']].sum()),
            APL_point_spikes=0, APL_local_u_mV_quantiles=np.quantile(k.u, [0, .5, .9, .99, 1]).tolist(),
            APL_receiver_conductance_sum=float(k.inhibition.sum()))
        return result

    def save_state(self, path):
        # Never use the locked v1 method with colliding native/adapter R8 names.
        return save(self, path)

    def load_state(self, path, sha256):
        metadata = load(self, path, sha256)
        self.last_counts.fill(0)  # Old action receipts cannot be reused.
        return metadata
