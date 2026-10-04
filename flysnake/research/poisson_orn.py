"""Independent native sensory-event hypothesis, with no action policy.

The Shiu model injects independent Poisson voltage jumps into sensory targets
and sets only those targets' refractory periods to zero. This extension retains
our full graph and all other existing LIF/KC/event-LTD operations. It does not
claim that the rest of this engineering model reproduces that paper or a fly.
"""
from dataclasses import dataclass, asdict
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .input_isolation import IsolatedInputBrain, InputConfig
from .compartment_native import CompartmentConfig

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / 'kernels/poisson_sensory.cpp'
VERSION = 'native-independent-poisson-ORN-events-v1'


@dataclass(frozen=True)
class PoissonConfig:
    seed: int = 100201
    maximum_hz: float = 60.
    jump_mV: float = .275 * 250
    refractory_free: bool = True


def draw_events(rng, indices, rates_hz, clock, steps, dt_ms):
    """Independent Poisson events binned at the existing native dt.

    Conditional on the Poisson count, independent uniform arrival ticks are
    the discretized homogeneous Poisson process; multiple events are retained.
    The result contains neuron indices/ticks, never game state or directions.
    """
    ix = np.asarray(indices)
    rates = np.asarray(rates_hz, float)
    if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or (ix < 0).any()
            or len(np.unique(ix)) != len(ix) or rates.shape != ix.shape or not np.isfinite(rates).all()
            or (rates < 0).any() or type(steps) is not int or steps <= 0
            or type(clock) is not int or clock < 0 or not np.isfinite(dt_ms) or dt_ms <= 0):
        raise ValueError('Explicit independent finite sensory rates required')
    positive = rates > 0
    multiplicity = rng.poisson(rates[positive] * steps * dt_ms / 1000.)
    targets = np.repeat(ix[positive], multiplicity).astype(np.int32)
    ticks = rng.integers(0, steps, size=len(targets), dtype=np.int64) + clock
    order = np.argsort(ticks, kind='stable')
    return np.ascontiguousarray(ticks[order]), np.ascontiguousarray(targets[order])


def build(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(SOURCE.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    if (hashlib.sha256((ROOT / provenance['original_source']).read_bytes()).hexdigest()
            != provenance['original_sha256']
            or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != provenance['candidate_sha256']):
        raise ValueError('Retained or candidate native source changed')
    library = directory / 'poisson_sensory.dll'; path = directory / 'build.json'
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['source'] != provenance or record['binary_sha256'] != hashlib.sha256(library.read_bytes()).hexdigest():
            raise ValueError('Preserve the existing sensory-event build')
        return library, record
    command = [sys.executable, '-m', 'ziglang', 'c++', '-O3', '-std=c++17', '-shared',
               '-target', 'x86_64-windows-gnu', str(SOURCE), '-o', str(library)]
    with (directory / 'compile.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Independent sensory-event compilation failed')
    record = {'version': VERSION, 'source': provenance, 'command': command,
              'binary_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return library, record


class EventKernel:
    """Append physical events to one native call; original export stays usable."""
    def __init__(self, original_advance, library, n, indices, jump_mV):
        self.library = C.CDLL(str(Path(library).resolve()))
        self.plain = self.library.memory_advance
        self.plain.argtypes = original_advance.argtypes
        self.plain.restype = original_advance.restype
        self.extended = self.library.memory_advance_inputs
        self.extended.argtypes = original_advance.argtypes + [C.c_int, C.c_void_p,
                                 C.c_void_p, C.c_void_p, C.c_float]
        self.extended.restype = None
        self.argtypes = original_advance.argtypes; self.restype = None
        self.mask = np.zeros(n, np.uint8); self.mask[indices] = 1
        self.jump_mV = jump_mV
        self.ticks = np.empty(0, np.int64); self.targets = np.empty(0, np.int32)

    def __call__(self, *args):
        self.extended(*args, len(self.ticks), self.ticks.ctypes.data,
                      self.targets.ctypes.data, self.mask.ctypes.data, self.jump_mV)


class PoissonORNBrain(IsolatedInputBrain):
    def __init__(self, build_directory, events=PoissonConfig(),
                 config=CompartmentConfig(central_headroom_mV=0.),
                 inputs=InputConfig(vision_input=False, lamina_bias_mV=0., zero_odor_is_silent=True)):
        if (type(events.seed) is not int or events.seed < 0
                or not np.isfinite([events.maximum_hz, events.jump_mV]).all()
                or min(events.maximum_hz, events.jump_mV) <= 0 or type(events.refractory_free) is not bool):
            raise ValueError('Explicit finite sensory-event configuration required')
        super().__init__(config=config, inputs=inputs)
        self.events_config = events
        self.rng = np.random.default_rng(events.seed)
        self.orn_arrays = [groups[side] for groups in self.orn.values() for side in ('L', 'R')]
        self.event_indices = np.concatenate(self.orn_arrays).astype(np.int32)
        if len(np.unique(self.event_indices)) != len(self.event_indices):
            raise ValueError('Sensory target identity overlap')
        library, record = build(build_directory)
        b = self.b
        original_step = b._neural_step
        self.original_neural_step = original_step
        self.original_advance = b.advance
        self.event_kernel = EventKernel(b.advance, library, b.n, self.event_indices, events.jump_mV)
        if not events.refractory_free:
            self.event_kernel.mask.fill(0)
        b.advance = self.event_kernel
        self._concentrations = np.zeros((2, 2))
        self.event_counts = np.zeros(len(self.event_indices), np.int64)

        def physical_step(light, duration_ms, **kwargs):
            # Remove only the superclass's explicit ORN tonic-current proxy.
            # Retinal/DAN/other declared interventions remain unchanged.
            filtered = []
            for indices, current in kwargs.get('stimulation', []):
                ix = np.asarray(indices)
                if any(np.array_equal(ix, orn) for orn in self.orn_arrays):
                    continue
                if np.isin(ix, self.event_indices).any():
                    raise ValueError('Ambiguous override of event-driven ORN targets')
                filtered.append((indices, current))
            kwargs['stimulation'] = filtered
            rates = np.concatenate([np.full(len(ix), value * events.maximum_hz)
                                    for ix, value in zip(self.orn_arrays, self._concentrations.flat)])
            ticks, targets = draw_events(self.rng, self.event_indices, rates, b.cursor,
                                        int(round(duration_ms / b.dt)), b.dt)
            self.event_kernel.ticks = ticks; self.event_kernel.targets = targets
            self.event_counts += np.bincount(np.searchsorted(self.sorted_indices, targets),
                                            minlength=len(self.event_indices))[self.unsort_counts]
            return original_step(light, duration_ms, **kwargs)

        self.sorted_indices = np.sort(self.event_indices)
        self.unsort_counts = np.searchsorted(self.sorted_indices, self.event_indices)
        b._neural_step = physical_step
        self.identity.update(version=VERSION, sensory_events=asdict(events),
            event_native_kernel=record,
            event_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sensory_assumption='Independent sensory voltage events; rate=concentration*maximum_hz, '
                'zero external odor events at zero concentration; target-only refractory override is '
                'explicit in sensory_events. No action noise or external policy.')

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        c = np.zeros((2, 2)) if concentrations is None else np.asarray(concentrations, float)
        if c.shape != (2, 2) or not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            raise ValueError('Physical two-channel bilateral concentrations required')
        self._concentrations = c
        self.event_counts.fill(0)
        receipt = super().integrate(rgb, duration_ms, c, stimulation=stimulation, learning=learning)
        receipt.update(protocol=VERSION, sensory_event_counts=self.event_counts.tolist(),
                       sensory_event_count=int(self.event_counts.sum()), sensory_events=asdict(self.events_config))
        return receipt

    def checkpoint(self, path, metadata):
        return super().checkpoint(path, {'protocol_state': metadata,
                                  'sensory_rng': self.rng.bit_generator.state})

    def restore(self, path, sha256):
        metadata = super().restore(path, sha256)
        self.rng.bit_generator.state = metadata['state']['sensory_rng']
        metadata['state'] = metadata['state']['protocol_state']
        return metadata
