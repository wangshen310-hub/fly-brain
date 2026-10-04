"""Independent frozen conductance diagnostic on the entire retained graph.

Reversal parameters are engineering hypotheses, not adult MBON measurements.
Su & O'Dowd (2003) establishes distinct ACh/GABA reversal currents in cultured
pupal neurons, not the -70 mV used here. Amin et al. (2020) shows local nonspiking
APL physiology; this version explicitly DOES NOT reconstruct it. No reward,
gradient, game state or extra action selector enters the membrane equations.
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
SOURCE = Path(__file__).parent / 'kernels/conductance_diagnostic.cpp'
VERSION = 'full-connectome-frozen-conductance-diagnostic-v1'


@dataclass(frozen=True)
class ConductanceConfig:
    excitatory_reversal_mV: float = 0.
    inhibitory_reversal_mV: float = -70.


def build(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    library = directory / 'conductance_diagnostic.dll'; record_path = directory / 'build.json'
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (record['source_sha256'] != sha or
                record['binary_sha256'] != hashlib.sha256(library.read_bytes()).hexdigest()):
            raise ValueError('Preserve the previous independent conductance build')
        return library, record
    command = [sys.executable, '-m', 'ziglang', 'c++', '-O3', '-std=c++17', '-shared',
               '-target', 'x86_64-windows-gnu', str(SOURCE), '-o', str(library)]
    with (directory / 'compile.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode: raise RuntimeError('Independent conductance compilation failed')
    record = {'version': VERSION, 'source_sha256': sha, 'command': command,
              'binary_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return library, record


class ConductanceKernel:
    def __init__(self, original_advance, library, n, config):
        self.library = C.CDLL(str(Path(library).resolve()))
        self.advance = self.library.memory_advance_conductance
        self.argtypes = original_advance.argtypes; self.restype = None
        self.advance.argtypes = list(self.argtypes) + [C.c_void_p, C.c_void_p, C.c_float, C.c_float]
        self.advance.restype = None
        self.ge = np.zeros(n, np.float32); self.gi = np.zeros(n, np.float32)
        self.config = config

    def __call__(self, *args):
        # MemoryBrain ABI field 31 is learning_enabled. Refuse before native
        # dispatch, rather than pretending this frozen candidate can learn.
        if args[31]: raise ValueError('Conductance diagnostic has no validated plasticity implementation')
        self.advance(*args, self.ge.ctypes.data, self.gi.ctypes.data,
                     self.config.excitatory_reversal_mV, self.config.inhibitory_reversal_mV)


class ConductanceNativeBrain(IsolatedInputBrain):
    def __init__(self, directory, conductance=ConductanceConfig(),
                 config=CompartmentConfig(central_headroom_mV=0.),
                 inputs=InputConfig(vision_input=False, lamina_bias_mV=0., zero_odor_is_silent=True)):
        super().__init__(config=config, inputs=inputs)
        self.conductance_config = conductance
        b = self.b
        if (not np.isfinite([conductance.excitatory_reversal_mV, conductance.inhibitory_reversal_mV]).all()
                or conductance.excitatory_reversal_mV <= max(float(b.rest.max()), -45.)
                or conductance.inhibitory_reversal_mV >= float(b.rest.min())):
            raise ValueError('Finite reversal potentials outside all resting voltages required')
        library, record = build(directory)
        self.original_advance = b.advance
        self.conductance_kernel = ConductanceKernel(b.advance, library, b.n, conductance)
        b.advance = self.conductance_kernel
        b.conductance_E = self.conductance_kernel.ge; b.conductance_I = self.conductance_kernel.gi
        for name in ('conductance_E', 'conductance_I'):
            b.fields.append(name); b.initial[name] = getattr(b, name).copy()
        self.identity.update(version=VERSION, conductance=asdict(conductance),
            conductance_native_kernel=record,
            conductance_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            diagnostic_only=True, kernel_event_plasticity_used=False,
            membrane_equation='20 dv/dt = rest-v+drive-adaptation+gE*(EE-v)+gI*(EI-v)',
            weight_conversion='existing positive W/(EE-rest); negative W/(EI-rest); signs and endpoints unchanged',
            numerical_scheme='0.1ms frozen-coefficient exponential midpoint; g/adaptation exact decay',
            refractory_model='voltage clamped to rest for 2.2ms; conductances accept inputs and decay',
            g_readout='legacy g stores instantaneous signed effective synaptic current; separate E/I states retained',
            limitations=['Unfitted reversal potentials and per-contact efficacies',
                         'APL remains a spiking point neuron with nonlocal output',
                         'Modulators still lack validated fast/receptor physiology',
                         'No validated plasticity, gameplay or learning in this diagnostic'])

    def integrate(self, rgb, duration_ms, concentrations=None, *, stimulation=None, learning=False):
        if learning: raise ValueError('Frozen physiology diagnostic cannot be used for training')
        result = super().integrate(rgb, duration_ms, concentrations, stimulation=stimulation, learning=False)
        result.update(protocol=VERSION, conductance=asdict(self.conductance_config))
        return result
