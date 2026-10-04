"""Independent graded-lamina candidate with reversal-based fast inhibition.

Original negative event weights become nonnegative inhibitory conductance.
No voltage clipping, changed edge signs, game labels or fitted action targets.
The common -70mV reversal and conversion at the resting voltage are explicit
unfitted assumptions, not measured receptor parameters for every cell type.
"""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .graded_lamina import GradedLaminaBrain, derivative_source as graded_source
from .apl_local import SOURCE, file_hash
from .paired import digest

VERSION = 'native-RGB-graded-lamina-fast-inhibitory-conductance-v1'


def derivative_source():
    text = graded_source()
    declaration = '''
static float* fast_inhibition = nullptr;
static bool fast_inhibition_enabled = false;
extern "C" EXPORT void fast_inhibition_setup(float* state,int enabled){
  fast_inhibition=state;fast_inhibition_enabled=enabled!=0;
}
'''
    replacements = (
        ('// Same ABI as the archived event kernel,', declaration + '\n// Same ABI as the archived event kernel,'),
        ('if(apl_inhibition[i]==0.f){', 'if(apl_inhibition[i]==0.f && fast_inhibition[i]==0.f){'),
        ('float in=apl_inhibition[i]*std::sqrt(ag[1]);',
         'float in=(apl_inhibition[i]+fast_inhibition[i])*std::sqrt(ag[1]);'),
        ('apl_inhibition[i]*=ag[1];', 'apl_inhibition[i]*=ag[1];fast_inhibition[i]*=ag[1];'),
        ('if(refractory[j]==0){g[j]+=weight[e];awaken(j);}',
         'if(refractory[j]==0){\n'
         '         if(fast_inhibition_enabled && weight[e]<0.f)\n'
         '           fast_inhibition[j]+=-weight[e]/(rest[j]-inhibitory_reversal);\n'
         '         else g[j]+=weight[e];awaken(j);\n'
         '       }'),
        ('g[j]+=positive;g[j]-=negative;',
         'g[j]+=positive;\n'
         '           if(fast_inhibition_enabled)fast_inhibition[j]+=negative/(rest[j]-inhibitory_reversal);\n'
         '           else g[j]-=negative;'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Locked graded kernel differs from conductance insertion point: ' + old)
        text = text.replace(old, new)
    return text


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'conductance_lamina.cpp'
    library = directory / 'conductance_lamina.dll'
    record_path = directory / 'build.json'
    text = derivative_source()
    expected = {'original_source_sha256': file_hash(SOURCE), 'wrapper_sha256': file_hash(__file__),
        'graded_parent_sha256': file_hash(Path(__file__).with_name('graded_lamina.py')),
        'generated_text_sha256': digest(np.frombuffer(text.encode('utf-8'), np.uint8))}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (any(record.get(k) != v for k, v in expected.items())
                or source.read_text(encoding='utf-8') != text
                or file_hash(library) != record['binary_sha256']):
            raise ValueError('Preserve the existing conductance build')
        return library, record
    with source.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    command = [sys.executable, '-m', 'ziglang', 'c++', '-O3', '-std=c++17', '-shared',
        '-target', 'x86_64-windows-gnu', str(source), '-o', str(library)]
    with (directory / 'compile.log').open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Conductance compilation failed; inspect compile.log')
    record = dict(expected, protocol=VERSION, generated_source_sha256=file_hash(source),
                  binary_sha256=file_hash(library), command=command)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return library, record


class ConductanceLaminaBrain(GradedLaminaBrain):
    def __init__(self, directory, *, attachment, build_directory, enabled=True,
                 inhibition_enabled=True):
        if type(inhibition_enabled) is not bool:
            raise ValueError('Explicit receiver diagnostic switch required')
        super().__init__(directory, attachment=attachment,
            build_directory=Path(build_directory) / 'graded-parent', enabled=enabled)
        library, record = build(build_directory)
        self.conductance_library = C.CDLL(str(Path(library).resolve()))
        advance = self.conductance_library.memory_advance_apl_local
        advance.argtypes = list(self.kernel.advance.argtypes)
        advance.restype = None
        self.kernel.advance = advance
        setup = self.conductance_library.lamina_graded_setup
        setup.argtypes = list(self.graded_setup.argtypes)
        setup.restype = None
        self.graded_setup = setup
        self.set_graded_enabled(enabled)
        self.fast_inhibition = np.zeros(self.b.n, np.float32)
        self.b.fast_synaptic_inhibition = self.fast_inhibition
        self.b.fields.append('fast_synaptic_inhibition')
        self.b.initial['fast_synaptic_inhibition'] = self.fast_inhibition.copy()
        self.inhibition_setup = self.conductance_library.fast_inhibition_setup
        self.inhibition_setup.argtypes = [C.c_void_p, C.c_int]
        self.inhibition_setup.restype = None
        self.set_inhibition_enabled(inhibition_enabled)
        if np.any(self.b.rest <= self.apl_config.inhibitory_reversal_mV):
            raise ValueError('Inhibitory reversal must lie below every retained resting voltage')
        self.identity.update(version=VERSION, conductance_kernel=record,
            conductance_source_sha256=file_hash(__file__),
            inhibition_enabled=inhibition_enabled,
            fast_negative_event_semantics='-weight/(rest-E_inhib) conductance, exponential 5ms decay; '
                'same original delay and refractory acceptance; positive events retain their original g increments',
            inhibitory_reversal_mV=self.apl_config.inhibitory_reversal_mV,
            voltage_clipping=False, physiological_fit=False,
            limitations='Common reversal and resting-point conversion are unfitted. '
                'This fixes unbounded negative-current semantics, not cell-type calibration, '
                'appropriate sparse representations or reinforcement learning.')

    def set_inhibition_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Explicit receiver switch required')
        if not enabled and np.any(self.fast_inhibition):
            raise ValueError('Reset before disabling a nonzero conductance state')
        self.inhibition_setup(self.fast_inhibition.ctypes.data, int(enabled))
        self.inhibition_enabled = enabled
        self.identity['inhibition_enabled'] = enabled
