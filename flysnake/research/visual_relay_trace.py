"""Read-only event/voltage instrumentation of the locked complete native graph.

This is a diagnostic kernel, not a new controller or a physiological repair.
The generated derivative source changes only two observation hooks. Native
equations, topology, weights, plasticity switch and RGB input remain unchanged.
"""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .apl_local import SOURCE, file_hash
from .paired import digest

VERSION = 'native-RGB-visual-relay-delivery-and-voltage-audit-v1'
HEADER = Path(__file__).parent / 'kernels/visual_relay_trace_hooks.hpp'
EVENT_COLUMNS = ('accepted_positive', 'accepted_negative_magnitude',
    'refractory_rejected_positive', 'refractory_rejected_negative_magnitude',
    'accepted_R1_R6_positive', 'accepted_R1_R6_negative_magnitude',
    'accepted_R8_positive', 'accepted_R8_negative_magnitude',
    'accepted_edge_events', 'refractory_rejected_edge_events')


def derivative_source():
    text = SOURCE.read_text(encoding='utf-8')
    replacements = (
        ('#endif\n', '#endif\n#include "' + HEADER.resolve().as_posix() + '"\n'),
        ('   last[i]=now;', '   visual_trace_voltage(i,v[i]);\n   last[i]=now;'),
        ('       if(refractory[j]==0){g[j]+=weight[e];awaken(j);}',
         '       visual_trace_receive(i,j,weight[e],refractory[j]==0);\n'
         '       if(refractory[j]==0){g[j]+=weight[e];awaken(j);}'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Locked event kernel no longer matches the trace insertion point')
        text = text.replace(old, new)
    return text


def build_trace(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'visual_relay_trace.cpp'
    library = directory / 'visual_relay_trace.dll'
    record_path = directory / 'build.json'
    text = derivative_source()
    expected = {'original_source_sha256': file_hash(SOURCE),
        'hooks_sha256': file_hash(HEADER), 'wrapper_sha256': file_hash(__file__),
        'generated_text_sha256': digest(np.frombuffer(text.encode('utf-8'), np.uint8))}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (any(record.get(k) != value for k, value in expected.items())
                or source.read_text(encoding='utf-8') != text
                or record['binary_sha256'] != file_hash(library)):
            raise ValueError('Preserve existing trace evidence; use a new build directory')
        return library, record
    with source.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    command = [sys.executable, '-m', 'ziglang', 'c++', '-O3', '-std=c++17',
        '-shared', '-target', 'x86_64-windows-gnu', str(source), '-o', str(library)]
    with (directory / 'compile.log').open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Trace compilation failed; inspect compile.log')
    record = dict(expected, protocol=VERSION, binary_sha256=file_hash(library),
                  generated_source_sha256=file_hash(source), command=command)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return library, record


class RelayRecorder:
    def __init__(self, brain, library):
        self.brain = brain
        self.library = C.CDLL(str(Path(library).resolve()))
        self.advance = self.library.memory_advance_apl_local
        self.advance.argtypes = list(brain.kernel.advance.argtypes)
        self.advance.restype = None
        self.setup = self.library.visual_trace_setup
        self.setup.argtypes = [C.c_int] + [C.c_void_p] * 4
        self.setup.restype = None
        self.original = brain.kernel.advance
        self.kind = np.zeros(brain.b.n, np.uint8)
        self.kind[brain.b.retina] = 1
        self.kind[brain.b.r8] = 2
        self.events = np.zeros((len(EVENT_COLUMNS), brain.b.n), np.float64)
        self.minimum = np.empty(brain.b.n, np.float32)
        self.maximum = np.empty_like(self.minimum)

    def install(self):
        self.brain.kernel.advance = self.advance
        self.disable()

    def disable(self):
        self.setup(0, None, None, None, None)

    def begin_bin(self):
        self.events.fill(0)
        self.minimum[:] = self.brain.b.v
        self.maximum[:] = self.brain.b.v
        self.setup(self.brain.b.n, self.kind.ctypes.data, self.events.ctypes.data,
                   self.minimum.ctypes.data, self.maximum.ctypes.data)


def dynamic_hashes(brain):
    """Include all actual native fields, local APL and pending delayed events."""
    b = brain.b
    arrays = {name: getattr(b, name) for name in b.fields}
    arrays.update(adapter_retinal_light=brain.retinal_light,
                  adapter_r8_light=brain.r8_light, tonic=b.tonic, all_weights=b.weight)
    return dict(cursor=int(b.cursor), sim_ms=float(b.sim_ms),
        total_spikes=int(b.total_spikes), arrays={name: digest(value) for name, value in arrays.items()})
