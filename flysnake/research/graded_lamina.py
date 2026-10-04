"""Frozen RGB candidate: passive L1-L5 voltages with bounded graded release.

The existing membrane/current equation is retained without point spikes in
these cells. Release uses their actual voltage and original signed outgoing
weights. Unknown electrical/release parameters remain explicit hypotheses;
none are selected using game performance. The remainder of the full graph,
local APL, RGB boundary and actual motor spike decoder are retained.
"""
from dataclasses import dataclass, asdict
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyarrow.feather as feather
from .apl_local import SOURCE, file_hash
from .visual_apl import VisualAPLBrain
from .paired import digest

ROOT = Path(__file__).resolve().parents[2]
HEADER = Path(__file__).parent / 'kernels/graded_lamina_hooks.hpp'
VERSION = 'native-RGB-graded-L1-L5-frozen-relay-candidate-v1'


@dataclass(frozen=True)
class GradedLaminaConfig:
    resting_release_hz: float = 50.
    release_span_mV: float = 12.
    maximum_release_hz: float = 100.
    release_interval_ms: float = 1.

    def validate(self):
        values = np.array(list(asdict(self).values()))
        if (not np.isfinite(values).all() or (values <= 0).any()
                or self.resting_release_hz > self.maximum_release_hz
                or self.maximum_release_hz > 500
                or not .1 <= self.release_interval_ms <= 2
                or not np.isclose(self.release_interval_ms / .1, round(self.release_interval_ms / .1))):
            raise ValueError('Finite bounded release and integral native stride required')


def derivative_source():
    text = SOURCE.read_text(encoding='utf-8')
    block = r'''
   if(lamina_enabled){
     // Deliver only along retained L1-L5 pair edges, with the existing delay.
     // Positive/negative budgets are kept separately until the receiver.
     for(int j=0;j<n;j++){
       float &positive=lamina_positive_queue[slot*n+j];
       float &negative=lamina_negative_queue[slot*n+j];
       if(positive!=0.f || negative!=0.f){
         evolve(j,*clock,drive[j]);
         if(refractory[j]==0){
           g[j]+=positive;g[j]-=negative;
           lamina_delivered[j]+=positive;lamina_delivered[n+j]+=negative;
           if(!apl_mask[j])awaken(j);
         }
         positive=0.f;negative=0.f;
       }
     }
     if((*clock+1)%lamina_stride==0){
       const float interval=dt*lamina_stride;
       for(int k=0;k<lamina_count;k++){
         int i=lamina_indices[k];evolve(i,*clock,drive[i]);
         float rate=std::clamp(lamina_rest_rate_hz*(1.f+(v[i]-rest[i])/lamina_release_span_mV),
                               0.f,lamina_maximum_hz);
         lamina_rates[k]=rate;
         const float amount=rate*interval/1000.f;
         for(int64_t e=ptr[i];e<ptr[i+1];e++){
           int j=post[e];float w=weight[e];
           if(w>=0.f)lamina_positive_queue[future*n+j]+=w*amount;
           else lamina_negative_queue[future*n+j]-=w*amount;
         }
       }
     }
   }
'''
    replacements = (
        ('#endif\n', '#endif\n#include "' + HEADER.resolve().as_posix() + '"\n'),
        ('if(apl_mask[i]){flags[i]=0;continue;}',
         'if(apl_mask[i] || (lamina_enabled && lamina_mask[i])){flags[i]=0;continue;}'),
        ('queue_count[slot]=0;', block + '   queue_count[slot]=0;'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Locked kernel differs from the declared graded insertion points')
        text = text.replace(old, new)
    return text


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'graded_lamina.cpp'
    library = directory / 'graded_lamina.dll'
    record_path = directory / 'build.json'
    text = derivative_source()
    expected = {'original_source_sha256': file_hash(SOURCE), 'hooks_sha256': file_hash(HEADER),
        'wrapper_sha256': file_hash(__file__),
        'generated_text_sha256': digest(np.frombuffer(text.encode('utf-8'), np.uint8))}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (any(record.get(k) != v for k, v in expected.items())
                or source.read_text(encoding='utf-8') != text
                or file_hash(library) != record['binary_sha256']):
            raise ValueError('Preserve existing graded build; use a new directory')
        return library, record
    with source.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    command = [sys.executable, '-m', 'ziglang', 'c++', '-O3', '-std=c++17', '-shared',
        '-target', 'x86_64-windows-gnu', str(source), '-o', str(library)]
    with (directory / 'compile.log').open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Graded candidate compilation failed')
    record = dict(expected, protocol=VERSION, generated_source_sha256=file_hash(source),
                  binary_sha256=file_hash(library), command=command)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return library, record


class GradedLaminaBrain(VisualAPLBrain):
    def __init__(self, directory, *, attachment, build_directory,
                 graded=GradedLaminaConfig(), enabled=True):
        graded.validate()
        if type(enabled) is not bool:
            raise ValueError('Explicit graded diagnostic switch required')
        super().__init__(directory, attachment=attachment)
        self.graded_config = graded
        b = self.b
        self.base_native_fields = tuple(b.fields)
        a = feather.read_table(ROOT / 'data/malecns_v1/annotations.feather',
            columns=['bodyId', 'type']).to_pandas().set_index('bodyId').loc[b.ids]
        self.graded_indices = np.flatnonzero(a.type.isin(['L1', 'L2', 'L3', 'L4', 'L5']).to_numpy()).astype(np.int32)
        self.graded_mask = np.zeros(b.n, np.uint8)
        self.graded_mask[self.graded_indices] = 1
        # This independent candidate has no invented graded-to-cable adapter.
        # Verify the retained L1-L5 topology has no APL receiver before running.
        for i in self.graded_indices:
            if np.any(self.kernel.mask[b.post[b.ptr[i]:b.ptr[i+1]]]):
                raise ValueError('Graded lamina has an APL target: explicit cable delivery implementation required')
        library, record = build(build_directory)
        self.graded_library = C.CDLL(str(Path(library).resolve()))
        function = self.graded_library.memory_advance_apl_local
        function.argtypes = list(self.kernel.advance.argtypes)
        function.restype = None
        self.kernel.advance = function
        self.graded_setup = self.graded_library.lamina_graded_setup
        self.graded_setup.argtypes = [C.c_int] * 3 + [C.c_void_p] * 6 + [C.c_float] * 3 + [C.c_int]
        self.graded_setup.restype = None
        self.graded_rates = np.zeros(len(self.graded_indices), np.float32)
        self.graded_positive_queue = np.zeros((19, b.n), np.float32)
        self.graded_negative_queue = np.zeros_like(self.graded_positive_queue)
        self.graded_delivered = np.zeros((2, b.n), np.float64)
        for name, value in (
            ('lamina_graded_rates', self.graded_rates),
            ('lamina_graded_positive_queue', self.graded_positive_queue),
            ('lamina_graded_negative_queue', self.graded_negative_queue),
            ('lamina_graded_delivered', self.graded_delivered)):
            setattr(b, name, value)
            b.fields.append(name)
            b.initial[name] = value.copy()
        self.set_graded_enabled(enabled)
        self.identity.update(version=VERSION, graded_config=asdict(graded),
            graded_cell_ids_sha256=digest(b.ids[self.graded_indices]), graded_cells=len(self.graded_indices),
            graded_kernel=record, graded_source_sha256=file_hash(__file__),
            original_edge_weights_and_signs_retained=True,
            physiological_fit=False, graded_enabled=enabled,
            assumptions='L1-L5 passive 20ms point voltage without spikes; threshold-linear release '
                'with declared resting 50Hz, 12mV span and 100Hz ceiling; unfitted parameters chosen '
                'before any stimulus/game result. Receptors and remaining non-APL cells retain '
                'their locked LIF proxies. No lamina/central external current or action labels.')

    def set_graded_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Explicit graded switch required')
        if hasattr(self, 'graded_enabled') and self.graded_enabled != enabled:
            if np.any(self.graded_positive_queue) or np.any(self.graded_negative_queue):
                raise ValueError('Reset the native state before changing the graded mechanism')
        c = self.graded_config
        self.graded_setup(self.b.n, len(self.graded_indices), round(c.release_interval_ms / .1),
            self.graded_indices.ctypes.data, self.graded_mask.ctypes.data, self.graded_rates.ctypes.data,
            self.graded_positive_queue.ctypes.data, self.graded_negative_queue.ctypes.data,
            self.graded_delivered.ctypes.data, c.resting_release_hz, c.release_span_mV,
            c.maximum_release_hz, int(enabled))
        self.graded_enabled = enabled
        self.identity['graded_enabled'] = enabled

    def integrate(self, rgb, duration_ms):
        self.graded_delivered.fill(0)
        result = super().integrate(rgb, duration_ms)
        if not np.isfinite(self.graded_rates).all():
            raise FloatingPointError('Nonfinite graded release')
        if self.graded_enabled and np.any(self.last_counts[self.graded_indices]):
            raise AssertionError('A declared nonspiking L1-L5 cell emitted a point spike')
        result.update(protocol=VERSION, graded_enabled=self.graded_enabled,
            graded_release_hz_quantiles=np.quantile(self.graded_rates, [0, .5, .9, 1]).tolist(),
            graded_delivered_positive=float(self.graded_delivered[0].sum()),
            graded_delivered_negative_magnitude=float(self.graded_delivered[1].sum()))
        return result
