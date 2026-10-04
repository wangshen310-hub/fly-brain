"""Independent optional kernel candidate, disabled unless a full audit passes.

No existing kernel or checkpoint dependency is overwritten. The one-tick fast
path preserves expression order and retains the complete general path and graph.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).parent/'kernels/full_graph_tick_fast.cpp'
VERSION='full-connectome-identical-tick-fast-candidate-v1'


def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    provenance=json.loads(SOURCE.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT/provenance['original_source']).read_bytes()).hexdigest()!=provenance['original_sha256']:
        raise ValueError('Retained baseline kernel source changed')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=provenance['candidate_sha256']:
        raise ValueError('Candidate kernel source changed')
    library=directory/'full_tick_fast.dll';metadata=directory/'build.json'
    if metadata.exists() and library.exists():
        record=json.loads(metadata.read_text())
        if record['source']==provenance and record['binary_sha256']==hashlib.sha256(library.read_bytes()).hexdigest():
            return library,record
    # Same conservative target and compiler flags as the retained kernel.
    # No fast-math, coarser dt, SIMD reorder, thread reorder or FMA added.
    command=[sys.executable,'-m','ziglang','c++','-O3','-std=c++17','-shared','-target','x86_64-windows-gnu',str(SOURCE),'-o',str(library)]
    with (directory/'compile.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Independent native candidate compilation failed')
    record={'version':VERSION,'source':provenance,'command':command,
            'binary_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
    metadata.write_text(json.dumps(record,indent=2),encoding='utf-8');return library,record


def bind(brain,library):
    loaded=C.CDLL(str(Path(library).resolve()));advance=loaded.memory_advance
    advance.argtypes=brain.advance.argtypes;advance.restype=brain.advance.restype
    return loaded,advance
