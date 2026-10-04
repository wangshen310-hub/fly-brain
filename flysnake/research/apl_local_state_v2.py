"""Independent local-APL state protocol; native/adapter fields cannot collide.

The locked v1 diagnostic and its failed replay checkpoint are preserved. This
module does not migrate missing state by guessing; new checkpoints retain it.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import numpy as np
from .apl_local import file_hash
from .paired import digest

VERSION='native-local-APL-complete-state-v2'


def save(brain,path):
    b=brain.b
    metadata={'version':VERSION,'source_sha256':file_hash(__file__),
        'identity':brain.identity,'current_cable_config':asdict(brain.kernel.config),
        'cursor':b.cursor,'neural_ms':brain.neural_ms,'sim_ms':b.sim_ms,'total_spikes':b.total_spikes,
        'all_weights_sha256':digest(b.weight),'native_fields':list(b.fields)}
    arrays={'native_'+name:getattr(b,name) for name in b.fields}
    arrays.update(plastic_weight=b.weight[brain.edges],native_tonic=b.tonic,
        adapter_retinal_light=brain.retinal_light,adapter_r8_light=brain.r8_light,
        metadata=np.array(json.dumps(metadata,sort_keys=True)))
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return file_hash(path)


def load(brain,path,sha):
    if file_hash(path)!=sha:raise ValueError('Complete APL checkpoint hash differs')
    b=brain.b
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']))
        if (metadata['version']!=VERSION or metadata['source_sha256']!=file_hash(__file__)
                or metadata['identity']!=json.loads(json.dumps(brain.identity))
                or metadata['current_cable_config']!=asdict(brain.kernel.config)
                or metadata['native_fields']!=list(b.fields)):
            raise ValueError('Complete APL checkpoint source, graph or cable configuration differs')
        destinations={'native_'+name:getattr(b,name) for name in b.fields}
        destinations.update(native_tonic=b.tonic,adapter_retinal_light=brain.retinal_light,adapter_r8_light=brain.r8_light)
        for name,destination in destinations.items():
            value=z[name]
            if value.shape!=destination.shape or value.dtype!=destination.dtype or not np.isfinite(value).all():
                raise ValueError('Complete APL checkpoint invalid field: '+name)
        weights=z['plastic_weight']
        if (weights.shape!=brain.base.shape or weights.dtype!=brain.base.dtype or not np.isfinite(weights).all()
                or (weights<brain.base*.1).any() or (weights>brain.base).any()):
            raise ValueError('Complete APL checkpoint invalid memory weights')
        prospective=b.weight.copy();prospective[brain.edges]=weights
        if digest(prospective)!=metadata['all_weights_sha256']:raise ValueError('Complete APL full-graph weight hash differs')
        if (type(metadata['cursor']) is not int or metadata['cursor']<0
                or not np.isfinite([metadata['neural_ms'],metadata['sim_ms'],metadata['total_spikes']]).all()
                or metadata['sim_ms']!=metadata['cursor']*.1):
            raise ValueError('Complete APL checkpoint invalid clock')
        # All validation precedes any in-memory mutation.
        for name,destination in destinations.items():destination[:]=z[name]
        b.weight[brain.edges]=weights
    b.cursor=metadata['cursor'];brain.neural_ms=metadata['neural_ms'];b.sim_ms=metadata['sim_ms'];b.total_spikes=metadata['total_spikes']
    return metadata
