"""Explicit, scene-independent intrinsic excitability hypothesis."""
from dataclasses import dataclass,asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.feather as feather


@dataclass(frozen=True)
class OperatingPoint:
    headroom_mV: float = 4.
    groups: tuple = ('cb_intrinsic','visual_projection','visual_centrifugal')


def apply_operating_point(brain,config=OperatingPoint()):
    if not np.isfinite(config.headroom_mV) or config.headroom_mV<=0:
        raise ValueError('A positive subthreshold headroom is required')
    b=brain.b
    metadata=feather.read_table(brain.native.data/'normalized/neurons.feather',
        columns=['source_id','superclass']).to_pandas().set_index('source_id').loc[b.ids]
    indices=np.flatnonzero(metadata.superclass.isin(config.groups))
    if not len(indices):raise ValueError('No annotated cells in operating point')
    if np.intersect1d(indices,np.r_[b.retina,b.r8,b.lamina]).size:
        raise ValueError('Background central drive must not replace the visual input transfer')
    current=np.maximum(-45-b.rest[indices]-config.headroom_mV,0)
    b.tonic[indices]=current
    report={'version':'central-subthreshold-headroom-v1','config':asdict(config),'cells':len(indices),
            'cell_ids_sha256':hashlib.sha256(b.ids[indices].tobytes()).hexdigest(),
            'current_range':[float(current.min()),float(current.max())],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'assumption':'Fixed intrinsic excitability, independent of scene, action and reward; not established in-vivo physiology'}
    brain.source_signature=hashlib.sha256((brain.source_signature+json.dumps(report,sort_keys=True)).encode()).hexdigest()
    return report
