"""Measure whether native motor afferents contain opposite-cue drive directions."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import cue_scene


brain=LabBrain();b=brain.b
samples=[]
for seed in (444,555):
    for side in ('left','right'):
        r=brain.observe(cue_scene(seed,side),learning=False)
        samples.append(brain.last['counts'].copy())
samples=np.array(samples)
mean_left=samples[::2].mean(axis=0);mean_right=samples[1::2].mean(axis=0)
rows=[]
for label,cell in [('DNp20_L',10162),('DNp20_R',10059),('DNpe017_A',10527),('DNpe017_B',555871)]:
    ix=int(np.flatnonzero(b.ids==cell)[0]);edges=np.flatnonzero(b.post==ix)
    pre=np.searchsorted(b.ptr,edges,side='right')-1
    fast=b.modulation_mask[pre]==0;edges=edges[fast];pre=pre[fast]
    weighted_difference=(mean_right[pre]-mean_left[pre])*b.weight[edges]
    active=(mean_left[pre]+mean_right[pre])>0
    row={'cell':cell,'label':label,'motor_counts':samples[:,ix].tolist(),
         'fast_afferents':len(edges),'active_afferents':int(active.sum()),
         'net_prefers_left':int((weighted_difference<-.01).sum()),
         'net_prefers_right':int((weighted_difference>.01).sum()),
         'strongest_left':float(weighted_difference.min()),'strongest_right':float(weighted_difference.max()),
         'varying_edges':[{'edge':int(edges[j]),'pre_id':int(b.ids[pre[j]]),'weight':float(b.weight[edges[j]]),
                          'left_spikes':float(mean_left[pre[j]]),'right_spikes':float(mean_right[pre[j]])}
                         for j in np.flatnonzero(np.abs(weighted_difference)>.01)]}
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='varying_edges'}),flush=True)
Path('diagnostics/motor-afferent-information-2026-09-22.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
