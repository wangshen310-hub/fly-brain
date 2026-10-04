"""Development-only assay of signed existing turn afferents; no saved game memory."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import cue_scene,neutral_scene


brain=LabBrain();b=brain.b
motor=np.r_[brain.native.groups['left'],brain.native.groups['right']]
brain.edges=np.flatnonzero(np.isin(b.post,motor)&(b.weight!=0))
brain.pre=np.searchsorted(b.ptr,brain.edges,side='right')-1
brain.edge_group=np.where(np.isin(b.post[brain.edges],brain.native.groups['left']),0,2)
brain.base=b.weight[brain.edges].copy();brain.fraction=np.ones(len(brain.edges))
brain.trace=np.zeros(len(brain.edges));brain.pre_baseline=np.zeros(len(brain.edges));brain.pre_scale=np.ones(len(brain.edges))*10
brain.calibrate([neutral_scene(i+901) for i in range(8)])
print('signed_existing_turn_edges',len(brain.edges),flush=True)
rows=[]
for i in range(64):
 side='left' if i%2==0 else 'right';rgb=cue_scene(99000+i,side)
 result=brain.observe(rgb,learning=False)
 # A presynaptic spike qualifies an existing synapse; inhibitory magnitudes
 # have the opposite effect on postsynaptic excitability to excitatory ones.
 pre=np.clip(brain.last['counts'][brain.pre]/.3/brain.pre_scale,0,3)
 scores=result['scores'];gap=(scores['left']-scores['right'])/2
 post=np.clip(np.where(brain.edge_group==0,gap,-gap),-3,3)
 brain.trace=(1-np.exp(-.3))*pre*post*np.sign(brain.base)
 target='right' if side=='left' else 'left'
 correct=result['action']==target
 feedback=brain.feedback('reward' if correct else 'aversive')
 rows.append({'i':i,'side':side,'action':result['action'],'correct':correct,'feedback':feedback})
 if i%8==7:print(json.dumps({'through':i+1,'correct_last8':sum(x['correct'] for x in rows[-8:]),'fraction_min':float(brain.fraction.min()),'fraction_max':float(brain.fraction.max())}),flush=True)
Path('diagnostics/pilot-signed-plasticity.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
