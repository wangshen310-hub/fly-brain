"""Bounded frozen calibration of the proposed linear photoreceptor interface."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flysnake.neural import RealFlyBrain
from flysnake.game import SnakeGame

brain = RealFlyBrain(learning=False)
b = brain.brain
g = SnakeGame()
g.snake = [(6,6),(6,7),(6,8)]
ix = np.concatenate(list(brain.groups.values()))
edges = np.flatnonzero(np.isin(b.post, ix) & (b.weight > 0))
pre = np.searchsorted(b.ptr, edges, side='right')-1
print('motor_afferents', len(edges), 'unique_pre',len(np.unique(pre)), flush=True)
for side,food in [('left',(5,6)),('right',(7,6)),('gray',None)]:
 g.food=food
 frame=g.render_observation() if side!='gray' else np.full((180,320,3),128,dtype=np.uint8)
 h,w=frame.shape[:2]
 def pixels(uv):
  a=frame[(uv[:,1]*(h-1)).astype(int),(uv[:,0]*(w-1)).astype(int)].astype(float)/255
  return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
 l=pixels(b.uv)@np.array([.2126,.7152,.0722]); rgb=pixels(b.r8_uv); r8=rgb[np.arange(len(rgb)),b.r8_channel]
 for gain in [20,30]:
  b.reset(); b.weights_frozen=True
  total=np.zeros(b.n,dtype=np.int32)
  for n in range(30):
   counts,_=b.step(np.zeros(len(b.retina)),10,learning=False,stimulation=[(b.retina,gain*l),(b.r8,gain*r8)])
   total+=counts
  print(json.dumps({'scene':side,'gain':gain,'counts':{k:total[v].tolist() for k,v in brain.groups.items()},'afferent_active':int(np.count_nonzero(total[pre])),'mbon':total[b.circuit['mb']].tolist()}),flush=True)
