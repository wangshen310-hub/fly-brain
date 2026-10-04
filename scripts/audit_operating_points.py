"""Label-free excitability screen of existing central/visual projection cells.

Anatomy and signed weights remain unchanged. Constant, subthreshold background
currents are explicit engineering hypotheses; they contain no scene/action info.
Only the diagnostic reports, never this script, select actual game actions.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.research.vision import RateRetinaBrain,ReadableGame


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output')
    scenes=json.loads((ROOT/'runtime/board-representation-v1/status.json').read_text())['scenes'][:8]
    brain=RateRetinaBrain();b=brain.b
    metadata=feather.read_table(ROOT/'data/malecns_v1/normalized/neurons.feather',columns=['source_id','superclass']).to_pandas().set_index('source_id').loc[b.ids]
    wake=np.flatnonzero(metadata.superclass.isin(['cb_intrinsic','visual_projection','visual_centrifugal']))
    targets=np.array([np.flatnonzero(b.ids==cell)[0] for cell in (523769,10360)])
    edges=np.flatnonzero(np.isin(b.post,targets)&(b.weight!=0))
    pre=np.searchsorted(b.ptr,edges,side='right')-1
    pre=np.unique(pre[b.modulation_mask[pre]==0])
    report={'protocol':'subthreshold-central-excitability-screen-v1','status':'running',
            'candidate_cells':len(wake),'groups':['cb_intrinsic','visual_projection','visual_centrifugal'],
            'motor_tonic_current':12.,'rows':[], 'scenes':scenes,
            'interpretation':'Label-free operating point screen only. Not physiological validation or proof of learning.'}
    start=time.time()
    for headroom in (None,4.,2.,1.):
        b.tonic.fill(0)
        if headroom is not None:
            b.tonic[wake]=np.maximum(-45-b.rest[wake]-headroom,0)
        b.tonic[targets]=12.
        features=[];motors=[];totals=[]
        for scene in scenes:
            game=ReadableGame(seed=scene['seed'],size=4)
            game.snake=[tuple(x) for x in scene['snake']];game.direction=scene['direction'];game.food=tuple(scene['food'])
            brain.observe(game.render_observation(),learning=False)
            counts=brain.last['counts']
            features.append(counts[pre].copy());motors.append(counts[targets].copy());totals.append(int(counts.sum()))
            # An assay ceiling, not a filtered game action. Preserve failed points.
            if totals[-1]>2000000:break
        x=np.array(features)
        varied=np.ptp(x,axis=0)>0
        unique=np.unique(x[:,varied].T,axis=0)
        row={'headroom_mV':headroom,'scenes_completed':len(features),
             'active_afferents':int(np.any(x>0,axis=0).sum()),'varying_afferents':int(varied.sum()),
             'distinct_varying_patterns':len(unique),'mean_total_spikes':float(np.mean(totals)),
             'maximum_total_spikes':max(totals),'motor_counts':np.array(motors).tolist(),
             'elapsed':time.time()-start,'aborted_saturation':len(features)<len(scenes)}
        report['rows'].append(row)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    report['status']='complete'
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
