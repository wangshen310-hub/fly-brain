"""Frozen visual, full-gain-reset and one-synapse ablations of actual spikes."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import Dynamics,file_hash
from flysnake.research.teacher_afferent import PlasticAfferentBrain


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise ValueError('Frozen completed probe required')
    checkpoint=status['best_checkpoint'];payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    torch.set_num_threads(4);b=PlasticAfferentBrain.load_graph(args.graph,config=Dynamics(**payload['identity']['dynamics']))
    b.restore_checkpoint(checkpoint['path'],checkpoint['sha256']);before=b.parameter_hash()
    dataset=np.load(Path(status['config']['dataset'])/'dataset.npz',allow_pickle=False);frames=dataset['train_rgb'][status['train_indices']]
    graph=np.load(Path(args.graph)/'graph.npz',allow_pickle=False);input_ids=graph['ids'][b.aff_pre.cpu().numpy()]
    target=int(b.motor[0]);which=np.flatnonzero((input_ids==13920)&(b.aff_post.cpu().numpy()==target))
    if len(which)!=1:raise ValueError('Expected one existing observed inhibitory synapse')
    index=int(which[0]);factor=float(torch.exp(4.*torch.tanh(b.afferent_log_gain[index])))
    base=float(-b.aff_abs[index] if not b.aff_exc[index] else b.aff_abs[index])
    conditions={}
    for arm in ['trained','vision-off','all-afferent-gains-reset','input-13920-gain-reset']:
        b.restore_checkpoint(checkpoint['path'],checkpoint['sha256'])
        with torch.no_grad():
            if arm=='all-afferent-gains-reset':b.afferent_log_gain.zero_()
            if arm=='input-13920-gain-reset':b.afferent_log_gain[index]=0.
        receipts=[b.act(frame,visual_enabled=arm!='vision-off')[0] for frame in frames]
        conditions[arm]={'actions':[r['action'] for r in receipts],'receipts':receipts,
            'parameter_sha256':b.parameter_hash()}
    b.restore_checkpoint(checkpoint['path'],checkpoint['sha256'])
    if b.parameter_hash()!=before:raise ValueError('Counterfactual audit failed to restore candidate')
    result={'status':'complete','conditions':conditions,'checkpoint':checkpoint,
        'existing_synapse':{'pre_id':13920,'post_id':523769,'base_signed_weight':base,
            'learned_gain_factor':factor,'learned_signed_weight':base*factor},
        'final_parameter_sha256':before,'teacher_imported':any(k.endswith('teacher_data') for k in sys.modules),
        'scope':'Three training images only; mechanistic causal ablations, not heldout or ordinary-game success.'}
    with (out/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'actions':{k:v['actions'] for k,v in conditions.items()},'existing_synapse':result['existing_synapse']}),flush=True)


if __name__=='__main__':main()
