"""Audit actual non-selected changed memory and prepare its independent replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome,Dynamics,decode_spikes,file_hash,STATE_ATOL


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    committed=json.loads((run/'committed.json').read_text(encoding='utf-8'));checkpoint=committed['checkpoint']
    payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    torch.set_num_threads(4);b=SpikingConnectome.load_graph(args.graph,config=Dynamics(**payload['identity']['dynamics']))
    data=np.load(Path(args.dataset)/'dataset.npz',allow_pickle=False)
    meta=json.loads((Path(args.dataset)/'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset)/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('Dataset changed')
    pose=meta['rows']['train'][0]['pose_key'];indices=[i for i,r in enumerate(meta['rows']['train']) if r['pose_key']==pose and r['adjacent_eat']]
    images=data['train_rgb'][indices];results={};array_records={};parameter_records={}
    initial={'path':str(run/'initial.pt'),'sha256':file_hash(run/'initial.pt')}
    for arm,point in [('initial',initial),('latest_nonselected',checkpoint)]:
        b.restore_checkpoint(point['path'],point['sha256']);rows=[];allcounts=[];allvoltage=[]
        parameter_records[arm]={k:v.detach().cpu().numpy().copy() for k,v in b.named_parameters()}
        for i,frame in enumerate(images):
            with torch.no_grad():c,m,s=b.integrate_encoded(b.encode(frame[None]))
            raw=m[0].cpu().to(torch.int64).tolist();counts={'left':[raw[0]],'right':[raw[1]],'forward':raw[2:]}
            rows.append({**decode_spikes(counts),'active_neurons':int(torch.count_nonzero(c)),
                'actual_total_spikes':int(c.sum()),'KC_total_spikes':int(c[b.kc].sum()),
                'KC_active_neurons':int(torch.count_nonzero(c[b.kc])),
                'non_graded_cells_within_0_05_of_threshold':int(torch.count_nonzero((torch.abs(s['v'][:,0]-b.config.threshold)<.05)&(~b.graded_mask))),
                'input_sha256':hashlib.sha256(frame.tobytes()).hexdigest()})
            allcounts.append(c.cpu().numpy());allvoltage.append(s['v'].cpu().numpy())
        changed=np.logical_or.reduce([x!=allcounts[0] for x in allcounts[1:]])
        voltage_diff=np.maximum.reduce([np.abs(v-allvoltage[0]) for v in allvoltage[1:]])
        results[arm]={'parameter_sha256':b.parameter_hash(),'rows':rows,
            'food_changes_any_actual_spike_count':bool(changed.any()),'food_changes_neuron_counts':int(changed.sum()),
            'food_changes_motor_counts':any(x['counts']!=rows[0]['counts'] for x in rows[1:]),
            'food_changes_motor_voltage':float(np.max(voltage_diff[b.motor.cpu().numpy()])),
            'food_changes_subthreshold_neurons_over_1e_5':int(np.sum(voltage_diff>1e-5))}
        array_records[arm+'_counts']=np.stack(allcounts);array_records[arm+'_voltage']=np.stack(allvoltage)
        print(json.dumps({'arm':arm,**results[arm]}),flush=True)
    changes={k:{'changed':int(np.count_nonzero(parameter_records['latest_nonselected'][k]!=value)),
                'max_abs':float(np.max(np.abs(parameter_records['latest_nonselected'][k]-value)))}
             for k,value in parameter_records['initial'].items()}
    if results['initial']['parameter_sha256']==b.parameter_hash():raise ValueError('Expected actually changed training memory')
    np.savez_compressed(out/'candidate_arrays.npz',**array_records)
    with (out/'audit.json').open('x',encoding='utf-8') as f:json.dump({'results':results,'parameter_changes':changes,
        'checkpoint':checkpoint,'arrays_sha256':file_hash(out/'candidate_arrays.npz'),
        'selected_for_game':False,'interpretation':'Changed internal parameters are not evidence of direction learning.'},f,indent=2)
    sequence=data['validation_rgb'][:4];state=None
    for frame in sequence[:2]:_,state=b.act(frame,state)
    point=b.save_checkpoint(out/'replay.pt',state=state,parent_checkpoint=checkpoint)
    expected=[];replay_states={}
    for i,frame in enumerate(sequence[2:]):
        receipt,state=b.act(frame,state)
        hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        expected.append({'receipt':receipt,'state':hashes})
        for k,v in state.items():
            if isinstance(v,torch.Tensor):replay_states[f'{i}_{k}']=v.cpu().numpy()
    np.savez_compressed(out/'replay-frames.npz',frames=sequence[2:]);np.savez_compressed(out/'replay-states.npz',**replay_states)
    with (out/'replay.json').open('x',encoding='utf-8') as f:json.dump({'checkpoint':point,'expected':expected,
        'expected_states_sha256':file_hash(out/'replay-states.npz'),'frames_sha256':file_hash(out/'replay-frames.npz'),
        'float_state_atol':STATE_ATOL,'scope':'Actually changed non-selected training parameters, nonzero complete neural state, independent inference replay.'},f,indent=2)


if __name__=='__main__':main()
