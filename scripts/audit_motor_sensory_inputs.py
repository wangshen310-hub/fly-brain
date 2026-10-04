"""Measure actual existing motor-afferent release at every neural millisecond."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome,Dynamics,file_hash,decode_spikes,STATE_ATOL


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    checkpoint=json.loads((Path(args.run)/'committed.json').read_text(encoding='utf-8'))['checkpoint']
    payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    torch.set_num_threads(4);b=SpikingConnectome.load_graph(args.graph,config=Dynamics(**payload['identity']['dynamics']))
    b.restore_checkpoint(checkpoint['path'],checkpoint['sha256'])
    graph=np.load(Path(args.graph)/'graph.npz',allow_pickle=False);post=graph['post'];pre=graph['pre'];weight=graph['signed_base']
    motor=b.motor.cpu().numpy();selected=np.isin(post,motor)&(weight!=0);source=np.unique(pre[selected])
    device_source=torch.as_tensor(source,device=b.base_rest.device)
    meta=json.loads((Path(args.dataset)/'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset)/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('Dataset changed')
    rows=meta['rows']['train'];pose=rows[0]['pose_key'];ids=[i for i,r in enumerate(rows) if r['pose_key']==pose and r['adjacent_eat']]
    data=np.load(Path(args.dataset)/'dataset.npz',allow_pickle=False);frames=data['train_rgb'][ids]
    arrays={};records=[]
    for i,frame in enumerate(frames):
        light=b.encode(frame[None]);state=b.zero_state(1);releases=[];voltages=[];motor_counts=torch.zeros(4,device=light.device)
        with torch.no_grad():
            for tick in range(b.config.window_ms):
                releases.append(state['release_queue'][state['cursor'],device_source,0].cpu().numpy())
                _,m,state=b.integrate_encoded(light,state,steps=1);motor_counts+=m[0]
                voltages.append(state['v'][b.motor,0].cpu().numpy())
        arrays[f'{i}_arrivals']=np.stack(releases);arrays[f'{i}_motor_voltage']=np.stack(voltages)
        raw=motor_counts.cpu().to(torch.int64).tolist();counts={'left':[raw[0]],'right':[raw[1]],'forward':raw[2:]}
        records.append({'frame':i,'input_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),**decode_spikes(counts)})
    motor_results={}
    for k,cell in zip(('left','right','forward_left','forward_right'),motor):
        edges=(post==cell)&(weight!=0);local_sources=np.unique(pre[edges]);where=np.searchsorted(source,local_sources)
        difference=np.maximum.reduce([np.max(np.abs(arrays[f'{i}_arrivals'][:,where]-arrays['0_arrivals'][:,where]),axis=0) for i in range(1,len(frames))])
        changing=difference>STATE_ATOL
        ix=list(motor).index(cell)
        motor_results[k]={'cell_id':int(graph['ids'][cell]),'existing_fast_edges':int(edges.sum()),'input_neurons':len(local_sources),
            'food_changes_actual_afferent_release_time_series':int(changing.sum()),
            'changing_input_cell_ids':graph['ids'][local_sources[changing]].tolist(),
            'maximum_release_difference':float(difference.max()),
            'maximum_voltage_time_series_difference':max(float(np.max(np.abs(arrays[f'{i}_motor_voltage'][:,ix]-arrays['0_motor_voltage'][:,ix]))) for i in range(1,len(frames)))}
    arrays['input_neuron_ids']=graph['ids'][source];np.savez_compressed(out/'actual_inputs.npz',**arrays)
    result={'status':'complete','checkpoint':checkpoint,'parameter_sha256':b.parameter_hash(),'frames':records,
        'motor_input_results':motor_results,'all_neurons':b.n,'anatomical_edges':b.manifest['anatomical_edges'],
        'sampling':'Actual delayed release_queue arriving at each 1ms tick, before existing full-graph delivery; not inferred from final counts.',
        'arrays_sha256':file_hash(out/'actual_inputs.npz'),'source_sha256':file_hash(__file__),
        'scope':'One fixed pose, three actual food images; not all poses or a proof of biological incapacity.'}
    with (out/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(motor_results),flush=True)


if __name__=='__main__':main()
