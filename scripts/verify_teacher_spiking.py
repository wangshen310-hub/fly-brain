"""Fresh-process replay of actual trained full-graph state, with no teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome, Dynamics, file_hash, STATE_ATOL


def state_hashes(state):
    return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            if isinstance(v,torch.Tensor) else v for k,v in state.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--run',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    run=Path(args.run);record=json.loads((run/'replay.json').read_text(encoding='utf-8'))
    payload=torch.load(record['checkpoint']['path'],map_location='cpu',weights_only=True)
    cfg=Dynamics(**payload['identity']['dynamics']);torch.set_num_threads(4)
    b=SpikingConnectome.load_graph(args.graph,device=args.device,config=cfg)
    state,_=b.restore_checkpoint(record['checkpoint']['path'],record['checkpoint']['sha256'])
    if (file_hash(run/'replay-frames.npz')!=record['frames_sha256']
            or file_hash(run/'replay-states.npz')!=record['expected_states_sha256']
            or record['float_state_atol']!=STATE_ATOL):raise ValueError('Replay fixture integrity/tolerance mismatch')
    frames=np.load(run/'replay-frames.npz',allow_pickle=False)['frames']
    expected_states=np.load(run/'replay-states.npz',allow_pickle=False)
    actual=[];checks=[]
    for i,frame in enumerate(frames):
        receipt,state=b.act(frame,state)
        actual.append({'receipt':receipt,'state':state_hashes(state)})
        reference=record['expected'][i];expected_receipt=reference['receipt']
        float_fields=('voltage_min','voltage_max')
        receipts_exact=all(receipt[k]==v for k,v in expected_receipt.items() if k not in float_fields)
        numeric={k:float(np.max(np.abs(v.cpu().numpy()-expected_states[f'{i}_{k}'])))
                 for k,v in state.items() if isinstance(v,torch.Tensor)}
        floats_valid=all(v<=STATE_ATOL for k,v in numeric.items() if k!='refractory')
        ticks_valid=all(state[k]==reference['state'][k] for k in ('ticks','cursor'))
        receipts_float_valid=all(abs(receipt[k]-expected_receipt[k])<=STATE_ATOL for k in float_fields)
        checks.append({'actual_spikes_and_receipt_exact':receipts_exact,'refractory_exact':numeric['refractory']==0,
            'timing_exact':ticks_valid,'float_state_within_preregistered_atol':floats_valid,
            'receipt_float_within_atol':receipts_float_valid,'max_abs':numeric})
    match=all(all(v for k,v in c.items() if k!='max_abs') for c in checks)
    result={'status':'complete','fresh_process':True,'matched':match,'conditions':len(actual),
        'all_state_hashes_byte_exact':actual==record['expected'],'checks':checks,'float_state_atol':STATE_ATOL,
        'parameter_sha256':b.parameter_hash(),'expert_imported':any(k.endswith('teacher_data') for k in sys.modules),
        'ordinary_game_learning_validated':False,'actual':actual,
        'interpretation':'Finite full-state continuation of the new trained protocol, not old game memory migration.'}
    if result['expert_imported']:raise ValueError('Teacher must be absent from inference process')
    output=Path(args.output)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='actual'}),flush=True)
    if not match:raise ValueError('Independent complete state continuation mismatch')


if __name__=='__main__':main()
