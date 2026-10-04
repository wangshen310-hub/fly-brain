"""Actual nonzero local-APL state saved then replayed by a different interpreter."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.apl_local import LocalAPLBrain,file_hash
from flysnake.research.apl_local_state_v2 import save,load,VERSION
from flysnake.research.paired import digest,writer_lock
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import inputs


def snapshot(brain):
    return {'native':{name:digest(getattr(brain.b,name)) for name in brain.b.fields},
        'adapter_retinal_light':digest(brain.retinal_light),'adapter_r8_light':digest(brain.r8_light),
        'tonic':digest(brain.b.tonic),'weights':digest(brain.b.weight),'cursor':brain.b.cursor}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parent=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if parent['status']!='complete':raise ValueError('Completed local APL bundle required')
    for name,sha in parent['sources'].items():
        if file_hash(ROOT/name)!=sha or file_hash(args.run/'sources'/name)!=sha:raise ValueError('Locked native source differs')
    path=args.output/'status.json'
    if not args.replay:
        if path.exists():raise ValueError('Preserve previous v2 state experiment')
        sources=dict(parent['sources'])
        for name in ('scripts/assay_apl_local_state.py','flysnake/research/apl_local_state_v2.py'):sources[name]=file_hash(ROOT/name)
        report={'version':VERSION,'status':'loading','process_id':os.getpid(),'started_at':time.time(),
            'parent_run':str(args.run),'parent_status_sha256':file_hash(args.run/'status.json'),'sources':sources,
            'old_v1_checkpoint_unchanged':True,'game_played':False,'learning':False}
        for name in sources:
            target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        atomic_json(path,report)
    else:
        report=json.loads(path.read_text(encoding='utf-8'))
        for name,sha in report['sources'].items():
            if file_hash(ROOT/name)!=sha or file_hash(args.output/'sources'/name)!=sha:raise ValueError('State protocol source differs')
        if report['process_id']==os.getpid():raise ValueError('Distinct OS process required')
    brain=LocalAPLBrain(args.run,attachment=args.run/'geometry');_,rgb,c=inputs(brain)[1]
    if args.replay:
        checkpoint=report['checkpoint'];load(brain,args.output/checkpoint['file'],checkpoint['sha256'])
        if snapshot(brain)!=report['checkpoint_state']:raise AssertionError('Initial saved-state reconstruction differs')
        receipt=brain.integrate(rgb,100,c);state=snapshot(brain)
        if receipt!=report['continuation_receipt'] or state!=report['continuation_state']:
            raise AssertionError('Fresh-process actual neural continuation differs')
        atomic_json(args.output/'replay.json',{'status':'complete','process_id':os.getpid(),'parent_process_id':report['process_id'],
            'all_native_and_adapter_states_equal':True,'receipt_equal':True,'observation_ms':100,
            'native_arrays':len(state['native']),'local_u_nonzero':bool(np.any(brain.kernel.u))})
        print(json.dumps({'replay':'complete','native_arrays':len(state['native'])}),flush=True);return
    brain.reset_trial();brain.integrate(rgb,150,c)
    if not np.any(brain.kernel.u) or not np.any(brain.kernel.inhibition):raise AssertionError('Replay checkpoint must contain nonzero APL state')
    report['checkpoint_state']=snapshot(brain)
    report['checkpoint']={'file':'state-v2.npz','sha256':save(brain,args.output/'state-v2.npz')}
    report['continuation_receipt']=brain.integrate(rgb,100,c);report['continuation_state']=snapshot(brain)
    report.update(status='saved-awaiting-independent-replay');atomic_json(path,report)
    command=[sys.executable,str(Path(__file__).resolve()),'--run',str(args.run),'--output',str(args.output),'--replay']
    with (args.output/'replay.out.log').open('w',encoding='utf-8') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Independent state replay failed; see replay.out.log')
    replay=json.loads((args.output/'replay.json').read_text(encoding='utf-8'))
    report.update(status='complete',completed_at=time.time(),replay=replay,
        interpretation='Nonzero local cable, receptor, delay queue, eligibility, full native state and adapter restored exactly; not learning evidence.')
    atomic_json(path,report);print(json.dumps({'status':'complete','replay':replay}),flush=True)


if __name__=='__main__':main()
