"""Finish the comparison after an interrupted isolated restart wrapper.

The original isolated trainer is resumed at its saved boundary, rather than
inventing a completed verification or overwriting its original memory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_chemical_controls as trainer
from flysnake.lab.experiment import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);args=p.parse_args()
    path=args.evidence/'result.json';result=json.loads(path.read_text(encoding='utf-8'))
    original=Path(result['parent_run']);clone=args.evidence/'isolated-run'
    actual=json.loads((clone/'status.json').read_text(encoding='utf-8'))
    continuous=json.loads((original/'status.json').read_text(encoding='utf-8'))
    cycle=result['checkpoint']['cycles_completed']
    if actual['sources']!=trainer.sources() or continuous['sources']!=trainer.sources():
        raise ValueError('Verified native sources changed')
    if actual['status']!='stopped' or actual['cycles_completed']!=cycle+1:
        raise ValueError('Actual isolated full cycle has not committed and exited')
    if len(continuous['cycles'])<=cycle:raise ValueError('Continuous full cycle has not committed')
    if actual['cycles'][cycle]['batch']!=continuous['cycles'][cycle]['batch']:
        raise RuntimeError('Isolated physical worlds differ')
    trials=0
    for arm in trainer.engine.ARMS:
        a,b=continuous['cycles'][cycle]['arms'][arm],actual['cycles'][cycle]['arms'][arm]
        if a['proposal']!=b['proposal'] or a['update']!=b['update']:raise RuntimeError('Restarted proposal/update differs')
        if len(a['groups'])!=len(b['groups']):raise RuntimeError('Restarted candidate count differs')
        for ga,gb in zip(a['groups'],b['groups']):
            for key in ('candidate','parameters_sha256','weights_sha256','food','return'):
                if ga[key]!=gb[key]:raise RuntimeError('Restarted candidate differs: '+key)
            if [trainer.engine.trial_signature(row) for row in ga['episodes']]!=[
                    trainer.engine.trial_signature(row) for row in gb['episodes']]:
                raise RuntimeError('Restarted native spikes/actions/feedback differ')
            trials+=len(gb['episodes'])
    expected_cp=next(cp for cp in continuous['checkpoints'] if cp['cycles_completed']==cycle+1)
    restored_cp=actual['checkpoints'][-1]
    for directory,cp in ((original,expected_cp),(clone,restored_cp)):
        if Path(cp['file']).name!=cp['file'] or hashlib.sha256((directory/cp['file']).read_bytes()).hexdigest()!=cp['sha256']:
            raise ValueError('Committed coordinator memory hash differs')
    with np.load(original/expected_cp['file'],allow_pickle=False) as a,np.load(clone/restored_cp['file'],allow_pickle=False) as b:
        for arm in trainer.engine.ARMS:
            if not np.array_equal(a[arm],b[arm]):raise RuntimeError('Restarted committed parameters differ')
        ma,mb=json.loads(str(a['metadata'])),json.loads(str(b['metadata']))
        if ma['rng']!=mb['rng'] or ma['weights']!=mb['weights']:raise RuntimeError('Restarted RNG/weights differ')
    result.update(status='complete',full_native_trials_match=trials,continuous_match=True,
        committed_parameters_equal=True,effective_weights_equal=True,rng_equal=True,
        isolated_uncommitted_cycle_interrupted_and_redone=True,
        comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_has_nonzero_learning=result['checkpoint']['parameters_sha256']['learning']!=result['checkpoint']['parameters_sha256']['frozen'],
        completed_at=time.time())
    atomic_json(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
