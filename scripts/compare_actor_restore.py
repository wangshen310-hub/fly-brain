"""Read-only comparison of independent native replay with continuous trials."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from train_actor_credit import ARMS,trial_signature
from flysnake.lab.experiment import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous comparison')
    report=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    replay=json.loads(args.replay.read_text(encoding='utf-8'));cp=replay['checkpoint'];index=cp['cycles_completed']
    if replay['status']!='complete' or not replay['independent_reloads_equal']:
        raise ValueError('Independent native restoration has not passed')
    if len(report['cycles'])<=index:raise ValueError('Wait for continuous next cycle to commit')
    if cp not in report['checkpoints']:raise ValueError('Replay checkpoint is not committed in this run')
    actual=report['cycles'][index];r=replay['replay']
    if actual['batch']!=r['batch'] or actual['noise_sha256']!=r['noise_sha256']:
        raise ValueError('Continuous world or neural current noise differs')
    count=0
    for arm in ARMS:
        expected=[row['signature'] for row in r['arms'][arm]]
        received=[trial_signature(row) for row in actual['arms'][arm]['experience'][:len(expected)]]
        if expected!=received:raise ValueError('Continuous actual native trials differ')
        count+=len(expected)
    next_cp=next(c for c in report['checkpoints'] if c['cycles_completed']==index+1)
    file=args.run/next_cp['file']
    if hashlib.sha256(file.read_bytes()).hexdigest()!=next_cp['sha256']:raise ValueError('Continuous memory hash differs')
    with np.load(file,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    if meta['rng']!=r['next_rng']:raise ValueError('Continuous committed RNG differs')
    result={'status':'complete','actual_native_trials_equal':True,'native_trials_compared':count,
            'checkpoint_cycle':index,'continuous_cycle':index+1,'next_rng_equal':True,
            'replay_source_sha256':hashlib.sha256(args.replay.read_bytes()).hexdigest(),
            'continuous_source_sha256':hashlib.sha256((args.run/'status.json').read_bytes()).hexdigest(),
            'interpretation':'Exploration trial subset and RNG matched with actual continuous run; not a complete coordinator restart or behavioral learning test'}
    atomic_json(args.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
