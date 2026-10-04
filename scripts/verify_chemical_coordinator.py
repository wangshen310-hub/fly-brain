"""Actual isolated coordinator restart at a committed chemical memory.

An independent copy resumes exactly one FULL three-arm cycle, then exits at
its committed boundary. Native rollouts, updates, effective weights and RNG
are compared to the original uninterrupted cycle. Original files are read-only.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_chemical_controls as trainer
from flysnake.lab.experiment import atomic_json


def safe_copy(run,name,destination,sha=None):
    if Path(name).name!=name:raise ValueError('Unsafe copied checkpoint/baseline path')
    source=run/name
    if sha and hashlib.sha256(source.read_bytes()).hexdigest()!=sha:raise ValueError('Original memory corrupted')
    shutil.copyfile(source,destination/name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-cycle',type=int,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier coordinator evidence')
    original=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if original['sources']!=trainer.sources():raise ValueError('Chemical trainer source changed')
    cps=[cp for cp in original['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle]
    if not cps or args.checkpoint_cycle>=original['settings']['cycles']:
        raise ValueError('Requested complete-cycle continuation missing')
    args.output.mkdir(parents=True);clone=args.output/'isolated-run';clone.mkdir()
    cp=cps[-1];selected=copy.deepcopy(original)
    selected.update(status='stopped',cycles_completed=args.checkpoint_cycle,
                    cycles=selected['cycles'][:args.checkpoint_cycle],checkpoints=[cp],after={})
    safe_copy(args.run,'perception-baseline.json',clone)
    perception=json.loads((clone/'perception-baseline.json').read_text(encoding='utf-8'))
    safe_copy(args.run,perception['file'],clone,perception['sha256'])
    chemical=selected['chemical_basis'];safe_copy(args.run,chemical['file'],clone,chemical['sha256'])
    safe_copy(args.run,cp['file'],clone,cp['sha256'])
    shutil.copytree(args.run/'sources',clone/'sources')
    atomic_json(clone/'status.json',selected)
    (clone/'STOP_AFTER_CYCLE').write_text('Only the next complete isolated cycle; preserve original trainer',encoding='utf-8')
    settings=selected['settings']
    command=[sys.executable,'-u',str(ROOT/'scripts/train_chemical_controls.py'),
        '--output',str(clone.resolve()),'--baseline-run',settings['baseline_run'],
        '--seed',str(settings['seed']),'--size',str(settings['size']),'--stage',settings['stage'],
        '--limit',str(settings['limit']),'--cycles',str(settings['cycles']),
        '--poses-per-heading',str(settings['poses_per_heading']),
        '--synaptic-step',str(settings['synaptic_step']),'--evaluation-n',str(settings['evaluation_n']),'--resume']
    result={'status':'running','parent_run':str(args.run.resolve()),'checkpoint':cp,
        'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'One actual full isolated coordinator cycle, not ordinary-Snake learning evidence'}
    atomic_json(args.output/'result.json',result)
    with (args.output/'out.log').open('w',encoding='utf-8') as out,(args.output/'err.log').open('w',encoding='utf-8') as err:
        finished=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
    if finished.returncode:
        result.update(status='error',returncode=finished.returncode);atomic_json(args.output/'result.json',result)
        raise RuntimeError('Isolated chemical coordinator failed')
    actual=json.loads((clone/'status.json').read_text(encoding='utf-8'))
    latest=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if actual['status']!='stopped' or actual['cycles_completed']!=args.checkpoint_cycle+1:
        raise RuntimeError('Isolated run did not stop at the next committed boundary')
    if len(latest['cycles'])<=args.checkpoint_cycle:
        result.update(status='awaiting_continuous_cycle',isolated_complete=True);atomic_json(args.output/'result.json',result)
        return
    expected=latest['cycles'][args.checkpoint_cycle];restarted=actual['cycles'][args.checkpoint_cycle]
    if expected['batch']!=restarted['batch']:raise RuntimeError('Full restarted physical worlds differ')
    trials=0
    for arm in trainer.engine.ARMS:
        a,b=expected['arms'][arm],restarted['arms'][arm]
        if a['proposal']!=b['proposal'] or a['update']!=b['update']:raise RuntimeError('Full restarted proposal/update differs')
        for ga,gb in zip(a['groups'],b['groups']):
            for key in ('candidate','parameters_sha256','weights_sha256','food','return'):
                if ga[key]!=gb[key]:raise RuntimeError('Full restarted candidate differs: '+key)
            if [trainer.engine.trial_signature(row) for row in ga['episodes']]!=[
                    trainer.engine.trial_signature(row) for row in gb['episodes']]:
                raise RuntimeError('Full restarted native trials differ')
            trials+=len(gb['episodes'])
    expected_cp=next(cp for cp in latest['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle+1)
    restarted_cp=actual['checkpoints'][-1]
    with np.load(args.run/expected_cp['file'],allow_pickle=False) as a,np.load(clone/restarted_cp['file'],allow_pickle=False) as b:
        for arm in trainer.engine.ARMS:
            if not np.array_equal(a[arm],b[arm]):raise RuntimeError('Full restarted committed parameters differ')
        ma,mb=json.loads(str(a['metadata'])),json.loads(str(b['metadata']))
        if ma['rng']!=mb['rng'] or ma['weights']!=mb['weights']:raise RuntimeError('Full restarted RNG/effective weights differ')
    result.update(status='complete',full_native_trials_match=trials,continuous_match=True,
                  committed_parameters_equal=True,effective_weights_equal=True,rng_equal=True,
                  checkpoint_has_nonzero_learning=cp['parameters_sha256']['learning']!=cp['parameters_sha256']['frozen'],
                  completed_at=time.time())
    atomic_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
