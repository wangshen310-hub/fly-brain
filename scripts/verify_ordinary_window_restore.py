"""Real ordinary-Snake window continuation after a full native reload.

First play a prefix and commit its dynamic state. Compare an actual continuous
suffix with an independently loaded model restored at that window boundary.
This checks recurrent state and game progress, not behavioral learning.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_chemical_controls as trainer
import run_ordinary_chemical_snake as ordinary
from verify_chemical_restore import load_brain
from flysnake.research.continuous_state import save_window,restore_window,state_arrays,game_state
from flysnake.research.pulse_balance import PulseBalancedGame
from flysnake.research.paired import digest,writer_lock
from flysnake.lab.experiment import atomic_json


def make_brain_game(run,parent,seed):
    brain=load_brain(run,parent);memory,_=trainer.engine.restore_memory(run,parent,brain)
    brain.set_parameters(memory['learning'])
    game=PulseBalancedGame(seed=seed,size=parent['settings']['size'],stage='uniform_food',
                          geometry=brain.geometry,calibration=brain.brain.calibration)
    brain.reset_trial();return brain,game


def step(brain,game):
    if game.terminal:game.new_episode();brain.reset_trial()
    result=ordinary.advance(brain,game)
    receipt=result['receipt']
    signature={'receipt':{k:receipt[k] for k in ('counts','action','olfactory_receptor_spikes',
        'observation_sha256','eligibility_sha256','memory_sha256','neural_time_start','neural_time_end')},
        'sense':result['sense'],'event':result['event'],'reward':result['reward'],
        'feedback':result['feedback'],'game':game_state(game),
        'native_dynamic_state':{k:digest(v) for k,v in state_arrays(brain).items()},
        'native_cursor':brain.brain.b.cursor,'neural_ms':brain.brain.neural_ms}
    return json.loads(json.dumps(signature))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-cycle',type=int,required=True)
    p.add_argument('--prefix',type=int,default=2);p.add_argument('--suffix',type=int,default=3)
    p.add_argument('--seed',type=int,default=930902);args=p.parse_args()
    if min(args.prefix,args.suffix)<1:raise ValueError('Positive actual native prefix/suffix required')
    if args.output.exists():raise ValueError('Preserve earlier ordinary restore evidence')
    parent=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if parent['sources']!=trainer.sources():raise ValueError('Native parent source changed')
    cps=[cp for cp in parent['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle]
    if not cps:raise ValueError('Requested committed brain memory missing')
    parent['checkpoints']=[cps[-1]]
    args.output.mkdir(parents=True)
    with writer_lock(args.output):
        path=args.output/'result.json'
        result={'status':'running','parent_checkpoint':cps[-1],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'interpretation':'Actual ordinary multi-step window continuation after native reload; no claim of learning'}
        atomic_json(path,result)
        source=ordinary.sources(parent)
        source['scripts/verify_ordinary_window_restore.py']=result['source_sha256']
        brain,game=make_brain_game(args.run,json.loads(json.dumps(parent)),args.seed)
        report={'sources':source,'settings':{'seed':args.seed,'stage':'uniform_food','frozen':True},
                'totals':{},'windows_completed':args.prefix,'parent_checkpoint':cps[-1],'checkpoints':[]}
        prefix=[step(brain,game) for _ in range(args.prefix)]
        cp=save_window(args.output,report,brain,game)
        expected=[step(brain,game) for _ in range(args.suffix)]
        del brain,game;gc.collect()
        brain,game=make_brain_game(args.run,json.loads(json.dumps(parent)),args.seed)
        restore_window(args.output,report,brain,game)
        actual=[step(brain,game) for _ in range(args.suffix)]
        if expected!=actual:
            result.update(status='error',expected=expected,actual=actual);atomic_json(path,result)
            raise RuntimeError('Restored ordinary native windows differ from continuous play')
        result.update(status='complete',prefix=prefix,continuation=actual,window_checkpoint=cp,
                      independent_native_reload_matches_continuous=True,
                      all_dynamic_arrays_equal_after_every_suffix_window=True,
                      ordinary_prefix_windows=args.prefix,ordinary_suffix_windows=args.suffix,
                      checkpoint_has_nonzero_learning=cps[-1]['parameters_sha256']['learning']!=cps[-1]['parameters_sha256']['frozen'],
                      completed_at=time.time())
        atomic_json(path,result);print(json.dumps({k:result[k] for k in ('status',
            'independent_native_reload_matches_continuous','ordinary_suffix_windows','checkpoint_has_nonzero_learning')}),flush=True)


if __name__=='__main__':main()
