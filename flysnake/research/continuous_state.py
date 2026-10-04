"""Atomic ordinary-Snake window checkpoints, including neural dynamic state.

These are boundary snapshots, never mid-kernel files. Frozen synaptic inference
can be resumed with its genuine recurrent state and future food RNG intact.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from flysnake.lab.experiment import atomic_json
from .paired import digest

VERSION='native-continuous-window-state-v1'


def nested_tuple(value):
    return tuple(nested_tuple(v) for v in value) if isinstance(value,list) else value


def game_state(game):
    if game.pending_feedback is not None or game._feedback_observation is not None:
        raise ValueError('Feedback must finish before an ordinary-game checkpoint')
    return {'seed':game.seed,'size':game.size,'episode_id':game.episode_id,'step_id':game.step_id,
        'snake':[list(p) for p in game.snake],'direction':game.direction,
        'food':list(game.food) if game.food is not None else None,'score':game.score,
        'terminal':game.terminal,'death_reason':game.death_reason,'outcome':game.outcome,
        'last_event':game.last_event,'last_neural_end':game._last_neural_end,
        'completed_feedback':sorted(game._completed_feedback),'rng':game._rng.getstate()}


def restore_game(game,state):
    if game.seed!=state['seed'] or game.size!=state['size']:raise ValueError('Ordinary game identity differs')
    for key in ('episode_id','step_id','direction','score','terminal','death_reason','outcome','last_event'):
        setattr(game,key,state[key])
    game.snake=[tuple(p) for p in state['snake']]
    game.food=tuple(state['food']) if state['food'] is not None else None
    game._last_neural_end=state['last_neural_end']
    game._completed_feedback=set(state['completed_feedback'])
    game._rng.setstate(nested_tuple(state['rng']))
    game._pending_feedback=None;game._feedback_observation=None


def state_arrays(brain):
    b=brain.brain.b
    return {'native_'+name:getattr(b,name).copy() for name in b.fields}|{
        'retinal_light':brain.brain.retinal_light.copy(),'r8_light':brain.brain.r8_light.copy(),
        'parameters':brain.coefficients.copy(),'concentration':brain.concentration.copy(),
        'tonic':b.tonic.copy()}


def save_window(output,report,brain,game):
    arrays=state_arrays(brain);b=brain.brain.b
    meta={'version':VERSION,'identity':brain.identity,'sources':report['sources'],
        'settings':report['settings'],'game':game_state(game),'totals':report['totals'],
        'windows_completed':report['windows_completed'],'parent_checkpoint':report['parent_checkpoint'],
        'cursor':b.cursor,'native_total_spikes':b.total_spikes,'neural_ms':brain.brain.neural_ms,
        'weights_sha256':brain.weight_hash(),'array_sha256':{k:digest(a) for k,a in arrays.items()}}
    name=f'window-{report["windows_completed"]:08d}-{time.time_ns()}.npz'
    with (output/name).open('xb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(meta,sort_keys=True),**arrays)
        stream.flush();os.fsync(stream.fileno())
    cp={'file':name,'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),
        'windows_completed':report['windows_completed'],'cycles_completed':report['parent_checkpoint']['cycles_completed'],
        'weights_sha256':meta['weights_sha256']}
    report['checkpoints'].append(cp);atomic_json(output/'status.json',report)
    return cp


def restore_window(output,report,brain,game):
    cp=report['checkpoints'][-1]
    if Path(cp['file']).name!=cp['file']:raise ValueError('Unsafe ordinary checkpoint file')
    path=output/cp['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cp['sha256']:raise ValueError('Ordinary checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        expected=json.loads(json.dumps({'version':VERSION,'identity':brain.identity,
                                       'sources':report['sources'],'settings':report['settings']}))
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Ordinary source/graph/configuration differs')
        if meta['windows_completed']!=cp['windows_completed']:raise ValueError('Ordinary committed window differs')
        # Validate ALL arrays before changing any native dynamic state.
        current=state_arrays(brain)
        if set(current)!=set(meta['array_sha256']):raise ValueError('Ordinary native state fields differ')
        for key,target in current.items():
            if (key not in z or z[key].shape!=target.shape or z[key].dtype!=target.dtype
                    or digest(z[key])!=meta['array_sha256'][key]):
                raise ValueError('Ordinary native state corrupted: '+key)
        brain.set_parameters(z['parameters'])
        if brain.weight_hash()!=meta['weights_sha256'] or brain.weight_hash()!=cp['weights_sha256']:
            raise ValueError('Ordinary effective weights differ')
        b=brain.brain.b
        for name in b.fields:getattr(b,name)[:]=z['native_'+name]
        brain.brain.retinal_light[:]=z['retinal_light'];brain.brain.r8_light[:]=z['r8_light']
        brain.concentration=z['concentration'].copy();b.tonic[:]=z['tonic']
    b.cursor=meta['cursor'];b.sim_ms=b.cursor*b.dt;b.total_spikes=meta['native_total_spikes']
    brain.brain.neural_ms=meta['neural_ms'];brain.brain.last=None
    brain.motor_noise=np.zeros(2)
    restore_game(game,meta['game']);report['windows_completed']=meta['windows_completed']
    report['totals']=meta['totals'];report['parent_checkpoint']=meta['parent_checkpoint']
    return meta
