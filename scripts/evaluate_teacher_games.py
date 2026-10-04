"""Frozen ordinary multi-step games; actual brain alone supplies every action."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome,Dynamics,file_hash
from flysnake.research.teacher_game import SpikingSnake,attach_game_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=24);p.add_argument('--max-decisions',type=int,default=100)
    p.add_argument('--size',type=int,default=12);p.add_argument('--seed',type=int,default=71003)
    p.add_argument('--arm',choices=['trained','initial','vision-off','latest-nonselected'],default='trained');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run);status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise ValueError('Evaluate a completed selected frozen run')
    checkpoint=status['best_checkpoint'] if args.arm!='initial' else {'path':str(run/'initial.pt'),'sha256':file_hash(run/'initial.pt')}
    if args.arm=='latest-nonselected':checkpoint=json.loads((run/'committed.json').read_text(encoding='utf-8'))['checkpoint']
    payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);torch.set_num_threads(4)
    b=SpikingConnectome.load_graph(args.graph,config=Dynamics(**payload['identity']['dynamics']))
    b.restore_checkpoint(checkpoint['path'],checkpoint['sha256']);parameter_hash=b.parameter_hash()
    game=SpikingSnake(seed=args.seed,size=args.size,parameter_sha256=parameter_hash,window_ms=b.config.window_ms)
    records=[];events=[];start=time.perf_counter()
    for episode in range(args.episodes):
        if episode:game.new_episode()
        # Episode trials are independent; every decision WITHIN each trial
        # carries full neural voltage, conductance, delays and refractory state.
        state=None;initial=game.snapshot();reward=0.;silent=0;actions=[]
        for k in range(args.max_decisions):
            rgb=game.render_observation();receipt,state=b.act(rgb,state,visual_enabled=args.arm!='vision-off')
            event=game.apply_receipt(attach_game_receipt(receipt,game,parameter_hash))
            if not event['accepted']:raise ValueError('Rejected actual neural receipt: '+event['reason'])
            reward+=event['scalar_reward'];silent+=receipt['action'] is None;actions.append(receipt['action'])
            events.append({'episode':episode,'decision':k,'receipt':receipt,'event':event})
            # Frozen evaluation records outcomes; no feedback-driven parameter
            # update or artificial neuronal reward stimulus is claimed.
            if game.pending_feedback:game.complete_feedback(event['event_id'])
            if game.terminal:break
        row={'episode':episode,'score':game.score,'decisions':len(actions),'silent':silent,'reward':reward,
            'death_reason':game.death_reason,'terminal':game.terminal,'truncated':not game.terminal,
            'initial':initial,'final':game.snapshot(),'actions':actions}
        records.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('initial','final','actions')}),flush=True)
    if b.parameter_hash()!=parameter_hash:raise ValueError('Frozen evaluation changed trained parameters')
    if any(k.endswith('teacher_data') for k in sys.modules):raise ValueError('Teacher present in inference')
    summary={'status':'complete','arm':args.arm,'size':args.size,'episodes':args.episodes,'seed':args.seed,
        'mean_food_score':float(np.mean([r['score'] for r in records])),'total_food':sum(r['score'] for r in records),
        'deaths':sum(r['terminal'] and r['death_reason'] is not None for r in records),
        'silent_decisions':sum(r['silent'] for r in records),'parameter_sha256':parameter_hash,
        'expert_or_external_action_policy':False,'state':'continuous within episode, reset between independent trials',
        'reward_learning_enabled':False,'wall_seconds':time.perf_counter()-start,
        'game_source_sha256':file_hash(ROOT/'flysnake/research/teacher_game.py'),
        'scope':'Single frozen training seed, ordinary random food and growing body. Comparison required; not a final three-seed learning claim.'}
    for name,value in [('summary',summary),('episodes',records),('events',events)]:
        with (out/(name+'.json')).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
