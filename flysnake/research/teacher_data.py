"""Training-only demonstrator and reproducible scene split.

Never imported by the neural inference module. Coordinates are used here only
to generate physical RGB scenes and teacher labels explicitly authorized by
the user. The teacher is forbidden from selecting live neural game actions.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from flysnake.game import DIRECTIONS
from .vision import ReadableGame

VERSION = 'RGB-supervised-pose-disjoint-short-Snake-scenes-v1'
ACTIONS = ('left','forward','right')


def teacher_action(snake, direction, food, size):
    """First shortest safe step, training ONLY; no live caller exists."""
    if not snake or food is None:
        return None
    head=tuple(snake[0]); food=tuple(food)
    blocked=set(map(tuple,snake[:-1]))
    candidates=[]
    for action,turn in [('forward',0),('left',-1),('right',1)]:
        d=(direction+turn)%4;dx,dy=DIRECTIONS[d];start=(head[0]+dx,head[1]+dy)
        if not (0<=start[0]<size and 0<=start[1]<size) or start in blocked:
            continue
        pending=deque([(start,1)]);seen={start};distance=None
        while pending:
            point,n=pending.popleft()
            if point==food:
                distance=n;break
            for dx,dy in DIRECTIONS:
                q=(point[0]+dx,point[1]+dy)
                if (0<=q[0]<size and 0<=q[1]<size and q not in blocked and q not in seen):
                    seen.add(q);pending.append((q,n+1))
        if distance is not None:candidates.append((distance,len(candidates),action))
    return min(candidates)[2] if candidates else None


def pose(rng,size):
    direction=int(rng.integers(4));dx,dy=DIRECTIONS[direction]
    head=(int(rng.integers(1,size-1)),int(rng.integers(1,size-1)))
    neck=(head[0]-dx,head[1]-dy)
    if not (0<=neck[0]<size and 0<=neck[1]<size):return None
    snake=[head,neck]
    for _ in range(int(rng.integers(1,4))):
        options=[(snake[-1][0]+x,snake[-1][1]+y) for x,y in DIRECTIONS]
        options=[q for q in options if 0<=q[0]<size and 0<=q[1]<size and q not in snake]
        if not options:break
        snake.append(options[int(rng.integers(len(options)))])
    return snake,direction


def render_scene(snake,direction,food,size=6):
    game=ReadableGame(seed=0,size=size)
    game.snake=list(map(tuple,snake));game.direction=int(direction);game.food=tuple(food) if food is not None else None
    return game.render_observation()


def build_dataset(path, *, size=6, seed=20261003, train_poses=24, validation_poses=8, test_poses=8, foods=9):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed);used_poses=set();used_frames=set();arrays={};records={}
    for split,nposes in [('train',train_poses),('validation',validation_poses),('test',test_poses)]:
        images=[];labels=[];rows=[];pose_count=0;attempts=0
        while pose_count<nposes:
            attempts+=1
            if attempts>10000:raise ValueError('Cannot construct disjoint physical poses')
            value=pose(rng,size)
            if value is None:continue
            snake,direction=value;key=json.dumps([snake,direction])
            if key in used_poses:continue
            adjacent=[]
            for turn in [-1,0,1]:
                dx,dy=DIRECTIONS[(direction+turn)%4];q=(snake[0][0]+dx,snake[0][1]+dy)
                if 0<=q[0]<size and 0<=q[1]<size and q not in snake:adjacent.append(q)
            free=[(x,y) for y in range(size) for x in range(size) if (x,y) not in snake and (x,y) not in adjacent]
            rng.shuffle(free)
            selected=adjacent+free[:max(0,foods-len(adjacent))]
            prepared=[]
            for food in selected:
                action=teacher_action(snake,direction,food,size)
                if action is None:continue
                image=render_scene(snake,direction,food,size)
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                if digest in used_frames:raise ValueError('Cross-split image overlap')
                dx,dy=DIRECTIONS[(direction+{'left':-1,'forward':0,'right':1}[action])%4]
                eat_next=(snake[0][0]+dx,snake[0][1]+dy)==food
                prepared.append((image,ACTIONS.index(action),{'snake':snake,'direction':direction,
                    'food':food,'pose_key':key,'sha256':digest,'teacher':action,'adjacent_eat':eat_next}))
            if not prepared:continue
            used_poses.add(key);pose_count+=1
            for image,label,row in prepared:
                used_frames.add(row['sha256']);images.append(image);labels.append(label);rows.append(row)
        arrays[split+'_rgb']=np.stack(images)
        arrays[split+'_labels']=np.array(labels,np.int64)
        records[split]=rows
    with (path/'dataset.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    summary={'version':VERSION,'seed':seed,'size':size,'foods_per_pose':foods,
        'pose_counts':{'train':train_poses,'validation':validation_poses,'test':test_poses},
        'rows':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256((path/'dataset.npz').read_bytes()).hexdigest(),
        'class_counts':{k:np.bincount(arrays[k+'_labels'],minlength=3).tolist() for k in records},
        'cross_split_pose_overlap':False,'cross_split_image_overlap':False,
        'scope':'short physical scenes, not ordinary-game learning or final heldout protocol'}
    (path/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary
