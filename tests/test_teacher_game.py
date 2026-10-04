import hashlib
import pytest
pytest.importorskip('torch')
from flysnake.research.teacher_game import SpikingSnake,CELL_IDS,attach_game_receipt
from flysnake.research.teacher_spiking import VERSION,decode_spikes


PARAM='1'*64


def receipt(game,counts,start=0):
    return attach_game_receipt({'version':VERSION,**decode_spikes(counts),
        'selected_cell_ids':CELL_IDS,'input_sha256':game.observation_hash(),
        'start_tick':start,'end_tick':start+100},game,PARAM)


def test_host_never_overrides_real_spikes_and_self_collision_punishes():
    game=SpikingSnake(seed=0,size=6,parameter_sha256=PARAM)
    game.snake=[(2,2),(2,3),(1,3),(1,2),(1,1)];game.direction=0;game.food=(4,4)
    raw=receipt(game,{'left':[8],'right':[2],'forward':[0,0]})
    fake={**raw,'action':'forward'}
    assert not game.apply_receipt(fake)['accepted']
    event=game.apply_receipt(raw)
    assert event['accepted'] and event['death_reason']=='self' and event['scalar_reward']==-1


def test_no_output_penalizes_then_keeps_processing_neural_windows():
    game=SpikingSnake(seed=0,size=6,parameter_sha256=PARAM);before=list(game.snake)
    event=game.apply_receipt(receipt(game,{'left':[0],'right':[0],'forward':[0,0]}))
    assert event['scalar_reward']==-.1 and not game.terminal and game.step_id==1 and game.snake==before
    game.complete_feedback(event['event_id'])
    old=receipt(game,{'left':[5],'right':[5],'forward':[0,0]},start=0)
    assert not game.apply_receipt(old)['accepted']
    actual=receipt(game,{'left':[5],'right':[5],'forward':[0,0]},start=100)
    assert game.apply_receipt(actual)['accepted']


def test_food_reward_grows_body_and_camera_stays_fixed():
    game=SpikingSnake(seed=0,size=6,parameter_sha256=PARAM)
    game.snake=[(2,2),(2,3),(2,4)];game.direction=0;game.food=(2,1)
    geometry=game.observation_geometry()
    event=game.apply_receipt(receipt(game,{'left':[5],'right':[5],'forward':[0,0]}))
    assert event['scalar_reward']==1 and game.score==1 and len(game.snake)==4
    game.complete_feedback(event['event_id'])
    assert game.observation_geometry()==geometry
