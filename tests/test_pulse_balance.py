"""User pulse-balance rule and fresh-receipt gate, not learning evidence."""
import copy
import hashlib
import numpy as np
from flysnake.research.pulse_balance import decode_balanced,PulseBalancedGame,CELL_IDS
from flysnake.research.projection import coverage_geometry
from flysnake.lab.protocol import decode

CALIBRATION={'baseline_hz':[15,40,4],'scale_hz':[3,3,3]}


def test_equal_turn_pulses_override_offset_artifact():
    counts={'left':[2],'forward':[12,11],'right':[2]}
    assert decode(counts,300,CALIBRATION)['action']=='right'
    assert decode_balanced(counts,300,CALIBRATION)['action']=='forward'


def test_small_pulse_difference_goes_forward_and_total_silence_stays_invalid():
    assert decode_balanced({'left':[10],'forward':[10,10],'right':[9]},300,CALIBRATION)['action']=='forward'
    assert decode_balanced({'left':[0],'forward':[10,10],'right':[0]},300,CALIBRATION)['action']=='forward'
    assert decode_balanced({'left':[0],'forward':[0,0],'right':[0]},300,CALIBRATION)['action'] is None
    neutral={'baseline_hz':[0,0,0],'scale_hz':[1,1,1]}
    assert decode_balanced({'left':[0],'forward':[1,0],'right':[0]},300,neutral)['action']=='forward'


def test_game_checks_correct_raw_balance_action_and_rejects_old_artifact():
    # Geometry is only a fixture; actual native rendering is tested separately.
    geometry={'board_cells':4,'tile_pixels':14,'board_origin':[130,16]}
    game=PulseBalancedGame(seed=123,size=4,stage='uniform_food',geometry=geometry,calibration=CALIBRATION)
    counts={'left':[2],'forward':[12,11],'right':[2]}
    receipt=decode_balanced(counts,300,CALIBRATION)
    receipt.update(episode_id=game.episode_id,step_id=game.step_id,selected_cell_ids=CELL_IDS,
                   observation_sha256=game.observation_hash(),calibration=CALIBRATION,
                   neural_time_start=100,neural_time_end=400)
    bad=copy.deepcopy(receipt);bad['action']='right'
    assert game.apply_receipt(bad)['reason']=='non_neural_action'
    assert game.apply_receipt(receipt)['accepted']
