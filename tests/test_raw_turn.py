"""Raw BCI direction invariants; these are not learning evidence."""
import copy
import pytest
from flysnake.research.raw_turn import decode_raw_turn,RawTurnGame
from flysnake.research.pulse_balance import CELL_IDS


def test_stronger_turn_neuron_cannot_be_reversed_by_neutral_offsets():
    assert decode_raw_turn({'left':[5],'forward':[12,10],'right':[2]},300)['action']=='left'
    assert decode_raw_turn({'left':[2],'forward':[12,10],'right':[5]},300)['action']=='right'
    assert decode_raw_turn({'left':[10],'forward':[12,10],'right':[8]},300)['action']=='forward'
    assert decode_raw_turn({'left':[0],'forward':[1,0],'right':[0]},300)['action']=='forward'
    assert decode_raw_turn({'left':[0],'forward':[0,0],'right':[0]},300)['action'] is None
    with pytest.raises(ValueError):decode_raw_turn({'left':[-1],'forward':[1,0],'right':[0]},300)


def test_raw_game_validates_actual_readout_and_rejects_direction_replacement():
    calibration={'baseline_hz':[15,40,4],'scale_hz':[3,3,3]}
    game=RawTurnGame(seed=42,size=4,stage='uniform_food',
                    geometry={'board_cells':4,'tile_pixels':14,'board_origin':[130,16]},calibration=calibration)
    receipt=decode_raw_turn({'left':[5],'forward':[12,10],'right':[2]},300)
    receipt.update(episode_id=game.episode_id,step_id=game.step_id,selected_cell_ids=CELL_IDS,
                   observation_sha256=game.observation_hash(),calibration=calibration,
                   neural_time_start=100,neural_time_end=400)
    wrong=copy.deepcopy(receipt);wrong['action']='right'
    assert game.apply_receipt(wrong)['reason']=='non_neural_action'
    assert game.apply_receipt(receipt)['accepted']
