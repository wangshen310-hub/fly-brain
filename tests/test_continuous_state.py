"""Interrupted ordinary progress/RNG/state tests on explicit small fixtures.

These validate the snapshot transaction, never native physiology or learning.
"""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from flysnake.research.continuous_state import state_arrays,save_window,restore_window,game_state
from flysnake.research.paired import digest
from flysnake.research.pulse_balance import PulseBalancedGame


class BrainFixture:
    identity={'fixture':True,'purpose':'boundary persistence only'}
    motor_noise=np.zeros(2)
    def __init__(self):
        b=SimpleNamespace(fields=['v','queue','counts'],dt=.1,cursor=40,total_spikes=100,
                          v=np.array([-49.,-51.],np.float32),queue=np.array([[0,1],[1,0]],np.int32),
                          counts=np.array([2,3],np.int32),tonic=np.array([4.,5.],np.float32))
        self.brain=SimpleNamespace(b=b,retinal_light=np.array([.1,.4]),r8_light=np.array([.3,.2]),
                                   neural_ms=12300,last=None)
        self.coefficients=np.array([.3,-.2,.1,.2]);self.concentration=np.array([.1,.2])
    def set_parameters(self,theta):self.coefficients=np.asarray(theta).copy()
    def weight_hash(self):return digest(np.array([1.,-1.,2.,-2.])*np.exp(self.coefficients))


def game_fixture():
    return PulseBalancedGame(seed=4242,size=4,stage='uniform_food',
        geometry={'board_cells':4,'tile_pixels':14,'board_origin':[130,16]},
        calibration={'baseline_hz':[1,1,1],'scale_hz':[1,1,1]})


def test_committed_window_restores_dynamic_state_progress_and_future_food_rng(tmp_path):
    brain=BrainFixture();game=game_fixture();game.new_episode();game._last_neural_end=12300
    report={'sources':{'source':'fixture'},'settings':{'seed':4242},'totals':{'food':2},
        'windows_completed':4,'parent_checkpoint':{'cycles_completed':3,'file':'parent-fixture'},'checkpoints':[]}
    original_arrays=state_arrays(brain);original_game=game_state(game)
    save_window(tmp_path,report,brain,game)
    # Simulate losing uncommitted progress after a window: weights, hidden
    # states, visual adaptation, food RNG and reported counters all advanced.
    brain.brain.b.v[:]=0;brain.brain.b.queue[:]=9;brain.coefficients[:]=0
    brain.brain.retinal_light[:]=1;brain.concentration[:]=1
    game.new_episode();report['windows_completed']=5;report['totals']['food']=99
    restore_window(tmp_path,report,brain,game)
    assert all(np.array_equal(state_arrays(brain)[key],value) for key,value in original_arrays.items())
    assert game_state(game)==original_game
    assert report['windows_completed']==4 and report['totals']=={'food':2}
    assert brain.brain.neural_ms==12300 and brain.brain.b.cursor==40
    fresh=game_fixture();from flysnake.research.continuous_state import restore_game
    restore_game(fresh,original_game)
    for _ in range(5):assert fresh.new_episode()==game.new_episode()
    report['sources']['source']='changed'
    with pytest.raises(ValueError,match='source'):restore_window(tmp_path,report,brain,game)


def test_unfinished_feedback_cannot_be_committed(tmp_path):
    game=game_fixture();game._pending_feedback={'event_id':'pending fixture'}
    with pytest.raises(ValueError,match='Feedback'):game_state(game)
