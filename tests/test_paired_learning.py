"""Optimizer/persistence contracts; these fixtures do not prove Snake learning."""
from dataclasses import replace
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest
from flysnake.research.paired import (PairedSearch, SearchConfig, digest, writer_lock,
                                     save_checkpoint, restore_checkpoint)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_paired_native_snake import CurriculumGame, episode


class BankFixture:
    def __init__(self):
        self.base = np.array([.5, -2., .3, -1.], dtype=np.float32)
        self.group = np.arange(4)
        self.theta = np.zeros(4)
        self.identity = {'base': digest(self.base), 'group': digest(self.group), 'calibration': [1, 2]}

    def set_parameters(self, theta):
        self.theta = np.array(theta)

    def weight_hash(self):
        return digest((self.base*np.exp(self.theta[self.group])).astype(self.base.dtype))


def test_reward_only_search_can_improve_an_unknown_smooth_objective():
    s = PairedSearch(19, SearchConfig(dimensions=4))
    optimum = np.array([.8, -.7, .4, -1.])
    reward = lambda x: -float(np.sum((x-optimum)**2))
    before = reward(s.theta)
    for _ in range(40):
        eps = s.directions(16)
        scores = [[reward(s.candidate(e, sign)) for sign in (1, -1)] for e in eps]
        s.update(eps, scores)
    assert reward(s.theta) > before + 1.7


def test_frozen_arm_preserves_parameters_and_identical_pair_returns_give_no_update():
    s = PairedSearch(18, SearchConfig(dimensions=4))
    eps = s.directions(4)
    s.update(eps, [[1, -1]]*4, 'frozen')
    np.testing.assert_array_equal(s.theta, 0)
    s.update(eps, [[-1, -1]]*4)
    np.testing.assert_array_equal(s.theta, 0)
    with pytest.raises(ValueError):
        s.update(eps, [[np.nan, 0]]*4)


def test_resume_replays_next_candidates_and_optimizer_update(tmp_path):
    a = PairedSearch(18, SearchConfig(dimensions=4)); b = BankFixture()
    eps = a.directions(4)
    a.update(eps, [[1, -1], [0, 1], [2, 0], [1, 1]])
    b.set_parameters(a.theta)
    path = tmp_path/'memory.npz'
    sha = save_checkpoint(path, b, a, {'source': 'test'})
    restored = PairedSearch(99, a.config); bank = BankFixture()
    restore_checkpoint(path, sha, bank, restored, {'source': 'test'})
    assert b.weight_hash() == bank.weight_hash()
    e1, e2 = a.directions(4), restored.directions(4)
    np.testing.assert_array_equal(e1, e2)
    for x in (a, restored):
        x.update(e1, [[0, 1], [1, 0], [2, 0], [1, 1]], 'shuffled')
    np.testing.assert_array_equal(a.theta, restored.theta)
    np.testing.assert_array_equal(a.v, restored.v)


def test_bad_sources_and_corrupt_checkpoint_leave_memory_unchanged(tmp_path):
    s = PairedSearch(2, SearchConfig(dimensions=4)); b = BankFixture()
    path = tmp_path/'memory.npz'; sha = save_checkpoint(path, b, s, {'source': 'v1'})
    before = b.weight_hash()
    with pytest.raises(ValueError, match='Incompatible'):
        restore_checkpoint(path, sha, b, s, {'source': 'v2'})
    assert b.weight_hash() == before
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='file hash'):
        restore_checkpoint(path, sha, b, s, {'source': 'v1'})
    assert b.weight_hash() == before


def test_checkpoint_refuses_uncommitted_candidate(tmp_path):
    s = PairedSearch(2, SearchConfig(dimensions=4)); b = BankFixture()
    b.set_parameters([.2, 0, 0, 0])
    with pytest.raises(ValueError, match='exploratory'):
        save_checkpoint(tmp_path/'memory.npz', b, s, {})


def test_os_writer_lock_excludes_duplicate_and_releases(tmp_path):
    with writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            with writer_lock(tmp_path):
                pass
    with writer_lock(tmp_path):
        pass


def test_same_game_outcomes_have_identical_return_despite_different_DAN_amplitudes():
    from types import SimpleNamespace
    from flysnake.lab.protocol import decode
    class ReceiptFixture:
        def __init__(self, amplitude):
            self.amplitude = amplitude
            self.time = 0
            self.brain = SimpleNamespace(calibration={'baseline_hz': [0., 0., 0.],
                'scale_hz': [1., 1., 1.]}, last=None, reset_trial=lambda: None)
        def observe(self, rgb):
            result = decode({'left': [1], 'forward': [0, 0], 'right': [1]},
                            300, self.brain.calibration)
            result.update(observation_sha256=digest(rgb), calibration=self.brain.calibration,
                          neural_time_start=self.time, neural_time_end=self.time+300)
            self.time += 300; self.brain.last = True
            return result
        def feedback(self, kind):
            self.brain.last = None
            return {'delivered_spikes': 1, 'modulation': self.amplitude*(1 if kind == 'reward' else -1)}
    a = episode(ReceiptFixture(.01), 19, 'fixed_pose', 4, 4)
    b = episode(ReceiptFixture(.99), 19, 'fixed_pose', 4, 4)
    assert a['food'] == b['food'] and a['reason'] == b['reason']
    assert a['return'] == b['return'] == a['food']-1


@pytest.mark.parametrize('size', (4, 6, 8, 12))
def test_curriculum_changes_initial_conditions_without_invalid_body_or_food(size):
    for stage in ('fixed_pose', 'nearby_food', 'uniform_food'):
        for seed in range(12):
            g = CurriculumGame(seed=seed, stage=stage, size=size)
            assert len(set(g.snake)) == 3
            assert all(g._inside(p) for p in g.snake)
            assert g.food not in g.snake and g._inside(g.food)
            if stage != 'uniform_food':
                assert sum(abs(a-b) for a, b in zip(g.food, g.snake[0])) == 1
