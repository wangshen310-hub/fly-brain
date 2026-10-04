"""Persistence fixtures are not evidence of biological or behavioral learning."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_node_learning import save_training, restore_training, novel_scene
from pilot_online_native_snake import NearbyFoodGame
from flysnake.lab.environment import cue_scene


class MemoryFixture:
    def __init__(self, seed):
        self.edges = np.array([2, 5, 7])
        self.base = np.array([1., -2., 3.], dtype=np.float32)
        self.group = np.arange(3)
        self.theta = np.zeros(3)
        self.mean_pre = np.zeros(3)
        self.reward_baseline = 0.
        self.tonic = 12.
        self.rng = np.random.default_rng(seed)
        self.brain = SimpleNamespace(calibration={'baseline_hz': [1., 2., 3.], 'scale_hz': [1., 1., 1.]})

    def set_parameters(self, theta):
        self.theta = np.array(theta).copy()

    def weight_hash(self):
        weight = (self.base * np.exp(self.theta)).astype(np.float32)
        return hashlib.sha256(weight.tobytes()).hexdigest()

    def save(self, path):
        np.savez_compressed(path, edges=self.edges, base=self.base, group=self.group,
                            theta=self.theta, calibration=json.dumps(self.brain.calibration))


def test_resume_preserves_weights_learning_statistics_and_next_neural_noise(tmp_path):
    original = MemoryFixture(123)
    original.set_parameters([.2, -.8, 1.])
    original.mean_pre[:] = [1.5, 4.2, 2.1]
    original.reward_baseline = -.2
    original.rng.normal(size=7)
    path = tmp_path / 'trial.npz'
    save_training(original, path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = MemoryFixture(999)
    restore_training(restored, path, expected_sha)
    assert original.weight_hash() == restored.weight_hash()
    np.testing.assert_array_equal(original.mean_pre, restored.mean_pre)
    assert original.reward_baseline == restored.reward_baseline
    np.testing.assert_array_equal(original.rng.normal(size=20), restored.rng.normal(size=20))


def test_corrupt_checkpoint_is_rejected_without_mutating_memory(tmp_path):
    original = MemoryFixture(123)
    path = tmp_path / 'trial.npz'
    save_training(original, path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b'corrupt')
    restored = MemoryFixture(999)
    before = restored.weight_hash()
    with pytest.raises(ValueError, match='file hash'):
        restore_training(restored, path, expected_sha)
    assert before == restored.weight_hash()


def test_novel_validation_pixels_do_not_repeat_any_training_frame():
    digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
    train = {digest(cue_scene(6200+i//2, 'left' if i%2 == 0 else 'right')) for i in range(320)}
    test = {digest(novel_scene(s, side)) for s in range(40700, 40764) for side in ('left', 'right')}
    assert train.isdisjoint(test)


def test_magnified_board_remains_fixed_when_head_moves():
    game = NearbyFoodGame(seed=1, size=4)
    game.snake = [(2, 2), (2, 3), (1, 3)]
    game.food = (1, 1)
    game.direction = 0
    before = game.render_observation()
    game.snake = [(3, 2), (2, 2), (2, 3)]
    game.direction = 1
    after = game.render_observation()
    np.testing.assert_array_equal(before[66:90, 136:160], after[66:90, 136:160])
    np.testing.assert_array_equal(before[18:42], after[18:42])
    assert tuple(before[80, 148]) == (180, 200, 255)
    assert before.shape == after.shape == (180, 320, 3)
