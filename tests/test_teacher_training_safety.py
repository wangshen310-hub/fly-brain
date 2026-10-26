"""Small CPU correctness fixtures, not full-brain learning evidence."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from scripts.train_teacher_spiking import (
    checked_optimizer_step, compact, evaluate, metrics, two_window_gate,
)
from flysnake.research.teacher_spiking import Dynamics, SpikingConnectome
from flysnake.research.teacher_afferent import PlasticAfferentBrain
from test_teacher_afferent import arrays as afferent_arrays
from test_teacher_spiking import fixture


COUNTS = np.array([[8, 2, 0, 0], [5, 5, 0, 0], [2, 8, 0, 0]])


def rows(labels):
    return [{'adjacent_eat': False, 'teacher': ('left', 'forward', 'right')[int(k)]}
            for k in labels]


def score(predictions, labels):
    motor = np.array([COUNTS[p] if p >= 0 else [0, 0, 0, 0] for p in predictions])
    return metrics(motor, np.array(labels), rows(labels))


def test_absent_classes_are_unassessed_not_perfect_balanced_accuracy():
    result = score([1], [1])
    assert result['accuracy'] == 1.
    assert result['balanced_accuracy'] is None
    assert result['recall'] == [None, 1., None]
    assert result['class_counts'] == [0, 1, 0]
    assert not result['class_coverage_complete']
    assert not two_window_gate([result, result], 100)['passed']


@pytest.mark.parametrize('motor,labels,nrows', [
    (np.empty((0, 4)), np.array([], dtype=int), 0),
    (COUNTS, np.array([0., 1., 2.]), 3),
    (COUNTS, np.array([0, 1, 3]), 3),
    (COUNTS, np.array([0, 1, 2]), 2),
    (COUNTS[:, :3], np.array([0, 1, 2]), 3),
    (np.array([[1.5, 1, 0, 0]]), np.array([0]), 1),
    (np.array([[np.nan, 1, 0, 0]]), np.array([0]), 1),
    (np.array([[np.inf, 1, 0, 0]]), np.array([0]), 1),
    (np.array([[-1, 1, 0, 0]]), np.array([0]), 1),
])
def test_metric_input_validation_never_truncates_or_rounds(motor, labels, nrows):
    with pytest.raises(ValueError):
        metrics(motor, labels, rows([0] * nrows))


def test_decoder_margin_and_silence_are_not_overridden():
    # At margin .2, exactly one pulse difference out of five is still forward.
    motor = np.array([[5, 4, 0, 0], [5, 4, 0, 0], [0, 0, 0, 0]])
    labels = np.array([0, 1, 2])
    usual = metrics(motor, labels, rows(labels), margin=.2)
    narrower = metrics(motor, labels, rows(labels), margin=.1)
    assert usual['predicted_actions'] == [0, 2, 0]
    assert narrower['predicted_actions'] == [2, 0, 0]
    assert usual['silent'] == narrower['silent'] == 1
    assert usual['recall'] == [0., 1., 0.]
    assert usual['balanced_accuracy'] == pytest.approx(1/3)


def test_perfect_cold_window_cannot_hide_warm_collapse():
    cold = score([0, 1, 2], [0, 1, 2])
    warm = score([2, 2, 2], [0, 1, 2])
    assert cold['balanced_accuracy'] == 1.
    assert warm['balanced_accuracy'] == pytest.approx(1/3)
    assert not two_window_gate([cold, warm], 100)['passed']


def test_every_class_recall_must_pass_even_when_both_bas_are_above_threshold():
    labels = [0]*4 + [1]*4 + [2]*4
    weak = score([0, 0, 1, 1] + [1]*4 + [2]*4, labels)
    boundary = score([0, 0, 0, 1] + [1, 1, 1, 2] + [2]*4, labels)
    assert weak['balanced_accuracy'] > .8 and weak['recall'][0] == .5
    assert not two_window_gate([weak, boundary], 100)['passed']
    gate = two_window_gate([boundary, boundary], 100)
    assert gate['passed'] and not gate['full_brain_learning_confirmed']
    assert not two_window_gate([boundary], 100)['passed']
    assert not two_window_gate([boundary, boundary], 20)['passed']


class ContinuationSpy:
    """Deliberately synthetic control-flow fixture, never a neural policy."""
    config = SimpleNamespace(window_ms=100, margin=.2)

    def __init__(self):
        self.previous = None
        self.calls = []

    def integrate_encoded(self, light, state, *, visual_enabled):
        assert not torch.is_grad_enabled()
        if state is None:
            self.light = light
            tick = 100
        else:
            assert state is self.previous  # not an incomplete reconstructed state
            assert light is self.light
            assert torch.equal(state['release_queue'], torch.tensor([7., 9.]))
            tick = 200
        output = {'ticks': tick, 'release_queue': torch.tensor([7., 9.])}
        self.calls.append((state, output, visual_enabled))
        self.previous = output
        indices = light[0].to(torch.int64).numpy() % 3
        motor = COUNTS[indices] if tick == 100 else np.tile(COUNTS[2], (len(indices), 1))
        return None, torch.tensor(motor, dtype=torch.float32), output

    def validate_state(self, state):
        assert state is self.previous


def test_evaluator_continues_exact_state_and_queue_in_each_batch():
    b = ContinuationSpy()
    labels = np.arange(9) % 3
    light = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    result = evaluate(b, light, labels, rows(labels), batch=4, visual=False)
    report = result['continuous_two_window']
    assert len(b.calls) == 6
    assert all(b.calls[i][0] is None for i in (0, 2, 4))
    assert all(not call[2] for call in b.calls)
    assert result['balanced_accuracy'] == 1.
    assert report['windows'][1]['balanced_accuracy'] == pytest.approx(1/3)
    assert not report['gate']['passed']
    assert [m['end_tick'] for m in report['windows']] == [100, 200]
    assert report['windows'][0]['class_counts'] == [3, 3, 3]
    assert report['windows'][1]['predicted_actions'] == [0, 0, 9]
    assert torch.equal(light, torch.arange(9).reshape(1, 9))
    stripped = compact(result)
    assert 'motor_counts' not in stripped
    assert all('receipts' not in m for m in stripped['continuous_two_window']['windows'])


@pytest.mark.parametrize('plastic', [False, True])
def test_real_tiny_neural_fixture_two_windows_equal_uninterrupted_200ms(monkeypatch, plastic):
    cls = PlasticAfferentBrain if plastic else SpikingConnectome
    b = cls(afferent_arrays() if plastic else fixture(), {}, device='cpu', config=Dynamics(window_ms=100))
    if plastic:
        with torch.no_grad():
            b.afferent_log_gain.copy_(torch.tensor([-.2, .3]))
    light = torch.tensor([[.1, .3, .8]])
    original = b.integrate_encoded
    with torch.no_grad():
        whole, whole_motor, final = original(light, steps=200)
    observed = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(b, 'integrate_encoded', record)
    before = b.parameter_hash()
    evaluate(b, light, np.array([0, 1, 2]), rows([0, 1, 2]), batch=3)
    assert torch.equal(observed[0][0] + observed[1][0], whole)
    assert torch.equal(observed[0][1] + observed[1][1], whole_motor)
    for name, expected in final.items():
        actual = observed[1][2][name]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), name
        else:
            assert actual == expected, name
    assert b.parameter_hash() == before
    assert all(p.grad is None for p in b.parameters())


def test_evaluator_rejects_reset_or_discontinuous_clock(monkeypatch):
    b = ContinuationSpy()
    original = b.integrate_encoded

    def reset_clock(*args, **kwargs):
        counts, motor, state = original(*args, **kwargs)
        state['ticks'] = 100
        return counts, motor, state

    monkeypatch.setattr(b, 'integrate_encoded', reset_clock)
    with pytest.raises(ValueError, match='clock mismatch'):
        evaluate(b, torch.arange(3, dtype=torch.float32).reshape(1, 3),
                 np.array([0, 1, 2]), rows([0, 1, 2]))


@pytest.mark.parametrize('gradient', [float('inf'), float('nan'), 1e20])
def test_later_bad_gradient_cannot_mutate_parameter_or_adam_memory(gradient):
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.Adam([p], lr=.01)
    p.grad = torch.tensor([.25, .25])
    assert checked_optimizer_step(optimizer, iter([p])) == pytest.approx(2**.5*.25)
    before = p.detach().clone()
    memory = copy.deepcopy(optimizer.state_dict())
    p.grad = torch.tensor([gradient, gradient])
    with pytest.raises(RuntimeError, match='non-finite'):
        checked_optimizer_step(optimizer, [p])
    assert torch.equal(p, before)
    after = optimizer.state_dict()
    assert after['param_groups'] == memory['param_groups']
    for key, value in memory['state'][0].items():
        assert torch.equal(after['state'][0][key], value), key


def test_no_gradient_update_is_rejected():
    p = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([p])
    with pytest.raises(ValueError, match='requires parameter gradients'):
        checked_optimizer_step(optimizer, [p])
