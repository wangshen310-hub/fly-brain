"""CPU-only 7-cell/control-flow fixtures; never evidence of brain learning."""
import copy
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from scripts import train_teacher_spiking as trainer
from flysnake.research.teacher_afferent import PlasticAfferentBrain
from flysnake.research.teacher_spiking import Dynamics, SpikingConnectome, file_hash
from test_teacher_afferent import arrays
from test_teacher_training_safety import score, rows


def tiny_brain(plastic=False):
    cls = PlasticAfferentBrain if plastic else SpikingConnectome
    b = cls(arrays(), {}, device='cpu', config=Dynamics(window_ms=100))
    if plastic:
        with torch.no_grad():
            b.afferent_log_gain.copy_(torch.tensor([-.2, .3]))
    return b


@pytest.mark.parametrize('plastic', [False, True])
def test_warm_only_loss_backpropagates_through_first_window_voltage_and_queue(monkeypatch, plastic):
    b = tiny_brain(plastic)
    original = b.integrate_encoded
    calls = []

    def record(light, state=None, **kwargs):
        if calls:
            assert state is calls[0][1]
            assert light is calls[0][0]
        result = original(light, state, **kwargs)
        calls.append((light, result[2]))
        return result

    monkeypatch.setattr(b, 'integrate_encoded', record)
    result = trainer.training_objective(b, torch.tensor([[.1, .3, .8]]),
        torch.tensor([0, 1, 2]), trainer.CONTINUOUS_PROTOCOL)
    assert len(calls) == 2 and calls[-1][1]['ticks'] == 200
    warm_loss = result['window_teacher_losses'][1]
    first = calls[0][1]
    cross = torch.autograd.grad(warm_loss, (first['v'], first['release_queue']), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in cross)
    parameters = [p for p in b.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(warm_loss, parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
    assert torch.equal(result['teacher_loss'], torch.stack(result['window_teacher_losses']).mean())
    assert torch.equal(result['silence_loss'], torch.stack(result['window_silence_losses']).mean())


def test_detach_negative_control_severs_the_cross_window_gradient(monkeypatch):
    b = tiny_brain(True)
    original = b.integrate_encoded
    states = []

    def deliberately_detached(light, state=None, **kwargs):
        result = original(light, b.detach_state(state) if state is not None else None, **kwargs)
        states.append(result[2])
        return result

    monkeypatch.setattr(b, 'integrate_encoded', deliberately_detached)
    result = trainer.training_objective(b, torch.tensor([[.1, .3, .8]]),
        torch.tensor([0, 1, 2]), trainer.CONTINUOUS_PROTOCOL)
    gradients = torch.autograd.grad(result['window_teacher_losses'][1],
        (states[0]['v'], states[0]['release_queue']), allow_unused=True)
    assert gradients == (None, None)


def test_reset_negative_control_is_rejected_as_discontinuous_training(monkeypatch):
    b = tiny_brain()
    original = b.integrate_encoded
    monkeypatch.setattr(b, 'integrate_encoded', lambda light, state=None: original(light, None))
    with pytest.raises(ValueError, match='training clock mismatch'):
        trainer.training_objective(b, torch.tensor([[.1, .3, .8]]),
            torch.tensor([0, 1, 2]), trainer.CONTINUOUS_PROTOCOL)


@pytest.mark.parametrize('plastic', [False, True])
def test_legacy_loss_and_gradients_are_exactly_the_original_one_window_formula(plastic):
    a = tiny_brain(plastic)
    b = tiny_brain(plastic)
    light = torch.tensor([[.1, .3, .8]])
    labels = torch.tensor([0, 1, 2])
    _, motor, _ = a.integrate_encoded(light)
    old_ce = torch.nn.functional.cross_entropy(a.logits(motor), labels)
    old_silence = .05 * torch.nn.functional.softplus(2.-motor.sum(dim=1)).mean()
    new = trainer.training_objective(b, light, labels)  # default must remain legacy
    assert len(new['window_teacher_losses']) == 1
    assert torch.equal(old_ce, new['teacher_loss'])
    assert torch.equal(old_silence, new['silence_loss'])
    old_grad = torch.autograd.grad(old_ce+old_silence, [p for p in a.parameters() if p.requires_grad])
    new_grad = torch.autograd.grad(new['teacher_loss']+new['silence_loss'],
                                   [p for p in b.parameters() if p.requires_grad])
    assert all(torch.equal(x, y) for x, y in zip(old_grad, new_grad))


def report(first, second):
    windows = [copy.deepcopy(first), copy.deepcopy(second)]
    for i, m in enumerate(windows):
        m.update(start_tick=i*100, end_tick=(i+1)*100)
    return {**windows[0], 'continuous_two_window': {'windows': windows,
        'same_input_each_window': True, 'state_reset_between_windows': False,
        'gate': {'passed': True}}}  # caller's cached boolean must not be trusted


def test_cold_perfect_warm_bad_cannot_be_selected_even_with_no_incumbent():
    perfect = score([0, 1, 2], [0, 1, 2])
    collapsed = score([2, 2, 2], [0, 1, 2])
    candidate = report(perfect, collapsed)
    assert trainer.checkpoint_score(candidate, trainer.LEGACY_PROTOCOL, 100) == 1.
    assert trainer.checkpoint_score(candidate, trainer.CONTINUOUS_PROTOCOL, 100) is None


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -.1, 1.1, True])
def test_selection_rejects_invalid_or_nonfinite_metrics(bad):
    perfect = score([0, 1, 2], [0, 1, 2])
    candidate = report(perfect, perfect)
    candidate['continuous_two_window']['windows'][1]['recall'][0] = bad
    assert trainer.checkpoint_score(candidate, trainer.CONTINUOUS_PROTOCOL, 100) is None


def test_selection_rejects_missing_classes_inconsistent_score_and_missing_fields():
    perfect = score([0, 1, 2], [0, 1, 2])
    missing = score([1], [1])
    assert trainer.checkpoint_score(report(missing, missing), trainer.CONTINUOUS_PROTOCOL, 100) is None
    candidate = report(perfect, perfect)
    candidate['continuous_two_window']['windows'][1]['balanced_accuracy'] = .99
    assert trainer.checkpoint_score(candidate, trainer.CONTINUOUS_PROTOCOL, 100) is None
    assert trainer.checkpoint_score({}, trainer.CONTINUOUS_PROTOCOL, 100) is None
    assert trainer.checkpoint_score({'balanced_accuracy': float('nan')}, trainer.LEGACY_PROTOCOL, 100) is None


def test_eligible_checkpoint_ranking_prioritizes_worst_window_not_cold_peak():
    labels = [0]*10 + [1]*10 + [2]*10
    perfect = score(labels, labels)
    eight = score([0]*8+[1]*2 + [1]*8+[2]*2 + [2]*8+[0]*2, labels)
    nine = score([0]*9+[1] + [1]*9+[2] + [2]*9+[0], labels)
    peaked = trainer.checkpoint_score(report(perfect, eight), trainer.CONTINUOUS_PROTOCOL, 100)
    balanced = trainer.checkpoint_score(report(nine, nine), trainer.CONTINUOUS_PROTOCOL, 100)
    assert balanced > peaked
    assert balanced == (.9, .9, .9)


def test_exact_eighty_percent_boundary_is_not_rejected_by_float_roundoff():
    labels = [0]*20 + [1]*20 + [2]*20
    boundary = score([0]*15+[1]*5 + [1]*15+[2]*5 + [2]*18+[0]*2, labels)
    assert boundary['recall'] == [.75, .75, .9]
    assert boundary['balanced_accuracy'] == pytest.approx(.8, abs=1e-15)
    assert trainer.checkpoint_score(report(boundary, boundary), trainer.CONTINUOUS_PROTOCOL, 100) is not None
    below = score([0]*15+[1]*5 + [1]*15+[2]*5 + [2]*17+[0]*3, labels)
    assert trainer.checkpoint_score(report(boundary, below), trainer.CONTINUOUS_PROTOCOL, 100) is None


def test_protocol_duration_and_resume_identity_are_enforced():
    with pytest.raises(ValueError, match='100 ms'):
        trainer.protocol_details(trainer.CONTINUOUS_PROTOCOL, 20)
    with pytest.raises(ValueError, match='Unknown'):
        trainer.protocol_details('approximate-replay', 100)
    config = {'protocol': trainer.CONTINUOUS_PROTOCOL, 'window_ms': 100}
    extra = {'config': config.copy(), 'dataset_sha256': 'dataset', 'protocol_source_sha256': 'source'}
    trainer.validate_training_resume(extra, config, 'dataset', 'source')
    for changed in ({**config, 'protocol': trainer.LEGACY_PROTOCOL}, {**config, 'window_ms': 20}):
        with pytest.raises(ValueError, match='resume'):
            trainer.validate_training_resume(extra, changed, 'dataset', 'source')
    with pytest.raises(ValueError, match='resume'):
        trainer.validate_training_resume(extra, config, 'dataset', 'different-source')


def run_fixture(tmp_path, monkeypatch, *, include_test=False):
    """Run the actual orchestration for ONE update on an explicit tiny graph."""
    dataset = tmp_path / 'synthetic-dataset'
    dataset.mkdir()
    content = {}
    metadata = {}
    for split in ('train', 'validation', *(['test'] if include_test else [])):
        # Identical RGB and conflicting three-class labels ensure this fixture
        # cannot pass the real classification gate. It is a control-flow test.
        content[split+'_rgb'] = np.zeros((3, 180, 320, 3), dtype=np.uint8)
        content[split+'_labels'] = np.array([0, 1, 2], dtype=np.int64)
        metadata[split] = rows([0, 1, 2])
    np.savez(dataset / 'dataset.npz', **content)
    (dataset / 'manifest.json').write_text(json.dumps({'dataset_sha256': file_hash(dataset/'dataset.npz'),
                                                    'rows': metadata}))
    b = tiny_brain()
    monkeypatch.setattr(trainer.SpikingConnectome, 'load_graph', lambda *args, **kwargs: b)
    original_threads = torch.set_num_threads
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: original_threads(2))
    run = tmp_path / 'run'
    argv = ['train_teacher_spiking.py', '--graph', 'explicit-seven-cell-fixture',
        '--dataset', str(dataset), '--run', str(run), '--device', 'cpu', '--steps', '1',
        '--eval-every', '1', '--batch', '3', '--protocol', trainer.CONTINUOUS_PROTOCOL]
    monkeypatch.setattr(sys, 'argv', argv)
    return run, argv, b


def test_no_eligible_run_saves_last_memory_and_never_opens_absent_test_arrays(tmp_path, monkeypatch):
    run, argv, b = run_fixture(tmp_path, monkeypatch)
    trainer.main()
    summary = json.loads((run/'summary.json').read_text())
    assert summary['selection_state'] == 'no_eligible_checkpoint'
    assert summary['best_checkpoint'] is None and not summary['heldout_test_evaluated']
    assert not list(run.glob('best-*.pt')) and not list(run.glob('test-*.json'))
    checkpoint = json.loads((run/'committed.json').read_text())['checkpoint']
    payload = torch.load(checkpoint['path'], weights_only=True)
    assert payload['parameter_sha256'] == b.parameter_hash()
    extra = payload['extra']
    assert extra['step'] == 1 and extra['best_checkpoint'] is None and extra['selected_step'] is None
    assert extra['optimizer']['state'] and json.loads(extra['rng_json'])['bit_generator'] == 'PCG64'
    assert extra['config']['protocol'] == trainer.CONTINUOUS_PROTOCOL
    assert len(json.loads((run/'status.json').read_text())['progress'][0]['window_teacher_losses']) == 2


def test_completed_run_rejects_cross_protocol_resume_before_loading_graph(tmp_path, monkeypatch):
    run, argv, _ = run_fixture(tmp_path, monkeypatch)
    trainer.main()
    monkeypatch.setattr(trainer.SpikingConnectome, 'load_graph',
                        lambda *args, **kwargs: pytest.fail('graph loaded before resume identity check'))
    changed = list(argv)
    changed[-1] = trainer.LEGACY_PROTOCOL
    monkeypatch.setattr(sys, 'argv', changed+['--resume'])
    with pytest.raises(ValueError, match='resume protocol/config/source'):
        trainer.main()


def test_no_eligible_committed_memory_restores_after_simulated_interruption(tmp_path, monkeypatch):
    run, argv, first = run_fixture(tmp_path, monkeypatch)
    trainer.main()
    parameter_hash = first.parameter_hash()
    committed = json.loads((run/'committed.json').read_text())
    status = json.loads((run/'status.json').read_text())
    status['status'] = 'failed'  # simulate process loss after commit, before completion
    (run/'status.json').write_text(json.dumps(status))
    restored = tiny_brain()
    monkeypatch.setattr(trainer.SpikingConnectome, 'load_graph', lambda *args, **kwargs: restored)
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    trainer.main()
    summary = json.loads((run/'summary.json').read_text())
    assert restored.parameter_hash() == parameter_hash
    assert summary['last_memory_checkpoint'] == committed['checkpoint']
    assert summary['selection_state'] == 'no_eligible_checkpoint'
    assert not summary['heldout_test_evaluated']
    assert json.loads((run/'committed.json').read_text()) == committed


def test_qualifying_initial_checkpoint_is_reported_as_unchanged_not_learned(tmp_path, monkeypatch):
    run, _, _ = run_fixture(tmp_path, monkeypatch, include_test=True)
    perfect = score([0, 1, 2], [0, 1, 2])
    # Gate-qualified result is SYNTHETIC solely to exercise the initial branch.
    monkeypatch.setattr(trainer, 'evaluate', lambda *args, **kwargs: report(perfect, perfect))
    trainer.main()
    summary = json.loads((run/'summary.json').read_text())
    assert summary['selected_training_step'] == 0
    assert summary['selected_checkpoint_origin'] == 'unchanged_initial'
    assert summary['eligible_checkpoint_found'] and summary['heldout_test_evaluated']
    assert not summary['full_brain_learning_confirmed']
    assert not summary['qualified_updated_checkpoint_selected']
    assert summary['interpretation'].startswith('Unchanged initial checkpoint')


def test_qualified_resume_restores_tuple_selection_optimizer_and_rng(tmp_path, monkeypatch):
    run, argv, first = run_fixture(tmp_path, monkeypatch, include_test=True)
    perfect = score([0, 1, 2], [0, 1, 2])
    monkeypatch.setattr(trainer, 'evaluate', lambda *args, **kwargs: report(perfect, perfect))
    # Interrupt only AFTER committed memory, before any selected final replay.
    monkeypatch.setattr(first, 'restore_checkpoint',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('simulated interruption')))
    with pytest.raises(RuntimeError, match='simulated interruption'):
        trainer.main()
    point = json.loads((run/'committed.json').read_text())['checkpoint']
    extra = torch.load(point['path'], weights_only=True)['extra']
    assert extra['best'] == (1., 1., 1.) and extra['selected_step'] == 0
    fresh = tiny_brain()
    monkeypatch.setattr(trainer.SpikingConnectome, 'load_graph', lambda *args, **kwargs: fresh)
    loaded = []
    original_load = torch.optim.Adam.load_state_dict

    def record_optimizer(optimizer, state):
        loaded.append(copy.deepcopy(state))
        return original_load(optimizer, state)

    monkeypatch.setattr(torch.optim.Adam, 'load_state_dict', record_optimizer)
    generators = []
    original_rng = np.random.default_rng

    def record_rng(*args, **kwargs):
        rng = original_rng(*args, **kwargs)
        generators.append(rng)
        return rng

    monkeypatch.setattr(np.random, 'default_rng', record_rng)
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    trainer.main()
    assert loaded[0]['param_groups'] == extra['optimizer']['param_groups']
    for index, fields in extra['optimizer']['state'].items():
        assert all(torch.equal(loaded[0]['state'][index][name], value) for name, value in fields.items())
    assert generators[0].bit_generator.state == json.loads(extra['rng_json'])
    summary = json.loads((run/'summary.json').read_text())
    assert summary['selected_training_step'] == 0
    assert summary['selected_checkpoint_origin'] == 'unchanged_initial'
    assert not summary['qualified_updated_checkpoint_selected']
