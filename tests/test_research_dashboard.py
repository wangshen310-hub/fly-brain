"""Dashboard records must not hide measured results or invent completion."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import serve_learning_research as dashboard


def fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    directory = tmp_path/'runtime/paired-native-snake-fixture'
    directory.mkdir(parents=True)
    data = {'protocol': 'native-two-hop-paired-reward-v1', 'status': 'running',
            'arm': 'learning', 'iterations': [{'generation': 1}], 'checkpoints': [{}],
            'before': {'uniform_food': {'food': 2, 'n': 6}},
            'settings': {'generations': 9}, 'progress': {'phase': 'training'},
            'brain': {'mutable_edges': 874654}}
    (directory/'status.json').write_text(json.dumps(data))
    return directory


def test_paired_protocol_exposes_food_counts_and_is_not_goal_completion(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch)
    snapshot = dashboard.snapshot()
    assert snapshot['goal_complete'] is False
    run = snapshot['runs'][0]
    assert run['food']['before_uniform_food'] == {'food': 2, 'n': 6}
    assert run['completed'] == 1 and run['memory_saved'] == 1
    assert run['progress'] == {'phase': 'training'}


def test_unknown_replay_path_is_rejected(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Unknown'):
        dashboard.replay('../../secret')


def test_corrupt_run_is_not_shown_as_zero_score(tmp_path, monkeypatch):
    directory = fixture_run(tmp_path, monkeypatch)
    (directory/'status.json').write_text('{corrupt')
    assert dashboard.snapshot()['runs'] == []
