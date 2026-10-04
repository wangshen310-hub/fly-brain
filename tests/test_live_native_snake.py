"""Visible game uses the same neural receipts and fixed-board rules as training."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from flysnake.lab.protocol import decode
from flysnake.research.live_server import LiveSession, handler
from flysnake.research.vision import ReadableGame


class FakePulseBrain:
    """Test double supplies spikes, never game actions or food coordinates."""
    def __init__(self):
        self.time = 0
        self.brain = SimpleNamespace(calibration={'baseline_hz': [0., 0., 0.],
            'scale_hz': [1., 1., 1.]}, last=None, reset_trial=lambda: None)

    def observe(self, rgb):
        counts = {'left': [1], 'forward': [0, 0], 'right': [1]}
        result = decode(counts, 300, self.brain.calibration)
        result.update(observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                      calibration=self.brain.calibration, neural_time_start=self.time,
                      neural_time_end=self.time+300)
        self.time += 300; self.brain.last = True
        return result

    def feedback(self, kind):
        self.brain.last = None
        return {'delivered_spikes': 3, 'kind': kind}


def test_live_observer_shows_exact_pixels_passed_to_spike_controller(tmp_path):
    from train_paired_native_snake import episode
    session = LiveSession(run=tmp_path/'run', parent=tmp_path/'parent', limit=4)
    result = episode(FakePulseBrain(), 44, 'uniform_food', 4, 4, on_step=session.observe)
    current = session.view()
    assert current['visual_input'].startswith('data:image/png;base64,')
    assert current['brain_output']['pixel_sha256'] == current['observation_sha256']
    assert current['brain_output']['action'] == 'forward'
    assert result['rows'][0]['receipt']['action'] == 'forward'
    assert result['rows'][0]['receipt']['counts']['left'] == [1]


def test_bad_displayed_neural_pixels_are_rejected(tmp_path):
    session = LiveSession(run=tmp_path/'run', parent=tmp_path/'parent')
    game = ReadableGame(seed=1, size=4)
    rgb = game.render_observation()
    session.observe('input', game, rgb, None, None, None)
    with pytest.raises(RuntimeError, match='differs'):
        session.observe('result', game, rgb, {'observation_sha256': 'wrong'}, None, None)


def test_browser_api_cannot_submit_a_move(tmp_path):
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    session = LiveSession(run=tmp_path/'run', parent=tmp_path/'parent')
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(session))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/state') as response:
            assert b'live-native-paired-snake-v1' in response.read()
        request = Request(f'http://127.0.0.1:{server.server_port}/api/move',
                          data=b'{"action":"left"}', method='POST')
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 404
        assert session.view()['brain_output'] is None
    finally:
        server.shutdown();server.server_close();thread.join()
