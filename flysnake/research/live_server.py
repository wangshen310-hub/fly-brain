"""Persistent, browser-independent Snake driven by native MaleCNS spikes.

Every displayed move is an accepted lab receipt. Paired reward search changes
only existing synaptic weights at complete generation boundaries. The browser
has read access to the game; it never supplies an action or a target direction.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import (VERSION as BRAIN_VERSION, SearchConfig, PairedSearch,
    NativeSearchBrain, writer_lock, save_checkpoint, restore_checkpoint)
from train_paired_native_snake import episode

LIVE_VERSION = 'live-native-paired-snake-v1'
DEFAULT_RUN = ROOT/'runtime/live-paired-snake-v1'
DEFAULT_PARENT = ROOT/'runtime/paired-native-snake-event-v2'


def png_data(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def sources():
    paths = [Path(__file__), ROOT/'scripts/train_paired_native_snake.py',
             ROOT/'flysnake/research/paired.py', ROOT/'flysnake/research/vision.py',
             ROOT/'flysnake/research/operating.py', ROOT/'flysnake/lab/brain.py',
             ROOT/'flysnake/lab/protocol.py', ROOT/'flysnake/lab/environment.py']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


class LiveSession:
    def __init__(self, run=DEFAULT_RUN, parent=DEFAULT_PARENT, seed=20260924,
                 batch_size=3, pairs=2, limit=16, size=4):
        if pairs < 2 or batch_size < 1 or limit < 1 or size not in (4, 6, 8, 12):
            raise ValueError('Invalid game or paired-learning configuration')
        self.run = Path(run).resolve(); self.parent = Path(parent).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.guard = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.public = {'protocol': LIVE_VERSION, 'phase': 'loading', 'message': '加载完整果蝇连接组',
                       'active': False, 'visual_input': None, 'board_image': None,
                       'brain_output': None, 'last_feedback': None,
                       'memory_saved': False, 'size': size, 'stage': 'uniform_food',
                       'batch_size': batch_size, 'pairs': pairs, 'limit': limit}
        self.settings = {'seed': seed, 'batch_size': batch_size, 'pairs': pairs,
                         'limit': limit, 'size': size}
        self.source = sources()

    def view(self):
        with self.guard:
            value = dict(self.public)
        value['active'] = bool(self.thread and self.thread.is_alive())
        return value

    def publish(self, **fields):
        with self.guard:
            self.public.update(fields)

    def observe(self, phase, game, rgb, receipt, event, feedback):
        if phase == 'input':
            self.publish(phase='observing', message='完整连接组正在计算下一次动作',
                         visual_input=png_data(rgb), board_image=png_data(rgb),
                         observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                         game={'score': game.score, 'step': game.step_id, 'size': game.size,
                               'head': list(game.snake[0]), 'food': list(game.food) if game.food else None,
                               'death_reason': game.death_reason})
            return
        if receipt['observation_sha256'] != hashlib.sha256(rgb.tobytes()).hexdigest():
            raise RuntimeError('Displayed neural input differs from actual receipt')
        kind = 'no_output' if receipt['action'] is None else (event or {}).get('kind', 'none')
        self.publish(phase='moving', message='动作已由真实神经脉冲决定',
                     board_image=png_data(game.render_observation()),
                     brain_output={'action': receipt['action'], 'counts': receipt['counts'],
                                   'scores': receipt['scores'], 'reason': receipt['reason'],
                                   'pixel_sha256': receipt['observation_sha256'],
                                   'neural_time_end': receipt['neural_time_end']},
                     last_feedback={'kind': kind, 'delivered_spikes':
                                    None if feedback is None else feedback['delivered_spikes']},
                     game={'score': game.score, 'step': game.step_id, 'size': game.size,
                           'head': list(game.snake[0]), 'food': list(game.food) if game.food else None,
                           'death_reason': game.death_reason})

    def _state_path(self):
        return self.run/'status.json'

    def _save(self, report, brain, search, provenance):
        name = f'memory-{search.updates:06d}-{time.time_ns()}.npz'
        sha = save_checkpoint(self.run/name, brain, search, provenance)
        report['checkpoints'].append({'file': name, 'sha256': sha,
                                      'generation': search.updates,
                                      'memory_sha256': brain.weight_hash()})
        atomic_json(self._state_path(), report)
        self.publish(generation=search.updates, memory_sha256=brain.weight_hash(),
                     memory_saved=True, last_saved_at=time.time(),
                     saved_checkpoints=len(report['checkpoints']))

    def _load(self, brain, search):
        path = self._state_path()
        provenance = {'protocol': LIVE_VERSION, 'settings': self.settings,
                      'source': self.source, 'brain_protocol': BRAIN_VERSION}
        if path.is_file():
            report = json.loads(path.read_text(encoding='utf-8'))
            if (report.get('protocol') != LIVE_VERSION or report.get('settings') != self.settings
                    or report.get('source') != self.source or not report.get('checkpoints')):
                raise ValueError('Existing live run has incompatible source, settings or checkpoint')
            c = report['checkpoints'][-1]
            restore_checkpoint(self.run/c['file'], c['sha256'], brain, search, provenance)
            if search.updates != report['generation']:
                raise ValueError('Live generation and checkpoint disagree')
        else:
            original_path = self.parent/'status.json'
            original = json.loads(original_path.read_text(encoding='utf-8'))
            if original.get('status') != 'complete' or not original.get('checkpoints'):
                raise ValueError('Completed paired-learning memory is required')
            c = original['checkpoints'][-1]
            restore_checkpoint(self.parent/c['file'], c['sha256'], brain, search,
                               {'settings': original['settings'], 'sources': original['sources']})
            report = {'protocol': LIVE_VERSION, 'status': 'running', 'settings': self.settings,
                      'source': self.source, 'generation': search.updates,
                      'source_memory': {'directory': str(self.parent), 'checkpoint': c},
                      'brain': brain.identity, 'checkpoints': [], 'recent_generations': [],
                      'episodes': 0, 'food': 0, 'deaths': 0, 'no_output': 0,
                      'evaluation': {'status': 'unproven', 'reference':
                          {'source': 'completed six-generation development run',
                           'ordinary_food_before': original['before']['uniform_food']['food'],
                           'ordinary_food_after': original['after']['uniform_food']['food'],
                           'episodes_per_assessment': original['after']['uniform_food']['n']}}}
            self._save(report, brain, search, provenance)
        self.publish(phase='ready', generation=search.updates,
                     saved_checkpoints=len(report['checkpoints']),
                     memory_sha256=brain.weight_hash(), memory_saved=True,
                     episodes=report['episodes'], food=report['food'], deaths=report['deaths'],
                     no_output=report['no_output'], evaluation=report['evaluation'],
                     message='果蝇脑已恢复记忆，开始连续贪吃蛇')
        return report, provenance

    def _run_episode(self, brain, report, generation, pair, sign, seed):
        row = episode(brain, seed, 'uniform_food', self.settings['size'],
                      self.settings['limit'], on_step=self.observe)
        summary = {k: row[k] for k in ('seed', 'stage', 'size', 'food', 'steps',
                                      'decisions', 'silent', 'reason', 'return')}
        summary.update(generation=generation, pair=pair, sign=sign,
                       candidate_memory_sha256=brain.weight_hash(), at=time.time())
        with (self.run/'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(summary, ensure_ascii=False)+'\n')
            stream.flush()
        report['episodes'] += 1; report['food'] += row['food'];
        report['deaths'] += row['reason'] in ('wall', 'self')
        report['no_output'] += row['silent']
        # Episode statistics are committed independently. Parameters change only
        # after every paired candidate has played the same world seeds.
        atomic_json(self._state_path(), report)
        self.publish(phase='new_episode', episodes=report['episodes'], food=report['food'],
                     deaths=report['deaths'], no_output=report['no_output'],
                     last_episode=summary, message='本局已结束，下一局立即开始')
        return summary

    def work(self):
        try:
            with writer_lock(self.run):
                search = PairedSearch(self.settings['seed'], SearchConfig())
                brain = NativeSearchBrain(search.config.dimensions)
                report, provenance = self._load(brain, search)
                while not self.stop.is_set():
                    generation = search.updates+1
                    world_seeds = np.random.default_rng(self.settings['seed']+10000+generation).integers(
                        100000, 200000, self.settings['batch_size'])
                    directions = search.directions(self.settings['pairs'])
                    returns = []; candidates = []
                    for pair, vector in enumerate(directions, 1):
                        pair_returns = []
                        for sign in (1, -1):
                            brain.set_parameters(search.candidate(vector, sign))
                            self.publish(phase='training', generation=generation, pair=pair,
                                         sign=sign, message='同一批游戏比较正负突触扰动')
                            rows = [self._run_episode(brain, report, generation, pair, sign, int(seed))
                                    for seed in world_seeds]
                            pair_returns.append(float(np.mean([r['return'] for r in rows])))
                            candidates.append({'pair': pair, 'sign': sign,
                                               'returns': pair_returns[-1],
                                               'food': sum(r['food'] for r in rows),
                                               'memory_sha256': brain.weight_hash()})
                        returns.append(pair_returns)
                    update = search.update(directions, returns)
                    brain.set_parameters(search.theta)
                    report['generation'] = search.updates
                    report['recent_generations'].append({'generation': search.updates,
                                                         'candidates': candidates, 'update': update})
                    report['recent_generations'] = report['recent_generations'][-30:]
                    self._save(report, brain, search, provenance)
                    self.publish(phase='saved', message='本代奖惩已写入连接组突触并自动保存',
                                 last_update=update)
                report['status'] = 'stopped'; atomic_json(self._state_path(), report)
                self.publish(phase='stopped', message='已在完整存档边界停止')
        except Exception as exc:
            self.publish(phase='error', message=str(exc), error=type(exc).__name__)
            if self._state_path().is_file():
                report = json.loads(self._state_path().read_text(encoding='utf-8'))
                report['status'] = 'error'; report['error'] = f'{type(exc).__name__}: {exc}'
                atomic_json(self._state_path(), report)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.work, name='native-paired-snake', daemon=True)
        self.thread.start()


def handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type='application/json; charset=utf-8', status=200):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/api/state':
                return self.respond(session.view())
            if path in ('/', '/index.html'):
                return self.respond((Path(__file__).with_name('live.html')).read_bytes(),
                                    'text/html; charset=utf-8')
            return self.respond({'error': 'Not found'}, status=404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.respond({'error': 'Origin denied'}, status=403)
            if urlparse(self.path).path != '/api/stop':
                return self.respond({'error': 'Not found'}, status=404)
            session.stop.set()
            return self.respond({'requested': 'save_and_stop_at_generation_boundary'})
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8769)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--size', type=int, choices=(4, 6, 8, 12), default=4)
    args = p.parse_args()
    session = LiveSession(args.run, args.parent, args.seed, size=args.size)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(session))
    session.start()
    print(f'Native paired Snake: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    finally:
        session.stop.set()
        if session.thread:
            session.thread.join()


if __name__ == '__main__':
    main()
