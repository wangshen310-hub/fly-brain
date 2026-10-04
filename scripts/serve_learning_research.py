"""Read-only loopback dashboard for committed research results."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import time
import base64
from io import BytesIO
from urllib.parse import urlparse, parse_qs
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = {}


def read(path):
    stamp = path.stat().st_mtime_ns
    old = CACHE.get(str(path))
    if old is None or old[0] != stamp:
        value = json.loads(path.read_text(encoding='utf-8'))
        CACHE[str(path)] = (stamp, value)
    return CACHE[str(path)][1]


def snapshot():
    runs = []
    patterns = ('node-validation-v1-*', 'node-perturbation-*', 'native-snake-search-*',
                'online-native-snake-*', 'search-multilayout-*', 'search-dna02-tonic12-*',
                'paired-native-snake-*')
    paths = sorted({p/'status.json' for pattern in patterns for p in (ROOT/'runtime').glob(pattern) if (p/'status.json').is_file()})
    for path in paths:
        try:
            d = read(path)
            rows = d.get('rows', [])
            run = {'name': path.parent.name, 'status': d['status'], 'protocol': d.get('protocol'),
                   'arm': d.get('arm'), 'completed': len(rows) or len(d.get('iterations', [])) or len(d.get('episodes', [])),
                   'phase': rows[-1].get('phase') if rows else None,
                   'path': str(path), 'evaluations': {}, 'curve': [], 'food': {},
                   'memory_saved': len(d.get('checkpoints', [])),
                   'progress': d.get('progress'), 'settings': d.get('settings'),
                   'updated_at': path.stat().st_mtime,
                   'error': d.get('error'),
                   'mechanism': {k: d.get('brain', {}).get(k) for k in
                                 ('neurons', 'graph_edges', 'mutable_edges', 'dimensions', 'direct_edges')}}
            ev = d.get('evaluations')
            if isinstance(ev, dict):
                run['evaluations'] = {phase: {k: {q: v[q] for q in ('accuracy', 'correct', 'n')}
                                             for k, v in evaluation.items()} for phase, evaluation in ev.items()}
            if rows:
                for i in range(0, len(rows), 40):
                    batch = rows[i:i+40]
                    run['curve'].append({'through': i+len(batch), 'accuracy': sum(bool(r.get('correct')) for r in batch)/len(batch)})
            if 'test_fitness' in d and 'snake' not in str(d.get('protocol')):
                test = d.get('test', [])
                run['development_test'] = {'correct': sum(r['reward'] > 0 for r in test), 'n': len(test),
                                           'accuracy': (1+d['test_fitness'])/2,
                                           'novelty_claim': 'pixel_disjoint' if d.get('pixel_disjoint') else 'not_established'}
            if 'snake' in str(d.get('protocol')) or path.parent.name.startswith('paired-native-snake-'):
                if isinstance(d.get('baseline'), dict):
                    base = d['baseline']['episodes']
                    run['food']['before'] = {'food': sum(r['food'] for r in base), 'n': len(base)}
                if isinstance(d.get('test'), list):
                    run['food']['after'] = {'food': sum(r['food'] for r in d['test']), 'n': len(d['test'])}
                for phase in ('before', 'after'):
                    if phase in d:
                        for kind, values in d[phase].items():
                            run['food'][f'{phase}_{kind}'] = {q: values[q] for q in ('food', 'n')}
                episodes = d.get('episodes', [])
                if episodes:
                    run['food']['training'] = {'food': sum(r['food'] for r in episodes), 'n': len(episodes)}
            runs.append(run)
        except (OSError, ValueError, KeyError, TypeError):
            # An uncommitted or incompatible run must not become a fake zero.
            continue
    return {'time': time.time(), 'goal_complete': False, 'runs': runs}


def replay(name):
    # Select only a known committed run. Never resolve an arbitrary user path.
    available = {r['name']: r for r in snapshot()['runs'] if r['name'].startswith('paired-native-snake-')}
    if name not in available:
        raise ValueError('Unknown paired experiment')
    path = Path(available[name]['path']); data = read(path)
    if 'after' in data:
        episode = data['after']['uniform_food']['episodes'][0]
        label = '训练后冻结评估 · 普通食物 · 第 1 局'
    elif data.get('iterations'):
        file = data['iterations'][-1]['candidates'][-1][-1]['file']
        if Path(file).name != file:
            raise ValueError('Invalid committed rollout filename')
        episode = read(path.parent/file)['episodes'][0]
        label = '最近完整代 · 探索候选 · 第 1 局（不是冻结成绩）'
    elif 'before' in data:
        episode = data['before']['uniform_food']['episodes'][0]
        label = '训练前基线 · 普通食物 · 第 1 局'
    else:
        return {'ready': False, 'label': '等待第一批完整轨迹'}
    from train_paired_native_snake import CurriculumGame
    game = CurriculumGame(seed=episode['seed'], stage=episode['stage'], size=episode['size'],
                          calibration=data['brain']['calibration'])
    frames = []
    for row in episode['rows']:
        receipt = row['receipt']; rgb = game.render_observation()
        if game.observation_hash() != receipt['observation_sha256']:
            raise ValueError('Replay pixels differ from actual recorded brain input')
        buf = BytesIO(); Image.fromarray(rgb).save(buf, format='PNG')
        frames.append({'image': 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
                       'action': receipt['action'], 'counts': receipt['counts'],
                       'reason': receipt['reason'], 'food': game.score,
                       'event': row['event'], 'feedback': row['feedback'],
                       'observation_sha256': receipt['observation_sha256']})
        if receipt['action'] is not None:
            event = game.apply_receipt(receipt)
            if not event['accepted']:
                raise ValueError('Replay receipt rejected')
            if event['kind'] != 'none':
                game.complete_feedback(event['event_id'])
    return {'ready': True, 'label': label, 'frames': frames, 'food': episode['food'],
            'reason': episode['reason'], 'seed': episode['seed'], 'pixels_verified': True}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/replay':
            try:
                body = json.dumps(replay(parse_qs(parsed.query).get('run', [''])[0]), ensure_ascii=False).encode('utf-8')
            except (ValueError, KeyError, OSError) as exc:
                self.send_error(400, str(exc)); return
            content = 'application/json; charset=utf-8'
        elif self.path == '/api/progress':
            body = json.dumps(snapshot(), ensure_ascii=False).encode('utf-8')
            content = 'application/json; charset=utf-8'
        elif self.path in ('/', '/index.html'):
            body = (ROOT/'flysnake/lab/paired_progress.html').read_bytes()
            content = 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8773)
    args = p.parse_args()
    print(f'Read-only research progress: http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
