"""Read-only checkpoint assay; writes only a separate diagnostic JSON.

Each trial resets short state, freezes all plasticity, and presents the same
image for five 100 ms windows. This is a sensitivity assay, not a play score.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.neural import RealFlyBrain
from flysnake.game import SnakeGame
from flysnake.decoder import decode_counts


def main():
    storage = ROOT / 'runtime/stronger-feedback-v6'
    pointer = storage / 'runs/latest_memory.json'
    record = json.loads(pointer.read_text(encoding='utf-8'))
    hashes = {str(pointer): hashlib.sha256(pointer.read_bytes()).hexdigest()}
    for name, expected in record['files'].items():
        path = storage / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected
        hashes[str(path)] = actual
    brain = RealFlyBrain(learning=False)
    brain.restore(storage / record['path'])
    brain.learning = False
    b = brain.brain
    b.weights_frozen = True
    trained = [b.weight[b.circuit['edges']].copy(), b.memory_u.copy(), b.memory_w.copy()]
    game = SnakeGame()
    game.snake = [(6, 6), (6, 7), (6, 8)]
    game.direction = 0
    scenes = {}
    for name, food in [('food_left', (3, 6)), ('food_right', (9, 6)),
                       ('food_ahead', (6, 3)), ('food_absent', None)]:
        game.food = food
        scenes[name] = game.render_observation()
    game.food = (6, 3)
    game.snake = [(3, 6), (3, 7), (3, 8)]
    scenes['head_shifted'] = game.render_observation()
    scenes['neutral_gray'] = np.full((180, 320, 3), 128, dtype=np.uint8)
    scenes['food_left_repeat'] = scenes['food_left'].copy()
    rows = []
    for mode in ['trained_frozen', 'memory_removed_frozen']:
        for name, frame in scenes.items():
            if mode == 'trained_frozen':
                b.weight[b.circuit['edges']] = trained[0]
                b.memory_u[:] = trained[1]
                b.memory_w[:] = trained[2]
            else:
                b.weight[b.circuit['edges']] = b.baseline_plastic
                b.memory_u.fill(0)
                b.memory_w.fill(0)
            brain.reset_episode()
            before = brain.memory()
            windows = []
            total = np.zeros(b.n, dtype=np.int32)
            for _ in range(5):
                counts, _ = brain._integrate(frame, 100.0, learning=False)
                total += counts
                groups = {k: counts[ix].tolist() for k, ix in brain.groups.items()}
                windows.append({'counts': groups, 'decision': decode_counts(groups, .1)})
            assert brain.memory()['sha256'] == before['sha256']
            groups = {k: total[ix].tolist() for k, ix in brain.groups.items()}
            row = {'mode': mode, 'scene': name, 'windows': windows,
                   'counts_500ms': groups, 'decision_500ms': decode_counts(groups, .5),
                   'mbon_counts': total[b.circuit['mb']].tolist(),
                   'kc_spikes': int(total[b.circuit['kc']].sum()),
                   'all_spikes': int(total.sum()), 'memory_sha256': before['sha256']}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['mode', 'scene', 'counts_500ms', 'mbon_counts']}, ensure_ascii=False), flush=True)
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    output = ROOT / 'diagnostics/learning-audit-2026-09-13.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'checkpoint': record['checkpoint_id'],
        'protocol': 'identical short-state reset; frozen; 500ms static image; no feedback',
        'checkpoint_files_unchanged': True, 'trials': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
