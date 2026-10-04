"""Frozen RGB response survey of annotated descending neurons, no actions."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import cue_scene, neutral_scene

brain = LabBrain()
b = brain.b
from stonkfly.neural.common import annotations
a = annotations(b.ids)
samples, descriptions = [], []
for seed in (4081, 4082, 4083, 4084, 444, 555):
    for side in ('left', 'right'):
        brain.observe(cue_scene(seed, side), learning=False)
        samples.append(brain.last['counts'].copy())
        descriptions.append({'seed': seed, 'cue': side})
counts = np.array(samples)
dn = np.flatnonzero(a.type.fillna('').str.startswith('DN'))
rows = []
for ix in dn:
    if not counts[:, ix].any() and a.type.iloc[ix] not in ('DNa01', 'DNa02', 'DNa03', 'DNp09'):
        continue
    edges = np.flatnonzero(b.post == ix)
    pre = np.searchsorted(b.ptr, edges, side='right') - 1
    fast = b.modulation_mask[pre] == 0
    edges, pre = edges[fast], pre[fast]
    active = np.any(counts[:, pre] > 0, axis=0)
    left, right = counts[::2, ix].mean(), counts[1::2, ix].mean()
    row = {'id': int(b.ids[ix]), 'type': str(a.type.iloc[ix]), 'side': str(a.somaSide.iloc[ix]),
           'left_mean': float(left), 'right_mean': float(right), 'counts': counts[:, ix].tolist(),
           'fast_inputs': len(edges), 'active_inputs': int(active.sum())}
    rows.append(row)
    if a.type.iloc[ix] in ('DNa01', 'DNa02', 'DNa03', 'DNp09', 'DNp20', 'DNpe017'):
        print(json.dumps(row), flush=True)
out = ROOT / 'diagnostics/descending-rgb-survey-2026-09-22.json'
out.write_text(json.dumps({'samples': descriptions, 'rows': rows}, indent=2), encoding='utf-8')
print(json.dumps({'file': str(out), 'cells': len(rows)}), flush=True)
