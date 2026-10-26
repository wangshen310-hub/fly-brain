"""CPU-only source/archived-JSON diagnosis; never load model weights or a graph.

This re-scores EXISTING historical receipts and checks tiny optimizer fixtures.
It is not a new inference, training run, or measurement of full-brain learning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.train_teacher_spiking import checked_optimizer_step, compact, metrics, two_window_gate


def diagnose():
    source = ROOT / 'docs/evidence/timecourse/summary.json'
    historical = json.loads(source.read_text(encoding='utf-8'))
    conditions = ['left-food', 'forward-food', 'right-food']
    labels = np.array([0, 1, 2])
    rows = [{'teacher': name, 'adjacent_eat': False} for name in ('left', 'forward', 'right')]
    windows = []
    for window in range(2):
        records = [historical['conditions'][name]['windows'][window] for name in conditions]
        assert all((r['start_tick'], r['end_tick']) == (100*window, 100*(window+1)) for r in records)
        motor = np.array([r['counts']['left'] + r['counts']['right'] + r['counts']['forward']
                          for r in records])
        scored = metrics(motor, labels, rows)
        assert [r['action'] for r in scored['receipts']] == [r['action'] for r in records]
        windows.append(compact(scored))
    incomplete = metrics(np.array([[5, 5, 0, 0]]), np.array([1]), [rows[1]])
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.Adam([p], lr=.01)
    p.grad = torch.tensor([.25, .25])
    checked_optimizer_step(optimizer, [p])
    before = p.detach().clone()
    step = optimizer.state[p]['step'].clone()
    p.grad = torch.tensor([float('inf'), 0.])
    rejected = False
    try:
        checked_optimizer_step(optimizer, [p])
    except RuntimeError:
        rejected = True
    return {
        'scope': 'CPU safety diagnosis and re-score of existing archived JSON, NOT new brain inference',
        'source_base_commit': 'e582e7405f3f4afc172539162060240221d18bd9',
        'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'cuda_available': torch.cuda.is_available()},
        'archived_receipt_rescore': {
            'path': str(source.relative_to(ROOT)),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sample_scope': 'One archived training pose, three directional scenes; no generalization claim',
            'windows': windows, 'gate': two_window_gate(windows, 100),
            'new_inference_run': False,
        },
        'missing_class_fixture': compact(incomplete),
        'second_step_nonfinite_gradient_fixture': {
            'rejected_before_optimizer_step': rejected,
            'parameter_unchanged': torch.equal(p, before),
            'adam_step_unchanged': torch.equal(optimizer.state[p]['step'], step),
        },
        'complete_brain_learning_confirmed': False,
        'original_step_112_recovered': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON file; refuses to overwrite')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = json.dumps(diagnose(), indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
