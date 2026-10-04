"""Read-only audit of the Git index before publishing a research snapshot."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ('.venv', 'runtime/', 'runs/', 'data/', 'diagnostics/', '.env', '.cache/', 'artifacts/')
SECRETS = re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT)
    names = [item.split(b'\t', 1)[1].decode('utf-8') for item in raw.split(b'\0') if item]
    rejected, rows = [], []
    for name in names:
        path = ROOT / name
        size = path.stat().st_size
        if name.startswith(FORBIDDEN) or '__pycache__/' in name or size > 100_000_000:
            rejected.append({'path': name, 'reason': 'Excluded environment/cache/raw data or oversized Git file'})
        if SECRETS.search(path.read_bytes()):
            rejected.append({'path': name, 'reason': 'Credential-like bytes detected; no secret printed'})
        rows.append({'path': name, 'bytes': size})
    required = ['models/current/brain.pt', 'models/current/training-memory.pt',
                'models/current/replay.pt', 'models/current/model.json', 'models/current/graph-manifest.json',
                'scripts/download_current_graph.py', 'scripts/check_current_model.py',
                'tests/test_teacher_afferent.py', 'vendor/stonkfly/LICENSE',
                'reports/tests-complete-2026-10-04.xml', 'reports/model-verification-2026-10-04.json']
    for name in required:
        if name not in names:
            rejected.append({'path': name, 'reason': 'Required snapshot file missing from index'})
    result = {'status': 'passed' if not rejected else 'failed', 'git_files': len(names),
              'total_git_worktree_bytes': sum(row['bytes'] for row in rows),
              'largest_git_files': sorted(rows, key=lambda row: row['bytes'], reverse=True)[:8],
              'rejected': rejected, 'complete_graph_delivery': 'GitHub Release asset; not excluded from delivery',
              'secret_scan': 'Credential patterns scanned without printing matched contents'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if rejected:
        raise ValueError('Publication index audit failed')


if __name__ == '__main__':
    main()
