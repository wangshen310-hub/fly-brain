"""Copy the audited snapshot into portable, Git-sized artifacts.

This tool never trains, rewrites a checkpoint or changes historical runs.
The complete graph is released separately because it exceeds Git's limit.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TAG = 'v0.1.0-audit'
REPO = 'wangshen310-hub/fly-brain'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def copy(source, destination):
    source, destination = ROOT / source, ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination) != digest(source):
        raise ValueError('Refusing to replace a different artifact: ' + str(destination))
    if not destination.exists():
        shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise ValueError('Copy checksum mismatch')
    return destination


def main():
    run = ROOT / 'runtime/teacher-afferent-probe-20261003-normal-v1'
    status, committed, replay = [read(run / name) for name in ('status.json', 'committed.json', 'replay.json')]
    if status['status'] != 'complete' or committed['step'] != 60:
        raise ValueError('Completed current 60-step checkpoint required')
    bundle = ROOT / 'models/current'
    checkpoints = {'brain.pt': ('best-0060.pt', status['best_checkpoint']),
                   'training-memory.pt': ('memory-0060.pt', committed['checkpoint']),
                   'replay.pt': ('replay.pt', replay['checkpoint'])}
    for name, (original, record) in checkpoints.items():
        if digest(run / original) != record['sha256']:
            raise ValueError('Historical checkpoint changed: ' + original)
        copy((run / original).relative_to(ROOT), 'models/current/' + name)
    for name in ('replay-frames.npz', 'replay-states.npz', 'replay.json', 'status.json'):
        copy((run / name).relative_to(ROOT), 'models/current/' + name)
    write(bundle / 'committed.json', {'step': 60, 'checkpoint': {
        **committed['checkpoint'], 'path': 'models/current/training-memory.pt'}})

    graph = ROOT / 'runtime/supervised-graph-20261003-v4'
    graph_meta = read(graph / 'manifest.json')
    if digest(graph / 'graph.npz') != graph_meta['graph_sha256']:
        raise ValueError('Complete graph integrity mismatch')
    copy('runtime/supervised-graph-20261003-v4/manifest.json', 'models/current/graph-manifest.json')

    dataset = ROOT / 'runtime/teacher-scenes-20261003-v1'
    dataset_meta = read(dataset / 'manifest.json')
    if digest(dataset / 'dataset.npz') != dataset_meta['dataset_sha256']:
        raise ValueError('Dataset integrity mismatch')
    for name in ('dataset.npz', 'manifest.json'):
        copy((dataset / name).relative_to(ROOT), 'models/current/dataset/' + name)
    smoke = []
    with np.load(dataset / 'dataset.npz', allow_pickle=False) as data:
        frames = data['train_rgb'][status['train_indices']]
    for index, (frame, expected) in enumerate(zip(frames, status['final']['receipts'])):
        pixel_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        if pixel_hash != expected['input_sha256']:
            raise ValueError('Smoke image identity mismatch')
        name = f'smoke/frame-{index}.png'
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(bundle / name)
        smoke.append({'path': name, 'rgb_sha256': pixel_hash, 'expected_receipt': expected})

    # Preserve the actual upstream training seed and the matching wrong-teacher control.
    controls = [('runtime/teacher-spiking-fit-20261003-s1003-v1/memory-0200.pt', 'warm-upstream/brain.pt'),
                ('runtime/teacher-afferent-probe-20261003-normal-v1/initial.pt', 'initial-afferents.pt'),
                ('runtime/teacher-afferent-probe-20261003-mismatch-v1/memory-0060.pt', 'mismatch/training-memory.pt'),
                ('runtime/teacher-afferent-probe-20261003-mismatch-v1/status.json', 'mismatch/status.json')]
    for original, name in controls:
        copy(original, 'models/controls/' + name)
    warm = status['warm_checkpoint']
    if digest(ROOT / 'models/controls/warm-upstream/brain.pt') != warm['sha256']:
        raise ValueError('Wrong upstream seed')
    write(ROOT / 'models/controls/warm-upstream/committed.json', {'step': 200, 'checkpoint': {
        **warm, 'path': 'models/controls/warm-upstream/brain.pt'}})
    mismatch = read(ROOT / 'runtime/teacher-afferent-probe-20261003-mismatch-v1/committed.json')
    if digest(ROOT / 'models/controls/mismatch/training-memory.pt') != mismatch['checkpoint']['sha256']:
        raise ValueError('Wrong mismatch control')
    write(ROOT / 'models/controls/mismatch/committed.json', {'step': 60, 'checkpoint': {
        **mismatch['checkpoint'], 'path': 'models/controls/mismatch/training-memory.pt'}})

    evidence_dirs = {
        'causality': 'runtime/teacher-afferent-causality-20261003-normal-v1',
        'frozen': 'runtime/teacher-afferent-frozen-20261003-s1003-v1',
        'upstream': 'runtime/teacher-afferent-upstream-audit-20261003-normal-v1',
        'inputs': 'runtime/teacher-afferent-input-audit-20261003-normal-v1',
        'timecourse': 'runtime/teacher-afferent-timecourse-20261003-normal-v1',
        'timecourse-repeat': 'runtime/teacher-afferent-timecourse-repeat-20261003-normal-v1',
    }
    for name, original in evidence_dirs.items():
        copy(original + '/summary.json', 'docs/evidence/' + name + '/summary.json')
    for original in (ROOT / evidence_dirs['frozen']).glob('*.json'):
        copy(original.relative_to(ROOT), 'docs/evidence/frozen/' + original.name)
    copy(evidence_dirs['upstream'] + '/signal_arrays.npz', 'docs/evidence/upstream/signal_arrays.npz')
    diagnostics = [
        'teacher-afferent-frozen-comparison-20261003-v1.json',
        'teacher-afferent-replay-20261003-normal-v1.json',
        'teacher-afferent-timecourse-repeat-20261003-v1.json',
        'teacher-left-forward-upstream-final-20261003-v1.json',
        'teacher-actual-memory-replay-20261003-s1003-v1.json',
    ]
    for name in diagnostics:
        copy('diagnostics/' + name, 'docs/evidence/' + name)

    # Current report links resolve on GitHub. Raw records retain historical provenance paths.
    report = (ROOT / 'SUPERVISED_AUDIT_2026-10-03.md').read_text(encoding='utf-8')
    prefix = ROOT.as_posix() + '/'
    for name, original in evidence_dirs.items():
        report = report.replace(prefix + original + '/summary.json', 'evidence/' + name + '/summary.json')
    for name in diagnostics:
        report = report.replace(prefix + 'diagnostics/' + name, 'evidence/' + name)
    if '](C:/' in report:
        raise ValueError('Unmapped local path in current report')
    (ROOT / 'docs/current-model-audit.md').write_text(report, encoding='utf-8')

    source_paths = ['flysnake/research/teacher_spiking.py', 'flysnake/research/teacher_afferent.py',
                    'flysnake/research/optical_sampling.py', 'flysnake/research/teacher_game.py',
                    'scripts/probe_afferent_teacher.py', 'scripts/prepare_supervised_graph.py',
                    'scripts/prepare_supervised_csr.py']
    source_hashes = {name: digest(ROOT / name) for name in source_paths}
    identity = status['brain']
    for path, field in [('flysnake/research/teacher_spiking.py', 'source_sha256'),
                        ('flysnake/research/teacher_afferent.py', 'afferent_source_sha256'),
                        ('flysnake/research/optical_sampling.py', 'optics_source_sha256')]:
        if source_hashes[path] != identity[field]:
            raise ValueError('Model source bytes changed: ' + path)
    files = {p.relative_to(bundle).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(bundle.rglob('*')) if p.is_file() and p.name != 'model.json'}
    related = {p.relative_to(ROOT).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
               for folder in ('models/controls', 'docs/evidence') for p in sorted((ROOT / folder).rglob('*')) if p.is_file()}
    manifest = {'format': 'fly-brain-research-snapshot-v1', 'archived_on': '2026-10-04',
                'model_version': identity['version'], 'brain_identity': identity,
                'parameter_sha256': status['trained_parameter_sha256'],
                'checkpoint': 'brain.pt', 'training_memory': 'training-memory.pt', 'replay_checkpoint': 'replay.pt',
                'files': files, 'related_artifacts': related, 'source_sha256': source_hashes, 'smoke': smoke,
                'complete_graph': {'repo': REPO, 'release_tag': TAG, 'asset': 'graph.npz',
                    'manifest_asset': 'manifest.json', 'bytes': (graph / 'graph.npz').stat().st_size,
                    'sha256': graph_meta['graph_sha256'], 'manifest_sha256': digest(graph / 'manifest.json'),
                    'neurons': graph_meta['neurons'], 'anatomical_edges': graph_meta['anatomical_edges']},
                'ordinary_game_learning_validated': False,
                'reward_learning_enabled_in_frozen_evaluation': False,
                'summary': 'Limited three-training-image conditional right response; ordinary 12x12 Snake evaluation ate zero food.'}
    write(bundle / 'model.json', manifest)
    print(json.dumps({'bundle_files': len(files), 'related_files': len(related),
                      'git_model_bytes': sum(x['bytes'] for x in files.values()),
                      'complete_graph_bytes': manifest['complete_graph']['bytes'],
                      'complete_graph_sha256': manifest['complete_graph']['sha256']}))


if __name__ == '__main__':
    main()
