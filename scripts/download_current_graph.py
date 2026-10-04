"""Download the complete, checksum-locked graph from the same GitHub repository.

A private repository requires gh auth login. No credentials are stored here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(folder, graph):
    model, manifest = folder / graph['asset'], folder / graph['manifest_asset']
    if (model.stat().st_size != graph['bytes'] or digest(model) != graph['sha256']
            or digest(manifest) != graph['manifest_sha256']):
        raise ValueError('Downloaded complete graph or manifest integrity mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runtime/supervised-graph-20261003-v4')
    args = parser.parse_args()
    graph = json.loads((ROOT / 'models/current/model.json').read_text(encoding='utf-8'))['complete_graph']
    output = args.output.resolve()
    if (output / graph['asset']).exists():
        validate(output, graph)
        print('Existing complete graph and manifest verified: ' + str(output))
        return
    if (output / graph['manifest_asset']).exists():
        raise FileExistsError('Incomplete destination exists; choose a new --output directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fly-brain-download-', dir=output.parent) as temporary:
        downloaded = Path(temporary)
        subprocess.run(['gh', 'release', 'download', graph['release_tag'], '--repo', graph['repo'],
                        '--pattern', graph['asset'], '--pattern', graph['manifest_asset'], '--dir', str(downloaded)], check=True)
        validate(downloaded, graph)
        output.mkdir(parents=True, exist_ok=True)
        for name in (graph['asset'], graph['manifest_asset']):
            (downloaded / name).replace(output / name)
    validate(output, graph)
    print('Complete graph downloaded and verified: ' + str(output))


if __name__ == '__main__':
    main()
