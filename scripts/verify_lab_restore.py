"""Load a committed lab memory in an independent brain process, never train it."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flysnake.lab.experiment import Experiment, BrainProcess, DEFAULT_STORAGE


def main():
    storage=DEFAULT_STORAGE
    pointer=storage/'latest.json'
    before=hashlib.sha256(pointer.read_bytes()).hexdigest()
    e=Experiment(storage);e.brain=BrainProcess()
    try:
        if not e.restore():raise RuntimeError('No committed experiment')
        actual=e.brain.rpc('metadata')
        expected=e.state['brain']['memory_sha256']
        assert actual['memory_sha256']==expected
        assert hashlib.sha256(pointer.read_bytes()).hexdigest()==before
        report={'restored_memory_exactly':True,'memory_sha256':expected,
                'cursor':e.state['cursor'],'completed_trials':len(e.state['rows']),
                'phase':e.state['phase'],'pointer_unchanged':True}
        (storage/'restore-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    finally:e.brain.close()


if __name__=='__main__':main()
