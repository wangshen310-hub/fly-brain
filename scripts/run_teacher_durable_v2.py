"""Locked trainer plus recorded, bounded retry of Windows JSON rename."""
import json
from pathlib import Path
import shutil
import sys
import time
import train_teacher_spiking as engine

original_atomic=engine.atomic_json
out=None
provenance=None


def durable_atomic(path,value):
    if path.name=='status.json':
        support=out/'io-support';support.mkdir(exist_ok=True)
        source=Path(__file__);snapshot=support/source.name
        if snapshot.exists() and engine.file_hash(snapshot)!=engine.file_hash(source):
            raise ValueError('Preserve locked IO support source')
        if not snapshot.exists():shutil.copyfile(source,snapshot)
        if not (support/'provenance.json').exists():
            original_atomic(support/'provenance.json',provenance)
        value={**value,'io_support':provenance}
    for attempt in range(10):
        try:return original_atomic(path,value)
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05*(attempt+1))


def main():
    global out,provenance
    if '--run' not in sys.argv:raise ValueError('Training run is required')
    out=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
    provenance={'purpose':'Bounded retry of transient atomic JSON rename only',
        'trainer_sha256':engine.file_hash(engine.__file__),'wrapper_sha256':engine.file_hash(__file__),
        'numerical_training_changed':False}
    engine.atomic_json=durable_atomic
    try:engine.main()
    except Exception as error:
        status=out/'status.json'
        if status.exists():
            with status.open(encoding='utf-8') as f:value=json.load(f)
            value.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
            durable_atomic(status,value)
        raise


if __name__=='__main__':main()
