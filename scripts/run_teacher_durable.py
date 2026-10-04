"""Run the locked trainer with an explicitly recorded Windows IO retry only.

The original numerical training source is preserved. This supplements atomic
status/manifest writes when a concurrent reader briefly denies Windows rename;
it never changes parameters, losses, data selection or neural dynamics.
"""
import json
from pathlib import Path
import shutil
import sys
import time
import train_teacher_spiking as engine

original_atomic=engine.atomic_json


def durable_atomic(path,value):
    for attempt in range(10):
        try:return original_atomic(path,value)
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05*(attempt+1))


def main():
    if '--run' not in sys.argv:raise ValueError('Training run is required')
    out=Path(sys.argv[sys.argv.index('--run')+1]);out.mkdir(parents=True,exist_ok=True)
    patch=out/'io-support';patch.mkdir(exist_ok=True)
    source=Path(__file__);snapshot=patch/source.name
    if snapshot.exists() and engine.file_hash(snapshot)!=engine.file_hash(source):
        raise ValueError('Preserve locked IO support source')
    if not snapshot.exists():shutil.copyfile(source,snapshot)
    durable_atomic(patch/'provenance.json',{'purpose':'Retry transient atomic JSON rename PermissionError only',
        'numerical_trainer_sha256':engine.file_hash(engine.__file__),
        'wrapper_sha256':engine.file_hash(source),'parameter_or_learning_change':False})
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
