"""Memory transaction checks with a tiny fixture, not behavioral evidence."""
import sys
from pathlib import Path
import copy
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_sensory_basis import save_memory,restore_memory,ARMS
from flysnake.research.paired import digest


class Fixture:
    identity={'graph':'fixture','sensory_basis':'locked','calibration':[1,2,3]}
    def set_parameters(self,a):
        a=np.asarray(a)
        if a.shape!=(2,) or not np.isfinite(a).all() or (abs(a)>3).any():
            raise ValueError('Invalid fixture parameters')
        self.a=a.copy()
    def weight_hash(self):return digest(np.exp(self.a)*np.array([1.,-1.]))


def stored(tmp_path):
    report={'cycles_completed':2,'cycles':[{'cycle':1},{'cycle':2}],
            'sources':{'fixture':'hash'},'settings':{'seed':123},'checkpoints':[]}
    memory={a:np.array([i*.1,-i*.2]) for i,a in enumerate(ARMS)}
    rng=np.random.default_rng(123);rng.normal(size=8)
    save_memory(tmp_path,report,memory,rng,Fixture())
    return report,memory,rng


def test_memory_restores_all_arms_and_exact_next_random_draw(tmp_path):
    report,memory,rng=stored(tmp_path)
    # An uncommitted comparison must never become the next starting memory.
    report['cycles_completed']=3;report['cycles'].append({'cycle':3})
    loaded,restored=restore_memory(tmp_path,report,Fixture())
    assert report['cycles_completed']==2 and len(report['cycles'])==2
    for arm in ARMS:assert np.array_equal(loaded[arm],memory[arm])
    assert np.array_equal(restored.normal(size=20),rng.normal(size=20))


def test_corrupt_file_is_rejected_and_left_intact(tmp_path):
    report,_,_=stored(tmp_path);path=tmp_path/report['checkpoints'][-1]['file']
    bytes_=bytearray(path.read_bytes());bytes_[40]^=1;path.write_bytes(bytes_)
    with pytest.raises(ValueError,match='hash'):restore_memory(tmp_path,report,Fixture())
    assert path.read_bytes()==bytes(bytes_)


def test_changed_source_or_identity_cannot_silently_load_memory(tmp_path):
    report,_,_=stored(tmp_path);changed=copy.deepcopy(report);changed['sources']['fixture']='new'
    with pytest.raises(ValueError,match='configuration'):restore_memory(tmp_path,changed,Fixture())
    other=Fixture();other.identity={'graph':'other','sensory_basis':'locked','calibration':[1,2,3]}
    with pytest.raises(ValueError,match='configuration'):restore_memory(tmp_path,report,other)
