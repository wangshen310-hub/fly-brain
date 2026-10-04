from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from flysnake.research.apl_local import LocalAPLConfig
from flysnake.research.apl_local_state_v2 import save,load


def fixture_brain():
    b=SimpleNamespace(fields=['r8_light','queue'],r8_light=np.array([.25],np.float32),queue=np.array([1,0],np.int32),
        weight=np.array([2.,3.],np.float32),tonic=np.array([12.,0.],np.float32),cursor=10,sim_ms=1.,total_spikes=7)
    return SimpleNamespace(b=b,identity={'test':'field-isolation'},kernel=SimpleNamespace(config=LocalAPLConfig()),
        edges=np.array([1]),base=np.array([3.],np.float32),retinal_light=np.array([.4]),r8_light=np.array([.8]),neural_ms=1.)


def test_native_and_adapter_r8_fields_roundtrip_without_collision(tmp_path):
    brain=fixture_brain();path=tmp_path/'state.npz';sha=save(brain,path)
    brain.b.r8_light.fill(0);brain.r8_light.fill(0);brain.b.queue.fill(0)
    load(brain,path,sha)
    assert brain.b.r8_light.dtype==np.float32 and brain.b.r8_light[0]==.25
    assert brain.r8_light.dtype==np.float64 and brain.r8_light[0]==.8
    np.testing.assert_array_equal(brain.b.queue,[1,0])
    with pytest.raises(FileExistsError):save(brain,path)


def test_identity_rejected_before_state_mutation(tmp_path):
    brain=fixture_brain();path=tmp_path/'state.npz';sha=save(brain,path)
    brain.identity={'test':'other'};brain.b.queue.fill(0)
    with pytest.raises(ValueError,match='source, graph'):load(brain,path,sha)
    np.testing.assert_array_equal(brain.b.queue,0)


def test_corrupted_checkpoint_rejected_before_state_mutation(tmp_path):
    brain=fixture_brain();path=tmp_path/'state.npz';sha=save(brain,path)
    with path.open('ab') as stream:stream.write(b'corruption')
    brain.b.queue.fill(0)
    with pytest.raises(ValueError,match='hash differs'):load(brain,path,sha)
    np.testing.assert_array_equal(brain.b.queue,0)
