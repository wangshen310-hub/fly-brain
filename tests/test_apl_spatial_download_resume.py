"""Persistent byte-download behavior, independent of neural simulations."""
import base64
import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest

crc=pytest.importorskip('google_crc32c')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('apl_data_acquisition_tests',ROOT/'scripts/prepare_apl_spatial_data.py')
acquire=importlib.util.module_from_spec(spec);spec.loader.exec_module(acquire)


class FakePublicSource:
    def __init__(self,data):self.data=data;self.read_offsets=[]
    def open_input_file(self,path):return self
    def read_at(self,length,start):self.read_offsets.append(start);return self.data[start:start+length]


def fixture(tmp_path,monkeypatch):
    data=bytes(range(256))*9
    source={'gcs_path':'fixture/public/bytes','bytes':len(data),'generation':'fixture-1',
            'crc32c_base64':base64.b64encode(crc.Checksum(data).digest()).decode('ascii')}
    report={'sources_remote':{'partners':source},'completed_chunks':{'0':hashlib.sha256(data[:1024]).hexdigest()}}
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'syn-partners.feather.partial').write_bytes(data[:1024]+bytes(len(data)-1024))
    monkeypatch.setattr(acquire,'source_metadata',lambda relative:source.copy())
    return data,source,report,FakePublicSource(data)


def test_verified_checkpointed_chunk_is_reused(tmp_path,monkeypatch):
    data,source,report,remote=fixture(tmp_path,monkeypatch)
    result=acquire.download_partner(remote,tmp_path,report,lambda:None,2,1024)
    assert result.read_bytes()==data and 0 not in remote.read_offsets
    assert set(remote.read_offsets)=={1024,2048}
    assert not (tmp_path/'cache/syn-partners.feather.partial').exists()
    assert report['partner_cache']['official_crc32c_verified']


def test_corrupted_saved_chunk_is_refused_before_new_ranges(tmp_path,monkeypatch):
    data,source,report,remote=fixture(tmp_path,monkeypatch)
    partial=tmp_path/'cache/syn-partners.feather.partial'
    damaged=bytearray(partial.read_bytes());damaged[0]^=1;partial.write_bytes(damaged)
    with pytest.raises(ValueError,match='Checkpointed download chunk changed'):
        acquire.download_partner(remote,tmp_path,report,lambda:None,2,1024)
    assert not remote.read_offsets


def test_changed_remote_generation_cannot_commit_cache(tmp_path,monkeypatch):
    data,source,report,remote=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(acquire,'source_metadata',lambda relative:{**source,'generation':'fixture-2'})
    with pytest.raises(ValueError,match='Official partner object changed'):
        acquire.download_partner(remote,tmp_path,report,lambda:None,2,1024)
    assert not (tmp_path/'cache/syn-partners.feather').exists()
