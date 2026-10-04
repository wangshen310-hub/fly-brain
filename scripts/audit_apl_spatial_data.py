"""Independent byte/geometry/raw-body-pair audit of completed APL acquisition.

The acquisition compares normalized native-index edges. This audit instead
groups body IDs with Arrow and reads original official body-body weights,
including objects excluded from the neural model. No acquisition comparison
function is reused. Full cache SHA256 and official CRC32C are independently read.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import time
import google_crc32c
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

ROOT=Path(__file__).resolve().parents[1]
APL_IDS=(10540,10977)


def hashes(path):
    sha=hashlib.sha256();crc=google_crc32c.Checksum()
    with path.open('rb') as stream:
        while block:=stream.read(32*1024*1024):sha.update(block);crc.update(block)
    return sha.hexdigest(),base64.b64encode(crc.digest()).decode('ascii')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous independent audit')
    status=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise ValueError('Only audit an actually completed data preparation')
    for name,sha in status['sources'].items():
        for path in (ROOT/name,args.run/'sources'/name):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Acquisition source changed')
    cache=args.run/status['partner_cache']['file'];sha,crc=hashes(cache)
    if (sha!=status['partner_cache']['sha256'] or
            crc!=status['sources_remote']['partners']['crc32c_base64']):
        raise ValueError('Full official partner cache integrity failed')
    print(json.dumps({'phase':'full-cache-independent-check','sha256_equal':True,'official_crc32c_equal':True}),flush=True)
    skeletons={}
    for body in APL_IDS:
        entry=status['skeletons'][str(body)]
        sha,crc=hashes(args.run/entry['file'])
        if sha!=entry['sha256'] or crc!=status['sources_remote'][str(body)]['crc32c_base64']:
            raise ValueError('Official APL SWC integrity failed')
        normalized=args.run/entry['normalized_file']
        if hashlib.sha256(normalized.read_bytes()).hexdigest()!=entry['normalized_sha256']:
            raise ValueError('Prepared skeleton changed')
        raw=np.loadtxt(args.run/entry['file'],comments='#',ndmin=2)
        with np.load(normalized,allow_pickle=False) as z:
            lookup={int(cell):i for i,cell in enumerate(raw[:,0])}
            parent=np.array([-1 if cell==-1 else lookup[int(cell)] for cell in raw[:,6]],np.int32)
            for actual,expected in ((z['node_ids'],raw[:,0].astype(np.int64)),
                                    (z['parent_index'],parent),
                                    (z['xyz_voxel'],raw[:,2:5]),
                                    (z['xyz_um'],raw[:,2:5]*.008),
                                    (z['radius_um'],raw[:,5]*.008)):
                if not np.array_equal(actual,expected):raise ValueError('SWC coordinate/topology conversion differs')
            components=np.bincount(z['component']).tolist()
        skeletons[str(body)]={'bytes_and_native_geometry_equal':True,'nodes':len(raw),
                              'component_nodes':components,'coordinates_um_verified':True}
    contact_path=args.run/status['APL_contacts']['file']
    if hashlib.sha256(contact_path.read_bytes()).hexdigest()!=status['APL_contacts']['sha256']:
        raise ValueError('Prepared contact table changed')
    contacts=feather.read_table(contact_path)
    pre_body=contacts['body_pre'].to_numpy();post_body=contacts['body_post'].to_numpy()
    if not (np.isin(pre_body,APL_IDS)|np.isin(post_body,APL_IDS)).all():raise ValueError('Non-APL contacts in extracted table')
    rows=contacts['global_partner_row'].to_numpy()
    expected_source_rows=json.loads((ROOT/'data/malecns_v1/normalized/report.json').read_text(encoding='utf-8'))['graph']['source_synaptic_contacts']
    if (status['APL_contacts']['source_rows_scanned']!=expected_source_rows or
            not (np.diff(rows)>0).all() or rows.max()>=expected_source_rows):
        raise ValueError('Full source scan or unique global row identity differs')
    ids=np.load(ROOT/'data/malecns_v1/normalized/neuron_ids.npy').astype(np.int64)
    lookup={int(cell):i for i,cell in enumerate(ids)}
    pre=np.fromiter((lookup.get(int(cell),-1) for cell in pre_body),np.int32,count=len(contacts))
    post=np.fromiter((lookup.get(int(cell),-1) for cell in post_body),np.int32,count=len(contacts))
    if (not np.array_equal(pre,contacts['pre_native_index'].to_numpy()) or
            not np.array_equal(post,contacts['post_native_index'].to_numpy()) or
            not np.array_equal((pre>=0)&(post>=0),contacts['retained_native_contact'].to_numpy())):
        raise ValueError('Published bodies and actual native node order differ')
    # Independent raw graph: no normalized indexing or shared aggregate helper.
    lock=json.loads((ROOT/'data/malecns_v1/source.lock.json').read_text(encoding='utf-8'))
    raw_path=ROOT/'data/malecns_v1/edges.feather'
    raw_sha,_=hashes(raw_path)
    if raw_sha!=lock['edges.feather']['sha256']:raise ValueError('Historical official raw weights changed')
    batches=[]
    with pa.memory_map(str(raw_path),'r') as stream:
        reader=pa.ipc.open_file(stream)
        for i in range(reader.num_record_batches):
            batch=reader.get_batch(i)
            pre=batch.column(0).to_numpy();post=batch.column(1).to_numpy()
            mask=np.isin(pre,APL_IDS)|np.isin(post,APL_IDS)
            if mask.any():batches.append(batch.filter(pa.array(mask)))
    expected=pa.Table.from_batches(batches).group_by(['body_pre','body_post']).aggregate([('weight','sum')])
    expected=expected.rename_columns(['body_pre','body_post','contact_count']).sort_by([('body_pre','ascending'),('body_post','ascending')])
    actual=contacts.select(['body_pre','body_post']).append_column('one',pa.array(np.ones(len(contacts),np.int64)))
    actual=actual.group_by(['body_pre','body_post']).aggregate([('one','sum')])
    actual=actual.rename_columns(['body_pre','body_post','contact_count']).sort_by([('body_pre','ascending'),('body_post','ascending')])
    if not actual.equals(expected,check_metadata=False):raise ValueError('Actual APL positions do not reproduce ALL official body-body counts')
    result={'version':'independent-apl-spatial-data-audit-v1','status':'complete','generated_at':time.time(),
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'acquisition_status_sha256':hashlib.sha256((args.run/'status.json').read_bytes()).hexdigest(),
            'full_partner_cache_sha256_equal':True,'full_partner_cache_official_crc32c_equal':True,
            'skeletons':skeletons,'extracted_contacts':len(contacts),'source_rows_scanned':expected_source_rows,
            'native_body_id_to_index_equal':True,'all_raw_official_APL_body_pair_counts_equal':True,
            'all_raw_APL_pairs':len(actual),'original_official_weights_sha256':raw_sha,
            'scope':'Acquisition and correspondence only. Coarse morphology and published contact coordinates; '
                    'no electrical compartment attachment, local APL dynamics, or learning demonstrated.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'complete','contacts':len(contacts),'all_raw_pairs':len(actual),'exact_correspondence':True}),flush=True)


if __name__=='__main__':main()
