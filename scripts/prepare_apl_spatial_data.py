"""Acquire official APL morphology/contacts, verify checksums and graph counts.

Anonymous public GCS reads only. Never reads credentials, modifies the neural
model, starts gameplay, or changes old memories. Partner cache is retained.
An interrupted download resumes only after source identity and chunk hashes
are verified. Published coordinates are retained even for excluded partners.
"""
from pathlib import Path
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import threading
import time
import traceback
from urllib.request import Request, urlopen
import google_crc32c
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.fs as pafs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.research.apl_spatial_data import VERSION,APL_IDS,parse_swc,indices_for_ids,compare_contacts_with_graph
from flysnake.research.paired import writer_lock
from flysnake.lab.experiment import atomic_json

BUCKET='flyem-male-cns'
PARTNER='v1.0/connectome-data/flat-connectome/syn-partners-male-cns-v1.0-minconf-0.5.feather'
SWC='v1.0/segmentation/skeletons-malecns/skeletons-swc/'
OFFICIAL='https://male-cns.janelia.org/download'


def source_metadata(relative):
    url=f'https://storage.googleapis.com/{BUCKET}/{relative}'
    with urlopen(Request(url,method='HEAD'),timeout=30) as response:
        hashes=dict(item.strip().split('=',1) for item in response.headers.get('x-goog-hash','').split(',') if '=' in item)
        result={'url':url,'gcs_path':f'{BUCKET}/{relative}','bytes':int(response.headers['Content-Length']),
                'generation':response.headers.get('x-goog-generation'),
                'crc32c_base64':hashes.get('crc32c'),'etag':response.headers.get('ETag'),
                'last_modified':response.headers.get('Last-Modified')}
    if not result['generation'] or not result['crc32c_base64']:raise ValueError('Missing official source generation/checksum')
    return result


def checksum(path,expected_crc):
    sha=hashlib.sha256();crc=google_crc32c.Checksum()
    with path.open('rb') as stream:
        while data:=stream.read(32*1024*1024):sha.update(data);crc.update(data)
    value=base64.b64encode(crc.digest()).decode('ascii')
    if value!=expected_crc:raise ValueError(f'Official CRC32C mismatch: {path.name}')
    return {'sha256':sha.hexdigest(),'crc32c_base64':value,'official_crc32c_verified':True}


def immutable_npz(path,arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                raise ValueError('Existing morphology artifact differs')
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_partner(fs,output,report,save,workers,chunk_size):
    source=report['sources_remote']['partners'];cache=output/'cache';cache.mkdir(exist_ok=True)
    complete=cache/'syn-partners.feather';partial=cache/'syn-partners.feather.partial'
    if complete.exists():
        if complete.stat().st_size!=source['bytes']:raise ValueError('Existing partner cache size differs')
        record=checksum(complete,source['crc32c_base64'])
        report['partner_cache']={'file':'cache/syn-partners.feather',**record};save();return complete
    if not partial.exists():
        if shutil.disk_usage(output).free<source['bytes']+2*1024**3:raise OSError('Insufficient space for official partner cache')
        with partial.open('xb') as stream:stream.truncate(source['bytes'])
    if partial.stat().st_size!=source['bytes']:raise ValueError('Partial cache size differs')
    chunks=(source['bytes']+chunk_size-1)//chunk_size
    completed=report.setdefault('completed_chunks',{})
    # Verify prior checkpointed chunks, not the apparent sparse file size.
    with partial.open('rb') as stream:
        for key,sha in completed.items():
            start=int(key)*chunk_size;length=min(chunk_size,source['bytes']-start)
            if length<=0:raise ValueError('Invalid checkpointed download offset')
            stream.seek(start)
            if hashlib.sha256(stream.read(length)).hexdigest()!=sha:raise ValueError('Checkpointed download chunk changed')
    local=threading.local();write_lock=threading.Lock()
    with partial.open('r+b') as destination:
        def transfer(index):
            if not hasattr(local,'stream'):local.stream=fs.open_input_file(source['gcs_path'])
            start=index*chunk_size;length=min(chunk_size,source['bytes']-start)
            data=local.stream.read_at(length,start)
            if len(data)!=length:raise IOError('Incomplete public GCS range')
            sha=hashlib.sha256(data).hexdigest()
            with write_lock:
                destination.seek(start);destination.write(data);destination.flush();os.fsync(destination.fileno())
            return index,sha
        report.update(status='running',phase='partner-download',total_chunks=chunks);save()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(transfer,i) for i in range(chunks) if str(i) not in completed]
            last_print=0.
            for future in as_completed(futures):
                index,sha=future.result();completed[str(index)]=sha
                report['downloaded_bytes']=sum(min(chunk_size,source['bytes']-int(i)*chunk_size) for i in completed)
                report['updated_at']=time.time();save()
                if time.monotonic()-last_print>=10 or len(completed)==chunks:
                    print(json.dumps({'phase':'partner-download','completed_chunks':len(completed),
                                      'total_chunks':chunks,'downloaded_bytes':report['downloaded_bytes']}),flush=True)
                    last_print=time.monotonic()
    report.update(phase='partner-checksum',updated_at=time.time());save()
    record=checksum(partial,source['crc32c_base64'])
    if source_metadata(PARTNER)!=source:raise ValueError('Official partner object changed during download')
    partial.rename(complete)
    report['partner_cache']={'file':'cache/syn-partners.feather',**record};save()
    return complete


def prepare_skeletons(fs,output,report,save):
    for body in APL_IDS:
        key=str(body);source=report['sources_remote'][key];path=output/f'{body}.swc'
        if not path.exists():
            with fs.open_input_file(source['gcs_path']) as stream:data=stream.read()
            if len(data)!=source['bytes']:raise ValueError('Incomplete skeleton download')
            with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        record=checksum(path,source['crc32c_base64'])
        if source_metadata(SWC+f'{body}.swc')!=source:raise ValueError('Official skeleton object changed')
        tree=parse_swc(path.read_text(encoding='utf-8'))
        arrays={k:v for k,v in tree.items() if k!='summary'}
        record.update(file=path.name,normalized_file=f'apl-{body}-skeleton.npz',
                      normalized_sha256=immutable_npz(output/f'apl-{body}-skeleton.npz',arrays),**tree['summary'])
        report.setdefault('skeletons',{})[key]=record;save()
        print(json.dumps({'phase':'skeleton','body_id':body,'nodes':record['nodes'],'components':record['components']}),flush=True)


def extract_contacts(cache,output,report,save):
    report.update(phase='APL-contact-filter',updated_at=time.time());save()
    with pa.memory_map(str(cache),'r') as stream:
        reader=pa.ipc.open_file(stream)
        body_columns=[reader.schema.get_field_index(k) for k in ('body_pre','body_post')]
        body_reader=pa.ipc.open_file(stream,options=pa.ipc.IpcReadOptions(included_fields=body_columns))
        batches=[];row_offset=0
        for i in range(reader.num_record_batches):
            body_batch=body_reader.get_batch(i)
            pre=body_batch.column(0).to_numpy();post=body_batch.column(1).to_numpy()
            mask=np.isin(pre,APL_IDS)|np.isin(post,APL_IDS)
            if mask.any():
                selected=reader.get_batch(i).filter(pa.array(mask))
                selected=selected.append_column('global_partner_row',pa.array(np.flatnonzero(mask).astype(np.uint64)+row_offset))
                batches.append(selected)
            row_offset+=body_batch.num_rows
            if (i+1)%100==0 or i+1==reader.num_record_batches:
                report.update(scanned_partner_batches=i+1,total_partner_batches=reader.num_record_batches,
                              scanned_partner_rows=row_offset,updated_at=time.time());save()
                if (i+1)%500==0 or i+1==reader.num_record_batches:
                    print(json.dumps({'phase':'APL-contact-filter','batches':i+1,'rows_scanned':row_offset}),flush=True)
    if not batches:raise ValueError('No actual APL partner contacts found')
    table=pa.Table.from_batches(batches)
    graph_ids=np.load(ROOT/'data/malecns_v1/normalized/neuron_ids.npy').astype(np.int64)
    with np.load(ROOT/'data/malecns_v1/graph.npz',allow_pickle=False) as z:
        if not np.array_equal(graph_ids,z['ids']):raise ValueError('Normalized and native graph identity order differs')
    if len(graph_ids)!=166700:raise ValueError('Different native graph coverage')
    pre=indices_for_ids(table['body_pre'].to_numpy(),graph_ids)
    post=indices_for_ids(table['body_post'].to_numpy(),graph_ids)
    table=table.append_column('pre_native_index',pa.array(pre)).append_column('post_native_index',pa.array(post))
    retained=(pre>=0)&(post>=0)
    table=table.append_column('retained_native_contact',pa.array(retained))
    metadata=dict(table.schema.metadata or {})
    metadata.update({b'apl_spatial_protocol':VERSION.encode(),b'coordinate_units':b'MaleCNS EM voxels; 1 voxel = 8nm',
                     b'official_source_generation':report['sources_remote']['partners']['generation'].encode(),
                     b'primary_post_scope':b'Published postsynaptic primary ROI; not reassigned to APL compartments'})
    table=table.replace_schema_metadata(metadata)
    target=output/'apl-partners.feather'
    if target.exists():
        if not feather.read_table(target).equals(table,check_metadata=True):raise ValueError('Existing APL contact artifact differs')
    else:
        with target.open('xb') as stream:feather.write_feather(table,stream,compression='zstd');stream.flush();os.fsync(stream.fileno())
    report['APL_contacts']={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                            'rows':len(table),'source_rows_scanned':row_offset,
                            'retained_rows':int(retained.sum()),'excluded_partner_rows':int((~retained).sum())}
    save();return table,graph_ids


def verify_correspondence(table,ids,report,save):
    report.update(phase='graph-correspondence',updated_at=time.time());save()
    apl_indices=indices_for_ids(APL_IDS,ids)
    neuron_table=feather.read_table(ROOT/'data/malecns_v1/normalized/neurons.feather')
    info=neuron_table.to_pandas().set_index('source_id').loc[list(APL_IDS)]
    if not info.cell_type.eq('APL').all() or not info.neurotransmitter.str.lower().eq('gaba').all():
        raise ValueError('APL identity/transmitter differs in native graph')
    expected_pre=[];expected_post=[];expected_count=[]
    edges_path=ROOT/'data/malecns_v1/normalized/edges.arrow'
    with pa.memory_map(str(edges_path),'r') as stream:
        reader=pa.ipc.open_file(stream)
        for i in range(reader.num_record_batches):
            batch=reader.get_batch(i);pre=batch.column(0).to_numpy();post=batch.column(1).to_numpy()
            mask=np.isin(pre,apl_indices)|np.isin(post,apl_indices)
            if mask.any():
                expected_pre.append(pre[mask]);expected_post.append(post[mask]);expected_count.append(batch.column(2).to_numpy()[mask])
    pre=table['pre_native_index'].to_numpy();post=table['post_native_index'].to_numpy()
    comparison=compare_contacts_with_graph(pre,post,np.concatenate(expected_pre),np.concatenate(expected_post),
                                           np.concatenate(expected_count),len(ids))
    report['graph_correspondence']=comparison
    pre_body=table['body_pre'].to_numpy();post_body=table['body_post'].to_numpy()
    for body in APL_IDS:
        outgoing=pre_body==body;incoming=post_body==body
        xyz_pre=np.c_[*[table[k].to_numpy()[outgoing] for k in ('x_pre','y_pre','z_pre')]]*.008
        xyz_post=np.c_[*[table[k].to_numpy()[incoming] for k in ('x_post','y_post','z_post')]]*.008
        report.setdefault('contact_summary',{})[str(body)]={
            'all_outgoing_contacts':int(outgoing.sum()),'all_incoming_contacts':int(incoming.sum()),
            'retained_outgoing_contacts':int(np.count_nonzero(outgoing&(post>=0))),
            'retained_incoming_contacts':int(np.count_nonzero(incoming&(pre>=0))),
            'unique_APL_presynaptic_locations':len(np.unique(xyz_pre,axis=0)),
            'unique_APL_postsynaptic_locations':len(np.unique(xyz_post,axis=0)),
            'presynaptic_bounds_um':[xyz_pre.min(axis=0).tolist(),xyz_pre.max(axis=0).tolist()],
            'postsynaptic_bounds_um':[xyz_post.min(axis=0).tolist(),xyz_post.max(axis=0).tolist()]}
    save()
    if not comparison['equal']:raise ValueError('Spatial partner contacts do not reproduce retained APL graph counts')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--chunk-mib',type=int,default=16)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not 1<=args.workers<=12 or not 4<=args.chunk_mib<=64:raise ValueError('Bounded download resources required')
    args.output.mkdir(parents=True,exist_ok=True)
    source_names=('scripts/prepare_apl_spatial_data.py','flysnake/research/apl_spatial_data.py','requirements-apl-spatial.txt')
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():
            if not args.resume:raise ValueError('Use --resume for an existing spatial preparation')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=sources or report['settings']!={'workers':args.workers,'chunk_mib':args.chunk_mib}:
                raise ValueError('Preserve source and chunk layout across a resume')
        else:
            report={'protocol':VERSION,'status':'starting','phase':'public-source-metadata','started_at':time.time(),
                    'sources':sources,'settings':{'workers':args.workers,'chunk_mib':args.chunk_mib},
                    'official_description':OFFICIAL,'neural_model_modified':False,'game_played':False,
                    'dependencies':{'pyarrow':pa.__version__,'google_crc32c':importlib.metadata.version('google-crc32c')}}
            for name in source_names:
                target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        def save():report['process_id']=os.getpid();report['updated_at']=time.time();atomic_json(path,report)
        save()
        try:
            remote={'partners':source_metadata(PARTNER),**{str(body):source_metadata(SWC+f'{body}.swc') for body in APL_IDS}}
            if 'sources_remote' in report and report['sources_remote']!=remote:raise ValueError('Official objects changed since earlier acquisition')
            report['sources_remote']=remote;save()
            fs=pafs.GcsFileSystem(anonymous=True,retry_time_limit=timedelta(seconds=30))
            prepare_skeletons(fs,args.output,report,save)
            cache=download_partner(fs,args.output,report,save,args.workers,args.chunk_mib*1024**2)
            table,ids=extract_contacts(cache,args.output,report,save)
            verify_correspondence(table,ids,report,save)
            report.update(status='complete',phase='verified-spatial-data',completed_at=time.time(),
                limitations=['Published SWC is coarse; it is not a measured electrical compartment model.',
                             'No contact-to-cable attachment or graded APL dynamics fitted in this data stage.',
                             'Primary_post is the source postsynaptic ROI, not an invented presynaptic compartment.',
                             'Spatial data/correspondence do not demonstrate associative learning or Snake learning.'])
            save();print(json.dumps({'status':'complete','APL_contacts':report['APL_contacts'],
                                    'graph_correspondence':report['graph_correspondence']}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
