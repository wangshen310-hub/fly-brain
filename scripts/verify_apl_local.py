"""Fresh-process state replay and independent completed local APL evidence audit."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import numpy as np
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.apl_local import LocalAPLBrain, LocalAPLConfig, file_hash
from flysnake.research.apl_local_state_v2 import load as load_state_v2
from flysnake.research.paired import digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import inputs


def independent_geometry_sample(run,arrays):
    """Brute-force all actual segments for 24 held-out actual contact positions.

    This intentionally does not call the BVH mapper and admits shared endpoints.
    The sample is not proof that coarse skeleton attachment is biologically exact.
    """
    data=ROOT/'runtime/apl-spatial-20261002-v1';t=feather.read_table(data/'apl-partners.feather')
    rows=t['global_partner_row'].to_numpy();results=[]
    for body in (10540,10977):
        with np.load(data/f'apl-{body}-skeleton.npz',allow_pickle=False) as z:
            xyz=z['xyz_um'];parent=z['parent_index'];child=np.flatnonzero(parent>=0);p=parent[child]
            a=xyz[p];direction=xyz[child]-a;denominator=np.sum(direction**2,axis=1)
        for side,selected_global,endpoint_key,alpha_key in (
            ('post',arrays['incoming_global_rows'],'in_child','in_alpha'),
            ('pre',arrays['outgoing_global_rows'],'out_child','out_alpha')):
            matching=np.flatnonzero(t[f'body_{side}'].to_numpy()==body)
            matching=matching[np.isin(rows[matching],selected_global)]
            chosen=matching[np.linspace(0,len(matching)-1,6).astype(int)]
            for original_index in chosen:
                global_row=int(rows[original_index]);index=int(np.flatnonzero(selected_global==global_row)[0])
                point=np.array([t[f'{axis}_{side}'][int(original_index)].as_py() for axis in 'xyz'])*.008
                fraction=np.clip(np.sum((point-a)*direction,axis=1)/denominator,0,1)
                projected=a+fraction[:,None]*direction;distances=np.linalg.norm(point-projected,axis=1)
                c=int(arrays[endpoint_key][index]);parent_key='in_parent' if side=='post' else 'out_parent'
                mapped_parent=int(arrays[parent_key][index]);alpha=float(arrays[alpha_key][index])
                mapped_point=arrays['xyz_um'][mapped_parent]+alpha*(arrays['xyz_um'][c]-arrays['xyz_um'][mapped_parent])
                mapped_distance=float(np.linalg.norm(point-mapped_point));minimum=float(distances.min())
                if abs(mapped_distance-minimum)>1e-10:raise AssertionError('BVH attachment disagrees with exhaustive actual morphology')
                if arrays['parent'][c]!=mapped_parent:raise AssertionError('Mapped attachment is not an actual cable edge')
                results.append({'body':body,'side':side,'global_row':global_row,'distance_um':mapped_distance,
                                'independent_exhaustive_minimum_um':minimum})
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-run',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous independent local APL verification')
    status_path=args.run/'status.json';status=json.loads(status_path.read_text(encoding='utf-8'))
    initial_status_sha=file_hash(status_path)
    if status['status']!='complete':raise ValueError('Only verify an actually completed native assay')
    if status['process_id']==os.getpid():raise ValueError('A distinct OS process is required for replay')
    for name,sha in status['sources'].items():
        if file_hash(ROOT/name)!=sha or file_hash(args.run/'sources'/name)!=sha:raise ValueError('Native assay source differs')
    record=status['native_arrays']
    if file_hash(args.run/record['file'])!=record['sha256']:raise ValueError('Actual native evidence arrays changed')
    geometry=args.run/'geometry';mapping=json.loads((geometry/'attachment.json').read_text(encoding='utf-8'))
    if file_hash(geometry/'attachment.npz')!=mapping['attachment_sha256']:raise ValueError('Attachment arrays changed')
    with np.load(geometry/'attachment.npz',allow_pickle=False) as z:attachment={name:z[name].copy() for name in z.files}
    sampled=independent_geometry_sample(args.run,attachment)
    actual_rows=[]
    with np.load(args.run/record['file'],allow_pickle=False) as z:
        counts=z['full_counts'];pre=z['gamma1_KC_indices'];mbon=z['mbon_indices'];dan=z['dan_indices']
        ids=z['cell_ids'];apl=np.flatnonzero(np.isin(ids,[10540,10977]))
        if np.any(counts[:,apl]):raise AssertionError('Nonspiking APL has actual point spikes')
        for condition in status['rows']:
            cues=[]
            for cue in condition['cues']:
                cue_counts=np.array([counts[r['native_count_array_index']] for r in cue['rows'] if r['phase']=='cue']).sum(axis=0)
                tail=np.array([counts[r['native_count_array_index']] for r in cue['rows'] if r['phase']=='wash'][-5:]).sum(axis=0)
                for row in cue['rows']:
                    if digest(counts[row['native_count_array_index']])!=row['receipt']['full_counts_sha256']:
                        raise AssertionError('Receipt differs from actual full-brain count array')
                if (int(cue_counts[pre].sum())!=cue['cue_KC_spikes'] or cue_counts[mbon].tolist()!=cue['cue_MBON']
                        or int(tail[pre].sum())!=cue['wash_tail_KC'] or tail[dan].tolist()!=cue['wash_tail_DAN']):
                    raise AssertionError('Summarized mechanism result differs from actual counts')
                cues.append({'cue':cue['cue'],'actual_cue_KC':int(cue_counts[pre].sum()),
                    'actual_cue_MBON':cue_counts[mbon].tolist(),'actual_tail_KC':int(tail[pre].sum()),
                    'actual_tail_DAN':tail[dan].tolist()})
            actual_rows.append({'label':condition['label'],'cues':cues,'gates':condition['gates']})
        native_APL_is_nonspiking=True
        cue_vectors=[]
        for cue in status['rows'][0]['cues']:
            indices=[r['native_count_array_index'] for r in cue['rows'] if r['phase']=='cue']
            cue_vectors.append(counts[indices][:,pre].sum(axis=0).astype(float))
        similarity=float(cue_vectors[0]@cue_vectors[1]/max(np.linalg.norm(cue_vectors[0])*np.linalg.norm(cue_vectors[1]),1e-15))
    # A new interpreter loads the same entire graph and exact mapping. Restore
    # nonzero cable/receptor/queue/eligibility state, not just trial-boundary W.
    # Load the exact immutable compiled bundle whose build identity is part of
    # the checkpoint. Recompiling to another path is a different build identity.
    brain=LocalAPLBrain(args.run,attachment=geometry)
    contacts=feather.read_table(ROOT/'runtime/apl-spatial-20261002-v1/apl-partners.feather')
    contacts=contacts.filter(contacts['retained_native_contact'])
    pre=contacts['pre_native_index'].to_numpy();post=contacts['post_native_index'].to_numpy()
    native_edges=attachment['all_contact_edge'];b=brain.b
    if (not np.array_equal(attachment['retained_global_rows'],contacts['global_partner_row'].to_numpy())
            or not np.array_equal(b.post[native_edges],post)
            or np.any(native_edges<b.ptr[pre]) or np.any(native_edges>=b.ptr[pre+1])):
        raise AssertionError('Actual contact-to-edge reconstruction differs')
    edge_counts=np.bincount(native_edges,minlength=len(b.weight))
    unique_edges=np.unique(native_edges)
    if (len(unique_edges)!=12027 or len(native_edges)!=443915
            or not np.allclose(np.abs(b.weight[unique_edges]),edge_counts[unique_edges]*.275,atol=1e-4,rtol=1e-7)):
        raise AssertionError('Contact mapping no longer reproduces retained original edge strengths')
    state_path=args.state_run/'status.json';state=json.loads(state_path.read_text(encoding='utf-8'))
    if state['status']!='complete' or state['parent_status_sha256']!=file_hash(status_path):
        raise ValueError('Completed corresponding v2 state protocol required; failed v1 state is not evidence')
    for name,sha in state['sources'].items():
        if file_hash(ROOT/name)!=sha or file_hash(args.state_run/'sources'/name)!=sha:raise ValueError('V2 state source differs')
    checkpoint=state['checkpoint'];load_state_v2(brain,args.state_run/checkpoint['file'],checkpoint['sha256'])
    _,rgb,concentration=inputs(brain)[1];receipt=brain.integrate(rgb,100,concentration)
    reference=state['continuation_receipt'];actual={name:digest(getattr(brain.b,name)) for name in brain.b.fields}
    expected=state['continuation_state']
    for key in ('full_counts_sha256','counts','action','readouts','memory_before','memory_after','APL_local_u_mV_quantiles'):
        if receipt[key]!=reference[key]:raise AssertionError('Fresh-process actual neural replay differs: '+key)
    if (actual!=expected['native'] or digest(brain.r8_light)!=expected['adapter_r8_light']
            or digest(brain.retinal_light)!=expected['adapter_retinal_light']):
        raise AssertionError('Fresh-process complete dynamic-state replay differs')
    if file_hash(status_path)!=initial_status_sha:raise AssertionError('Assay record changed during audit')
    result={'version':'independent-native-local-APL-audit-replay-v1','status':'complete','generated_at':time.time(),
        'process_id':os.getpid(),'assay_process_id':status['process_id'],'verification_source_sha256':file_hash(__file__),
        'assay_status_sha256':file_hash(status_path),'actual_native_counts_verified':True,'sampled_exact_segment_projections':sampled,
        'all_443915_contacts_reconstruct_12027_existing_edge_weights':True,
        'actual_point_APL_spikes_zero':native_APL_is_nonspiking,'actual_results':actual_rows,
        'first_configuration_gamma1_300ms_count_cosine_DM1_DM4':similarity,
        'fresh_process_full_dynamic_state_replay':True,'replay_observation_ms':100,
        'state_v2_status_sha256':file_hash(state_path),'failed_v1_checkpoint_used_as_evidence':False,
        'replayed_dynamic_array_count':len(actual),'replayed_all_dynamic_arrays':actual,
        'all_replayed_weights_sha256':digest(brain.b.weight),'learning_validated':False,
        'scope':'Native local APL integration, geometry sample, actual frozen counts and nonzero-state replay. '
                'No fitted adult physiology or successful reward learning inferred.'}
    atomic_json(args.output,result);print(json.dumps({'status':'complete','replay':True,'geometry_sample':len(sampled),
        'actual_neural_arrays':True,'DM1_DM4_cosine':similarity}),flush=True)


if __name__=='__main__':main()
