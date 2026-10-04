"""Targeted postcue silencing and early-time coding in the complete local-APL graph.

Currents are diagnostic interventions, not fitted biological currents and never
a controller. Each intervention starts after the same unmodified cue history.
Only one full brain is loaded; evidence arrays are compressed per condition.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.apl_local import LocalAPLBrain,file_hash
from flysnake.research.recurrence_protocol import anatomical_groups,cue_windows
from flysnake.research.paired import digest,writer_lock
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import inputs,immutable_arrays


def potential_budget(b,target,counts,groups,types):
    mask=np.zeros(b.n,bool);mask[target]=True;edges=np.flatnonzero(mask[b.post])
    pre=(np.searchsorted(b.ptr,edges,side='right')-1).astype(np.int32)
    value=counts[pre].astype(float)*b.weight[edges]
    fast=~b.modulation_mask[pre].astype(bool)
    summary={}
    for name,ix in groups.items():
        selected=np.isin(pre,ix)&fast
        summary[name]={'positive':float(value[selected&(value>0)].sum()),'negative':float(value[selected&(value<0)].sum())}
    active=fast&(value!=0);unique,inverse=np.unique(types[pre[active]],return_inverse=True)
    positive=np.bincount(inverse,weights=np.maximum(value[active],0),minlength=len(unique))
    negative=np.bincount(inverse,weights=np.minimum(value[active],0),minlength=len(unique))
    top_positive=np.argsort(positive)[-12:][::-1];top_negative=np.argsort(negative)[:12]
    return {'groups':summary,'all_fast_positive':float(value[fast&(value>0)].sum()),
        'all_fast_negative':float(value[fast&(value<0)].sum()),
        'largest_positive_types':[[str(unique[i]),float(positive[i])] for i in top_positive if positive[i]>0],
        'largest_negative_types':[[str(unique[i]),float(negative[i])] for i in top_negative if negative[i]<0],
        'scope':'Spikes times original weight: potential budget, not receptor current. Delay/refractory rejection not reconstructed. '
                'Graded APL does not have point spikes and is measured separately; modulator edges excluded.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve previous recurrence experiment')
        parent=json.loads((args.bundle/'status.json').read_text(encoding='utf-8'))
        if parent['status']!='complete':raise ValueError('Completed local APL source bundle required')
        source=dict(parent['sources'])
        for name in ('scripts/assay_apl_recurrence.py','flysnake/research/recurrence_protocol.py',
                     'vendor/stonkfly/neural/transmitters.py'):
            source[name]=file_hash(ROOT/name)
        for name,sha in parent['sources'].items():
            if file_hash(ROOT/name)!=sha:raise ValueError('Original source changed')
        settings=[('baseline-DM1','DM1',12.,None),('baseline-DM4','DM4',12.,None),
            ('motor-zero-DM1','DM1',0.,None),('motor-zero-DM4','DM4',0.,None),
            ('postcue-ALLN-ACh','DM1',12.,'ALLN_ACh'),('postcue-ALLN-uncertain','DM1',12.,'ALLN_uncertain'),
            ('postcue-ALLN','DM1',12.,'ALLN'),('postcue-ALPN','DM1',12.,'ALPN'),
            ('postcue-ORN','DM1',12.,'ORN'),('postcue-KC','DM1',12.,'KC')]
        report={'version':'native-local-APL-recurrence-causal-timecourse-v1','status':'loading','sources':source,
            'process_id':os.getpid(),'started_at':time.time(),'rows':[],
            'parent_status_sha256':file_hash(args.bundle/'status.json'),
            'settings':{'conditions':settings,'pre_ms':300,'cue_ms':300,'cue_bin_ms':10,'wash_ms':3000,
                'wash_bin_ms':100,'silencing_first_wash_ms':200,'silencing_current_mV':-10000.,
                'learning':False,'game_played':False,'fit':'No score or action target. Anatomical diagnostic interventions only.'}}
        for name in source:
            target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        atomic_json(path,report)
        try:
            brain=LocalAPLBrain(args.bundle,attachment=args.bundle/'geometry');b=brain.b
            report['brain']=brain.identity
            a=feather.read_table(ROOT/'data/malecns_v1/annotations.feather',columns=['bodyId','type','class']).to_pandas().set_index('bodyId').loc[b.ids]
            n=feather.read_table(ROOT/'data/malecns_v1/normalized/neurons.feather',columns=['source_id','neurotransmitter']).to_pandas().set_index('source_id').loc[b.ids]
            types=a.type.fillna('unknown').to_numpy(dtype=str)
            groups=anatomical_groups(a['class'].fillna(''),types,n.neurotransmitter.fillna(''),brain.memory_readouts['KC'])
            groups.update(gamma1_KC=np.unique(brain.pre),PPL101=brain.memory_readouts['PPL101'],MBON=brain.memory_readouts['gamma1_MBON11'])
            del a,n
            if any(not len(groups[g]) for _,_,_,g in settings if g):raise ValueError('Declared anatomical clamp group is empty')
            report['groups']={name:{'cells':len(ix),'native_ids':b.ids[ix].tolist(),'indices_sha256':digest(ix)} for name,ix in groups.items()}
            motor=np.r_[brain.groups['left'],brain.groups['right']];cue_records={};baseline_history=None
            original_tonic=b.tonic.copy()
            for index,(label,cue,motor_mV,clamp) in enumerate(settings):
                b.tonic[:]=original_tonic;b.tonic[motor]=motor_mV;brain.reset_trial()
                rgb,physical=[(rgb,c) for name,rgb,c in inputs(brain) if name==cue][0]
                rows=[];counts=[];local_summary=[]
                for phase,bins,duration,concentration in (('pre',3,100,np.zeros((2,2))),('cue',30,10,physical),('wash',30,100,np.zeros((2,2)))):
                    for bin_number in range(bins):
                        intervention=clamp if phase=='wash' and bin_number<2 else None
                        stimulation=[(groups[intervention],-10000.)] if intervention else None
                        receipt=brain.integrate(rgb,duration,concentration,stimulation=stimulation,learning=False)
                        c=brain.last_counts.copy();counts.append(c)
                        if phase!='cue':
                            expected=np.zeros(b.n,np.float32);expected[motor]=motor_mV
                            if intervention:expected[groups[intervention]]+=-10000.
                            if not np.array_equal(b.drive,expected):raise AssertionError('Undeclared external drive')
                        row={'phase':phase,'bin':bin_number+1,'duration_ms':duration,'receipt':receipt,
                            'array_index':len(counts)-1,'clamp':intervention,
                            'group_spikes':{name:int(c[ix].sum()) for name,ix in groups.items()},
                            'silenced_group_spikes':int(c[groups[intervention]].sum()) if intervention else None}
                        rows.append(row);local_summary.append(receipt['APL_local_u_mV_quantiles'])
                        report.update(status='running',phase=f'{label}/{phase}/{bin_number+1}',active_rows=rows,updated_at=time.time())
                        atomic_json(path,report)
                counts=np.array(counts);cue_counts=counts[3:33];tail=counts[-5:].sum(axis=0)
                cued=cue_counts.sum(axis=0);wash_counts=counts[33:].sum(axis=0)
                history=[r['receipt']['full_counts_sha256'] for r in rows[:33]]
                if label=='baseline-DM1':baseline_history=history
                history_equal=history==baseline_history if clamp else None
                if clamp and not history_equal:raise AssertionError('Causal intervention did not start from identical pre/cue history')
                result={'label':label,'cue':cue,'motor_mV':motor_mV,'clamp_group':clamp,'rows':rows,
                    'identical_history_before_intervention':history_equal,
                    'clamp_successful':all(r['silenced_group_spikes']==0 for r in rows if r['clamp']) if clamp else None,
                    'cue_group_spikes':{name:int(cued[ix].sum()) for name,ix in groups.items()},
                    'wash_tail_group_spikes':{name:int(tail[ix].sum()) for name,ix in groups.items()},
                    'wash_tail_full_spikes':int(tail.sum()),'cue_full_spikes':int(cued.sum()),
                    'last500ms_equals_silent_KC_DAN_baseline':int(tail[groups['gamma1_KC']].sum())==0 and int(tail[groups['PPL101']].sum())==0,
                    'potential_budgets':{name:potential_budget(b,groups[name],tail,groups,types) for name in ('gamma1_KC','ALPN','PPL101')},
                    'all_weights_unchanged':digest(b.weight)==brain.initial_all_weights_sha256}
                if not result['all_weights_unchanged']:raise AssertionError('Frozen recurrence diagnostic changed weights')
                result['native_arrays']={'file':f'condition-{index:02d}.npz','sha256':immutable_arrays(args.output/f'condition-{index:02d}.npz',
                    full_counts=counts,cell_ids=b.ids,local_u_quantiles=np.array(local_summary))}
                if not clamp:cue_records[label]=cue_counts.copy()
                report['rows'].append(result);report.pop('active_rows',None);atomic_json(path,report)
                print(json.dumps({'label':label,'clamp_successful':result['clamp_successful'],
                    'wash_tail_KC':result['wash_tail_group_spikes']['gamma1_KC'],
                    'wash_tail_DAN':result['wash_tail_group_spikes']['PPL101'],
                    'wash_tail_full_spikes':int(tail.sum())}),flush=True)
                del counts
            report['early_time_coding']={
                motor:{name:cue_windows(cue_records[f'{motor}-DM1'],cue_records[f'{motor}-DM4'],groups[name])
                       for name in ('ORN','ALPN','gamma1_KC','MBON')}
                for motor in ('baseline','motor-zero')}
            b.tonic[:]=original_tonic
            report.update(status='complete',completed_at=time.time(),
                interpretation='Causal postcue silencing and actual early count windows. Algebraic diagnostic currents are not '
                    'biological calibration or trained game control. Necessary/sufficient circuit origins require further controls.')
            atomic_json(path,report);print(json.dumps({'status':'complete','early_time_coding':report['early_time_coding']}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
