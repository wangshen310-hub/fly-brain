"""Frozen full-connectome conductance counterfactual, not a learning trial.

Settings are declared before loading. Same existing contact weights, sensory
adapter and raw spike readout; membrane/refractory dynamics explicitly differ.
Two APL->KC gains match previous diagnostic conditions, plus motor-zero control.
No outcome-based parameter selection, object coordinates, reward or game score.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.conductance_native import ConductanceNativeBrain, ConductanceConfig, VERSION
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays


def sources():
    result=parent_sources()
    for name in ('scripts/assay_native_conductance.py', 'flysnake/research/input_isolation.py',
                 'flysnake/research/conductance_native.py',
                 'flysnake/research/kernels/conductance_diagnostic.cpp', 'vendor/stonkfly/neural/state.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve previous conductance diagnostic')
        source=sources()
        report={'protocol':VERSION,'status':'loading','sources':source,'process_id':os.getpid(),
            'started_at':time.time(),'rows':[],'learning':False,'game_played':False,
            'settings':{'EE_mV':0.,'EI_mV':-70.,'pre_ms':300,'cue_ms':300,'wash_ms':3000,
                        'bin_ms':100,'APL_KC_gain_motor_mV':[[1.,12.],[2.,12.],[1.,0.]],
                        'gamma1_fraction_intervention':.8,
                        'parameter_fit':'None; engineering hypotheses, no measured adult MBON reversal claimed.'}}
        for name in source:
            target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        atomic_json(path,report)
        try:
            brain=ConductanceNativeBrain(args.output/'build'); b=brain.b
            full_base=b.weight.copy(); report['brain']=brain.identity
            unique_pre=np.unique(brain.pre)
            apl=np.array([np.flatnonzero(b.ids==cell)[0] for cell in (10540,10977)],np.int32)
            apl_all_edges=np.concatenate([np.arange(b.ptr[i],b.ptr[i+1],dtype=np.int64) for i in apl])
            apl_kc_edges=apl_all_edges[b.circuit['kc_mask'][b.post[apl_all_edges]].astype(bool)]
            if (full_base[apl_kc_edges]>=0).any():raise AssertionError('APL->KC sign identity differs')
            target=brain.memory_readouts['gamma1_MBON11']
            motor=np.r_[brain.groups['left'],brain.groups['right']]
            selected=np.unique(np.r_[unique_pre,apl,*brain.memory_readouts.values(),motor])
            phase_totals=[]; samples=[]; membranes=[]
            for apl_gain,motor_current in ((1.,12.),(2.,12.),(1.,0.)):
                b.weight[:]=full_base; b.weight[apl_kc_edges]*=np.float32(apl_gain)
                b.tonic[motor]=motor_current; expected_hash=digest(b.weight)
                condition={'APL_KC_gain':apl_gain,'motor_tonic_mV':motor_current,
                           'initial_all_weights_sha256':expected_hash,'cues':[],'perturbed':[]}
                for cue,rgb,concentration in inputs(brain)[1:]:
                    brain.reset_trial();rows=[]
                    for phase,bins,physical in (('pre',3,np.zeros((2,2))),('cue',3,concentration),
                                               ('wash',30,np.zeros((2,2)))):
                        total=np.zeros(b.n,np.int32)
                        for index in range(bins):
                            before=time.perf_counter()
                            receipt=brain.integrate(rgb,100,physical,learning=False)
                            c=brain.last_counts.copy();total+=c;samples.append(c[selected])
                            membranes.append(np.c_[b.v[target],b.conductance_E[target],b.conductance_I[target]])
                            if phase!='cue':
                                expected=np.zeros(b.n,np.float32);expected[motor]=motor_current
                                if not np.array_equal(b.drive,expected):raise AssertionError('Undeclared wash drive')
                            rows.append({'phase':phase,'bin':index+1,'receipt':receipt,
                                'wall_seconds':time.perf_counter()-before,
                                'gamma1_KC_active':int(np.count_nonzero(c[unique_pre])),
                                'gamma1_KC_spikes':int(c[unique_pre].sum()),'APL_spikes':c[apl].tolist(),
                                'MBON_v_mV':b.v[target].tolist(),'MBON_conductance_E':b.conductance_E[target].tolist(),
                                'MBON_conductance_I':b.conductance_I[target].tolist()})
                            # For these nonadapting MBONs with nonnegative drive,
                            # passive/inhibitory targets cannot fall below EI.
                            if (b.v[target]<-70.001).any():raise AssertionError('MBON inhibitory reversal violated')
                            report.update(status='running',phase=f'{apl_gain}/{motor_current}/{cue}/{phase}/{index+1}',
                                          active_rows=rows,updated_at=time.time());atomic_json(path,report)
                        phase_totals.append(total)
                        print(json.dumps({'APL_KC_gain':apl_gain,'motor_mV':motor_current,'cue':cue,'phase':phase,
                            'gamma1_KC_spikes':int(total[unique_pre].sum()),'MBON':total[target].tolist(),
                            'PPL101':total[brain.memory_readouts['PPL101']].tolist()}),flush=True)
                    cued=[r for r in rows if r['phase']=='cue'];tail=[r for r in rows if r['phase']=='wash'][-5:]
                    condition['cues'].append({'cue':cue,'rows':rows,
                        'cue_gamma1_KC_spikes':sum(r['gamma1_KC_spikes'] for r in cued),
                        'cue_MBON11_spikes':np.sum([r['receipt']['readouts']['gamma1_MBON11'] for r in cued],axis=0).tolist(),
                        'cue_PPL101_spikes':np.sum([r['receipt']['readouts']['PPL101'] for r in cued],axis=0).tolist(),
                        'last_500ms_gamma1_KC_spikes':sum(r['gamma1_KC_spikes'] for r in tail),
                        'last_500ms_PPL101_spikes':np.sum([r['receipt']['readouts']['PPL101'] for r in tail],axis=0).tolist(),
                        'cue_MBON_boundary_min_v_mV':min(min(r['MBON_v_mV']) for r in cued)})
                if digest(b.weight)!=expected_hash:raise AssertionError('Frozen diagnostic changed native weights')
                baseline=[]; altered=[]
                for cue,rgb,c in inputs(brain)[1:]:
                    brain.reset_trial();baseline.append(brain.integrate(rgb,300,c,learning=False))
                brain.set_fraction(np.full(len(brain.edges),.8))
                for cue,rgb,c in inputs(brain)[1:]:
                    brain.reset_trial();altered.append(brain.integrate(rgb,300,c,learning=False))
                brain.reset_memory()
                condition['perturbed']=[{'cue':cue,'before':a,'after':z} for cue,a,z in
                                        zip(('DM1','DM4'),baseline,altered)]
                condition['gates']={
                    'odor_reaches_gamma1_KC':all(c['cue_gamma1_KC_spikes']>0 for c in condition['cues']),
                    'odor_reaches_gamma1_MBON':all(sum(c['cue_MBON11_spikes'])>0 for c in condition['cues']),
                    'postcue_returns_to_silent_assay_baseline':all(c['last_500ms_gamma1_KC_spikes']==0 and
                        not any(c['last_500ms_PPL101_spikes']) for c in condition['cues']),
                    'gamma1_intervention_changes_MBON':any(a['readouts']['gamma1_MBON11']!=z['readouts']['gamma1_MBON11']
                        for a,z in zip(baseline,altered)),
                    'gamma1_intervention_changes_DNa02_counts':any(any(a['counts'][k]!=z['counts'][k]
                        for k in ('left','right')) for a,z in zip(baseline,altered))}
                condition['all_gates_pass']=all(condition['gates'].values())
                condition['all_frozen_weights_restored']=digest(b.weight)==expected_hash
                report['rows'].append(condition);report.pop('active_rows',None);atomic_json(path,report)
                print(json.dumps({'configuration':[apl_gain,motor_current],'gates':condition['gates']}),flush=True)
            report['native_spikes']={'file':'native-states.npz','sha256':immutable_arrays(
                args.output/'native-states.npz',phase_counts=np.array(phase_totals),selected_counts=np.array(samples),
                mbon_state=np.array(membranes),selected_indices=selected,cell_ids=b.ids,
                gamma1_KC_indices=unique_pre,mbon_ids=b.ids[target])}
            report.update(status='complete',completed_at=time.time(),
                interpretation='Frozen conductance counterfactual; does not reconstruct nonspiking/local APL or prove learning.')
            atomic_json(path,report);print(json.dumps({'status':'complete'}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
