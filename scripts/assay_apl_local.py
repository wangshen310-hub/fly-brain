"""Anatomical local APL integration, frozen whole-connectome causal validation.

All settings and gates precede native integration. No game scores, direction
labels, outside policy, or candidate return optimizer are used. Failed physiology
is recorded instead of being disguised as successful brain learning.
"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import ctypes as C
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.apl_local import LocalAPLBrain, LocalAPLConfig, VERSION, file_hash, cable_geometry
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as original_sources, inputs, immutable_arrays


def sources():
    result=original_sources()
    for name in ('scripts/assay_apl_local.py','flysnake/research/apl_local.py','flysnake/research/kernels/apl_local.cpp',
                 'flysnake/research/input_isolation.py','vendor/stonkfly/neural/state.py'):
        result[name]=file_hash(ROOT/name)
    return result


def actual_cable_assay(brain):
    """Electrical probe on actual published morphology; no synthetic skeleton."""
    k=brain.kernel;a=k.a;parent=a['parent'];xyz=a['xyz_um'];n=k.local_n
    original_nodes=np.flatnonzero((a['body_ids']==10540)&(a['component']==0))
    possible=a['in_child'][np.isin(a['in_child'],original_nodes)]
    stimulated=int(possible[len(possible)//2]);drive=np.zeros(n);drive[stimulated]=30.
    children=[[] for _ in parent]
    for i,p in enumerate(parent):
        if p>=0:children[p].append(i)
    distance=np.full(n,np.inf);distance[stimulated]=0.;visit=[stimulated]
    for i in visit:
        neighbors=children[i]+([int(parent[i])] if parent[i]>=0 else [])
        for j in neighbors:
            if np.isinf(distance[j]):distance[j]=distance[i]+np.linalg.norm(xyz[j]-xyz[i]);visit.append(j)
    fn=k.library.apl_cable_trace;fn.argtypes=[C.c_int,C.c_int,C.c_double]+[C.c_void_p]*4+[C.c_double]*3+[C.c_void_p]*5;fn.restype=None
    measurements=[];states=[]
    for interval in (.5,.25):
        u=np.zeros(n);ge=np.zeros(n);gi=np.zeros(n);steps=round(200/interval)
        output=np.empty((steps,n))
        fn(n,steps,interval,parent.ctypes.data,k.axial.ctypes.data,k.volume.ctypes.data,k.diagonal_axial.ctypes.data,
           k.config.membrane_tau_ms,52.,-18.,u.ctypes.data,ge.ctypes.data,gi.ctypes.data,drive.ctypes.data,output.ctypes.data)
        near=(distance<=25);far=(distance>=100)&np.isfinite(distance);other=np.isinf(distance)
        measurements.append({'dt_ms':interval,'near_mean_u_mV':float(u[near].mean()),
            'far_mean_u_mV':float(u[far].mean()),'other_component_max_abs_u_mV':float(np.abs(u[other]).max()),
            'stimulated_node_u_mV':float(u[stimulated]),'full_local_state_sha256':digest(u)})
        states.append(u.copy());del output
    relative=float(np.linalg.norm(states[0]-states[1])/max(np.linalg.norm(states[1]),1e-15))
    result={'stimulated_local_index':stimulated,'body':10540,'SWC_node_is_real':True,
        'local_drive_mV':30.,'duration_ms':200,'measurements':measurements,
        'dt_halving_relative_state_difference':relative,
        'locality_pass':all(m['near_mean_u_mV']>m['far_mean_u_mV'] for m in measurements),
        'disconnected_isolation_pass':all(m['other_component_max_abs_u_mV']==0 for m in measurements),
        'numerical_dt_gate_pass':relative<.02,
        'scope':'Passive electrical current probe, not endogenous odor, calcium imaging or learning'}
    return result,np.array(states),distance


def passthrough_assay(brain):
    b=brain.b;k=brain.kernel;native=b.advance
    original=b.library.memory_advance;original.argtypes=native.argtypes;original.restype=None
    saved=[]
    for function in (original,native):
        b.reset(keep_memory=True);k.diagnostic_passthrough=(function is native);b.advance=function
        rows=[]
        for index in range(10):
            groups=brain.orn['ORN_DM1'];stimulation=[(groups['L'],24.),(groups['R'],24.)]
            b._neural_step(np.zeros(len(b.retina)),10,learning=False,stimulation=stimulation,lamina_bias=0.)
            rows.append({name:digest(getattr(b,name)) for name in b.fields if not name.startswith('apl_')})
        saved.append(rows)
    k.diagnostic_passthrough=False;b.advance=native;b.reset(keep_memory=True)
    if saved[0]!=saved[1]:raise AssertionError('Local-kernel passthrough changed archived native dynamics')
    return {'actual_full_graph_boundaries':10,'bin_ms':10,'all_original_dynamic_arrays_bitwise_equal':True,
            'original_kernel_binary_used_as_reference':True,'scope':'Local extension disabled; not local-model equivalence'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve previous local APL assay')
        source=sources();report={'version':VERSION,'status':'loading','process_id':os.getpid(),'started_at':time.time(),
            'sources':source,'rows':[],'game_played':False,'learning':False,
            'predeclared_configurations':[asdict(LocalAPLConfig()),asdict(LocalAPLConfig(space_constant_um=50.)),
                asdict(LocalAPLConfig(release_gain=0.))],
            'predeclared_protocol':{'pre_ms':300,'cue_ms':300,'wash_ms':3000,'bin_ms':100,
                'gamma1_fraction_intervention':.8,'geometry_max_distance_gate_um':5.,
                'fit':'None; constants declared before measurements. No game performance selection.'}}
        for name in source:
            target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        atomic_json(path,report)
        try:
            brain=LocalAPLBrain(args.output);b=brain.b;k=brain.kernel
            report['brain']=brain.identity;report['mapping']=brain.mapping
            report.update(status='running',phase='actual-cable-validation');atomic_json(path,report)
            print(json.dumps({'phase':'mapped','geometry':brain.mapping['geometry']}),flush=True)
            report['cable_assay'],cable_states,cable_distance=actual_cable_assay(brain)
            report['native_passthrough']=passthrough_assay(brain);atomic_json(path,report)
            print(json.dumps({'phase':'cable-and-passthrough','cable':report['cable_assay'],'passthrough':report['native_passthrough']}),flush=True)
            unique_pre=np.unique(brain.pre);mbon=brain.memory_readouts['gamma1_MBON11'];dan=brain.memory_readouts['PPL101']
            full_counts=[];local_states=[];receiver_states=[];mbon_states=[]
            for label,configuration in (('local-25um',LocalAPLConfig()),('local-50um',LocalAPLConfig(space_constant_um=50.)),
                                         ('graded-release-off',LocalAPLConfig(release_gain=0.))):
                configuration.validate();k.config=configuration;brain.apl_config=configuration
                k.volume,k.axial,k.diagonal_axial=cable_geometry(k.a['xyz_um'],k.a['parent'],configuration.space_constant_um)
                condition={'label':label,'config':asdict(configuration),'cues':[],'perturbations':[]}
                baseline={}
                for cue,rgb,concentration in inputs(brain)[1:]:
                    brain.reset_trial();rows=[]
                    for phase,bins,physical in (('pre',3,np.zeros((2,2))),('cue',3,concentration),('wash',30,np.zeros((2,2)))):
                        for index in range(bins):
                            begin=time.perf_counter();receipt=brain.integrate(rgb,100,physical,learning=False)
                            counts=brain.last_counts.copy();full_counts.append(counts);local_states.append(k.u.copy())
                            receiver_states.append(k.inhibition[k.a['targets']].copy());mbon_states.append(b.v[mbon].copy())
                            if phase!='cue':
                                expected=np.zeros(b.n,np.float32);expected[np.r_[brain.groups['left'],brain.groups['right']]]=12.
                                if not np.array_equal(b.drive,expected):raise AssertionError('Undeclared postcue current')
                            row={'phase':phase,'bin':index+1,'receipt':receipt,'wall_seconds':time.perf_counter()-begin,
                                'gamma1_KC_spikes':int(counts[unique_pre].sum()),'gamma1_KC_active':int(np.count_nonzero(counts[unique_pre])),
                                'MBON_v_mV':b.v[mbon].tolist(),'local_u_sha256':digest(k.u),
                                'APL_receiver_conductance_sha256':digest(k.inhibition),
                                'native_count_array_index':len(full_counts)-1}
                            rows.append(row)
                            report.update(status='running',phase=f'{label}/{cue}/{phase}/{index+1}',active_rows=rows,updated_at=time.time())
                            atomic_json(path,report)
                        selected=[r for r in rows if r['phase']==phase]
                        print(json.dumps({'config':label,'cue':cue,'phase':phase,
                            'gamma1_KC':sum(r['gamma1_KC_spikes'] for r in selected),
                            'MBON':np.sum([r['receipt']['readouts']['gamma1_MBON11'] for r in selected],axis=0).tolist(),
                            'PPL101':np.sum([r['receipt']['readouts']['PPL101'] for r in selected],axis=0).tolist(),
                            'APL_u_max':max(r['receipt']['APL_local_u_mV_quantiles'][-1] for r in selected)}),flush=True)
                    cue_rows=[r for r in rows if r['phase']=='cue'];tail=[r for r in rows if r['phase']=='wash'][-5:]
                    baseline[cue]={'MBON':np.sum([r['receipt']['readouts']['gamma1_MBON11'] for r in cue_rows],axis=0).tolist(),
                        'motor':{name:np.sum([r['receipt']['counts'][name] for r in cue_rows],axis=0).tolist() for name in ('left','right')}}
                    condition['cues'].append({'cue':cue,'rows':rows,'cue_KC_spikes':sum(r['gamma1_KC_spikes'] for r in cue_rows),
                        'cue_MBON':baseline[cue]['MBON'],'cue_DAN':np.sum([r['receipt']['readouts']['PPL101'] for r in cue_rows],axis=0).tolist(),
                        'wash_tail_KC':sum(r['gamma1_KC_spikes'] for r in tail),
                        'wash_tail_DAN':np.sum([r['receipt']['readouts']['PPL101'] for r in tail],axis=0).tolist(),
                        'MBON_min_boundary_v_mV':min(min(r['MBON_v_mV']) for r in rows)})
                if digest(b.weight)!=brain.initial_all_weights_sha256:raise AssertionError('Frozen local physiology changed graph weights')
                brain.set_fraction(np.full(len(brain.edges),.8))
                for cue,rgb,concentration in inputs(brain)[1:]:
                    brain.reset_trial();brain.integrate(rgb,300,np.zeros((2,2)))
                    after=brain.integrate(rgb,300,concentration);full_counts.append(brain.last_counts.copy())
                    condition['perturbations'].append({'cue':cue,'before':baseline[cue],
                        'after':{'MBON':after['readouts']['gamma1_MBON11'],
                                 'motor':{name:after['counts'][name] for name in ('left','right')}},
                        'native_count_array_index':len(full_counts)-1})
                brain.reset_memory()
                condition['gates']={'odor_reaches_gamma1_KC':all(c['cue_KC_spikes']>0 for c in condition['cues']),
                    'odor_reaches_gamma1_MBON':all(sum(c['cue_MBON'])>0 for c in condition['cues']),
                    'postcue_returns_to_silent_assay_baseline':all(c['wash_tail_KC']==0 and not any(c['wash_tail_DAN']) for c in condition['cues']),
                    'MBON_boundary_above_assumed_inhibitory_reversal':all(c['MBON_min_boundary_v_mV']>=-70.001 for c in condition['cues']),
                    'gamma1_intervention_changes_MBON':any(c['before']['MBON']!=c['after']['MBON'] for c in condition['perturbations']),
                    'gamma1_intervention_changes_DNa02_counts':any(c['before']['motor']!=c['after']['motor'] for c in condition['perturbations'])}
                condition['all_gates_pass']=all(condition['gates'].values());report['rows'].append(condition)
                report.pop('active_rows',None);atomic_json(path,report)
                print(json.dumps({'config':label,'gates':condition['gates']}),flush=True)
            # Restore declared default candidate before continuation/replay checkpoint.
            k.config=LocalAPLConfig();brain.apl_config=k.config
            k.volume,k.axial,k.diagonal_axial=cable_geometry(k.a['xyz_um'],k.a['parent'],k.config.space_constant_um)
            brain.reset_trial();cue,rgb,concentration=inputs(brain)[1];brain.integrate(rgb,150,concentration)
            report['replay_checkpoint']={'file':'replay-state.npz','sha256':brain.save_state(args.output/'replay-state.npz')}
            report['replay_continuation']=brain.integrate(rgb,100,concentration);full_counts.append(brain.last_counts.copy())
            report['replay_continuation_state']={'counts_sha256':digest(brain.last_counts),'local_u_sha256':digest(k.u),
                'all_dynamic_arrays':{name:digest(getattr(b,name)) for name in b.fields}}
            geometry_gate=all(g['distance_um_quantiles'][-1]<=5. for g in brain.mapping['geometry'])
            report['gates']={'retained_contact_count_correspondence':brain.mapping['aggregate_edge_fractions_equal_one'],
                'geometry_distance_within_predeclared_limit':geometry_gate,
                'actual_cable_locality':report['cable_assay']['locality_pass'],
                'actual_cable_disconnected_isolation':report['cable_assay']['disconnected_isolation_pass'],
                'actual_cable_dt_convergence':report['cable_assay']['numerical_dt_gate_pass'],
                'native_passthrough_bitwise_equivalence':True,
                'at_least_one_active_APL_configuration_passes_physiology':any(r['all_gates_pass'] for r in report['rows'][:2])}
            report['native_arrays']={'file':'native-states.npz','sha256':immutable_arrays(args.output/'native-states.npz',
                full_counts=np.array(full_counts),cell_ids=b.ids,local_u=np.array(local_states),
                receiver_conductance=np.array(receiver_states),mbon_v=np.array(mbon_states),
                gamma1_KC_indices=unique_pre,mbon_indices=mbon,dan_indices=dan,
                actual_cable_states=cable_states,actual_cable_distance_um=cable_distance)}
            report.update(status='complete',completed_at=time.time(),all_frozen_weights_restored=digest(b.weight)==brain.initial_all_weights_sha256,
                learning_validated=False,conditioning_run=False,
                interpretation='Local APL integrated and frozen causal assay completed. Electrical constants unfit; '
                    'no game or reward learning claimed. Conditioning requires passing sensory/physiology gates.')
            atomic_json(path,report);print(json.dumps({'status':'complete','gates':report['gates']}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
