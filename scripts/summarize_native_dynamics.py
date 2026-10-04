"""Read-only, hash-checked synthesis of completed native dynamics diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNS=('native-clearance-20261002-v1','native-coupling-20261002-v1',
      'native-balance-20261002-v1','native-apl-clearance-20261002-v2',
      'native-tonic-apl-20261002-v1','native-mbon-state-20261002-v1',
      'native-conductance-20261002-v1')


def read(name):
    directory=ROOT/'runtime'/name;path=directory/'status.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['status']!='complete':raise ValueError(f'{name} is not a completed native assay')
    for relative,sha in data['sources'].items():
        for p in (ROOT/relative,directory/'sources'/relative):
            if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError(f'Source changed: {p}')
    entry=data.get('native_spikes',data.get('native_states'))
    if entry is None:raise ValueError('Missing native array evidence')
    path=directory/entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Native arrays changed')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    counts=arrays.get('counts',arrays.get('phase_counts'))
    ids=arrays['cell_ids'];index={int(cell):i for i,cell in enumerate(ids)}
    if counts.ndim!=2 or counts.shape[1]!=166700 or ids.shape!=(166700,):raise ValueError('Not full-native evidence')
    if counts.dtype.kind not in 'iu' or (counts<0).any():raise ValueError('Invalid native spike counts')
    return data,arrays,counts,index,{'run':name,'status_sha256':hashlib.sha256((directory/'status.json').read_bytes()).hexdigest(),
                                  'native_array_sha256':entry['sha256'],'sources_and_arrays_checked':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result={'version':'native-dynamics-evidence-synthesis-v1','generated_at':time.time(),
            'evidence':[],'clearance':[],'calibration':[],'mbon_state':[],
            'physiology_sources':{'APL':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7541083/',
                'synaptic_reversal':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6740836/',
                'transferred_coefficient':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/'},
            'gameplay_or_learning_in_these_assays':False}
    for name in RUNS:
        data,arrays,counts,index,evidence=read(name);result['evidence'].append(evidence)
        mbon=np.array([index[x] for x in (10704,11402)])
        ppl=np.array([index[x] for x in (11900,11327)])
        # Preserve the protocol's own readout order in comparisons.
        identity=data.get('brain',data.get('original_brain',{}))
        if identity:
            mbon=np.array([index[x] for x in identity['memory_readouts']['gamma1_MBON11']])
            ppl=np.array([index[x] for x in identity['memory_readouts']['PPL101']])
        if 'clearance-' in name and 'apl-' not in name:
            gamma=arrays['gamma1_KC_indices']; selected=arrays['selected_indices'];offset=0
            gamma_cols=np.searchsorted(selected,gamma);ppl_cols=np.searchsorted(selected,ppl)
            for row in data['rows']:
                bins=arrays['selected_counts'][offset:offset+36];offset+=36
                tail=bins[-5:]
                kc=int(tail[:,gamma_cols].sum());dan=tail[:,ppl_cols].sum(axis=0).tolist()
                if kc!=row['final_500ms_gamma1_KC_spikes'] or dan!=row['final_500ms_PPL101_spikes']:
                    raise ValueError('Clearance tail and actual counts differ')
                result['clearance'].append({'motor_tonic_mV':row['motor_tonic_mV'],'cue':row['cue'],
                    'last_500ms_gamma1_KC_spikes':kc,'last_500ms_PPL101_spikes':dan,
                    'returns_to_silent_assay_baseline':row['returns_to_silent_gamma1_baseline']})
            continue
        if 'mbon-state' in name:
            offset=0
            for row_index,row in enumerate(data['rows']):
                selected_rows=row['rows'];trace=arrays['mbon_state'][offset:offset+len(selected_rows)];offset+=len(selected_rows)
                expected=np.array([np.array([r['v_mV'],r['g_mV'],r['drive_mV']]).T for r in selected_rows],np.float32)
                if not np.array_equal(trace,expected):raise ValueError('Native voltage snapshots differ')
                if counts[row_index*3:row_index*3+3,mbon].tolist()!=row['phase_MBON_counts']:
                    raise ValueError('Native MBON phase counts differ')
                mask=np.array([r['phase']=='cue' for r in selected_rows]);cue=trace[mask]
                result['mbon_state'].append({'APL_KC_gain':row['apl_gain'],
                    'MBON_intrinsic_mV':row['MBON_intrinsic_mV'],'cue':row['cue'],
                    'cue_mean_v_mV':cue[:,:,0].astype(float).mean(axis=0).tolist(),
                    'cue_mean_signed_g_mV':cue[:,:,1].astype(float).mean(axis=0).tolist(),
                    'cue_actual_drive_mV':np.unique(cue[:,:,2],axis=0).tolist(),
                    'phase_MBON_counts':row['phase_MBON_counts'],
                    'potential_cue_budgets':row['potential_budgets'][1]['group_totals']})
            continue
        # Every remaining calibration stores cue as phase 1 of each pre/cue/wash triple.
        row_index=0
        for condition in data['rows']:
            cues=[]
            for cue in condition['cues']:
                actual=counts[row_index*3+1];row_index+=1
                if actual[mbon].tolist()!=cue['cue_MBON11_spikes']:
                    raise ValueError('Native cued MBON counts differ from receipt')
                cues.append({k:v for k,v in cue.items() if k!='rows'})
            settings=condition.get('configuration',{'APL_KC_gain':condition.get('APL_KC_gain'),
                                                   'motor_tonic_mV':condition.get('motor_tonic_mV')})
            result['calibration'].append({'run':name,'configuration':settings,'cues':cues,
                'gates':condition['gates'],'all_gates_pass':condition['all_gates_pass']})
    result['no_admissible_assay_configuration']=not any(x['all_gates_pass'] for x in result['calibration'])
    result['scope']='Actual frozen full-graph dynamics and potential weight*spike budgets. Budgets are not accepted EPSCs; '
    result['scope']+='quietness is an assay-baseline criterion, not a universal biological requirement. No learning score.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'checked_native_runs':len(result['evidence']),
                     'calibration_conditions':len(result['calibration']),
                     'any_admissible_configuration':not result['no_admissible_assay_configuration'],
                     'ordinary_snake_learning':False},ensure_ascii=False))


if __name__=='__main__':main()
