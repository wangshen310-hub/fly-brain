"""One-at-a-time ablation of external sensory background, no game or reward."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.input_isolation import IsolatedInputBrain,InputConfig,VERSION
from flysnake.research.paired import writer_lock,digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve previous isolation evidence')
        source=parent_sources()
        for relative in ('scripts/assay_input_isolation.py','flysnake/research/input_isolation.py'):
            source[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        report={'protocol':VERSION,'status':'loading','sources':source,'process_id':os.getpid(),
                'rows':[],'summaries':{},'started_at':time.time(),'learning':False,'game_played':False}
        for relative in source:
            target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        atomic_json(path,report)
        try:
            brain=IsolatedInputBrain();report['brain']=brain.identity;samples=[]
            conditions=[('original',InputConfig()),
                        ('retina_off',InputConfig(vision_input=False)),
                        ('retina_lamina_off',InputConfig(vision_input=False,lamina_bias_mV=0.)),
                        ('no_background',InputConfig(vision_input=False,lamina_bias_mV=0.,zero_odor_is_silent=True)),
                        ('retina_only',InputConfig(vision_input=True,lamina_bias_mV=0.,zero_odor_is_silent=True))]
            unique_pre=np.unique(brain.pre)
            for label,config in conditions:
                brain.set_input_configuration(config);block=[]
                for cue,rgb,concentration in inputs(brain):
                    brain.reset_trial();receipt=brain.integrate(rgb,300,concentration)
                    sample=brain.last_counts.copy();samples.append(sample);block.append(sample[unique_pre])
                    row={'condition':label,'cue':cue,'receipt':receipt}
                    report['rows'].append(row);report.update(status='running',updated_at=time.time())
                    atomic_json(path,report)
                    print(json.dumps({'condition':label,'cue':cue,'KC_active':receipt['KC_active'],
                        'unique_gamma1_KC_active':int(np.count_nonzero(block[-1])),
                        'readouts':receipt['readouts'],'motor':receipt['counts']}),flush=True)
                norm=np.linalg.norm(block[1])*np.linalg.norm(block[2])
                report['summaries'][label]={
                    'input_config':asdict(config),'gamma1_KC_count':len(unique_pre),
                    'active_gamma1_KC':[int(np.count_nonzero(x)) for x in block],
                    'cue_response_changed_from_neutral':[int(np.count_nonzero(x-block[0])) for x in block[1:]],
                    'raw_cue_cosine_similarity':float(np.dot(block[1].astype(float),block[2].astype(float))/norm) if norm else None,
                    'cue_pair_equal':bool(np.array_equal(block[1],block[2]))}
                atomic_json(path,report)
            report['native_spikes']={'file':'native-spikes.npz','sha256':immutable_arrays(
                args.output/'native-spikes.npz',counts=np.array(samples),cell_ids=brain.b.ids,
                plastic_edges=brain.edges,plastic_pre=brain.pre,base_weights=brain.base)}
            report.update(status='complete',completed_at=time.time(),
                all_weights_restored=digest(brain.b.weight)==brain.initial_all_weights_sha256,
                interpretation='Sensory background ablations, not learned behavior or a replacement game camera.')
            atomic_json(path,report);print(json.dumps(report['summaries']),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
