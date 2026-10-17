#!/usr/bin/env python3
"""41 fixed train-only interpolations between unchanged parent and a conditional candidate.
No full brain is evaluated; no sealed test is read.
"""
import argparse
from pathlib import Path
import json,time
import numpy as np
import torch
from continue_conditional_experiment import bind,restore,objective,safe_record

def rank(s):
    recall=np.array(s['group_recall'],dtype=float)
    ba=recall.mean(axis=1)
    return (float(ba.min()),float(recall.min()),float(ba.mean()),-float(s['loss']))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    Replay,loss_fn,Dynamics=bind(a.source)
    train, arr,train_id=restore('train',a.source,Replay,Dynamics)
    train.verify_reference()
    start=time.perf_counter()
    with np.load(a.proposal,allow_pickle=False) as d:
        proposal=[torch.from_numpy(d[n].copy()) for n in ('afferent_log_gain','relay_edge_log_gain','relay_rest_raw','node_gain_raw','node_tau_raw')]
    parent=[train.baseline_afferent_raw,train.baseline_relay_raw,train.baseline_relay_rest_raw,train.baseline_node_gain_raw,train.baseline_node_tau_raw]
    for orig,trial in zip(parent,proposal):
        assert orig.shape==trial.shape and orig.dtype==trial.dtype
        assert torch.isfinite(trial).all()
    scales=[i/40 for i in range(41)]
    trials=[];selected=None;best_score=None;best_alpha=None
    for i,alpha in enumerate(scales):
        params=[base+(q-base)*alpha for base,q in zip(parent,proposal)]
        with torch.no_grad():
            _,s=objective(train,params,loss_fn)
        trials.append({'alpha':alpha, 'train':safe_record(s)})
        if best_score is None or rank(s)>rank(best_score):
            selected=[v.detach().clone() for v in params];best_score=s;best_alpha=alpha
    val,_,val_id=restore('validation',a.source,Replay,Dynamics)
    val.verify_reference()
    with torch.no_grad():
        _,vnew=objective(val,selected,loss_fn)
        _,vbaseline=objective(val,parent,loss_fn)
    if not all(torch.equal(x,y) for x,y in zip(parent,[val.baseline_afferent_raw,val.baseline_relay_raw,val.baseline_relay_rest_raw,val.baseline_node_gain_raw,val.baseline_node_tau_raw])):
        raise ValueError('Split parent mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps({'version':'train-only-interpolation-screen-v1',
      'parent_source_sha256':train_id['data_sha256'],'steps':len(scales),'selected_alpha':best_alpha,
      'training_parent':trials[0]['train'],'training_selected':safe_record(best_score),
      'validation_parent':safe_record(vbaseline),'validation_selected':safe_record(vnew),
      'sealed_final_test_read':False,'complete_brain_scans_after_parameter_change':0,
      'status':'conditional-only-not-learned','wall_seconds':time.perf_counter()-start},indent=2,ensure_ascii=False))
    (a.output/'grid.json').write_text(json.dumps(trials,indent=2))
    np.savez_compressed(a.output/'conditional-selected-parameters.npz',
        afferent_log_gain=selected[0].numpy(),relay_edge_log_gain=selected[1].numpy(),
        relay_rest_raw=selected[2].numpy(),node_gain_raw=selected[3].numpy(),node_tau_raw=selected[4].numpy())
    print('Selected interpolation',best_alpha,'training',best_score['balanced_accuracy'],'developer validation',vnew['balanced_accuracy'])
if __name__=='__main__':main()
