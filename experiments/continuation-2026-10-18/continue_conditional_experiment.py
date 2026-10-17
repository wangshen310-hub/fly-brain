#!/usr/bin/env python3
"""Bounded, train-only five-cell native conditional continuation diagnostic.

NOT a replacement for the complete fly brain. External releases remain frozen from
an audited checkpoint. Use the full graph to accept a candidate before reporting
learning or running a sealed final test. All validation cases remain uninspected
until the training-selected candidate is fixed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def bind(source):
    sys.path.insert(0,str(source/'files'))
    from flysnake.research.learning_boundary_replay import ConditionalBoundaryReplay
    from flysnake.research.learning_decoder_margin import decoder_margin_loss
    from flysnake.research.teacher_spiking import Dynamics
    return ConditionalBoundaryReplay, decoder_margin_loss, Dynamics


def restore(split, source, model_type, config):
    base=source/'runtime/learning-feedback-anchor-s1101-v1'
    data=base/f'{split}.boundary.npz'
    meta=base/f'{split}.boundary.metadata.json'
    with np.load(data,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    info=json.loads(meta.read_text())
    model=model_type(arrays=arrays,identity=info['brain_identity'],config=config(**info['dynamics']))
    model.register_buffer('reference_forward_counts',torch.from_numpy(np.ascontiguousarray(arrays['motor_counts'][:,:,2:4])))
    model.register_buffer('labels',torch.from_numpy(np.ascontiguousarray(arrays['labels'])))
    return model,arrays,{'data_sha256':sha256(data),'metadata_sha256':sha256(meta),'split':split}


def objective(model, params, loss_fn, temporal_weight=0.0):
    turn, _=model(*params)
    motor=torch.cat((turn[:,:,3:5],model.reference_forward_counts),dim=-1)
    loss, scores=loss_fn(motor,model.labels.long())
    if temporal_weight:
        # The image/label is unchanged over both continuous windows; penalize
        # variation of the signed turn motor difference, not total firing.
        signed_difference = motor[...,1] - motor[...,0]
        loss = loss + temporal_weight * (signed_difference[1] - signed_difference[0]).square().mean()
    return loss, scores


def ranking(scores):
    rec=np.array(scores['group_recall'],dtype=np.float64)
    balanced=rec.mean(axis=1)
    # Never confuse low decoder-margin surrogate loss with true hard-action improvements.
    return (float(rec.min()),float(balanced.min()),float(balanced.mean()),-float(scores['loss']))


def safe_record(scores):
    return {'loss':scores['loss'], 'class_recall':scores['group_recall'],
            'balanced_accuracy':scores['balanced_accuracy'],
            'minimum_class_recall':scores['minimum_class_recall'],
            'minimum_window_balanced_accuracy':scores['minimum_window_balanced_accuracy'],
            'correct_cases':[sum(int(scores['hard_predictions'][w][j]==scores['assigned_labels'][j]) for j in range(len(scores['assigned_labels']))) for w in range(2)]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--max-seconds',type=int,default=180)
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--seed',type=int,default=20261018)
    parser.add_argument('--temporal-weight',type=float,default=0.0)
    args=parser.parse_args()
    if not (1<=args.steps<=96 and 0<args.lr<=0.05 and 0<=args.temporal_weight<=2 and 20<=args.max_seconds<=1200):
        raise SystemExit('Invalid limited search parameters')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Replay, decoder_margin_loss, Dynamics = bind(args.source)
    start=time.perf_counter()
    train,train_arrays,train_id=restore('train',args.source,Replay,Dynamics)
    # Original complete brain -> conditional replay exactness at the starting point.
    proof=train.verify_reference()
    print('Train reference exact on 48 scenes:', proof['parent_millisecond_spike_timing_exact'],flush=True)
    # Never load validation until a training-only parameter point is finalized.
    baselines=[train.baseline_afferent_raw,train.baseline_relay_raw,
               train.baseline_relay_rest_raw,train.baseline_node_gain_raw,train.baseline_node_tau_raw]
    with torch.no_grad():
        base_loss,base_score=objective(train,baselines,decoder_margin_loss)
    best=[p.clone() for p in baselines]
    best_score=base_score
    history=[{'step':0,'candidate_role':'recorded_original_full_brain','train':safe_record(base_score),'accepted':True}]
    print('Baseline recall:',base_score['group_recall'],'objective',base_score['loss'],flush=True)

    # Only existing positive-gain raw variables, with a tight, symmetric, parent-centred box.
    trainable=[torch.nn.Parameter(x.clone()) for x in baselines[:2]]
    constants=[x.clone() for x in baselines[2:]]
    opt=torch.optim.Adam(trainable,lr=args.lr)
    count=0
    for step in range(1,args.steps+1):
        if time.perf_counter()-start>args.max_seconds:break
        opt.zero_grad(set_to_none=True)
        loss, surrogate_score = objective(train, [*trainable,*constants],decoder_margin_loss,args.temporal_weight)
        loss.backward()
        norms=[float(p.grad.norm()) for p in trainable]
        if not all(np.isfinite(norms)):raise ValueError('Non-finite gradient')
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        opt.step()
        with torch.no_grad():
            for p,original in zip(trainable,baselines[:2]):
                p.clamp_(min=original-0.12,max=original+0.12)
            actual_loss,actual_score=objective(train,[*trainable,*constants],decoder_margin_loss)
        count=step
        better=ranking(actual_score)>ranking(best_score)
        if better:
            best=[t.detach().clone() for t in [*trainable,*constants]]
            best_score=actual_score
        history.append({'step':step,'candidate_role':'conditional_not_full_graph','train':safe_record(actual_score),
                        'surrogate_loss_before_update':float(loss.detach()),
                        'grad_norms_before_clip':norms,'train_selected':better,
                        'parameter_max_abs_delta': [float(torch.max(torch.abs(t-b)).detach()) for t,b in zip(trainable,baselines[:2])]})
        print(f'step {step} loss {actual_score["loss"]:.3f} BA {actual_score["balanced_accuracy"]} selected {better}',flush=True)
    # Freeze training selected point first; only now open validation.
    val,validation_arrays,val_id=restore('validation',args.source,Replay,Dynamics)
    val_proof=val.verify_reference()
    with torch.no_grad():
        _,vbaseline=objective(val,[val.baseline_afferent_raw,val.baseline_relay_raw,
                                 val.baseline_relay_rest_raw,val.baseline_node_gain_raw,val.baseline_node_tau_raw],decoder_margin_loss)
        _,vtrial=objective(val,best,decoder_margin_loss)
    # Ensure no baseline drift between splits.
    assert all(torch.equal(a,b) for a,b in zip(baselines,[val.baseline_afferent_raw,val.baseline_relay_raw,
                            val.baseline_relay_rest_raw,val.baseline_node_gain_raw,val.baseline_node_tau_raw]))
    np.savez_compressed(args.output/'conditional-selected-parameters.npz',
        afferent_log_gain=best[0].detach().numpy(),relay_edge_log_gain=best[1].detach().numpy(),
        relay_rest_raw=best[2].detach().numpy(),node_gain_raw=best[3].detach().numpy(),node_tau_raw=best[4].detach().numpy())
    summary={'version':'flybrain-continued-conditional-boundary-diagnostic-v1',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'experiment_scope':'five native cells with recorded external releases, not complete recurrent brain',
      'source_commit':'3fdac042eb8ba18d04365b9124608d7441b8345e',
      'split_data_ids':{'train':train_id,'validation':val_id},
      'original_reference_verified_exact':True,
      'train_samples':48,'development_validation_samples':24,'heldout_sealed_test_accessed':False,
      'steps_requested':args.steps,'steps_completed':count,'learning_rate':args.lr,'temporal_weight':args.temporal_weight,'max_seconds':args.max_seconds,
      'updates_applied_to_native_graph':0,'native_full_graph_runs_after_updates':0,
      'best_train':safe_record(best_score),'baseline_train':safe_record(base_score),
      'baseline_validation':safe_record(vbaseline),'selected_validation':safe_record(vtrial),
      'conditional_validation_is_native_validation':False,'learning_gate_passed':False,
      'wall_seconds':time.perf_counter()-start,'control':'original unchanged snapshot',
      'finding':'Conditional candidate requires complete-graph verification and closed-loop game test before any learning success claim'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (args.output/'history.json').write_text(json.dumps(history,indent=2,ensure_ascii=False))
    print('Finished:',args.output, 'best validation',vtrial['balanced_accuracy'],flush=True)

if __name__=='__main__':main()
