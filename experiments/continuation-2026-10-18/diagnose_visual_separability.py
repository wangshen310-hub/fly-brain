#!/usr/bin/env python3
"""External offline representation probe only. Readout NOT added to fly-brain."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    R=args.source/'runtime/learning-feedback-anchor-s1101-v1'
    def load(split):
        with np.load(R/(split+'.boundary.npz'),allow_pickle=False) as f:
            return {k:f[k] for k in ['external_arriving','node_counts','motor_counts','labels','pose_id','rgb_sha256']}
    train=load('train'); groups=train['pose_id']
    folds=list(GroupKFold(n_splits=4).split(np.arange(len(groups)),train['labels'],groups))
    results={}
    for signal in ['external_first','external_both','relay_motor_counts']:
        def prep(a):
            if signal=='external_first':x=a['external_arriving'][:100].mean(axis=0)
            elif signal=='external_both':x=np.concatenate([a['external_arriving'][:100].mean(axis=0),a['external_arriving'][100:].mean(axis=0)],axis=1)
            else:x=a['node_counts'].transpose(1,0,2).reshape(-1,10)
            return x.astype(np.float64)
        X=prep(train);y=train['labels'];CV=[]
        for alpha in [1.0,100.0,10000.0]:
            fold_score=[]
            for tr,va in folds:
                m=RidgeClassifier(alpha=alpha)
                m.fit(X[tr],y[tr]);fold_score.append(float((m.predict(X[va])==y[va]).mean()))
            CV.append({'alpha':alpha,'mean':float(np.mean(fold_score)),'fold_accuracy':fold_score})
        chosen=sorted(CV,key=lambda r:(-r['mean'],r['alpha']))[0]['alpha']
        m=RidgeClassifier(alpha=chosen);m.fit(X,y)
        train_pred=m.predict(X)
        # Alpha is selected strictly from the 48 training examples before opening validation.
        val=load('validation');pred=m.predict(prep(val));yv=val['labels']
        results[signal]={'cv_train':CV,'alpha_selected_from_train_only':chosen,
           'train_accuracy':float((train_pred==y).mean()),
           'developer_validation_accuracy':float((pred==yv).mean()),
           'developer_validation_recall':[float(np.mean(pred[yv==c]==c)) for c in range(3)],
           'readout':'offline ridge classifier, not part of the neural model'}
    val=load('validation')
    report={'version':'offline-separability-sanity-check-v1','results':results,
      'train_validation_rgb_sha256_overlap':len(set(train['rgb_sha256'].tolist())&set(val['rgb_sha256'].tolist())),
      'train_validation_pose_id_overlap':len(set(train['pose_id'].tolist())&set(val['pose_id'].tolist())),
      'causal_or_native_learning_claim':False,'sealed_final_test_accessed':False,
      'limitation':'Readout is external; images differ but some pose IDs repeat across train and development validation.'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'separability.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    for k,v in results.items():print(k,'CV',v['cv_train'],'validation',v['developer_validation_accuracy'])
if __name__=='__main__':main()
