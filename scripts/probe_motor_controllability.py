"""Reward-only search over real signed motor synapses, not an action policy.

Development assay only. No legacy files are changed. The optimizer sees scalar
reward; cue labels are owned by the environment. No decoder remapping, appended
neurons/edges, supervised weight fit, or post-hoc action replacement is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import cue_scene,neutral_scene
from flysnake.lab.experiment import atomic_json


class SynapticSearchBrain:
    def __init__(self,basis='sign'):
        self.brain=LabBrain()
        self.brain.calibrate([neutral_scene(i+901) for i in range(8)])
        b=self.brain.b
        targets=np.r_[self.brain.native.groups['left'],self.brain.native.groups['right']]
        edges=np.flatnonzero(np.isin(b.post,targets)&(b.weight!=0))
        # Modulatory transmitters do not deliver fast synaptic current in this
        # kernel. Keep them unchanged instead of optimizing ineffective weights.
        pre=np.searchsorted(b.ptr,edges,side='right')-1
        keep=b.modulation_mask[pre]==0
        self.edges=edges[keep];self.pre=pre[keep]
        self.base=b.weight[self.edges].copy()
        side=np.isin(b.post[self.edges],self.brain.native.groups['right']).astype(int)
        inhibitory=(self.base<0).astype(int)
        self.group=side*2+inhibitory
        self.theta=np.zeros(4)
        self.evaluations=0
        self.basis=basis
        if basis=='activity':
            # Unsupervised grouping from neural activity, without any desired
            # action/reward. PCA polarity is arbitrary; reward search must learn
            # which parameter directions help. Nothing is recomputed at inference.
            samples=[]
            for seed in (4081,4082,4083,4084):
                for cue in ('left','right'):
                    self.brain.observe(cue_scene(seed,cue),learning=False)
                    samples.append(self.brain.last['counts'][self.pre].copy())
            samples=np.array(samples,dtype=float)
            centered=(samples-samples.mean(axis=0))/np.maximum(samples.std(axis=0),1.)
            _,singular,vh=np.linalg.svd(centered,full_matrices=False)
            loading=vh[0]*np.sign(self.base)
            threshold=max(np.max(np.abs(loading))*.1,1e-12)
            preference=np.where(loading < -threshold,0,np.where(loading>threshold,2,1))
            self.group=side*3+preference
            self.theta=np.zeros(6)
            self.basis_report={'variance_ratio':float(singular[0]**2/max(np.sum(singular**2),1e-12)),
                               'group_sizes':np.bincount(self.group,minlength=6).tolist(),
                               'unlabelled_calibration_seeds':[4081,4082,4083,4084]}
        else:self.basis_report={'group_sizes':np.bincount(self.group,minlength=4).tolist()}

    def weight_hash(self):
        return hashlib.sha256(self.brain.b.weight[self.edges].tobytes()).hexdigest()

    def set_parameters(self,theta):
        theta=np.asarray(theta,dtype=float)
        if theta.shape!=self.theta.shape or not np.isfinite(theta).all() or np.abs(theta).max()>4.01:
            raise ValueError('Invalid bounded synaptic log gains')
        self.theta=theta.copy()
        self.brain.b.weight[self.edges]=self.base*np.exp(theta[self.group])
        assert np.all(np.sign(self.brain.b.weight[self.edges])==np.sign(self.base))

    def observe(self,rgb):
        result=self.brain.observe(rgb,learning=False,task='choice')
        result['memory_sha256']=self.weight_hash()
        return result

    def feedback(self,kind):
        before=self.weight_hash()
        result=self.brain.feedback(kind,learning=False)
        assert self.weight_hash()==before
        result.update(memory_before=before,memory_after=before,weight_update='between candidate trials via reward-only search')
        return result


def evaluate(brain,seeds,mapping,feedback=False):
    rows=[]
    for seed in seeds:
        for side in ('left','right'):
            r=brain.observe(cue_scene(seed,side))
            correct=side if mapping=='same' else ('right' if side=='left' else 'left')
            reward=1 if r['action']==correct else -1
            receipt=brain.feedback('reward' if reward>0 else 'aversive') if feedback else None
            if receipt is not None and receipt['delivered_spikes']<=0:raise RuntimeError('Undelivered feedback')
            rows.append({'seed':seed,'cue':side,'action':r['action'],'reward':reward,
                         'counts':r['counts'],'memory_sha256':r['memory_sha256'],'feedback':receipt})
    return float(np.mean([r['reward'] for r in rows])),rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=92201)
    p.add_argument('--candidates',type=int,default=64);p.add_argument('--output',type=Path,default=ROOT/'runtime/search-development-v1')
    p.add_argument('--basis',choices=('sign','activity'),default='sign')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'status.json').exists():raise ValueError('Use a new output directory; never overwrite an earlier experiment')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(args.seed);brain=SynapticSearchBrain(args.basis);start=time.time()
    report={'protocol':'signed-native-motor-gain-search-dev-v2','status':'running','basis':args.basis,'basis_report':brain.basis_report,
            'seed':args.seed,'mutable_edges':len(brain.edges),
            'bounds_log_gain':[-4,4],'decoder_unchanged':True,'topology_unchanged':True,
            'legacy_memory_imported':False,'iterations':[]}
    dimension=len(brain.theta)
    best_theta=np.zeros(dimension);best=-float('inf')
    for iteration in range(args.candidates):
        if iteration==0:theta=np.zeros(dimension)
        elif iteration%3==0:theta=np.clip(best_theta+rng.normal(0,.7,dimension),-4,4)
        else:theta=rng.uniform(-4,4,dimension)
        brain.set_parameters(theta)
        # Fixed development layouts. Held-out evaluation is never fed back.
        fitness,rows=evaluate(brain,[6121],'opposite',feedback=True)
        if fitness>best:
            best=fitness;best_theta=theta.copy()
        record={'iteration':iteration,'theta':theta.tolist(),'fitness':fitness,'rows':rows,
                'best_fitness':best,'elapsed_seconds':time.time()-start}
        report['iterations'].append(record)
        report.update(best_theta=best_theta.tolist(),best_fitness=best)
        atomic_json(args.output/'status.json',report)
        print(json.dumps({k:record[k] for k in ('iteration','fitness','best_fitness','elapsed_seconds')}),flush=True)
        if best==1:break
    brain.set_parameters(best_theta)
    fitness,test=evaluate(brain,range(7310,7318),'opposite',feedback=False)
    # Preserve actual effective weights and identities, not only a claimed score.
    np.savez_compressed(args.output/'candidate.npz',edges=brain.edges,base=brain.base,
                        weight=brain.brain.b.weight[brain.edges],theta=best_theta,group=brain.group,
                        calibration=json.dumps(brain.brain.calibration))
    report.update(status='complete',test_fitness=fitness,test=test,
                  candidate_sha256=hashlib.sha256((args.output/'candidate.npz').read_bytes()).hexdigest(),
                  interpretation='Feasibility/development only; matched controls and Snake remain required')
    atomic_json(args.output/'status.json',report)
    print(json.dumps({'status':'complete','test_fitness':fitness,'theta':best_theta.tolist()}),flush=True)


if __name__=='__main__':main()
