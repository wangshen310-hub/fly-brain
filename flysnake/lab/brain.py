"""Full MaleCNS dynamics with explicitly engineered sensory/motor plasticity.

Only RGB, task mode and signed outcome stimuli cross this interface. No game
coordinates, target action, path, collision mask or policy enters this class.
The 858 existing excitatory motor afferents are a candidate engineering locus;
their in-vivo plasticity has NOT been established by this project.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from flysnake.neural import RealFlyBrain
from .protocol import ACTIONS, VERSION, Config, sample_linear, decode, eligibility_update, plastic_update


class LabBrain:
    def __init__(self):
        self.config=Config()
        self.native=RealFlyBrain(learning=False)
        self.b=self.native.brain
        b=self.b
        b.weights_frozen=True
        motor=np.concatenate(list(self.native.groups.values()))
        self.edges=np.flatnonzero(np.isin(b.post,motor)&(b.weight>0))
        self.pre=np.searchsorted(b.ptr,self.edges,side='right')-1
        self.edge_group=np.zeros(len(self.edges),dtype=int)
        for i,name in enumerate(ACTIONS):
            self.edge_group[np.isin(b.post[self.edges],self.native.groups[name])]=i
        self.base=b.weight[self.edges].copy()
        self.fraction=np.ones(len(self.edges))
        self.trace=np.zeros(len(self.edges))
        self.pre_baseline=np.zeros(len(self.edges))
        self.pre_scale=np.ones(len(self.edges))*10
        self.calibration={'baseline_hz':[0.,0.,0.],'scale_hz':[1.,1.,1.]}
        self.retinal_light=np.zeros(len(b.retina))
        self.r8_light=np.zeros(len(b.r8))
        self.neural_ms=0
        self.last=None
        self.calibrated=False
        self.source_signature=hashlib.sha256(''.join(
            hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
            for f in ('brain.py','protocol.py','environment.py')).encode()).hexdigest()

    def reset_trial(self):
        self.b.reset(keep_memory=True)
        self.b.weights_frozen=True
        self.retinal_light.fill(0); self.r8_light.fill(0); self.trace.fill(0)
        self.last=None
        self.integrate(np.full((180,320,3),85,dtype=np.uint8),100)

    def integrate(self,rgb,duration,stimulus=None):
        cfg=self.config; b=self.b
        linear=sample_linear(rgb,b.uv)
        target=linear@np.asarray([.2126,.7152,.0722])
        r8rgb=sample_linear(rgb,b.r8_uv)
        target8=r8rgb[np.arange(len(b.r8)),b.r8_channel]
        counts=np.zeros(b.n,dtype=np.int32)
        for _ in range(duration//10):
            alpha=1-np.exp(-10/cfg.sensory_tau_ms)
            self.retinal_light+=alpha*(target-self.retinal_light)
            self.r8_light+=alpha*(target8-self.r8_light)
            pulses=[(b.retina,cfg.retinal_gain*self.retinal_light),(b.r8,cfg.retinal_gain*self.r8_light)]
            if stimulus is not None: pulses.append(stimulus)
            # Black native light avoids the old saturating transfer. Measured
            # linear currents enter only the real annotated photoreceptors.
            c,_=b.step(np.zeros(len(b.retina)),10,learning=False,stimulation=pulses)
            counts+=c
        self.neural_ms+=duration
        if not np.isfinite(b.v).all() or not np.isfinite(b.g).all():
            raise FloatingPointError('Nonfinite connectome state')
        return counts

    def set_fraction(self,fraction):
        f=np.asarray(fraction,dtype=float)
        if f.shape!=self.fraction.shape or not np.isfinite(f).all() or (f<.25).any() or (f>2).any():
            raise ValueError('Invalid synaptic efficacy')
        self.fraction[:]=f
        self.b.weight[self.edges]=self.base*f

    def observe(self,rgb,task='choice',learning=True,reset=True):
        if reset: self.reset_trial()
        start=self.neural_ms
        counts=self.integrate(rgb,self.config.observation_ms)
        groups={k:counts[ix].tolist() for k,ix in self.native.groups.items()}
        result=decode(groups,self.config.observation_ms,self.calibration,task,self.config.turn_margin)
        scores=np.array([result['scores'][k] for k in ACTIONS])
        if learning:
            self.trace=eligibility_update(self.trace,counts[self.pre]*1000/self.config.observation_ms,
                self.pre_baseline,self.pre_scale,scores,self.edge_group,
                self.config.observation_ms/1000,self.config.trace_seconds)
        result.update(neural_time_start=start,neural_time_end=self.neural_ms,
            observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            calibration=self.calibration,memory_sha256=self.memory_hash(),
            total_spikes=int(counts.sum()),active_afferents=int(np.count_nonzero(counts[self.pre])),
            afferent_counts=counts[self.pre].tolist(),selected_cell_ids=self.native.ids)
        self.last={'rgb':rgb.copy(),'receipt':result,'counts':counts}
        return result

    def feedback(self,kind,learning=True):
        if kind not in ('reward','aversive') or self.last is None:
            raise ValueError('Feedback requires an observation and a valid stimulus')
        before=self.memory_hash()
        targets=self.b.circuit[kind]
        counts=self.integrate(self.last['rgb'],self.config.feedback_ms,
                              (targets,self.config.feedback_current))
        target_spikes=int(counts[targets].sum())
        # Delivered DAN activity controls magnitude; valence is the declared
        # experimental reward/aversive channel, not an inferred natural sign.
        modulation=(1 if kind=='reward' else -1)*float(np.tanh(target_spikes/len(targets)/5))
        old=self.fraction.copy()
        if learning:
            self.set_fraction(plastic_update(self.fraction,self.trace,modulation,self.config))
        self.trace.fill(0)
        self.last=None # A decision cannot receive a second reward.
        return {'kind':kind,'delivered_spikes':target_spikes,'stimulus_ms':self.config.feedback_ms,
                'stimulus_current':self.config.feedback_current,'modulation':modulation,
                'learning':learning,'memory_before':before,'memory_after':self.memory_hash(),
                'changed_edges':int(np.count_nonzero(old!=self.fraction)),
                'mean_abs_change':float(np.abs(old-self.fraction).mean()),'action':None}

    def calibrate(self,frames):
        if self.calibrated: raise ValueError('Calibration is locked; create another experiment')
        motor=[]; afferents=[]
        for rgb in frames:
            result=self.observe(rgb,learning=False)
            motor.append([result['rates_hz'][a] for a in ACTIONS])
            afferents.append(self.last['counts'][self.pre]*1000/self.config.observation_ms)
        motor=np.array(motor); afferents=np.array(afferents)
        self.calibration={'baseline_hz':motor.mean(axis=0).tolist(),
                          'scale_hz':np.maximum(motor.std(axis=0),1000/self.config.observation_ms).tolist()}
        self.pre_baseline=afferents.mean(axis=0)
        self.pre_scale=np.maximum(afferents.std(axis=0),10)
        self.calibrated=True
        self.last=None
        return {'calibration':self.calibration,'samples':len(frames),'label_free':True}

    def intervention(self,rgb,group,current):
        self.reset_trial()
        counts=self.integrate(rgb,self.config.observation_ms,(self.native.groups[group],current))
        groups={k:counts[ix].tolist() for k,ix in self.native.groups.items()}
        return decode(groups,self.config.observation_ms,self.calibration,task='choice')

    def memory_hash(self):
        return hashlib.sha256(self.fraction.tobytes()).hexdigest()

    def metadata(self):
        return {'version':VERSION,'config':asdict(self.config),'neurons':self.b.n,
            'graph_edges':len(self.b.post),'candidate_plastic_edges':len(self.edges),
            'candidate_rule':'Engineered three-factor plasticity on existing excitatory motor afferents; in-vivo plasticity unvalidated',
            'topology_modified':False,'external_policy':False,'legacy_memory':'Preserved separately, not imported into the new rule',
            'selected_cell_ids':self.native.ids,'calibration':self.calibration,'calibrated':self.calibrated,
            'memory_sha256':self.memory_hash(),'near_lower_bound':int((self.fraction<=.26).sum()),
            'near_upper_bound':int((self.fraction>=1.99).sum()),'neural_ms':self.neural_ms}

    def checkpoint(self,path):
        """Assay checkpoints are trial boundaries, not ongoing neural windows."""
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        metadata={'protocol':self.config.signature(),'source':self.source_signature,
                  'calibration':self.calibration,'calibrated':self.calibrated,'neural_ms':self.neural_ms,
                  'memory_sha256':self.memory_hash(),
                  'base_sha256':hashlib.sha256(self.base.tobytes()).hexdigest(),
                  'edges_sha256':hashlib.sha256(self.edges.tobytes()).hexdigest()}
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=json.dumps(metadata),fraction=self.fraction,
                                pre_baseline=self.pre_baseline,pre_scale=self.pre_scale)
        return metadata

    def restore(self,path):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']))
            if (m['protocol']!=self.config.signature() or m['source']!=self.source_signature or
                m['base_sha256']!=hashlib.sha256(self.base.tobytes()).hexdigest() or
                m['edges_sha256']!=hashlib.sha256(self.edges.tobytes()).hexdigest()):
                raise ValueError('Incompatible experimental memory; previous file retained')
            for key in ('pre_baseline','pre_scale'):
                a=z[key]
                if a.shape!=self.fraction.shape or not np.isfinite(a).all(): raise ValueError('Invalid calibration memory')
                setattr(self,key,a.copy())
            if (self.pre_scale<=0).any(): raise ValueError('Invalid calibration scale')
            self.set_fraction(z['fraction'])
            if self.memory_hash()!=m['memory_sha256']: raise ValueError('Memory hash mismatch')
            self.calibration=m['calibration']; self.calibrated=m['calibrated']; self.neural_ms=m['neural_ms']
        self.reset_trial()
        return self.metadata()


def worker(connection):
    try:
        brain=LabBrain()
        connection.send({'result':brain.metadata()})
        while True:
            request=connection.recv(); operation=request.pop('operation')
            if operation=='close': break
            if operation not in ('observe','feedback','calibrate','intervention','set_fraction','metadata','checkpoint','restore'):
                raise ValueError('Unsupported operation')
            result=getattr(brain,operation)(**request)
            connection.send({'result':result})
    except EOFError:
        pass
    except Exception as exc:
        import traceback
        connection.send({'error':str(exc),'traceback':traceback.format_exc()})
    finally:
        connection.close()
