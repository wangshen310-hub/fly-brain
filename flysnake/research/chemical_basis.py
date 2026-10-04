"""Static native synaptic axes measured by reward-free chemical perturbations.

Fixed RGB is held constant while actual left/right ORNs are stimulated. The
sensor regression chooses STATIC changes to EXISTING synapses. It never encodes
a game frame or selects an action. Inference remains the complete native brain.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .pulse_balance import PulseBalancedBrain
from .paired import digest

VERSION='native-chemical-causal-basis-actual-reward-search-v1'


@dataclass(frozen=True)
class ChemicalConfig:
    target_mV: float=2.
    coefficient_bound: float=3.
    gain_bound: float=2.


def synaptic_candidates(theta,cycle,step=1.,bound=3.):
    """A fixed coordinate/paired-axis search, without a scene or action label.

    Proposals are static synaptic settings. A candidate must still play each
    physical world through the complete native model to earn its return.
    """
    theta=np.asarray(theta,float)
    if (theta.shape!=(4,) or not np.isfinite(theta).all()
            or type(cycle) is not int or cycle<0
            or not np.isfinite([step,bound]).all() or min(step,bound)<=0
            or (abs(theta)>bound).any()):
        raise ValueError('Four bounded native synaptic coordinates required')
    paired=np.array([[1,0,-1,0],[1,0,1,0],[0,1,0,-1],[0,1,0,1]],float)
    directions=np.vstack([np.eye(4),paired])
    direction=directions[cycle%len(directions)]
    candidates=np.array([theta,np.clip(theta+step*direction,-bound,bound),
                        np.clip(theta-step*direction,-bound,bound)])
    return candidates,{'direction':direction.tolist(),'step':step,
                       'physical_input_used':False,'action_label_used':False,
                       'eligibility_used':False}


def chemical_axes(counts,concentrations,frame_groups,base,sides,target_mV=2.,duration=300.):
    counts=np.asarray(counts,float);c=np.asarray(concentrations,float)
    groups=np.asarray(frame_groups);base=np.asarray(base);sides=np.asarray(sides)
    if (counts.ndim!=2 or counts.shape[1]!=len(base) or c.shape!=(len(counts),2)
            or groups.shape!=(len(counts),) or sides.shape!=base.shape
            or not all(np.isfinite(x).all() for x in (counts,c,base))
            or (counts<0).any() or (base==0).any() or not np.isin(sides,[0,1]).all()
            or (c<0).any() or (c>1).any()
            or not np.isfinite([target_mV,duration]).all() or min(target_mV,duration)<=0):
        raise ValueError('Matched actual native chemical activity required')
    # Physical sensor variables are calibration conditions, never desired
    # actions, food positions or labels. Remove a separate fixed image effect.
    design=np.c_[c[:,0]-c[:,1],c.mean(1)]
    y=counts.copy();x=design.copy()
    for group in np.unique(groups):
        mask=groups==group
        if mask.sum()<5:raise ValueError('At least five matched chemical levels per image required')
        x[mask]-=x[mask].mean(0);y[mask]-=y[mask].mean(0)
    if np.linalg.matrix_rank(x)!=2:raise ValueError('Independent bilateral concentration conditions required')
    tuning=np.linalg.lstsq(x,y,rcond=None)[0]
    basis=np.zeros((len(base),4));scales=[]
    for side in (0,1):
        for channel in (0,1):
            direction=np.sign(base)*tuning[channel]
            direction[sides!=side]=0
            current=y@(base*direction)*5./duration
            sensitivity=float(np.std(current))
            vector=np.clip(direction*target_mV/max(sensitivity,.1),-1.,1.)
            basis[:,side*2+channel]=vector
            scales.append({'side':side,'physical_channel':'contrast' if channel==0 else 'mean_strength',
                           'unclipped_current_sd_mV':sensitivity,'nonzero_edges':int(np.count_nonzero(vector))})
    return basis,{'samples':len(counts),'fixed_images':len(np.unique(groups)),
        'varying_afferents':int(np.count_nonzero(np.std(y,axis=0))),
        'tuning_sha256':digest(tuning),'axes':scales,
        'actions_used':False,'reward_used':False,'frame_effect_removed':True}


class NativeChemicalBasisBrain(PulseBalancedBrain):
    def __init__(self,baseline_directory,config=None,chemical=ChemicalConfig()):
        values=list(asdict(chemical).values())
        if not np.isfinite(values).all() or min(values)<=0 or chemical.gain_bound>2.:
            raise ValueError('Finite bounded native chemical configuration required')
        self.chemical_ready=False
        if config is None:super().__init__(baseline_directory)
        else:super().__init__(baseline_directory,config=config)
        self.chemical=chemical;self.chemical_basis=None
        self.identity.update(version=VERSION,chemical_config=asdict(chemical),chemical_basis_fitted=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            learning_surrogate_used_for_proposals=False,
            physical_contrast_used_in_frame_encoding=False,
            assumption='Static axes of actual sensory perturbation responses and scalar reward-only native synaptic search; in-vivo plasticity unvalidated')

    def fit_chemical_basis(self,frames):
        if self.chemical_ready or np.any(self.coefficients):raise ValueError('Chemical calibration must precede any reward update')
        if len(frames)<4:raise ValueError('At least four independent fixed RGB frames required')
        # Five concentrations independently cover contrast and common strength.
        levels=[(.2,.2),(.8,.2),(.2,.8),(.8,.8),(.5,.5)]
        counts=[];conditions=[];groups=[];inputs=[]
        for group,rgb in enumerate(frames):
            for concentration in levels:
                self.reset_trial();receipt,_=self.observe(rgb,concentration)
                counts.append(self.brain.last['counts'][self.pre].copy())
                conditions.append(concentration);groups.append(group)
                inputs.append({'rgb':receipt['observation_sha256'],'concentration':list(concentration)})
        basis,measured=chemical_axes(counts,conditions,groups,self.base,self.sides,
            self.chemical.target_mV,self.brain.config.observation_ms)
        self.chemical_counts=np.asarray(counts)
        self.chemical_conditions=np.asarray(conditions)
        self.chemical_frame_groups=np.asarray(groups)
        measured.update(native_counts_sha256=digest(self.chemical_counts),
                        physical_conditions_sha256=digest(self.chemical_conditions),
                        frame_groups_sha256=digest(self.chemical_frame_groups))
        self._accept_chemical_basis(basis,inputs,measured)
        return measured

    def _accept_chemical_basis(self,basis,inputs,measured):
        basis=np.asarray(basis,float)
        if basis.shape!=(len(self.edges),4) or not np.isfinite(basis).all() or (abs(basis)>1).any():
            raise ValueError('Bounded native chemical synaptic axes required')
        self.chemical_basis=basis.copy();self.chemical_ready=True
        self.coefficients=np.zeros(4)
        self.identity.update(dimensions=4,chemical_basis_fitted=True,chemical_basis_sha256=digest(basis),
                             chemical_calibration_inputs=inputs,chemical_measurement=measured)
        self.set_parameters(self.coefficients)

    def set_parameters(self,theta):
        if not self.chemical_ready:return super().set_parameters(theta)
        theta=np.asarray(theta,float)
        if theta.shape!=(4,) or not np.isfinite(theta).all() or (abs(theta)>self.chemical.coefficient_bound).any():
            raise ValueError('Bounded static chemical-axis synaptic gains required')
        gains=np.clip(self.chemical_basis@theta,-self.chemical.gain_bound,self.chemical.gain_bound)
        super().set_parameters(gains)
        self.coefficients=theta.copy()

    def save_chemical_basis(self,path):
        if not self.chemical_ready:raise ValueError('Chemical native calibration incomplete')
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,basis=self.chemical_basis,counts=self.chemical_counts,
                conditions=self.chemical_conditions,frame_groups=self.chemical_frame_groups,
                identity=json.dumps(self.identity,sort_keys=True))
            stream.flush();os.fsync(stream.fileno())
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def restore_chemical_basis(self,path,sha):
        if self.chemical_ready or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Chemical native baseline hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            identity=json.loads(str(z['identity']))
            for key,value in self.identity.items():
                if key=='chemical_basis_fitted':continue
                # Dimension is four after fitting, independent of the inherited
                # actor's initial per-edge parameter representation.
                if key=='dimensions':continue
                if identity.get(key)!=json.loads(json.dumps(value)):raise ValueError('Chemical baseline identity differs: '+key)
            if digest(z['basis'])!=identity['chemical_basis_sha256']:raise ValueError('Chemical axes array mismatch')
            measured=identity['chemical_measurement']
            for name,key in (('counts','native_counts_sha256'),('conditions','physical_conditions_sha256'),
                             ('frame_groups','frame_groups_sha256')):
                if digest(z[name])!=measured[key]:raise ValueError('Chemical calibration evidence mismatch')
            calculated,_=chemical_axes(z['counts'],z['conditions'],z['frame_groups'],self.base,self.sides,
                self.chemical.target_mV,self.brain.config.observation_ms)
            if not np.array_equal(calculated,z['basis']):raise ValueError('Measured chemical axes cannot be reproduced')
            self.chemical_counts=z['counts'].copy();self.chemical_conditions=z['conditions'].copy()
            self.chemical_frame_groups=z['frame_groups'].copy()
            self._accept_chemical_basis(z['basis'],identity['chemical_calibration_inputs'],identity['chemical_measurement'])
