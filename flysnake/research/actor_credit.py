"""Self-action reward credit on existing native afferents, never an actor net.

The three soft probabilities below are a LEARNING SURROGATE only. They neither
sample nor select actions; hard actions still use actual full-brain spikes and
the locked decoder. Actual candidate games can reject either gradient sign.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .sensory_basis import NativeSensoryBasisBrain,BasisConfig
from .olfaction import OdorConfig
from .paired import digest

VERSION='native-self-action-credit-verified-reward-v1'


@dataclass(frozen=True)
class ActorCreditConfig:
    temperature: float=.5
    learning_rate: float=.2
    maximum_edge_step: float=.12
    gain_bound: float=2.
    noise_sigma: float=3.
    discount: float=.95
    activity_floor: float=3.
    minimum_log_slope: float=.05


def surrogate_probabilities(gap,margin=.2,temperature=.5):
    if not np.isfinite([gap,margin,temperature]).all() or min(margin,temperature)<=0:
        raise ValueError('Finite positive surrogate readout scales required')
    def logistic(x):return 1./(1.+np.exp(-np.clip(x,-60.,60.)))
    left=logistic((-gap-margin)/temperature)
    right=logistic((gap-margin)/temperature)
    forward=1.-left-right
    dl=-left*(1.-left)/temperature;dr=right*(1.-right)/temperature
    return ({'left':left,'forward':max(forward,1e-14),'right':right},
            {'left':dl,'forward':-dl-dr,'right':dr})


def self_action_credit(pre_counts,mean_counts,weights,sides,receipt,calibration,config=ActorCreditConfig()):
    pre_counts,mean_counts,weights,sides=map(np.asarray,(pre_counts,mean_counts,weights,sides))
    if (pre_counts.ndim!=1 or any(a.shape!=pre_counts.shape for a in (mean_counts,weights,sides))
            or not all(np.isfinite(a).all() for a in (pre_counts,mean_counts,weights))
            or (pre_counts<0).any() or (mean_counts<0).any() or not np.isin(sides,[0,1]).all()):
        raise ValueError('Matched native spikes, frozen activity and signed synapses required')
    action=receipt['action']
    if action is None:return np.zeros(len(weights))
    if action not in ('left','forward','right'):raise ValueError('Invalid native self action')
    scores=receipt['scores'];gap=scores['right']-scores['left']
    margin=.2*max(abs(scores['left']),abs(scores['right']),1.)
    probabilities,derivatives=surrogate_probabilities(gap,margin,config.temperature)
    scale=np.asarray(calibration['scale_hz'])[[0,2]]
    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Locked native calibration required')
    # Signed EPSC and neutral-mean compensation, with a declared variance
    # preconditioner. Recurrence and spike thresholds make this approximate,
    # not a true policy gradient of the deterministic full spiking model.
    feature=(pre_counts-mean_counts)/np.sqrt(mean_counts+config.activity_floor)
    current_slope=weights*feature*5./300.
    contrast_slope=np.where(sides==1,1.,-1.)/scale[sides]
    slope=derivatives[action]/max(probabilities[action],1e-6)
    # A saturated, incorrect hard decision must still be able to receive
    # negative credit. This declared floor makes the surrogate approximate;
    # it is never used to choose an action or change the locked decoder.
    if action in ('left','right'):
        slope=(-1. if action=='left' else 1.)*max(abs(slope),config.minimum_log_slope)
    return slope*contrast_slope*current_slope


def displaced_feedback(values,seed):
    """A random nonzero rotation: every trial receives another trial's return."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('At least two finite actual feedback values required')
    offset=int(np.random.default_rng(seed).integers(1,len(values)))
    permutation=np.roll(np.arange(len(values)),offset)
    if np.any(permutation==np.arange(len(values))):raise RuntimeError('Feedback identity permutation')
    return values[permutation],permutation.tolist()


def reward_proposal(theta,episode_traces,returns,*,arm,seed,config=ActorCreditConfig()):
    theta,traces,values=np.asarray(theta,float),np.asarray(episode_traces,float),np.asarray(returns,float)
    if (theta.ndim!=1 or traces.shape!=(len(values),len(theta)) or len(values)<2
            or not all(np.isfinite(a).all() for a in (theta,traces,values))
            or arm not in ('learning','frozen','shuffled')):
        raise ValueError('Matched native eligibility and actual game return batch required')
    assigned=values.copy();permutation=list(range(len(values)))
    if arm=='shuffled':assigned,permutation=displaced_feedback(values,seed)
    # A fixed zero baseline allows all-negative experience to discourage a
    # bad self action. It is not fitted from held-out labels or game geometry.
    gradient=np.mean(assigned[:,None]*traces,axis=0)
    rms=float(np.sqrt(np.mean(gradient**2)))
    step=np.clip(config.learning_rate*gradient/max(rms,1e-8),
                 -config.maximum_edge_step,config.maximum_edge_step)
    if arm=='frozen':step[:]=0.
    candidates=np.array([theta,np.clip(theta+step,-config.gain_bound,config.gain_bound),
                        np.clip(theta-step,-config.gain_bound,config.gain_bound)])
    return candidates,{'gradient_rms':rms,'step_rms':float(np.sqrt(np.mean(step**2))),
        'feedback_permutation':permutation,'assigned_returns':assigned.tolist(),
        'gradient_sha256':digest(gradient),'step_sha256':digest(step),'exact_policy_gradient':False}


def accept_game_candidate(candidates,returns,*,arm,seed):
    candidates,values=np.asarray(candidates,float),np.asarray(returns,float)
    if (candidates.ndim!=2 or len(candidates)!=3 or values.shape!=(3,)
            or not np.isfinite(candidates).all() or not np.isfinite(values).all()
            or arm not in ('learning','frozen','shuffled')):
        raise ValueError('Three actual native candidate game returns required')
    assigned=values.copy();permutation=[0,1,2]
    if arm=='shuffled':assigned,permutation=displaced_feedback(values,seed)
    winner=int(np.argmax(assigned))
    if arm=='frozen' or assigned[winner]<=assigned[0]:winner=0
    return candidates[winner].copy(),{'winner':winner,'accepted':winner!=0,
        'actual_returns':values.tolist(),'assigned_returns':assigned.tolist(),
        'feedback_permutation':permutation,'step_rms':float(np.sqrt(np.mean((candidates[winner]-candidates[0])**2)))}


class NativeActorCreditBrain(NativeSensoryBasisBrain):
    def __init__(self,baseline_directory,config=ActorCreditConfig()):
        if (not np.isfinite(list(asdict(config).values())).all()
                or min(asdict(config).values())<=0 or config.discount>1):
            raise ValueError('Positive finite native learning configuration required')
        baseline_directory=Path(baseline_directory)
        manifest=json.loads((baseline_directory/'perception-baseline.json').read_text(encoding='utf-8'))
        self.actor_ready=False;self.motor_noise=np.zeros(2)
        super().__init__(size=manifest['size'],config=BasisConfig(**manifest['config']),
                         odor=OdorConfig(**manifest['odor']))
        self.restore_basis(baseline_directory/manifest['file'],manifest['sha256'])
        # The saved SVD matrix is unused by this rule. Only its unrewarded mean
        # and locked display/decoder provenance are inherited, without trained
        # weights, reward labels or prior held-out evaluation results.
        self.actor_config=config
        self.coefficients=np.zeros(len(self.edges));self.actor_ready=True
        self.identity.update(version=VERSION,dimensions=len(self.edges),
            credit_config=asdict(config),perception_baseline=manifest,
            inherited_sampling_source_sha256=self.identity['source_sha256'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            static_svd_used_for_parameter_proposals=False,exact_policy_gradient=False,
            eligibility='Centered actual native presynaptic spikes x signed EPSC x learning-only self-action surrogate; no desired action',
            acceptance='Actual matched games test both proposal signs before committed update',
            assumption='Engineered self-action eligibility with saturated slope floor and reward-verified synaptic updates; natural fly plasticity unvalidated')
        self.set_parameters(self.coefficients)

    def set_sensation(self,concentration):
        super().set_sensation(concentration)
        if self.actor_ready:
            for i,name in enumerate(('left','right')):
                self.brain.b.tonic[self.brain.native.groups[name]]+=self.motor_noise[i]

    def set_parameters(self,theta):
        if not self.actor_ready:return super().set_parameters(theta)
        theta=np.asarray(theta,float)
        if (theta.shape!=(len(self.edges),) or not np.isfinite(theta).all()
                or (abs(theta)>self.actor_config.gain_bound+1e-9).any()):
            raise ValueError('Bounded existing native synaptic gains required')
        weights=(self.base*np.exp(theta)).astype(self.base.dtype)
        if not np.array_equal(np.sign(weights),np.sign(self.base)):raise RuntimeError('Native synaptic sign changed')
        self.brain.b.weight[self.edges]=weights;self.coefficients=theta.copy()
        correction=np.zeros(self.brain.b.n)
        np.add.at(correction,self.post,-(weights-self.base)*self.mean_counts*5./300.)
        self.correction=np.clip(correction,-self.config.mean_current_limit,self.config.mean_current_limit)
        self.set_sensation(self.concentration)

    def observe(self,rgb,concentration,noise=None):
        noise=np.zeros(2) if noise is None else np.asarray(noise,float)
        if noise.shape!=(2,) or not np.isfinite(noise).all():raise ValueError('Two finite neural exploration currents required')
        self.motor_noise=noise.copy()
        receipt=super().observe(rgb,concentration)
        counts=self.brain.last['counts'][self.pre]
        trace=self_action_credit(counts,self.mean_counts,self.brain.b.weight[self.edges],self.sides,
                                receipt,self.brain.calibration,self.actor_config)
        receipt.update(motor_current_noise=noise.tolist(),eligibility_sha256=digest(trace),
                       learning_surrogate_selects_action=False)
        return receipt,trace

    def reset_trial(self):
        self.motor_noise=np.zeros(2);super().reset_trial()
