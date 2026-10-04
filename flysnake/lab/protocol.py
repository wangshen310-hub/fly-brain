"""Pure sensory, motor-interface and plasticity mathematics, without a policy."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np

VERSION = 'malecns-lab-linear-motor-three-factor-v4'
ACTIONS = ('left', 'forward', 'right')


@dataclass(frozen=True)
class Config:
    observation_ms: int = 300
    feedback_ms: int = 200
    retinal_gain: float = 30.0
    sensory_tau_ms: float = 10.0
    feedback_current: float = 30.0
    learning_rate: float = 0.08
    trace_seconds: float = 1.0
    minimum_fraction: float = 0.25
    maximum_fraction: float = 2.0
    turn_margin: float = 0.20

    def signature(self):
        return hashlib.sha256(json.dumps({'version':VERSION, **asdict(self)}, sort_keys=True).encode()).hexdigest()


def sample_linear(rgb, uv):
    rgb = np.asarray(rgb)
    if rgb.shape != (180,320,3) or rgb.dtype != np.uint8:
        raise ValueError('Expected fixed 320x180 RGB uint8, no board features')
    a = rgb[(uv[:,1]*179).astype(int), (uv[:,0]*319).astype(int)].astype(np.float32)/255
    return np.where(a <= .04045, a/12.92, ((a+.055)/1.055)**2.4)


def decode(counts, duration_ms, calibration, task='snake', margin=.2):
    """Only fresh counts and a locked neutral calibration can enter this BCI."""
    if task not in ('choice', 'snake') or duration_ms <= 0:
        raise ValueError('Invalid readout task/window')
    if set(counts) != set(ACTIONS):
        raise ValueError('Invalid cell groups')
    rates=[]
    for name,n in zip(ACTIONS,(1,2,1)):
        c=np.asarray(counts[name],dtype=float)
        if c.shape!=(n,) or not np.isfinite(c).all() or (c<0).any() or (c!=np.floor(c)).any():
            raise ValueError('Malformed fresh counts')
        rates.append(float(c.mean()*1000/duration_ms))
    base=np.asarray(calibration['baseline_hz'],dtype=float)
    scale=np.asarray(calibration['scale_hz'],dtype=float)
    if base.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(base).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid locked calibration')
    scores=(np.array(rates)-base)/scale
    result={'action':None,'counts':counts,'rates_hz':dict(zip(ACTIONS,rates)),
            'scores':dict(zip(ACTIONS,scores.tolist())),'reason':'silent','version':VERSION}
    if sum(sum(c) for c in counts.values())==0:
        return result
    gap=float(scores[2]-scores[0])
    if abs(gap)<1e-9:
        result.update(action='forward' if task=='snake' and rates[0]+rates[2]>0 else None,reason='turn_balance')
    elif task=='snake' and abs(gap)<=margin*max(abs(scores[0]),abs(scores[2]),1):
        result.update(action='forward',reason='turn_balance')
    else:
        result.update(action='right' if gap>0 else 'left',reason='calibrated_turn')
    return result


def eligibility_update(trace, pre_hz, pre_baseline, pre_scale, post_group_scores, edge_group, seconds, tau):
    """Activity-only three-factor candidate; receives neither reward nor action labels.

    Signed presynaptic deviation gates a local postsynaptic motor competition
    signal. Competition is an engineered neural-rate term, not anatomical proof.
    """
    pre=np.clip((pre_hz-pre_baseline)/pre_scale,-3,3)
    post=np.asarray(post_group_scores,dtype=float)
    competition=np.clip(post-post.mean(),-3,3)
    return np.exp(-seconds/tau)*trace+(1-np.exp(-seconds/tau))*pre*competition[edge_group]


def plastic_update(fraction, trace, modulation, config):
    if not np.isfinite(modulation) or not np.isfinite(trace).all():
        raise ValueError('Nonfinite feedback/eligibility')
    return np.clip(fraction+config.learning_rate*modulation*trace,
                   config.minimum_fraction,config.maximum_fraction)
