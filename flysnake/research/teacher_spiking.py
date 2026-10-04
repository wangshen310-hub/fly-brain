"""Full signed graph with internal trainable dynamics and a fixed spike decoder.

The only observation boundary is RGB. This module has no teacher, game state,
semantic feature encoder, navigation algorithm or trainable action readout.
Continuous early visual release and surrogate training derivatives are explicit
engineering assumptions, not a claim about natural fruit-fly learning.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .optical_sampling import ApertureConfig, sample_aperture_linear

VERSION = 'complete-signed-RGB-teacher-surrogate-spiking-v1'
ACTIONS = ('left', 'forward', 'right')
STATE_ATOL = 1e-6  # Pre-training bound: 0.00002 mV for normalized voltage.


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@dataclass(frozen=True)
class Dynamics:
    dt_ms: float = 1.
    window_ms: int = 100
    delay_steps: int = 2
    refractory_steps: int = 2
    synapse_ms: float = 5.
    threshold: float = .75
    exc_reversal: float = 3.
    inh_reversal: float = -.5
    derivative_width: float = .15
    margin: float = .2

    def __post_init__(self):
        if (self.dt_ms != 1. or type(self.window_ms) is not int or self.window_ms < 10
                or type(self.delay_steps) is not int or self.delay_steps < 1
                or type(self.refractory_steps) is not int or self.refractory_steps < 1
                or not all(math.isfinite(v) for v in asdict(self).values())
                or not 0 < self.margin < 1 or self.derivative_width <= 0
                or self.synapse_ms <= 0 or not self.inh_reversal < 0 < self.threshold < self.exc_reversal):
            raise ValueError('Invalid declared dynamics')


class _FixedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, matrix, transpose):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, x)

    @staticmethod
    def backward(ctx, grad):
        # Sparse matrices are frozen buffers. Only dense neural state gets a
        # derivative; no dense N-by-N gradient is ever constructed.
        return torch.sparse.mm(ctx.transpose, grad.contiguous()), None, None


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_minus_threshold, width):
        ctx.save_for_backward(voltage_minus_threshold)
        ctx.width = width
        return (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        derivative = 1. / (2. * ctx.width * (1. + x.abs() / ctx.width).square())
        return grad * derivative, None


def decode_spikes(counts, margin=.2):
    if set(counts) != {'left', 'right', 'forward'} or not 0 < margin < 1:
        raise ValueError('Fixed motor groups required')
    values = counts['left'] + counts['right'] + counts['forward']
    if (len(counts['left']) != 1 or len(counts['right']) != 1
            or len(counts['forward']) != 2
            or any(type(v) is not int or v < 0 for v in values)):
        raise ValueError('Actual nonnegative integer spike counts required')
    left, right = counts['left'][0], counts['right'][0]
    gap = margin * max(left, right, 1)
    action = None if not sum(values) else (
        'forward' if abs(right-left) <= gap else 'right' if right > left else 'left')
    return {'action': action, 'reason': 'silent' if action is None else 'fixed_raw_turn',
        'difference': right-left, 'balance_limit': gap, 'counts': counts}


class SpikingConnectome(nn.Module):
    def __init__(self, arrays, manifest, *, device='cuda', seed=0, config=Dynamics()):
        super().__init__()
        self.config, self.n = config, len(arrays['ids'])
        self.manifest = manifest
        self.graph_identity = manifest.get('graph_sha256', 'test-fixture')
        self.device_name = device
        dev = torch.device(device)
        torch.use_deterministic_algorithms(True)
        for prefix in ('exc', 'inh'):
            for kind, marker in [('matrix', ''), ('transpose', 't')]:
                matrix = torch.sparse_csr_tensor(
                    torch.as_tensor(arrays[prefix+'_'+marker+'crow'], dtype=torch.int32, device=dev),
                    torch.as_tensor(arrays[prefix+'_'+marker+'col'], dtype=torch.int32, device=dev),
                    torch.as_tensor(arrays[prefix+'_'+marker+'value'], dtype=torch.float32, device=dev),
                    size=(self.n, self.n), device=dev, check_invariants=True)
                self.register_buffer(prefix+'_'+kind, matrix, persistent=False)
        def buffer(name, value, dtype):
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype, device=dev), persistent=False)
        buffer('base_rest', arrays['rest'], torch.float32)
        graded = np.unique(np.r_[arrays['retina'], arrays['r8'], arrays['graded_lamina']])
        mask = np.zeros(self.n, bool); mask[graded] = True
        buffer('graded_mask', mask, torch.bool)
        receptor = np.r_[arrays['retina'], arrays['r8']]
        buffer('receptor', receptor, torch.long)
        buffer('lamina', arrays['graded_lamina'], torch.long)
        buffer('kc', arrays['KC'], torch.long)
        motor = np.r_[arrays['left'], arrays['right'], arrays['forward_left'], arrays['forward_right']]
        buffer('motor', motor, torch.long)
        plastic_bias = np.ones(self.n, np.float32)
        plastic_bias[np.r_[motor, receptor]] = 0.
        buffer('bias_mask', plastic_bias, torch.float32)
        motor_drive = np.zeros(self.n, np.float32); motor_drive[motor[:2]] = .6
        buffer('motor_drive', motor_drive, torch.float32)
        rng = np.random.default_rng(seed)
        self.rest_delta = nn.Parameter(torch.as_tensor(rng.normal(0, .002, self.n), dtype=torch.float32, device=dev))
        self.gain_raw = nn.Parameter(torch.full((self.n,), math.log(.95/3.), device=dev))
        self.tau_raw = nn.Parameter(torch.full((self.n,), math.log(15./20.), device=dev))
        self.uv = np.array(arrays['uv'], copy=True)
        self.r8_uv = np.array(arrays['r8_uv'], copy=True)
        self.r8_channel = np.array(arrays['r8_channel'], copy=True)
        self.retina_count = len(arrays['retina'])
        self.aperture = ApertureConfig(radius_pixels=2.25, subdivisions=3)
        self.motor_ids = {k: [int(arrays['ids'][i]) for i in arrays[k]] for k in
                          ('left', 'right', 'forward_left', 'forward_right')}
        self.identity = {'version': VERSION, 'graph_sha256': self.graph_identity,
            'dynamics': asdict(config), 'source_sha256': file_hash(__file__),
            'optics_source_sha256': file_hash(Path(__file__).with_name('optical_sampling.py')),
            'motor_ids': self.motor_ids, 'input': 'fixed RGB uint8 only',
            'numerics': {'torch':str(torch.__version__),'cuda':torch.version.cuda,
                'device_type':dev.type,'float_state_atol':STATE_ATOL,
                'integer_spikes_and_timing':'exact',
                'GPU_bitwise_state_replay':False},
            'teacher_or_game_state_at_inference': False, 'external_action_network': False,
            'learned': ['neuron incoming gain', 'intrinsic rest delta', 'membrane time constant'],
            'fixed': ['all graph endpoints', 'fast signs', 'optical sampling', 'motor decoder'],
            'assumptions': 'New normalized conductance dynamics and graded early vision; '
                'surrogate derivative during engineering training only. Not legacy native equivalence.'}

    @classmethod
    def load_graph(cls, path, **kwargs):
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if (manifest['neurons'], manifest['anatomical_edges']) != (166700, 25582938):
            raise ValueError('Complete pinned MaleCNS export required')
        if file_hash(path/'graph.npz') != manifest['graph_sha256']:
            raise ValueError('Export graph bytes changed')
        with np.load(path/'graph.npz', allow_pickle=False) as arrays:
            return cls(arrays, manifest, **kwargs)

    def encode(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (180, 320, 3):
            raise ValueError('Only B x 180 x 320 x 3 uint8 RGB is accepted')
        result = []
        for frame in rgb:
            light = sample_aperture_linear(frame, self.uv, self.aperture) @ np.array([.2126,.7152,.0722])
            color = (sample_aperture_linear(frame, self.r8_uv, self.aperture)
                     if len(self.r8_uv) else np.empty((0,3)))
            r8 = color[np.arange(len(self.r8_channel)), self.r8_channel]
            result.append(np.r_[light, r8])
        return torch.as_tensor(np.array(result).T.copy(), dtype=torch.float32, device=self.base_rest.device)

    def zero_state(self, batch):
        v = self.base_rest[:, None].expand(self.n, batch).clone()
        zero = torch.zeros_like(v)
        return {'v': v, 'ge': zero.clone(), 'gi': zero.clone(),
            'adaptation': zero.clone(),
            'refractory': torch.zeros_like(v, dtype=torch.int32),
            'release_queue': torch.zeros(self.config.delay_steps, self.n, batch,
                                         device=v.device), 'cursor': 0, 'ticks': 0}

    @staticmethod
    def detach_state(state):
        return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k,v in state.items()}

    def integrate_encoded(self, light, state=None, *, visual_enabled=True, steps=None):
        if (light.ndim != 2 or light.shape[0] != len(self.receptor) or light.shape[1]<1
                or light.dtype!=torch.float32 or light.device!=self.base_rest.device
                or not torch.isfinite(light).all()):
            raise ValueError('Encoded receptor radiance dimension mismatch')
        batch = light.shape[1]
        if state is not None:
            self.validate_state(state)
            if state['v'].shape[1]!=batch:raise ValueError('Neural continuation batch mismatch')
        state = self.zero_state(batch) if state is None else state
        cfg = self.config
        steps = cfg.window_ms if steps is None else steps
        if type(steps) is not int or steps <= 0:
            raise ValueError('Positive integral neural steps required')
        v, ge, gi = state['v'], state['ge'], state['gi']
        adaptation = state['adaptation']
        refractory = state['refractory']
        queue = list(state['release_queue'].unbind(0))
        cursor = state['cursor']
        counts = torch.zeros_like(v)
        rest = self.base_rest[:, None] + .4 * torch.tanh(self.rest_delta)[:, None] * self.bias_mask[:, None]
        target_rest = rest + self.motor_drive[:, None]
        gain = (.05 + 3.95 * torch.sigmoid(self.gain_raw))[:, None]
        tau = (5. + 35. * torch.sigmoid(self.tau_raw))[:, None]
        syn_decay = math.exp(-cfg.dt_ms/cfg.synapse_ms)
        receptor_release = (.005 + .055*light.clamp(0,1)) * cfg.dt_ms if visual_enabled else torch.zeros_like(light)
        for _ in range(steps):
            adaptation = adaptation*math.exp(-cfg.dt_ms/200.)
            arriving = queue[cursor]
            ge = ge*syn_decay + gain*_FixedMM.apply(arriving, self.exc_matrix, self.exc_transpose)
            gi = gi*syn_decay + gain*_FixedMM.apply(arriving, self.inh_matrix, self.inh_transpose)
            total = 1. + ge + gi + adaptation
            equilibrium = (target_rest + cfg.exc_reversal*ge + cfg.inh_reversal*(gi+adaptation))/total
            v = equilibrium + (v-equilibrium)*torch.exp(-cfg.dt_ms*total/tau)
            active = refractory == 0
            spike = _Spike.apply(v-cfg.threshold, cfg.derivative_width)
            spike = spike * active * (~self.graded_mask[:, None])
            counts = counts + spike
            # Detached reset avoids letting a fictitious derivative through the
            # discrete reset overwrite the documented surrogate spike gradient.
            reset = spike.detach().bool()
            v = torch.where(reset | ~active, rest, v)
            refractory = torch.where(reset, cfg.refractory_steps,
                                       (refractory-1).clamp_min(0))
            adaptation_next = adaptation.clone()
            adaptation_next[self.kc] = adaptation[self.kc] + .4*spike[self.kc]
            adaptation = adaptation_next
            release = spike.clone()
            lamina_rate = (50.*(1.+(v[self.lamina]-rest[self.lamina])/.6)).clamp(0,100)
            release[self.lamina] = lamina_rate*cfg.dt_ms/1000.
            release[self.receptor] = receptor_release
            queue[cursor] = release
            cursor = (cursor+1) % cfg.delay_steps
        new_state = {'v': v, 'ge': ge, 'gi': gi, 'adaptation': adaptation, 'refractory': refractory,
            'release_queue': torch.stack(queue), 'cursor': cursor, 'ticks': state['ticks']+steps}
        motor = counts[self.motor].T
        return counts, motor, new_state

    def logits(self, motor):
        # Fixed differentiable proxy of the SAME raw pulse margin, never a
        # trained policy. The numerical forward motor counts are actual spikes.
        left, right = motor[:,0], motor[:,1]
        difference = right-left
        limit = self.config.margin*torch.maximum(torch.maximum(left,right), torch.ones_like(left))
        return torch.stack((-difference-limit, torch.zeros_like(left), difference-limit), dim=1)

    def act(self, rgb, state=None, *, visual_enabled=True):
        with torch.no_grad():
            light = self.encode(np.asarray(rgb)[None])
            all_counts, motor, state = self.integrate_encoded(light, state, visual_enabled=visual_enabled)
            raw = motor[0].cpu().to(torch.int64).tolist()
            counts = {'left':[raw[0]], 'right':[raw[1]], 'forward':raw[2:]}
            result = decode_spikes(counts, self.config.margin)
            result.update(version=VERSION, selected_cell_ids=self.motor_ids,
                input_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
                full_counts_sha256=hashlib.sha256(all_counts.cpu().numpy().tobytes()).hexdigest(),
                actual_total_spikes=int(all_counts.sum().item()),
                voltage_min=float(state['v'].min()), voltage_max=float(state['v'].max()),
                start_tick=state['ticks']-self.config.window_ms, end_tick=state['ticks'])
            return result, self.detach_state(state)

    def parameter_hash(self):
        h=hashlib.sha256()
        for name,p in self.named_parameters():
            h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def validate_state(self, state):
        """Reject incomplete or incompatible continuation before using it."""
        keys={'v','ge','gi','adaptation','refractory','release_queue','cursor','ticks'}
        if not isinstance(state,dict) or set(state)!=keys:
            raise ValueError('Neural continuation schema mismatch')
        v=state['v']
        if (not isinstance(v,torch.Tensor) or v.ndim!=2 or v.shape[0]!=self.n
                or v.shape[1]<1):
            raise ValueError('Neural continuation shape mismatch')
        for name in ('v','ge','gi','adaptation','release_queue'):
            value=state[name]
            expected=(self.config.delay_steps,*v.shape) if name=='release_queue' else v.shape
            if (not isinstance(value,torch.Tensor) or value.shape!=expected
                    or value.dtype!=torch.float32 or value.device!=self.base_rest.device
                    or not torch.isfinite(value).all()
                    or (name!='v' and (value<0).any())):
                raise ValueError('Invalid neural continuation '+name)
        refractory=state['refractory']
        if (not isinstance(refractory,torch.Tensor) or refractory.shape!=v.shape
                or refractory.dtype!=torch.int32 or refractory.device!=v.device
                or (refractory<0).any() or (refractory>self.config.refractory_steps).any()
                or type(state['ticks']) is not int or state['ticks']<0
                or type(state['cursor']) is not int
                or state['cursor']!=state['ticks']%self.config.delay_steps):
            raise ValueError('Invalid neural continuation timing')

    def save_checkpoint(self, path, state=None, **extra):
        path=Path(path)
        if path.exists():raise FileExistsError('Immutable checkpoint generation required')
        if state is not None:self.validate_state(state)
        payload={'identity':self.identity, 'parameters':{k:v.detach().cpu() for k,v in self.named_parameters()},
                 'state':None if state is None else {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()},
                 'parameter_sha256':self.parameter_hash(), 'extra':extra}
        temp=path.with_suffix(path.suffix+'.partial')
        with temp.open('xb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        temp.replace(path)
        return {'path':str(path),'sha256':file_hash(path),'parameter_sha256':payload['parameter_sha256']}

    def restore_checkpoint(self, path, sha256):
        if file_hash(path)!=sha256:raise ValueError('Checkpoint integrity mismatch')
        payload=torch.load(path,map_location=self.base_rest.device,weights_only=True)
        if payload['identity']!=self.identity:raise ValueError('Checkpoint source/graph/dynamics mismatch')
        parameters=dict(self.named_parameters())
        if set(parameters)!=set(payload['parameters']):raise ValueError('Parameter schema mismatch')
        h=hashlib.sha256()
        for name,value in parameters.items():
            saved=payload['parameters'][name]
            if (not isinstance(saved,torch.Tensor) or saved.shape!=value.shape
                    or saved.dtype!=value.dtype or not torch.isfinite(saved).all()):
                raise ValueError('Invalid neural parameters')
            h.update(name.encode());h.update(saved.detach().cpu().numpy().tobytes())
        if h.hexdigest()!=payload['parameter_sha256']:
            raise ValueError('Parameter replay mismatch')
        if payload['state'] is not None:self.validate_state(payload['state'])
        with torch.no_grad():
            for name,value in payload['parameters'].items():
                parameters[name].copy_(value)
        if self.parameter_hash()!=payload['parameter_sha256']:raise ValueError('Parameter replay mismatch')
        return payload['state'],payload['extra']
