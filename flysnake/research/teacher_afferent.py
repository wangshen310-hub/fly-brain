"""Plasticity on 2319 existing DNa02 input synapses, with fixed upstream brain.

The anatomical graph, every existing fast sign, RGB encoder, real threshold
spikes and fixed action decoder remain the base protocol. Individual existing
synapses can change efficacy; no external action network or new edge exists.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .teacher_spiking import SpikingConnectome,_FixedMM,_Spike,file_hash

VERSION='complete-signed-RGB-existing-turn-afferent-plasticity-v2'


class PlasticAfferentBrain(SpikingConnectome):
    def __init__(self,arrays,manifest,**kwargs):
        super().__init__(arrays,manifest,**kwargs)
        self.base_identity=dict(self.identity)
        for parameter in self.parameters():parameter.requires_grad_(False)
        motor=self.motor[:2].cpu().numpy();post=np.asarray(arrays['post']);pre=np.asarray(arrays['pre']);weight=np.asarray(arrays['signed_base'])
        selected=np.flatnonzero(np.isin(post,motor)&(weight!=0))
        if self.n==166700 and len(selected)!=2319:raise ValueError('Pinned 2319 existing turn input synapses required')
        dev=self.base_rest.device
        self.register_buffer('aff_pre',torch.as_tensor(pre[selected],dtype=torch.long,device=dev),persistent=False)
        self.register_buffer('aff_post',torch.as_tensor(post[selected],dtype=torch.long,device=dev),persistent=False)
        self.register_buffer('aff_abs',torch.as_tensor(np.abs(weight[selected]),dtype=torch.float32,device=dev),persistent=False)
        self.register_buffer('aff_exc',torch.as_tensor(weight[selected]>0,device=dev),persistent=False)
        self.aff_targets=[int(x) for x in motor]
        for channel in ('exc','inh'):
            for slot,target in enumerate(motor):
                select_channel=weight[selected]>0 if channel=='exc' else weight[selected]<0
                indices=np.flatnonzero(select_channel & (post[selected]==target))
                self.register_buffer(f'aff_{channel}_{slot}',torch.as_tensor(indices,dtype=torch.long,device=dev),persistent=False)
        self.afferent_log_gain=nn.Parameter(torch.zeros(len(selected),device=dev))
        self.identity={**self.identity,'version':VERSION,'afferent_source_sha256':file_hash(__file__),
            'afferent_edges':len(selected),'afferent_identity_sha256':hashlib.sha256(np.stack([pre[selected],post[selected]],axis=1).tobytes()+weight[selected].tobytes()).hexdigest(),
            'learned':['Individual efficacy of existing DNa02 incoming synapses only'],
            'frozen_upstream_neuron_parameters':True,'gain_factor_bound':[math.exp(-4.),math.exp(4.)],
            'assumptions':self.identity['assumptions']+' Existing afferent plasticity is an engineering locus, not established in-vivo fruit-fly learning.'}

    def warm_start_checkpoint(self,path,sha256):
        if file_hash(path)!=sha256:raise ValueError('Warm checkpoint integrity mismatch')
        payload=torch.load(path,map_location=self.base_rest.device,weights_only=True)
        if payload['identity']!=self.base_identity:raise ValueError('Only the locked compatible upstream protocol can seed this model')
        names=('rest_delta','gain_raw','tau_raw');h=hashlib.sha256()
        for name in names:
            value=payload['parameters'][name];target=getattr(self,name)
            if value.shape!=target.shape or value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('Invalid warm neuronal parameters')
            h.update(name.encode());h.update(value.cpu().numpy().tobytes())
        if h.hexdigest()!=payload['parameter_sha256']:raise ValueError('Warm parameter hash mismatch')
        with torch.no_grad():
            for name in names:getattr(self,name).copy_(payload['parameters'][name])

    def delivery(self,arriving,prefix):
        delivered=_FixedMM.apply(arriving,getattr(self,prefix+'_matrix'),getattr(self,prefix+'_transpose'))
        correction=self.aff_abs*(torch.exp(4.*torch.tanh(self.afferent_log_gain))-1.)
        # Fixed ordered reductions for two existing targets. Corrections change
        # only the gain of anatomical afferents already included in base CSR.
        outputs=[]
        for slot,target in enumerate(self.aff_targets):
            indices=getattr(self,f'aff_{prefix}_{slot}')
            outputs.append((correction[indices,None]*arriving[self.aff_pre[indices]]).sum(dim=0))
        result=delivered.clone()
        result[self.motor[:2]]=delivered[self.motor[:2]]+torch.stack(outputs)
        return result

    def integrate_encoded(self,light,state=None,*,visual_enabled=True,steps=None):
        if (light.ndim!=2 or light.shape[0]!=len(self.receptor) or light.shape[1]<1 or light.dtype!=torch.float32
                or light.device!=self.base_rest.device or not torch.isfinite(light).all()):raise ValueError('Encoded radiance mismatch')
        batch=light.shape[1]
        if state is not None:
            self.validate_state(state)
            if state['v'].shape[1]!=batch:raise ValueError('Neural batch mismatch')
        state=self.zero_state(batch) if state is None else state;cfg=self.config
        steps=cfg.window_ms if steps is None else steps
        if type(steps) is not int or steps<=0:raise ValueError('Positive neural steps required')
        v,ge,gi=state['v'],state['ge'],state['gi'];adaptation=state['adaptation'];refractory=state['refractory']
        queue=list(state['release_queue'].unbind(0));cursor=state['cursor'];counts=torch.zeros_like(v)
        rest=self.base_rest[:,None]+.4*torch.tanh(self.rest_delta)[:,None]*self.bias_mask[:,None]
        target_rest=rest+self.motor_drive[:,None]
        gain=(.05+3.95*torch.sigmoid(self.gain_raw))[:,None];tau=(5.+35.*torch.sigmoid(self.tau_raw))[:,None]
        decay=math.exp(-cfg.dt_ms/cfg.synapse_ms)
        receptor_release=(.005+.055*light.clamp(0,1))*cfg.dt_ms if visual_enabled else torch.zeros_like(light)
        for _ in range(steps):
            adaptation=adaptation*math.exp(-cfg.dt_ms/200.);arriving=queue[cursor]
            ge=ge*decay+gain*self.delivery(arriving,'exc');gi=gi*decay+gain*self.delivery(arriving,'inh')
            total=1.+ge+gi+adaptation
            equilibrium=(target_rest+cfg.exc_reversal*ge+cfg.inh_reversal*(gi+adaptation))/total
            v=equilibrium+(v-equilibrium)*torch.exp(-cfg.dt_ms*total/tau);active=refractory==0
            spike=_Spike.apply(v-cfg.threshold,cfg.derivative_width)*active*(~self.graded_mask[:,None]);counts=counts+spike
            reset=spike.detach().bool();v=torch.where(reset | ~active,rest,v)
            refractory=torch.where(reset,cfg.refractory_steps,(refractory-1).clamp_min(0))
            newer=adaptation.clone();newer[self.kc]=adaptation[self.kc]+.4*spike[self.kc];adaptation=newer
            release=spike.clone();lamina_rate=(50.*(1.+(v[self.lamina]-rest[self.lamina])/.6)).clamp(0,100)
            release[self.lamina]=lamina_rate*cfg.dt_ms/1000.;release[self.receptor]=receptor_release
            queue[cursor]=release;cursor=(cursor+1)%cfg.delay_steps
        state={'v':v,'ge':ge,'gi':gi,'adaptation':adaptation,'refractory':refractory,'release_queue':torch.stack(queue),
            'cursor':cursor,'ticks':state['ticks']+steps}
        return counts,counts[self.motor].T,state

    def act(self,*args,**kwargs):
        receipt,state=super().act(*args,**kwargs)
        receipt['model_version']=VERSION
        return receipt,state
