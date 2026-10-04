"""Local, graded APL cable on actual MaleCNS skeletons and synaptic coordinates.

Only the two APL point cells are replaced. Native fast events, modulatory events,
gamma1 plasticity and raw motor-spike decoder remain separate and unchanged.
Cable constants/release curve are explicit engineering hypotheses, not measured
adult-male electrical parameters. No game coordinates or action target enter.
"""
from dataclasses import dataclass, asdict
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyarrow.feather as feather
from .input_isolation import IsolatedInputBrain, InputConfig
from .compartment_native import CompartmentConfig
from .paired import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / 'kernels/apl_local.cpp'
VERSION = 'native-spatial-graded-APL-cable-v1'
APL_IDS = (10540, 10977)


@dataclass(frozen=True)
class LocalAPLConfig:
    space_constant_um: float = 25.
    membrane_tau_ms: float = 20.
    cable_dt_ms: float = .5
    input_reference_um: float = 1.
    release_half_mV: float = 15.
    release_max_hz: float = 100.
    release_gain: float = 1.
    inhibitory_reversal_mV: float = -70.

    def validate(self):
        values=np.array(list(asdict(self).values()),float)
        if (not np.isfinite(values).all() or min(values[:6])<=0 or self.release_gain<0
                or not .1<=self.cable_dt_ms<=1. or not np.isclose(self.cable_dt_ms/.1,round(self.cable_dt_ms/.1))
                or self.inhibitory_reversal_mV>=-60. or self.space_constant_um>500.):
            raise ValueError('Finite positive cable constants, integral native stride and inhibitory reversal required')


def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    library=directory/'apl_local.dll';record_path=directory/'build.json';sha=file_hash(SOURCE)
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding='utf-8'))
        if record['source_sha256']!=sha or record['binary_sha256']!=file_hash(library):
            raise ValueError('Preserve the locked local APL build; choose a new directory')
        return library,record
    command=[sys.executable,'-m','ziglang','c++','-O3','-std=c++17','-shared',
             '-target','x86_64-windows-gnu',str(SOURCE),'-o',str(library)]
    with (directory/'compile.log').open('w',encoding='utf-8') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Local APL compilation failed; inspect compile.log')
    record={'version':VERSION,'source_sha256':sha,'binary_sha256':file_hash(library),'command':command}
    record_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    return library,record


def attach_segments(library, xyz, child, parent, queries):
    xyz=np.ascontiguousarray(xyz,np.float64);q=np.ascontiguousarray(queries,np.float64)
    child=np.ascontiguousarray(child,np.int32);parent=np.ascontiguousarray(parent,np.int32)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or q.ndim!=2 or q.shape[1]!=3
            or not np.isfinite(xyz).all() or not np.isfinite(q).all() or not len(child)
            or child.shape!=parent.shape or min(child.min(),parent.min())<0
            or max(child.max(),parent.max())>=len(xyz)):
        raise ValueError('Valid finite skeleton segments and queries required')
    edge=np.empty(len(q),np.int32);alpha=np.empty(len(q));distance=np.empty(len(q))
    function=C.CDLL(str(Path(library).resolve())).apl_attach_segments
    function.argtypes=[C.c_int]+[C.c_void_p]*3+[C.c_int]+[C.c_void_p]*4;function.restype=None
    function(len(child),xyz.ctypes.data,child.ctypes.data,parent.ctypes.data,len(q),q.ctypes.data,
             edge.ctypes.data,alpha.ctypes.data,distance.ctypes.data)
    return edge,alpha,distance


def forest_order(parent):
    parent=np.asarray(parent,np.int32)
    if parent.ndim!=1 or (parent<-1).any() or (parent>=len(parent)).any():raise ValueError('Invalid forest parent')
    children=[[] for _ in parent]
    for i,p in enumerate(parent):
        if p>=0:children[p].append(i)
    order=np.flatnonzero(parent<0).tolist()
    for i in order:order.extend(children[i])
    if len(order)!=len(parent):raise ValueError('Forest contains a cycle')
    order=np.array(order,np.int32);inverse=np.empty(len(order),np.int32);inverse[order]=np.arange(len(order))
    return order,inverse


def cable_geometry(xyz, parent, space_constant_um):
    xyz=np.asarray(xyz,float);parent=np.asarray(parent,np.int32)
    if (xyz.shape!=(len(parent),3) or not np.isfinite(xyz).all() or not np.isfinite(space_constant_um)
            or space_constant_um<=0 or (parent>=np.arange(len(parent))).any() or (parent<-1).any()):
        raise ValueError('Finite cable with parents preceding children required')
    child=np.flatnonzero(parent>=0);length=np.linalg.norm(xyz[child]-xyz[parent[child]],axis=1)
    if (length<=0).any():raise ValueError('Zero-length cable edge')
    volume=np.zeros(len(parent));np.add.at(volume,child,length/2);np.add.at(volume,parent[child],length/2)
    if (volume<=0).any():raise ValueError('Isolated zero-volume node cannot have invented cable length')
    axial=np.zeros(len(parent));axial[child]=space_constant_um**2/length
    diagonal=axial.copy();np.add.at(diagonal,parent[child],axial[child])
    return volume,axial,diagonal


def edge_lookup(b, pre, post):
    keys,inverse=np.unique(np.asarray(pre,np.int64)*b.n+np.asarray(post,np.int64),return_inverse=True)
    pre=(keys//b.n).astype(np.int32);post=(keys%b.n).astype(np.int32);result=np.empty(len(keys),np.int64)
    for source in np.unique(pre):
        select=np.flatnonzero(pre==source);edges=np.arange(b.ptr[source],b.ptr[source+1],dtype=np.int64)
        sort=np.argsort(b.post[edges]);ordered=b.post[edges][sort];position=np.searchsorted(ordered,post[select])
        if (position>=len(ordered)).any() or not np.array_equal(ordered[position],post[select]):
            raise ValueError('Spatial contact lacks a retained original edge')
        result[select]=edges[sort[position]]
    return result[inverse]


def prepare_attachment(b, directory, library, data):
    """Map every retained APL-related contact once, without artificial new wires."""
    data=Path(data);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    status=json.loads((data/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete' or status['APL_contacts']['sha256']!=file_hash(data/'apl-partners.feather'):
        raise ValueError('Completed verified APL contact data required')
    table=feather.read_table(data/'apl-partners.feather')
    retained=table['retained_native_contact'].to_numpy();table=table.filter(retained)
    body_pre=table['body_pre'].to_numpy();body_post=table['body_post'].to_numpy()
    pre=table['pre_native_index'].to_numpy();post=table['post_native_index'].to_numpy()
    if not np.array_equal(b.ids[pre],body_pre) or not np.array_equal(b.ids[post],body_post):
        raise ValueError('Spatial table native identity differs')
    edge=edge_lookup(b,pre,post);unique,count=np.unique(edge,return_counts=True)
    fraction=1./count[np.searchsorted(unique,edge)]
    local_child_pre=np.full(len(table),-1,np.int32);local_parent_pre=local_child_pre.copy()
    local_child_post=local_child_pre.copy();local_parent_post=local_child_pre.copy()
    alpha_pre=np.zeros(len(table));alpha_post=np.zeros(len(table));dist_pre=np.full(len(table),np.nan);dist_post=dist_pre.copy()
    parents=[];points=[];components=[];bodies=[];offset=0;geometry=[]
    for body in APL_IDS:
        path=data/f'apl-{body}-skeleton.npz'
        if file_hash(path)!=status['skeletons'][str(body)]['normalized_sha256']:raise ValueError('APL skeleton changed')
        with np.load(path,allow_pickle=False) as z:
            xyz=z['xyz_um'].copy();original_parent=z['parent_index'].copy();component=z['component'].copy()
        order,inverse=forest_order(original_parent);child=np.flatnonzero(original_parent>=0).astype(np.int32)
        parent=original_parent[child];newparent=np.where(original_parent[order]>=0,
            inverse[np.maximum(original_parent[order],0)]+offset,-1).astype(np.int32)
        parents.append(newparent);points.append(xyz[order]);components.append(component[order]);bodies.append(np.full(len(order),body,np.int64))
        for side,original,lc,lp,alpha,dist in (
                ('pre',body_pre,local_child_pre,local_parent_pre,alpha_pre,dist_pre),
                ('post',body_post,local_child_post,local_parent_post,alpha_post,dist_post)):
            select=np.flatnonzero(original==body)
            queries=np.c_[*[table[f'{axis}_{side}'].to_numpy()[select] for axis in 'xyz']]*.008
            found,a,d=attach_segments(library,xyz,child,parent,queries)
            lc[select]=inverse[child[found]]+offset;lp[select]=inverse[parent[found]]+offset
            alpha[select]=a;dist[select]=d
            geometry.append({'body':body,'side':side,'contacts':len(select),
                'distance_um_quantiles':np.quantile(d,[0,.5,.9,.99,1]).tolist(),
                'over_5um':int((d>5).sum()),'attached_component_counts':np.bincount(component[child[found]]).tolist()})
        offset+=len(order)
    incoming=np.flatnonzero(np.isin(body_post,APL_IDS) & ~np.isin(body_pre,APL_IDS))
    incoming=incoming[np.argsort(pre[incoming],kind='stable')]
    outgoing=np.flatnonzero(np.isin(body_pre,APL_IDS))
    targets=np.unique(post[outgoing][~np.isin(body_post[outgoing],APL_IDS)]).astype(np.int32)
    out_target=np.full(len(outgoing),-1,np.int32);normal=~np.isin(body_post[outgoing],APL_IDS)
    out_target[normal]=np.searchsorted(targets,post[outgoing][normal])
    in_ptr=np.r_[0,np.cumsum(np.bincount(pre[incoming],minlength=b.n))].astype(np.int64)
    arrays={'parent':np.concatenate(parents),'xyz_um':np.concatenate(points),'body_ids':np.concatenate(bodies),
        'component':np.concatenate(components),'in_ptr':in_ptr,'in_edge':edge[incoming],
        'in_child':local_child_post[incoming],'in_parent':local_parent_post[incoming],
        'in_alpha':alpha_post[incoming],'in_fraction':fraction[incoming],
        'out_edge':edge[outgoing],'out_child':local_child_pre[outgoing],'out_parent':local_parent_pre[outgoing],
        'out_alpha':alpha_pre[outgoing],'out_fraction':fraction[outgoing],'out_target':out_target,
        'out_local_child':local_child_post[outgoing],'out_local_parent':local_parent_post[outgoing],
        'out_local_alpha':alpha_post[outgoing],'targets':targets,
        'retained_global_rows':table['global_partner_row'].to_numpy(),
        'incoming_global_rows':table['global_partner_row'].to_numpy()[incoming],
        'outgoing_global_rows':table['global_partner_row'].to_numpy()[outgoing],
        'all_contact_edge':edge,'distance_pre_um':dist_pre,'distance_post_um':dist_post}
    totals=np.zeros(len(unique));np.add.at(totals,np.searchsorted(unique,edge),fraction)
    if not np.allclose(totals,1.,rtol=0,atol=1e-10):raise AssertionError('Per-contact fractions do not reconstruct edges')
    npz=directory/'attachment.npz'
    with npz.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    report={'version':VERSION,'data_status_sha256':file_hash(data/'status.json'),
        'contact_source_sha256':file_hash(data/'apl-partners.feather'),'attachment_sha256':file_hash(npz),
        'APL_ids':list(APL_IDS),'local_nodes':offset,'retained_contacts':len(table),'retained_pairs':len(unique),
        'incoming_non_APL_contacts':len(incoming),'outgoing_APL_contacts':len(outgoing),
        'APL_to_APL_contacts':int((out_target<0).sum()),'normal_output_targets':len(targets),
        'geometry':geometry,'aggregate_edge_fractions_equal_one':True,
        'method':'Exact nearest segment in original EM space; barycentric input/output; forest components retained',
        'limitations':'Coarse skeleton and nearby branches introduce unresolved attachment ambiguity; no electrical radius fitted'}
    (directory/'attachment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return arrays,report


class LocalAPLKernel:
    def __init__(self, original, library, b, attachment, config):
        self.config=config;config.validate();self.a=attachment;self.n=b.n
        self.library=C.CDLL(str(Path(library).resolve()));self.advance=self.library.memory_advance_apl_local
        self.argtypes=original.argtypes;self.restype=None
        self.advance.argtypes=list(self.argtypes)+[C.c_void_p,C.c_int]+[C.c_void_p]*14+[C.c_int]+[C.c_void_p]*9+[
            C.c_int]+[C.c_void_p]*4+[C.c_int]+[C.c_double]*5+[C.c_float]
        self.advance.restype=None
        self.mask=np.isin(b.ids,APL_IDS).astype(np.uint8)
        self.local_n=len(attachment['parent']);self.targets_n=len(attachment['targets'])
        self.volume,self.axial,self.diagonal_axial=cable_geometry(attachment['xyz_um'],attachment['parent'],config.space_constant_um)
        self.u=np.zeros(self.local_n);self.ge=np.zeros(self.local_n);self.gi=np.zeros(self.local_n);self.drive=np.zeros(self.local_n)
        self.inhibition=np.zeros(b.n,np.float32);self.release_queue=np.zeros((19,self.targets_n),np.float32)
        self.local_release_queue=np.zeros((19,self.local_n))
        self.diagnostic_passthrough=False
        self.zero_mask=np.zeros(b.n,np.uint8);self.zero_in_ptr=np.zeros(b.n+1,np.int64)

    def __call__(self,*args):
        a=self.a;c=self.config
        extra=[self.mask.ctypes.data,self.local_n,*[x.ctypes.data for x in (
            a['parent'],self.axial,self.volume,self.diagonal_axial,self.u,self.ge,self.gi,self.drive,
            a['in_ptr'],a['in_edge'],a['in_child'],a['in_parent'],a['in_alpha'],a['in_fraction'])],
            len(a['out_edge']),*[a[name].ctypes.data for name in (
                'out_edge','out_child','out_parent','out_alpha','out_fraction','out_target',
                'out_local_child','out_local_parent','out_local_alpha')],
            self.targets_n,a['targets'].ctypes.data,self.inhibition.ctypes.data,
            self.release_queue.ctypes.data,self.local_release_queue.ctypes.data,
            round(c.cable_dt_ms/.1),c.membrane_tau_ms,c.input_reference_um,c.release_half_mV,
            c.release_max_hz,c.release_gain,c.inhibitory_reversal_mV]
        if self.diagnostic_passthrough:
            # Read-only equivalence control: no local dynamics or substitution.
            # This flag is never used to choose a gameplay action.
            if np.any(self.inhibition):raise ValueError('Passthrough requires a clean initial state')
            extra[0]=self.zero_mask.ctypes.data;extra[1]=0;extra[10]=self.zero_in_ptr.ctypes.data
            extra[16]=0;extra[26]=0
        self.advance(*args,*extra)


class LocalAPLBrain(IsolatedInputBrain):
    def __init__(self,directory,*,data=ROOT/'runtime/apl-spatial-20261002-v1',attachment=None,
                 apl=LocalAPLConfig(),config=CompartmentConfig(central_headroom_mV=0.),
                 inputs=InputConfig(vision_input=False,lamina_bias_mV=0.,zero_odor_is_silent=True)):
        apl.validate();super().__init__(config=config,inputs=inputs);b=self.b
        directory=Path(directory);library,record=build(directory/'build')
        if attachment is None:arrays,mapping=prepare_attachment(b,directory/'geometry',library,data)
        else:
            attachment=Path(attachment);mapping=json.loads((attachment/'attachment.json').read_text(encoding='utf-8'))
            if file_hash(attachment/'attachment.npz')!=mapping['attachment_sha256']:raise ValueError('Attachment file changed')
            if file_hash(Path(data)/'status.json')!=mapping['data_status_sha256']:raise ValueError('Attachment data differs')
            with np.load(attachment/'attachment.npz',allow_pickle=False) as z:arrays={name:z[name].copy() for name in z.files}
        self.kernel=LocalAPLKernel(b.advance,library,b,arrays,apl);b.advance=self.kernel
        self.apl_config=apl;self.mapping=mapping
        state={'apl_local_u':self.kernel.u,'apl_local_ge':self.kernel.ge,'apl_local_gi':self.kernel.gi,
               'apl_local_drive':self.kernel.drive,'apl_inhibition':self.kernel.inhibition,
               'apl_release_queue':self.kernel.release_queue,'apl_local_release_queue':self.kernel.local_release_queue}
        for name,value in state.items():setattr(b,name,value);b.fields.append(name);b.initial[name]=value.copy()
        self.identity.update(version=VERSION,local_APL_config=asdict(apl),local_APL_mapping=mapping,
            local_APL_kernel=record,local_APL_source_sha256=file_hash(__file__),
            physiological_fit=False,point_APL_spikes_disabled=True,
            existing_non_APL_fast_event_dynamics_retained=True,
            assumptions='Constant-radius finite-volume cable; implicit tree solve; graded saturating release with zero '
                'resting output; APL-only conductance receiver. Other inhibitory currents remain legacy/unfitted. '
                'Monoamine input remains modulatory at original APL identity, without an invented fast cable receptor.')

    def integrate(self,rgb,duration_ms,concentrations=None,*,stimulation=None,learning=False):
        result=super().integrate(rgb,duration_ms,concentrations,stimulation=stimulation,learning=learning)
        k=self.kernel
        if not np.isfinite(k.u).all() or not np.isfinite(k.inhibition).all():raise FloatingPointError('Nonfinite local APL state')
        if np.any(self.last_counts[k.mask.astype(bool)]):raise AssertionError('Graded APL emitted a point spike')
        result.update(protocol=VERSION,APL_point_spikes=0,APL_local_u_mV_quantiles=np.quantile(k.u,[0,.5,.9,.99,1]).tolist(),
            APL_receiver_conductance_sum=float(k.inhibition.sum()))
        return result

    def save_state(self,path):
        metadata={'identity':self.identity,'cursor':self.b.cursor,'neural_ms':self.neural_ms,
                  'sim_ms':self.b.sim_ms,'total_spikes':self.b.total_spikes,'weights_sha256':digest(self.b.weight)}
        arrays={name:getattr(self.b,name) for name in self.b.fields}
        arrays.update(plastic_weight=self.b.weight[self.edges],retinal_light=self.retinal_light,r8_light=self.r8_light,
                      metadata=np.array(json.dumps(metadata,sort_keys=True)))
        with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        return file_hash(path)

    def load_state(self,path,sha):
        if file_hash(path)!=sha:raise ValueError('Local APL state hash differs')
        with np.load(path,allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata']))
            if metadata['identity']!=json.loads(json.dumps(self.identity)):raise ValueError('Local APL state identity differs')
            for name in self.b.fields:
                value=z[name];expected=getattr(self.b,name)
                if value.shape!=expected.shape or value.dtype!=expected.dtype or not np.isfinite(value).all():
                    raise ValueError('Local APL state array differs')
            weights=z['plastic_weight'];current=self.b.weight.copy();current[self.edges]=weights
            if digest(current)!=metadata['weights_sha256']:raise ValueError('Full graph state weight hash differs')
            for name in self.b.fields:getattr(self.b,name)[:]=z[name]
            self.b.weight[self.edges]=weights;self.retinal_light[:]=z['retinal_light'];self.r8_light[:]=z['r8_light']
        self.b.cursor=metadata['cursor'];self.neural_ms=metadata['neural_ms'];self.b.sim_ms=metadata['sim_ms'];self.b.total_spikes=metadata['total_spikes']
        return metadata
