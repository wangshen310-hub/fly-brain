"""Validation of published APL morphology and contact-to-graph correspondence.

No neural dynamics, synthetic coordinates, action policy, or game labels.
All coordinates use the official MaleCNS EM voxel space (8 nm per voxel).
Disconnected skeleton components are retained and reported, never glued together.
"""
from io import StringIO
import json
import numpy as np

VERSION='malecns-apl-spatial-contact-data-v1'
APL_IDS=(10540,10977)
MICRONS_PER_VOXEL=.008


def parse_swc(text):
    metadata=[]
    for line in text.splitlines():
        if line.startswith('#${'):
            try:metadata.append(json.loads(line[2:]))
            except json.JSONDecodeError:raise ValueError('Invalid SWC JSON metadata')
    array=np.loadtxt(StringIO(text),comments='#',dtype=np.float64,ndmin=2)
    if array.ndim!=2 or array.shape[1]!=7 or len(array)==0 or not np.isfinite(array).all():
        raise ValueError('Finite nonempty seven-column SWC required')
    integer_columns=array[:,[0,1,6]]
    if not np.array_equal(integer_columns,np.floor(integer_columns)):
        raise ValueError('SWC identifiers must be integers')
    ids=array[:,0].astype(np.int64);parent_ids=array[:,6].astype(np.int64)
    if (ids<=0).any() or len(np.unique(ids))!=len(ids) or (array[:,5]<0).any():
        raise ValueError('Unique positive SWC node IDs and nonnegative radii required')
    lookup={int(cell):i for i,cell in enumerate(ids)}
    parents=np.empty(len(ids),np.int32)
    for i,cell in enumerate(parent_ids):
        if cell==-1:parents[i]=-1
        elif int(cell) in lookup:parents[i]=lookup[int(cell)]
        else:raise ValueError('SWC parent refers to a missing node')
    # Linear-time directed forest check; no requirement that the published
    # coarse morphology have one root, and no fabricated component joining.
    color=np.zeros(len(ids),np.uint8);components=np.full(len(ids),-1,np.int32)
    roots=np.flatnonzero(parents<0);root_component={int(r):i for i,r in enumerate(roots)}
    for start in range(len(ids)):
        if color[start]==2:continue
        chain=[];node=start
        while node>=0 and color[node]==0:
            color[node]=1;chain.append(node);node=int(parents[node])
        if node>=0 and color[node]==1:raise ValueError('SWC contains a directed cycle')
        component=int(components[node]) if node>=0 else root_component[chain[-1]]
        for node in reversed(chain):color[node]=2;components[node]=component
    xyz_voxel=array[:,2:5].copy();xyz_um=xyz_voxel*MICRONS_PER_VOXEL
    children=np.flatnonzero(parents>=0)
    lengths=np.linalg.norm(xyz_um[children]-xyz_um[parents[children]],axis=1)
    return {'node_ids':ids,'parent_index':parents,'kind':array[:,1].astype(np.int32),
            'xyz_voxel':xyz_voxel,'xyz_um':xyz_um,'radius_um':array[:,5]*MICRONS_PER_VOXEL,
            'component':components,'edge_child_index':children.astype(np.int32),'edge_length_um':lengths,
            'summary':{'nodes':len(ids),'edges':len(children),'components':len(roots),
                       'coordinate_units':'MaleCNS EM voxels; 1 voxel = 8nm',
                       'bounds_um':[xyz_um.min(axis=0).tolist(),xyz_um.max(axis=0).tolist()],
                       'total_cable_length_um':float(lengths.sum()),
                       'maximum_edge_length_um':float(lengths.max(initial=0)),
                       'zero_length_edges':int(np.count_nonzero(lengths==0)),
                       'published_swc_metadata':metadata}}


def indices_for_ids(values,graph_ids):
    values=np.asarray(values,np.int64);ids=np.asarray(graph_ids,np.int64)
    if ids.ndim!=1 or len(ids)==0 or len(np.unique(ids))!=len(ids):
        raise ValueError('Unique nonempty graph IDs required')
    order=np.argsort(ids);ordered=ids[order];position=np.searchsorted(ordered,values)
    safe=np.minimum(position,len(ids)-1)
    retained=(position<len(ids)) & (ordered[safe]==values)
    result=np.full(values.shape,-1,np.int32);result[retained]=order[safe[retained]]
    return result


def aggregate_pairs(pre,post,count,n):
    pre=np.asarray(pre,np.int64);post=np.asarray(post,np.int64);count=np.asarray(count,np.int64)
    if (type(n) is not int or n<=0 or pre.ndim!=1 or pre.shape!=post.shape or pre.shape!=count.shape
            or (pre<0).any() or (post<0).any() or (pre>=n).any() or (post>=n).any() or (count<=0).any()):
        raise ValueError('Valid retained graph pairs and positive contact counts required')
    keys=pre*n+post;unique,inverse=np.unique(keys,return_inverse=True)
    weights=np.zeros(len(unique),np.int64);np.add.at(weights,inverse,count)
    return unique,weights


def compare_contacts_with_graph(pre,post,expected_pre,expected_post,expected_count,n):
    pre=np.asarray(pre,np.int32);post=np.asarray(post,np.int32)
    retained=(pre>=0)&(post>=0)
    actual_keys,actual_counts=aggregate_pairs(pre[retained],post[retained],np.ones(int(retained.sum()),np.int64),n)
    expected_keys,expected_counts=aggregate_pairs(expected_pre,expected_post,expected_count,n)
    keys=np.union1d(actual_keys,expected_keys)
    actual=np.zeros(len(keys),np.int64);expected=np.zeros(len(keys),np.int64)
    actual[np.searchsorted(keys,actual_keys)]=actual_counts
    expected[np.searchsorted(keys,expected_keys)]=expected_counts
    mismatch=actual!=expected
    return {'equal':not bool(mismatch.any()),'retained_contacts':int(retained.sum()),
            'excluded_partner_contacts':int((~retained).sum()),'retained_pairs':len(actual_keys),
            'expected_pairs':len(expected_keys),'expected_contacts':int(expected_counts.sum()),
            'mismatched_pairs':int(mismatch.sum()),
            'first_mismatches':[{'pre_index':int(k//n),'post_index':int(k%n),
                                  'actual':int(a),'expected':int(e)} for k,a,e in
                                zip(keys[mismatch][:10],actual[mismatch][:10],expected[mismatch][:10])]}
