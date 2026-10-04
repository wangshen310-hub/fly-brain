"""Offline recurrence diagnostics, never sensory transduction or action choice."""
import numpy as np


def anatomical_groups(classes,types,transmitters,kc_indices):
    classes=np.asarray(classes,str);types=np.asarray(types,str);nt=np.asarray(transmitters,str)
    if classes.ndim!=1 or classes.shape!=types.shape or classes.shape!=nt.shape:
        raise ValueError('Aligned original anatomical annotations required')
    cholinergic=np.array(['acetylcholine' in str(v).lower().split(',') for v in nt])
    uncertain=[]
    for v in nt:
        tokens=set(str(v).lower().split(','));signs=set()
        if 'acetylcholine' in tokens:signs.add(1)
        if tokens&{'gaba','glutamate','histamine'}:signs.add(-1)
        uncertain.append(len(signs)!=1)
    ln=classes=='ALLN';pn=classes=='ALPN';orn=np.char.startswith(types,'ORN_')
    groups={'ORN':np.flatnonzero(orn),'ALPN':np.flatnonzero(pn),'ALLN':np.flatnonzero(ln),
        'ALLN_ACh':np.flatnonzero(ln&cholinergic),'ALLN_uncertain':np.flatnonzero(ln&np.array(uncertain)),
        'KC':np.asarray(kc_indices,np.int32)}
    return {name:np.asarray(ix,np.int32) for name,ix in groups.items()}


def vector_comparison(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=1 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or min(x.min(),y.min())<0:
        raise ValueError('Aligned nonnegative actual spike count vectors required')
    norm=np.linalg.norm(x)*np.linalg.norm(y);union=np.count_nonzero((x>0)|(y>0))
    return {'cosine':float(x@y/norm) if norm else None,
        'active_jaccard':float(np.count_nonzero((x>0)&(y>0))/union) if union else None,
        'active_cells':[int(np.count_nonzero(x)),int(np.count_nonzero(y))],
        'spikes':[int(x.sum()),int(y.sum())]}


def cue_windows(counts_a,counts_b,indices):
    a=np.asarray(counts_a);b=np.asarray(counts_b);ix=np.asarray(indices,np.int32)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=30:raise ValueError('Thirty actual 10ms cue bins required')
    windows=[('0-30ms',0,3),('0-50ms',0,5),('0-100ms',0,10),('100-300ms',10,30),('0-300ms',0,30)]
    return [{'window':label,**vector_comparison(a[start:end][:,ix].sum(axis=0),b[start:end][:,ix].sum(axis=0))}
            for label,start,end in windows]
