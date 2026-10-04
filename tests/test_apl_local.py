"""Independent geometry and cable checks; fixtures are not learning evidence."""
import ctypes as C
import numpy as np
import pytest
from flysnake.research.apl_local import build, attach_segments, forest_order, cable_geometry, LocalAPLConfig


@pytest.fixture(scope='module')
def library(tmp_path_factory):return build(tmp_path_factory.mktemp('apl-local'))[0]


def cable(library,xyz,parent,*,steps=100,dt=.1,lam=2.,u=None,ge=None,gi=None,drive=None):
    xyz=np.array(xyz,float);parent=np.array(parent,np.int32);n=len(parent)
    volume,axial,diagonal=cable_geometry(xyz,parent,lam)
    states=[np.zeros(n) if x is None else np.array(x,float) for x in (u,ge,gi,drive)]
    output=np.empty((steps,n));fn=C.CDLL(str(library)).apl_cable_trace
    fn.argtypes=[C.c_int,C.c_int,C.c_double]+[C.c_void_p]*4+[C.c_double]*3+[C.c_void_p]*5
    fn.restype=None
    fn(n,steps,dt,parent.ctypes.data,axial.ctypes.data,volume.ctypes.data,diagonal.ctypes.data,
       20.,52.,-18.,*[x.ctypes.data for x in states],output.ctypes.data)
    return output,states,(volume,axial,diagonal)


def test_segment_attachment_matches_independent_exhaustive_projection(library):
    rng=np.random.default_rng(42);xyz=rng.normal(size=(61,3));child=np.arange(1,61,dtype=np.int32)
    parent=np.arange(60,dtype=np.int32);q=rng.normal(size=(47,3))
    e,a,d=attach_segments(library,xyz,child,parent,q)
    direction=xyz[child]-xyz[parent];difference=q[:,None,:]-xyz[parent][None,:,:]
    expected_a=np.clip(np.sum(difference*direction[None,:,:],axis=2)/np.sum(direction**2,axis=1),0,1)
    distance=np.linalg.norm(difference-expected_a[:,:,None]*direction[None,:,:],axis=2)
    expected_e=distance.argmin(axis=1)
    # Adjacent segments share an endpoint; equal minima can name either edge.
    np.testing.assert_allclose(d,distance.min(axis=1),atol=1e-14,rtol=0)
    np.testing.assert_allclose(a,expected_a[np.arange(len(q)),e],atol=1e-14,rtol=0)
    np.testing.assert_allclose(d,distance[np.arange(len(q)),e],atol=1e-14,rtol=0)


def test_attachment_uses_segment_interior_instead_of_nearest_node(library):
    edge,alpha,d=attach_segments(library,[[0,0,0],[10,0,0]],[1],[0],[[5,1,0]])
    assert edge[0]==0 and alpha[0]==.5 and d[0]==1.


def test_parent_order_and_disconnected_forest_are_retained():
    order,inverse=forest_order([2,-1,1,-1,3]);assert len(order)==5
    assert (inverse[np.array([2,1,3])]<inverse[np.array([0,2,4])]).all()
    with pytest.raises(ValueError,match='cycle'):forest_order([1,0])


def test_cable_equilibrium_matches_independent_dense_linear_system(library):
    xyz=[[0,0,0],[1,0,0],[2,0,0],[1,1,0]];parent=[-1,0,1,1];drive=np.array([8,0,0,0.])
    result,states,(volume,axial,diagonal)=cable(library,xyz,parent,steps=5000,dt=.5,drive=drive)
    matrix=np.diag(volume+diagonal)
    for i,p in enumerate(parent):
        if p>=0:matrix[i,p]=matrix[p,i]=-axial[i]
    expected=np.linalg.solve(matrix,volume*drive)
    np.testing.assert_allclose(result[-1],expected,atol=1e-10,rtol=0)


def test_constant_passive_voltage_follows_backward_euler_closed_form(library):
    result,_,_=cable(library,[[0,0,0],[1,0,0]],[-1,0],u=[7,7],steps=100,dt=.5)
    expected=7*(1+.5/20.)**(-np.arange(1,101))
    np.testing.assert_allclose(result[:,0],expected,atol=1e-12,rtol=0)


def test_local_drive_decays_with_cable_distance_and_does_not_cross_components(library):
    xyz=[[i,0,0] for i in range(11)]+[[0,1,0],[1,1,0]];parent=[-1]+list(range(10))+[-1,11]
    drive=np.zeros(13);drive[0]=10
    result,_,_=cable(library,xyz,parent,drive=drive,steps=4000,dt=.5)
    assert np.all(np.diff(result[-1,:11])<0)
    np.testing.assert_array_equal(result[:,-2:],0.)


def test_strong_inhibitory_conductance_stays_between_reversals(library):
    result,states,_=cable(library,[[0,0,0],[1,0,0]],[-1,0],ge=[1e5,0],gi=[0,1e6],steps=100)
    assert result.min()>=-18.00001 and result.max()<=52.00001
    assert np.isfinite(result).all()
    assert states[1].sum()<1e5 and states[2].sum()<1e6


def test_halving_dt_converges_to_independent_continuous_matrix_solution(library):
    xyz=[[0,0,0],[1,0,0],[2,0,0]];parent=[-1,0,1];u=np.array([9.,0,0])
    errors=[]
    for dt in (.5,.25,.125):
        result,_,(volume,axial,diagonal)=cable(library,xyz,parent,u=u,steps=round(10/dt),dt=dt)
        matrix=np.diag(volume+diagonal)
        for i,p in enumerate(parent):
            if p>=0:matrix[i,p]=matrix[p,i]=-axial[i]
        capacitance=20*volume;operator=matrix/np.sqrt(capacitance[:,None]*capacitance[None,:])
        eigen,vec=np.linalg.eigh(operator)
        exact=(vec@(np.exp(-10*eigen)*(vec.T@(np.sqrt(capacitance)*u))))/np.sqrt(capacitance)
        errors.append(np.linalg.norm(result[-1]-exact))
    assert errors[2]<errors[1]<errors[0] and errors[1]<errors[0]*.55


def test_invalid_geometry_and_config_are_rejected():
    with pytest.raises(ValueError):cable_geometry([[0,0,0],[0,0,0]],[-1,0],25.)
    for config in (LocalAPLConfig(cable_dt_ms=.35),LocalAPLConfig(release_gain=-1),LocalAPLConfig(space_constant_um=float('nan'))):
        with pytest.raises(ValueError):config.validate()
