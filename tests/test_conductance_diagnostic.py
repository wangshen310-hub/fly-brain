"""Numerical and delayed-delivery tests, never evidence of animal learning."""
import ctypes as C
from types import SimpleNamespace
import numpy as np
import pytest
from flysnake.research.conductance_native import build, ConductanceKernel, ConductanceConfig

ABI = ([C.c_int] + [C.c_void_p]*11 + [C.c_int, C.c_float] + [C.c_void_p]*5
       + [C.c_void_p]*4 + [C.c_int] + [C.c_void_p]*4
       + [C.c_float]*3 + [C.c_int] + [C.c_void_p]*5 + [C.c_float]*2)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp('conductance-numerical'))[0]


def trace(library, initial, steps=100, dt=.1, rest=-52., drive=0., ref=0):
    dll=C.CDLL(str(library)); fn=dll.conductance_cell_trace
    fn.argtypes=[C.c_int]+[C.c_float]*6+[C.c_void_p]*3; fn.restype=None
    state=np.array(initial, np.float32); refractory=np.array([ref],np.int16)
    result=np.empty((steps,4),np.float32)
    fn(steps,dt,0.,-70.,rest,drive,200.,state.ctypes.data,refractory.ctypes.data,result.ctypes.data)
    return result, refractory[0]


def test_passive_response_matches_closed_form(library):
    result,_=trace(library,[-64.,0.,0.,0.],drive=3.)
    times=np.arange(1,101)*.1
    expected=-49.+(-64.+49.)*np.exp(-times/20.)
    np.testing.assert_allclose(result[:,0],expected,rtol=0,atol=.0005)


def test_inhibition_has_reversal_instead_of_unbounded_negative_current(library):
    result,_=trace(library,[-52.,0.,600.,0.])
    assert result[:,0].min()>=-70.0001
    assert result[-1,0]<-69.
    below,_=trace(library,[-90.,0.,100.,0.],steps=1)
    assert below[0,0]>-90.  # Same inhibitory channel depolarizes below its reversal.


def test_midpoint_voltage_matches_independent_continuous_rk4(library):
    result,_=trace(library,[-57.,2.,9.,4.],steps=100,drive=6.)
    state=np.array([-57.,2.,9.,4.],float); fine=.0025
    def rhs(x):
        v,e,i,a=x
        return np.array([(-52.-v+6.-a+e*(0.-v)+i*(-70.-v))/20.,-e/5.,-i/5.,-a/200.])
    samples=[]
    for step in range(4000):
        k1=rhs(state); k2=rhs(state+fine*k1/2); k3=rhs(state+fine*k2/2); k4=rhs(state+fine*k3)
        state+=fine*(k1+2*k2+2*k3+k4)/6
        if (step+1)%40==0: samples.append(state.copy())
    np.testing.assert_allclose(result,np.array(samples),rtol=0,atol=.001)


def test_refractory_clamps_voltage_but_receptors_keep_decaying(library):
    result,ref=trace(library,[-49.,3.,5.,2.],steps=2,ref=2)
    np.testing.assert_array_equal(result[:,0],[-52.,-52.])
    np.testing.assert_allclose(result[-1,1:3],np.array([3.,5.])*np.exp(-.2/5.),rtol=1e-6)
    assert ref==0


def fixture_graph(library):
    n=2
    arrays=[np.array([0,1,1],np.int64),np.array([1],np.int32),np.array([5.2],np.float32),
            np.array([-44.,-52.],np.float32),np.zeros(n,np.float32),np.array([0,30],np.int16),
            np.zeros(n,np.float32),np.zeros(n,np.float32),np.zeros((19,n),np.int32),
            np.zeros(19,np.int32),np.array([0],np.int64)]
    ending=[np.zeros(n,np.int32),np.zeros(n,np.int32),np.zeros(n,np.uint8),
            np.zeros(1,np.int32),np.full(n,-1,np.int64),np.zeros(n,np.uint8),
            np.full(n,-1,np.int8),np.zeros(n,np.float64),np.zeros(n,np.int64)]
    plastic=[np.empty(0,np.int64),np.empty(0,np.int32),np.empty(0,np.float32),np.empty(0,np.float32)]
    extra=[np.zeros(n,np.float32),np.zeros(n,np.int64),np.zeros(n,np.uint8),
           np.full(n,-52.,np.float32),np.zeros(n,np.float32)]
    kernel=ConductanceKernel(SimpleNamespace(argtypes=ABI),library,n,ConductanceConfig())
    def advance(steps,learning=0):
        ending[0].fill(0)
        args=[n,*[x.ctypes.data for x in arrays],steps,.1,*[x.ctypes.data for x in ending],
              0,*[x.ctypes.data for x in plastic],.001,1000.,.1,learning,
              *[x.ctypes.data for x in extra],8.,200.]
        assert len(args)==39 and args[31]==learning
        kernel(*args)
        return ending[0].copy()
    return kernel,arrays,ending,extra,advance


def test_delayed_event_survives_refractory_and_call_boundaries(library):
    k,a,e,x,advance=fixture_graph(library)
    np.testing.assert_array_equal(advance(10),[1,0])
    assert k.ge[1]==0.
    advance(10)
    assert k.ge[1]>0. and a[5][1]>0
    assert a[3][1]==-52.
    k2,a2,e2,x2,advance2=fixture_graph(library)
    np.testing.assert_array_equal(advance2(20),[1,0])
    for left,right in zip(a,a2):np.testing.assert_array_equal(left,right)
    np.testing.assert_array_equal(k.ge,k2.ge)
    np.testing.assert_array_equal(k.gi,k2.gi)


def test_training_is_refused_before_native_dispatch(library):
    kernel,arrays,ending,extra,advance=fixture_graph(library)
    with pytest.raises(ValueError,match='no validated plasticity'): advance(1,learning=1)
    assert arrays[10][0]==0
