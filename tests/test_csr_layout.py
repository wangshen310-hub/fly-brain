import numpy as np
from flysnake.research.csr_layout import canonical_columns


def test_unsorted_sparse_delivery_keeps_physical_weighted_sum():
    crow=np.array([0,3,4,5]);col=np.array([2,0,1,2,1]);value=np.array([.3,.7,1.2,2.,-.2])
    sorted_col,sorted_value=canonical_columns(crow,col,value,3)
    input_state=np.array([2.,-4.,7.])
    def deliver(c,v):return np.array([np.sum(v[crow[i]:crow[i+1]]*input_state[c[crow[i]:crow[i+1]]]) for i in range(3)])
    assert np.allclose(deliver(col,value),deliver(sorted_col,sorted_value))
    assert sorted_col.tolist()==[0,1,2,2,1]
