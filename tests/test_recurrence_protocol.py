import numpy as np
import pytest
from flysnake.research.recurrence_protocol import anatomical_groups,vector_comparison,cue_windows


def test_groups_use_official_classes_and_record_ambiguous_fast_signs():
    groups=anatomical_groups(['ALLN','ALLN','ALPN','x','ALLN'],
        ['lLN2T_a','v2LN30','DM1_lPN','ORN_DM1','mixed'],
        ['acetylcholine','unclear','acetylcholine','acetylcholine','acetylcholine,gaba'],[4])
    np.testing.assert_array_equal(groups['ALLN'],[0,1,4])
    np.testing.assert_array_equal(groups['ALLN_uncertain'],[1,4])
    np.testing.assert_array_equal(groups['ALPN'],[2]);np.testing.assert_array_equal(groups['ORN'],[3])


def test_empty_response_is_not_misreported_as_good_discrimination():
    assert vector_comparison(np.zeros(5),np.zeros(5))['cosine'] is None
    assert vector_comparison([1,0],[0,1])['cosine']==0
    with pytest.raises(ValueError):vector_comparison([1,-1],[0,1])


def test_early_and_late_windows_remain_separate():
    a=np.zeros((30,3),np.int32);b=a.copy();a[:5,0]=1;b[:5,1]=1;a[5:,2]=10;b[5:,2]=10
    rows=cue_windows(a,b,[0,1,2])
    assert rows[1]['cosine']==0 and rows[-1]['cosine']>.999
    assert rows[1]['active_jaccard']==0 and rows[-1]['active_jaccard']==1/3
