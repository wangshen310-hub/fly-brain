"""Data validation fixtures; no synthetic fixture is claimed as neural evidence."""
import numpy as np
import pytest
from flysnake.research.apl_spatial_data import parse_swc,indices_for_ids,compare_contacts_with_graph


def test_swc_preserves_units_forest_and_parent_identity():
    text='#${"downresLevel":"coarse"}\n40 0 125 0 0 32 10\n10 0 0 0 0 64 -1\n80 0 0 250 0 0 -1\n'
    tree=parse_swc(text)
    np.testing.assert_array_equal(tree['parent_index'],[1,-1,-1])
    np.testing.assert_array_equal(tree['component'],[0,0,1])
    np.testing.assert_allclose(tree['xyz_um'],[[1,0,0],[0,0,0],[0,2,0]])
    np.testing.assert_allclose(tree['edge_length_um'],[1])
    assert tree['summary']['components']==2
    assert tree['summary']['published_swc_metadata'][0]['downresLevel']=='coarse'


@pytest.mark.parametrize('text',[
    '1 0 0 0 0 1 2\n2 0 1 0 0 1 1\n',
    '1 0 0 0 0 1 99\n',
    '1 0 0 0 0 1 -1\n1 0 1 0 0 1 -1\n',
    '1 0 nan 0 0 1 -1\n',
    '1.1 0 0 0 0 1 -1\n',
])
def test_invalid_swc_is_rejected(text):
    with pytest.raises(ValueError):parse_swc(text)


def test_source_id_mapping_preserves_native_order_and_excludes_unknown():
    np.testing.assert_array_equal(indices_for_ids([10977,999,10540,900],[900,10540,10977]),[2,-1,1,0])


def test_contacts_match_counts_but_excluded_objects_are_reported():
    result=compare_contacts_with_graph([1,1,2,1],[2,2,1,-1],[1,2],[2,1],[2,1],3)
    assert result['equal'] and result['retained_contacts']==3
    assert result['excluded_partner_contacts']==1 and result['retained_pairs']==2
    changed=compare_contacts_with_graph([1,1,2],[2,2,1],[1,2],[2,1],[1,1],3)
    assert not changed['equal'] and changed['mismatched_pairs']==1
