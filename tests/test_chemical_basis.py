"""Physical tuning isolation mathematics, not native learning evidence."""
import numpy as np
from flysnake.research.chemical_basis import chemical_axes


def test_fixed_image_background_cannot_determine_chemical_tuning():
    levels=np.array([[.2,.2],[.8,.2],[.2,.8],[.8,.8],[.5,.5]])
    c=np.tile(levels,(4,1));groups=np.repeat(np.arange(4),5)
    tuning=np.array([[2.,-3.,4.,-5.],[1.,2.,-3.,4.]])
    design=np.c_[c[:,0]-c[:,1],c.mean(1)]
    counts=100+design@tuning
    base=np.array([1.,-1.,1.,-1.]);sides=np.array([0,0,1,1])
    a,m=chemical_axes(counts,c,groups,base,sides)
    offsets=np.array([10,30,-20,50])[groups,None]*np.array([1.,2.,3.,4.])[None,:]
    b,n=chemical_axes(counts+offsets,c,groups,base,sides)
    assert np.allclose(a,b,atol=1e-12)
    assert not m['actions_used'] and not m['reward_used']
    assert np.all(a[sides==1,:2]==0) and np.all(a[sides==0,2:]==0)


def test_static_synaptic_search_is_bounded_and_combines_bilateral_axes():
    from flysnake.research.chemical_basis import synaptic_candidates
    theta=np.array([2.8,-2.8,.1,.2])
    first,proposal=synaptic_candidates(theta,0,1.,3.)
    assert np.array_equal(first[0],theta) and first[1,0]==3 and np.isclose(first[2,0],1.8)
    assert not proposal['physical_input_used'] and not proposal['action_label_used']
    paired,_=synaptic_candidates(np.zeros(4),4)
    assert np.array_equal(paired[1],[1,0,-1,0])


def test_grouped_training_worlds_cover_every_food_at_each_selected_pose():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_chemical_controls import grouped_worlds
    from train_sensory_basis import pose_split,food_sites
    poses,_=pose_split(4)
    batch=grouped_worlds(np.random.default_rng(42),poses,4,'adjacent_one_step',1)
    for heading in range(4):
        rows=[row for row in batch if row['pose']['heading']==heading]
        assert rows and all(row['pose']==rows[0]['pose'] for row in rows)
        assert set(tuple(row['food']) for row in rows)==set(food_sites(rows[0]['pose'],4,'adjacent_one_step'))
