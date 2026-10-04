import hashlib
import json
import numpy as np

from flysnake.research.teacher_data import teacher_action,build_dataset


def test_teacher_is_training_only_and_respects_immediate_obstacles():
    snake=[(2,2),(2,3),(1,3)]
    assert teacher_action(snake,0,(1,2),5)=='left'
    assert teacher_action(snake,0,(2,1),5)=='forward'
    assert teacher_action(snake,0,(3,2),5)=='right'
    # Requested left neighbour is a body segment: teacher must route around it.
    assert teacher_action([(2,2),(1,2),(1,3)],0,(0,2),5)!='left'


def test_disjoint_pose_and_image_splits_reproduce(tmp_path):
    first=build_dataset(tmp_path/'a',seed=42,train_poses=3,validation_poses=2,test_poses=2,foods=4)
    second=build_dataset(tmp_path/'b',seed=42,train_poses=3,validation_poses=2,test_poses=2,foods=4)
    # ZIP timestamps can differ; compare physical arrays and rows, not archive bytes.
    with np.load(tmp_path/'a/dataset.npz') as a,np.load(tmp_path/'b/dataset.npz') as b:
        for k in a.files:assert np.array_equal(a[k],b[k])
    assert first['rows']==second['rows']
    pose=set();image=set()
    for split in ('train','validation','test'):
        rows=first['rows'][split];keys={r['pose_key'] for r in rows};frames={r['sha256'] for r in rows}
        assert not keys&pose and not frames&image
        pose.update(keys);image.update(frames)
