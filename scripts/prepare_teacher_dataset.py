import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flysnake.research.teacher_data import build_dataset

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--size',type=int,default=6);p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--train-poses',type=int,default=24);p.add_argument('--validation-poses',type=int,default=8)
    p.add_argument('--test-poses',type=int,default=8);p.add_argument('--foods',type=int,default=9)
    a=p.parse_args();m=build_dataset(a.output,size=a.size,seed=a.seed,train_poses=a.train_poses,
        validation_poses=a.validation_poses,test_poses=a.test_poses,foods=a.foods)
    print(json.dumps({k:m[k] for k in ['version','size','seed','pose_counts','class_counts','dataset_sha256']}))
