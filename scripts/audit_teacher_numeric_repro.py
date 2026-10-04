"""Isolate GPU sparse reduction reproducibility before any model training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    b=SpikingConnectome.load_graph(args.graph)
    frame=np.load(Path(args.dataset)/'dataset.npz')['train_rgb'][0]
    light=b.encode(frame[None]);rows=[]
    with torch.no_grad():
        for batch in [1,2,3,12]:
            encoded=light.expand(light.shape[0],batch).contiguous()
            c,_,s=b.integrate_encoded(encoded);z,_,t=b.integrate_encoded(encoded)
            diff={k:float(torch.max(torch.abs(v-t[k]))) for k,v in s.items() if isinstance(v,torch.Tensor)}
            rows.append({'batch':batch,'counts_exact':torch.equal(c,z),'state_exact':all(x==0 for x in diff.values()),'max_abs':diff})
            print(json.dumps(rows[-1]),flush=True)
    result={'rows':rows,'torch':str(torch.__version__),'deterministic_algorithms':True,'trained':False}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
