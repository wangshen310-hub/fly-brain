"""A read-only live view of actual native sensory/reward experiments."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import time
ROOT=Path(__file__).resolve().parents[1]


def writer_active(directory):
    path=directory/'writer.lock'
    if not path.is_file():return False
    try:
        with path.open('r+b') as stream:
            if os.name=='nt':
                import msvcrt
                try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                except OSError:return True
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except OSError:return True
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    except OSError:return None
    return False


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def snapshot(directory):
    current=read(directory/'current.json');status=read(directory/'status.json')
    result={'current':current,'status':status.get('status','waiting'),
        'writer_active':writer_active(directory),'cycles_completed':status.get('cycles_completed',0),
        'checkpoints':status.get('checkpoints',[]),'settings':status.get('settings'),
        'brain':{k:v for k,v in status.get('brain',{}).items() if k not in ('baseline_inputs',)},
        'evaluations':{},'cycles':[],'learning_proven':False,'server_time':time.time()}
    if status.get('before'):
        result['evaluations']['before']={k:v for k,v in status['before'].items() if k!='episodes'}
    for arm,ev in status.get('after',{}).items():
        result['evaluations'][arm]={k:v for k,v in ev.items() if k!='episodes'}
    for cycle in status.get('cycles',[]):
        result['cycles'].append({'cycle':cycle['cycle'],
            'arms':{arm:values['update'] for arm,values in cycle['arms'].items()}})
    if status.get('error'):result['error']=status['error']
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--port',type=int,default=8773)
    args=p.parse_args();directory=args.run.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/','/index.html'):
                body=(ROOT/'flysnake/research/sensory.html').read_bytes();kind='text/html; charset=utf-8'
            elif self.path=='/api/state':
                body=json.dumps(snapshot(directory),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)
        def log_message(self,*_):pass
    print(f'Native sensory Snake: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
