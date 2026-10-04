"""Local experiment viewer. The worker runs without depending on browser tabs."""
import argparse
import json
import multiprocessing as mp
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse
from .experiment import Experiment, DEFAULT_STORAGE


class Controller:
    def __init__(self,storage):
        self.storage=Path(storage).resolve();self.guard=threading.Lock()
        self.thread=None;self.stop=threading.Event();self.error=None
        self.closing=False

    def state(self):
        p=self.storage/'status.json'
        state=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'phase':'not_started','message':'等待启动验证实验','gates':{}}
        # Complete trial records remain on disk. UI receives compact progress.
        rows=state.pop('rows',[])
        state['completed_trials']=len(rows)
        state['stage_completed_trials']=sum(row.get('stage')==state.get('stage') for row in rows)
        state['recent_trials']=rows[-12:]
        state.pop('assay_samples',None)
        state['active']=bool(self.thread and self.thread.is_alive())
        if self.error:state['server_error']=self.error
        return state

    def start(self):
        with self.guard:
            if self.closing: raise ValueError('Server is closing')
            if self.thread and self.thread.is_alive():return {'already_running':True}
            self.stop.clear();self.error=None
            def run():
                try:Experiment(self.storage,stop=self.stop.is_set).run()
                except Exception as error:self.error=str(error)
            self.thread=threading.Thread(target=run,name='bounded-experiment',daemon=False)
            self.thread.start()
        return {'started':True}

    def close(self):
        with self.guard:self.closing=True;self.stop.set()
        if self.thread:self.thread.join()


def handler(controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,value,status=200,content_type='application/json; charset=utf-8'):
            body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/state':return self.respond(controller.state())
            if path=='/':return self.respond(Path(__file__).with_name('index.html').read_bytes(),content_type='text/html; charset=utf-8')
            return self.respond({'error':'Not found'},404)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):
                return self.respond({'error':'Origin denied'},403)
            if urlparse(self.path).path!='/api/command':return self.respond({'error':'Not found'},404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024 or 'application/json' not in self.headers.get('Content-Type',''):raise ValueError('Invalid command')
                action=json.loads(self.rfile.read(length))['action']
                if action=='run':return self.respond(controller.start())
                if action=='pause':
                    controller.stop.set();return self.respond({'requested':'save_then_pause_at_trial_boundary'})
                if action=='shutdown':
                    self.respond({'requested':'save_then_exit'})
                    def close():controller.close();self.server.shutdown()
                    threading.Thread(target=close,daemon=True).start();return
                raise ValueError('Unknown action')
            except (ValueError,KeyError,TypeError) as error:return self.respond({'error':str(error)},400)
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8772)
    p.add_argument('--storage',type=Path,default=DEFAULT_STORAGE);p.add_argument('--run',action='store_true')
    args=p.parse_args();controller=Controller(args.storage)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(controller))
    if args.run:controller.start()
    print(f'Fly learning laboratory: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:controller.close();server.server_close()


if __name__=='__main__':mp.freeze_support();main()
