"""Read-only display of real native chemical-response learning trials."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from serve_sensory_snake import snapshot,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--port',type=int,default=8773);args=p.parse_args();directory=args.run.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/','/index.html'):
                body=(ROOT/'flysnake/research/chemical.html').read_bytes();kind='text/html; charset=utf-8'
            elif self.path=='/api/state':
                data=snapshot(directory);report=read(directory/'status.json')
                data.update(chemical_basis=report.get('chemical_basis'),run=str(directory))
                body=json.dumps(data,ensure_ascii=False).encode();kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)
        def log_message(self,*_):pass
    print(f'Native chemical Snake: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
