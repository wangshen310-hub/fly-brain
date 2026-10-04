"""Hidden service-created worker; it does not inherit Codex's process job.

The native script and arguments are visible in the OS command line. Standard
streams are opened by the worker itself, never held by the calling shell.
"""
from pathlib import Path
import argparse
import contextlib
import json
import os
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--script',required=True)
    p.add_argument('--launch-data',type=Path,required=True);args=p.parse_args()
    script=(ROOT/args.script).resolve();data=args.launch_data.resolve()
    if not script.is_relative_to(ROOT/'scripts') or not data.is_relative_to(ROOT):
        raise ValueError('Only this workspace native worker may be launched')
    spec=json.loads(data.read_text(encoding='utf-8'))
    if spec['script']!=args.script or Path(spec['cwd']).resolve()!=ROOT:
        raise ValueError('Worker launch identity differs')
    for name in ('stdout','stderr'):
        if not Path(spec[name]).resolve().is_relative_to(ROOT):raise ValueError('Workspace worker logs required')
    if not isinstance(spec['arguments'],list) or not all(isinstance(a,str) for a in spec['arguments']):
        raise ValueError('Explicit native CLI argument strings required')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    sys.argv=[str(script),*spec['arguments']]
    with Path(spec['stdout']).open('a',encoding='utf-8',buffering=1) as out,Path(spec['stderr']).open('a',encoding='utf-8',buffering=1) as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:runpy.run_path(str(script),run_name='__main__')
            except BaseException:
                import traceback
                traceback.print_exc();raise


if __name__=='__main__':main()
