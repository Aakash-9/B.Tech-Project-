"""Real process/HTTP smoke check. Requires webcam access and localhost sockets.

The startup probes terminate only their own process trees. Graceful simulation
shutdown is separately exercised by test_worker_feedback and test_realtime.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil


def request(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=5)


def main():
    out = Path('audit_results/final')
    out.mkdir(parents=True, exist_ok=True)
    for mode in ('sim', 'camera'):
        url = 'http://127.0.0.1:5011'
        cmd = [sys.executable, '-u', 'dashboard.py', '--port', '5011']
        if mode == 'camera': cmd.append('--camera')
        with (out / f'startup-{mode}.log').open('w', encoding='utf8') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                state = None
                for _ in range(60):
                    if proc.poll() is not None: break
                    try:
                        with request(url+'/api/state') as r: state = json.load(r)
                        break
                    except OSError: time.sleep(1)
                assert state is not None, f'{mode}: dashboard did not start; see {log.name}'
                assert state['is_live'] == (mode == 'camera')
                with request(url) as r:
                    assert b'feedback-note' in r.read()
                for panel in ('camera', 'sim'):
                    with request(url+'/video/'+panel) as r:
                        assert b'image/jpeg' in r.read(2000)
                time.sleep(8)
                with request(url+'/api/state') as r: state = json.load(r)
                print(f'[PASS] {mode}: page, source, camera JPEG, simulation JPEG', flush=True)
                print(json.dumps({'fps': state['fps'], 'calibration': state['calibration']}),flush=True)
                if mode == 'sim':
                    t = time.perf_counter()
                    with request(url+'/api/command', {'text': 'pick the red cube'}) as r:
                        assert json.load(r)['ok']
                    response_time = time.perf_counter()-t
                    assert response_time < .5, f'HTTP command stalled {response_time:.2f}s'
                    for _ in range(160):
                        with request(url+'/api/state') as r: state=json.load(r)
                        if state['holding'] and not state['busy']: break
                        time.sleep(.15)
                    assert state['holding'] and not state['busy'], state['log']
                    print(f'[PASS] HTTP pick completes; request returned in {response_time*1000:.1f} ms',flush=True)
            finally:
                children = psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
                if proc.poll() is None:
                    proc.terminate(); proc.wait(timeout=10)
                for child in children:
                    if child.is_running(): child.terminate()
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive: child.kill()
                psutil.wait_procs(alive, timeout=5)
                assert not any(c.is_running() for c in children), 'orphaned probe process'
                print(f'[PASS] {mode}: owned startup process tree cleaned up',flush=True)
    print('PASSED 5 / 5 dashboard startup checks')


if __name__ == '__main__': main()
