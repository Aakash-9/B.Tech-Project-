"""Check every part of the dashboard and say exactly which one is broken.

    venv/Scripts/python.exe doctor.py            check the simulated mode
    venv/Scripts/python.exe doctor.py --camera   check the live camera mode

Prints a PASS/FAIL line per component so a failure points at one thing instead of
"it doesn't work".
"""
import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return ok


def port_free(port):
    with socket.socket() as s:
        s.settimeout(0.7)
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    print("\n=== 1. Python packages ===")
    for mod in ("cv2", "numpy", "flask", "pybullet", "ultralytics"):
        try:
            importlib.import_module(mod)
            check(f"{mod} importable", True)
        except Exception as e:
            check(f"{mod} importable", False, str(e)[:70])

    print("\n=== 2. Project files ===")
    for f in ("dashboard.py", "dashboard_engine.py", "dashboard_sources.py",
              "templates/dashboard.html", "robot_arm.urdf"):
        check(f"{f} present", os.path.exists(os.path.join(HERE, f)))

    print("\n=== 3. Port ===")
    free = port_free(args.port)
    check(f"port {args.port} is free", free,
          "" if free else f"something is already using it — try --port {args.port + 1}")

    if args.camera:
        print("\n=== 4. Camera ===")
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        ok_read, frame = (cap.read() if opened else (False, None))
        cap.release()
        check("webcam opens", opened, "" if opened else "another program may be using it")
        check("webcam returns frames", bool(ok_read),
              f"{frame.shape}, brightness {frame.mean():.0f}" if ok_read else "")

    print("\n=== 5. Starting the real dashboard ===")
    port = args.port if free else args.port + 1
    cmd = [PY, "dashboard.py", "--port", str(port)] + (["--camera"] if args.camera else [])
    print(f"  running: {' '.join(cmd[1:])}")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    state, deadline = None, time.time() + (120 if args.camera else 60)
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3) as r:
                state = json.loads(r.read())
            break
        except Exception:
            time.sleep(1.0)

    if proc.poll() is not None:
        out = proc.stdout.read()
        check("dashboard starts", False, "it exited — error below")
        print("\n  ---- server output ----")
        for line in out.strip().splitlines()[-15:]:
            print("   ", line)
    else:
        check("dashboard starts", state is not None,
              "" if state else "started but never answered /api/state")
        if state:
            check("serves the web page",
                  b"Vision Dashboard" in urllib.request.urlopen(
                      f"http://127.0.0.1:{port}/", timeout=5).read())
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/video/camera",
                                        timeout=8) as r:
                check("camera panel streams", b"image/jpeg" in r.read(8000))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/video/sim",
                                        timeout=8) as r:
                check("simulation panel streams", b"image/jpeg" in r.read(8000))
            n = len(state["objects"])
            check("objects detected", n > 0,
                  f"{n} found" if n else
                  "nothing in view matching the prompts (see PROMPTS in dashboard_sources.py)")
            cal = state["calibration"]
            check("calibration usable", bool(cal["ok"]), cal["text"])
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()

    bad = results.count(False)
    print(f"\n{'=' * 60}")
    if bad == 0:
        print(f"EVERYTHING WORKS — open http://127.0.0.1:{port} in your browser")
    else:
        print(f"{bad} CHECK(S) FAILED — the FAIL lines above say which part")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
