"""PyBullet in its own process, because it cannot share one with the camera.

Measured: `getCameraImage` with the software renderer takes ~56 ms and holds the CPython
GIL for every one of them — the extension never releases it. Running it in a thread
alongside the capture loop therefore did not just slow the renderer down, it dragged the
camera from 30 fps to 19 and the detector to a crawl, because both spent most of their time
waiting for a lock a C function was sitting on. No amount of thread tuning fixes that; a
separate interpreter does, since each process has its own GIL and the OS puts them on
different cores.

The parent sends small, picklable dicts over a Pipe and gets JPEG frames back. Masks and
raw frames are deliberately NOT sent: the simulation only needs a box's size, pose and
colour, so the message stays a few hundred bytes.
"""
import math
import queue
import threading
import time

import cv2
import numpy as np
import pybullet as p

from vision_measure import describe

# The software renderer costs about 0.63 us per pixel on this machine, measured:
#   640x480 254 ms (3.9 fps)   480x360 138 ms (7.3)   400x300 95 ms (10.5)
#   360x270  78 ms (12.8)      320x240  61 ms (16.4)  256x192 39 ms (25.7)
# Smooth motion reads as "real time" far more than resolution does, and the browser scales
# the panel up anyway, so take the size that makes the target frame rate actually reachable.
VIEW_W, VIEW_H = 320, 240
SIM_FPS = 15
LIFT_CM = 6.0


def to_payload(objects):
    """Strip a measured object down to what the simulation actually needs."""
    out = []
    for o in objects:
        if o.get("track_id") is None:
            continue
        out.append({
            "track_id": o["track_id"],
            "length_cm": float(o["length_cm"]), "width_cm": float(o["width_cm"]),
            "height_cm": float(o["height_cm"]), "angle_deg": float(o.get("angle_deg", 0.0)),
            "center_cm": (float(o["center_cm"][0]), float(o["center_cm"][1])),
            "colour_bgr": tuple(int(c) for c in (o.get("colour_bgr") or (160, 160, 160))),
            "label_text": f"#{o['track_id']} {describe(o)}",
        })
    return out


def _labels(sim, held, view, proj):
    """Project each body's label into the image ourselves.

    PyBullet's debug text is a GUI-only overlay; getCameraImage never draws it, which is
    why the headless panel showed unlabelled blocks.
    """
    mvp = np.array(proj).reshape(4, 4).T @ np.array(view).reshape(4, 4).T
    for tid, rec in sim.objects.items():
        clip = mvp @ np.array([*rec["text_pos"], 1.0])
        if clip[3] <= 0:
            continue
        sx = int((clip[0] / clip[3] * 0.5 + 0.5) * VIEW_W)
        sy = int((1.0 - (clip[1] / clip[3] * 0.5 + 0.5)) * VIEW_H)
        yield rec["label"], sx, sy, (0, 140, 0) if tid == held else (20, 20, 20)


def main(conn):
    from ik_solver import L1, L2
    from pybullet_sim import ArmSim

    sim = ArmSim(gui=False)
    jobs = queue.Queue()
    latest = {"objects": []}
    stop = threading.Event()

    def listen():
        """Reading the pipe must never stall the physics, so it gets its own thread."""
        while not stop.is_set():
            try:
                kind, arg = conn.recv()
            except (EOFError, OSError):
                stop.set()
                return
            if kind == "sync":              # keep only the newest scene, drop the backlog
                latest["objects"] = arg
            else:
                jobs.put((kind, arg))

    threading.Thread(target=listen, daemon=True).start()

    view = p.computeViewMatrix(cameraEyePosition=[0.46, -0.40, 0.40],
                               cameraTargetPosition=[0.0, 0.01, 0.02],
                               cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=46, aspect=VIEW_W / VIEW_H,
                                        nearVal=0.01, farVal=3.0)
    held = None
    next_render = 0.0
    frames = 0
    fps_t = time.perf_counter()
    fps = 0.0

    while not stop.is_set():
        while True:
            try:
                kind, arg = jobs.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "workspace":
                    sim.set_workspace(*arg)
                elif kind == "reset":
                    sim.release()
                    held = None
                    sim.sync_objects([])
                    sim.set_workspace(*arg)
                    latest["objects"] = []
                elif kind == "release":
                    sim.release()
                    held = None
                    conn.send(("held", None))
                elif kind == "pick":
                    x, y, z, tid = arg
                    conn.send(("busy", True))
                    sim.go_to(x, y, z + 5.0, settle_steps=130)   # approach from above
                    sim.go_to(x, y, z, settle_steps=130)
                    ok = sim.grasp(tid)
                    if ok:
                        held = tid
                        sim.go_to(x, y, z + LIFT_CM, settle_steps=180)
                    conn.send(("picked", (ok, tid if ok else None)))
                    conn.send(("busy", False))
                elif kind == "quit":
                    stop.set()
            except Exception as exc:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
                conn.send(("busy", False))

        sim.sync_objects(latest["objects"], skip=held)
        p.stepSimulation()

        now = time.perf_counter()
        if now >= next_render:
            next_render = now + 1.0 / SIM_FPS
            img = p.getCameraImage(VIEW_W, VIEW_H, view, proj,
                                   renderer=p.ER_TINY_RENDERER)
            rgba = np.reshape(img[2], (VIEW_H, VIEW_W, 4)).astype(np.uint8)
            out = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            for text, sx, sy, col in _labels(sim, held, view, proj):
                for thick, c in ((3, (255, 255, 255)), (1, col)):   # halo, then the text
                    cv2.putText(out, text, (sx - 42, sy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.36, c, thick, cv2.LINE_AA)
            frames += 1
            dt = now - fps_t
            if dt >= 0.5:
                fps, frames, fps_t = frames / dt, 0, now
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                conn.send(("frame", (buf.tobytes(), round(fps, 1))))
        # Windows' default timer granularity is ~15 ms, so a short sleep here would
        # overshoot the render deadline. Sleep only for the slack we actually have.
        slack = next_render - time.perf_counter()
        time.sleep(min(0.004, slack) if slack > 0 else 0)
    conn.close()
