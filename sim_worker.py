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
# Cost is linear in pixels on the software renderer. Measured here WITH shadows on:
#   320x240 33 ms (30 fps ceiling)   400x300 52 ms (19)   448x336 69 ms (14)   512x384 88 ms (11)
# 400x300 is the largest size that still clears the 15 fps target with headroom, and it is
# visibly sharper than 320x240 in a browser panel that upscales it anyway.
VIEW_W, VIEW_H = 400, 300
SIM_FPS = 15
# Shadows cost 1.3 ms of a 24 ms frame here — measured — and they are what make a flat
# render read as 3D at a glance: without them an object and its label float with no sense
# of what is resting on what.
LIGHT_DIR = [0.6, -0.9, 1.2]


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


def _mvp(view, proj):
    return np.array(proj).reshape(4, 4).T @ np.array(view).reshape(4, 4).T


def _project(mvp, point_m):
    """World position in METRES -> pixel, or None when it is behind the camera."""
    clip = mvp @ np.array([point_m[0], point_m[1], point_m[2], 1.0])
    if clip[3] <= 1e-6:
        return None
    return (int((clip[0] / clip[3] * 0.5 + 0.5) * VIEW_W),
            int((1.0 - (clip[1] / clip[3] * 0.5 + 0.5)) * VIEW_H))


def _ground_ring(mvp, radius_cm, segments=72):
    """A circle of this radius on the table, as a projected pixel polygon."""
    r = radius_cm / 100.0
    pts = [_project(mvp, (r * math.cos(2 * math.pi * i / segments),
                          r * math.sin(2 * math.pi * i / segments), 0.0))
           for i in range(segments)]
    pts = [q for q in pts if q is not None]
    return np.asarray(pts, np.int32) if len(pts) > 8 else None


FONT = cv2.FONT_HERSHEY_SIMPLEX
CHIP_SCALE = 0.34


def _chip(img, text, x, y, fg, bg):
    """Text on a filled pill. Plain text over a busy render is hard to read; this isn't."""
    (tw, th), _ = cv2.getTextSize(text, FONT, CHIP_SCALE, 1)
    x = max(2, min(x, VIEW_W - tw - 6))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 3), bg, -1)
    cv2.putText(img, text, (x, y), FONT, CHIP_SCALE, fg, 1, cv2.LINE_AA)
    return (x - 3, y - th - 4, x + tw + 3, y + 3)


def _lay_out_labels(entries):
    """Stop labels from landing on top of each other.

    With four objects clustered near the middle of the table, every chip wanted the same
    patch of screen and they overprinted into an unreadable stack — the single worst thing
    for a view whose whole job is being understandable at a glance. Walk them top-down and
    push any chip that would collide with one already placed far enough down to clear it.
    """
    placed = []
    out = []
    for anchor, text, carried in sorted(entries, key=lambda e: e[0][1]):
        (tw, th), _ = cv2.getTextSize(text, FONT, CHIP_SCALE, 1)
        x, y = anchor[0] - tw // 2, anchor[1] - 8
        for _ in range(14):                       # bounded: never loop forever on a pile-up
            box = (x - 3, y - th - 4, x + tw + 3, y + 3)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                break
            y += th + 7
        placed.append((x - 3, y - th - 4, x + tw + 3, y + 3))
        out.append((anchor, text, carried, x, y))
    return out


def _draw_overlay(img, sim, held, status, mvp, reach_min_cm, reach_max_cm):
    """Everything the raw render cannot say for itself.

    A physics render shows WHAT is there but not what it MEANS. The reach rings answer the
    first question anyone asks watching this ("can it actually get to that one?") without
    needing the answer explained, and the status line says what the arm is doing right now
    rather than leaving the viewer to infer it from a frozen frame.
    """
    for radius, colour in ((reach_max_cm, (95, 180, 110)), (reach_min_cm, (110, 135, 180))):
        ring = _ground_ring(mvp, radius)
        if ring is not None:
            cv2.polylines(img, [ring], True, colour, 1, cv2.LINE_AA)

    base = _project(mvp, (0.0, 0.0, 0.0))
    if base is not None:
        cv2.drawMarker(img, base, (200, 200, 200), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)

    entries = []
    for tid, rec in sim.objects.items():
        pos = rec['text_pos']
        if tid == held:
            xyz = p.getBasePositionAndOrientation(rec['body'])[0]
            pos = (xyz[0], xyz[1], xyz[2] + rec['half'][2] + .03)
        anchor = _project(mvp, pos)
        if anchor is not None:
            entries.append((anchor, rec["label"], tid == held))

    for anchor, text, carried, x, y in _lay_out_labels(entries):
        cv2.circle(img, anchor, 2, (245, 245, 245), -1, cv2.LINE_AA)
        if carried:
            cv2.circle(img, anchor, 12, (70, 220, 90), 1, cv2.LINE_AA)
        # A leader line, because once a chip has been pushed aside to avoid a collision it
        # is no longer obvious which block it belongs to.
        cv2.line(img, anchor, (x + 6, y + 2), (120, 120, 125), 1, cv2.LINE_AA)
        _chip(img, text, x, y,
              (255, 255, 255) if carried else (245, 245, 245),
              (40, 120, 55) if carried else (45, 45, 50))

    cv2.rectangle(img, (0, 0), (VIEW_W, 20), (26, 30, 36), -1)
    cv2.putText(img, status, (7, 14), FONT, 0.40, (235, 240, 245), 1, cv2.LINE_AA)
    # The legend needs its own dark strip: as bare text it sat on whatever the render put
    # underneath and was unreadable against the light tabletop.
    cv2.rectangle(img, (0, VIEW_H - 16), (VIEW_W, VIEW_H), (26, 30, 36), -1)
    cv2.circle(img, (10, VIEW_H - 8), 4, (95, 180, 110), 1, cv2.LINE_AA)
    # :.1f on the inner radius — it is 0.5 cm here, and %.0f rendered it as a flat "0",
    # which reads as "no inner limit" when there is one.
    cv2.putText(img, f"arm reach {reach_min_cm:.1f}-{reach_max_cm:.1f} cm", (20, VIEW_H - 5),
                FONT, 0.33, (205, 225, 210), 1, cv2.LINE_AA)
    return img


def main(conn):
    from ik_solver import L1, L2
    from pybullet_sim import ArmSim
    from visual_servo import CONTROL_DT, ServoPick

    sim = ArmSim(gui=False)
    jobs = queue.Queue()
    latest = {"scene": ([], 0.)}
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
                latest["scene"] = arg
            else:
                jobs.put((kind, arg))

    threading.Thread(target=listen, daemon=True).start()

    view = p.computeViewMatrix(cameraEyePosition=[0.46, -0.40, 0.40],
                               cameraTargetPosition=[0.0, 0.01, 0.02],
                               cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=46, aspect=VIEW_W / VIEW_H,
                                        nearVal=0.01, farVal=3.0)
    sim.go_to(12, 0, 8)  # one-time startup pose, before feedback is enabled
    picker = ServoPick(sim)
    last_scene = None
    next_render = 0.0
    frames = 0
    fps_t = time.perf_counter()
    fps = 0.0

    try:
      while not stop.is_set():
        tick_start = time.perf_counter()
        snapshot = latest['scene']
        objects, observed_at = snapshot
        if snapshot is not last_scene:
            sim.sync_objects(objects, skip=picker.held)
            last_scene = snapshot
        while True:
            try:
                kind, arg = jobs.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "workspace":
                    sim.set_workspace(*arg)
                elif kind == "reset":
                    picker.cancel('Looking at the table')
                    sim.sync_objects([])
                    sim.set_workspace(*arg)
                    latest["scene"] = ([], 0.)
                    objects, observed_at = [], 0.
                elif kind == "release":
                    picker.cancel()
                elif kind == "pick":
                    x, y, z, tid = arg
                    # x/y are the initial selection only. Each subsequent tick
                    # observes the selected body's RGB marker, so motion matters.
                    picker.start(tid, z, time.perf_counter())
                elif kind == "quit":
                    stop.set()
            except Exception as exc:
                picker.cancel('Stopped after a simulation error')
                conn.send(("error", f"{type(exc).__name__}: {exc}"))

        if stop.is_set(): break
        was_busy = picker.busy
        try:
            picker.tick(time.perf_counter(), {o['track_id'] for o in objects}, observed_at)
        except Exception as exc:
            picker.fail(f'Stopped: feedback error ({type(exc).__name__}: {exc})')
        for event in picker.events: conn.send(event)
        picker.events.clear()
        conn.send(('status', picker.status))
        for _ in range(16): p.stepSimulation()  # 240 Hz simulated physics / 15 Hz control

        now = time.perf_counter()
        if now >= next_render:
            next_render = now + 1.0 / SIM_FPS
            if was_busy and picker.camera.rgb is not None:
                # Reuse the actual feedback RGB frame; never render twice per tick.
                out = cv2.cvtColor(picker.camera.rgb, cv2.COLOR_RGB2BGR)
                overlay_mvp = _mvp(picker.camera.view, picker.camera.proj)
            else:
                img = p.getCameraImage(VIEW_W, VIEW_H, view, proj,
                                   renderer=p.ER_TINY_RENDERER,
                                   shadow=1, lightDirection=LIGHT_DIR)
                rgba = np.reshape(img[2], (VIEW_H, VIEW_W, 4)).astype(np.uint8)
                out = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                overlay_mvp = _mvp(view, proj)
            _draw_overlay(out, sim, picker.held, picker.status, overlay_mvp,
                          abs(L1 - L2), L1 + L2)
            frames += 1
            dt = now - fps_t
            if dt >= 0.5:
                fps, frames, fps_t = frames / dt, 0, now
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                conn.send(("frame", (buf.tobytes(), round(fps, 1))))
        # Windows' default timer granularity is ~15 ms, so a short sleep here would
        # overshoot the render deadline. Sleep only for the slack we actually have.
        slack = CONTROL_DT - (time.perf_counter() - tick_start)
        stop.wait(max(0., slack))
    except (EOFError, BrokenPipeError, OSError):
        pass  # parent closed its pipe
    finally:
        stop.set()
        sim.disconnect()
        conn.close()
