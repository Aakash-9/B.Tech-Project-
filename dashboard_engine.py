"""The engine behind the dashboard: perceives, reasons, and drives the arm.

Runs the SAME pipeline the rest of the project is tested against — nothing here is a
special demo path. It works from either a simulated scene or the live webcam; the only
difference is that the simulated one knows ground truth, so accuracy can be shown rather
than asserted.

THREE LOOPS, THREE RATES. Measured on this machine: grabbing a frame costs 31 ms, the
segmenter costs 264 ms, and PyBullet's software renderer costs 56 ms, while annotating and
JPEG-encoding cost under 1 ms together. Running those in one loop meant the whole dashboard
advanced at the speed of its slowest part — about 2 fps. They are now decoupled: video
streams at camera rate, perception publishes whenever it finishes, and the overlay draws
the most recent detections onto the newest frame. This is the ordinary way a perception
stack is built, and it is why the panel looks live even though the network does not run
live.

PyBullet does not just lack thread safety — its software renderer holds the GIL for the
whole 56 ms it draws, so keeping it in this process stalled the camera no matter how the
threads were arranged. It therefore runs in a separate process (sim_worker.py) reached
through SimProxy, and nothing here ever blocks on it.
"""
import math
import threading
import time

import cv2
import numpy as np

from dashboard_sources import make_source
from ik_solver import L1, L2
from sim_proxy import SimProxy
from vision_measure import _rect_world, describe, match_command

# How far the object we think we are holding may drift on the table before we admit we are
# not holding it. The gripper cannot move a real object, so if vision keeps seeing it
# somewhere new, the grasp is fiction and holding onto it just freezes the display.
GRASP_SLIP_CM = 3.0


class _Rate:
    """Measured frames per second, for the header. Cheap enough to leave on."""

    def __init__(self):
        self.t, self.n, self.fps = time.perf_counter(), 0, 0.0

    def tick(self):
        self.n += 1
        dt = time.perf_counter() - self.t
        if dt >= 0.5:
            self.fps, self.n, self.t = self.n / dt, 0, time.perf_counter()


class Engine:
    def __init__(self, source="sim", cam_index=0):
        self.grab_lock = threading.Lock()      # held by the capture loop
        self.proc_lock = threading.Lock()      # held by the perception loop
        self.source = make_source(source, cam_index)
        self.sim = SimProxy(on_event=self._on_sim_event)   # PyBullet, in its own process
        self._pending_grasp_at = None

        self.frame = None
        self.objects = []
        self.grasp_at = None                   # where the held object was when we gripped it
        self.switching = False
        self._reject_seen = set()
        self.log = []
        self.camera_jpeg = None
        self.cam_rate, self.det_rate = _Rate(), _Rate()

        self.sim.workspace(self.source.workspace_cm())
        self._say(f"{self.source.name} started. {self.source.calibration_status()['text']}")

    def _say(self, text):
        self.log.append({"t": time.strftime("%H:%M:%S"), "text": text})
        del self.log[:-40]

    def set_source(self, kind):
        """Swap between the simulated scene and the live webcam without restarting.

        Loading the detector takes several seconds, so `switching` is exposed for the UI
        to show progress instead of looking frozen.
        """
        if kind == ("camera" if self.source.name == "Live camera" else "sim"):
            return
        self.switching = True
        self._say(f"Switching to {'live camera' if kind == 'camera' else 'simulated scene'}…")
        try:
            new = make_source(kind)          # built BEFORE the swap: a failure changes nothing
            with self.grab_lock, self.proc_lock:
                old, self.source = self.source, new
                self.frame, self.objects = None, []
                self._reject_seen.clear()
                old.close()
            self.sim.reset(new.workspace_cm())
            self.grasp_at = None
            self._say(f"{new.name} active. {new.calibration_status()['text']}")
        except Exception as exc:
            self._say(f"Could not switch: {exc}")
        finally:
            self.switching = False

    # --------------------------------------------------------------- rendering
    def annotate(self, frame, objects):
        img = frame.copy()
        if self.source.H0 is None:
            cv2.putText(img, "No calibration - show ArUco markers", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img
        Wt = np.linalg.inv(self.source.H0)
        for o in objects:
            px = cv2.perspectiveTransform(
                np.array([[o["center_cm"]]], np.float32), Wt)[0][0]
            x, y = int(px[0]), int(px[1])
            hot = (o.get("track_id") == self.held)
            col = (60, 255, 60) if hot else (0, 235, 255)
            cv2.polylines(img, [np.asarray(o["mask_px"], np.int32)], True,
                          (90, 90, 90), 1, cv2.LINE_AA)
            # The mask is what was detected; this rectangle is what was MEASURED. Drawing
            # both means a wrong size is visible on the picture instead of only in the table.
            rect = _rect_world(o["length_cm"], o["width_cm"], o.get("angle_deg", 0.0),
                               *o["center_cm"]).astype(np.float32)
            box = cv2.perspectiveTransform(rect.reshape(1, -1, 2), Wt)[0]
            cv2.polylines(img, [box.astype(np.int32)], True, col, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, col, -1)
            cv2.putText(img, f"#{o.get('track_id')} {describe(o)}", (x - 78, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ loops
    @property
    def sim_jpeg(self):
        return self.sim.jpeg

    # held/busy live in the simulation process; mirror them so callers read one place.
    @property
    def held(self):
        return self.sim.held

    @property
    def busy(self):
        return self.sim.busy

    def _on_sim_event(self, kind, arg):
        """Called from the pipe-reader thread when the simulation reports something."""
        if kind == "picked":
            ok, _tid = arg
            self.grasp_at = self._pending_grasp_at if ok else None
            self._say("Gripped and lifted." if ok else "Could not grip it.")
        elif kind == "error":
            self._say("Simulation error: " + str(arg))

    def capture_loop(self):
        """Camera rate. Never waits for the detector — that is the whole point."""
        while True:
            if self.switching:
                time.sleep(0.05)
                continue
            with self.grab_lock:
                frame = self.source.grab()
            if frame is None:
                time.sleep(0.02)
                continue
            self.frame = frame
            img = self.annotate(frame, self.objects)       # newest frame, latest detections
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.camera_jpeg = buf.tobytes()
            self.cam_rate.tick()

    def perceive_loop(self):
        """Detector rate. Publishes objects to the UI and the simulation; blocks nothing."""
        while True:
            frame = self.frame
            if self.switching or frame is None:
                time.sleep(0.02)
                continue
            with self.proc_lock:
                try:
                    objs = self.source.process(frame)
                except Exception as exc:
                    self._say(f"Perception error: {exc}")
                    time.sleep(0.2)
                    continue
                rejected = list(getattr(self.source, "rejected", []))
                notes = list(getattr(self.source, "notes", []))
            self.objects = objs
            if not self.busy:            # do not move the scene under a moving arm
                self.sim.sync(objs)
            self._check_grasp(objs)
            for name, why in rejected:
                key = (name, why.split()[0])
                if key not in self._reject_seen:
                    self._reject_seen.add(key)
                    self._say(f"Ignored a detection — {name}: {why}")
            for name, why in notes:      # kept, but something about it was downgraded
                key = (name, why.split()[0])
                if key not in self._reject_seen:
                    self._reject_seen.add(key)
                    self._say(f"{name}: {why}")
            self.det_rate.tick()

    def _check_grasp(self, objs):
        """Is the thing we believe we are holding still lying on the table somewhere else?

        There is no real arm yet, so nothing stops the operator picking the object up and
        putting it down by hand. When that happens the grasp is stale, and the honest
        response is to let go and follow reality rather than keep showing a fiction.
        """
        if self.held is None or self.busy or self.grasp_at is None:
            return
        for o in objs:
            if o.get("track_id") != self.held:
                continue
            moved = math.dist(o["center_cm"], self.grasp_at)
            if moved > GRASP_SLIP_CM:
                self.grasp_at = None
                self.sim.release()
                self._say(f"{describe(o, with_size=False)} moved {moved:.0f} cm on the "
                          f"table — it is not in the gripper. Releasing.")
            return

    def run_forever(self):
        threading.Thread(target=self.perceive_loop, daemon=True).start()
        self.capture_loop()

    def step(self):
        """One synchronous pass — for tests and scripts, not the dashboard."""
        frame = self.source.grab()
        if frame is None:
            return
        self.frame = frame
        self.objects = self.source.process(frame)
        self.sim.sync(self.objects)
        ok, buf = cv2.imencode(".jpg", self.annotate(frame, self.objects),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.camera_jpeg = buf.tobytes()

    # ---------------------------------------------------------------- reporting
    def state(self):
        rows = []
        for o in sorted(self.objects, key=lambda o: -o["area_cm2"]):
            t = self.source.truth_of(o["label"])
            measured = [round(o["length_cm"], 1), round(o["width_cm"], 1),
                        round(o["height_cm"], 1)]
            row = {
                "id": o.get("track_id"),
                "label": o["label"],
                "colour": o.get("colour_name"),
                "shape": o["shape"],
                "measured": measured,
                "conf": round(float(o.get("conf", 0)), 2),
                "pos": [round(o["center_cm"][0], 1), round(o["center_cm"][1], 1)],
                "height_measured": bool(o.get("height_measured")),
                "held": o.get("track_id") == self.held,
                "true": None, "err_mm": None, "err_dim": None, "err_each": None,
            }
            if t:
                # Worst of the dimensions, not just length — length alone read as "0.4 mm"
                # for a book whose height was out by 9 mm. Height counts ONLY when it was
                # genuinely measured: when the estimator declines and a fallback is shown,
                # the gap is the assumption being wrong, not the vision.
                errs = {"L": abs(o["length_cm"] - t["length_cm"]) * 10,
                        "W": abs(o["width_cm"] - t["width_cm"]) * 10}
                if o.get("height_measured"):
                    errs["H"] = abs(o["height_cm"] - t["height_cm"]) * 10
                worst = max(errs, key=errs.get)
                row.update({
                    "true": [round(t["length_cm"], 1), round(t["width_cm"], 1),
                             round(t["height_cm"], 1)],
                    "err_mm": round(errs[worst], 1),
                    "err_dim": worst,
                    "err_each": {k: round(v, 1) for k, v in errs.items()},
                })
            rows.append(row)

        return {"objects": rows, "log": self.log[-12:], "busy": self.busy,
                "switching": self.switching,
                "is_live": self.source.name == "Live camera",
                "holding": self.held is not None,
                "reach": [round(abs(L1 - L2), 1), round(L1 + L2, 1)],
                "source": self.source.name,
                "has_truth": self.source.has_truth,
                "fps": {"camera": round(self.cam_rate.fps, 1),
                        "detect": round(self.det_rate.fps, 1),
                        "sim": round(self.sim.fps, 1)},
                "calibration": self.source.calibration_status()}

    # ---------------------------------------------------------------- commands
    def command(self, text):
        text = (text or "").strip()
        if not text:
            return
        if text.lower() in ("drop", "release"):
            self.grasp_at = None
            self.sim.release()
            self._say("Released.")
            return
        threading.Thread(target=self._pick, args=(text,), daemon=True).start()

    def _pick(self, text):
        objects = list(self.objects)
        if not objects:
            self._say(f"Nothing is detected, so '{text}' cannot be matched.")
            return
        target, why = match_command(objects, text)
        if target is None:
            seen = ", ".join(describe(o, with_size=False) for o in objects)
            self._say(f"No match for '{text}'. On the table: {seen}")
            return

        x, y = target["center_cm"]
        z = target["height_cm"] / 2.0
        d = math.hypot(x, y, z)
        self._say(f"'{text}' -> {describe(target)} at ({x:.1f}, {y:.1f}) cm  [{why}]")
        if not (abs(L1 - L2) <= d <= L1 + L2):
            self._say(f"Out of reach: {d:.1f} cm (arm reaches "
                      f"{abs(L1 - L2):.1f}-{L1 + L2:.1f} cm).")
            return

        # Posted, not executed: the arm moves in the simulation process and the outcome
        # comes back through _on_sim_event. Nothing on this thread waits for it.
        self._pending_grasp_at = tuple(target["center_cm"])
        self.sim.pick(x, y, z, target.get("track_id"))
