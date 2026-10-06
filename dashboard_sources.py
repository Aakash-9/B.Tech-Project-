"""Where the dashboard's objects come from: a simulated scene, or the real webcam.

Both return the same thing — a frame plus measured, colour-named, tracked objects — so the
dashboard and the arm do not care which is running.

The important difference is TRUTH. The simulated scene knows each object's real size, so
accuracy can be shown rather than claimed. A real camera cannot know that, so live mode
reports detector confidence and calibration status instead, and never invents a "true"
column it has no way to fill.
"""
import os

import cv2
import numpy as np

from aruco_workspace import (area_size, height_gradient, make_detector,
                             workspace_from_corners, MARKER_CM)
from object_tracker import TrailTracker, TRACKER
from sim_accuracy import calibrate_from_frame
from simulator import Camera, cluttered_scene
from vision_measure import (apply_height_hint, filter_inside_workspace, measure_scene,
                            sample_colour)

SIM_TILT_DEG = 22.0
SCENE_SPEED = 0.30

# Live-camera settings (mirrors live_pybullet.py — same reasoning applies there).
PROMPTS = {
    "phone": "smartphone",               # "smartphone" outscored "cell phone" on a real
                                          # phone: 0.57 vs 0.26 on the same frame
    # "water bottle" scored ~0 on a real dark, insulated bottle — CLIP's prior for that
    # phrase leans toward clear plastic bottles, so it lost outright to "drinking cup"
    # (0.66 vs 0.28), which is the exact misclassification that fed a wrong height hint
    # into a 68 cm phantom footprint. Describing the SHAPE instead of naming the product
    # category fixed it: 0.70 vs 0.04 on the same frame, tested against the live screenshot.
    "bottle": "tall narrow drinking bottle with a cap",
    "box": "flat rectangular product box",
    "pen stand": "cylindrical pen holder",
    "cup": "drinking cup",
}
HEIGHT_HINTS_CM = {"bottle": 22.0, "pen stand": 11.0, "cup": 9.0, "box": 4.0, "phone": 1.0}
MODEL = "yoloe-11s-seg.pt"
# 960, not 640. A phone lying on a table is SMALL in frame, and inference size dominates
# there — measured on one real frame: 0.20 at 640, 0.46 at 800, 0.57 at 960, with the same
# prompts. It costs speed, but a confident detection is worth more than a fast miss.
INFER_SIZE = 960
# 0.18, not 0.30. That phone scored 0.26 and was being thrown away just under the old
# threshold, which is why the table read "nothing detected" while it sat in plain view.
MIN_CONF = 0.18
MAX_FOOTPRINT_CM2 = 400
MAX_SPAN_CM = 25.0
FALLBACK_HEIGHT_CM = 3.0
DEFAULT_WORKSPACE_CM = (42.0, 30.0)


class SimSource:
    """A synthetic scene. Knows ground truth, so accuracy is verifiable."""

    name = "Simulated scene"
    has_truth = True

    def __init__(self):
        self.scene = cluttered_scene(camera=Camera(height_cm=50.0, tilt_deg=SIM_TILT_DEG),
                                     moving=True)
        frame = self.scene.render()
        self.H0, self.W0, self.D, self.centres = calibrate_from_frame(frame)
        if self.H0 is None:
            raise RuntimeError("calibration failed on the simulated scene")
        self.trailer = TrailTracker()
        self.rejected = []
        self.t = 0.0
        self._result = self.scene.detection_result()

    def calibration_status(self):
        L, W = area_size(self.centres)
        return {"ok": True,
                "text": f"exact (synthetic camera) — workspace {L:.1f} x {W:.1f} cm"}

    def grab(self):
        """Cheap: advance the scene and render it. No measurement."""
        self.t += SCENE_SPEED
        self.scene.advance(self.t)
        self._result = self.scene.detection_result()   # snapshot, so process() stays in sync
        return self.scene.render()

    def process(self, frame):
        _, objs = measure_scene(self._result, self.H0, self.W0, self.D,
                                max_footprint_cm2=1e9)
        for o in objs:
            o["colour_bgr"], o["colour_name"] = sample_colour(frame, o["mask_px"])
        self.trailer.update(objs)
        objs = self.trailer.apply_remembered_heights(objs)
        return self.trailer.stabilize_sizes(objs)

    def read(self, frozen=False):
        frame = self.scene.render() if frozen else self.grab()
        return frame, self.process(frame)

    def workspace_cm(self):
        return area_size(self.centres)

    def truth_of(self, label):
        for t in self.scene.truth():
            if t["name"] == label:
                return t
        return None

    def close(self):
        pass


class CameraSource:
    """The real webcam: open-vocabulary detection, live ArUco calibration."""

    name = "Live camera"
    has_truth = False

    def __init__(self, cam_index=0):
        from ultralytics import YOLOE                       # slow import; only when needed
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {cam_index} could not be opened")
        # Without this the driver queues frames while the detector thinks, and the panel
        # shows the past. One slot means grab() always returns something current.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.model = YOLOE(MODEL)
        phrases = list(PROMPTS.values())
        self.alias = {v: k for k, v in PROMPTS.items()}
        self.model.set_classes(phrases, self.model.get_text_pe(phrases))

        self.aruco = make_detector()
        self.trailer = TrailTracker()
        self.H0 = self.W0 = self.D = self.centres = None
        self.rejected = []
        self.notes = []               # kept, but with something downgraded — not dropped
        self.live_calib = False
        self.calib_note = "waiting for ArUco markers"

        if os.path.exists("homography_matrix.npy"):
            self.H0 = np.load("homography_matrix.npy")
            if os.path.exists("plane_calib.npz"):
                with np.load("plane_calib.npz") as c:
                    self.W0, self.D = c["W0"], c["D"]
            self.calib_note = ("saved calibration file — sizes are only right if the "
                               "camera has not moved since it was made")

    def calibration_status(self):
        if self.live_calib and self.centres is not None:
            L, W = area_size(self.centres)
            return {"ok": True,
                    "text": f"live ArUco — workspace measured {L:.1f} x {W:.1f} cm"}
        return {"ok": False, "text": self.calib_note}

    def grab(self):
        """Just pull a frame. Runs on its own thread at camera rate (~30 fps)."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def process(self, frame):
        """Everything expensive. Runs on its own thread, as fast as it manages."""
        corners, ids, _ = self.aruco.detectMarkers(frame)
        if ids is not None and len(ids) >= 3:
            Hn, cn = workspace_from_corners(
                [corners[i] for i in np.argsort(ids.flatten())], MARKER_CM)
            if Hn is not None:
                self.H0, self.centres = Hn, cn
                self.W0, self.D = height_gradient(self.H0, frame.shape)
                self.live_calib = True
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if self.H0 is None:
            return []

        r = self.model.track(frame, persist=True, tracker=TRACKER, verbose=False,
                             imgsz=INFER_SIZE)[0]
        # The size filters exist to reject scene-sized detections like the table itself.
        # But they work on centimetres derived from the calibration — so when the
        # calibration is NOT verified, those centimetres are not trustworthy and filtering
        # on them throws away real objects. Measured here: a phone came out as "30 cm" on a
        # stale calibration and was silently dropped for exceeding a 25 cm limit while it
        # sat in plain view. Only apply the size limits once ArUco has confirmed the scale.
        self.rejected = []
        limits = dict(max_footprint_cm2=MAX_FOOTPRINT_CM2, max_span_cm=MAX_SPAN_CM) \
            if self.live_calib else dict(max_footprint_cm2=1e9, max_span_cm=None)
        _, objs = measure_scene(r, self.H0, self.W0, self.D,
                                fallback_height_cm=FALLBACK_HEIGHT_CM,
                                min_conf=MIN_CONF, rejected=self.rejected, **limits)
        if self.live_calib:
            objs = filter_inside_workspace(objs, self.centres)
        for o in objs:
            o["colour_bgr"], o["colour_name"] = sample_colour(frame, o["mask_px"])
            o["label"] = self.alias.get(o["label"], o["label"])
        self.trailer.update(objs)
        objs = self.trailer.apply_remembered_heights(objs)
        # Stabilised BEFORE the height-hint loop, not after: the hint lookup keys off the
        # label, so a track that flickers "cup"/"bottle" between frames should get the
        # bottle's hint consistently, not whichever one won this one frame's coin flip.
        objs = self.trailer.stabilize_labels(objs)
        self.notes = []
        for o in objs:                       # a told height beats a default known to be wrong
            hint = HEIGHT_HINTS_CM.get(o["label"])
            if hint is None or o["height_measured"]:
                continue
            before = dict(o)
            before_len = o["length_cm"]
            apply_height_hint(o, hint, self.H0, self.W0, self.D)
            # A height hint is a guess tied to the DETECTED label. If that label is wrong
            # (a bottle read as "cup" gets the cup's 9 cm instead of the bottle's 22 cm), the
            # correction is known to blow up badly — see footprint_polygon()'s docstring: a
            # wrong height doesn't shrink the footprint, it slides the two unprojected planes
            # apart and stretches their overlap into a long diagonal sliver.
            #
            # This check does NOT gate on live_calib. MAX_SPAN_CM is used here as a hard
            # physical-plausibility bound ("nothing pickable is longer than the table"), not
            # a claim about calibration accuracy — a stale calibration was measured to scale
            # real objects by roughly 2x, not the 5-10x this failure mode produces. And
            # reverting is always safe: it falls back to `before`, the exact fallback-height
            # measurement that would have been shown anyway had no hint ever been applied, so
            # this can only ever remove an implausible number, never introduce one.
            #
            # The SAME sliding-planes mechanism can also collapse the footprint instead of
            # stretching it — seen live: a bottle came back as "3 x 0 x 22 cm", a physically
            # meaningless zero-width sliver, which neither the growth check nor the absolute
            # cap catches (it's small in every dimension that check looks at). Guard on shape
            # too: a real handheld object doesn't have a near-zero width or a 15:1 aspect.
            grew = o["length_cm"] > before_len * 1.3
            absurd = o["length_cm"] > MAX_SPAN_CM or o["area_cm2"] > MAX_FOOTPRINT_CM2
            degenerate = (o["width_cm"] < 1.0
                         or o["length_cm"] > o["width_cm"] * 15.0)
            if grew or absurd or degenerate:
                why = (f"height hint ({hint:.0f} cm) turned {before_len:.0f} cm into "
                       f"{o['length_cm']:.0f} x {o['width_cm']:.1f} cm — likely a "
                       f"misclassification; showing the unhinted measurement instead")
                self.notes.append((o["label"], why))
                o.clear()
                o.update(before)
        # Stabilised LAST: this is the final answer for the frame (after any hint-guard
        # revert), so it's both what gets shown and what future frames smooth against.
        return self.trailer.stabilize_sizes(objs)

    def read(self, frozen=False):
        frame = self.grab()
        return (None, []) if frame is None else (frame, self.process(frame))

    def workspace_cm(self):
        # Without a verified calibration the workspace size is unknown, so show the arm's
        # own reach rather than a made-up table that would look authoritative and be wrong.
        if self.live_calib and self.centres is not None:
            return area_size(self.centres)
        return DEFAULT_WORKSPACE_CM

    def truth_of(self, label):
        return None                          # a real camera has no ground truth

    def close(self):
        self.cap.release()


def make_source(kind, cam_index=0):
    return CameraSource(cam_index) if kind == "camera" else SimSource()
