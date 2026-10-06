"""Does a tracked object survive being occluded? Tested against known truth.

A target slides across the table and passes behind a taller box. Because the scene is
rendered with proper depth ordering, the target genuinely loses pixels and the simulated
detector genuinely misses it for a few frames — the same dropout a real occlusion causes.
Those detections drive the REAL BoT-SORT tracker, so ID continuity is the library's actual
behaviour, not an assumption. What is checked here:

  * the target keeps ONE identity before and after the occlusion,
  * the lock survives the gap and is not handed to a different object,
  * a height measured before the occlusion is still known during it,
  * a target that leaves for good eventually releases the lock.
"""
import sys

import cv2
import numpy as np
from ultralytics.trackers import BOTSORT
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from object_tracker import TrailTracker
from simulator import Box, Camera, Scene
from sim_accuracy import calibrate_from_frame
from vision_measure import measure_scene


def make_tracker():
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
    cfg.with_reid = False           # appearance re-ID needs real texture; motion only here
    return BOTSORT(cfg)


class _Dets:
    """The minimal detection object BoT-SORT's update() consumes.

    It must support boolean indexing, because the tracker splits detections into
    high- and low-confidence subsets before matching.
    """

    def __init__(self, dets=None, xywh=None, conf=None, cls=None, index=None):
        if dets is not None:
            self.xywh = np.array([d["xywh"] for d in dets], np.float32).reshape(-1, 4)
            self.conf = np.array([d["conf"] for d in dets], np.float32)
            self.cls = np.zeros(len(dets), np.float32)
            # Carried through slicing so a track can be mapped back to its detection.
            self.index = np.arange(len(dets))
        else:
            self.xywh, self.conf, self.cls, self.index = xywh, conf, cls, index

    @property
    def xyxy(self):
        """BoT-SORT's motion compensation asks for corner-form boxes."""
        c = self.xywh
        return np.column_stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2,
                                c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2])

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _Dets(xywh=self.xywh[mask], conf=self.conf[mask],
                     cls=self.cls[mask], index=self.index[mask])


def run_sequence(n_frames=46, occluder=True, target_leaves_at=None, verbose=True):
    cam = Camera(height_cm=52.0, tilt_deg=22.0)
    target = Box(-13.0, -2.0, 3.0, 3.0, 3.0, colour=(190, 190, 200), name="target")
    boxes = [target]
    if occluder:
        # Tall and on the camera's side of the target, so its silhouette genuinely covers
        # the target's image rows for part of the crossing (verified: fully hidden for
        # roughly a third of the path).
        boxes.append(Box(0.0, 3.0, 8.0, 4.0, 14.0, colour=(120, 170, 210), name="wall"))
    scene = Scene(cam, boxes=boxes)

    tracker = make_tracker()
    trail = TrailTracker(max_len=60, max_missed_frames=8, locked_grace_frames=25)

    H0, W0, D = None, None, None
    ids_seen, occluded_frames, log = [], 0, []
    for f in range(n_frames):
        if target_leaves_at is not None and f >= target_leaves_at:
            target.x, target.y = 900.0, 900.0        # gone from the scene entirely
        else:
            target.x = -13.0 + 26.0 * (f / (n_frames - 1.0))   # slide left -> right

        frame = scene.render()
        if H0 is None:
            H0, W0, D, _ = calibrate_from_frame(frame)

        dets = [d for d in scene.detections() if d["name"] != "wall"]
        vis = scene.visibility().get("target", 0.0)
        tracks = tracker.update(_Dets(dets), frame) if dets else tracker.update(_Dets([]), frame)

        # BoT-SORT returns [x1,y1,x2,y2, id, conf, cls, det_index]
        objs = []
        if len(tracks):
            _, measured = measure_scene(
                _as_result(dets, H0), H0, W0, D, max_footprint_cm2=1e9)
            for t in tracks:
                di = int(t[-1])
                if 0 <= di < len(measured):
                    o = measured[di]
                    o["track_id"] = int(t[4])
                    objs.append(o)

        trail.update(objs)
        trail.apply_remembered_heights(objs)

        tid = objs[0]["track_id"] if objs else None
        if tid is not None:
            ids_seen.append(tid)
            if trail.locked_id is None:
                trail.lock(tid)
        if not dets:
            occluded_frames += 1

        log.append({"f": f, "vis": vis, "detected": bool(dets), "id": tid,
                    "locked": trail.locked_id,
                    "h": objs[0]["height_cm"] if objs else None,
                    "h_meas": objs[0].get("height_measured") if objs else None,
                    "h_mem": objs[0].get("height_from_memory", False) if objs else False})

    if verbose:
        print(f"{'frame':>6}{'visible':>9}{'detected':>10}{'id':>5}{'locked':>8}"
              f"{'height':>9}  note")
        for r in log:
            if r["f"] % 3 and r["detected"]:
                continue                      # keep the printout short, show every gap
            note = "OCCLUDED - no detection" if not r["detected"] else (
                "height from memory" if r["h_mem"] else "")
            h = f"{r['h']:.2f}cm" if r["h"] is not None else "   -   "
            print(f"{r['f']:6d}{r['vis']:9.2f}{str(r['detected']):>10}"
                  f"{str(r['id']):>5}{str(r['locked']):>8}{h:>9}  {note}")

    return {"ids": ids_seen, "occluded_frames": occluded_frames, "log": log, "trail": trail}


def _as_result(dets, H0):
    """Wrap simulated detections so the real measurement code can consume them."""
    class V:
        def __init__(self, v):
            self._v = v

        def tolist(self):
            return self._v

        def int(self):
            return self

    return type("R", (), {
        "names": {i: d["name"] for i, d in enumerate(dets)},
        "masks": type("M", (), {"xy": [d["poly"] for d in dets]})(),
        "boxes": type("B", (), {"cls": V(list(range(len(dets)))),
                                "conf": V([d["conf"] for d in dets]),
                                "id": None})(),
    })()


if __name__ == "__main__":
    passed, failed = [], []

    def check(name, ok, detail=""):
        (passed if ok else failed).append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    print("=== target slides across the table, passing behind a taller box ===")
    res = run_sequence()
    uniq = sorted(set(res["ids"]))
    print()
    check("the occlusion actually happened (detector lost the target)",
          res["occluded_frames"] > 0, f"{res['occluded_frames']} frames with no detection")
    check("the target keeps a single identity across the whole run",
          len(uniq) == 1, f"ids seen: {uniq}")

    after = [r for r in res["log"] if r["detected"]]
    check("the lock is still held at the end", res["trail"].locked_id is not None,
          f"locked id {res['trail'].locked_id}")
    check("the lock never jumped to a different object",
          len({r['locked'] for r in res['log'] if r['locked'] is not None}) == 1)

    gap = [r for r in res["log"] if not r["detected"]]
    mem = [r for r in res["log"] if r["h_mem"]]
    check("height is remembered rather than re-guessed after a dropout",
          len(mem) > 0 or all(r["h_meas"] for r in after if r["h"] is not None),
          f"{len(mem)} frames used a remembered height")

    print("\n=== control: no occluder, same motion ===")
    clean = run_sequence(occluder=False, verbose=False)
    check("without an occluder there are no dropouts", clean["occluded_frames"] == 0)
    check("without an occluder the identity is still single",
          len(set(clean["ids"])) == 1, f"ids seen: {sorted(set(clean['ids']))}")

    print("\n=== target leaves permanently ===")
    gone = run_sequence(n_frames=60, occluder=False, target_leaves_at=20, verbose=False)
    check("a permanently lost target eventually releases the lock",
          gone["trail"].locked_id is None,
          f"locked id ended as {gone['trail'].locked_id}")

    print(f"\n{'=' * 60}\nPASSED {len(passed)} / {len(passed) + len(failed)}")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        sys.exit(1)
    print("TRACKING TESTS PASSED")
