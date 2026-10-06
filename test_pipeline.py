"""Regression tests for the whole pipeline, checked against exact known geometry.

Run this after any change to the vision or kinematics maths:
    venv/Scripts/python.exe test_pipeline.py

Everything here is scored against ground truth rather than eyeballed, so a number moving
in the wrong direction shows up immediately instead of at demo time.
"""
import sys

import cv2
import numpy as np

from aruco_workspace import (area_size, height_gradient, make_detector,
                             workspace_from_corners, MARKER_CM)
from ik_solver import forward_kinematics, inverse_kinematics
from object_tracker import TrailTracker
from simulator import Camera, default_scene
from vision_measure import (filter_inside_workspace, fit_two_plane_calibration,
                            homography_at_height, measure_polygon, measure_scene,
                            select_target)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def synthetic_camera(cam_h=55.0, tilt=20.0, f=600.0, size=(640, 480)):
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    a = np.radians(tilt)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) \
        @ np.diag([1.0, -1.0, -1.0])
    t = np.array([0.0, 0.0, cam_h])
    w2i = lambda h: K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * h + t])
    proj = lambda p, h: cv2.perspectiveTransform(
        np.asarray(p, np.float32).reshape(-1, 1, 2), w2i(h)).reshape(-1, 2)
    return proj


def rect(length, width, angle_deg, cx, cy):
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    pts = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                    [length / 2, width / 2], [-length / 2, width / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


# ------------------------------------------------------------------ kinematics
print("\n=== 1. Kinematics ===")
b, s, e, w = inverse_kinematics(12.0, 8.0, 5.0)
pts = forward_kinematics(b, s, float(np.hypot(12.0, 8.0)), 5.0)
check("forward kinematics lands exactly on the IK target",
      max(abs(pts[2][i] - v) for i, v in enumerate((12.0, 8.0, 5.0))) < 1e-9)
check("base angle points at the target",
      abs(b - np.degrees(np.arctan2(8.0, 12.0))) < 1e-9, f"base={b:.2f} deg")

for dist, should_work, why in ((20.4, True, "just inside outer reach"),
                               (20.6, False, "just outside outer reach"),
                               (0.9, True, "just outside inner limit"),
                               (0.2, False, "inside the inner limit")):
    try:
        inverse_kinematics(dist, 0.0, 0.0)
        ok = should_work
    except ValueError:
        ok = not should_work
    check(f"{why} ({dist} cm)", ok)

_, _, e_near, _ = inverse_kinematics(6.0, 0.0, 0.0)
_, _, e_far, _ = inverse_kinematics(19.0, 0.0, 0.0)
check("elbow straightens as the target gets further away", e_far > e_near,
      f"{e_near:.0f} deg at 6 cm -> {e_far:.0f} deg at 19 cm")

# ------------------------------------------------- two-plane calibration maths
print("\n=== 2. Two-plane calibration ===")
proj = synthetic_camera()
A4 = np.array([[-10.5, -14.85], [10.5, -14.85], [10.5, 14.85], [-10.5, 14.85]], np.float32)
H0, W0, D = fit_two_plane_calibration(proj(A4, 0.0), proj(A4, 5.0), A4, 5.0)
check("second plane yields a height gradient", D is not None)
_, _, D_none = fit_two_plane_calibration(proj(A4, 0.0), None, A4, 5.0)
check("skipping the second plane yields no gradient", D_none is None)

probe = np.array([[4.0, -6.0], [-8.0, 9.0], [0.0, 0.0]], np.float32)
for h in (0.0, 3.0, 5.0, 8.0):
    got = cv2.perspectiveTransform(proj(probe, h).reshape(-1, 1, 2),
                                   homography_at_height(W0, D, h)).reshape(-1, 2)
    err = np.abs(got - probe).max()
    check(f"mapping exact at height {h:.0f} cm{' (extrapolated)' if h > 5 else ''}",
          err < 0.01, f"max error {err * 10:.4f} mm")

# ------------------------------------------------------- measurement accuracy
print("\n=== 3. Measurement on a known plane ===")


def fake_result(polys, labels=None):
    labels = labels or [f"o{i}" for i in range(len(polys))]

    class V:
        def __init__(self, v):
            self._v = v

        def tolist(self):
            return self._v

        def int(self):
            return self

    return type("R", (), {
        "names": {i: n for i, n in enumerate(labels)},
        "masks": type("M", (), {"xy": polys})(),
        "boxes": type("B", (), {"cls": V(list(range(len(polys)))),
                                "conf": V([0.9] * len(polys)),
                                "id": V(list(range(1, len(polys) + 1)))})(),
    })()


m = measure_polygon(proj(rect(8.0, 5.0, 30.0, 3.0, -2.0), 0.0), H0)
check("planar object measured exactly", abs(m["length_cm"] - 8) < 0.05 and
      abs(m["width_cm"] - 5) < 0.05, f"{m['length_cm']:.2f} x {m['width_cm']:.2f} cm")
check("long-axis angle recovered", abs(m["angle_deg"] - 30.0) < 1.5,
      f"{m['angle_deg']:.1f} deg")
check("grasp angle is perpendicular to the long axis",
      abs(((m["grasp_deg"] - m["angle_deg"]) % 180) - 90) < 1.5)

for name, poly in (("square", rect(4, 4, 0, 0, 0)),
                   ("rectangle", rect(12, 4, 0, 0, 0)),
                   ("long strip", rect(18, 2, 0, 0, 0)),
                   ("circle", [[4 * np.cos(t), 4 * np.sin(t)]
                               for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]),
                   ("triangle", [[-4, -3], [4, -3], [0, 4]])):
    got = measure_polygon(proj(np.asarray(poly, np.float32), 0.0), H0)
    check(f"'{name}' classified correctly", got["shape"] == name, f"got '{got['shape']}'")

check("a frame with no masks returns nothing and does not crash",
      measure_scene(type("R", (), {"names": {}, "masks": None, "boxes": None})(), H0)[1] == [])

# ------------------------------------------------------ end-to-end on a render
print("\n=== 4. End-to-end on rendered scenes (real ArUco + real measurement) ===")
det = make_detector()
worst_len = worst_pos = 0.0
h_errs = []
for tilt in (10.0, 18.0, 28.0, 40.0):
    scene = default_scene(camera=Camera(height_cm=50.0, tilt_deg=tilt))
    frame = scene.render()
    corners, ids, _ = det.detectMarkers(frame)
    check(f"all 4 markers detected at {tilt:.0f} deg tilt",
          ids is not None and len(ids) == 4, f"found {0 if ids is None else len(ids)}")
    if ids is None or len(ids) < 3:
        continue
    order = np.argsort(ids.flatten())
    Hs, centres = workspace_from_corners([corners[i] for i in order], MARKER_CM)
    aw, ah = area_size(centres)
    check(f"workspace size within 5% at {tilt:.0f} deg tilt",
          abs(aw - 32.0) / 32.0 < 0.05 and abs(ah - 24.0) / 24.0 < 0.05,
          f"{aw:.2f} x {ah:.2f} cm (true 32.00 x 24.00)")

    Ws, Ds = height_gradient(Hs, frame.shape)
    truth = {t["name"]: t for t in scene.truth()}
    _, objs = measure_scene(scene.detection_result(), Hs, Ws, Ds, max_footprint_cm2=1e9)
    for o in objs:
        t = truth[o["label"]]
        worst_len = max(worst_len, abs(o["length_cm"] - t["length_cm"]))
        worst_pos = max(worst_pos, np.hypot(o["center_cm"][0] - t["center_cm"][0],
                                            o["center_cm"][1] - t["center_cm"][1]))
        if o["height_measured"]:
            h_errs.append(abs(o["height_cm"] - t["height_cm"]))

# 20 mm, not 10: the simulator now returns VISIBLE, RASTERISED outlines like a real
# segmentation mask, instead of exact geometric hulls. That pixel quantisation costs a few
# mm on small objects at steep tilt (worst seen: a 5 cm cube at 40 deg). Tightening this
# back would only be possible by making the test input less realistic than reality.
check("worst length error under 20 mm across all tilts", worst_len < 2.0,
      f"{worst_len * 10:.1f} mm")
check("worst position error under 10 mm across all tilts", worst_pos < 1.0,
      f"{worst_pos * 10:.1f} mm")
check("mean height error under 5 mm where measured", np.mean(h_errs) < 0.5,
      f"{np.mean(h_errs) * 10:.1f} mm over {len(h_errs)} measurements")

# --------------------------------------------------- selection and workspace
print("\n=== 5. Selection, workspace filter ===")
scene = default_scene(camera=Camera(height_cm=50.0, tilt_deg=20.0))
frame = scene.render()
corners, ids, _ = det.detectMarkers(frame)
order = np.argsort(ids.flatten())
Hs, centres = workspace_from_corners([corners[i] for i in order], MARKER_CM)
Ws, Ds = height_gradient(Hs, frame.shape)
_, objs = measure_scene(scene.detection_result(), Hs, Ws, Ds, max_footprint_cm2=1e9)

check("'smallest' picks the cube", select_target(objs, size="smallest")["label"] == "cube")
check("'biggest' picks the book", select_target(objs, size="biggest")["label"] == "book")
check("'square' shape query returns a square",
      (select_target(objs, shape="square") or {}).get("shape") == "square")
check("querying an absent shape returns nothing",
      select_target(objs, shape="triangle") is None)

inside = filter_inside_workspace(objs, centres)
check("all scene objects are inside the workspace", len(inside) == len(objs))
far = [dict(objs[0], center_cm=(200.0, 200.0))]
check("an object far outside the workspace is filtered out",
      len(filter_inside_workspace(far, centres)) == 0)

# -------------------------------------------------------- tracking behaviour
print("\n=== 6. Tracking, lock and height memory ===")
tr = TrailTracker(max_len=3, max_missed_frames=4, locked_grace_frames=10)
for x in range(5):
    tr.update([{"center_cm": (x, 0.0), "track_id": 1, "height_measured": False}])
check("trail caps at max_len", len(tr.trail_of(1)) == 3)
check("trail keeps the most recent point", list(tr.trail_of(1)[-1]) == [4.0, 0.0])

tr.lock(1)
for _ in range(tr.max_missed + 2):
    tr.update([])
check("locked id survives longer than an ordinary trail would", tr.locked_id == 1)
for _ in range(tr.locked_grace + 2):
    tr.update([])
check("locked id is released once genuinely lost", tr.locked_id is None)

tr2 = TrailTracker()
tr2.update([{"center_cm": (0, 0), "track_id": 7, "height_cm": 3.0, "height_measured": True}])
objs2 = [{"center_cm": (0, 0), "track_id": 7, "height_cm": 9.9, "height_measured": False}]
tr2.update(objs2)
tr2.apply_remembered_heights(objs2)
check("a remembered height fills in when a frame's estimate declines",
      objs2[0]["height_measured"] and abs(objs2[0]["height_cm"] - 3.0) < 1e-9,
      f"height now {objs2[0]['height_cm']:.2f} cm")

# ------------------------------------------------- realistic (ragged) masks
print("\n=== 7. Degraded masks (what a real segmentation actually returns) ===")
rng = np.random.default_rng(0)
for bias, jitter, rnd, note in ((0.0, 0.0, 1.0, "corners rounded"),
                                (0.0, 0.3, 0.0, "border wobble"),
                                (1.0, 0.3, 1.0, "over-segmented"),
                                (-1.0, 0.3, 1.0, "under-segmented")):
    lens, hts = [], []
    for tilt in (14.0, 20.0, 28.0):
        sc = default_scene(camera=Camera(height_cm=50.0, tilt_deg=tilt))
        frame = sc.render()
        corners, ids, _ = det.detectMarkers(frame)
        if ids is None or len(ids) < 3:
            continue
        Hd, _ = workspace_from_corners([corners[i] for i in np.argsort(ids.flatten())],
                                       MARKER_CM)
        Wd, Dd = height_gradient(Hd, frame.shape)
        truth = {t["name"]: t for t in sc.truth()}
        sils = sc.degraded_silhouettes(rng, bias, jitter, rnd)
        if not sils:
            continue
        _, objs = measure_scene(fake_result([p for _, p in sils], [n for n, _ in sils]),
                                Hd, Wd, Dd, max_footprint_cm2=1e9)
        for o in objs:
            t = truth[o["label"]]
            lens.append(abs(o["length_cm"] - t["length_cm"]))
            if o["height_measured"]:
                hts.append(abs(o["height_cm"] - t["height_cm"]))
    check(f"length stays under 10 mm with a {note} mask",
          lens and np.mean(lens) < 1.0, f"{np.mean(lens) * 10:.1f} mm")
    # Ragged masks used to produce ~50 mm height errors; the simplification plus the
    # "needs a second agreeing edge" rule is what keeps this bounded.
    check(f"height stays under 15 mm with a {note} mask",
          (not hts) or np.mean(hts) < 1.5,
          f"{np.mean(hts) * 10:.1f} mm over {len(hts)} measurements" if hts
          else "none measured (declined, which is safe)")

print(f"\n{'=' * 62}\nPASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
print("ALL TESTS PASSED")
