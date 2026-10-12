"""Calibrating from a plain sheet of paper must give true metric measurements.

This exists because running on a STALE saved calibration is not a mild degradation: observed
live, it turned a 7 cm bottle into "73 x 6 cm" and placed an object on the table "behind the
arm", while the dashboard reported both with full confidence. The ArUco path prevents that
but needs a printer, and the click-four-corners script needs a GUI and a steady hand. A
sheet of A4 is 21.0 x 29.7 cm everywhere, needs nothing, and carries the same metric scale.

Tests run against a synthetic camera whose geometry is known exactly, so "correct" here
means correct against ground truth, not merely self-consistent.
"""
import sys

import cv2
import numpy as np

from paper_calib import (SHEET_CM, calibrate_from_sheet, find_sheet, order_corners,
                         residual_cm, sheet_world_points)
from simulator import Camera

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def render_sheet(cam, size_cm=SHEET_CM, yaw_deg=0.0, centre=(0.0, 0.0), bg=38, fg=238):
    """A bright sheet lying flat on a dark table, as this camera would see it."""
    img = np.full((cam.h, cam.w, 3), bg, np.uint8)
    sw, sl = size_cm
    a = np.radians(yaw_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-sw / 2, -sl / 2], [sw / 2, -sl / 2],
                        [sw / 2, sl / 2], [-sw / 2, sl / 2]])
    corners = (corners @ rot.T) + np.asarray(centre)
    px = cam.project(np.column_stack([corners, np.zeros(4)]))
    cv2.fillPoly(img, [np.round(px).astype(np.int32)], (fg, fg, fg))
    return img, corners


def main():
    print("--- a sheet is found and measured correctly from several viewpoints ---")
    for tilt, yaw, centre in ((38.0, 0.0, (0.0, 0.0)), (30.0, 25.0, (2.0, -3.0)),
                              (45.0, -40.0, (-1.5, 2.0)), (50.0, 70.0, (0.0, 1.0))):
        cam = Camera(height_cm=55.0, tilt_deg=tilt)
        img, truth_corners = render_sheet(cam, yaw_deg=yaw, centre=centre)
        H0, W0, D, corners, size_cm = calibrate_from_sheet(img)
        if H0 is None:
            check(f"tilt {tilt:.0f} yaw {yaw:.0f}: sheet detected", False, "not found")
            continue
        check(f"tilt {tilt:.0f} yaw {yaw:.0f}: sheet detected", True,
              f"read as {size_cm[0]:.1f} x {size_cm[1]:.1f} cm")
        world, _ = sheet_world_points(corners, SHEET_CM)
        res = residual_cm(H0, corners, world)
        check(f"tilt {tilt:.0f} yaw {yaw:.0f}: fit residual under 2 mm", res < 0.2,
              f"{res*10:.2f} mm")

        # The real question: does a KNOWN distance on the table measure correctly?
        probe = cam.project(np.array([[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0]]))
        got = cv2.perspectiveTransform(probe.reshape(-1, 1, 2).astype(np.float32),
                                       H0).reshape(-1, 2)
        span = float(np.linalg.norm(got[1] - got[0]))
        check(f"tilt {tilt:.0f} yaw {yaw:.0f}: a true 12 cm span reads 12 cm",
              abs(span - 12.0) < 0.3, f"{span:.2f} cm")

    print("\n--- it declines rather than inventing a calibration ---")
    cam = Camera(height_cm=55.0, tilt_deg=38.0)
    blank = np.full((cam.h, cam.w, 3), 40, np.uint8)
    check("an empty table yields no sheet", find_sheet(blank) is None)
    noise = np.random.default_rng(0).integers(0, 255, (cam.h, cam.w, 3), dtype=np.uint8)
    got = find_sheet(noise)
    check("random noise yields no sheet", got is None,
          "found something" if got is not None else "")
    circle = np.full((cam.h, cam.w, 3), 38, np.uint8)
    cv2.circle(circle, (cam.w // 2, cam.h // 2), 120, (240, 240, 240), -1)
    check("a round bright blob is not mistaken for a sheet", find_sheet(circle) is None)

    print("\n--- corner ordering is rotation-proof ---")
    base = np.array([[100.0, 80.0], [300.0, 80.0], [300.0, 220.0], [100.0, 220.0]])
    for turn in range(4):
        out = order_corners(np.roll(base, turn, axis=0))
        check(f"rotation {turn}: same four corners, canonical order",
              sorted(map(tuple, np.round(out))) == sorted(map(tuple, np.round(base))))

    print("\n--- the long side is assigned to the long physical dimension ---")
    _, s1 = sheet_world_points(np.array([[0., 0.], [300., 0.], [300., 100.], [0., 100.]]))
    check("a landscape sheet maps 29.7 across", s1 == (29.7, 21.0), str(s1))
    _, s2 = sheet_world_points(np.array([[0., 0.], [100., 0.], [100., 300.], [0., 300.]]))
    check("a portrait sheet maps 21.0 across", s2 == (21.0, 29.7), str(s2))

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("PAPER CALIBRATION OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
