"""Marking four points by hand must give true metric measurements.

This is the calibration route with no prerequisites: no printer, no sheet the detector
might refuse, and — unlike a found sheet — the operator chooses where the origin sits,
which matters because everything downstream treats world (0,0) as the arm's base.

It is also the route with the most room for a silent mistake. A homography through four
points always "succeeds"; click them in the wrong order and the result is mirrored or
sheared while still reporting a clean fit. So these tests check the recovered scale against
distances the calibration never saw, and check that a bad click order is visible in the
residual rather than hidden.
"""
import sys

import cv2
import numpy as np

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


def fit(points_px, w_cm, h_cm):
    """The same fit CameraSource.calibrate_from_points performs."""
    pts = np.asarray(points_px, np.float32).reshape(-1, 2)
    world = np.array([[-w_cm / 2, h_cm / 2], [w_cm / 2, h_cm / 2],
                      [w_cm / 2, -h_cm / 2], [-w_cm / 2, -h_cm / 2]], np.float32)
    H0, _ = cv2.findHomography(pts, world)
    got = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H0).reshape(-1, 2)
    return H0, float(np.max(np.linalg.norm(got - world, axis=1)))


def span_cm(H0, cam, a, b):
    """Measure a real table distance through the calibration."""
    p = cam.project(np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]]))
    w = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), H0).reshape(-1, 2)
    return float(np.linalg.norm(w[1] - w[0]))


def main():
    cam = Camera(height_cm=55.0, tilt_deg=40.0)
    W, H = 24.0, 18.0                      # the rectangle the operator marks and measures
    # In the order the UI tells the operator to click: top-left on screen, then clockwise.
    # For a down-looking camera that is the far edge first (+y), then the near edge.
    rect = np.array([[-W / 2, H / 2], [W / 2, H / 2], [W / 2, -H / 2], [-W / 2, -H / 2]])
    clicks = cam.project(np.column_stack([rect, np.zeros(4)]))

    print("--- a correctly clicked rectangle gives true metric scale ---")
    H0, res = fit(clicks, W, H)
    check("corner fit residual under 0.5 mm", res * 10 < 0.5, f"{res*10:.3f} mm")
    for a, b, true in (((-9, -6), (9, -6), 18.0), ((0, -8), (0, 8), 16.0),
                       ((-5, -5), (5, 5), float(np.hypot(10, 10))), ((2, 2), (4, 2), 2.0)):
        got = span_cm(H0, cam, a, b)
        check(f"a true {true:.1f} cm distance measures {true:.1f} cm",
              abs(got - true) < 0.15, f"{got:.3f} cm")

    # Distances survive a reflection, so the span checks above would ALL still pass with a
    # mirrored frame. Only a signed position catches it — and a mirrored frame sends the
    # arm to the wrong side of the table with nothing visibly wrong on screen.
    print("\n--- positions are not mirrored (what a bad winding silently does) ---")
    for probe in ((6.0, 4.0), (-7.0, 3.0), (5.0, -6.0)):
        got = cv2.perspectiveTransform(
            cam.project(np.array([[probe[0], probe[1], 0.0]])).reshape(-1, 1, 2)
            .astype(np.float32), H0).reshape(2)
        check(f"world {probe} comes back as itself",
              float(np.linalg.norm(got - np.asarray(probe))) < 0.1,
              f"({got[0]:+.2f}, {got[1]:+.2f})")

    print("\n--- the origin lands where the operator put the rectangle ---")
    centre = cv2.perspectiveTransform(
        cam.project(np.array([[0.0, 0.0, 0.0]])).reshape(-1, 1, 2).astype(np.float32),
        H0).reshape(2)
    check("the rectangle's centre is world (0,0)", float(np.linalg.norm(centre)) < 0.05,
          f"({centre[0]:.3f}, {centre[1]:.3f})")

    # The residual CANNOT catch this: a homography through four points always fits, and a
    # transposed pair still fits to 0.0 mm while mirroring every coordinate afterwards.
    # Measured that directly before adding the geometric checks below.
    print("\n--- a wrong click ORDER is caught, which the residual alone cannot do ---")
    swapped = clicks[[0, 2, 1, 3]]         # corners 2 and 3 transposed: a bow-tie
    _, res_bad = fit(swapped, W, H)
    check("the residual really is blind to it (why the extra checks exist)",
          res_bad * 10 < 1.0, f"crossed order still fits to {res_bad*10:.3f} mm")

    import dashboard_sources as ds
    src = ds.CameraSource.__new__(ds.CameraSource)
    src.frame_shape = (480, 640, 3)
    try:
        src.calibrate_from_points(swapped, W, H, save=False)
        check("crossed corners are refused", False, "accepted a bow-tie")
    except ValueError as exc:
        check("crossed corners are refused", "cross over" in str(exc), str(exc)[:46])
    try:
        src.calibrate_from_points(clicks[::-1], W, H, save=False)
        check("anticlockwise corners are refused", False, "accepted a mirrored frame")
    except ValueError as exc:
        check("anticlockwise corners are refused", "anticlockwise" in str(exc),
              str(exc)[:46])

    print("\n--- stated size is what sets the scale ---")
    H_half, _ = fit(clicks, W / 2, H / 2)
    got = span_cm(H_half, cam, (-9, -6), (9, -6))
    check("halving the stated size halves every measurement",
          abs(got - 9.0) < 0.1, f"18 cm reads {got:.2f} cm when the rectangle is called half size")

    print("\n--- bad input is refused, not fitted ---")
    import dashboard_sources as ds
    src = ds.CameraSource.__new__(ds.CameraSource)   # no camera needed for the guards
    src.frame_shape = (480, 640, 3)
    for pts, w, h, why in ((clicks[:3], W, H, "only three points"),
                           (clicks, 0.0, H, "zero width"),
                           (clicks, W, -5.0, "negative height")):
        try:
            src.calibrate_from_points(pts, w, h, save=False)
            check(f"{why} is refused", False, "it was accepted")
        except (ValueError, AttributeError) as exc:
            check(f"{why} is refused", True, str(exc)[:48])

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("MANUAL CALIBRATION OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
