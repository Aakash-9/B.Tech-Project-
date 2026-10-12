"""Can the camera's tilt be measured from the calibration, accurately enough to act on?

Camera tilt is the single biggest factor in whether object heights can be measured at all.
Swept on the synthetic scene: below 30 degrees the heights that do come back are wrong by
up to 48 mm, while at 36-44 degrees the error collapses to 1-3 mm. Nobody can judge that
angle by eye, so the dashboard measures it and tells the operator which way to move.

The estimate comes free from the existing calibration: it already recovers r3, the table's
normal in camera coordinates, and |r3[2]| is cos(tilt). (Getting that backwards — arcsin
instead of arccos — returns the exact complement of the right answer, which looks
plausible, so the straight-angle cases below are the guard against that.)
"""
import sys

from aruco_workspace import GOOD_TILT_DEG, camera_tilt_deg, tilt_advice
from sim_accuracy import calibrate_from_frame
from simulator import Camera, cluttered_scene

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def estimate(true_tilt, cam_height=50.0):
    scene = cluttered_scene(camera=Camera(height_cm=cam_height, tilt_deg=float(true_tilt)),
                            moving=False)
    frame = scene.render()
    H0, _, D, _ = calibrate_from_frame(frame)
    if H0 is None:
        return None
    return camera_tilt_deg(H0, D, frame.shape)


def main():
    print("--- tilt is recovered accurately across the usable band ---")
    errs = []
    for true_tilt in (32, 36, 40, 44, 48):
        est = estimate(true_tilt)
        if est is None:
            check(f"{true_tilt} deg estimated", False, "no estimate")
            continue
        err = abs(est - true_tilt)
        errs.append(err)
        check(f"{true_tilt} deg -> {est:.1f} deg", err < 5.0, f"error {err:.1f} deg")
    if errs:
        check("average error stays under 3 degrees", sum(errs) / len(errs) < 3.0,
              f"{sum(errs)/len(errs):.1f} deg")

    print("\n--- not the complement (the arcsin/arccos trap) ---")
    est = estimate(25)
    check("a near-overhead camera reads near-overhead (small angle)",
          est is not None and est < 45,
          f"25 deg -> {est:.1f} deg" if est else "no estimate")
    est = estimate(50)
    check("a flatter camera reads flatter (large angle)", est is not None and est > 40,
          f"50 deg -> {est:.1f} deg" if est else "no estimate")

    # The direction of the advice is the part that was once WRONG: the angle is measured
    # from the table normal (0 = overhead), but the message described it as "below
    # horizontal" and derived the direction from that, so it told operators to move the
    # camera the opposite way. These check the physical direction, not just the wording.
    print("\n--- the advice points the operator the RIGHT way ---")
    lo, hi = GOOD_TILT_DEG
    too_overhead = tilt_advice(lo - 12) or ""
    too_flat = tilt_advice(hi + 12) or ""
    check("a too-OVERHEAD camera is never told to aim further down",
          "DOWN" not in too_overhead.upper(), too_overhead)
    check("a too-OVERHEAD camera is told it is too overhead",
          "overhead" in too_overhead, too_overhead)
    check("a too-FLAT camera IS told to aim further down",
          "DOWN" in too_flat.upper(), too_flat)
    check("a good angle is confirmed", "good" in (tilt_advice((lo + hi) / 2) or ""),
          tilt_advice((lo + hi) / 2))
    check("the message is phrased below-horizontal, not raw normal angle",
          "below horizontal" in (tilt_advice((lo + hi) / 2) or ""),
          tilt_advice((lo + hi) / 2))
    check("unknown tilt says nothing rather than guessing", tilt_advice(None) is None)

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("CAMERA TILT OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
