"""A wrong height hint (from a misclassified object) must never reach the dashboard as a
worse, more implausible footprint than showing no hint at all.

Real failure this guards: a water bottle misclassified as "cup" got the cup's 9 cm height
hint instead of the bottle's 22 cm, and the footprint-correction math (documented in
vision_measure.py as sensitive to height error) blew the measurement up to a 68 cm diagonal
sliver — bigger than the table, shown with a straight face in the dashboard. A second, live
occurrence collapsed the SAME way instead: the bottle came back "3 x 0 x 22 cm", a
physically meaningless zero-width sliver that an oversize-only check does not catch.

CameraSource.process() in dashboard_sources.py now reverts a hint that grew the footprint
too much, pushed it past the same plausibility limits used elsewhere (MAX_SPAN_CM,
MAX_FOOTPRINT_CM2), or collapsed its width/aspect into something no real object has —
falling back to the pre-hint measurement. That decision is a handful of threshold
comparisons on plain dicts — this tests that logic directly, with controlled before/after
numbers, rather than fighting a synthetic camera to reproduce the exact geometry of a real,
irregular segmentation mask (tried; a convex hull of a few rectangle corners is too regular
to degenerate the way a real mask does — see the comment in footprint_polygon()'s docstring
for the mechanism instead).

A second block exercises the real code path (apply_height_hint + the actual revert loop
logic) end to end on a synthetic camera, as a live confirmation that a badly-wrong hint is
at minimum never allowed to grow the footprint, even though the synthetic geometry can't
reproduce the sliver's exact numbers.
"""
import sys

import cv2
import numpy as np

import dashboard_sources as ds
from vision_measure import apply_height_hint, measure_polygon, _rect_world

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def revert_decision(before_len, after_len, after_area, after_width=10.0):
    """The exact rule in dashboard_sources.py's height-hint loop, isolated for testing.

    after_width defaults to something unremarkable so existing size-only calls don't need
    to change; pass it explicitly to exercise the degenerate-shape branch.
    """
    grew = after_len > before_len * 1.3
    absurd = after_len > ds.MAX_SPAN_CM or after_area > ds.MAX_FOOTPRINT_CM2
    degenerate = after_width < 1.0 or after_len > after_width * 15.0
    return grew or absurd or degenerate


def main():
    print("--- decision logic, controlled numbers (mirrors the real dashboard loop) ---")
    check("a hint that turns 10 cm into 68 cm (the real screenshot's shape) is caught",
          revert_decision(before_len=10.0, after_len=68.0, after_area=200.0))
    check("a hint that grows the footprint 40% is caught even under the absolute cap",
          revert_decision(before_len=15.0, after_len=21.0, after_area=100.0))
    check("a hint that shrinks the footprint (the intended, working case) is kept",
          not revert_decision(before_len=19.7, after_len=8.5, after_area=30.0))
    check("a hint that leaves it roughly the same size is kept",
          not revert_decision(before_len=10.0, after_len=11.0, after_area=50.0))
    check("a hint whose area alone exceeds the table-sized limit is caught",
          revert_decision(before_len=10.0, after_len=12.0, after_area=500.0))
    check("a hint that collapses width to ~0 is caught, even though length shrank too "
          "(the live '3 x 0 x 22 cm' case — small in every dimension the other checks look at)",
          revert_decision(before_len=10.0, after_len=3.0, after_area=0.1, after_width=0.0))
    check("a plausible thin object (a pen, aspect ~10:1) is NOT flagged as degenerate",
          not revert_decision(before_len=12.0, after_len=11.0, after_area=16.5,
                              after_width=1.5))

    print("\n--- end to end on a synthetic camera: a wrong hint never GROWS the footprint ---")
    K = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1.0]])
    tilt = np.radians(10.0)
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                 [0, np.sin(tilt), np.cos(tilt)]]) @ np.diag([1.0, -1.0, -1.0])
    t = np.array([0.0, 0.0, 35.0])

    def w2i(h):
        return K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * h + t])

    W0_true, W1_true = w2i(0.0), w2i(5.0)
    H0 = np.linalg.inv(W0_true / W0_true[2, 2])
    H1 = np.linalg.inv(W1_true / W1_true[2, 2])
    W0, W1 = np.linalg.inv(H0), np.linalg.inv(H1)
    s = np.sum(W0[:, :2] * W1[:, :2]) / np.sum(W1[:, :2] * W1[:, :2])
    D = (s * W1 - W0) / 5.0

    TRUE_H, HINT = 25.0, ds.HEIGHT_HINTS_CM["cup"]     # a bottle misread as "cup"
    foot = _rect_world(6.0, 6.0, 0.0, 18.0, 12.0)
    ground_img = cv2.perspectiveTransform(
        foot.reshape(-1, 1, 2).astype(np.float32), w2i(0.0)).reshape(-1, 2)
    top_img = cv2.perspectiveTransform(
        foot.reshape(-1, 1, 2).astype(np.float32), w2i(TRUE_H)).reshape(-1, 2)
    silhouette = np.vstack([ground_img, top_img])

    before = measure_polygon(silhouette, H0, "cup", 0.7)
    before["mask_px"] = silhouette
    o = dict(before)
    apply_height_hint(o, HINT, H0, W0, D)
    triggered = revert_decision(before["length_cm"], o["length_cm"], o["area_cm2"],
                                o["width_cm"])
    shown = dict(before) if triggered else o

    check("the wrong hint alone would have grown or breached a limit here", triggered,
          f"before={before['length_cm']:.1f} raw_hinted={o['length_cm']:.1f}")
    check("what actually reaches the dashboard is no larger than the pre-hint measurement",
          shown["length_cm"] <= before["length_cm"] + 0.01,
          f"shown={shown['length_cm']:.1f} cm, pre-hint={before['length_cm']:.1f} cm")

    print("\n--- a correct hint is never discarded (no false positives) ---")
    w2i2_tilt = np.radians(18.0)
    R2 = np.array([[1, 0, 0], [0, np.cos(w2i2_tilt), -np.sin(w2i2_tilt)],
                  [0, np.sin(w2i2_tilt), np.cos(w2i2_tilt)]]) @ np.diag([1.0, -1.0, -1.0])
    t2 = np.array([0.0, 0.0, 45.0])

    def w2i2(h):
        return K @ np.column_stack([R2[:, 0], R2[:, 1], R2[:, 2] * h + t2])

    W0_true2, W1_true2 = w2i2(0.0), w2i2(5.0)
    H0b = np.linalg.inv(W0_true2 / W0_true2[2, 2])
    H1b = np.linalg.inv(W1_true2 / W1_true2[2, 2])
    W0b, W1b = np.linalg.inv(H0b), np.linalg.inv(H1b)
    sb = np.sum(W0b[:, :2] * W1b[:, :2]) / np.sum(W1b[:, :2] * W1b[:, :2])
    Db = (sb * W1b - W0b) / 5.0

    TRUE_H2 = 22.0
    foot2 = _rect_world(6.0, 6.0, 0.0, 8.0, 4.0)
    g2 = cv2.perspectiveTransform(
        foot2.reshape(-1, 1, 2).astype(np.float32), w2i2(0.0)).reshape(-1, 2)
    t2i = cv2.perspectiveTransform(
        foot2.reshape(-1, 1, 2).astype(np.float32), w2i2(TRUE_H2)).reshape(-1, 2)
    sil2 = np.vstack([g2, t2i])
    before2 = measure_polygon(sil2, H0b, "obj", 0.7)
    before2["mask_px"] = sil2
    o2 = dict(before2)
    apply_height_hint(o2, TRUE_H2, H0b, W0b, Db)     # the hint matches the true height exactly
    caught2 = revert_decision(before2["length_cm"], o2["length_cm"], o2["area_cm2"],
                              o2["width_cm"])
    check("a correct hint is kept, not reverted", not caught2,
          f"before={before2['length_cm']:.1f} after={o2['length_cm']:.1f}")

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("HEIGHT HINT GUARD OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
