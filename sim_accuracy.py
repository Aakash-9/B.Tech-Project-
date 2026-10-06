"""End-to-end accuracy check: render a scene, run the REAL pipeline on it, score the result.

Every number the pipeline produces is compared against the scene's exact ground truth, so
this measures the pipeline itself rather than how convincing a demo looks. Run it after any
change to the vision maths to see immediately whether accuracy moved.
"""
import cv2
import numpy as np

from aruco_workspace import (make_detector, workspace_from_corners, area_size,
                             height_gradient, MARKER_CM)
from ik_solver import inverse_kinematics, L1, L2
from simulator import Camera, Scene, default_scene
from vision_measure import measure_scene, select_target


def calibrate_from_frame(frame):
    """Run the real ArUco workspace calibration on a rendered frame."""
    corners, ids, _ = make_detector().detectMarkers(frame)
    if ids is None or len(ids) < 3:
        return None, None, None, None
    order = np.argsort(ids.flatten())
    H0, centres = workspace_from_corners([corners[i] for i in order], MARKER_CM)
    if H0 is None:
        return None, None, None, None
    W0, D = height_gradient(H0, frame.shape)
    return H0, W0, D, centres


def score_scene(scene, label="scene", verbose=True):
    frame = scene.render()
    H0, W0, D, centres = calibrate_from_frame(frame)
    if H0 is None:
        print(f"[{label}] FAILED: workspace calibration could not be recovered")
        return None

    truth = {t["name"]: t for t in scene.truth()}
    _, measured = measure_scene(scene.detection_result(), H0, W0, D,
                                max_footprint_cm2=1e9, fallback_height_cm=3.0)

    area = area_size(centres)
    if verbose:
        span = scene.marker_centres
        true_area = (span[:, 0].max() - span[:, 0].min(),
                     span[:, 1].max() - span[:, 1].min())
        print(f"\n=== {label} ===")
        print(f"workspace: measured {area[0]:.2f} x {area[1]:.2f} cm | "
              f"true {max(true_area):.2f} x {min(true_area):.2f} cm")
        print(f"{'object':<11}{'L meas':>8}{'L true':>8}{'W meas':>8}{'W true':>8}"
              f"{'H meas':>8}{'H true':>8}{'pos err':>9}")

    rows = []
    for m in measured:
        t = truth.get(m["label"])
        if t is None:
            continue
        pos_err = np.hypot(m["center_cm"][0] - t["center_cm"][0],
                           m["center_cm"][1] - t["center_cm"][1])
        h_shown = m["height_cm"] if m["height_measured"] else float("nan")
        rows.append({
            "name": t["name"],
            "dL": abs(m["length_cm"] - t["length_cm"]),
            "dW": abs(m["width_cm"] - t["width_cm"]),
            "dH": abs(m["height_cm"] - t["height_cm"]) if m["height_measured"] else None,
            "pos": pos_err,
            "shape": m["shape"],
        })
        if verbose:
            hs = f"{h_shown:8.2f}" if m["height_measured"] else "    n/a "
            print(f"{t['name']:<11}{m['length_cm']:8.2f}{t['length_cm']:8.2f}"
                  f"{m['width_cm']:8.2f}{t['width_cm']:8.2f}{hs}{t['height_cm']:8.2f}"
                  f"{pos_err * 10:8.1f}mm")

    if verbose and rows:
        dl = np.mean([r["dL"] for r in rows])
        dw = np.mean([r["dW"] for r in rows])
        hs = [r["dH"] for r in rows if r["dH"] is not None]
        pos = np.mean([r["pos"] for r in rows]) * 10
        print(f"mean error -> length {dl * 10:.1f} mm, width {dw * 10:.1f} mm, "
              f"height {np.mean(hs) * 10:.1f} mm ({len(hs)}/{len(rows)} measured), "
              f"position {pos:.1f} mm")
    return rows


if __name__ == "__main__":
    print("Rendered scenes are fed through the real ArUco calibration, measurement and IK.")

    all_rows = []
    for tilt in (10.0, 18.0, 28.0, 40.0):
        sc = default_scene(camera=Camera(height_cm=50.0, tilt_deg=tilt))
        r = score_scene(sc, label=f"camera tilt {tilt:.0f} deg")
        if r:
            all_rows += r

    # Reachability: the arm sits at the origin, so check each object against its real limits.
    print("\n=== reachability (arm at origin, reach "
          f"{abs(L1 - L2):.1f}-{L1 + L2:.1f} cm) ===")
    sc = default_scene(camera=Camera(height_cm=50.0, tilt_deg=18.0))
    H0, W0, D, _ = calibrate_from_frame(sc.render())
    _, measured = measure_scene(sc.detection_result(), H0, W0, D,
                                max_footprint_cm2=1e9, fallback_height_cm=3.0)
    for m in measured:
        wx, wy = m["center_cm"]
        z = m["height_cm"] / 2
        try:
            b, s, e, w = inverse_kinematics(wx, wy, z)
            print(f"   {m['label']:<11} at ({wx:5.1f},{wy:5.1f}) -> base {b:6.1f} "
                  f"shoulder {s:5.1f} elbow {e:5.1f}")
        except ValueError as exc:
            print(f"   {m['label']:<11} at ({wx:5.1f},{wy:5.1f}) -> {exc}")

    print("\n=== selection queries ===")
    for q in ({"size": "smallest"}, {"size": "biggest"}, {"shape": "square"}):
        pick = select_target(measured, **q)
        if pick is None:
            print(f"   {str(q):<28} -> nothing matched")
        else:
            print(f"   {str(q):<28} -> {pick['label']} "
                  f"({pick['length_cm']:.1f}x{pick['width_cm']:.1f}cm, {pick['shape']})")

    if all_rows:
        hs = [r["dH"] for r in all_rows if r["dH"] is not None]
        print(f"\nOVERALL across {len(all_rows)} object-measurements: "
              f"length {np.mean([r['dL'] for r in all_rows]) * 10:.1f} mm, "
              f"height {np.mean(hs) * 10:.1f} mm, "
              f"position {np.mean([r['pos'] for r in all_rows]) * 10:.1f} mm")
