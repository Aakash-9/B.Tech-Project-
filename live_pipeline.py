"""Webcam -> open-vocab tracking -> auto workspace calibration -> IK -> live 3D arm plot.

Simulation only. Detection is YOLOE: name the target in plain words below, no training.
Workspace calibration is LIVE from 4 printed ArUco markers (see aruco_workspace.py) —
tape them at the corners of your work area and the system measures that area's real size
every frame, at whatever angle or height the camera happens to be. No sheet-clicking step.
If markers aren't visible this frame, the last good calibration is reused; if none has ever
been seen, it falls back to a manual homography_matrix.npy from calibrate_homography.py.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # cv2 window + an interactive matplotlib window crash together on Windows
import matplotlib.pyplot as plt
from ultralytics import YOLOE

from ik_solver import inverse_kinematics, forward_kinematics, L1, L2
from vision_measure import measure_scene, select_target, filter_inside_workspace, measure_polygon
from object_tracker import TrailTracker, TRACKER
from aruco_workspace import (make_detector, workspace_from_corners, area_size,
                             height_gradient, MARKER_CM)
from arm_render import draw_arm, draw_workspace, set_view

# Name the objects to look for, in plain words. No training, no COCO class list.
PROMPTS = ["small wooden cube", "plastic container", "cell phone", "book"]
MODEL = "yoloe-11s-seg.pt"

# What to pick: shape ("square"/"rectangle"/"circle"/"triangle"/"long strip"), label text,
# and/or relative size ("smallest"/"biggest"). None = don't filter on that axis.
TARGET_SHAPE = None
TARGET_LABEL = None
TARGET_SIZE = None
TRACK_MOVING_TARGET = True   # once a target is picked, keep following that SAME object

OBJECT_HEIGHT_CM = 3.0   # fallback only — height is measured per object when possible
MAX_FOOTPRINT_CM2 = 400  # a pick candidate must be small by area AND longest side, or an
MAX_SPAN_CM = 25.0       # irregular table/mat mask can sneak under an area-only limit
MIN_CONF = 0.35
PLOT_EVERY = 5           # update the 3D plot every Nth frame so it doesn't lag the camera
INFER_SIZE = 512         # 1.5x faster than 640 on CPU for ~0.3% dimension error
# For report screenshots: show the raw (uncorrected) silhouette size next to the corrected
# footprint, so the difference the correction makes is visible in a single frame.
SHOW_FOOTPRINT_COMPARISON = False
TRAIL_LEN = 25

aruco_det = make_detector()
have_live_calib = False
H_table = W0 = D = workspace_centres = None
if os.path.exists("homography_matrix.npy"):
    H_table = np.load("homography_matrix.npy")
    if os.path.exists("plane_calib.npz"):
        with np.load("plane_calib.npz") as c:      # close the npz; open handle blocks recalib
            W0, D = c["W0"], c["D"]
    print("Manual calibration loaded as a fallback (used only while no ArUco markers are seen).")
else:
    print("No homography_matrix.npy and no ArUco markers seen yet — waiting for 4 markers "
          "in view. Print aruco_workspace.py's save_marker_sheet() output and tape it down, "
          "or run calibrate_homography.py for the manual one-time method.")

model = YOLOE(MODEL)
model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
print("Looking for:", ", ".join(PROMPTS))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found (index 0). Check the webcam connection.")
    raise SystemExit(1)

fig = plt.figure(figsize=(6, 5), dpi=100)
ax = fig.add_subplot(projection="3d")
trailer = TrailTracker(max_len=TRAIL_LEN)
surface_area_reported = None

frame_i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed.")
        break

    # --- live workspace calibration: re-detect markers every frame, at any angle/height ---
    corners, ids, _ = aruco_det.detectMarkers(frame)
    annotated = frame.copy()
    if ids is not None and len(ids) >= 3:
        order = np.argsort(ids.flatten())
        H_new, centres_new = workspace_from_corners([corners[i] for i in order], MARKER_CM)
        if H_new is not None:
            H_table, workspace_centres = H_new, centres_new
            W0, D = height_gradient(H_table, frame.shape)
            have_live_calib = True
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            L, Wd = area_size(workspace_centres)
            if surface_area_reported != (round(L, 1), round(Wd, 1)):
                print(f"Workspace measured live: {L:.1f} x {Wd:.1f} cm "
                      f"({len(ids)} markers, focal {'known' if D is not None else 'n/a'})")
                surface_area_reported = (round(L, 1), round(Wd, 1))

    if H_table is None:
        cv2.putText(annotated, "Show 3-4 ArUco markers to calibrate", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # --- detect + track + measure ---
    results = model.track(frame, persist=True, tracker=TRACKER, verbose=False, imgsz=INFER_SIZE)
    r = results[0]
    annotated = r.plot(img=annotated)
    _, objects = measure_scene(r, H_table, W0, D, MAX_FOOTPRINT_CM2, OBJECT_HEIGHT_CM,
                               max_span_cm=MAX_SPAN_CM, min_conf=MIN_CONF)
    if have_live_calib:
        objects = filter_inside_workspace(objects, workspace_centres)   # only pick FROM inside
    trailer.update(objects)
    # Height only measures when an object shows a usable vertical edge, so carry a
    # previously measured height forward on frames where this one's geometry declines.
    objects = trailer.apply_remembered_heights(objects)

    W_table = np.linalg.inv(H_table)
    for obj in objects:
        px = cv2.perspectiveTransform(
            np.array([[obj["center_cm"]]], dtype=np.float32), W_table)[0][0]
        tag = "~" if not obj["height_measured"] else ""
        idtag = f"#{obj['track_id']} " if obj.get("track_id") is not None else ""
        cv2.putText(annotated,
                    f"{idtag}{obj['shape']} {obj['length_cm']:.1f}x{obj['width_cm']:.1f}"
                    f"x{tag}{obj['height_cm']:.1f}cm",
                    (int(px[0]) - 60, int(px[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 2)
        if SHOW_FOOTPRINT_COMPARISON:
            # Raw = the mask measured directly on the table plane (no height correction),
            # exactly what the pipeline used to report before the intersection fix.
            raw = measure_polygon(obj["mask_px"], H_table)
            cv2.putText(annotated,
                        f"raw {raw['length_cm']:.1f}x{raw['width_cm']:.1f}cm | "
                        f"corrected {obj['length_cm']:.1f}x{obj['width_cm']:.1f}cm",
                        (int(px[0]) - 90, int(px[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 140, 255), 2)
    cv2.imshow("detection", annotated)
    frame_i += 1

    # Lock onto one physical object once picked, so a moving target keeps being followed
    # even if a re-classification would otherwise have picked a different-looking one.
    target = trailer.find_locked(objects) if TRACK_MOVING_TARGET else None
    if target is None:
        target = select_target(objects, TARGET_SHAPE, TARGET_LABEL, TARGET_SIZE)
        if target is not None and TRACK_MOVING_TARGET and target.get("track_id") is not None:
            trailer.lock(target["track_id"])

    if frame_i % PLOT_EVERY == 0 and target is not None:
        wx, wy = target["center_cm"]
        grasp_z = target["height_cm"] / 2   # grip the middle of the object's side

        reach, reach_min = L1 + L2, abs(L1 - L2)
        ax.cla()
        draw_workspace(ax, reach, reach_min, z=0.0, workspace_centres=workspace_centres)

        # Every object the camera can see, so the scene reads as a scene, not one dot.
        for o in objects:
            c = o["contour_cm"]
            is_t = o is target
            ax.plot(c[:, 0], c[:, 1], o["height_cm"],
                    "-", color="#e65100" if is_t else "#9e9e9e",
                    lw=2.0 if is_t else 1.0, alpha=1.0 if is_t else 0.6)

        trail = trailer.trail_of(target.get("track_id"))
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], grasp_z, "-", color="purple",
                    lw=1.2, alpha=0.6, label="path")
        ax.scatter([wx], [wy], [grasp_z], color="red", s=70, label="target")

        try:
            base, shoulder, elbow, wrist = inverse_kinematics(wx, wy, grasp_z)
        except ValueError as exc:
            ax.set_title(f"{target['label']} — {exc}", fontsize=10, color="#c62828")
        else:
            pts = forward_kinematics(base, shoulder, float(np.hypot(wx, wy)), grasp_z)
            draw_arm(ax, pts, grasp_deg=target["grasp_deg"],
                     gripper_open_cm=max(2.0, target["width_cm"] + 1.0))
            tag = "" if target["height_measured"] else "~"
            idtag = f"#{target['track_id']} " if target.get("track_id") is not None else ""
            ax.set_title(
                f"{idtag}{target['shape']} ({target['label']}) {target['length_cm']:.1f}x"
                f"{target['width_cm']:.1f}x{tag}{target['height_cm']:.1f}cm "
                f"@ ({wx:.1f}, {wy:.1f})\nbase {base:.0f}  shoulder {shoulder:.0f}  "
                f"elbow {elbow:.0f}  grasp {target['grasp_deg']:.0f}", fontsize=10)
        set_view(ax, reach)
        ax.legend(loc="upper left", fontsize=7)

        fig.canvas.draw()
        plot_img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        cv2.imshow("arm simulation", plot_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")
