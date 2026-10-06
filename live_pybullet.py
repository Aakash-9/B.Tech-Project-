"""Live camera -> objects appear in PyBullet -> type a command -> the arm picks that one.

    venv/Scripts/python.exe live_pybullet.py

Two windows open: the camera feed with every detected object measured and colour-named, and
a PyBullet physics window where those same objects appear as 3D blocks at their measured
positions, sizes and colours. Type a command in the terminal, e.g.

    pick the black mobile phone
    pick the smallest square
    grab the red cube
    drop

and the arm finds that specific object among everything on the table, moves to it under
real physics, grips it and lifts it.

Calibration: uses printed ArUco markers if they are in view (best, and the camera can move),
otherwise whatever calibrate_homography.py last saved. Without either, positions are
meaningless — the console says which one is active.
"""
import math
import os
import queue
import threading

import cv2
import numpy as np
from ultralytics import YOLOE

from aruco_workspace import (area_size, height_gradient, make_detector,
                             workspace_from_corners, MARKER_CM)
from ik_solver import L1, L2
from object_tracker import TrailTracker, TRACKER
from pybullet_sim import ArmSim
from vision_measure import (apply_height_hint, describe, filter_inside_workspace,
                            match_command, measure_scene, sample_colour)
import pybullet as p

# What to look for: {short name shown on screen: phrase used to detect it}.
#
# The two are separate on purpose. Detection works better with a descriptive phrase —
# measured on a real dim scene, generic "box" found one box at 0.16 confidence while
# "black product box" found BOTH at 0.67 and 0.25. But a phrase containing a colour word
# then fights the colour actually measured, printing nonsense like "grey black product
# box". So detect with the phrase, display the short name, and let the measured colour be
# the only thing that names a colour.
#
# Keep the list short and distinct: an open-vocabulary detector scores EVERY phrase against
# EVERY object, so near-synonyms compete — listing "cup", "box", "mouse" and "remote"
# together got a dark phone labelled "cup". Edit this to match your own table.
PROMPTS = {
    # Both re-worded after measuring real confusions on live frames (see
    # dashboard_sources.py for the numbers): "water bottle" scored near zero on a dark
    # insulated bottle and lost to "cup" outright; "cell phone" scored 0.26 against
    # "smartphone"'s 0.57 on the same phone.
    "bottle": "tall narrow drinking bottle with a cap",
    "box": "flat rectangular product box",
    "phone": "smartphone",
    "pen stand": "cylindrical pen holder",
    "cup": "drinking cup",
}
MODEL = "yoloe-11s-seg.pt"

# Typical height in cm for each short name above, used ONLY when the geometry could not
# measure a height itself. This matters most for tall things: a 7x7x22 cm bottle measured
# with the 3 cm default reported a 21.8 x 17.0 cm footprint, because the perspective
# correction needs the real height; told 22 cm it reported 7.8 x 7.6 cm. Measure yours
# roughly with a ruler. Leave a name out and it falls back to OBJECT_HEIGHT_CM.
HEIGHT_HINTS_CM = {
    "bottle": 22.0,
    "pen stand": 11.0,
    "cup": 9.0,
    "box": 4.0,
    "phone": 1.0,
}
OBJECT_HEIGHT_CM = 3.0
MAX_FOOTPRINT_CM2 = 400
MAX_SPAN_CM = 25.0
MIN_CONF = 0.30
# 512 is ~1.5x faster on CPU and costs only ~0.3% in measured dimensions, but 640 detects
# noticeably better in dim or low-contrast scenes (bottle 0.40 -> 0.46 on the test scene).
# Detection reliability is the bottleneck here, not measurement precision, so 640 wins.
INFER_SIZE = 640
SYNC_EVERY = 3            # refresh the physics scene every Nth camera frame
LIFT_CM = 6.0             # how far to raise an object after gripping it


def command_reader(q):
    """Read typed commands on a background thread so the camera loop never blocks."""
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            q.put(None)
            return
        q.put(line.strip())


def main():
    aruco = make_detector()
    H_table = W0 = D = centres = None
    live_calib = False
    if os.path.exists("homography_matrix.npy"):
        H_table = np.load("homography_matrix.npy")
        if os.path.exists("plane_calib.npz"):
            with np.load("plane_calib.npz") as c:
                W0, D = c["W0"], c["D"]
        print("Calibration: saved file (stale if the camera has moved since).")
    else:
        print("Calibration: none yet — show 4 ArUco markers, or run calibrate_homography.py.")

    model = YOLOE(MODEL)
    phrases = list(PROMPTS.values())
    alias = {phrase: short for short, phrase in PROMPTS.items()}
    model.set_classes(phrases, model.get_text_pe(phrases))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found (index 0).")
        raise SystemExit(1)

    sim = ArmSim(gui=True)
    sim.draw_reach_circles()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)          # hide the cluttered side panels
    p.resetDebugVisualizerCamera(cameraDistance=0.62, cameraYaw=50, cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0.03])

    trailer = TrailTracker()
    commands = queue.Queue()
    threading.Thread(target=command_reader, args=(commands,), daemon=True).start()

    print(f"\nLooking for: {', '.join(PROMPTS)}")
    print("Type a command and press Enter, e.g. 'pick the black mobile phone'.")
    print("'drop' releases what is held. 'q' in the camera window quits.\n")

    frame_i, objects = 0, []
    reject_seen = set()        # so a recurring rejection is explained once, not every frame
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        frame_i += 1

        corners, ids, _ = aruco.detectMarkers(frame)
        annotated = frame.copy()
        if ids is not None and len(ids) >= 3:
            Hn, cn = workspace_from_corners(
                [corners[i] for i in np.argsort(ids.flatten())], MARKER_CM)
            if Hn is not None:
                H_table, centres = Hn, cn
                W0, D = height_gradient(H_table, frame.shape)
                if not live_calib:
                    a = area_size(centres)
                    print(f"ArUco calibration live: workspace {a[0]:.1f} x {a[1]:.1f} cm")
                live_calib = True
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        if H_table is not None:
            r = model.track(frame, persist=True, tracker=TRACKER, verbose=False,
                            imgsz=INFER_SIZE)[0]
            annotated = r.plot(img=annotated)
            rejected = []
            _, objects = measure_scene(r, H_table, W0, D, MAX_FOOTPRINT_CM2,
                                       OBJECT_HEIGHT_CM, max_span_cm=MAX_SPAN_CM,
                                       min_conf=MIN_CONF, rejected=rejected)
            # A detection visible on the camera but absent from PyBullet is confusing;
            # say why it was dropped instead of letting it vanish silently.
            for name, why in rejected:
                key = (name, why.split()[0])
                if key not in reject_seen:
                    reject_seen.add(key)
                    print(f"   (not a pick candidate: {name} — {why})")
            if live_calib:
                objects = filter_inside_workspace(objects, centres)
            trailer.update(objects)
            objects = trailer.apply_remembered_heights(objects)

            for o in objects:                      # colour is what makes "the black one" work
                bgr, name = sample_colour(frame, o["mask_px"])
                o["colour_bgr"], o["colour_name"] = bgr, name
                # Show the short name, not the detection phrase.
                o["label"] = alias.get(o["label"], o["label"])
                # Geometry could not measure this one's height: use the operator's figure
                # rather than a default that is known to distort tall objects badly.
                hint = HEIGHT_HINTS_CM.get(o["label"])
                if hint is not None and not o["height_measured"]:
                    apply_height_hint(o, hint, H_table, W0, D)

            Wt = np.linalg.inv(H_table)
            for o in objects:
                px = cv2.perspectiveTransform(
                    np.array([[o["center_cm"]]], np.float32), Wt)[0][0]
                cv2.putText(annotated, f"#{o.get('track_id')} {describe(o)}",
                            (int(px[0]) - 70, int(px[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 255, 255), 1)
            if frame_i % SYNC_EVERY == 0 and sim.held is None:
                sim.sync_objects(objects)          # freeze the scene while carrying something
        else:
            cv2.putText(annotated, "No calibration - show ArUco markers", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("camera", annotated)
        if not sim.connected():          # user closed the PyBullet window — quit tidily
            print("PyBullet window closed.")
            break
        p.stepSimulation()

        try:
            cmd = commands.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd is not None:
            low = cmd.lower().strip()
            if low in ("drop", "release"):
                sim.release()
                print("released.\n")
            elif low in ("list", "what do you see", "?"):
                if not objects:
                    print("nothing detected right now.\n")
                else:
                    for o in objects:
                        print(f"   #{o.get('track_id')} {describe(o)}")
                    print()
            elif set_height_command(low):
                pass                      # handled: a height was set for an object type
            elif cmd:
                handle_pick(sim, objects, cmd)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    sim.disconnect()
    print("Exited cleanly.")


def set_height_command(text):
    """Handle 'bottle is 22' / 'set bottle 22cm' — adjust a height hint without restarting.

    A fixed laptop camera cannot be tilted to whatever angle would let the geometry measure
    a tall object's height, so the operator's figure is the reliable route. Being able to
    correct it live beats editing the file and restarting mid-demo.
    """
    words = text.replace("cm", " ").replace("=", " ").split()
    if not words:
        return False
    number = None
    for w in reversed(words):
        try:
            number = float(w)
            break
        except ValueError:
            continue
    if number is None or not (0.2 <= number <= 100.0):
        return False

    name = " ".join(w for w in words
                    if w not in ("set", "is", "height", "tall", "to", str(number))
                    and not w.replace(".", "", 1).isdigit()).strip()
    if not name:
        return False
    known = {k.lower(): k for k in PROMPTS}
    key = known.get(name, name)
    if key not in PROMPTS:
        print(f"'{name}' is not one of: {', '.join(PROMPTS)}\n")
        return True
    HEIGHT_HINTS_CM[key] = number
    print(f"height for '{key}' set to {number:g} cm (used when it cannot be measured).\n")
    return True


def handle_pick(sim, objects, cmd):
    if not objects:
        print("nothing is being detected right now.\n")
        return
    target, why = match_command(objects, cmd)
    if target is None:
        seen = ", ".join(describe(o, with_size=False) for o in objects)
        print(f"{why}\n   currently visible: {seen}\n")
        return

    x, y = target["center_cm"]
    z = target["height_cm"] / 2.0
    d = math.hypot(x, y, z)
    print(f"-> {describe(target)} at ({x:.1f}, {y:.1f})  [{why}]")
    if not (abs(L1 - L2) <= d <= L1 + L2):
        print(f"   but it is out of reach ({d:.1f} cm, arm reaches "
              f"{abs(L1 - L2):.1f}-{L1 + L2:.1f} cm)\n")
        return

    sim.go_to(x, y, z + 5.0, settle_steps=150)      # approach from above, like a real pick
    sim.go_to(x, y, z, settle_steps=150)
    if sim.grasp(target.get("track_id")):
        sim.go_to(x, y, z + LIFT_CM, settle_steps=200)
        print("   gripped and lifted. Type 'drop' to release.\n")
    else:
        print("   could not grip it (object not in the physics scene yet).\n")


if __name__ == "__main__":
    main()
