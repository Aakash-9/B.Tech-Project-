"""Watchable end-to-end demo, all in simulation: a cluttered table with a moving object.

    venv/Scripts/python.exe sim_demo_interactive.py

Two windows: what the simulated camera sees (with every object measured and colour-named),
and a PyBullet physics view where those same objects appear as 3D blocks and the arm works.
Type commands in the terminal:

    pick the biggest blue box     two blue objects are present on purpose
    pick the smallest blue one
    pick the black cube           this one is moving
    list                          what is detected right now
    drop
    q (in the camera window)      quit

Detection here is the simulator's own exact masks, so this exercises the geometry, colour
naming, tracking, reasoning and kinematics — everything except whether a neural detector
finds your real objects, which only a real camera can answer.
"""
import math
import queue
import threading
import time

import cv2
import numpy as np
import pybullet as p

from ik_solver import L1, L2
from object_tracker import TrailTracker
from pybullet_sim import ArmSim
from sim_accuracy import calibrate_from_frame
from simulator import Camera, cluttered_scene
from vision_measure import describe, match_command, measure_scene, sample_colour

TILT_DEG = 22.0
LIFT_CM = 6.0
SPEED = 0.35          # seconds of scene time per frame, so the moving object visibly drifts


def reader(q):
    while True:
        try:
            q.put(input().strip())
        except (EOFError, KeyboardInterrupt):
            q.put(None)
            return


def annotate(frame, objects, H0, target_id=None):
    img = frame.copy()
    Wt = np.linalg.inv(H0)
    for o in objects:
        px = cv2.perspectiveTransform(
            np.array([[o["center_cm"]]], np.float32), Wt)[0][0]
        x, y = int(px[0]), int(px[1])
        hot = (o.get("track_id") == target_id)
        col = (60, 255, 60) if hot else (0, 235, 255)
        cv2.circle(img, (x, y), 4, col, -1)
        cv2.putText(img, f"#{o.get('track_id')} {describe(o)}", (x - 75, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return img


def pick(sim, objects, cmd):
    target, why = match_command(objects, cmd)
    if target is None:
        seen = ", ".join(describe(o, with_size=False) for o in objects)
        print(f"   {why}\n   on the table: {seen}\n")
        return None
    x, y = target["center_cm"]
    z = target["height_cm"] / 2.0
    d = math.hypot(x, y, z)
    print(f"-> {describe(target)} at ({x:.1f}, {y:.1f})   [{why}]")
    if not (abs(L1 - L2) <= d <= L1 + L2):
        print(f"   out of reach: {d:.1f} cm (arm reaches "
              f"{abs(L1 - L2):.1f}-{L1 + L2:.1f} cm)\n")
        return None
    sim.go_to(x, y, z + 5.0, settle_steps=140)      # approach from above
    sim.go_to(x, y, z, settle_steps=140)
    if not sim.grasp(target.get("track_id")):
        print("   could not grip it\n")
        return None
    sim.go_to(x, y, z + LIFT_CM, settle_steps=200)
    print("   gripped and lifted. 'drop' to release.\n")
    return target.get("track_id")


def main():
    scene = cluttered_scene(camera=Camera(height_cm=50.0, tilt_deg=TILT_DEG), moving=True)
    H0, W0, D, centres = calibrate_from_frame(scene.render())
    if H0 is None:
        print("calibration failed on the simulated scene")
        return

    sim = ArmSim(gui=True)
    sim.draw_reach_circles()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.65, cameraYaw=50, cameraPitch=-38,
                                 cameraTargetPosition=[0, 0, 0.03])

    trailer = TrailTracker()
    commands = queue.Queue()
    threading.Thread(target=reader, args=(commands,), daemon=True).start()

    print("\nOn the table: two BLUE objects (different sizes), red, green, yellow, and a")
    print("BLACK cube that is moving. Try:")
    print("   pick the biggest blue box     |   pick the smallest blue one")
    print("   pick the black cube           |   list        |   drop")
    print("   q in the camera window quits\n")

    held, t, frame_i = None, 0.0, 0
    while True:
        frame_i += 1
        if held is None:                 # freeze the world while carrying something
            t += SPEED
            scene.advance(t)
        frame = scene.render()

        _, objects = measure_scene(scene.detection_result(), H0, W0, D,
                                   max_footprint_cm2=1e9)
        for o in objects:
            bgr, name = sample_colour(frame, o["mask_px"])
            o["colour_bgr"], o["colour_name"] = bgr, name
        trailer.update(objects)
        objects = trailer.apply_remembered_heights(objects)

        if held is None:
            sim.sync_objects(objects)
        cv2.imshow("simulated camera", annotate(frame, objects, H0, held))

        if not sim.connected():
            print("PyBullet window closed.")
            break
        p.stepSimulation()

        try:
            cmd = commands.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd is not None:
            low = cmd.lower()
            if low in ("drop", "release"):
                sim.release()
                held = None
                print("released.\n")
            elif low in ("list", "?"):
                for o in objects:
                    print(f"   #{o.get('track_id')} {describe(o)}")
                print()
            elif cmd:
                got = pick(sim, objects, cmd)
                if got is not None:
                    held = got

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    sim.disconnect()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
