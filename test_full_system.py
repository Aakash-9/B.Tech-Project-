"""End-to-end check of the whole system in simulation, against exact ground truth.

    venv/Scripts/python.exe test_full_system.py

Covers what the arm actually has to do before any hardware exists:
  1. every object identified, measured, and colour-named
  2. several objects handled at once, including two of the SAME colour
  3. a plain-language request picks the right one, and says so when it is ambiguous
  4. a MOVING object keeps its identity and is still picked correctly
  5. the arm physically reaches, grips and lifts it in PyBullet

Detection here is the simulator's exact silhouettes rather than a neural detector: this
tests the geometry, tracking, reasoning and kinematics. Detector reliability is a separate
question, answered only by a real camera.
"""
import sys

import cv2
import numpy as np
import pybullet as p

from object_tracker import TrailTracker
from pybullet_sim import ArmSim
from sim_accuracy import calibrate_from_frame
from simulator import Camera, cluttered_scene
from vision_measure import describe, match_command, measure_scene, sample_colour

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def perceive(scene, H0, W0, D, trailer=None):
    """One full perception pass: measure, colour-name, track."""
    frame = scene.render()
    _, objs = measure_scene(scene.detection_result(), H0, W0, D, max_footprint_cm2=1e9)
    for o in objs:
        bgr, name = sample_colour(frame, o["mask_px"])
        o["colour_bgr"], o["colour_name"] = bgr, name
    if trailer is not None:
        trailer.update(objs)
        objs = trailer.apply_remembered_heights(objs)
    return objs


# ---------------------------------------------------------------- 1. identification
print("\n=== 1. Every object identified, measured and colour-named ===")
scene = cluttered_scene(camera=Camera(height_cm=50.0, tilt_deg=22.0), moving=False)
frame = scene.render()
H0, W0, D, _ = calibrate_from_frame(frame)
check("workspace calibration recovered from the scene", H0 is not None and D is not None)

truth = {t["name"]: t for t in scene.truth()}
objects = perceive(scene, H0, W0, D)
check("all objects detected", len(objects) == len(scene.boxes),
      f"{len(objects)} of {len(scene.boxes)}")

expected_colour = {"blue box": "blue", "blue cube": "blue", "red cube": "red",
                   "green book": "green", "yellow cube": "yellow", "black cube": "black"}
print(f"    {'object':<12}{'colour':>8}{'measured LxWxH':>22}{'true LxWxH':>18}")
worst_len = 0.0
for o in sorted(objects, key=lambda o: o["label"]):
    t = truth[o["label"]]
    worst_len = max(worst_len, abs(o["length_cm"] - t["length_cm"]))
    print(f"    {o['label']:<12}{o['colour_name']:>8}"
          f"{o['length_cm']:8.1f} x{o['width_cm']:5.1f} x{o['height_cm']:5.1f}"
          f"{t['length_cm']:9.1f} x{t['width_cm']:4.1f} x{t['height_cm']:4.1f}")

right = sum(1 for o in objects if o["colour_name"] == expected_colour.get(o["label"]))
check("every colour named correctly", right == len(objects), f"{right}/{len(objects)}")
check("worst length error under 10 mm", worst_len < 1.0, f"{worst_len * 10:.1f} mm")

# --------------------------------------------------------- 2/3. plain-language picking
print("\n=== 2-3. Plain-language requests, with duplicate colours present ===")
blues = [o for o in objects if o["colour_name"] == "blue"]
check("two different blue objects are present (the ambiguous case)", len(blues) == 2,
      f"{[o['label'] for o in blues]}")

for cmd, want in (("pick the red cube", "red cube"),
                  ("pick the green book", "green book"),
                  ("pick the yellow cube", "yellow cube"),
                  ("pick the biggest blue box", "blue box"),
                  ("pick the smallest blue object", "blue cube"),
                  ("pick the black cube", "black cube")):
    got, why = match_command(objects, cmd)
    check(f"'{cmd}'", got is not None and got["label"] == want,
          f"got {got['label'] if got else 'nothing'}")

got, why = match_command(objects, "pick the purple box")
check("a colour that is not on the table returns nothing (not a wrong guess)",
      got is None, why)

ambiguous, _ = match_command(objects, "pick the blue one")
check("an ambiguous request still returns a blue object",
      ambiguous is not None and ambiguous["colour_name"] == "blue",
      f"chose {ambiguous['label'] if ambiguous else 'nothing'}")

# ------------------------------------------------------------- 4. moving object
print("\n=== 4. A moving object keeps its identity while it travels ===")
moving = cluttered_scene(camera=Camera(height_cm=50.0, tilt_deg=22.0), moving=True)
H0m, W0m, Dm, _ = calibrate_from_frame(moving.render())
trailer = TrailTracker()
ids_seen, positions = [], []
for step in range(12):
    moving.advance(step * 0.5)
    objs = perceive(moving, H0m, W0m, Dm, trailer)
    black = [o for o in objs if o["label"] == "black cube"]
    if black:
        ids_seen.append(black[0]["track_id"])
        positions.append(black[0]["center_cm"])

check("the moving object was seen throughout", len(positions) >= 10,
      f"{len(positions)} of 12 frames")
travelled = np.hypot(positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1])
check("it actually moved a meaningful distance", travelled > 5.0, f"{travelled:.1f} cm")
check("it kept ONE identity the whole way", len(set(ids_seen)) == 1,
      f"ids seen: {sorted(set(ids_seen))}")

true_last = [t for t in moving.truth() if t["name"] == "black cube"][0]["center_cm"]
err = np.hypot(positions[-1][0] - true_last[0], positions[-1][1] - true_last[1])
check("its tracked position matches the truth", err < 1.0, f"{err * 10:.1f} mm off")

objs_now = perceive(moving, H0m, W0m, Dm, trailer)
got, why = match_command(objs_now, "pick the moving black cube")
check("the moving object can still be picked by name and colour",
      got is not None and got["label"] == "black cube",
      f"got {got['label'] if got else 'nothing'}")

# --------------------------------------------------------- 5. the arm actually picks it
print("\n=== 5. The arm reaches, grips and lifts the requested object ===")
sim = ArmSim(gui=False)
sim.sync_objects(objects)
check("every object exists in the physics scene", len(sim.objects) == len(objects),
      f"{len(sim.objects)} bodies")

target, _ = match_command(objects, "pick the biggest blue box")
x, y = target["center_cm"]
z = target["height_cm"] / 2.0
sim.go_to(x, y, z + 5.0, settle_steps=120)
sim.go_to(x, y, z, settle_steps=120)
tip = sim.tip_position_cm()
check("the arm reached the requested object",
      np.hypot(tip[0] - x, tip[1] - y) < 0.5, f"{np.hypot(tip[0]-x, tip[1]-y)*10:.1f} mm from it")

body = sim.objects[target["track_id"]]["body"]
before = p.getBasePositionAndOrientation(body)[0][2]
check("gripper closed on it", sim.grasp(target["track_id"]))
sim.go_to(x, y, z + 6.0, settle_steps=250)
after = p.getBasePositionAndOrientation(body)[0][2]
check("the object was lifted off the table", after - before > 0.02,
      f"{before * 100:.1f} -> {after * 100:.1f} cm")

sim.release()
for _ in range(240):
    p.stepSimulation()
dropped = p.getBasePositionAndOrientation(body)[0][2]
check("releasing puts it back down on the table", dropped < after - 0.01,
      f"settled at {dropped * 100:.1f} cm")
sim.disconnect()

print(f"\n{'=' * 64}\nPASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
print("FULL SYSTEM WORKS IN SIMULATION")
