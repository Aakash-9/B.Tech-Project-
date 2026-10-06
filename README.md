# Vision-Guided 4-DOF Robotic Arm — Software & Simulation

A camera-to-decision pipeline: detect objects from a webcam, measure their real-world
size and position in centimetres, understand a natural-language pick command, plan the
arm's motion, and simulate the physics — all validated in software before any hardware
is built.

## What this is

- **Perception**: open-vocabulary instance segmentation (YOLOE) + multi-object tracking
  (BoT-SORT) with persistent IDs that survive occlusion.
- **Measurement**: homography-based pixel→cm conversion, single-view height estimation
  via vanishing-point geometry, footprint correction for tall objects, and
  self-calibrating workspace detection via ArUco markers (works at any camera
  placement, no fixed rig).
- **Reasoning**: natural-language target selection ("pick the biggest blue box") that
  disambiguates between multiple objects of the same colour/shape.
- **Motion**: exact analytical inverse kinematics for the 2-link arm, verified to
  0.00 mm against PyBullet's own forward kinematics.
- **Simulation**: full PyBullet physics — real gravity, collision, and grasp
  constraints — the arm actually lifts simulated objects.
- **Dashboard**: a live Flask web app with two real-time video panels (camera +
  physics), a command box, and a measured-objects table.

## Status

Software and simulation phase. Hardware (servos, PCA9685, physical build) has not
started — the explicit priority was to prove the full pipeline works in simulation
first. See `Progress_Report_v2.md` for the detailed write-up, including what's been
solved and what's still open (shallow-camera-angle height estimation is a known
geometric limitation, not a bug — see the comments in `vision_measure.py`).

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Model weights (`yoloe-11s-seg.pt` and friends) are **not** in this repo — they're
large pretrained checkpoints that `ultralytics` downloads automatically on first run.

```bash
python dashboard.py              # simulated scene (ground truth known, accuracy provable)
python dashboard.py --camera     # live webcam
```

Then open `http://127.0.0.1:5000`.

## Tests

```bash
python test_pipeline.py          # 52 checks — core measurement math
python test_full_system.py       # 23 checks — end-to-end pipeline
python sim_tracking.py           # 8 checks — occlusion-robust tracking
python object_tracker.py         # tracker self-test (label/size stability)
python pybullet_sim.py           # IK vs. physics ground truth (0.00 mm)
python vision_measure.py         # homography/height-correction self-test
python test_realtime.py          # 11 checks — real-time architecture
python test_height_hint_guard.py # 10 checks — misclassification containment
python test_classification.py    # 4 checks — detection prompts, on real captured frames
```

## Layout

| File | What it does |
|---|---|
| `vision_measure.py` | Core measurement math: homography, height, footprint correction |
| `aruco_workspace.py` | Self-calibrating workspace from ArUco markers |
| `object_tracker.py` | Multi-frame tracking + temporal smoothing (label/size/height) |
| `ik_solver.py` | Analytical inverse kinematics |
| `pybullet_sim.py` / `sim_worker.py` / `sim_proxy.py` | Physics simulation, run in its own process |
| `simulator.py` | Synthetic test scene with known ground truth |
| `dashboard.py` / `dashboard_engine.py` / `dashboard_sources.py` | Live web dashboard |
| `robot_arm.urdf` | 4-DOF arm model for PyBullet |
| `test_*.py` | Automated test suites (116+ checks total) |
