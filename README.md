# Vision-Guided 4-DOF Robotic Arm — Software & Simulation

A local Python prototype connecting a webcam or synthetic scene to object
measurement, typed target selection and a simulated arm in a web dashboard.
There is no hardware-actuation code.

## Current scope

Priority 1 adds closed-loop **simulated RGB feedback**. During a pick, the worker
observes rendered tool/target fiducials, calculates image error, converts it through
an image Jacobian into bounded joint velocities, and repeats through approach,
descent and lift. It requires convergence before creating the grasp constraint.
Moving targets continue to update during approach. Missing or stale observations
pause motion and eventually abort; release interrupts an active pick.

This validates control in the simulated scene. In webcam mode, the simulator
mirrors the webcam's measured objects; its feedback cannot correct errors in the
webcam's real-world calibration, classifications or assumed object heights.

## Run locally

Tested here with Windows and Python 3.11. Run commands from this folder:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe dashboard.py
```

Open http://127.0.0.1:5000 . Try `pick the red cube`, then `drop`.

```powershell
.\venv\Scripts\python.exe dashboard.py --camera
```

Use `--cam-index 1` for a different camera or `--port 5001` if the port is occupied.
Ctrl+C in the server terminal closes its engine and simulation child process.
Closing the browser alone leaves the server running.

Installation needs Git for the YOLOE CLIP dependency. PyBullet may require Visual
Studio C++ Build Tools when a compatible Windows wheel is unavailable. Model
weights are downloaded by Ultralytics on first use and remain gitignored; the
YOLOE text encoder checkpoint alone is roughly 600 MB. Once dependencies and
weights are present, inference is local.

## Webcam calibration and measurement

Print `aruco_markers.png`, or regenerate it:

```powershell
.\venv\Scripts\python.exe -c "from aruco_workspace import save_marker_sheet; save_marker_sheet()"
```

Measure the printed marker's black-square side and set `MARKER_CM` in
`aruco_workspace.py` (default **4 cm**). Place the markers flat, with the same
orientation, around the workspace. Known marker size supplies metric scale.
The mean marker centre defines the synthetic arm origin; a physical arm would
require a verified camera-to-base transform.

Alternatively run `calibrate_homography.py`, following its known-sheet and optional
raised-plane instructions. Saved calibration applies only to that camera setup.
Recalibrate after changing camera position, zoom or resolution. A green ArUco
badge is not an accuracy certificate: calibration currently persists after marker
loss, so a later camera move can invalidate it.

- Planar homography maps a calibrated plane; it does not measure arbitrary depth.
- Height estimation needs suitable vertical silhouette edges. Unmeasured heights
  use explicit defaults or class hints (`HEIGHT_HINTS_CM`) and display `~`.
- Footprint correction is a silhouette-intersection heuristic. Bad masks, height
  assumptions or calibration can distort both dimensions and positions.
- The tilt function measures angle from the table normal (zero is overhead),
  despite older comments/advice describing it as an angle below horizontal.
- YOLOE prompts and colour voting reduce some observed confusions; neither
  guarantees correct classification. Tests cover two captured-image fixtures.
- Typed commands use keyword matching. Unmatched label keywords currently fall
  back to other filters; inspect the selected object and activity log.

## What the feedback experiment proves

`test_visual_servo.py` runs 24 independently seeded trials plus one preserved
previously failing case, with camera variations, 0.15-pixel Gaussian observation
noise, brief dropouts and a target shift in 13 of the 25 trials.
The initial IK estimate has a signed 5 mm error on each axis. Cartesian error is
scored independently against PyBullet's link frame and target body, never passed
to the image controller. CSV traces and a convergence PNG **per trial** are written
to `audit_results/servo_trials/`; the script fails if any final error is >=2 mm or
the controller fails to converge. See `PRIORITY1_RESULTS.md` for measured results.

The observer renders deliberately coloured virtual markers at known offsets
(tool +10 cm, target +18 cm). Those are simulation instrumentation, not automatic
gripper/object feature detection in a webcam image. The controller assumes camera
calibration, target height and joint state are available. Image-point alignment
alone cannot determine 3D separation along the viewing ray. The test therefore
does **not** establish general monocular height accuracy or hardware readiness.

Other limits:

- Fixed constraint attachment approximates a grasp; fingers do not close or
  validate friction/contact. Arm-to-table collisions remain disabled.
- There is no collision-free trajectory planner or guaranteed moving-object
  interception. Out-of-reach approaches and observation loss stop the pick.
- Synthetic semantic labels and IDs are supplied by the scene. Synthetic tests
  assess geometry/control, not neural detector recognition accuracy.
- Camera, perception and physics remain decoupled. PyBullet stays in its own OS
  process. All pipe writes run on a sender thread, outside Flask request paths.
- Legacy standalone demos still use their original open-loop motion. The new
  feedback path is used by the dashboard and its dedicated tests.

## Verification

Use the virtual environment's Python for each script:

```text
test_pipeline.py             test_full_system.py
sim_tracking.py              object_tracker.py
pybullet_sim.py              vision_measure.py
test_realtime.py             test_height_hint_guard.py
test_classification.py       test_camera_tilt.py
test_scene_feedback.py       test_servo_integration.py
test_worker_feedback.py      test_visual_servo.py
test_dashboard_startup.py    (requires localhost and webcam access)
```

For example: `.\venv\Scripts\python.exe test_visual_servo.py`.
`AUDIT_PRIORITY1.md` records the pre-change audit and limitations of older tests.

## Main files

| Files | Purpose |
|---|---|
| `dashboard.py`, `dashboard_engine.py`, `templates/dashboard.html` | HTTP, capture/perception loops and dashboard |
| `dashboard_sources.py`, `simulator.py` | Live detector and synthetic frame sources |
| `vision_measure.py`, `aruco_workspace.py`, `object_tracker.py` | Measurement, calibration, tracking and smoothing |
| `ik_solver.py`, `robot_arm.urdf`, `pybullet_sim.py` | Arm model, kinematics and simulation primitives |
| `visual_servo.py` | RGB observer, image controller, ticked pick sequence |
| `sim_proxy.py`, `sim_worker.py` | Non-blocking transport and isolated physics process |

Priorities 2–4 (kinematic depth, depth-model fallback, hardware adapters), a broad
UI redesign and physical calibration/validation remain separate work. None is
claimed complete by the Priority 1 results.
