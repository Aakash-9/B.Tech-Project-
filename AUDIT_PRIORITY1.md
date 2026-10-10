# Priority 1 audit — 10 October 2026

Scope: Step 0 and closed-loop visual servoing only. Priorities 2–4 and the general
dashboard redesign are deferred. No hardware actuation is included.

## Baseline, before code changes

Read README first, then all 27 project-root Python files, the URDF and dashboard
template. Git was clean at `f574f4a`.

| Command (venv Python) | Actual result |
|---|---|
| test_pipeline.py | PASSED 52 / 52 |
| test_full_system.py | PASSED 23 / 23 |
| sim_tracking.py | PASSED 8 / 8 |
| object_tracker.py | OK (15 assert statements, some in loops) |
| pybullet_sim.py | PASS; four targets; worst printed error 0.00 mm |
| vision_measure.py | OK; four assertions |
| test_realtime.py | PASSED 11 / 11 |
| test_height_hint_guard.py | PASSED 10 / 10 |
| test_classification.py | PASSED 4 / 4 |
| test_camera_tilt.py | PASSED 12 / 12 |

The numbered suites total 120 checks. Self-test assertions and four physics cases
are reported separately, rather than invented as additional numbered checks.

Both dashboard entry points started, served `/api/state`, and produced JPEGs from
both video endpoints. Windows sandbox denied localhost sockets and webcam access;
the checks succeeded outside the sandbox. Baseline on port 5011:

```
sim:    camera 57.2 fps, detection 121.1 fps, simulation 14.8 fps
camera: camera 30.4 fps, detection   3.2 fps, simulation 10.8 fps
camera calibration: ok=false (saved calibration, not verified)
cleanup children remaining: 0 (audit explicitly terminated its child processes)
```

These are short samples, not sustained latency benchmarks. Baseline application
shutdown itself has no engine stop path; audit cleanup is not proof that it does.

## Claim verification

- PyBullet's independent link-frame readback reproduces the rounded 0.00 mm IK
  error on the four existing cases. This is not physical arm accuracy.
- Planar mapping is exact in the algebraic tests; rendered markers and silhouettes
  introduce error. No live metric ground truth is available.
- Height sweep: cluttered_scene, fixed scene, heights 50/70/90 and tilt parameters
  22/30/40: **20/54 = 37.0%** measured. The prompt's 34% does not reproduce on
  this explicit grid. At 50/22, 50/30, 50/40 coverage is 2/6, 4/6, 4/6.
- At 50/40, measured height mean error is **1.601 mm**, worst **1.753 mm**;
  the quoted 2.9 mm worst does not reproduce. A 10–30 degree sweep in 2-degree
  increments reproduces **47.765 mm** worst at 28 degrees (blue box).
- Tilt tests pass, but the implementation measures angle **from the table normal**
  (zero is looking down), contrary to its docstring's "below horizontal" claim.
  Camera.height_cm is camera-axis translation, not vertical world height at tilt.
- Classification checks cover only two saved images, one with baked-in overlays.
  They are regression fixtures, not accuracy statistics.
- CLAHE comparison and the claimed GIL speed ratio have no saved reproducible
  benchmark here. Not independently reproduced; no CLAHE or process change made.
- KineDepth's abstract reports 22.1% depth error reduction and 52% downstream
  success increase: https://arxiv.org/abs/2409.19490 . These are that paper's results,
  not this project's; sparse correspondences alone do not guarantee dense depth.
- The generic industrial ±0.4 mm statement lacks an identified experiment and is
  not used as a result or acceptance criterion here.

## Inventory

| Component | What exists / limitation |
|---|---|
| IK, URDF, ArmSim | Working analytical mapping; plotting FK receives target r/z and is not independent verification. Arm/table collisions are disabled. |
| Measurement, ArUco | Working geometry and synthetic tests; footprint intersection is a heuristic. Same-orientation flat markers of known size required. Calibration does not expire after marker loss. |
| Tracker | Trails, votes, medians, retained height; finite dropout tolerance. Synthetic scene IDs follow visible-list order and can change when visibility/order changes. |
| Detector | Working local YOLOE and two fixture checks; no general classification guarantee. |
| Dashboard | Separate capture/perception threads and physics process; sync stops while busy. Sim frame and detection snapshot can race. |
| Worker | Open-loop blocking approach/descend/lift; renderer and cancellation wait for the moves. Physics timestep does not follow wall time. |
| Grasp | Fixed constraint, no proximity requirement, visual fixed fingers; not friction-based grasp validation. |
| Commands | Keyword parser, not a language model; unmatched label words are ignored, so a wrong object may be selected. |
| Tests | Height guard mirrors rather than invokes production branch. Full-system moving test checks tracking then picks a static snapshot. |
| Dependencies | requirements.txt omits Flask, PyBullet and direct YOLOE text/tracking dependencies. |
| Legacy entry points | live_pipeline/live_pybullet/pybullet_demo/sim_demo_interactive and matplotlib demos are separate runnable demos, not dashboard control paths. |
| Dead/stub code | No hardware driver exists. SimSource.close is intentionally empty. Unused imports/locals exist (e.g. pybullet_sim time, simulator.visible_silhouettes cam_c); no wholesale deletion warranted. |

## Priority 1 design boundary

Use rendered RGB fiducials on the simulated target and tool, convert image error
through an image Jacobian to bounded joint velocities, and keep an explicit known
target-height constraint. Two pixel coordinates alone cannot constrain three
translation coordinates. Simulated fiducials and joint readback are privileged
simulation instrumentation; ground truth positions may generate the images and
score results but must not be fed directly into the controller as Cartesian error.

Reference for image features and interaction matrices:
https://faculty.cc.gatech.edu/~seth/res.php?u=vs .

Keep the physics worker process, replace blocking pick with a ticked state machine,
keep target updates during approach, stop on lost/stale observations, bound runtime,
and require convergence before attachment. Verify at least 20 seeded trials with
PyBullet scoring and one convergence plot per trial, including perturbed initial
estimates. Document that this validates simulated feedback, not real-camera metric
accuracy or arbitrary unknown object height.
