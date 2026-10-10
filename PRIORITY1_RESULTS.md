# Priority 1 results — 10 October 2026

Closed-loop control is green within the instrumented simulation described below.
This completes Priority 1, not the full sim-to-real master task. The baseline audit
is in `AUDIT_PRIORITY1.md`. No hardware actuation or Priorities 2–4 were added.

## What changed

- The dashboard worker now corrects image error each control tick through approach,
  descent and lift, instead of making three blocking one-shot IK moves.
- PyBullet remains in a separate process. A sender thread handles pipe writes;
  Flask commands return without waiting for the arm or renderer.
- Target observations continue during approach. Missing/stale observations pause
  motion and then abort. Release interrupts motion; duplicate picks are refused.
- Synthetic object IDs remain stable when another object leaves the view. Each
  synthetic frame owns its matching detection snapshot, avoiding a thread race.
- Synthetic motion now follows elapsed time rather than capture frame count.
- Fixed-grasp constraint frames were corrected so attachment does not teleport
  the body. A proximity check follows image convergence before attachment.
- Dashboard status discloses simulated RGB feedback and the height assumption.
  Application cleanup closes the camera and simulation worker.

## Accuracy experiment

The controller observes RGB centroids of virtual magenta tool and cyan target
markers. The calibrated image Jacobian converts their error to bounded joint
velocities. Known target height and joint state supply the additional vertical
constraint: two pixel coordinates alone cannot determine arbitrary 3D depth.
PyBullet ground truth generates the simulated sensor and scores error; Cartesian
target error is not an input to the controller.

Twenty-four trials have independent deterministic random seeds; the 25th preserves
an actual earlier failure. Targets span 8–16 cm radius and 1.5–4 cm height, with
camera variation, 0.15-pixel observation noise, three missing observations, and
an 8.5 mm target shift during 13 trials. This is a bounded sample, not exhaustive
workspace validation or a statistical reliability guarantee.

| Measurement | Result |
|---|---:|
| Trials converging below 2 mm | 25 / 25 |
| Deliberately biased one-shot IK baseline | 8.660 mm |
| Final feedback error, mean | 0.686 mm |
| Final feedback error, worst | 1.606 mm |
| Final feedback error, best | 0.215 mm |
| Preserved failure, after fix | 1.146 mm |
| Control ticks to convergence | 27–38 |

The baseline uses signed 5 mm errors on each axis (8.660 mm combined). The feedback
run starts 3 cm above that biased estimate; convergence plots therefore start
higher than the baseline comparison. Plot time is simulation time, not measured
wall-clock latency. Every trial has a CSV and PNG in `audit_results/servo_trials/`.
The worst final-error plot is `trial_06.png`; it was visually checked.

During development, a small RGB marker produced false convergence at **2.607 mm**.
Increasing its simulated radius from 3 to 6 mm gave more pixels for centroid
estimation. That exact old trial remains in the suite. A target marker at +6 cm
was also occluded by the forearm in the dashboard cluttered scene; +18 cm clears
that approach. These are disclosed simulation instrumentation choices, not fixes
for ordinary webcam objects. Loaded lifting settled 0.45 mm below the desired
height; a 0.6 mm vertical convergence tolerance accommodates that residual while
retaining the independent strict 2 mm final-error test.

## Full verification: actual terminal results

All **15 scripts** passed. The numbered checks total **170**, including 25 accuracy
trials, two controller setup checks, and five dashboard startup checks. The three
legacy self-tests are reported separately rather than assigning invented counts.

```text
test_pipeline.py              PASSED 52 / 52
test_full_system.py           PASSED 23 / 23
sim_tracking.py               PASSED 8 / 8
object_tracker.py             OK
pybullet_sim.py               worst error: 0.00 mm; PASS
vision_measure.py             OK — planar measurement exact, height correction validated
test_realtime.py              PASSED 11 / 11
test_height_hint_guard.py     PASSED 10 / 10
test_classification.py        PASSED 4 / 4
test_camera_tilt.py           PASSED 12 / 12
test_scene_feedback.py        PASSED 2 / 2 scene feedback checks
test_servo_integration.py     PASSED 11 / 11 integration checks
test_worker_feedback.py       PASSED 5 / 5 worker feedback checks
test_visual_servo.py:
  [PASS] rendered RGB features, bounded correction, missing feature stops
  [PASS] independent analytic pose agrees with URDF frame
  [PASS] trial 25: open-loop 8.660 mm -> feedback 1.146 mm; 37 ticks
  PASSED 25 / 25 accuracy trials; worst 1.606 mm
test_dashboard_startup.py     PASSED 5 / 5 dashboard startup checks
```

Raw logs remain locally in `audit_results/final/`. This directory is gitignored:
plots/logs are reproducible and camera-related artifacts need not be published.
Tests cover moved targets during approach, loaded lift, cancellation, observation
loss, stale input, timeouts, out-of-reach targets, duplicate commands, stable IDs,
constraint attachment, nonblocking transport, and process/thread cleanup.

The HTTP check launched the real `dashboard.py` entry point in both modes, verified
the page and both JPEG streams, and completed `pick the red cube` in simulation.
Actual output:

```text
[PASS] sim: page, source, camera JPEG, simulation JPEG
fps: camera=52.3, detect=36.9, sim=12.5
[PASS] HTTP pick completes; request returned in 6.0 ms
[PASS] sim: owned startup process tree cleaned up
[PASS] camera: page, source, camera JPEG, simulation JPEG
fps: camera=30.3, detect=3.3, sim=10.9
camera calibration: ok=false (saved calibration file)
[PASS] camera: owned startup process tree cleaned up
PASSED 5 / 5 dashboard startup checks
```

Windows sandbox blocks localhost/webcam access, so this check ran with permission
outside the sandbox. Startup probes terminate their own process trees; separate
worker and engine tests exercise application cleanup. A manual Windows Ctrl+C
event and browser visual usability were not independently tested.

## Before / after and limits

| Item | Audited baseline | Final check |
|---|---|---|
| Pick control | Blocking one-shot IK moves | Ticked rendered-RGB corrections |
| Target updates while busy | Suppressed | Continue until attachment |
| Sim capture / perception / rendering FPS | 57.2 / 121.1 / 14.8 | 52.3 / 36.9 / 12.5 |
| Webcam capture / perception / rendering FPS | 30.4 / 3.2 / 10.8 | 30.3 / 3.3 / 10.9 |
| Live metric calibration | Unverified saved file | Still unverified |

FPS values are short startup samples, not controlled sustained benchmarks. The
baseline synthetic perception loop repeatedly processed the same frame; it now
processes each captured snapshot once. The reduced synthetic rate is not directly
comparable detector throughput. No camera speedup is claimed.

Feedback operates on rendered scene markers, with known camera calibration,
target height and simulated joint readback. In webcam mode that scene is built
from webcam estimates; feedback cannot repair an incorrectly measured real object.
The virtual target marker is not a general object/gripper detector. Colour-feature
ambiguity and larger viewpoint/occlusion changes remain outside the tested rig.
The arm uses a fixed constraint for grasping; fingers and friction are not verified,
and arm/table collisions remain disabled. This does not prove hardware precision,
general moving-object interception, or guaranteed classification.

Priorities 2–4 remain deferred as requested: no dense depth model, kinematic depth
self-calibration or hardware adapter. The broad UI redesign is also deferred.
Height coverage remains the audited **37.0% on the specified synthetic grid**, not
90%; live dimensions need fresh verified calibration and physical ground truth.
Classification remains covered by only two image fixtures. Dependency installation
was not repeated in a fresh environment; these checks used the existing venv.
