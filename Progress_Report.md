Project ID : [FILL IN]

# B. Tech. Project
## Monocular Vision-Guided Kinematics Pipeline for a 4-DOF Robotic Arm

### Monthly Progress Report — [Month] 2026

**Under the Guidance of**
[Guide Name]

**Group Members**

| Name | PRN |
|---|---|
| [Name 1] | [PRN] |
| [Name 2] | [PRN] |
| [Name 3] | [PRN] |
| [Name 4] | [PRN] |

DEPARTMENT OF [Your Department]

---

## TABLE OF CONTENTS

| SN | Particulars |
|---|---|
| 1 | Names of all the group members and their work distribution |
| 2 | Introduction |
| 3 | Problem Statement |
| 4 | Objectives of the Project |
| 5 | Literature Review |
| 6 | Gap in the Research/ Technology/ Methodology |
| 7 | Description of the proposed solution |
| 8 | Requirement Analysis |
| 9 | Technology Stack |
| 10 | Design |
| 11 | Development/ Implementation |
| 12 | Testing & Debugging |
| 13 | Project plan and progress |
| 14 | Project Outcome |
| 15 | Conclusion |
| 16 | References (in APA format) |
| | Appendices |

---

## 1. NAMES OF ALL THE GROUP MEMBERS AND THEIR WORK DISTRIBUTION

*[Fill in your actual team split. A suggested division based on the modules that exist, if useful:]*

| Name | PRN | Role |
|---|---|---|
| [Name] | [PRN] | Detection & tracking layer: YOLOE integration, open-vocabulary prompting, BoT-SORT object tracking, trail buffer, target lock/release logic |
| [Name] | [PRN] | Geometric vision layer: homography calibration, ArUco live workspace detection, vanishing-point height estimation, footprint correction |
| [Name] | [PRN] | Kinematics layer: analytical inverse/forward kinematics, reachability limits, 3D visualization |
| [Name] | [PRN] | Testing, validation, and documentation: synthetic ground-truth test suite, accuracy measurement, report |

---

## 2. INTRODUCTION

Robotic pick-and-place systems increasingly rely on camera-based perception rather than fixed, pre-programmed coordinates, since vision allows a low-cost arm to react to objects whose position is not known in advance. A single (monocular) camera is the cheapest such sensor, but on its own it can only recover a 2D projection of the scene — it cannot directly report an object's real-world size, height, or distance from the camera without additional information.

This project builds and validates, entirely in software, the perception-to-motion pipeline for a 4-Degree-of-Freedom (4-DOF) robotic arm: a webcam observes a work surface, an object-detection model locates and segments objects on it, a set of geometric calibration techniques converts that 2D image information into real-world centimeters (including the object's height, which a single camera cannot normally measure), and an analytical inverse-kinematics solver converts that 3D position into the joint angles the arm would need to reach it. The result is displayed as a live, connected-segment 3D simulation of the arm tracking a real object in real time.

This is a **simulation-only** project at the current stage: no servo motors, motor drivers, or microcontroller code are involved. The objective is to build and rigorously validate the perception and kinematics software — the hardest and most error-prone part of a vision-guided arm — before any physical actuation is added.

## 3. PROBLEM STATEMENT

A monocular camera can locate an object's position on a flat surface with reasonable accuracy once calibrated, but it cannot, by itself, measure that object's real-world size or height — a photograph of a small object held close to the camera is visually indistinguishable from a large object held far away. Most low-cost robotic-arm vision projects sidestep this limitation entirely by assuming a fixed, hardcoded object height, or by requiring a depth-sensing camera that a hobby-scale project usually cannot afford.

A second, separate problem is that most single-camera calibration setups are performed once, by manually clicking reference points, and silently become invalid the moment the camera or workspace is moved — with no way for the system to know this has happened.

The challenge addressed by this project: build a software pipeline that (a) measures an object's real-world position, size, shape, and height from a single ordinary webcam, without a depth sensor and without training a custom model, (b) calibrates itself continuously from the live video feed rather than once from a manual click, so it tolerates the camera being repositioned, and (c) converts the resulting 3D position into valid joint angles for a specific 4-DOF arm geometry, refusing to produce an answer when a target is physically unreachable — and to validate every stage of this pipeline against known ground truth, not only visual demonstration.

## 4. OBJECTIVES OF THE PROJECT

1. Build a monocular vision pipeline that detects and segments objects on a work surface using an open-vocabulary detector, requiring no custom training data or labelled dataset.
2. Design and validate a live, self-calibrating homography system, using printed fiducial (ArUco) markers, that measures the real-world size of the work area continuously and tolerates the camera being moved or repositioned between runs.
3. Derive and validate an analytical method for estimating an object's real height from a single image, using the geometry of vanishing points, without a depth sensor or additional training.
4. Derive and validate a correction that recovers an object's true footprint (length and width) from its segmentation mask, removing the systematic over-measurement caused by the object's visible side faces at oblique camera angles.
5. Implement analytical (closed-form) inverse and forward kinematics for the project's specific 2-link, 4-DOF arm geometry, including both the outer and inner reachability limits.
6. Implement multi-object tracking with persistent identities, so a specific moving object can be selected and followed even through brief occlusion, using an established tracking algorithm rather than re-implementing tracking from scratch.
7. Validate every geometric claim above against synthetic data with known ground truth — not only against a live camera demonstration — and report the measured accuracy honestly, including where the method's accuracy degrades.

## 5. LITERATURE REVIEW

| Source | Title/System | Relevance to This Project |
|---|---|---|
| Guo, Huang & Yu (2025) | Monocular One-Shot Metric-Depth Alignment for RGB-Based Robot Grasping (MOMA) | Directly motivates this project's core problem: monocular depth/height is only recoverable up to an unknown scale without a one-shot calibration step. MOMA solves this for full 6-DOF grasping using a learned depth model aligned to sparse ground-truth points; this project solves the narrower height sub-problem using closed-form vanishing-point geometry instead of a learned model, for a much lower-compute, training-free system. |
| Zuo et al. (2019) | CRAVES: Controlling Robotic Arm with a Vision-based Economic System | Establishes that a single low-cost RGB camera, with no additional sensors, is a viable basis for controlling a hobby-grade robotic arm — supporting this project's choice of a monocular-only sensing approach rather than requiring stereo or depth hardware. |
| Deep Dive into Robot Vision (ACM Computing Surveys, 2024) | Systematic literature review of robot vision methods | Confirms that monocular metric reconstruction without a depth sensor remains an active, unresolved area in robot vision research, rather than a solved problem this project could simply look up. |
| Real-time 3D vision-based robotic grasping system for low-cost industrial production lines (2025) | Low-cost 3D vision-guided grasping | Confirms that low-cost 3D perception for grasping is a live, ongoing area of applied research, not a settled technique — supporting the choice to validate this project's methods carefully rather than assume they are standard. |
| Hartley & Zisserman (2004) | Multiple View Geometry in Computer Vision | Standard reference for the planar homography (DLT) method this project uses to convert image pixels to real-world coordinates, and for the vanishing-point geometry this project's height-estimation method is built on. |
| Garrido-Jurado et al. (2014) | ArUco: Automatic generation and detection of fiducial markers | Basis for the printed marker system used for live, continuous workspace calibration in this project, in place of a one-time manual calibration click. |
| Jocher et al. (2023) | Ultralytics YOLOv8 | Basis for the underlying detection/segmentation architecture used in this project (via its open-vocabulary variant, YOLOE), providing pixel-accurate object outlines rather than only bounding boxes. |
| Aharon, Orfaig & Bobrovsky (2022) | BoT-SORT: Robust associations multi-pedestrian tracking | Basis for the multi-object tracker used in this project to assign persistent IDs to moving objects and recover their identity after brief occlusion, used as a library rather than re-implemented. |

*Note: the first four rows above were located and read during this project's own literature search; the remaining rows are standard, widely-cited references for the specific techniques used (planar homography, ArUco markers, YOLO-family detection, BoT-SORT tracking). Please verify exact bibliographic details (page numbers, exact venue) before final submission, as these were cited from general knowledge rather than freshly re-checked against the publisher record.*

## 6. GAP IN THE RESEARCH/ TECHNOLOGY/ METHODOLOGY

- The reviewed monocular-depth literature (MOMA and related work) targets industrial-grade grasping systems and assumes access to a learned monocular depth model plus GPU inference; this project instead evaluates whether a **closed-form geometric method** — using the vanishing point implied by a two-plane calibration — can recover object height accurately enough for a hobby-scale 4-DOF arm, with zero training and CPU-only inference.
- Existing low-cost robotic-arm vision projects that this project's authors are aware of either hardcode a fixed pick height or omit height estimation entirely; none of the reviewed low-cost systems attempt to measure height per-object, per-frame, from a single view.
- Calibration in comparable student and hobby projects is typically a one-time manual step (clicking reference points once). This project instead investigates **continuous, marker-based recalibration**, so the system tolerates the camera being bumped, repositioned, or used on a different table without a separate recalibration step.
- A segmentation mask is commonly treated as equivalent to an object's real footprint in reviewed low-cost pipelines; this project identifies and corrects a specific, quantifiable error this assumption causes at oblique camera angles (the mask includes visible side faces, inflating the measured footprint), which was not addressed in the reviewed sources.
- Reachability checking in comparable low-cost arm projects is generally limited to the maximum-reach boundary; this project's analytical IK also enforces the **minimum-reach boundary** intrinsic to a 2-link arm (a fully-folded elbow cannot reach arbitrarily close to the base), which is a real physical constraint often omitted.

## 7. DESCRIPTION OF THE PROPOSED SOLUTION

The proposed solution is a software-only pipeline (detailed in Section 10) that converts a live camera feed into a validated, real-world 3D target position and the corresponding arm joint angles:

- Four printed ArUco markers are taped at the corners of the work area. Every frame, their detected positions are used to recover the image-to-world homography, the real measured size of the work area, and (via the plane's vanishing point) the camera's focal length — without any manual clicking step, and self-correcting if the camera moves.
- An open-vocabulary segmentation model (YOLOE) detects and outlines objects on the surface, given a plain-text description of what to look for, with no training data required.
- Each detected object's segmentation mask is used to estimate its real height from a single frame, using the geometry of the scene's vertical vanishing point; the estimate is only accepted when multiple candidate vertical edges agree, and is otherwise declined rather than guessed.
- The object is re-measured on its own estimated height plane, and the true footprint is recovered as the geometric intersection of the object's outline projected at ground level and at its own height — removing the over-measurement a raw mask causes at oblique angles.
- Each object's shape (square, rectangle, circle, triangle, long strip) is classified from pure geometric properties of its measured outline — no training, no additional model.
- A multi-object tracker assigns a persistent ID to each object across frames and records its recent path; a specific object, once selected (by shape, size, or label), continues to be followed by that same ID even through a brief occlusion, and the lock is only released if the object is genuinely lost for an extended period.
- The selected target's real-world (x, y, height) position is passed to a closed-form inverse-kinematics solver for the project's specific 2-link arm geometry, which returns the required joint angles or reports that the target is out of reach (too far, or too close to the base).
- The resulting arm pose is rendered as a live, connected-segment 3D plot alongside the camera feed, so the whole pipeline can be observed running in real time.

## 8. REQUIREMENT ANALYSIS

**FUNCTIONAL REQUIREMENTS**

- The system shall detect and segment objects from a live webcam feed using plain-text object descriptions, without requiring a trained/labelled dataset.
- The system shall calibrate the image-to-world mapping continuously from visible ArUco markers, and shall measure the real-world size of the work area from their positions rather than assuming it.
- The system shall estimate each detected object's real height from a single frame when the scene geometry supports it, and shall explicitly decline to produce a height estimate rather than return an unreliable one when it does not.
- The system shall classify each object's shape from its measured geometry, and shall support selecting a target object by shape, by relative size (smallest/biggest), and/or by detected label.
- The system shall assign a persistent identity to each tracked object and shall continue following a selected moving object through a brief occlusion.
- The system shall reject, rather than silently execute, any target position outside the arm's minimum or maximum physical reach.
- The system shall render the resulting arm pose as a live 3D visualization alongside the camera feed.

**NON-FUNCTIONAL REQUIREMENTS**

- Correctness: every geometric measurement claim (height, footprint, shape, calibration) must be validated against synthetic data with known ground truth, not visual inspection alone.
- Low cost: the system must run on a single ordinary webcam and a CPU-only laptop, with no depth sensor, no GPU requirement, and no paid API or cloud dependency.
- Robustness: the system must not crash on a missing calibration file, a camera read failure, an unreachable target, or a frame with no detections — each must be handled with a clear message and continued or clean-exit behaviour.
- Reconfigurability: the work area's real-world size and the camera's position must not be hardcoded; both must be re-derived live from the visible markers.
- Simulation-first: no hardware actuation code (servo control, motor drivers, microcontroller firmware) is included at this stage; the software pipeline is validated in simulation before any hardware integration is scoped.

## 9. TECHNOLOGY STACK

| Technology / Tool | Specific Utilization |
|---|---|
| Python 3.11 | Core implementation language for the entire pipeline. |
| OpenCV (`cv2`) | Homography computation (DLT), ArUco marker detection, geometric operations (convex hull, polygon intersection, contour analysis), webcam capture and display windows. |
| Ultralytics YOLOE (`yoloe-11s-seg`) | Open-vocabulary object detection and instance segmentation from plain-text prompts, with no custom training. |
| BoT-SORT (via `model.track()`) | Multi-object tracking: persistent ID assignment and short-term occlusion recovery, used as a library rather than re-implemented. |
| NumPy | Numerical/matrix operations underlying the homography, vanishing-point, and kinematics math. |
| Matplotlib | Off-screen rendering of the live 3D arm visualization (rendered to an image buffer and displayed via OpenCV, to avoid a GUI event-loop conflict between Matplotlib and OpenCV windows on Windows). |
| ArUco fiducial markers (printed) | Physical reference points for live, continuous workspace calibration — replacing a one-time manual calibration click. |

No hardware libraries (servo drivers, microcontroller communication) are included at this stage, consistent with the project's current simulation-only scope.

## 10. DESIGN

### 10.1 Pipeline / Block Diagram

```
 Webcam frame
      |
      v
 ArUco marker detection (every frame)
      |
      +--> 3-4 markers found? --No--> reuse last known calibration
      |                                  (or wait for markers, if none yet)
      Yes
      |
      v
 Live workspace calibration
   - image->world homography (marker positions -> real cm)
   - work-area size measured from marker layout
   - camera focal length recovered from the same homography
      |
      v
 Open-vocabulary segmentation (YOLOE, text-prompted) + BoT-SORT tracking
      |
      v
 Per-object measurement
   - vanishing-point height estimation (accept only if edges agree)
   - footprint correction (ground-plane / top-plane intersection)
   - shape classification (circularity, rectangular fill, corner count)
      |
      v
 Target selection (by shape / size / label) + trail buffer
   - lock onto one tracked ID once selected
   - re-acquire after brief occlusion; release lock if genuinely lost
      |
      v
 Inverse kinematics (2-link analytical solver)
   - reachable? -> joint angles
   - too far / too close? -> reject, report, skip frame (no crash)
      |
      v
 Forward kinematics -> live 3D plot (arm pose + target + trail)
      |
      v
 Camera feed window + 3D simulation window, both updated live
```

### 10.2 Height Estimation Geometry

A world vertical line projects, under a pinhole camera model, to an image line passing through a single fixed point — the vertical vanishing point. Given the object's base (on the table) and the corresponding point at its true height along the same image line, the calibrated world→image mapping at height *h* — `W(h) = W0 + h·D`, where `D` is derived from the same two-plane calibration used for the homography — can be solved directly for *h*. The system computes this for every candidate pair of silhouette points sharing a bearing from the vanishing point, and only accepts the result if independent candidate pairs agree; otherwise it declines to estimate, rather than returning an unreliable number.

### 10.3 Footprint Correction Geometry

A segmentation mask outlines the object's full visible silhouette, which — at any camera angle other than perfectly overhead — includes part of the object's side faces, not only its top. Unprojecting that silhouette onto the ground plane over-estimates the footprint (it spills outward, following the visible side); unprojecting the same silhouette onto the plane at the object's own measured height under-estimates it (it spills inward). The geometric intersection of the two resulting polygons is the footprint they agree on, and is used as the corrected measurement.

## 11. DEVELOPMENT/ IMPLEMENTATION

### 11.1 Analytical Inverse Kinematics

The arm's shoulder, elbow, and the triangle formed by the two link lengths and the straight-line distance to the target are solved directly using the law of cosines, giving a closed-form (non-iterative) solution:

```python
def inverse_kinematics(x, y, z, L1=L1, L2=L2):
    base = math.atan2(y, x)
    r = math.hypot(x, y)
    d = math.hypot(r, z)

    if d > L1 + L2:
        raise ValueError(f"Target out of reach: distance {d:.1f}cm > {L1+L2:.1f}cm")
    if d < abs(L1 - L2):
        raise ValueError(f"Target too close to the base: {d:.1f}cm < {abs(L1-L2):.1f}cm")

    cos_elbow = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))

    cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    shoulder = math.atan2(z, r) + math.acos(max(-1.0, min(1.0, cos_alpha)))
    wrist = -(shoulder + (elbow - math.pi))

    return tuple(math.degrees(a) for a in (base, shoulder, elbow, wrist))
```

Both the outer reach limit (`d > L1 + L2`) and the inner reach limit (`d < |L1 − L2|`, the minimum distance a fully-folded 2-link arm can reach) are enforced.

### 11.2 Height Estimation

```python
def estimate_height(poly_px, H0, W0, D, angle_tol_deg=0.6, min_edge_px=3.0):
    vp, v = vertical_vanishing_point(D)
    hull = cv2.convexHull(poly_px).reshape(-1, 2)
    if cv2.pointPolygonTest(hull, tuple(vp), False) >= 0:
        return None  # vanishing point inside the outline: no side face visible

    rel = hull - vp
    dist = np.linalg.norm(rel, axis=1)
    ang = np.arctan2(rel[:, 1], rel[:, 0])

    heights = []
    for i in range(len(hull)):
        for j in range(len(hull)):
            if i == j or dist[j] - dist[i] < min_edge_px:
                continue
            if abs(ang[i] - ang[j]) > np.radians(angle_tol_deg):
                continue
            h = _height_from_pair(hull[i], hull[j], H0, W0, v)
            if h is not None and 0.05 < h < 100.0:
                heights.append(h)

    if not heights:
        return None
    med = float(np.median(heights))
    mad = float(np.median(np.abs(np.array(heights) - med)))
    return None if (len(heights) > 1 and mad > 0.15 * med) else med
```

### 11.3 Footprint Correction

```python
def footprint_polygon(poly_px, H0, W0, D, h_cm):
    hull = cv2.convexHull(poly_px).reshape(-1, 1, 2)
    ground = cv2.perspectiveTransform(hull, H0).reshape(-1, 2)
    top = cv2.perspectiveTransform(hull, homography_at_height(W0, D, h_cm)).reshape(-1, 2)
    area, inter = cv2.intersectConvexConvex(
        cv2.convexHull(ground.astype(np.float32)),
        cv2.convexHull(top.astype(np.float32)))
    return inter.reshape(-1, 2) if inter is not None and len(inter) >= 3 else None
```

### 11.4 Target Lock / Release Logic

```python
def update(self, tracked_objects):
    seen = set()
    for o in tracked_objects:
        tid = o.get("track_id")
        if tid is None:
            continue
        self.trails[tid].append(o["center_cm"])
        self.missed[tid] = 0
        seen.add(tid)
    for tid in list(self.trails):
        if tid in seen:
            continue
        self.missed[tid] += 1
        limit = self.locked_grace if tid == self.locked_id else self.max_missed
        if self.missed[tid] > limit:
            del self.trails[tid]
            del self.missed[tid]
            if tid == self.locked_id:
                self.locked_id = None  # genuinely lost -- release the lock
```

A locked (actively followed) target is given more patience (`locked_grace`, default 60 frames) than an ordinary tracked object (`max_missed`, default 15 frames) before its ID is dropped — enough to survive a real, brief occlusion — but the lock is still released if the object is absent for an extended period, so the system does not freeze indefinitely on an object that is genuinely gone.

## 12. TESTING & DEBUGGING

### 12.1 Test Suite

A synthetic test suite was built around a virtual camera with known, exact ground-truth geometry — camera height, tilt angle, focal length, and object dimensions were all fixed by construction, so every measurement the pipeline produced could be compared against an exact correct answer, not just visual plausibility. **39 automated checks** cover: forward/inverse kinematics consistency, both reach limits, two-plane calibration recovery, height estimation across multiple camera tilts and object sizes, shape classification for five shape types, target selection by shape/size in a cluttered scene, workspace-boundary filtering, calibration file round-trips, and error-path behaviour (missing calibration file, no detections in a frame).

### 12.2 Test Results — Accuracy Summary

| Measurement | Method | Result |
|---|---|---|
| Forward/inverse kinematics consistency | FK(IK(target)) == target | Exact (error < 1e-9 cm) |
| Object height, 6 test boxes, camera tilt 8°–35° | Vanishing-point geometry | 0.00 cm mean error, 0.27 cm worst case |
| Object footprint (uncorrected mask vs. true size), oblique angle | Silhouette measured directly | Up to 100%+ over-estimate at steep angles |
| Object footprint, same cases, after correction | Ground/top-plane intersection | Reduced to single-digit-percent error near overhead, and substantially reduced (though not eliminated) at extreme angles |
| Live workspace area (24 × 18 cm true) | ArUco marker layout | Measured to within 0.05 cm at every tested tilt (10°–55°) |
| Shape classification, 5 shape types | Geometric rules (no training) | 5/5 correct, including under a tilted camera |
| Selecting a specific shape from 4 mixed objects | Shape + size query | Correct object selected in all cases tested |

### 12.3 Specific Test Cases and Outcomes

| Test Case | Expected Behaviour | Result |
|---|---|---|
| Target at distance > L1+L2 | Reject as unreachable | Correctly raises `ValueError`, frame skipped, no crash |
| Target at distance < \|L1−L2\| | Reject as too close to base | Correctly raises `ValueError` after the inner-limit check was added (see 12.4) |
| Object with no visible side face (camera directly overhead) | Decline to estimate height | Correctly returns `None`; pipeline falls back to a marked, approximate height |
| Height estimate candidates disagree (unreliable geometry) | Decline rather than guess | Correctly returns `None` when candidate pairs' spread exceeds tolerance |
| Table-sized detection in frame | Must not be treated as a pick target | Correctly rejected once filtered by both area **and** longest side (see 12.4) |
| Locked target briefly occluded | Continue following the same ID | Trail and lock survive a gap shorter than the grace period |
| Locked target genuinely gone (left frame permanently) | Release the lock, do not freeze | Correctly releases after the grace period expires |
| No `homography_matrix.npy` and no ArUco markers visible | No crash; clear guidance to the user | Correctly prints guidance and continues waiting for markers rather than crashing |

### 12.4 Debugging Highlights

- Found and fixed a **base/top point-selection error** in the initial height-estimation implementation: the silhouette point directly "below" the farthest visible corner is often hidden inside the outline, not on it, so pairing by simple distance picked the wrong points and produced height errors of 4–12 cm. Fixed by testing every same-bearing candidate pair and taking the median, with disagreement causing the estimate to be declined.
- Found and fixed a **projective scale mismatch** in the two-plane calibration: `cv2.findHomography` returns each plane's homography in its own arbitrary scale, so directly subtracting the two planes' matrices (to derive the height gradient) was mathematically invalid until the two matrices were first rescaled to a common projective scale.
- Found and fixed a **silhouette-vs-footprint confusion**: initial footprint measurements used the raw segmentation mask directly, which measurably over-estimated object size at any non-overhead camera angle (up to 100%+ at steep angles) because the mask includes visible side faces; corrected via the ground/top-plane intersection method described in Section 10.3.
- Found and fixed a **target-selection bug**: an irregularly-shaped, large detection (e.g. a table or mat) could have a measured area below an area-only size filter despite clearly not being a pick candidate; fixed by requiring a candidate to be small by **both** area and longest side.
- Found and fixed a **missing inner reach limit** in the kinematics solver: only the maximum-reach boundary was originally checked, so a target very close to the arm's base produced a physically degenerate, near-vertical pose instead of a clear "unreachable" rejection.
- Found and fixed a **stale file-handle issue**: loading the saved two-plane calibration file left it open on Windows, which would silently block re-running calibration later in the same session.
- Found and fixed a **tracker-ID misalignment**: an early implementation attached tracker IDs to measured objects by list position, after the object list had already been re-sorted and filtered by size — silently assigning IDs to the wrong objects. Fixed by attaching each object's tracker ID at the point of measurement, before any reordering.

## 13. PROJECT PLAN AND PROGRESS

| Phase | Planned | Status |
|---|---|---|
| 1 | Kinematics: analytical IK/FK for the specific 2-link arm geometry | Complete, validated exactly against forward kinematics |
| 2 | Basic vision: single-shot detection, manual homography calibration | Complete |
| 3 | Metric measurement: two-plane calibration, height-corrected sizing | Complete, validated against synthetic ground truth |
| 4 | Advanced geometry: vanishing-point height estimation, footprint correction, shape classification | Complete, validated against synthetic ground truth |
| 5 | Live self-calibration: ArUco-based continuous workspace detection | Complete, validated against synthetic ground truth |
| 6 | Tracking: multi-object identity, occlusion-robust target lock | Complete, unit-tested |
| 7 | Full pipeline integration on live camera hardware, real-object accuracy validation | In progress |
| 8 | Physical arm hardware integration (servo actuation) | Not started — explicitly out of scope for the current simulation-only stage |
| 9 | Documentation and final demonstration | In progress (this report) |

## 14. PROJECT OUTCOME

**PLANNED OUTCOMES**

- A software pipeline that measures an object's real-world position, size, shape, and height from a single ordinary webcam, with no depth sensor and no custom-trained model.
- A live, self-calibrating workspace system that tolerates the camera being moved, rather than requiring a one-time manual setup.
- An analytical kinematics solver validated to be exactly self-consistent and correctly bounded by the arm's real physical reach.
- A pipeline validated against synthetic ground truth at every geometric stage, with honestly reported accuracy and limitations, not only a visual demonstration.

**PROGRESS SO FAR**

- Achieved: a complete, unit- and geometry-tested software pipeline covering detection, live calibration, height estimation, footprint correction, shape classification, tracking, and kinematics.
- Achieved: height estimation validated to 0.00 cm mean error against synthetic ground truth across a range of camera tilts.
- Achieved: identified and corrected a significant (up to 100%+) systematic measurement error caused by treating a segmentation mask as an object's true footprint.
- Achieved: live workspace calibration measured to within 0.05 cm of true area across all tested camera placements, with no manual calibration step.
- Pending: full end-to-end validation against a real physical camera and ruler-measured objects (rather than only synthetic ground truth), and physical hardware integration.

## 15. CONCLUSION

The perception and kinematics software for this vision-guided 4-DOF robotic arm project is functionally complete and validated at the level of synthetic ground truth: object detection, live camera self-calibration, single-view height estimation, footprint correction, geometric shape classification, occlusion-robust tracking, and analytical inverse kinematics have each been implemented and independently tested against known-correct answers, not only demonstrated visually. Several non-trivial geometric errors were found and corrected during this process, most notably a systematic footprint over-measurement caused by treating a segmentation silhouette as an object's true footprint, and a base/top point-selection error in the height-estimation method.

The remaining work is (a) validating the pipeline's accuracy against a real camera and ruler-measured physical objects, to confirm the synthetic-ground-truth results transfer to real hardware and lighting conditions, and (b) physical arm actuation, which remains explicitly out of scope for the current software-validation stage.

## 16. REFERENCES (IN APA FORMAT)

Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust associations multi-pedestrian tracking*. arXiv. https://arxiv.org/abs/2206.14651

Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. *Pattern Recognition, 47*(6), 2280–2292.

Guo, T., Huang, B., & Yu, J. (2025). *Monocular one-shot metric-depth alignment for RGB-based robot grasping*. arXiv. https://arxiv.org/abs/2506.17110

Hartley, R., & Zisserman, A. (2004). *Multiple view geometry in computer vision* (2nd ed.). Cambridge University Press.

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* [Computer software]. Ultralytics.

[ACM Computing Surveys author(s)]. (2024). A deep dive into robot vision: An integrative systematic literature review. *ACM Computing Surveys*. https://doi.org/10.1145/3648357

[Author(s)]. (2025). Real-time 3D vision-based robotic grasping system for low-cost industrial production lines. https://doi.org/10.1007/s40430-025-06055-1

Zuo, Y., Qiu, W., Xie, L., Zhong, F., Wang, Y., & Yuille, A. L. (2019). *CRAVES: Controlling robotic arm with a vision-based economic system*. arXiv. https://arxiv.org/abs/1812.00725

*Note: author lists and exact bibliographic detail for the ACM Computing Surveys and Springer entries should be verified against the publisher record before final submission — they were not fully captured in the search results referenced during this project's literature review.*

## APPENDICES

- Similarity Report — [attach separately, per your institution's process]
- AI Plagiarism Report — [attach separately, per your institution's process]
