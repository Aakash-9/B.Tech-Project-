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

## 1. Group Members and Work Distribution

*[Fill in your actual split — here's one way it could be divided, based on how the modules were built:]*

| Name | PRN | Role |
|---|---|---|
| [Name] | [PRN] | Built the detection and tracking side — wiring up YOLOE for open-vocabulary detection, getting BoT-SORT tracking working, and the trail-buffer plus lock/release logic for following a moving object |
| [Name] | [PRN] | Handled the geometric calibration work — the homography math, live ArUco-based workspace detection, the vanishing-point height estimator, and the footprint correction |
| [Name] | [PRN] | Worked on the kinematics — closed-form inverse and forward kinematics for the arm, reach-limit checks, and the 3D visualization |
| [Name] | [PRN] | Owned testing and documentation — building the ground-truth test suite, running the accuracy checks, and putting this report together |

---

## 2. Introduction

Camera-driven perception is becoming the default way robotic pick-and-place systems locate objects, since it lets a low-cost arm respond to things whose position isn't known ahead of time — unlike a hardcoded-coordinate approach. A regular webcam is about as cheap as a sensor gets, but it comes with a real limitation: on its own, it only gives you a flattened, 2D view of the world. It has no way to tell you an object's real size, its height off the table, or how far away it actually is, unless something extra is done to recover that information.

What this project does is build and check, purely in software, everything a 4-DOF robotic arm would need between "camera sees something" and "arm knows where to move." A webcam looks down at a work surface; an object-detection model finds and outlines whatever's sitting there; a set of geometric calibration steps turns those pixel outlines into real centimeters — including height, which a single camera normally can't give you at all — and a closed-form kinematics solver turns that 3D position into the joint angles the arm would need. All of this shows up as a live 3D plot of the arm tracking a real object as it moves.

Right now this is **software only** — no servos, no motor driver, no microcontroller code anywhere in it. The idea is to get the perception and math side working and genuinely verified first, since that's the part most likely to quietly go wrong, before spending time or money wiring up actual hardware.

---

## 3. Problem Statement

Once calibrated, a single camera can locate where an object sits on a flat surface reasonably well — but it has no built-in way of knowing that object's real size or height. A small object close to the lens and a large object farther away can produce an identical-looking image; there's no way to tell them apart from pixels alone. Most low-cost arm projects don't really solve this — they either hardcode a fixed object height and hope it's close enough, or they reach for a depth camera, which usually isn't in a hobby project's budget.

There's a second, unrelated issue too. Most single-camera setups are calibrated once, by clicking a few reference points by hand, and that calibration quietly stops being correct the moment the camera or the work area gets moved — and the system has no way of knowing that's happened.

What this project set out to do: build something that (a) works out an object's real position, size, shape, and height from one ordinary webcam — no depth sensor, no custom training — (b) keeps recalibrating itself continuously off the live feed instead of relying on a one-off manual click, so moving the camera doesn't silently break it, and (c) turns that 3D position into valid joint angles for this specific arm's geometry, correctly refusing a target it physically cannot reach. And, just as importantly, to actually check each of these claims against known-correct answers rather than just eyeballing a demo and calling it done.

---

## 4. Objectives of the Project

1. Get a monocular vision pipeline working that can detect and outline objects on a surface just from a plain-text description — no dataset, no labelling, no training run.
2. Build and verify a homography setup that calibrates itself continuously using printed ArUco markers, measuring the real size of the work area on the fly and surviving the camera being moved between sessions.
3. Work out — and actually test — a way to estimate an object's real height from a single frame, using vanishing-point geometry, without a depth sensor or any learned model.
4. Work out and test a correction for an object's true footprint, since a raw segmentation mask over-measures size at any angle other than straight-down (it picks up visible side faces along with the top).
5. Write closed-form inverse and forward kinematics for this project's specific 2-link, 4-DOF arm, including both the outer reach limit and the often-skipped inner one.
6. Get multi-object tracking working with persistent IDs, so a specific moving object can be picked out and followed through a brief occlusion, using an existing, well-tested tracking algorithm rather than building one from scratch.
7. Check every geometric claim above against synthetic ground truth, not only a live camera demo, and report the real accuracy — including the cases where it's worse, not just the cases where it's perfect.

---

## 5. Literature Review

| Source | Title/System | Relevance to This Project |
|---|---|---|
| Guo, Huang & Yu (2025) | Monocular One-Shot Metric-Depth Alignment for RGB-Based Robot Grasping (MOMA) | This is essentially the paper that names the exact problem this project is built around: a single camera can only give you depth up to an unknown scale and shift unless there's a one-shot calibration step. MOMA solves the full problem with a learned depth model aligned against sparse ground-truth points; this project tackles the narrower height piece with closed-form vanishing-point geometry instead — no learned model, far less compute. |
| Zuo et al. (2019) | CRAVES: Controlling Robotic Arm with a Vision-based Economic System | Shows that a single cheap RGB camera, nothing else, is enough to meaningfully control a hobby-grade arm — backs up the decision here to stay monocular-only rather than reaching for stereo or depth hardware. |
| A Deep Dive into Robot Vision (ACM Computing Surveys, 2024) | Systematic review of robot-vision methods | Confirms that monocular metric reconstruction without a depth sensor is still an open research problem, not something with a settled, textbook answer to just copy. |
| Real-time 3D vision-based robotic grasping system for low-cost industrial production lines (2025) | Low-cost 3D vision-guided grasping | Same point from a different angle — low-cost 3D perception for grasping is still actively being worked on in the literature, which is exactly why this project's own methods needed to be checked carefully rather than assumed correct. |
| Hartley & Zisserman (2004) | Multiple View Geometry in Computer Vision | The standard reference behind the planar homography (DLT) approach used here to map image pixels to real-world coordinates, and behind the vanishing-point geometry the height estimator is built on. |
| Garrido-Jurado et al. (2014) | ArUco: Automatic generation and detection of fiducial markers | The basis for the printed-marker system used here for continuous live calibration, replacing a one-time manual click. |
| Jocher et al. (2023) | Ultralytics YOLOv8 | The underlying architecture behind the detector used here (via its open-vocabulary variant, YOLOE), which is what gives pixel-level object outlines rather than just boxes. |
| Aharon, Orfaig & Bobrovsky (2022) | BoT-SORT: Robust associations multi-pedestrian tracking | The tracking algorithm used here to keep a persistent ID on each object and recover it after a brief occlusion — used off the shelf, not reimplemented. |

*Note: the first four entries were found and read specifically during this project's literature search; the rest are well-known, standard references for the individual techniques used. Double-check exact bibliographic details before final submission — these last four are cited from general knowledge rather than freshly pulled from the publisher record.*

---

## 6. Gap in the Research/Technology/Methodology

- The monocular-depth papers reviewed here (MOMA especially) are aimed at industrial-scale grasping and assume a learned depth model plus GPU inference is available. This project instead checks whether a much simpler, closed-form geometric trick — using the vanishing point that falls out of a two-plane calibration — can get height accurate enough for a hobby-scale 4-DOF arm, with no training at all and running fine on a CPU.
- The low-cost arm projects we came across either hardcode a single pick height or skip height altogether. None of them try to measure it per object, per frame, from a single image the way this project does.
- Calibration in comparable student projects is usually a one-off manual step — click some points, done. This project instead tries continuous, marker-based recalibration, so bumping the camera or moving to a different table doesn't force a redo.
- It's common in low-cost pipelines to just treat a segmentation mask as if it were the object's real footprint. This project actually measured how wrong that assumption gets at an angle (the mask picks up visible side faces, not just the top) and built a correction for it — something the sources reviewed here don't touch.
- Reach-limit checking in comparable projects usually stops at the outer boundary. This project's kinematics also enforces the inner boundary — a 2-link arm with a fully folded elbow genuinely cannot reach arbitrarily close to its own base, and that's a real constraint that tends to get left out.

---

## 7. Description of the Proposed Solution

The solution here is a software pipeline that takes a live camera feed and turns it into a checked, real-world 3D target position plus the arm angles needed to reach it:

- Four printed ArUco markers sit at the corners of the work area. Every single frame, their detected positions are used to recompute the image-to-world homography, the actual measured size of the work area, and — through the plane's vanishing point — the camera's focal length. No manual clicking, and it self-corrects if the camera gets moved mid-session.
- YOLOE, an open-vocabulary segmentation model, finds and outlines objects on the surface from a plain-text description — no training data needed at all.
- Each object's mask is used to work out its real height from that single frame, based on where the scene's vertical vanishing point falls. The estimate only gets accepted if several candidate vertical edges agree with each other; otherwise the system declines to guess.
- The object then gets re-measured on its own estimated height plane, and its true footprint is recovered as the overlap between its outline projected onto the ground and onto its own height — which removes the over-measurement a raw mask causes at any oblique angle.
- Shape — square, rectangle, circle, triangle, long strip — is worked out from pure geometry of the measured outline. No training, no extra model.
- A tracker keeps a persistent ID on every object and remembers its recent path. Once a target is picked (by shape, size, or label), the system keeps following that same ID through a brief occlusion, only letting go of the lock if the object is genuinely gone for a while.
- The chosen target's real (x, y, height) goes into a closed-form inverse-kinematics solver built for this specific 2-link arm, which either returns the joint angles or reports the target is unreachable — too far, or too close to the base.
- The resulting pose gets drawn as a live 3D plot next to the camera feed, so the whole thing can be watched running in real time.

---

## 8. Requirement Analysis

**Functional Requirements**

- Detect and outline objects from a live feed using plain-text descriptions, with no trained or labelled dataset required.
- Keep the image-to-world mapping calibrated continuously from visible ArUco markers, and measure the real size of the work area from their layout rather than assuming a fixed number.
- Estimate each object's real height from a single frame where the geometry allows it, and explicitly decline rather than guess when it doesn't.
- Classify each object's shape from its measured geometry, and support picking a target by shape, by relative size, and/or by detected label.
- Give every tracked object a persistent identity, and keep following a selected moving object through a short occlusion.
- Reject, rather than quietly attempt, any target position outside the arm's minimum or maximum reach.
- Show the resulting arm pose as a live 3D visualization next to the camera feed.

**Non-Functional Requirements**

- Correctness: every geometric measurement — height, footprint, shape, calibration — has to be checked against synthetic ground truth, not just judged by eye.
- Low cost: needs to run on one ordinary webcam and a CPU-only laptop — no depth sensor, no GPU, no paid API or cloud dependency.
- Robustness: a missing calibration file, a camera read failure, an unreachable target, or a frame with nothing detected should never crash the program — each needs a clear message and either graceful continuation or a clean exit.
- Reconfigurability: neither the work area's size nor the camera's position should be hardcoded anywhere — both need to come from the visible markers, live.
- Simulation-first: no hardware actuation code — no servo control, no motor driver, no firmware — belongs in this stage; the software gets validated first, hardware comes later.

---

## 9. Technology Stack

| Technology / Tool | Specific Utilization |
|---|---|
| Python 3.11 | The language everything in this pipeline is written in. |
| OpenCV (`cv2`) | Homography (DLT), ArUco marker detection, geometric work (convex hulls, polygon intersection, contour analysis), and the camera capture/display windows. |
| Ultralytics YOLOE (`yoloe-11s-seg`) | Open-vocabulary detection and segmentation from a plain-text prompt — no custom training involved. |
| BoT-SORT (via `model.track()`) | Multi-object tracking — persistent IDs and short-term occlusion recovery, used as-is rather than rebuilt. |
| NumPy | The matrix and numerical work underneath the homography, vanishing-point, and kinematics math. |
| Matplotlib | Renders the 3D arm view off-screen (to an image buffer, then shown through OpenCV) — needed because an interactive Matplotlib window and an OpenCV window fight over the same event loop on Windows if run together. |
| Printed ArUco markers | The physical reference points the live workspace calibration is built on, replacing a one-time manual click. |

No hardware libraries — servo drivers, microcontroller communication — appear anywhere in this stage, in line with the project staying software-only for now.

---

## 10. Design

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

Under a pinhole camera model, every vertical line in the real world projects to an image line that passes through one fixed point — the vertical vanishing point. If you know an object's base (sitting on the table) and the corresponding point at its true height along that same image line, the calibrated world-to-image mapping at height *h* — `W(h) = W0 + h·D`, where `D` comes from the same two-plane calibration that produces the homography — can be solved directly for *h*. The system runs this for every candidate pair of silhouette points that share a bearing from the vanishing point, and only trusts the result if independent pairs agree with each other; if they don't, it declines rather than returning a number it can't stand behind.

### 10.3 Footprint Correction Geometry

A segmentation mask traces the object's entire visible outline — and at any angle other than dead overhead, that includes part of the sides, not just the top. Projecting that outline onto the ground plane over-measures the footprint (the side spills outward); projecting the same outline onto the plane at the object's own measured height under-measures it (it spills inward instead). Taking the overlap of those two shapes gives the footprint both agree on, and that's what gets reported.

---

## 11. Development / Implementation

### 11.1 Analytical Inverse Kinematics

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

Both limits are checked here — the usual outer boundary (`d > L1 + L2`), and the inner one too (`d < |L1 − L2|`), which is the closest a fully folded 2-link arm can ever physically get to its own base.

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

The object currently being followed gets more patience than any other tracked object — 60 missed frames of grace instead of the usual 15 — enough to ride out a real occlusion. But it isn't unlimited: if the object stays gone past that grace period, the lock is released rather than leaving the system frozen, waiting on something that's genuinely no longer there.

---

## 12. Testing & Debugging

### 12.1 Test Suite

The test suite is built around a virtual camera whose geometry — height above the table, tilt angle, focal length, object dimensions — is fixed exactly by construction. That means every number the pipeline produces can be checked against a known correct answer instead of just looking plausible. There are **39 automated checks** in total, covering forward/inverse kinematics consistency, both reach limits, the two-plane calibration recovery, height estimation across several camera tilts and object sizes, shape classification across five shape types, picking a target by shape/size out of a cluttered scene, the workspace-boundary filter, calibration file round-trips, and the error paths (missing calibration file, a frame with nothing detected).

### 12.2 Test Results — Accuracy Summary

| Measurement | Method | Result |
|---|---|---|
| Forward/inverse kinematics consistency | FK(IK(target)) == target | Exact — error under 1e-9 cm |
| Object height, 6 test boxes, camera tilt 8°–35° | Vanishing-point geometry | 0.00 cm mean error, 0.27 cm worst case |
| Object footprint, raw mask, oblique angle | Silhouette measured directly | Over 100% over-estimate at steep angles |
| Object footprint, same cases, corrected | Ground/top-plane intersection | Down to single-digit-percent error near overhead; meaningfully reduced (though not zero) at extreme angles |
| Live workspace area, 24 × 18 cm true | ArUco marker layout | Within 0.05 cm at every tilt tested (10°–55°) |
| Shape classification, 5 shape types | Geometric rules, no training | 5/5 correct, even under a tilted camera |
| Picking one shape out of 4 mixed objects | Shape + size query | Correct object picked every time it was tested |

### 12.3 Specific Test Cases and Outcomes

| Test Case | Expected Behaviour | Result |
|---|---|---|
| Target beyond L1+L2 | Reject as unreachable | Raises `ValueError` correctly; frame skipped, no crash |
| Target closer than \|L1−L2\| | Reject as too close to the base | Raises `ValueError` correctly, once this check was added |
| No visible side face (camera straight down) | Decline the height estimate | Returns `None`, correctly falls back to a marked approximate height |
| Height candidates disagree | Decline rather than guess | Returns `None` once the spread between candidates exceeds tolerance |
| A table-sized shape in frame | Must not be treated as a pick target | Correctly excluded, once filtered by area **and** longest side together |
| Locked target briefly occluded | Keep following the same ID | Trail and lock both survive a gap shorter than the grace period |
| Locked target genuinely gone | Release the lock, don't freeze | Correctly released once the grace period runs out |
| No calibration file, no markers visible | No crash, clear guidance | Prints guidance and waits for markers instead of crashing |

### 12.4 Debugging Highlights

- The first version of the height estimator had a **base/top point-selection error**: the silhouette point sitting "below" the farthest visible corner is often hidden inside the outline rather than actually being on it, so pairing points by simple distance grabbed the wrong ones and produced height errors of 4–12 cm. Fixed by checking every same-bearing candidate pair and taking the median, with disagreement between pairs causing the estimate to be dropped instead of trusted.
- There was a **projective scale mismatch** in the two-plane calibration — `cv2.findHomography` hands back each plane's homography in its own arbitrary scale, so subtracting the two matrices directly (to get the height gradient) was mathematically meaningless until both were rescaled onto a shared projective scale first.
- Early on, footprint size was measured straight off the raw segmentation mask, which over-measured objects at any non-overhead angle by up to 100%+, since the mask includes visible side faces along with the top. That's what the ground/top-plane intersection fix in Section 10.3 addresses.
- There was a **target-selection bug** where a large, irregularly shaped detection — a table or a mat — could slip under an area-only size filter despite obviously not being something to pick up. Fixed by requiring a candidate to be small by both area *and* longest side.
- The kinematics solver was originally **missing the inner reach limit** — only the outer boundary was checked, so a target very close to the base produced a strange, nearly vertical pose instead of a clean "unreachable" rejection.
- There was a **file-handle leak**: loading the saved calibration file left it open on Windows, which would quietly block re-running calibration later in the same session.
- Tracker IDs were originally attached to measured objects **by list position**, after that list had already been re-sorted and filtered by size — which silently handed IDs to the wrong objects. Fixed by attaching each object's tracker ID at the point of measurement, before any reordering happens.

---

## 13. Project Plan and Progress

| Phase | Planned | Status |
|---|---|---|
| 1 | Kinematics — closed-form IK/FK for this specific 2-link geometry | Done, checked exactly against forward kinematics |
| 2 | Basic vision — single-shot detection, manual homography calibration | Done |
| 3 | Metric measurement — two-plane calibration, height-corrected sizing | Done, checked against synthetic ground truth |
| 4 | Advanced geometry — vanishing-point height, footprint correction, shape classification | Done, checked against synthetic ground truth |
| 5 | Live self-calibration — ArUco-based continuous workspace detection | Done, checked against synthetic ground truth |
| 6 | Tracking — multi-object identity, occlusion-robust lock | Done, unit-tested |
| 7 | Full pipeline on live camera hardware, real-object accuracy check | In progress |
| 8 | Physical arm hardware integration | Not started — deliberately out of scope for now |
| 9 | Documentation and final demonstration | In progress (this report) |

---

## 14. Project Outcome

**Planned Outcomes**

- A pipeline that works out an object's real position, size, shape, and height from one ordinary webcam — no depth sensor, no custom-trained model.
- A live, self-calibrating workspace setup that survives the camera being moved instead of needing a manual redo.
- Kinematics that are exactly self-consistent and correctly bounded by the arm's real reach.
- Every geometric stage checked against synthetic ground truth, with the real accuracy and limitations reported honestly rather than glossed over.

**Progress So Far**

- Done: a complete pipeline covering detection, live calibration, height estimation, footprint correction, shape classification, tracking, and kinematics, with tests behind each piece.
- Done: height estimation checked to 0.00 cm mean error against synthetic ground truth across several camera tilts.
- Done: found and fixed a real measurement error (up to 100%+) caused by treating a segmentation mask as if it were the object's true footprint.
- Done: live workspace calibration measured to within 0.05 cm of the true area across every camera placement tested, with no manual step.
- Still pending: checking all of this against a real physical camera and ruler-measured objects rather than only synthetic ground truth, and the actual hardware integration.

---

## 15. Conclusion

The perception and kinematics side of this vision-guided 4-DOF arm project is functionally done and checked against synthetic ground truth: detection, live self-calibration, single-view height estimation, footprint correction, geometric shape classification, occlusion-robust tracking, and closed-form kinematics have all been built and tested against known-correct answers, not just shown off in a demo. A few real, non-trivial geometry bugs turned up and got fixed along the way — most notably a footprint-measurement error from treating a mask as the true footprint, and a base/top point-mixup in the height estimator.

What's left is (a) checking this all against a real camera and objects measured with an actual ruler, to see whether the synthetic-ground-truth numbers hold up under real lighting and real hardware, and (b) the physical arm itself, which is deliberately outside the scope of this software-validation stage.

---

## 16. References (APA Format)

Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust associations multi-pedestrian tracking*. arXiv. https://arxiv.org/abs/2206.14651

Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. *Pattern Recognition, 47*(6), 2280–2292.

Guo, T., Huang, B., & Yu, J. (2025). *Monocular one-shot metric-depth alignment for RGB-based robot grasping*. arXiv. https://arxiv.org/abs/2506.17110

Hartley, R., & Zisserman, A. (2004). *Multiple view geometry in computer vision* (2nd ed.). Cambridge University Press.

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* [Computer software]. Ultralytics.

[ACM Computing Surveys author(s)]. (2024). A deep dive into robot vision: An integrative systematic literature review. *ACM Computing Surveys*. https://doi.org/10.1145/3648357

[Author(s)]. (2025). Real-time 3D vision-based robotic grasping system for low-cost industrial production lines. https://doi.org/10.1007/s40430-025-06055-1

Zuo, Y., Qiu, W., Xie, L., Zhong, F., Wang, Y., & Yuille, A. L. (2019). *CRAVES: Controlling robotic arm with a vision-based economic system*. arXiv. https://arxiv.org/abs/1812.00725

*Note: verify exact author lists for the ACM Computing Surveys and Springer entries before submission — not fully captured in the earlier search results.*

## APPENDICES

- Similarity Report — [attach separately, per your institution's process]
- AI Plagiarism Report — [attach separately, per your institution's process]
