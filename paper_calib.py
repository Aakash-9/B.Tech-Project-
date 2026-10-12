"""Calibrate from a plain sheet of paper, automatically.

Why this exists: the ArUco path needs a printer, and calibrate_homography.py needs someone
to click four corners in a GUI window. Both are real barriers, and the cost of skipping
calibration is not "slightly worse numbers" — it is a stale homography that silently turns
a 7 cm bottle into "73 x 6 cm" while the dashboard reports it with a straight face. So
offer the lowest-friction option that still carries true metric scale: an ordinary A4 sheet
is 21.0 x 29.7 cm everywhere in the world, to a fraction of a millimetre, and everyone has
one. Put it on the table, and this finds its corners without being told where they are.

The sheet gives the SAME thing four ArUco markers give — four image points whose real-world
separation is known — so everything downstream (homography, height gradient, measurement)
is unchanged. This is a new way to obtain the calibration, not a new kind of calibration.

Limitation worth stating: one flat sheet calibrates the table plane and, via
height_gradient's focal estimate, the height mapping. It cannot see the arm, so it does not
establish where the camera is relative to the robot base; that still needs the arm itself
(see arm_driver / the kinematic-probe route).
"""
import cv2
import numpy as np

from aruco_workspace import height_gradient

# A4 short and long side in cm. Letter paper is 21.6 x 27.9 — set SHEET_CM to that if the
# sheet is US Letter, or the whole scale will be off by a few percent.
SHEET_CM = (21.0, 29.7)

# A sheet has to be a decent fraction of the frame to calibrate well: corner noise is
# amplified by the ratio of workspace size to reference size, which is the same reason
# single-marker ArUco calibration was abandoned earlier in this project.
MIN_AREA_FRAC = 0.02
MAX_AREA_FRAC = 0.90


def order_corners(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Sorting by angle about the centroid is what makes this robust to how the sheet happens
    to be rotated on the table; naming the corners by raw x/y comparisons breaks as soon as
    the sheet is turned past 45 degrees.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(ang)]
    # Rotate so the point closest to the top-left of the image comes first.
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def find_sheet(frame_bgr, min_area_frac=MIN_AREA_FRAC, max_area_frac=MAX_AREA_FRAC):
    """Four ordered corners of the largest bright quadrilateral, or None.

    Otsu rather than a fixed threshold, because the whole point is to work on whatever
    lighting the room happens to have — the same dim, desaturated scene that defeats fixed
    colour thresholds elsewhere in this project.
    """
    if frame_bgr is None:
        return None
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if not (min_area_frac * frame_area <= area <= max_area_frac * frame_area):
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # A sheet seen from any sane angle still fills most of its own bounding quad;
        # this rejects L-shaped highlights and paper that is folded or partly covered.
        if area < 0.80 * abs(cv2.contourArea(cv2.convexHull(approx))):
            continue
        best = approx.reshape(-1, 2).astype(np.float64)
        break
    return None if best is None else order_corners(best)


def sheet_world_points(corners_px, sheet_cm=SHEET_CM):
    """World coordinates (cm) for the ordered corners, centred on the sheet.

    Which physical side is 21 cm and which is 29.7 cm cannot be read off a single
    quadrilateral, so decide it by comparing the two pairs of opposite edges in pixels and
    giving the longer physical side to the longer pair. Getting this backwards does not
    fail loudly — it produces a confident calibration that is wrong by the aspect ratio —
    so the caller gets the decision back and can check it against a known distance.
    """
    short_cm, long_cm = sorted(float(s) for s in sheet_cm)
    c = np.asarray(corners_px, dtype=np.float64)
    e_a = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[3] - c[2])) / 2.0
    e_b = (np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[0] - c[3])) / 2.0
    if e_a >= e_b:
        w, h = long_cm, short_cm
    else:
        w, h = short_cm, long_cm
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                     [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float64), (w, h)


def calibrate_from_sheet(frame_bgr, sheet_cm=SHEET_CM):
    """(H0, W0, D, corners_px, size_cm) from one frame containing a sheet, or all None.

    H0 maps image -> world cm on the table; W0/D give the world -> image mapping at any
    height, exactly as the ArUco path produces them, so callers need no special case.
    """
    corners = find_sheet(frame_bgr)
    if corners is None:
        return None, None, None, None, None
    world, size_cm = sheet_world_points(corners, sheet_cm)
    H0, _ = cv2.findHomography(corners.astype(np.float32), world.astype(np.float32))
    if H0 is None:
        return None, None, None, None, None
    W0, D = height_gradient(H0, frame_bgr.shape)
    return H0, W0, D, corners, size_cm


def residual_cm(H0, corners_px, world_pts):
    """Worst corner error in cm after fitting — the honest quality number for this fit."""
    got = cv2.perspectiveTransform(
        np.asarray(corners_px, np.float32).reshape(-1, 1, 2), H0).reshape(-1, 2)
    return float(np.max(np.linalg.norm(got - np.asarray(world_pts), axis=1)))


if __name__ == "__main__":
    # Corner ordering must survive the sheet being rotated to any angle on the table.
    base = np.array([[100.0, 80.0], [300.0, 80.0], [300.0, 220.0], [100.0, 220.0]])
    for turn in (0, 1, 2, 3):
        shuffled = np.roll(base, turn, axis=0)[::-1]      # rotated AND reversed winding
        out = order_corners(shuffled)
        assert np.allclose(out.sum(axis=1).argmin(), 0), "first corner must be top-left-most"
        # same cyclic set of points, whatever order they arrived in
        assert sorted(map(tuple, np.round(out))) == sorted(map(tuple, np.round(base)))

    # The long physical side must be assigned to the longer pixel pair, either way round.
    wide = np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 100.0], [0.0, 100.0]])
    w, size = sheet_world_points(wide)
    assert size == (29.7, 21.0), size
    tall = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 300.0], [0.0, 300.0]])
    w, size = sheet_world_points(tall)
    assert size == (21.0, 29.7), size

    # A synthetic sheet photographed by a known camera must recover its true size.
    from simulator import Camera
    cam = Camera(height_cm=55.0, tilt_deg=38.0)
    truth = np.array([[-10.5, -14.85], [10.5, -14.85], [10.5, 14.85], [-10.5, 14.85]])
    px = cam.project(np.column_stack([truth, np.zeros(4)]))
    ordered = order_corners(px)
    world, size = sheet_world_points(ordered)
    H0, _ = cv2.findHomography(ordered.astype(np.float32), world.astype(np.float32))
    err = residual_cm(H0, ordered, world)
    assert err < 0.01, f"fit residual {err:.4f} cm"
    # and a measurement through that homography must come back metric
    probe = cam.project(np.array([[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0]]))
    got = cv2.perspectiveTransform(probe.reshape(-1, 1, 2).astype(np.float32), H0).reshape(-1, 2)
    span = float(np.linalg.norm(got[1] - got[0]))
    assert abs(span - 10.0) < 0.02, f"10 cm measured as {span:.3f} cm"
    print(f"OK - sheet calibration exact on a synthetic camera "
          f"(residual {err*10:.3f} mm, 10 cm span read {span:.3f} cm)")
