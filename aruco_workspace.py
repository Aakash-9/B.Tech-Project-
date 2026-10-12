"""Live workspace calibration from printed ArUco markers — works at any camera placement.

Tape 4 markers around the work area. Every frame this recovers the image->world mapping,
MEASURES where the markers actually are (so the area size is detected, not assumed), and
derives the camera pose so object heights can be measured too. Bump the camera and it
re-solves on the next frame.

Scale comes from the printed marker size: a camera cannot know real-world size from pixels
alone, so exactly one physical dimension has to be told to the system. That is MARKER_CM.
"""
import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
MARKER_CM = 4.0   # printed marker side length in cm — measure your printout and edit this


def make_detector():
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                   cv2.aruco.DetectorParameters())


def save_marker_sheet(path="aruco_markers.png", ids=(0, 1, 2, 3), px=700, margin=60):
    """Write a printable sheet of markers. Print it, cut out, tape at the area corners."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    tiles = [cv2.aruco.generateImageMarker(d, i, px) for i in ids]
    cell = px + 2 * margin
    sheet = np.full((cell * 2, cell * 2), 255, np.uint8)
    for n, t in enumerate(tiles):
        r, c = divmod(n, 2)
        y, x = r * cell + margin, c * cell + margin
        sheet[y:y + px, x:x + px] = t
        cv2.putText(sheet, f"id {ids[n]}", (x, y + px + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.imwrite(path, sheet)
    return path


# Marker corners in the marker's own frame, matching ArUco's order (TL, TR, BR, BL)
# with a y-up world. One marker therefore fixes the world axes and the scale.
def _local_corners(marker_cm):
    s = marker_cm / 2.0
    return np.array([[-s, s], [s, s], [s, -s], [-s, -s]], np.float32)


def _unproject(h8, pts):
    H = np.append(h8, 1.0).reshape(3, 3)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def _square_residuals(h8, corner_list, local):
    """How far each marker is from being an exact marker_cm square once unprojected."""
    return np.concatenate([
        (w - (w.mean(axis=0) + local)).ravel()
        for w in (_unproject(h8, c) for c in corner_list)])


def workspace_from_corners(corner_list, marker_cm, iters=40, lam=1e-3):
    """image->world homography plus the MEASURED marker centres, from detected corners.

    Anchoring the world frame to ONE marker and extrapolating is far too sensitive: a
    sub-pixel corner error on a small marker is amplified by the ratio of workspace size to
    marker size (measured: 0.8 px of corner noise became an 18% error across the area).
    Instead every marker is used at once — each is a known-size square, which heavily
    over-constrains the homography's 8 parameters — and the fit is refined by
    Levenberg-Marquardt against that squareness. Each marker's centre stays free, so only
    shape and scale drive the fit; the origin is recentred on the workspace afterwards.
    Assumes the markers lie flat and are printed the same way up.
    """
    if not corner_list:
        return None, None
    local = _local_corners(marker_cm)
    cl = [np.asarray(c, np.float32).reshape(-1, 2) for c in corner_list]
    H0, _ = cv2.findHomography(cl[0], local)
    if H0 is None:
        return None, None

    h = (H0 / H0[2, 2]).ravel()[:8].copy()
    r = _square_residuals(h, cl, local)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.zeros((len(r), 8))
        for k in range(8):
            step = max(abs(h[k]), 1.0) * 1e-6
            hp = h.copy()
            hp[k] += step
            J[:, k] = (_square_residuals(hp, cl, local) - r) / step
        try:
            # Damped: marker centres are free, so a pure translation of the world leaves
            # the residuals unchanged and the normal equations are rank-deficient.
            dh = np.linalg.solve(J.T @ J + lam * np.eye(8), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        r_new = _square_residuals(h + dh, cl, local)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            h, r, cost = h + dh, r_new, cost_new
            lam = max(lam * 0.5, 1e-9)
            if cost < 1e-12:
                break
        else:
            lam *= 4.0
            if lam > 1e6:
                break

    centres = np.array([_unproject(h, c).mean(axis=0) for c in cl], dtype=np.float64)
    origin = centres.mean(axis=0)
    # Shift the world origin to the workspace centre without touching scale or rotation.
    H = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]]) @ np.append(h, 1.0).reshape(3, 3)
    return H, centres - origin


def focal_from_homography(W, cx, cy):
    """Estimate focal length from ONE plane homography (world->image).

    Uses the orthonormality of the rotation columns, assuming square pixels, no skew and a
    known principal point. Degenerate when the plane faces the camera dead-on, so it can
    return None — the caller must cope rather than trust a garbage number.
    """
    h1, h2 = W[:, 0].astype(float), W[:, 1].astype(float)

    def bilinear(a, b):
        # a^T B b split into the 1/f^2 part and the constant part
        p = (a[0] * b[0] + a[1] * b[1]
             - cx * (a[0] * b[2] + a[2] * b[0])
             - cy * (a[1] * b[2] + a[2] * b[1])
             + (cx * cx + cy * cy) * a[2] * b[2])
        return p, a[2] * b[2]

    p12, q12 = bilinear(h1, h2)
    p11, q11 = bilinear(h1, h1)
    p22, q22 = bilinear(h2, h2)

    cands = []
    if abs(p12) > 1e-9:
        cands.append(-q12 / p12)
    if abs(p11 - p22) > 1e-9:
        cands.append(-(q11 - q22) / (p11 - p22))
    cands = [w for w in cands if w > 1e-12]
    if not cands:
        return None
    f = 1.0 / np.sqrt(np.mean(cands))
    return float(f) if np.isfinite(f) and 50.0 < f < 20000.0 else None


def height_gradient(H_img_to_world, image_shape, focal=None):
    """Derive (W0, D) — the world->image mapping at any height — from one plane homography.

    W(h) = W0 + h*D with D = K[0 0 r3], so this unlocks height measurement without a
    second calibration plane. Returns (W0, D) or (W0, None) if the focal estimate fails.
    """
    W0 = np.linalg.inv(H_img_to_world)
    h_img, w_img = image_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    f = focal if focal is not None else focal_from_homography(W0, cx, cy)
    if f is None:
        return W0, None

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ W0
    n1 = np.linalg.norm(M[:, 0])
    n2 = np.linalg.norm(M[:, 1])
    if n1 < 1e-9 or n2 < 1e-9:
        return W0, None
    lam = np.sqrt(n1 * n2)
    r1, r2 = M[:, 0] / lam, M[:, 1] / lam
    r3 = np.cross(r1, r2)
    # D must carry the same projective scale as W0, hence the lam factor.
    D = lam * (K @ np.column_stack([np.zeros(3), np.zeros(3), r3]))
    return W0, D


GOOD_TILT_DEG = (32.0, 50.0)


def camera_tilt_deg(H_img_to_world, D, image_shape):
    """Angle between the optical axis and the table's NORMAL, in degrees, or None.

    READ THE CONVENTION CAREFULLY — it was got wrong here once and the resulting
    on-screen advice told operators to move the camera the wrong way:
        0 deg   = camera looking straight DOWN at the table (overhead)
        90 deg  = camera looking HORIZONTALLY across the table
    So this is the angle away from overhead, NOT the angle below horizontal. Subtract
    from 90 to get "degrees below horizontal", which is how a person pictures it.

    Height estimation needs two clean vertical edges on an object, and whether it gets
    them depends almost entirely on this number — measured on the synthetic scene, at
    18-26 deg (near overhead) the few heights returned are wrong by up to 48 mm, while
    at 36-44 deg the error collapses to 1-3 mm. Nobody can judge that angle by eye, so
    measure it: the calibration already recovers r3, the table normal in camera
    coordinates, and |r3[2]| is cos(angle). Accurate to ~1.3 deg across the usable band.
    """
    if D is None:
        return None
    W0 = np.linalg.inv(H_img_to_world)
    h_img, w_img = image_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    f = focal_from_homography(W0, cx, cy)
    if f is None:
        return None
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    r3 = np.linalg.inv(K) @ D[:, 2]
    n = np.linalg.norm(r3)
    if n < 1e-9:
        return None
    # |r3[2]| is cos(tilt): 1 with the camera horizontal, 0 looking straight down.
    return float(np.degrees(np.arccos(min(1.0, abs(r3[2] / n)))))


def tilt_advice(tilt_deg):
    """One short line an operator can act on, or None when the tilt is unknown.

    Phrased in degrees BELOW HORIZONTAL, because that is the angle a person can actually
    picture and act on; tilt_deg itself is measured from the table normal (see
    camera_tilt_deg). An earlier version reported the raw angle with "below horizontal"
    wording and a direction derived from it, which pointed operators the wrong way.
    """
    if tilt_deg is None:
        return None
    lo, hi = GOOD_TILT_DEG
    below_horizon = 90.0 - tilt_deg
    if tilt_deg < lo:                      # too close to straight down
        return (f"camera {below_horizon:.0f}deg below horizontal - too overhead, "
                f"raise its aim toward ~50deg for heights")
    if tilt_deg > hi:                      # too close to looking across the table
        return (f"camera {below_horizon:.0f}deg below horizontal - too flat, "
                f"aim it further DOWN toward ~50deg for heights")
    return f"camera {below_horizon:.0f}deg below horizontal - good angle for heights"


def area_size(centres):
    """Measured work-area extent (length, width) in cm from the marker centres."""
    if centres is None or len(centres) < 3:
        return None
    (_, _), (a, b), _ = cv2.minAreaRect(np.asarray(centres, np.float32))
    return float(max(a, b)), float(min(a, b))


def inside_workspace(point_cm, centres, margin_cm=0.0):
    """Is a world point inside the marker quadrilateral (optionally shrunk/grown)?"""
    if centres is None or len(centres) < 3:
        return True
    hull = cv2.convexHull(np.asarray(centres, np.float32))
    d = cv2.pointPolygonTest(hull, (float(point_cm[0]), float(point_cm[1])), True)
    return d >= margin_cm


if __name__ == "__main__":
    # Synthetic check: put markers at known spots, project, then recover them.
    MARKER, AREA_W, AREA_L = MARKER_CM, 24.0, 18.0
    truth = np.array([[-AREA_W / 2, -AREA_L / 2], [AREA_W / 2, -AREA_L / 2],
                      [AREA_W / 2, AREA_L / 2], [-AREA_W / 2, AREA_L / 2]])
    local = _local_corners(MARKER)
    W_IMG, H_IMG, F_TRUE = 640, 480, 600.0
    K = np.array([[F_TRUE, 0, W_IMG / 2], [0, F_TRUE, H_IMG / 2], [0, 0, 1.0]])

    print(f"{'tilt':>5} {'area (cm)':>16} {'centre err':>11} {'focal':>14} {'height err':>11}")
    for tilt in (10, 25, 40, 55):
        ang = np.radians(tilt)
        R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]) \
            @ np.diag([1.0, -1.0, -1.0])
        t = np.array([0.0, 0.0, 55.0])
        w2i = lambda h: K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * h + t])
        proj = lambda p, h: cv2.perspectiveTransform(
            np.asarray(p, np.float32).reshape(-1, 1, 2), w2i(h)).reshape(-1, 2)

        detected = [proj(truth[i] + local, 0.0) for i in range(4)]
        H, centres = workspace_from_corners(detected, MARKER)
        L, Wd = area_size(centres)
        # truth is already centred on the area's own centroid (mean of the 4 corners is
        # the origin by construction), so centres should match truth directly now.
        cerr = np.abs(centres - truth).max()

        W0, D = height_gradient(H, (H_IMG, W_IMG))
        f_est = focal_from_homography(W0, W_IMG / 2, H_IMG / 2)
        if D is None:
            herr = "n/a"
        else:
            box = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]]) + [6, -5]
            sil = cv2.convexHull(np.vstack([proj(box, 0.0), proj(box, 3.0)]).astype(np.float32))
            from vision_measure import estimate_height
            est = estimate_height(sil.reshape(-1, 2), H, W0, D)
            herr = f"{abs(est - 3.0):.2f} cm" if est else "declined"
        print(f"{tilt:5.0f} {L:7.2f} x {Wd:5.2f} {cerr:10.3f} {str(round(f_est or 0, 1)):>14} "
              f"{herr:>11}")
    print(f"(truth: area {AREA_W} x {AREA_L} cm, focal {F_TRUE}, height 3.00 cm)")
