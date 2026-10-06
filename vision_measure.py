"""Turn segmentation masks into real-world measurements (cm) on the calibrated table plane.

The homography is exact for points on a known plane, so a mask's outline transformed into
world coordinates gives a metrically correct footprint: length, width, orientation, area.

With two-plane calibration (plane_calib.npz) the mapping is available at ANY height, so an
object standing 3 cm tall is measured on ITS OWN plane instead of the table — removing the
magnification error that otherwise inflates every measurement.
"""
import cv2
import numpy as np


def homography_at_height(W0, D, h_cm):
    """image->world homography for the plane at height h_cm. W0/D come from plane_calib.npz."""
    return np.linalg.inv(W0 + h_cm * D)


def fit_two_plane_calibration(pts0, pts1, world_points, ref_height_cm):
    """Fit the table plane and (optionally) the height correction from two sets of clicks.

    Returns (H0, W0, D): H0 maps image->world on the table; the world->image homography
    at any height h is W0 + h*D. D is None when the second plane was skipped.
    """
    H0, _ = cv2.findHomography(pts0, world_points)
    W0 = np.linalg.inv(H0)
    if pts1 is None:
        return H0, W0, None

    H1, _ = cv2.findHomography(pts1, world_points)
    W1 = np.linalg.inv(H1)
    # findHomography returns each plane in its own projective scale. The first two
    # columns (K*r1, K*r2) do not depend on height, so rescale W1 to match W0 there —
    # without this the two planes are incomparable and the correction is wrong.
    s = np.sum(W0[:, :2] * W1[:, :2]) / np.sum(W1[:, :2] * W1[:, :2])
    return H0, W0, (s * W1 - W0) / ref_height_cm


def classify_shape(contour_world, length, width):
    """Name the object's footprint shape from geometry alone — no training, no model.

    Uses circularity (4*pi*A / P^2, = 1 for a circle), rectangular fill (contour area vs
    its bounding rectangle) and aspect ratio.
    """
    area = abs(cv2.contourArea(contour_world.astype(np.float32)))
    perim = cv2.arcLength(contour_world.astype(np.float32), True)
    if perim <= 0 or area <= 0 or width <= 0:
        return "unknown", 0.0
    circularity = 4 * np.pi * area / (perim ** 2)
    fill = area / (length * width)          # 1.0 = perfectly fills its bounding rect
    aspect = length / width
    approx = cv2.approxPolyDP(contour_world.astype(np.float32), 0.03 * perim, True)

    if circularity > 0.85 and len(approx) > 5:
        return "circle", circularity
    if len(approx) == 3 and fill < 0.80:
        return "triangle", 1.0 - abs(fill - 0.5) * 2
    if fill > 0.80:
        if aspect < 1.15:
            return "square", fill
        if aspect >= 3.5:
            return "long strip", fill
        return "rectangle", fill
    if circularity > 0.60:
        return "round-ish", circularity
    return "irregular", fill


# Hue bands in OpenCV's 0-179 scale. Naming by hue only works once a colour is known to
# BE colourful — see colour_name() for why greys have to be split off first.
_HUE_BANDS = ((8, "red"), (22, "orange"), (33, "yellow"), (85, "green"),
              (100, "cyan"), (130, "blue"), (150, "purple"), (170, "pink"), (180, "red"))

# Every name colour_name() can return — the vocabulary a spoken command can filter on.
NAMED_COLOUR_WORDS = frozenset(
    ["black", "white", "grey", "gray", "brown"] + [n for _, n in _HUE_BANDS])


def colour_name(bgr):
    """Name a BGR colour the way a person would.

    Plain RGB distance to reference colours fails badly: it called a light green 'grey'
    and a tan 'pink', because lightness dominates the distance. Splitting saturation and
    value off from hue first fixes that — decide grey/black/white by how colourful and how
    bright it is, and only then pick a hue.
    """
    px = np.uint8([[[max(0, min(255, int(c))) for c in bgr]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].tolist()

    if v < 55:
        return "black"
    if s < 55:
        return "white" if v > 185 else "grey"
    if s < 90 and v < 150:
        return "brown"              # dark and only weakly coloured reads as brown
    if h < 20 and v < 145:
        return "brown"              # dark orange/red is brown, not red
    for upper, name in _HUE_BANDS:
        if h < upper:
            return name
    return "red"


def sample_colour(frame_bgr, poly_px, erode_px=3):
    """Average colour inside an object's mask, plus a human-readable colour name.

    The mask is eroded first: a segmentation outline usually clips a little background,
    and those edge pixels would drag the average toward whatever the object is sitting on.
    Returns (bgr_tuple, colour_name) or (None, None) if nothing usable is inside.
    """
    poly = np.asarray(poly_px, np.float32)
    if len(poly) < 3:
        return None, None
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1,) * 2, np.uint8)
        eroded = cv2.erode(mask, k)
        if eroded.any():          # keep the raw mask if erosion wipes out a small object
            mask = eroded
    if not mask.any():
        return None, None

    # MEDIAN, not mean. A mask often catches some background, and a partly occluded object
    # catches pixels belonging to whatever is in front of it — measured here, a black cube
    # passing behind a red one averaged out to dark red and got named "brown". The mean is
    # dragged by any such minority; the median ignores it.
    pixels = frame_bgr[mask > 0]
    if pixels.size == 0:
        return None, None
    med = np.median(pixels.reshape(-1, 3), axis=0)
    return tuple(float(c) for c in med), colour_name(med)


def vertical_vanishing_point(D):
    """Image point where all vertical lines meet. D = K[0 0 r3], so its 3rd column is it."""
    v = D[:, 2]
    if abs(v[2]) < 1e-12:
        return None, v          # camera looking dead-on: verticals stay parallel
    return v[:2] / v[2], v


def _height_from_pair(base_px, top_px, H0, W0, v):
    """Solve for height given a base point (on the table) and the top of its vertical."""
    X = cv2.perspectiveTransform(
        np.asarray(base_px, np.float32).reshape(1, 1, 2), H0).reshape(2)
    p0 = W0 @ np.array([X[0], X[1], 1.0])
    # top ~ p0 + h*v in homogeneous coords; solve both image coords for h, least squares.
    A = np.array([v[i] - top_px[i] * v[2] for i in range(2)])
    r = np.array([top_px[i] * p0[2] - p0[i] for i in range(2)])
    denom = float(A @ A)
    return None if denom < 1e-12 else float((A @ r) / denom)


def estimate_height(poly_px, H0, W0, D, angle_tol_deg=0.6, min_edge_px=3.0,
                    simplify_frac=0.004):
    """Estimate an object's height in cm from ONE view, using its own vertical edges.

    World verticals project to lines through the vertical vanishing point, so the base and
    top of a vertical edge share a bearing from that point. The visible vertical edges are
    the silhouette's side tangents (the base corner under the FARTHEST top corner is hidden
    inside the outline, so distance alone picks the wrong pair). Every same-bearing pair is
    solved and the median taken, which ignores spurious pairs from a ragged mask.

    A real segmentation mask is a staircase of pixel steps, not a clean polygon, and a
    staircase has no true corners for the bearing test to latch onto — measured on degraded
    masks, skipping the simplification below dropped usable estimates from 22/36 to 4/36.
    So the outline is simplified first to recover its real corners.

    Returns None when the geometry is degenerate (object under the camera, no visible side).
    """
    vp, v = vertical_vanishing_point(D)
    if vp is None or len(poly_px) < 3:
        return None
    hull = cv2.convexHull(np.asarray(poly_px, np.float32))
    if simplify_frac > 0 and len(hull) > 3:
        hull = cv2.approxPolyDP(hull, simplify_frac * cv2.arcLength(hull, True), True)
    hull = hull.reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return None
    if cv2.pointPolygonTest(hull.astype(np.float32), (float(vp[0]), float(vp[1])), False) >= 0:
        return None      # vanishing point inside the outline: no side face visible

    rel = hull - vp
    dist = np.linalg.norm(rel, axis=1)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    # vp is outside the hull, so the outline spans a small arc — unwrap around its mean.
    ang = np.arctan2(np.sin(ang - ang.mean()), np.cos(ang - ang.mean()))

    # Every same-bearing pair is a candidate vertical edge. Real edges agree closely;
    # spurious pairs (the base corner under the farthest top corner is hidden inside the
    # outline) scatter. Take the median, and refuse to answer when they disagree —
    # a declined estimate is worth more than a confidently wrong one.
    heights = []
    for i in range(len(hull)):
        for j in range(len(hull)):
            if i == j or dist[j] - dist[i] < min_edge_px:
                continue          # j must be the far end (the top) by a usable margin
            if abs(ang[i] - ang[j]) > np.radians(angle_tol_deg):
                continue          # not the same vertical line
            h = _height_from_pair(hull[i], hull[j], H0, W0, v)
            if h is not None and 0.05 < h < 100.0:
                heights.append(h)

    # A single candidate has nothing to corroborate it, and on ragged masks those lone
    # estimates were the ones that went badly wrong — requiring a second agreeing edge cut
    # the worst observed error from 54 mm to 9 mm, at the cost of some coverage. Coverage
    # is cheaper to recover than accuracy: a frame that declines is filled in from the
    # object's remembered height (see object_tracker.TrailTracker).
    if len(heights) < 2:
        return None
    med = float(np.median(heights))
    mad = float(np.median(np.abs(np.array(heights) - med)))
    if mad > 0.15 * med:
        return None               # candidate edges disagree — geometry is unreliable here
    return med


def measure_polygon(poly_px, H, label="object", conf=1.0):
    """Measure one mask outline on the plane described by H. Returns a dict, or None."""
    poly = np.asarray(poly_px, dtype=np.float64)
    if len(poly) < 3:
        return None
    # Mask outline (image px) -> world plane (cm)
    world = cv2.perspectiveTransform(
        poly.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)

    (cx, cy), (a, b), ang = cv2.minAreaRect(world)
    # OpenCV returns the angle of the rect's first edge; normalise so the reported
    # angle always describes the LONG axis of the object.
    long_axis_deg = ang if a >= b else ang + 90.0
    length, width = float(max(a, b)), float(min(a, b))
    shape, shape_score = classify_shape(world, length, width)
    return {
        "label": label,
        "conf": float(conf),
        "shape": shape,
        "shape_score": float(shape_score),
        "center_cm": (float(cx), float(cy)),
        "length_cm": length,
        "width_cm": width,
        "angle_deg": float(long_axis_deg % 180.0),           # object's long axis
        "grasp_deg": float((long_axis_deg + 90.0) % 180.0),  # gripper closes across it
        "area_cm2": float(cv2.contourArea(world)),
        "contour_cm": world,
        "mask_px": poly,
    }


def measure_objects(result, H, max_footprint_cm2=1000.0):
    """Measure every segmented object. Returns a list of dicts.

    Attaches each detection's tracker ID (result.boxes.id, from model.track()) HERE, zipped
    alongside the polygon it belongs to — not by position after the fact, because
    measure_scene() sorts and filters this list by area and that would silently mismatch
    IDs to the wrong objects.
    """
    out = []
    if result.masks is None:
        return out

    names = result.names
    ids = result.boxes.id
    ids = ids.int().tolist() if ids is not None else [None] * len(result.masks.xy)
    for poly, cls, conf, tid in zip(result.masks.xy, result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist(), ids):
        m = measure_polygon(poly, H, names[int(cls)], conf)
        if m is not None:
            m["track_id"] = tid
            out.append(m)

    return [o for o in out if o["area_cm2"] <= max_footprint_cm2]


def footprint_polygon(poly_px, H0, W0, D, h_cm):
    """Recover an object's true footprint (cm) from its silhouette.

    A segmentation mask is the whole silhouette — top face AND the visible side faces —
    so unprojecting it onto the table over-reports size badly at oblique angles. Unproject
    it on the ground plane and the top spills OUTWARD; unproject on the top plane and the
    base spills INWARD. Those two errors point opposite ways, so the intersection of the
    two polygons is the footprint they agree on.

    NOTE ON A WRONG h_cm (e.g. a bad height hint from a misclassified object): the two
    unprojections don't just get bigger or smaller together, they slide apart — the
    top-plane homography translates points sideways as well as scaling them, so two
    similar-sized hulls end up OFFSET rather than nested, and intersect in a thin diagonal
    sliver instead of a shrunken copy of either. Measured with a synthetic camera: a wrong
    height on a shallow-tilt view turned a true 6x6 cm footprint into 62.6x23.9 cm, while
    `area` stayed positive the whole time, so it alone can't catch this. A geometric guard
    was tried here and dropped: comparing the intersection area against the two source
    hulls also flags CORRECT heights, because this algorithm's own hulls are supposed to be
    bigger than the true footprint (that's how it corrects for outward/inward spill), so
    "the intersection is much smaller than its inputs" is normal, not a fault signal. The
    guard that actually holds lives one layer up, in dashboard_sources.py: re-check the
    result against the same size limits used to filter every other candidate, AFTER a
    height hint recomputes it, and drop the hint instead of trusting a wrong answer.
    """
    hull = cv2.convexHull(np.asarray(poly_px, np.float32)).reshape(-1, 1, 2)
    ground = cv2.perspectiveTransform(hull, H0).reshape(-1, 2)
    top = cv2.perspectiveTransform(hull, homography_at_height(W0, D, h_cm)).reshape(-1, 2)
    a = cv2.convexHull(ground.astype(np.float32))
    b = cv2.convexHull(top.astype(np.float32))
    area, inter = cv2.intersectConvexConvex(a, b)
    if inter is None or len(inter) < 3 or area <= 0:
        return None
    return inter.reshape(-1, 2)


def measure_scene(result, H0, W0=None, D=None, max_footprint_cm2=400.0,
                  fallback_height_cm=3.0, min_footprint_cm2=1.0, max_span_cm=None,
                  min_conf=0.0, rejected=None):
    """Measure the whole scene: the work surface, plus every pick candidate.

    Each candidate's height is measured from its own vertical edges when the two-plane
    calibration is available, then it is re-measured on ITS OWN plane so length/width and
    position are free of the magnification error. Falls back to a fixed height when the
    geometry cannot support an estimate.

    Returns (surface, objects). surface is the largest detection (the table), or None.
    Pass a list as `rejected` to find out WHY a detection did not become a candidate —
    otherwise an object visible in the camera view just silently fails to appear.
    """
    everything = measure_objects(result, H0, max_footprint_cm2=1e12)
    if not everything:
        return None, []
    everything.sort(key=lambda o: -o["area_cm2"])

    surface = everything[0] if everything[0]["area_cm2"] > max_footprint_cm2 else None
    objects = []
    for o in everything:
        # A pick candidate must be small by BOTH area and longest side: a table's mask is
        # irregular enough that its area alone can slip under an area-only limit.
        why = None
        if o["area_cm2"] > max_footprint_cm2:
            why = f"footprint {o['area_cm2']:.0f} cm2 > {max_footprint_cm2:.0f} limit"
        elif o["area_cm2"] < min_footprint_cm2:
            why = f"footprint {o['area_cm2']:.1f} cm2 < {min_footprint_cm2:.0f} minimum"
        elif max_span_cm is not None and o["length_cm"] > max_span_cm:
            why = f"longest side {o['length_cm']:.0f} cm > {max_span_cm:.0f} limit"
        elif o["conf"] < min_conf:
            why = f"confidence {o['conf']:.2f} < {min_conf:.2f}"
        if why:
            if rejected is not None:
                rejected.append((o.get("label", "object"), why))
            continue
        height, measured = fallback_height_cm, False
        if D is not None and W0 is not None:
            est = estimate_height(o["mask_px"], H0, W0, D)
            if est is not None:
                height, measured = est, True
            # Re-measure the true footprint, with the side faces removed.
            fp = footprint_polygon(o["mask_px"], H0, W0, D, height)
            re = (measure_polygon(fp, np.eye(3), o["label"], o["conf"])
                  if fp is not None else
                  measure_polygon(o["mask_px"], homography_at_height(W0, D, height),
                                  o["label"], o["conf"]))
            if re is not None:
                mask_px, tid = o["mask_px"], o.get("track_id")
                o = re
                o["mask_px"] = mask_px      # keep the pixel outline for drawing
                o["track_id"] = tid
        o["height_cm"] = float(height)
        o["height_measured"] = measured
        objects.append(o)
    return surface, objects


def select_target(objects, shape=None, label=None, size=None):
    """Pick one object by shape, label, and/or relative size ('smallest'/'biggest')."""
    pool = objects
    if shape:
        pool = [o for o in pool if o["shape"] == shape]
    if label:
        pool = [o for o in pool if label.lower() in o["label"].lower()]
    if not pool:
        return None
    if size == "smallest":
        return min(pool, key=lambda o: o["area_cm2"])
    if size == "biggest":
        return max(pool, key=lambda o: o["area_cm2"])
    return max(pool, key=lambda o: (o["shape_score"] if shape else 0.0, o["conf"]))


SHAPE_WORDS = ("square", "rectangle", "circle", "triangle", "long strip", "round-ish",
               "irregular")
SIZE_WORDS = {"smallest": "smallest", "small": "smallest", "tiniest": "smallest",
              "biggest": "biggest", "largest": "biggest", "big": "biggest",
              "large": "biggest"}
_STOPWORDS = {"pick", "up", "the", "a", "an", "that", "this", "object", "one", "please",
              "get", "grab", "take", "me", "it", "and", "of"}

# Everyday words mapped onto the label text a detector might actually produce, so asking
# for a "phone" still matches a detection labelled "cell phone".
LABEL_SYNONYMS = {
    "phone": {"phone", "cell", "mobile", "smartphone"},
    "mobile": {"phone", "cell", "mobile", "smartphone"},
    "cellphone": {"phone", "cell", "mobile"},
    "cube": {"cube", "box", "block"},
    "box": {"box", "cube", "carton", "block"},
    "block": {"block", "cube", "box"},
    "bottle": {"bottle", "flask"},
    "cup": {"cup", "mug", "glass"},
    "mug": {"mug", "cup"},
    "book": {"book", "notebook"},
    "remote": {"remote", "controller"},
}


def parse_command(text):
    """Turn 'pick up the black mobile phone' into filters the matcher can apply.

    Deliberately keyword-based rather than an LLM: it runs offline, instantly, and its
    behaviour is inspectable — which matters more here than handling unusual phrasing.
    """
    words = [w.strip(".,!?").lower() for w in text.split()]
    wanted = {"colour": None, "shape": None, "size": None, "keywords": []}

    lowered = " ".join(words)
    for shape in SHAPE_WORDS:                    # check multi-word shapes before tokens
        if shape in lowered:
            wanted["shape"] = shape
            lowered = lowered.replace(shape, " ")
    remaining = [w for w in lowered.split() if w]

    for w in remaining:
        if w in NAMED_COLOUR_WORDS and wanted["colour"] is None:
            wanted["colour"] = w
        elif w in SIZE_WORDS and wanted["size"] is None:
            wanted["size"] = SIZE_WORDS[w]
        elif w not in _STOPWORDS:
            wanted["keywords"].append(w)
    return wanted


def match_command(objects, text, verbose=False):
    """Pick the one object a spoken/typed instruction refers to.

    Colour, shape and label words are hard filters — asking for the black phone must never
    return a white book. Size is applied last, as a chooser among whatever survived.
    Returns (object_or_None, explanation).
    """
    want = parse_command(text)
    pool, reasons = list(objects), []

    if want["colour"]:
        pool = [o for o in pool if o.get("colour_name") == want["colour"]]
        reasons.append(f"colour={want['colour']} ({len(pool)} left)")
    if want["shape"]:
        pool = [o for o in pool if o.get("shape") == want["shape"]]
        reasons.append(f"shape={want['shape']} ({len(pool)} left)")
    for kw in want["keywords"]:
        terms = LABEL_SYNONYMS.get(kw, {kw})
        narrowed = [o for o in pool
                    if any(t in str(o.get("label", "")).lower() for t in terms)]
        if narrowed:
            pool = narrowed
            reasons.append(f"label~{kw} ({len(pool)} left)")
        else:
            # The detector's label is the least trustworthy thing we have — it has been
            # seen calling a phone a "cup" at 0.2 confidence, while colour and measured
            # size were right. So a label word that matches nothing is dropped rather than
            # allowed to empty the pool; colour and shape still did the real filtering.
            reasons.append(f"label~{kw} ignored (no detection is labelled that)")

    explanation = "; ".join(reasons) if reasons else "no filters understood"
    if not pool:
        return None, f"nothing matched: {explanation}"
    if want["size"] == "smallest":
        return min(pool, key=lambda o: o["area_cm2"]), f"{explanation}; smallest"
    if want["size"] == "biggest":
        return max(pool, key=lambda o: o["area_cm2"]), f"{explanation}; biggest"
    # No size given: prefer the most confident detection among what is left.
    return max(pool, key=lambda o: o.get("conf", 0.0)), explanation


def apply_height_hint(o, height_cm, H0, W0, D):
    """Re-measure an object assuming a known height, when the geometry could not find one.

    Height is the linchpin for a TALL object: measured on a 7x7x22 cm bottle, using the
    default 3 cm fallback reported its footprint as 21.8 x 17.0 cm, while the true 22 cm
    gave 7.8 x 7.6 cm. The vanishing-point estimator needs two clean vertical edges and
    tall objects here often present only one, so an operator-supplied height for known
    object types beats an assumption that is known to be wrong.

    Mutates and returns o. Leaves height_measured False: this is told, not measured.
    """
    if D is None or W0 is None:
        o["height_cm"] = float(height_cm)
        return o
    fp = footprint_polygon(o["mask_px"], H0, W0, D, float(height_cm))
    re = (measure_polygon(fp, np.eye(3), o["label"], o["conf"]) if fp is not None
          else measure_polygon(o["mask_px"], homography_at_height(W0, D, float(height_cm)),
                               o["label"], o["conf"]))
    if re is not None:
        keep = {k: o[k] for k in ("track_id", "mask_px", "colour_bgr", "colour_name")
                if k in o}
        o.update(re)
        o.update(keep)
    o["height_cm"] = float(height_cm)
    o["height_measured"] = False
    o["height_hinted"] = True
    return o


def describe(o, with_size=True):
    """Short human label for an object, without saying its colour twice.

    Prompts are often descriptive ("black product box"), so blindly prefixing the detected
    colour produced "black black product box".
    """
    label = str(o.get("label", "object"))
    colour = o.get("colour_name")
    name = label if (not colour or colour in label.lower()) else f"{colour} {label}"
    if not with_size:
        return name
    return (f"{name} {o['length_cm']:.0f}x{o['width_cm']:.0f}"
            f"x{'' if o.get('height_measured') else '~'}{o['height_cm']:.0f}cm")


def filter_inside_workspace(objects, workspace_centres, margin_cm=0.0):
    """Keep only objects whose centre falls inside the workspace boundary (or all, if None)."""
    if workspace_centres is None or len(workspace_centres) < 3:
        return objects
    hull = cv2.convexHull(np.asarray(workspace_centres, np.float32))
    kept = []
    for o in objects:
        d = cv2.pointPolygonTest(hull, (float(o["center_cm"][0]), float(o["center_cm"][1])), True)
        if d >= margin_cm:
            kept.append(o)
    return kept


def _fake_result(polygon_px, label="test"):
    """Minimal stand-in for a YOLO result, for the self-tests below."""
    return type("R", (), {
        "names": {0: label},
        "masks": type("M", (), {"xy": [polygon_px]}),
        "boxes": type("B", (), {"cls": type("t", (), {"tolist": lambda s: [0]})(),
                                "conf": type("t", (), {"tolist": lambda s: [1.0]})(),
                                "id": None}),
    })()


def _rect_world(length, width, angle_deg, cx, cy):
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    corners = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                        [length / 2, width / 2], [-length / 2, width / 2]])
    return (corners @ np.array([[c, s], [-s, c]])) + [cx, cy]


if __name__ == "__main__":
    # --- Test 1: planar measurement is metrically exact (uses a real calibration) ---
    img_pts = np.array([[51, 230], [454, 220], [588, 471], [12, 464]], dtype=np.float32)
    sq = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, sq)

    rect = _rect_world(8.0, 5.0, 30.0, 3.0, -2.0)
    rect_img = cv2.perspectiveTransform(
        rect.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(H)).reshape(-1, 2)
    m = measure_objects(_fake_result(rect_img), H)[0]
    print(f"[1] planar     truth 8.00 x 5.00 cm  ->  measured "
          f"{m['length_cm']:.2f} x {m['width_cm']:.2f} cm")
    assert abs(m["length_cm"] - 8.0) < 0.05 and abs(m["width_cm"] - 5.0) < 0.05

    # --- Test 2: height correction, against a synthetic camera with known geometry ---
    # Camera 60 cm above the table, tilted slightly, 640x480, focal 550 px.
    K = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1.0]])
    tilt = np.radians(15.0)
    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]]) @ np.diag([1.0, -1.0, -1.0])
    t = np.array([0.0, 0.0, 60.0])

    def world_to_image(h):  # exact world->image homography for the plane at height h
        return K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * h + t])

    W_true0, W_true1 = world_to_image(0.0), world_to_image(5.0)
    # Simulate what calibration actually produces: each plane in its own arbitrary scale.
    H0 = np.linalg.inv(W_true0 / W_true0[2, 2])
    H1 = np.linalg.inv(W_true1 / W_true1[2, 2])
    W0, W1 = np.linalg.inv(H0), np.linalg.inv(H1)
    s = np.sum(W0[:, :2] * W1[:, :2]) / np.sum(W1[:, :2] * W1[:, :2])
    D = (s * W1 - W0) / 5.0

    # A 4x4 cm cube top face, sitting 3 cm above the table.
    cube_h = 3.0
    top = _rect_world(4.0, 4.0, 0.0, 6.0, 4.0)
    top_img = cv2.perspectiveTransform(
        top.reshape(-1, 1, 2).astype(np.float32), world_to_image(cube_h)).reshape(-1, 2)

    wrong = measure_objects(_fake_result(top_img), H0)[0]
    right = measure_objects(_fake_result(top_img), homography_at_height(W0, D, cube_h))[0]
    err_wrong = abs(wrong["length_cm"] - 4.0) / 4.0 * 100
    err_right = abs(right["length_cm"] - 4.0) / 4.0 * 100
    print(f"[2] 4.00 cm cube 3 cm tall:")
    print(f"      table-plane homography -> {wrong['length_cm']:.2f} cm  ({err_wrong:.1f}% error)")
    print(f"      height-corrected       -> {right['length_cm']:.2f} cm  ({err_right:.1f}% error)")
    assert err_right < 0.5, "height-corrected measurement should be near-exact"
    assert err_wrong > err_right, "correction must beat the uncorrected measurement"

    # Position must also be corrected, not just size.
    dx = np.hypot(right["center_cm"][0] - 6.0, right["center_cm"][1] - 4.0)
    print(f"      position error after correction: {dx*10:.2f} mm")
    assert dx < 0.05
    print("OK — planar measurement exact, height correction validated")
