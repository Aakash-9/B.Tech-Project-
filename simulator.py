"""Synthetic scene renderer: a virtual camera looking at a table with the arm at its centre.

Renders a photographable-looking scene (table surface, ArUco markers, 3D boxes) through a
virtual pinhole camera, and reports the EXACT ground truth for everything in it. Because the
rendered frames go into the same pipeline the webcam feeds, accuracy can be measured
end-to-end without any physical setup — and any fix proven here is a fix in the real path.

World frame: Z=0 is the table, +Z is up, and the arm's base sits at the origin (0, 0, 0).
"""
import cv2
import numpy as np

from aruco_workspace import DICT, MARKER_CM

TABLE_COLOUR = (78, 102, 130)     # BGR, a muted wood tone
BG_COLOUR = (60, 60, 62)


class Camera:
    """Pinhole camera looking down at the table from a given height, tilt and yaw."""

    def __init__(self, height_cm=50.0, tilt_deg=18.0, yaw_deg=0.0,
                 focal=600.0, size=(640, 480)):
        self.w, self.h = size
        self.K = np.array([[focal, 0, self.w / 2.0],
                           [0, focal, self.h / 2.0],
                           [0, 0, 1.0]])
        a, y = np.radians(tilt_deg), np.radians(yaw_deg)
        tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        spin = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        # diag(1,-1,-1) points the camera down the world -Z axis before tilting.
        self.R = tilt @ np.diag([1.0, -1.0, -1.0]) @ spin
        self.t = np.array([0.0, 0.0, height_cm])
        self.height_cm = height_cm

    def project(self, pts3d):
        """World (X, Y, Z) in cm -> image pixels."""
        p = np.asarray(pts3d, dtype=float).reshape(-1, 3)
        cam = (self.R @ p.T).T + self.t
        cam[:, 2] = np.where(np.abs(cam[:, 2]) < 1e-6, 1e-6, cam[:, 2])
        img = (self.K @ cam.T).T
        return img[:, :2] / img[:, 2:3]

    def centre_world(self):
        return -self.R.T @ self.t


class Box:
    """An axis-aligned box of known size standing on the table, optionally rotated in yaw.

    Pass `motion` as a function of time -> (x, y) [optionally (x, y, yaw)] to make the
    object move, so tracking and picking moving objects can be tested against known truth.
    """

    def __init__(self, x, y, length, width, height, yaw_deg=0.0, colour=(200, 200, 205),
                 name="box", motion=None):
        self.x, self.y = float(x), float(y)
        self.length, self.width, self.height = float(length), float(width), float(height)
        self.yaw = float(yaw_deg)
        self.colour = colour
        self.name = name
        self.motion = motion

    def advance(self, t):
        """Move this object to where its motion function says it should be at time t."""
        if self.motion is None:
            return
        pose = self.motion(t)
        self.x, self.y = float(pose[0]), float(pose[1])
        if len(pose) > 2:
            self.yaw = float(pose[2])

    def footprint(self):
        """The 4 base corners in world XY (cm)."""
        L, W = self.length / 2.0, self.width / 2.0
        c, s = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W]])
        return pts @ np.array([[c, s], [-s, c]]) + [self.x, self.y]

    def corners3d(self):
        base = self.footprint()
        bottom = np.column_stack([base, np.zeros(4)])
        top = np.column_stack([base, np.full(4, self.height)])
        return bottom, top

    def truth(self):
        return {
            "name": self.name,
            "center_cm": (self.x, self.y),
            "length_cm": max(self.length, self.width),
            "width_cm": min(self.length, self.width),
            "height_cm": self.height,
            "yaw_deg": self.yaw,
        }


class Scene:
    """Table + ArUco markers + boxes, rendered through a Camera."""

    def __init__(self, camera, marker_span=(32.0, 24.0), marker_cm=MARKER_CM, boxes=None):
        self.cam = camera
        self.marker_cm = marker_cm
        sx, sy = marker_span[0] / 2.0, marker_span[1] / 2.0
        # Markers sit at the corners, centred on the arm base at the origin.
        self.marker_centres = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]])
        self.boxes = list(boxes or [])
        d = cv2.aruco.getPredefinedDictionary(DICT)
        self._tiles = [cv2.aruco.generateImageMarker(d, i, 240) for i in range(4)]

    def _draw_marker(self, img, idx):
        half = self.marker_cm / 2.0
        c = self.marker_centres[idx]
        # ArUco corner order is TL, TR, BR, BL in a y-up world.
        world = np.array([[-half, half], [half, half], [half, -half], [-half, -half]]) + c
        dst = self.cam.project(np.column_stack([world, np.zeros(4)])).astype(np.float32)
        tile = self._tiles[idx]
        n = tile.shape[0]
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), M,
                                     (self.cam.w, self.cam.h), borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), M,
                                   (self.cam.w, self.cam.h))
        img[mask > 0] = warped[mask > 0]

    def _draw_box(self, img, box):
        bottom, top = box.corners3d()
        cam_c = self.cam.centre_world()
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            quad = np.array([bottom[i], bottom[j], top[j], top[i]])
            faces.append((quad, 0.72))                      # side faces, shaded darker
        faces.append((top, 1.0))                            # top face, brightest
        # Painter's algorithm: draw the furthest face first so nearer ones cover it.
        faces.sort(key=lambda f: -np.linalg.norm(f[0].mean(axis=0) - cam_c))
        for quad, shade in faces:
            pts = self.cam.project(quad).astype(np.int32)
            col = tuple(int(np.clip(v * shade, 0, 255)) for v in box.colour)
            cv2.fillConvexPoly(img, pts, col, lineType=cv2.LINE_AA)
            cv2.polylines(img, [pts], True, (40, 40, 40), 1, cv2.LINE_AA)

    def render(self):
        img = np.full((self.cam.h, self.cam.w, 3), BG_COLOUR, np.uint8)
        # Table surface: a generous quad around the markers so the plane reads as a surface.
        pad = 14.0
        ext = self.marker_centres * (1.0 + pad / np.abs(self.marker_centres).max())
        table = self.cam.project(np.column_stack([ext, np.zeros(4)])).astype(np.int32)
        cv2.fillConvexPoly(img, table, TABLE_COLOUR, lineType=cv2.LINE_AA)
        for i in range(4):
            self._draw_marker(img, i)
        # Far boxes first, so a nearer box correctly occludes one behind it.
        for i in self._depth_order(near_first=False):
            self._draw_box(img, self.boxes[i])
        return img

    def truth(self):
        return [b.truth() for b in self.boxes]

    def advance(self, t):
        """Step every moving object to time t (seconds)."""
        for b in self.boxes:
            b.advance(t)

    def _depth_order(self, near_first=True):
        """Boxes ordered by how close they get to the camera.

        Ranked by NEAREST CORNER, not by base position: a tall box's top is far closer to
        the camera than its footprint suggests, and ranking by footprint alone makes a tall
        occluder sort as if it were behind what it actually hides. Still an approximation —
        it orders whole boxes rather than individual pixels — which is fine for boxes that
        do not interpenetrate.
        """
        cam_c = self.cam.centre_world()
        d = [min(np.linalg.norm(np.append(c, z) - cam_c)
                 for arr, z in ((b.footprint(), 0.0), (b.footprint(), b.height))
                 for c in arr)
             for b in self.boxes]
        idx = sorted(range(len(self.boxes)), key=lambda i: d[i])
        return idx if near_first else idx[::-1]

    def visibility(self):
        """Fraction of each box still visible once nearer boxes occlude it (0..1).

        Rasterises the silhouettes nearest-camera-first, so a box passing behind another
        genuinely loses pixels — which is what makes an occlusion test meaningful rather
        than assumed.
        """
        order = self._depth_order(near_first=True)
        occupied = np.zeros((self.cam.h, self.cam.w), np.uint8)
        sils = {n: p for n, p in self.silhouettes()}
        out = {}
        for i in order:                       # nearest first: it occludes everything behind
            b = self.boxes[i]
            own = np.zeros_like(occupied)
            cv2.fillConvexPoly(own, sils[b.name].astype(np.int32), 1)
            total = int(own.sum())
            if total == 0:
                out[b.name] = 0.0
                continue
            out[b.name] = float(((own == 1) & (occupied == 0)).sum()) / total
            occupied |= own
        return out

    def visible_silhouettes(self, min_visible=0.05):
        """Outlines of only the VISIBLE part of each object, as a real mask would be.

        silhouettes() returns each object's full geometric outline even where another
        object covers it. A segmentation model cannot do that — it labels visible pixels
        only — and the difference matters: sampling colour inside a full outline picked up
        the occluder's pixels and called a black cube "brown".
        """
        cam_c = self.cam.centre_world()
        occupied = np.zeros((self.cam.h, self.cam.w), np.uint8)
        sils = {n: p for n, p in self.silhouettes()}
        out = []
        for i in self._depth_order(near_first=True):     # nearest first, so it occludes
            b = self.boxes[i]
            own = np.zeros_like(occupied)
            cv2.fillConvexPoly(own, sils[b.name].astype(np.int32), 1)
            total = int(own.sum())
            visible = ((own == 1) & (occupied == 0)).astype(np.uint8)
            occupied |= own
            if total == 0 or visible.sum() < max(1, min_visible * total):
                continue
            cnts, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            biggest = max(cnts, key=cv2.contourArea)
            if len(biggest) >= 3:
                out.append((b.name, biggest.reshape(-1, 2).astype(np.float32)))
        return out

    def detections(self, min_visible=0.35):
        """Simulated detector output: an object too heavily occluded is simply missed.

        YOLOE cannot see flat-shaded synthetic boxes, so detection is modelled here rather
        than run for real. What this does test honestly is the pipeline's behaviour when
        detections drop out and come back — which is exactly what an occlusion causes.
        """
        vis = self.visibility()
        out = []
        for name, poly in self.silhouettes():
            if vis[name] < min_visible:
                continue
            x, y, w, h = cv2.boundingRect(poly.astype(np.float32))
            out.append({"name": name, "poly": poly, "visible": vis[name],
                        "xywh": (x + w / 2.0, y + h / 2.0, float(w), float(h)),
                        "conf": float(0.55 + 0.4 * vis[name])})
        return out

    def degraded_silhouettes(self, rng, bias_px=0.0, jitter_px=0.0, round_px=0.0):
        """Silhouettes degraded the way a real segmentation mask is.

        A neural mask is not the clean polygon the geometry has been tested on: its border
        wobbles by a pixel or two, sharp corners come out rounded, and the whole outline
        tends to sit slightly inside or outside the true object. This reproduces those three
        effects by rasterising, distorting, and re-extracting the contour — so the geometry
        can be scored against the kind of input it will actually be given.

        bias_px  : + dilates (over-segment), - erodes (under-segment)
        jitter_px: random per-pixel border roughness
        round_px : blur radius, which rounds corners like a low-resolution mask head
        """
        out = []
        for name, poly in self.silhouettes():
            m = np.zeros((self.cam.h, self.cam.w), np.uint8)
            cv2.fillConvexPoly(m, poly.astype(np.int32), 255)

            if jitter_px > 0:
                noise = rng.normal(0, jitter_px, m.shape).astype(np.float32)
                m = np.where(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
                             + noise * 40 > 127, 255, 0).astype(np.uint8)
            if round_px > 0:
                k = max(1, int(round_px) * 2 + 1)
                m = np.where(cv2.GaussianBlur(m, (k, k), 0) > 127, 255, 0).astype(np.uint8)
            if abs(bias_px) > 0.5:
                k = np.ones((int(abs(bias_px)) * 2 + 1,) * 2, np.uint8)
                m = cv2.dilate(m, k) if bias_px > 0 else cv2.erode(m, k)

            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            out.append((name, max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)))
        return out

    def silhouettes(self):
        """Each box's outline in image pixels — what a perfect segmentation would return.

        This is the union of the box's projected faces, i.e. top face AND visible sides,
        exactly the silhouette a real mask gives (and exactly what the footprint
        correction has to undo).
        """
        out = []
        for b in self.boxes:
            bottom, top = b.corners3d()
            pts = np.vstack([self.cam.project(bottom), self.cam.project(top)])
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            out.append((b.name, hull))
        return out

    def detection_result(self, track_ids=True):
        """A stand-in for a YOLO result, so rendered frames drive the real pipeline code.

        Using exact silhouettes isolates the GEOMETRY under test from detector noise: any
        error the pipeline reports here is its own, not the detector's.
        """
        # Visible-only outlines, because that is what a segmentation model produces. Using
        # full geometric outlines quietly made occlusion free, hiding a real failure mode.
        sils = self.visible_silhouettes()
        names = {i: n for i, (n, _) in enumerate(sils)}
        polys = [p for _, p in sils]
        n = len(polys)

        class _Ids:
            def __init__(self, v):
                self._v = v

            def int(self):
                return self

            def tolist(self):
                return self._v

        class _Vals:
            def __init__(self, v):
                self._v = v

            def tolist(self):
                return self._v

        boxes = type("B", (), {
            "cls": _Vals(list(range(n))),
            "conf": _Vals([0.95] * n),
            # Names are unique in the synthetic scene; visibility/depth ordering
            # can change each frame, so visible-list position is NOT an identity.
            "id": _Ids([next(i + 1 for i, box in enumerate(self.boxes) if box.name == name)
                         for name, _ in sils]) if track_ids else None,
        })()
        return type("R", (), {
            "names": names,
            "masks": type("M", (), {"xy": polys})(),
            "boxes": boxes,
        })()


def cluttered_scene(camera=None, moving=True, **cam_kwargs):
    """A deliberately awkward table: several colours, DUPLICATE colours, and movement.

    Two blue boxes of different sizes are the point — "pick the blue box" is genuinely
    ambiguous, so the system has to either disambiguate ("the biggest blue box") or say so,
    rather than quietly grabbing whichever it happened to see first.
    """
    cam = camera or Camera(**cam_kwargs)
    drift = (lambda t: (-11.0 + 2.2 * t, 6.5)) if moving else None
    boxes = [
        Box(7.0, -5.0, 6.0, 4.0, 3.0, yaw_deg=10, colour=(205, 90, 45),
            name="blue box"),                                   # big blue
        Box(-7.5, -6.0, 3.2, 3.2, 3.0, yaw_deg=-20, colour=(200, 80, 40),
            name="blue cube"),                                  # small blue — same colour
        Box(3.0, 8.0, 4.5, 4.5, 3.0, yaw_deg=35, colour=(45, 45, 205),
            name="red cube"),
        Box(-4.0, 1.5, 12.0, 8.0, 2.0, yaw_deg=-8, colour=(70, 170, 70),
            name="green book"),
        Box(10.0, 5.0, 3.0, 3.0, 3.0, yaw_deg=0, colour=(60, 215, 235),
            name="yellow cube"),
        Box(-11.0, 6.5, 3.5, 3.5, 3.0, yaw_deg=0, colour=(28, 28, 28),
            name="black cube", motion=drift),                   # the moving one
    ]
    return Scene(cam, boxes=boxes)


def default_scene(camera=None, **cam_kwargs):
    """The arm sits at the origin; four objects of known size are placed around it."""
    cam = camera or Camera(**cam_kwargs)
    boxes = [
        Box(6.5, -5.0, 3.0, 3.0, 3.0, yaw_deg=0, colour=(190, 190, 200), name="cube"),
        Box(-7.0, 4.5, 9.0, 5.5, 2.0, yaw_deg=25, colour=(120, 170, 210), name="phone"),
        Box(2.0, 8.5, 5.0, 5.0, 5.0, yaw_deg=40, colour=(150, 205, 160), name="tall cube"),
        Box(-6.0, -7.5, 13.0, 9.0, 1.5, yaw_deg=-15, colour=(205, 180, 140), name="book"),
    ]
    return Scene(cam, boxes=boxes)


if __name__ == "__main__":
    scene = default_scene()
    frame = scene.render()
    cv2.imwrite("sim_frame.png", frame)

    c, ids, _ = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICT),
        cv2.aruco.DetectorParameters()).detectMarkers(frame)
    print(f"rendered {frame.shape[1]}x{frame.shape[0]} -> sim_frame.png")
    print(f"markers the detector finds in the render: "
          f"{0 if ids is None else len(ids)} {sorted(ids.flatten().tolist()) if ids is not None else ''}")
    print("ground truth:")
    for t in scene.truth():
        print(f"   {t['name']:<10} {t['length_cm']:5.1f} x {t['width_cm']:4.1f} x "
              f"{t['height_cm']:4.1f} cm  at ({t['center_cm'][0]:5.1f}, {t['center_cm'][1]:5.1f})"
              f"  yaw {t['yaw_deg']:+.0f}")
