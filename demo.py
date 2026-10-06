"""One-command demo: run the whole pipeline and show measured vs. true, side by side.

    venv/Scripts/python.exe demo.py

Writes demo_result.png — the camera's view with every object's measured size printed on it,
next to the arm pose the kinematics solved for, plus a table of measured-vs-true numbers.
Because the scene is synthetic, the true answer is known exactly, so this is evidence the
measurements are right rather than a demo that merely looks convincing.
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aruco_workspace import area_size, height_gradient, make_detector, workspace_from_corners, MARKER_CM
from ik_solver import forward_kinematics, inverse_kinematics, L1, L2
from object_tracker import TrailTracker
from simulator import Camera, default_scene
from vision_measure import measure_scene, select_target

TILT_DEG = 20.0          # the measured sweet spot for this setup
FRAMES = 25              # a short "video", so heights can be measured across frames


def run():
    truth = {t["name"]: t for t in default_scene().truth()}
    detector = make_detector()
    trailer = TrailTracker()
    rng = np.random.default_rng(0)

    frame = H0 = W0 = D = centres = None
    objects = []
    for _ in range(FRAMES):
        # A real camera is never perfectly still, and that tiny variation is what lets a
        # height that one frame's geometry cannot support succeed on another. Rendering
        # the identical frame repeatedly would make every frame fail in exactly the
        # same way and misrepresent how this behaves on live video.
        scene = default_scene(camera=Camera(
            height_cm=50.0, tilt_deg=TILT_DEG + rng.normal(0, 1.2)))
        frame = scene.render()
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None or len(ids) < 3:
            continue
        H0, centres = workspace_from_corners(
            [corners[i] for i in np.argsort(ids.flatten())], MARKER_CM)
        W0, D = height_gradient(H0, frame.shape)
        _, objects = measure_scene(scene.detection_result(), H0, W0, D,
                                   max_footprint_cm2=1e9)
        trailer.update(objects)
        objects = trailer.apply_remembered_heights(objects)

    target = select_target(objects, size="smallest")
    return frame, H0, W0, D, centres, objects, truth, target, scene


def annotate(frame, objects, truth, H0, target):
    img = frame.copy()
    Wt = np.linalg.inv(H0)
    for o in objects:
        t = truth[o["label"]]
        px = cv2.perspectiveTransform(
            np.array([[o["center_cm"]]], np.float32), Wt)[0][0]
        x, y = int(px[0]), int(px[1])
        is_target = target is not None and o["label"] == target["label"]
        col = (0, 230, 255) if not is_target else (60, 255, 60)
        cv2.circle(img, (x, y), 5, col, -1)
        cv2.putText(img, f"{o['label']}", (x - 45, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
        cv2.putText(img, f"meas {o['length_cm']:.1f}x{o['width_cm']:.1f}x{o['height_cm']:.1f}",
                    (x - 45, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        cv2.putText(img, f"true {t['length_cm']:.1f}x{t['width_cm']:.1f}x{t['height_cm']:.1f}",
                    (x - 45, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
    return img


def main():
    frame, H0, W0, D, centres, objects, truth, target, scene = run()
    if H0 is None or not objects:
        print("Calibration or detection failed — nothing to show.")
        return

    aw, ah = area_size(centres)
    img = annotate(frame, objects, truth, H0, target)

    fig = plt.figure(figsize=(14, 6.4), dpi=110)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"What the camera sees (tilt {TILT_DEG:.0f}°)\n"
                  f"workspace measured {aw:.1f} x {ah:.1f} cm  (true 32.0 x 24.0)",
                  fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    reach = L1 + L2
    th = np.linspace(0, 2 * np.pi, 80)
    ax2.plot(reach * np.cos(th), reach * np.sin(th), 0, "g--", lw=1, label="arm reach")
    if centres is not None:
        ws = np.vstack([centres, centres[:1]])
        ax2.plot(ws[:, 0], ws[:, 1], 0, "b:", lw=1, label="workspace")
    for o in objects:
        c = o["contour_cm"]
        ax2.plot(c[:, 0], c[:, 1], o["height_cm"], "-", color="orange", lw=1.2)
    title = "no reachable target"
    if target is not None:
        wx, wy = target["center_cm"]
        gz = target["height_cm"] / 2
        ax2.scatter([wx], [wy], [gz], color="red", s=60, label="target")
        try:
            b, s, e, w = inverse_kinematics(wx, wy, gz)
            pts = forward_kinematics(b, s, float(np.hypot(wx, wy)), gz)
            xs, ys, zs = zip(*pts)
            ax2.plot(xs, ys, zs, "-o", color="steelblue", lw=3, ms=7, label="arm")
            title = (f"picking '{target['label']}'  "
                     f"{target['length_cm']:.1f}x{target['width_cm']:.1f}x"
                     f"{target['height_cm']:.1f} cm\n"
                     f"base {b:.0f}°  shoulder {s:.0f}°  elbow {e:.0f}°  "
                     f"wrist {w:.0f}°")
        except ValueError as exc:
            title = f"target unreachable: {exc}"
    ax2.set_xlim(-reach, reach); ax2.set_ylim(-reach, reach); ax2.set_zlim(0, reach)
    ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)"); ax2.set_zlabel("z (cm)")
    ax2.set_title(title, fontsize=10)
    ax2.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig("demo_result.png")
    print("wrote demo_result.png\n")

    print(f"workspace measured by the system: {aw:.2f} x {ah:.2f} cm   (true 32.00 x 24.00)")
    print(f"\n{'object':<11}{'measured (L x W x H cm)':>26}{'true':>22}"
          f"{'err L':>8}{'err W':>8}{'err H':>9}")
    for o in sorted(objects, key=lambda o: -o["area_cm2"]):
        t = truth[o["label"]]
        # A height that was never actually measured is an assumption, not a result, and
        # printing an error against it would be meaningless.
        h_err = (f"{abs(o['height_cm'] - t['height_cm']) * 10:7.1f}mm"
                 if o["height_measured"] else "  assumed")
        mark = "" if o["height_measured"] else "~"
        print(f"{o['label']:<11}"
              f"{o['length_cm']:8.2f} x{o['width_cm']:6.2f} x{mark}{o['height_cm']:5.2f}"
              f"{t['length_cm']:10.2f} x{t['width_cm']:5.2f} x{t['height_cm']:5.2f}"
              f"{abs(o['length_cm'] - t['length_cm']) * 10:7.1f}mm"
              f"{abs(o['width_cm'] - t['width_cm']) * 10:7.1f}mm{h_err}")
    n_meas = sum(1 for o in objects if o["height_measured"])
    print(f"\nheights actually measured: {n_meas}/{len(objects)} "
          f"(a '~' marks an assumed fallback, not a measurement)")

    print("\nqueries answered from the measurements themselves:")
    for q in ({"size": "smallest"}, {"size": "biggest"}, {"shape": "square"}):
        p = select_target(objects, **q)
        print(f"   {str(q):<26} -> {p['label'] if p else 'nothing'}")


if __name__ == "__main__":
    main()
