"""Live calibration check: can this camera measure real centimetres right now?

Run this, watch the window, and it tells you the answer instead of asking you to trust it.

    venv\\Scripts\\python.exe calibration_check.py

Put a plain A4 sheet flat on the table in view. The script finds it, calibrates from it,
then VERIFIES the result two ways:

  1. Fit residual -- how far the sheet's own corners land from where 21.0 x 29.7 cm says
     they should. This catches a bad corner detection.
  2. Re-measuring the sheet through the finished calibration. This must come back 21.0 x
     29.7 cm. It is a weak check on its own (the sheet calibrated itself) but it catches
     the long/short side being assigned the wrong way round, which is the one failure that
     otherwise produces a confident, plausible-looking, wrong answer.

Then hold a ruler or any object whose size you know across the sheet area and read the
measured span off the window. That is the check that actually proves metric scale, because
the object played no part in the calibration.

Press q to quit, s to save the current frame.
"""
import sys

import cv2
import numpy as np

from aruco_workspace import camera_tilt_deg, make_detector, tilt_advice, MARKER_CM
from aruco_workspace import workspace_from_corners
from paper_calib import SHEET_CM, calibrate_from_sheet, residual_cm, sheet_world_points

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, y, colour=(240, 240, 240), scale=0.52):
    cv2.putText(img, text, (12, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), FONT, scale, colour, 1, cv2.LINE_AA)


def measure_span(H0, p0, p1):
    """Distance in cm between two image points, through the calibration."""
    pts = np.array([p0, p1], np.float32).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(pts, H0).reshape(-1, 2)
    return float(np.linalg.norm(w[1] - w[0]))


def main(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"camera {cam_index} could not be opened")
        return 1
    aruco = make_detector()
    picked = []          # two clicked points, for measuring a known distance

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((float(x), float(y)))
            del picked[:-2]

    cv2.namedWindow("calibration check")
    cv2.setMouseCallback("calibration check", on_mouse)
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        H0 = None
        source = None

        # Markers first, exactly as the dashboard prefers them.
        corners, ids, _ = aruco.detectMarkers(frame)
        if ids is not None and len(ids) >= 3:
            Hn, centres = workspace_from_corners(
                [corners[i] for i in np.argsort(ids.flatten())], MARKER_CM)
            if Hn is not None:
                H0, source = Hn, f"{len(ids)} ArUco markers"
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        sheet_px = None
        if H0 is None:
            H0s, W0s, Ds, sheet_px, size_cm = calibrate_from_sheet(frame)
            if H0s is not None:
                H0, source = H0s, f"A4 sheet ({size_cm[0]:.1f} x {size_cm[1]:.1f} cm)"
                cv2.polylines(vis, [np.round(sheet_px).astype(np.int32)], True,
                              (0, 220, 255), 2)
                for i, p in enumerate(sheet_px):
                    cv2.circle(vis, tuple(np.round(p).astype(int)), 4, (0, 90, 255), -1)

        if H0 is None:
            put(vis, "NO CALIBRATION - cannot measure centimetres", 28, (90, 90, 255), 0.62)
            put(vis, "Put a plain A4 sheet flat on the table, fully in view.", 54)
            put(vis, "It must be the brightest flat rectangle in the picture.", 76)
        else:
            put(vis, f"CALIBRATED from {source}", 28, (120, 240, 140), 0.62)
            y = 54
            if sheet_px is not None:
                world, _ = sheet_world_points(sheet_px, SHEET_CM)
                res = residual_cm(H0, sheet_px, world) * 10
                col = (120, 240, 140) if res < 2.0 else (90, 200, 255)
                put(vis, f"corner fit residual {res:.2f} mm "
                         f"({'good' if res < 2.0 else 'check the sheet is flat and unfolded'})",
                    y, col)
                y += 22
                # Re-measure the sheet's own edges through the finished calibration.
                e1 = measure_span(H0, sheet_px[0], sheet_px[1])
                e2 = measure_span(H0, sheet_px[1], sheet_px[2])
                put(vis, f"sheet re-measured: {e1:.1f} x {e2:.1f} cm "
                         f"(should be 29.7 x 21.0 or 21.0 x 29.7)", y)
                y += 22

            tilt = None
            if sheet_px is not None:
                _, W0s, Ds, _, _ = calibrate_from_sheet(frame)
                tilt = camera_tilt_deg(H0, Ds, frame.shape)
            if tilt is not None:
                good = 32.0 <= tilt <= 50.0
                put(vis, tilt_advice(tilt), y,
                    (120, 240, 140) if good else (90, 200, 255))
                y += 22

            if len(picked) == 2:
                span = measure_span(H0, picked[0], picked[1])
                cv2.line(vis, tuple(map(int, picked[0])), tuple(map(int, picked[1])),
                         (60, 255, 255), 2)
                put(vis, f">>> clicked span measures {span:.2f} cm <<<", y, (60, 255, 255),
                    0.62)
                y += 22
                put(vis, "compare that against a ruler: THIS is the real proof", y)
            else:
                put(vis, "click two points to measure the distance between them", y)

        cv2.imshow("calibration check", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite("calibration_check.png", vis)
            print("saved calibration_check.png")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
