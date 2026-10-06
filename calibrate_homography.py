"""Two-plane one-shot calibration: pixels -> real cm, with metric height correction.

Plane 1: reference sheet flat on the table (Z = 0).
Plane 2 (optional): the same sheet raised onto a box of known height REF_HEIGHT_CM.

Why two planes: the world->image homography is exactly linear in height
(H(h) = K[r1 r2 t + h*r3]), so two calibrated planes give the mapping at ANY height.
Without it, an object 3 cm tall is measured through the ground plane and comes out
several percent too large — that magnification is the dominant error for a top-down camera.
"""
import cv2
import numpy as np

from vision_measure import fit_two_plane_calibration

# Reference sheet size in cm, and its 4 corners in world coordinates, in CLICK ORDER.
# Default: A4 (21.0 x 29.7 cm) centred on the arm base, so (0,0) is the middle of the sheet.
REF_W, REF_L = 21.0, 29.7
world_points = np.array([[-REF_W / 2, -REF_L / 2], [REF_W / 2, -REF_L / 2],
                         [REF_W / 2, REF_L / 2], [-REF_W / 2, REF_L / 2]], dtype=np.float32)

REF_HEIGHT_CM = 5.0  # measured height of the box you rest the sheet on for plane 2

clicked = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append([x, y])
        print(f"  point {len(clicked)}/4: pixel ({x}, {y})")


def collect_4_points(cap, banner):
    """Show the live feed until 4 corners are clicked. Returns them, or None if quit."""
    clicked.clear()
    print(banner)
    while len(clicked) < 4:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            return None
        for px, py in clicked:
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(frame, banner[:60], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == ord("s"):
            return "skip"
    return np.array(clicked, dtype=np.float32)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found (index 0). Check the webcam connection.")
    raise SystemExit(1)

cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)

print(f"PLANE 1 of 2 — lay the {REF_W}x{REF_L}cm sheet FLAT on the table.")
pts0 = collect_4_points(cap, "Plane 1: click 4 sheet corners (same order as world_points)")

if pts0 is None or isinstance(pts0, str):
    cap.release()
    cv2.destroyAllWindows()
    print("Calibration cancelled.")
    raise SystemExit(1)

H0, _, _ = fit_two_plane_calibration(pts0, None, world_points, REF_HEIGHT_CM)
np.save("homography_matrix.npy", H0)
print("\nSaved homography_matrix.npy (table plane):")
print(H0)

print(f"\nPLANE 2 of 2 — rest the SAME sheet on a box exactly {REF_HEIGHT_CM} cm tall,")
print("directly above the same spot. Then click its 4 corners again.")
print("Press 's' to skip (height correction stays off).")
pts1 = collect_4_points(cap, f"Plane 2: sheet raised {REF_HEIGHT_CM}cm - click 4 corners ('s'=skip)")

cap.release()
cv2.destroyAllWindows()

if pts1 is None:
    print("Plane 2 cancelled — height correction not saved.")
elif isinstance(pts1, str):
    print("Plane 2 skipped — height correction not saved.")
else:
    _, W0, D = fit_two_plane_calibration(pts0, pts1, world_points, REF_HEIGHT_CM)
    np.savez("plane_calib.npz", W0=W0, D=D, ref_height=REF_HEIGHT_CM)
    print("\nSaved plane_calib.npz — metric height correction is now ON.")
    print(f"Objects can be measured at their true height, not just on the table plane.")
