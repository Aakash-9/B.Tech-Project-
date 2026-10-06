"""Analytical IK/FK for a 4-DOF arm (base yaw + shoulder + elbow + wrist pitch)."""
import math

L1 = 10.5  # shoulder-to-elbow length, cm
L2 = 10.0  # elbow-to-wrist length, cm


def inverse_kinematics(x, y, z, L1=L1, L2=L2):
    """Return (base, shoulder, elbow, wrist) angles in degrees for target (x, y, z) cm."""
    # Base rotates so the rest of the arm works in a single vertical plane.
    base = math.atan2(y, x)
    r = math.hypot(x, y)          # horizontal reach in that plane
    d = math.hypot(r, z)          # straight-line distance shoulder -> target

    if d > L1 + L2:
        raise ValueError(f"Target out of reach: distance {d:.1f}cm > {L1 + L2:.1f}cm")
    # Inner limit: with the elbow fully folded the hand cannot come closer than |L1-L2|.
    if d < abs(L1 - L2):
        raise ValueError(f"Target too close to the base: {d:.1f}cm < {abs(L1 - L2):.1f}cm")

    # Law of cosines: d^2 = L1^2 + L2^2 - 2*L1*L2*cos(pi - elbow)
    # => cos(elbow) = (L1^2 + L2^2 - d^2) / (2*L1*L2), elbow = interior joint angle.
    cos_elbow = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))  # clamp float rounding
    elbow = math.acos(cos_elbow)

    # Shoulder = angle to target above horizontal + law-of-cosines correction
    # for the triangle formed by L1, L2 and d.
    cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    shoulder = math.atan2(z, r) + math.acos(cos_alpha)

    # Wrist pitch keeps the end-effector level with the table.
    wrist = -(shoulder + (elbow - math.pi))

    return tuple(math.degrees(a) for a in (base, shoulder, elbow, wrist))


def forward_kinematics(base_angle, shoulder, r, z, L1=L1):
    """Return 3 (x, y, z) points — base, elbow, end-effector — for plotting.

    base_angle/shoulder in degrees; (r, z) is the target in the arm's vertical plane.
    """
    b = math.radians(base_angle)
    s = math.radians(shoulder)
    base_pt = (0.0, 0.0, 0.0)
    elbow_pt = (L1 * math.cos(s) * math.cos(b),
                L1 * math.cos(s) * math.sin(b),
                L1 * math.sin(s))
    end_pt = (r * math.cos(b), r * math.sin(b), z)
    return [base_pt, elbow_pt, end_pt]


if __name__ == "__main__":
    x, y, z = 12.0, 8.0, 5.0
    angles = inverse_kinematics(x, y, z)
    print(f"Target ({x}, {y}, {z}) cm")
    print("Angles (deg): base=%.1f shoulder=%.1f elbow=%.1f wrist=%.1f" % angles)
    pts = forward_kinematics(angles[0], angles[1], math.hypot(x, y), z)
    print("Arm points:", [tuple(round(v, 2) for v in p) for p in pts])
    # end-effector from FK must land on the target
    assert all(abs(a - b) < 1e-6 for a, b in zip(pts[2], (x, y, z))), "FK/IK mismatch"
    print("OK")
