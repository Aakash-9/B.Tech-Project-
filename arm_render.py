"""Draw the 4-DOF arm as an actual arm — solid links, joints and a gripper — not a line.

The kinematics only ever produce three points (base, elbow, end-effector). This turns those
into something that reads as a robot: a base plinth, cylindrical links with visible joints,
and a two-finger gripper that opens across the object's short axis. It also draws the real
workspace: the outer reach circle AND the inner one a 2-link arm can never reach inside.
"""
import numpy as np

BASE_COLOUR = "#37474f"
LINK1_COLOUR = "#4a90d9"
LINK2_COLOUR = "#6ba9e6"
JOINT_COLOUR = "#263238"
GRIP_COLOUR = "#c62828"


def _cylinder(p0, p1, radius, n=18):
    """Mesh for a cylinder between two 3D points, for ax.plot_surface."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return None
    axis = axis / length
    # Any vector not parallel to the axis gives us a perpendicular basis.
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)

    t = np.linspace(0, 1, 2)
    theta = np.linspace(0, 2 * np.pi, n)
    T, TH = np.meshgrid(t, theta)
    centre = p0[:, None, None] + axis[:, None, None] * (T * length)
    offset = radius * (np.cos(TH) * u[:, None, None] + np.sin(TH) * w[:, None, None])
    pts = centre + offset
    return pts[0], pts[1], pts[2]


def _sphere(centre, radius, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _add_cylinder(ax, p0, p1, radius, colour, alpha=1.0):
    mesh = _cylinder(p0, p1, radius)
    if mesh is not None:
        ax.plot_surface(*mesh, color=colour, alpha=alpha, linewidth=0,
                        antialiased=True, shade=True)


def draw_workspace(ax, reach_max, reach_min, z=0.0, workspace_centres=None):
    """The ring the arm can actually reach: outer limit, and the inner dead zone."""
    t = np.linspace(0, 2 * np.pi, 120)
    ax.plot(reach_max * np.cos(t), reach_max * np.sin(t), z, "--",
            color="#2e7d32", lw=1.6, label=f"max reach {reach_max:.1f} cm")
    if reach_min > 0.2:
        ax.plot(reach_min * np.cos(t), reach_min * np.sin(t), z, ":",
                color="#c62828", lw=1.4, label=f"min reach {reach_min:.1f} cm")
    # A faint filled disc makes the reachable area read as an area, not just an outline.
    rr = np.linspace(reach_min, reach_max, 2)
    R, T = np.meshgrid(rr, t)
    ax.plot_surface(R * np.cos(T), R * np.sin(T), np.full_like(R, z),
                    color="#2e7d32", alpha=0.06, linewidth=0, shade=False)
    if workspace_centres is not None and len(workspace_centres) >= 3:
        ws = np.vstack([workspace_centres, workspace_centres[:1]])
        ax.plot(ws[:, 0], ws[:, 1], z, ":", color="#1565c0", lw=1.2, label="camera workspace")


def draw_arm(ax, points, grasp_deg=0.0, gripper_open_cm=3.0, base_height=2.2):
    """Draw the arm through the three kinematic points (base, elbow, end-effector)."""
    base, elbow, tip = [np.asarray(p, float) for p in points]
    shoulder = base + np.array([0.0, 0.0, base_height])

    # Plinth and shoulder, then the two links, thinner as they go outward.
    _add_cylinder(ax, base, shoulder, 2.6, BASE_COLOUR)
    ax.plot_surface(*_sphere(shoulder, 1.5), color=JOINT_COLOUR, linewidth=0, shade=True)
    _add_cylinder(ax, shoulder, elbow, 1.15, LINK1_COLOUR)
    ax.plot_surface(*_sphere(elbow, 1.2), color=JOINT_COLOUR, linewidth=0, shade=True)
    _add_cylinder(ax, elbow, tip, 0.95, LINK2_COLOUR)
    ax.plot_surface(*_sphere(tip, 0.9), color=JOINT_COLOUR, linewidth=0, shade=True)

    # Two fingers, opening across the object's short axis (that is what grasp_deg means).
    a = np.radians(grasp_deg)
    across = np.array([np.cos(a), np.sin(a), 0.0]) * (gripper_open_cm / 2.0)
    for sign in (1, -1):
        root = tip + sign * across
        _add_cylinder(ax, root, root - np.array([0.0, 0.0, 1.6]), 0.3, GRIP_COLOUR)
        _add_cylinder(ax, tip + sign * across * 0.15, root, 0.28, GRIP_COLOUR)


def set_view(ax, reach, z_max=None):
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(0, z_max or reach)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ik_solver import forward_kinematics, inverse_kinematics, L1, L2

    target = (11.0, 6.0, 2.0)
    b, s, e, w = inverse_kinematics(*target)
    pts = forward_kinematics(b, s, float(np.hypot(target[0], target[1])), target[2])

    fig = plt.figure(figsize=(7, 6), dpi=110)
    ax = fig.add_subplot(projection="3d")
    draw_workspace(ax, L1 + L2, abs(L1 - L2), z=0.0)
    draw_arm(ax, pts, grasp_deg=30.0)
    ax.scatter(*[[v] for v in target], color="red", s=70, label="target")
    set_view(ax, L1 + L2)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"reaching ({target[0]}, {target[1]}, {target[2]}) cm\n"
                 f"base {b:.0f}°  shoulder {s:.0f}°  elbow {e:.0f}°")
    fig.tight_layout()
    fig.savefig("arm_preview.png")
    print("wrote arm_preview.png")
