"""Drive the real robot_arm.urdf with ik_solver.py's angles, under real physics.

Joint-angle mapping (derived from ik_solver.py's geometry, verified against forward_kinematics'
own worked example (12, 8, 5) cm before trusting it):

  base_yaw        = base_angle_deg               (straight through)
  shoulder_pitch  = shoulder_angle_deg            (straight through — shoulder pivots at the
                                                    origin in ik_solver's model, matching this
                                                    URDF's shoulder joint sitting at (0,0,0))
  elbow_pitch     = elbow_angle_deg - 180         (ik_solver's elbow angle is the INTERIOR
                                                    triangle angle, 180 deg = arm fully
                                                    extended = zero bend at this joint)
  wrist_pitch     = wrist_angle_deg               (straight through)

Run standalone for a self-test: solve IK for a known target, drive the URDF, then read back
PyBullet's own computed end-effector position and check it lands back on that target.
"""
import math
import time

import pybullet as p
import pybullet_data

from ik_solver import inverse_kinematics, L1, L2
from vision_measure import describe

JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")


def angles_to_joint_targets(base_deg, shoulder_deg, elbow_deg, wrist_deg):
    """ik_solver.py's (base, shoulder, elbow, wrist) degrees -> this URDF's joint radians."""
    return (math.radians(base_deg), math.radians(shoulder_deg),
            math.radians(elbow_deg - 180.0), math.radians(wrist_deg))


class ArmSim:
    def __init__(self, gui=True, urdf="robot_arm.urdf"):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # The table surface must be exactly z=0: that is where the camera's world plane is
        # and where measured objects rest. But ik_solver puts the shoulder pivot AT z=0, so
        # the upper arm's 1.1 cm-radius cylinder permanently intersects the table there —
        # measured at 6745 N of contact force, which dragged the shoulder joint ~25 degrees
        # off its commanded angle. The arm is position-controlled and has no business
        # colliding with the table, so exclude every arm link from the plane. Objects still
        # collide with the plane (they sit on it) and with the arm (so grasping works).
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(urdf, basePosition=[0, 0, 0], useFixedBase=True)
        for link in range(-1, p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.plane, link, -1, enableCollision=0)

        self.joint_index = {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name = info[1].decode("utf-8")
            if name in JOINTS:
                self.joint_index[name] = i
        missing = [j for j in JOINTS if j not in self.joint_index]
        if missing:
            raise RuntimeError(f"URDF is missing expected joints: {missing}")
        self.gripper_link = next(
            i for i in range(p.getNumJoints(self.robot))
            if p.getJointInfo(self.robot, i)[12].decode("utf-8") == "gripper_base")
        self.objects = {}      # track_id -> {"body": pybullet id, "half": half-extents}
        self.held = None       # constraint id while an object is gripped
        self.table = None      # workspace slab, sized from the calibration

    def set_workspace(self, length_cm, width_cm):
        """Draw the table the camera actually calibrated, instead of an endless grey plane.

        Visual only: the plane at z=0 already stops objects falling, and giving this slab
        collision geometry would just re-create the shoulder-vs-table contact problem above.
        """
        if getattr(self, "table", None) is not None:
            p.removeBody(self.table)
        half = [max(length_cm, 5.0) / 200.0, max(width_cm, 5.0) / 200.0, 0.004]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.82, 0.78, 0.70, 1.0])
        # +1.5 mm: with its top exactly at z=0 it z-fights the ground plane and vanishes.
        self.table = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                       basePosition=[0, 0, -half[2] + 0.0015])

    def go_to(self, x_cm, y_cm, z_cm, settle_steps=120):
        """Solve IK for a target in cm and drive the real URDF joints toward it."""
        base, shoulder, elbow, wrist = inverse_kinematics(x_cm, y_cm, z_cm)
        targets = angles_to_joint_targets(base, shoulder, elbow, wrist)
        for name, angle in zip(JOINTS, targets):
            p.setJointMotorControl2(self.robot, self.joint_index[name],
                                    p.POSITION_CONTROL, targetPosition=angle,
                                    force=40, positionGain=0.25, velocityGain=1.0)
        for _ in range(settle_steps):
            p.stepSimulation()
        return (base, shoulder, elbow, wrist)

    def tip_position_cm(self):
        # getLinkState()[0] is the link's CENTER OF MASS (its inertial frame), which is
        # offset from the joint origin by that link's <inertial origin> — using it here
        # instead of [4] (the URDF link/joint frame) silently added a few cm of error.
        state = p.getLinkState(self.robot, self.gripper_link, computeForwardKinematics=True)
        return tuple(v * 100.0 for v in state[4])

    # ---------------------------------------------------------------- scene objects
    def sync_objects(self, objects, skip=None):
        """Mirror the camera's measured objects into the physics scene, keyed by track ID.

        Sizes and colours come straight from the measurements, so what appears here is what
        the vision pipeline actually believes is on the table — if a box looks wrong, the
        measurement is wrong, which makes this a visual check on the perception too.

        `skip` is the track ID currently in the gripper. Its body is owned by the grasp
        constraint, so re-posing it here would fight the constraint every frame and rebuilding
        it would silently destroy the grasp.
        """
        seen = set()
        for o in objects:
            tid = o.get("track_id")
            if tid is None:
                continue
            seen.add(tid)
            if tid == skip:
                continue
            half = [max(o["length_cm"], 0.5) / 200.0,      # cm -> m, and half-extents
                    max(o["width_cm"], 0.5) / 200.0,
                    max(o["height_cm"], 0.5) / 200.0]
            x, y = o["center_cm"]
            pos = [x / 100.0, y / 100.0, half[2]]          # sitting ON the table
            quat = p.getQuaternionFromEuler([0, 0, math.radians(o.get("angle_deg", 0.0))])
            b, g, r = o.get("colour_bgr") or (160, 160, 160)
            rgba = [r / 255.0, g / 255.0, b / 255.0, 1.0]

            label = o.get("label_text") or f"#{tid} {describe(o)}"
            text_pos = [pos[0], pos[1], pos[2] + half[2] + 0.03]

            prev = self.objects.get(tid)
            if prev and prev["half"] == half:
                p.resetBasePositionAndOrientation(prev["body"], pos, quat)
                prev.update(label=label, text_pos=text_pos)
                if self.gui:
                    prev["text"] = p.addUserDebugText(
                        label, text_pos, textColorRGB=[0.1, 0.1, 0.1], textSize=1.1,
                        replaceItemUniqueId=prev["text"])
                continue
            if prev:                                        # size changed: rebuild the body
                p.removeBody(prev["body"])
                if self.gui:
                    p.removeUserDebugItem(prev["text"])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            body = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=pos,
                                     baseOrientation=quat)
            text = p.addUserDebugText(label, text_pos, textColorRGB=[0.1, 0.1, 0.1],
                                      textSize=1.1) if self.gui else -1
            self.objects[tid] = {"body": body, "half": half, "text": text,
                                 "label": label, "text_pos": text_pos}

        for tid in [t for t in self.objects if t not in seen and t != skip]:
            gone = self.objects.pop(tid)
            p.removeBody(gone["body"])
            if self.gui:
                p.removeUserDebugItem(gone["text"])

    def grasp(self, track_id, max_distance_cm=None):
        """Attach an object to the gripper with a fixed constraint (a standard PyBullet
        stand-in for closing fingers on it — the URDF's fingers are visual only)."""
        entry = self.objects.get(track_id)
        if entry is None:
            return False
        state = p.getLinkState(self.robot, self.gripper_link, computeForwardKinematics=True)
        gp, gq = state[4], state[5]
        op, oq = p.getBasePositionAndOrientation(entry["body"])
        if max_distance_cm is not None and math.dist(gp, op) * 100 > max_distance_cm:
            return False
        self.release()
        # Both constraint frames must describe the SAME world pose. The old
        # child offset had the opposite sign and jumped the body on attachment.
        oi, qi = p.invertTransform(op, oq)
        child_pos, child_orn = p.multiplyTransforms(oi, qi, gp, gq)
        pi, pq = p.invertTransform(state[0], state[1])  # Bullet constraint frame is the CoM
        parent_pos, parent_orn = p.multiplyTransforms(pi, pq, gp, gq)
        self.held = p.createConstraint(
            self.robot, self.gripper_link, entry["body"], -1, p.JOINT_FIXED,
            [0, 0, 0], parent_pos, child_pos,
            parentFrameOrientation=parent_orn, childFrameOrientation=child_orn)
        return True

    def release(self):
        if self.held is not None:
            p.removeConstraint(self.held)
            self.held = None

    def draw_reach_circles(self):
        for radius, colour in ((L1 + L2, [0, 0.6, 0]), (abs(L1 - L2), [0.7, 0, 0])):
            r = radius / 100.0
            n = 60
            pts = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.001)
                  for i in range(n + 1)]
            for a, b in zip(pts, pts[1:]):
                p.addUserDebugLine(a, b, colour, lineWidth=2)

    def mark_target(self, x_cm, y_cm, z_cm, colour=(1, 0, 0)):
        p.addUserDebugPoints([[x_cm / 100, y_cm / 100, z_cm / 100]], [colour], pointSize=12)

    def connected(self):
        """False once the physics server is gone — e.g. the user closed the GUI window."""
        try:
            return bool(p.isConnected(self.client))
        except p.error:
            return False

    def disconnect(self):
        # Closing the GUI window already tears the server down, so disconnecting again
        # raised "Not connected to physics server" and dumped a traceback on what is a
        # perfectly normal way to quit.
        if self.connected():
            p.disconnect(self.client)


def self_test():
    """Prove the joint mapping is correct before trusting any visual demo built on it."""
    sim = ArmSim(gui=False)
    cases = [(12.0, 8.0, 5.0), (-9.0, 15.0, 3.0), (18.0, 0.0, 1.0), (0.0, -14.0, 8.0)]
    print(f"{'target (cm)':>22}   {'pybullet tip (cm)':>22}   {'error':>8}")
    worst = 0.0
    for x, y, z in cases:
        sim.go_to(x, y, z)
        got = sim.tip_position_cm()
        err = math.dist((x, y, z), got)
        worst = max(worst, err)
        print(f"({x:6.1f},{y:6.1f},{z:5.1f})   ({got[0]:6.2f},{got[1]:6.2f},{got[2]:5.2f})"
              f"   {err * 10:6.2f} mm")
    sim.disconnect()
    print(f"\nworst error: {worst * 10:.2f} mm")
    if worst > 0.01:  # 1 mm — URDF joints are exact, so this should be near machine precision
        print("FAIL — joint-angle mapping does not match ik_solver.py's geometry")
        raise SystemExit(1)
    print("PASS — PyBullet's own forward kinematics matches ik_solver.py exactly")


if __name__ == "__main__":
    self_test()
