"""Simulation-only RGB visual feedback with an explicit target-height constraint.

The camera renders cyan target and magenta tool fiducials above their reference
points. Their known offsets avoid mutual occlusion at convergence. ImageServo
receives only pixel observations, joints, camera calibration and assumed height.
It never receives the target body's Cartesian position or renderer depth buffer.
"""
import math

import cv2
import numpy as np
import pybullet as p

from ik_solver import L1, L2

# Baseline renderer sustained 10.8–14.8 fps; keep control at 15 Hz in its process.
CONTROL_DT = 1. / 15.
# Synthetic fiducial mounting dimensions, not measurements of real hardware (cm).
TOOL_OFFSET = 10.
# At +6 cm the forearm hid the target marker in the dashboard's red-cube
# approach (saved RGB regression). +18 cm clears that forearm in the test rig.
# This is virtual instrumentation, not a claim that ordinary objects have tags.
TARGET_OFFSET = 18.


def joint_pose(q):
    """Independent analytic TCP in cm; q uses the URDF's radian convention."""
    b, s, e = q[:3]
    radius = L1 * math.cos(s) + L2 * math.cos(s + e)
    return np.array([radius * math.cos(b), radius * math.sin(b),
                     L1 * math.sin(s) + L2 * math.sin(s + e)])


def project(P, xyz_cm):
    h = P @ np.r_[xyz_cm, 1.]
    if h[2] <= 0:
        return None
    return h[:2] / h[2]


def on_plane(P, uv, z):
    H = np.column_stack((P[:, 0], P[:, 1], P[:, 3] + z * P[:, 2]))
    h = np.linalg.solve(H, np.r_[uv, 1.])
    return np.r_[h[:2] / h[2], z]


class ImageServo:
    # Settings are validated by test_visual_servo's seeded noise/offset sweep.
    # 1 rad/s is below the URDF's 3 rad/s limit. Pixel and z tolerances leave
    # room under the requested 2 mm physical-error threshold at 400x300.
    max_speed = 1.0
    gain = 2.5
    damping = 1.0
    pixel_tol = 0.45
    # Loaded lift settles ~0.045 cm low under velocity control (clutter regression).
    # 0.06 cm accommodates that 0.45 mm residual within the 2 mm acceptance budget.
    height_tol_cm = 0.06
    z_weight = 10.0  # px/cm; same order as the rendered workspace's local scale

    def __init__(self, projection):
        self.P = np.asarray(projection)
        self.error_px = None

    def velocity(self, q, tool_px, target_px, target_z, goal_z):
        if tool_px is None or target_px is None:
            self.error_px = None
            return np.zeros(4), False
        target = on_plane(self.P, target_px, target_z + TARGET_OFFSET)
        target[2] = goal_z + TOOL_OFFSET
        wanted = project(self.P, target)
        tcp = joint_pose(q)
        observed_error = np.asarray(tool_px) - wanted
        self.error_px = float(np.linalg.norm(observed_error))
        e = np.r_[observed_error, self.z_weight * (tcp[2] - goal_z)]

        # Image Jacobian: differentiate the calibrated projection of the known
        # kinematic model. Feedback error itself comes from the rendered pixels.
        def features(angles):
            xyz = joint_pose(angles)
            return np.r_[project(self.P, xyz + [0, 0, TOOL_OFFSET]),
                         self.z_weight * xyz[2]]

        eps = 1e-5  # central difference step; checked against URDF FK in the test
        J = np.column_stack([(features(q + np.eye(4)[i] * eps)
                              - features(q - np.eye(4)[i] * eps)) / (2 * eps)
                             for i in range(3)])
        vel = -self.gain * J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(3), e)
        # Wrist keeps pitch level; include it when bounding ALL four speeds.
        wrist = -vel[1] - vel[2] - self.gain * (q[1] + q[2] + q[3])
        vel = np.r_[vel, wrist]
        vel *= min(1., self.max_speed / max(np.max(np.abs(vel)), 1e-9))
        # One tick cannot cross a joint limit. Boundaries come from the URDF.
        vel = np.clip(vel, (-math.pi - q) / CONTROL_DT, (math.pi - q) / CONTROL_DT)
        done = self.error_px < self.pixel_tol and abs(tcp[2] - goal_z) < self.height_tol_cm
        return vel, done


class FeedbackCamera:
    """Rendered RGB instrumentation. Physics truth is private to image generation."""
    width, height = 400, 300

    def __init__(self, sim, yaw_deg=0., pitch_deg=0.):
        self.sim = sim
        # Same field of view as the dashboard; fiducials must remain visible above
        # the workspace. Camera perturbations are exercised in the random trials.
        a = math.radians(yaw_deg)
        eye = np.array([0.46, -0.40, 0.55 + pitch_deg / 300.])
        eye[:2] = np.array([[math.cos(a), -math.sin(a)],
                           [math.sin(a), math.cos(a)]]) @ eye[:2]
        self.view = p.computeViewMatrix(eye, [0, 0, 0.055], [0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(46., self.width / self.height, 0.01, 3.)
        mvp = np.array(self.proj).reshape(4, 4).T @ np.array(self.view).reshape(4, 4).T
        self.projection = np.vstack(((mvp[0] + mvp[3]) * self.width / 2,
                                     (mvp[3] - mvp[1]) * self.height / 2, mvp[3]))
        self.projection[:, :3] /= 100.  # projection API uses centimetres
        self.markers = []
        for colour in ([1., 0., 1., 1.], [0., 1., 1., 1.]):
            # 3 mm-radius markers occupied only a few pixels; a noisy trial
            # declared convergence at 2.607 mm. A 6 mm radius supplies a larger
            # RGB patch for centroid estimation, without adding a depth sensor.
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.006, rgbaColor=colour)
            self.markers.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                                  basePosition=[0, 0, -10]))
        self.rgb = None

    @staticmethod
    def features(rgb):
        r, g, b = [rgb[:, :, i].astype(float) for i in range(3)]
        masks = [(r > 80) & (b > 80) & (g < r * .35) & (g < b * .35),
                 (g > 80) & (b > 80) & (r < g * .35) & (r < b * .35)]
        points = []
        for mask in masks:
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            valid = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]
            points.append(centers[max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])]
                          if valid else None)
        return tuple(points)

    def observe(self, target_body):
        # Simulator sensor construction only; these positions never leave the
        # observer. The controller gets RGB-derived centroids, not these values.
        tool = np.asarray(self.sim.tip_position_cm()) + [0, 0, TOOL_OFFSET]
        target = np.asarray(p.getBasePositionAndOrientation(target_body)[0]) * 100
        target += [0, 0, TARGET_OFFSET]
        for body, position in zip(self.markers, (tool, target)):
            p.resetBasePositionAndOrientation(body, position / 100, [0, 0, 0, 1])
        image = p.getCameraImage(self.width, self.height, self.view, self.proj,
                                 renderer=p.ER_TINY_RENDERER)
        self.rgb = np.asarray(image[2], np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        return self.features(self.rgb)

    def hide(self):
        for body in self.markers:
            p.resetBasePositionAndOrientation(body, [0, 0, -10], [0, 0, 0, 1])


class ServoPick:
    """One non-blocking feedback tick per call; no IK moves inside approach.

    State transitions use observed image residuals plus the explicit z constraint.
    Constraint grasping remains a simulation approximation. TCP is aimed 1 cm
    above the top face: the URDF's tool radius is 0.8 cm, leaving 2 mm clearance.
    """
    clearance_cm = 1.0
    approach_cm = 3.0
    lift_cm = 6.0
    # 3.2 Hz baseline inference means a sample every ~0.31 s. 1.5 s permits
    # four missed updates; after 2 s missing data abort rather than extrapolate.
    max_age = 1.5
    lost_timeout = 2.0
    phase_timeout = 12.0  # trials converge in <4 s; bounded margin for approach

    def __init__(self, sim):
        from pybullet_sim import JOINTS
        self.sim = sim
        self.ids = [sim.joint_index[n] for n in JOINTS]
        self.camera = FeedbackCamera(sim)
        self.servo = ImageServo(self.camera.projection)
        self.phase = 'idle'
        self.status = 'Looking at the table'
        self.held = None
        self.tid = None
        self.events = []
        self.stable = 0
        self.anchor = None
        self.missing_since = None

    @property
    def busy(self):
        return self.phase in ('approach', 'descend', 'lift')

    def hold_pose(self):
        q = [p.getJointState(self.sim.robot, i)[0] for i in self.ids]
        p.setJointMotorControlArray(self.sim.robot, self.ids, p.POSITION_CONTROL,
                                    targetPositions=q, forces=[40]*4)

    def cancel(self, reason='Released'):
        self.hold_pose()
        self.sim.release()
        self.held = self.tid = None
        self.phase = 'idle'
        self.status = reason
        self.camera.hide()
        self.events += [('held', None), ('busy', False), ('status', self.status)]

    def fail(self, reason):
        self.cancel(reason)
        self.events += [('error', reason), ('picked', (False, None))]

    def start(self, tid, target_z, now):
        if self.busy or self.held is not None:
            self.events.append(('error', 'Finish or release the current pick first.'))
            return False
        if tid not in self.sim.objects:
            self.events += [('error', 'Target is no longer visible.'), ('busy', False)]
            return False
        self.tid, self.target_z = tid, float(target_z)
        self.grasp_z = 2 * self.target_z + self.clearance_cm
        self.phase, self.phase_started = 'approach', now
        self.stable = 0
        self.missing_since = None
        self.status = 'Reaching for the selected object (simulated RGB feedback)'
        self.events += [('busy', True), ('status', self.status)]
        return True

    def tick(self, now, visible_ids, observed_at):
        if not self.busy:
            return
        if now - self.phase_started > self.phase_timeout:
            self.fail('Stopped: feedback did not converge before the time limit.')
            return
        entry = self.sim.objects.get(self.tid)
        valid = entry is not None and (self.phase == 'lift' or
                 (self.tid in visible_ids and 0 <= now - observed_at <= self.max_age))
        tool, target = self.camera.observe(entry['body']) if valid else (None, None)
        if tool is None or target is None:
            self.hold_pose()
            self.stable = 0
            if self.missing_since is None: self.missing_since = now
            self.status = 'Paused: waiting for a fresh view of the same target'
            if now - self.missing_since >= self.lost_timeout:
                self.fail('Stopped: target or tool observation was lost.')
            return
        self.missing_since = None
        goal_z = self.grasp_z + (self.approach_cm if self.phase == 'approach' else
                                  self.lift_cm if self.phase == 'lift' else 0)
        pixel = self.anchor if self.phase == 'lift' else target
        target_plane = self.target_z
        xy = on_plane(self.camera.projection, pixel, target_plane + TARGET_OFFSET)[:2]
        if np.linalg.norm(np.r_[xy, goal_z]) > L1 + L2:
            self.fail('Stopped: target or approach height is outside arm reach.')
            return
        q = np.array([p.getJointState(self.sim.robot, i)[0] for i in self.ids])
        velocity, done = self.servo.velocity(q, tool, pixel, target_plane, goal_z)
        p.setJointMotorControlArray(self.sim.robot, self.ids, p.VELOCITY_CONTROL,
                                    targetVelocities=velocity, forces=[40]*4)
        self.stable = self.stable + 1 if done else 0
        self.status = f'{self.phase.capitalize()}: correcting image error ({self.servo.error_px:.2f} px)'
        if self.stable < 3:
            return
        self.stable, self.phase_started = 0, now
        if self.phase == 'approach':
            self.phase = 'descend'
        elif self.phase == 'descend':
            # Image convergence first; distance is only a final simulation grasp
            # guard, never a Cartesian feedback signal to the controller.
            if not self.sim.grasp(self.tid, self.target_z + self.clearance_cm + .2):
                self.fail('Stopped: converged image did not satisfy the grasp distance check.')
                return
            self.held, self.anchor = self.tid, target.copy()
            self.phase = 'lift'
            self.events.append(('held', self.held))
        else:
            self.phase = 'holding'
            self.hold_pose()
            self.camera.hide()
            self.status = 'Picked it up (simulated constraint grasp)'
            self.events += [('picked', (True, self.held)), ('busy', False), ('status', self.status)]
