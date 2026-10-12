"""The one and only boundary between this project and a physical arm.

Everything above this file — vision, measurement, command parsing, inverse kinematics,
the dashboard — computes JOINT ANGLES IN DEGREES and nothing else. This module is the
only place that knows whether those degrees reach a PyBullet body or a servo horn. Nothing
outside it may branch on "am I in simulation or not"; if that check ever appears elsewhere,
the boundary has leaked and sim-to-real transfer stops being a one-file change.

The contract, in the units the rest of the project already uses:

    move_to_joint_angles(base, shoulder, elbow, wrist)   degrees, ik_solver convention
    open_gripper() / close_gripper()
    read_joint_angles()  -> the four angles, or the last commanded ones
    settle(seconds)      -> wait for motion to finish

ik_solver.py's convention is the source of truth: base yaw about +Z, shoulder and elbow
pitch measured from the +X axis in the arm's vertical plane, wrist pitch relative to the
forearm. pybullet_sim.angles_to_joint_targets() already maps that to the URDF, and
ServoDriver maps the same degrees to PWM below.

ServoDriver is written and testable WITHOUT any hardware attached: constructing it with
bus=None gives a dry-run driver that records what it would have sent. That matters because
the wiring cannot be debugged at the same time as the kinematics — get the numbers right
first, on a bench, then plug it in.
"""
import time

CHANNELS = {"base_yaw": 0, "shoulder_pitch": 1, "elbow_pitch": 2, "wrist_pitch": 3,
            "gripper": 4}

# Per-joint calibration. A real servo is never the ideal 500-2500 us / 0-180 deg part its
# datasheet claims: horns mount a few degrees off, linkages add offsets, and cheap servos
# run several percent fast or slow. These are the knobs to turn during bring-up — measure
# the arm at a few known angles and adjust, rather than editing kinematics to compensate,
# which would silently diverge from the simulation.
SERVO_CAL = {
    # joint:          (us_at_min, us_at_max, deg_min, deg_max, invert)
    "base_yaw":       (500.0, 2500.0, -90.0,  90.0, False),
    "shoulder_pitch": (500.0, 2500.0,   0.0, 180.0, False),
    "elbow_pitch":    (500.0, 2500.0,   0.0, 180.0, False),
    "wrist_pitch":    (500.0, 2500.0, -90.0,  90.0, False),
}
GRIPPER_US = {"open": 1800.0, "closed": 1200.0}

# Refuse commands outside what the mechanism can physically do. In simulation an impossible
# angle is a wrong picture; on hardware it is a stalled servo drawing locked-rotor current
# into a plastic gear train, so this is enforced in the driver and not left to callers.
JOINT_LIMITS_DEG = {
    "base_yaw":       (-90.0, 90.0),
    "shoulder_pitch": (0.0, 180.0),
    "elbow_pitch":    (0.0, 180.0),
    "wrist_pitch":    (-90.0, 90.0),
}
ORDER = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")


class JointLimitError(ValueError):
    """A commanded angle is outside the mechanism's safe travel."""


def clamp_or_raise(name, deg, strict=True):
    lo, hi = JOINT_LIMITS_DEG[name]
    if lo <= deg <= hi:
        return float(deg)
    if strict:
        raise JointLimitError(f"{name}={deg:.1f} deg outside [{lo}, {hi}]")
    return float(min(hi, max(lo, deg)))


def joint_violations(angles_deg):
    """Which commanded joints fall outside the mechanism's travel. Empty means reachable."""
    out = []
    for name, deg in zip(ORDER, angles_deg):
        lo, hi = JOINT_LIMITS_DEG[name]
        if not (lo <= deg <= hi):
            out.append((name, float(deg), (lo, hi)))
    return out


def reachable(x_cm, y_cm, z_cm):
    """Can the PHYSICAL arm reach this point? Returns (ok, reason).

    Radial distance alone is not reachability, and assuming it was hid a real problem: the
    arm's link lengths happily cover a point behind the base, but a standard hobby servo
    gives 180 degrees of travel total, so base yaw is limited to +/-90 and anything behind
    the shoulder needs 153 degrees. In simulation that silently clamped and the arm moved
    somewhere else entirely (measured 117 mm from the requested point) while still
    reporting success. On hardware the same command is a servo driven into its end stop.
    So reachability is checked against the JOINTS, not just the distance.
    """
    from ik_solver import inverse_kinematics
    try:
        angles = inverse_kinematics(x_cm, y_cm, z_cm)
    except ValueError as exc:
        return False, str(exc)
    bad = joint_violations(angles)
    if bad:
        name, deg, (lo, hi) = bad[0]
        if name == "base_yaw":
            return False, (f"behind the arm: needs {deg:.0f} deg of base rotation, "
                           f"the servo reaches {lo:.0f} to {hi:.0f}")
        return False, f"{name} would need {deg:.0f} deg, outside {lo:.0f} to {hi:.0f}"
    return True, "reachable"


def angle_to_us(name, deg):
    """Joint angle in degrees -> servo pulse width in microseconds, via SERVO_CAL."""
    us_lo, us_hi, d_lo, d_hi, invert = SERVO_CAL[name]
    span = d_hi - d_lo
    frac = 0.0 if abs(span) < 1e-9 else (deg - d_lo) / span
    frac = min(1.0, max(0.0, frac))
    if invert:
        frac = 1.0 - frac
    return us_lo + frac * (us_hi - us_lo)


class ArmDriver:
    """Interface. Both implementations below must honour it exactly."""

    def move_to_joint_angles(self, base, shoulder, elbow, wrist):
        raise NotImplementedError

    def open_gripper(self):
        raise NotImplementedError

    def close_gripper(self):
        raise NotImplementedError

    def read_joint_angles(self):
        raise NotImplementedError

    def settle(self, seconds=0.0):
        pass

    def close(self):
        pass


class SimDriver(ArmDriver):
    """Drives the PyBullet arm already used by the simulation process."""

    name = "simulation"
    has_feedback = True

    def __init__(self, sim):
        self.sim = sim
        self._last = (0.0, 90.0, 90.0, 0.0)

    def move_to_joint_angles(self, base, shoulder, elbow, wrist):
        angles = tuple(clamp_or_raise(n, d, strict=False)
                       for n, d in zip(ORDER, (base, shoulder, elbow, wrist)))
        self.sim.drive_joint_angles(*angles)
        self._last = angles
        return angles

    def open_gripper(self):
        self.sim.release()

    def close_gripper(self, track_id=None):
        return self.sim.grasp(track_id)

    def read_joint_angles(self):
        return self.sim.joint_angles_deg()

    def settle(self, seconds=0.0):
        self.sim.step_for(seconds)


class ServoDriver(ArmDriver):
    """Drives real servos over a PCA9685. Importable and testable with no hardware.

    Pass bus=None (the default) for a dry run: every pulse that WOULD be sent is recorded
    in `.sent` instead, so the angle-to-PWM mapping and the limit checks can be unit-tested
    on a laptop. Pass a configured adafruit_pca9685 instance to actually move metal.
    """

    name = "servo"
    has_feedback = False          # no encoders: position is what we last commanded

    def __init__(self, bus=None, freq_hz=50.0, settle_s=0.45):
        self.bus = bus
        self.freq_hz = float(freq_hz)
        self.settle_s = float(settle_s)
        self.sent = []            # (channel, microseconds) in dry-run and in real use
        self._last = (0.0, 90.0, 90.0, 0.0)

    def _write_us(self, channel, us):
        self.sent.append((channel, float(us)))
        if self.bus is None:
            return                # dry run: recorded, not transmitted
        # PCA9685 is a 12-bit PWM generator: duty = pulse / period, scaled to 4096 counts.
        counts = int(round(us * 1e-6 * self.freq_hz * 4096))
        self.bus.set_pwm(channel, 0, min(4095, max(0, counts)))

    def move_to_joint_angles(self, base, shoulder, elbow, wrist):
        # strict=True here: on hardware an out-of-range command is a stalled servo, so it
        # must fail loudly rather than be quietly clamped into something that looks fine.
        angles = tuple(clamp_or_raise(n, d, strict=True)
                       for n, d in zip(ORDER, (base, shoulder, elbow, wrist)))
        for name, deg in zip(ORDER, angles):
            self._write_us(CHANNELS[name], angle_to_us(name, deg))
        self._last = angles
        return angles

    def open_gripper(self):
        self._write_us(CHANNELS["gripper"], GRIPPER_US["open"])

    def close_gripper(self, track_id=None):
        self._write_us(CHANNELS["gripper"], GRIPPER_US["closed"])
        return True               # open-loop: no contact sensing, so this is a request

    def read_joint_angles(self):
        # No encoders on hobby servos. Returning the last command is the honest answer:
        # it is what we asked for, NOT a measurement of where the arm actually is.
        return self._last

    def settle(self, seconds=None):
        time.sleep(self.settle_s if seconds is None else seconds)


def make_driver(kind="sim", sim=None, bus=None):
    """The only place the rest of the project chooses an arm."""
    if kind == "sim":
        if sim is None:
            raise ValueError("SimDriver needs the ArmSim instance")
        return SimDriver(sim)
    if kind == "servo":
        return ServoDriver(bus=bus)
    raise ValueError(f"unknown driver kind: {kind!r}")


if __name__ == "__main__":
    # Dry-run self-test: the whole hardware path exercised with no hardware present.
    d = ServoDriver(bus=None)
    mid = d.move_to_joint_angles(0.0, 90.0, 90.0, 0.0)
    assert mid == (0.0, 90.0, 90.0, 0.0)
    chans = [c for c, _ in d.sent]
    assert chans == [0, 1, 2, 3], chans
    # centre of each calibrated range must land mid-pulse
    assert abs(d.sent[0][1] - 1500.0) < 1e-6, d.sent[0]
    assert abs(d.sent[1][1] - 1500.0) < 1e-6, d.sent[1]

    d.sent.clear()
    d.move_to_joint_angles(-90.0, 0.0, 180.0, 90.0)
    assert abs(d.sent[0][1] - 500.0) < 1e-6
    assert abs(d.sent[2][1] - 2500.0) < 1e-6

    try:
        d.move_to_joint_angles(0.0, 200.0, 90.0, 0.0)
    except JointLimitError:
        pass
    else:
        raise AssertionError("an out-of-range shoulder angle must raise, not clamp")

    d.sent.clear()
    d.open_gripper()
    d.close_gripper()
    assert [c for c, _ in d.sent] == [4, 4]
    assert d.sent[0][1] > d.sent[1][1], "open should be a longer pulse than closed"

    # invert flag actually inverts
    SERVO_CAL["base_yaw"] = (500.0, 2500.0, -90.0, 90.0, True)
    assert abs(angle_to_us("base_yaw", -90.0) - 2500.0) < 1e-6
    SERVO_CAL["base_yaw"] = (500.0, 2500.0, -90.0, 90.0, False)

    assert d.read_joint_angles() == (-90.0, 0.0, 180.0, 90.0)
    print("OK - servo path verified with no hardware attached")
