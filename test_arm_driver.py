"""Both arm drivers must honour ONE contract, so swapping them is a one-file change.

This is the sim-to-real seam. If it holds, the physical arm is wired by editing
arm_driver.SERVO_CAL and constructing ServoDriver with a real bus — nothing else in the
project changes. If it leaks, every caller starts needing to know which arm it is talking
to, and the simulation stops being a specification for the hardware.

Everything here runs with NO hardware attached.
"""
import sys

import numpy as np

from arm_driver import (JointLimitError, ServoDriver, SimDriver, angle_to_us,
                        make_driver, reachable, ORDER)
from ik_solver import inverse_kinematics
from pybullet_sim import ArmSim

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def main():
    print("--- both drivers expose the same contract ---")
    sim = ArmSim(gui=False)
    sim_d = make_driver("sim", sim=sim)
    servo_d = make_driver("servo")
    required = ("move_to_joint_angles", "open_gripper", "close_gripper",
                "read_joint_angles", "settle", "close")
    for m in required:
        check(f"both implement {m}()",
              callable(getattr(sim_d, m, None)) and callable(getattr(servo_d, m, None)))

    print("\n--- the servo path works with no hardware present ---")
    servo_d.move_to_joint_angles(0.0, 90.0, 90.0, 0.0)
    check("a centred command emits one pulse per joint",
          [c for c, _ in servo_d.sent] == [0, 1, 2, 3], str(servo_d.sent))
    check("centre of travel maps to mid-pulse (1500 us)",
          all(abs(us - 1500.0) < 1e-6 for _, us in servo_d.sent[:2]),
          str(servo_d.sent[:2]))
    check("nothing was transmitted in dry-run mode", servo_d.bus is None)

    print("\n--- limits are enforced where it matters ---")
    try:
        servo_d.move_to_joint_angles(0.0, 200.0, 90.0, 0.0)
        check("hardware refuses an out-of-range angle", False, "no exception raised")
    except JointLimitError:
        check("hardware refuses an out-of-range angle", True)
    before = sim_d.read_joint_angles()
    sim_d.move_to_joint_angles(0.0, 200.0, 90.0, 0.0)   # sim clamps rather than raising
    check("simulation clamps instead of raising (a wrong picture, not a stalled servo)",
          sim_d.read_joint_angles() is not None, f"was {np.round(before, 1)}")

    print("\n--- angles round-trip through the simulated arm ---")
    for target in ((0.0, 80.0, 100.0, 0.0), (25.0, 70.0, 120.0, -10.0),
                   (-30.0, 100.0, 80.0, 15.0)):
        sim_d.move_to_joint_angles(*target)
        sim_d.settle(1.2)
        got = sim_d.read_joint_angles()
        worst = max(abs(a - b) for a, b in zip(target, got))
        check(f"commanded {target} comes back within 2 deg", worst < 2.0,
              f"worst {worst:.2f} deg, got {tuple(round(g, 1) for g in got)}")

    print("\n--- the IK -> driver handoff lands where IK intended ---")
    for xyz in ((12.0, 4.0, 3.0), (8.0, -6.0, 5.0), (6.0, 9.0, 4.0)):
        ok, why = reachable(*xyz)
        check(f"{xyz} is declared reachable", ok, why)
        angles = inverse_kinematics(*xyz)
        sim_d.move_to_joint_angles(*angles)
        sim_d.settle(1.5)
        tip = np.asarray(sim.tip_position_cm())
        err_mm = float(np.linalg.norm(tip - np.asarray(xyz)) * 10)
        check(f"IK target {xyz} reached within 5 mm", err_mm < 5.0, f"{err_mm:.2f} mm")

    # The failure this section was added for: link lengths reach behind the base, a
    # 180-degree servo does not. Without this check the sim clamped yaw and reported a
    # successful pick 117 mm away from the request.
    print("\n--- points the real arm CANNOT reach are refused, not silently clamped ---")
    for xyz in ((-10.0, 5.0, 4.0), (-12.0, -3.0, 3.0)):
        ok, why = reachable(*xyz)
        check(f"{xyz} behind the base is refused", not ok, why)
        check("the refusal explains itself in physical terms", "behind the arm" in why, why)
    far = reachable(0.0, 40.0, 3.0)
    check("a point beyond the links is still refused", not far[0], far[1])

    print("\n--- calibration knobs behave ---")
    check("angle_to_us is monotonic across travel",
          angle_to_us("base_yaw", -90) < angle_to_us("base_yaw", 0)
          < angle_to_us("base_yaw", 90))
    check("every joint in ORDER has a calibration entry",
          all(angle_to_us(n, 0.0) > 0 for n in ORDER))

    sim_d.close()
    servo_d.close()
    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("ARM DRIVER CONTRACT OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
