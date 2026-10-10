"""RGB-feedback controller regression and seeded PyBullet accuracy trials.

Ground-truth Cartesian error is used ONLY for scoring, never by ImageServo.
Run with venv/Scripts/python.exe test_visual_servo.py.
"""
import math
from pathlib import Path

import numpy as np
import pybullet as p

from pybullet_sim import ArmSim, JOINTS
from visual_servo import FeedbackCamera, ImageServo, joint_pose, CONTROL_DT


def trials(sim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv

    folder = Path('audit_results/servo_trials')
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    ids = [sim.joint_index[n] for n in JOINTS]
    for trial in range(25):
        # Independent trial streams: changing convergence time in one trial must
        # not silently change all subsequent target/camera randomisations.
        rng = np.random.default_rng(20261010 + trial)
        if trial == 24:
            # Preserve the exact pre-fix 2.607 mm failure, including its noise.
            # These are the preceding trials' observed frame counts, used only
            # to restore the old shared RNG's state; no physics result is reused.
            rng = np.random.default_rng(20261010)
            for ticks in (27, 31, 29, 33, 30, 36, 32, 30, 30):
                rng.uniform(size=5)
                rng.choice([-1, 1], 3)
                rng.normal(size=(ticks-3, 4))
        radius, angle = rng.uniform(8., 16.), rng.uniform(-2.6, 2.6)
        target = np.array([radius * math.cos(angle), radius * math.sin(angle),
                           rng.uniform(1.5, 4.)])
        body = p.createMultiBody(baseMass=0, basePosition=target / 100.)
        camera = FeedbackCamera(sim, yaw_deg=rng.uniform(-20, 20), pitch_deg=rng.uniform(-5, 5))
        servo = ImageServo(camera.projection)
        # A biased one-shot IK estimate: 5 mm lateral and height errors. Open loop
        # must retain that error. The trial then starts 3 cm above this estimate.
        estimate = target + np.array([.5, -.5, .5]) * rng.choice([-1, 1], 3)
        sim.go_to(*estimate)
        open_error = np.linalg.norm(np.array(sim.tip_position_cm()) - target) * 10
        sim.go_to(*(estimate + [0, 0, 3]))
        rows, consecutive = [], 0
        for tick in range(180):
            if (trial % 2 or trial == 24) and tick == 15:
                target[:2] += [.6, -.6]  # 8.5 mm target shift DURING approach
                p.resetBasePositionAndOrientation(body, target / 100., [0, 0, 0, 1])
            q = np.array([p.getJointState(sim.robot, i)[0] for i in ids])
            tool, seen = camera.observe(body)
            if 5 <= tick < 8:
                seen = None  # repeatable short occlusion; zero velocity required
            else:
                if tool is not None: tool = tool + rng.normal(0, .15, 2)
                if seen is not None: seen = seen + rng.normal(0, .15, 2)
            v, done = servo.velocity(q, tool, seen, target[2], target[2])
            if seen is None:
                assert not v.any(), 'lost observation must zero every joint velocity'
            p.setJointMotorControlArray(sim.robot, ids, p.VELOCITY_CONTROL,
                                        targetVelocities=v, forces=[40] * 4)
            for _ in range(16):  # 240 Hz physics / 15 Hz feedback
                p.stepSimulation()
            truth = np.asarray(p.getBasePositionAndOrientation(body)[0]) * 100
            err = np.linalg.norm(np.asarray(sim.tip_position_cm()) - truth) * 10
            rows.append((tick * CONTROL_DT, err, servo.error_px))
            consecutive = consecutive + 1 if done else 0
            if consecutive >= 3: break
        p.setJointMotorControlArray(sim.robot, ids, p.VELOCITY_CONTROL,
                                    targetVelocities=[0]*4, forces=[40]*4)
        results.append((trial+1, open_error, rows[-1][1], consecutive >= 3, len(rows)))
        with (folder / f'trial_{trial+1:02}.csv').open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(('seconds', 'truth_error_mm', 'image_error_px'))
            writer.writerows(rows)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([r[0] for r in rows], [r[1] for r in rows], label='PyBullet 3D error')
        ax.axhline(2., color='red', linestyle='--', label='2 mm requirement')
        ax.set(xlabel='Simulation time (s)', ylabel='Error (mm)',
               title=f'Trial {trial+1}: RGB feedback, known target height')
        ax.legend(); fig.tight_layout(); fig.savefig(folder / f'trial_{trial+1:02}.png'); plt.close(fig)
        camera.hide()
        for marker in camera.markers: p.removeBody(marker)
        p.removeBody(body)
        print(f"[{'PASS' if consecutive >= 3 and err < 2 else 'FAIL'}] trial {trial+1:02}: "
              f"open-loop {open_error:.3f} mm -> feedback {err:.3f} mm; {len(rows)} ticks",flush=True)
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(('trial','open_loop_mm','closed_loop_mm','converged','ticks'))
        writer.writerows(results)
    assert all(r[1] > 2 for r in results), 'biased open-loop control must fail the 2 mm gate'
    assert all(r[2] < 2 and r[3] for r in results), 'all feedback trials must converge below 2 mm'
    print(f'PASSED {len(results)} / {len(results)} accuracy trials; worst {max(r[2] for r in results):.3f} mm')


def main():
    sim = ArmSim(gui=False)
    try:
        camera = FeedbackCamera(sim)
        target = np.array([12., 4., 2.])
        body = p.createMultiBody(baseMass=0, basePosition=target / 100.)
        sim.go_to(*(target + [0.5, -0.5, 3.]))
        q = np.array([p.getJointState(sim.robot, sim.joint_index[n])[0] for n in JOINTS])
        tool, observed = camera.observe(body)
        assert tool is not None and observed is not None, 'both RGB features must be visible'
        servo = ImageServo(camera.projection)
        v, arrived = servo.velocity(q, tool, observed, target[2], target[2])
        assert np.isfinite(v).all() and np.max(np.abs(v)) <= servo.max_speed
        assert not arrived, 'initial offset must require a correction'
        missing, arrived = servo.velocity(q, None, observed, target[2], target[2])
        assert np.all(missing == 0) and not arrived, 'missing feature must stop motion'
        print('[PASS] rendered RGB features, bounded correction, missing feature stops')
        assert np.linalg.norm(joint_pose(q) - np.array(sim.tip_position_cm())) < 0.001
        print('[PASS] independent analytic pose agrees with URDF frame')
        camera.hide()
        p.removeBody(body)
        for marker in camera.markers: p.removeBody(marker)
        trials(sim)
    finally:
        sim.disconnect()


if __name__ == '__main__':
    main()
