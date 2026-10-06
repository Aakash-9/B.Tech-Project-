"""Watch the real arm, in a real physics window, pick its way through a scene.

    venv/Scripts/python.exe pybullet_demo.py

Opens a PyBullet GUI window with the arm at the workspace centre, its reach circles drawn
on the floor, and four objects marked at real measured positions (from simulator.py's test
scene — the same numbers demo.py verified against ground truth). The arm visibly moves to
each one in turn, under real gravity and motor dynamics, then returns home.
"""
import math
import time

import pybullet as p

from ik_solver import L1, L2
from pybullet_sim import ArmSim
from simulator import default_scene

HOME = (14.0, 0.0, 8.0)   # a comfortable, clearly-reachable resting pose


def main():
    sim = ArmSim(gui=True)
    sim.draw_reach_circles()

    p.resetDebugVisualizerCamera(cameraDistance=0.55, cameraYaw=35, cameraPitch=-35,
                                 cameraTargetPosition=[0, 0, 0.05])

    scene = default_scene()
    objects = [t for t in scene.truth()]
    print(f"reach: {abs(L1 - L2):.1f} - {L1 + L2:.1f} cm\n")

    print("moving to a home pose...")
    sim.go_to(*HOME, settle_steps=200)
    time.sleep(1.0)

    for obj in objects:
        x, y = obj["center_cm"]
        z = obj["height_cm"] / 2.0
        d = math.hypot(x, y, z)
        reach = f"{d:.1f} cm from base"
        sim.mark_target(x, y, 0.0, colour=(1, 0.6, 0))       # footprint marker on the floor
        if d > L1 + L2 or d < abs(L1 - L2):
            print(f"{obj['name']:<10} at ({x:5.1f},{y:5.1f},{z:4.1f}) — "
                  f"OUT OF REACH ({reach})")
            continue
        print(f"{obj['name']:<10} at ({x:5.1f},{y:5.1f},{z:4.1f}) — moving there ({reach})")
        sim.mark_target(x, y, z, colour=(1, 0, 0))
        sim.go_to(x, y, z, settle_steps=200)
        time.sleep(1.2)

    print("\nreturning home...")
    sim.go_to(*HOME, settle_steps=200)
    time.sleep(1.0)

    print("done — close the PyBullet window, or press Ctrl+C here, to exit.")
    try:
        while sim.connected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except (KeyboardInterrupt, p.error):
        # p.error: the window can be closed between the connected() check and the step.
        pass
    sim.disconnect()
    print("closed.")


if __name__ == "__main__":
    main()
