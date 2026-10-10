"""Does the dashboard stay responsive, and does it let go when reality moves on?

Two complaints drove these checks, and both were real:

  1. The panels were laggy. The cause was one loop doing capture, detection, physics and
     rendering in sequence, so everything ran at the speed of the slowest stage.
  2. After picking an object, moving the real object did nothing — the display stayed
     frozen holding a thing that was no longer there. The cause was that perception was
     deliberately paused while `held` was set.

These tests fail if either regresses. They spawn the real simulation process, so they must
run under a __main__ guard.
"""
import sys
import threading
import time

import numpy as np

from dashboard_engine import GRASP_SLIP_CM, Engine

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


class SlowSource:
    """Stands in for the live camera: cheap to grab from, expensive to understand."""

    name, has_truth, H0 = "Slow fake", False, None

    def __init__(self, engine):
        self.engine = engine
        self.calls = 0

    def grab(self):
        time.sleep(0.005)                       # a fast camera
        return np.zeros((240, 320, 3), np.uint8)

    def process(self, frame):
        self.calls += 1
        time.sleep(0.40)                        # a slow detector, like the real one
        return []

    def workspace_cm(self):
        return (40.0, 30.0)

    def calibration_status(self):
        return {"ok": False, "text": "fake"}

    def truth_of(self, label):
        return None

    def close(self):
        pass


def wait_for(fn, timeout, poll=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(poll)
    return False


def main():
    print("\n--- a slow detector must not slow the video ---")
    e = Engine(source="sim")
    e.source.close()
    e.source = SlowSource(e)
    threading.Thread(target=e.capture_loop, daemon=True).start()
    threading.Thread(target=e.perceive_loop, daemon=True).start()
    time.sleep(6)
    cam, det = e.cam_rate.fps, e.det_rate.fps
    check("video keeps running while perception is slow", cam > 20,
          f"{cam:.0f} fps video vs {det:.1f} fps detection")
    check("the two loops run at genuinely different rates", cam > det * 5,
          f"{cam:.0f} vs {det:.1f}")
    check("the slow source really was being called", e.source.calls >= 10,
          f"{e.source.calls} calls")
    e.close()

    print("\n--- the simulation runs in its own process and keeps drawing ---")
    e = Engine(source="sim")
    threading.Thread(target=e.run_forever, daemon=True).start()
    check("simulation process is alive", e.sim.proc.is_alive())
    check("simulation sends frames", wait_for(lambda: e.sim.jpeg is not None, 20),
          f"{0 if e.sim.jpeg is None else len(e.sim.jpeg)} bytes")
    check("simulation keeps a steady frame rate", wait_for(lambda: e.sim.fps >= 5, 20),
          f"{e.sim.fps:.1f} fps")
    check("objects reached the simulation",
          wait_for(lambda: len(e.objects) >= 4, 20), f"{len(e.objects)} objects")

    print("\n--- a grasp survives a stationary object ---")
    e.command("pick the red cube")
    got = wait_for(lambda: e.held is not None and not e.busy, 45)
    check("the red cube was picked up", got,
          f"held={e.held}; status={e.sim.status}; log={[l['text'] for l in e.log][-3:]}")
    if got:
        held_id = e.held
        time.sleep(5)
        check("it is still held five seconds later", e.held == held_id,
              f"held={e.held}")

        print("\n--- but NOT one the operator moves away ---")
        box = next(b for b in e.source.scene.boxes if b.name == "red cube")
        box.x += GRASP_SLIP_CM + 6.0            # the operator picks it up and puts it down
        released = wait_for(lambda: e.held is None, 30)
        check("moving the real object releases the stale grasp", released,
              f"held={e.held}")
        check("and it said why", any("not in the gripper" in l["text"] for l in e.log),
              [l["text"] for l in e.log][-1])
    e.close()

    print("\n" + "=" * 58)
    print(f"PASSED {passed} / {passed + failed}")
    print("REAL-TIME BEHAVIOUR OK" if not failed else "REGRESSED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
