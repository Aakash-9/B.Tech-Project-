"""Production pick state-machine and worker transport checks, no test framework."""
import numpy as np
import pybullet as p

from pybullet_sim import ArmSim
from visual_servo import ServoPick, CONTROL_DT


def object_at(x=12., y=4.):
    return dict(track_id=7, length_cm=2, width_cm=2, height_cm=2,
                center_cm=(x, y), colour_bgr=(0, 180, 0))


def advance(sim):
    for _ in range(16): p.stepSimulation()


def pick_checks(sim):
    sim.sync_objects([object_at()])
    sim.go_to(12, 0, 8)
    picker = ServoPick(sim)
    assert picker.start(7, 1., 0.)
    assert not picker.start(7, 1., 0.), 'must reject overlapping pick requests'
    print('[PASS] duplicate pick rejected')
    for step in range(240):
        now = step * CONTROL_DT
        if step == 12:
            sim.sync_objects([object_at(11., 5.)])
        picker.tick(now, {7}, now)
        advance(sim)
        if picker.phase == 'holding': break
        assert picker.busy, picker.status
    assert picker.phase == 'holding', picker.status
    body = sim.objects[7]['body']
    z = p.getBasePositionAndOrientation(body)[0][2]
    assert z > .05, f'object must actually rise; got {z}'
    print('[PASS] target moved during approach, convergence gated grasp, object lifted')
    picker.cancel()
    assert picker.held is None and not picker.busy and sim.held is None
    print('[PASS] release clears constraint and controller state')

    sim.sync_objects([object_at()])
    assert picker.start(7, 1., 0.)
    picker.tick(.1, set(), .1)
    assert picker.busy and 'Paused' in picker.status
    picker.tick(2.2, set(), 2.2)
    assert not picker.busy and sim.held is None
    print('[PASS] missing target pauses then aborts without attachment')
    assert picker.start(7, 1., 0.)
    picker.tick(2., {7}, 0.)
    picker.tick(4.1, {7}, 0.)
    assert not picker.busy and sim.held is None
    print('[PASS] stale observation cannot authorize a grasp')
    assert picker.start(7, 1., 0.)
    picker.cancel('Cancelled by user')
    assert not picker.busy
    print('[PASS] cancel interrupts approach on its next tick')
    sim.sync_objects([object_at(25., 0.)])
    assert picker.start(7, 1., 0.)
    picker.tick(.1, {7}, .1)
    assert not picker.busy and 'reach' in picker.status
    print('[PASS] unreachable observed target refused')
    sim.sync_objects([object_at()])
    assert picker.start(7, 1., 0.)
    picker.tick(picker.phase_timeout + .1, {7}, picker.phase_timeout + .1)
    assert not picker.busy and sim.held is None
    print('[PASS] nonconvergent approach times out')
    assert not picker.start(999, 1., 0.)
    print('[PASS] absent target ID refused')
    picker.camera.hide()


def clutter_check(sim):
    from dashboard_sources import SimSource
    src = SimSource()
    objs = src.process(src.grab())
    sim.sync_objects(objs)
    sim.go_to(12, 0, 8)
    picker = ServoPick(sim)
    target = next(o for o in objs if o['label'] == 'red cube')
    picker.start(target['track_id'], target['height_cm']/2, 0.)
    for tick in range(240):
        now = tick*CONTROL_DT
        picker.tick(now, {o['track_id'] for o in objs}, now)
        advance(sim)
        if not picker.busy: break
    assert picker.phase == 'holding', picker.status
    picker.cancel()
    for marker in picker.camera.markers: p.removeBody(marker)
    src.close()
    print('[PASS] dashboard cluttered scene keeps feedback visible throughout pick')


def main():
    sim = ArmSim(gui=False)
    try:
        sim.sync_objects([dict(track_id=7, length_cm=2, width_cm=2, height_cm=2,
                               center_cm=(12, 4), colour_bgr=(0, 180, 0))])
        sim.go_to(12, 4, 4)
        body = sim.objects[7]['body']
        before = np.array(p.getBasePositionAndOrientation(body)[0])
        assert sim.grasp(7)
        for _ in range(16): p.stepSimulation()
        after = np.array(p.getBasePositionAndOrientation(body)[0])
        assert np.linalg.norm(after-before) < .002, 'attachment must preserve relative transform'
        print('[PASS] attachment does not teleport the object')
        sim.release()
        pick_checks(sim)
        clutter_check(sim)
        print('PASSED 11 / 11 integration checks')
    finally:
        sim.disconnect()


if __name__ == '__main__': main()
