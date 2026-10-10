"""Exercise the actual spawned worker, command queue and streaming during a pick."""
import time

from sim_proxy import SimProxy


def wait_for(fn, timeout=20):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if fn(): return True
        time.sleep(.02)
    return False


def main():
    events = []
    sim = SimProxy(lambda kind, arg: events.append((kind, arg)))
    obj = dict(track_id=41, length_cm=2., width_cm=2., height_cm=2.,
               center_cm=(12., 4.), colour_bgr=(0, 180, 0), label='box')
    try:
        assert wait_for(lambda: sim.jpeg is not None)
        sim.sync([obj])
        time.sleep(.3)
        start = time.perf_counter()
        assert sim.pick(12, 4, 1, 41)
        assert time.perf_counter() - start < .1, 'posting must not await physics'
        assert not sim.pick(12, 4, 1, 41)
        print('[PASS] pick posting is non-blocking; concurrent pick refused')
        frames, previous, moved, deadline = 0, sim.jpeg, False, time.perf_counter()+25
        while time.perf_counter() < deadline and sim.busy:
            if not moved and 'Approach:' in sim.status:
                obj['center_cm'] = (11., 5.)
                moved = True
            sim.sync([obj])
            if sim.jpeg is not previous:
                frames += 1; previous = sim.jpeg
            time.sleep(.05)
        assert moved and sim.held == 41 and not sim.busy, (sim.status, events[-5:])
        assert frames >= 8, f'stream must continue throughout approach; got {frames}'
        print(f'[PASS] worker follows target update, grasps and keeps streaming ({frames} frames)')
        sim.release()
        assert wait_for(lambda: sim.status == 'Released')
        sim.sync([obj])
        assert sim.pick(11, 5, 1, 41)
        sim.release()
        assert wait_for(lambda: not sim.busy and sim.held is None)
        print('[PASS] cancel is processed during motion')
        sim.sync([obj], observed_at=time.perf_counter()-10)
        assert sim.pick(11, 5, 1, 41)
        assert wait_for(lambda: not sim.busy and 'lost' in sim.status, timeout=8)
        assert sim.held is None
        print('[PASS] stale scene update stops worker without false grasp')
    finally:
        sim.close()
    assert not sim.proc.is_alive()
    assert not sim._sender_thread.is_alive() and not sim._reader_thread.is_alive()
    print('[PASS] worker and both transport threads shut down')
    print('PASSED 5 / 5 worker feedback checks')


if __name__ == '__main__': main()
