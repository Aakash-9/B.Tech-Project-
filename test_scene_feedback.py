"""Camera observations must retain identity and belong to the frame being processed."""
from dashboard_sources import SimSource
from simulator import Camera, Box, Scene


def main():
    a, b = Box(-8, 0, 2, 2, 2, name='a'), Box(8, 0, 2, 2, 2, name='b')
    scene = Scene(Camera(), boxes=[a, b])
    def ids(result):
        return {result.names[int(c)]: tid for c, tid in
                zip(result.boxes.cls.tolist(), result.boxes.id.tolist())}
    old = ids(scene.detection_result())
    a.x = 500
    assert ids(scene.detection_result())['b'] == old['b'], 'visibility must not renumber IDs'
    print('[PASS] IDs survive another object leaving the camera')
    src = SimSource()
    frame = src.grab()
    before = {o['label']: o['center_cm'] for o in src.process(frame)}
    src.scene.boxes[0].x += 5
    src.grab()
    after = {o['label']: o['center_cm'] for o in src.process(frame)}
    assert before == after, 'processing an older frame must use its own detections'
    print('[PASS] capture/perception use the same immutable frame snapshot')
    src.close()
    print('PASSED 2 / 2 scene feedback checks')


if __name__ == '__main__': main()
