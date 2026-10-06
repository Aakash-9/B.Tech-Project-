"""Assign a stable ID to each object across frames and remember its recent path.

Uses Ultralytics' built-in BoT-SORT tracker (model.track(persist=True)) for ID assignment —
it already does motion-predicted re-identification, so an object that is briefly hidden
behind another keeps its ID instead of being counted as new. This module only adds the
world-space trail buffer and the "lock onto one ID" logic pick_by_id needs.
"""
from collections import Counter, defaultdict, deque

import numpy as np

TRACKER = "botsort.yaml"   # ships with ultralytics; does short-term occlusion re-ID


class TrailTracker:
    """Keeps each track ID's recent world-space (x, y) positions for drawing a path.

    Also the ONLY place per-frame noise gets smoothed. A single detection is a single
    frame's segmentation mask run through the whole measurement pipeline, and any one of
    those frames can be a bad one — a motion-blurred edge, a marginal 0.2-confidence guess,
    a height estimate the geometry couldn't quite support. Reacting to a single bad frame
    (rejecting it outright, as the dashboard's height-hint guard does) stops the worst
    outcomes, but it does not stop the table from flickering between plausible-looking
    numbers every ~300ms, which reads as "buggy" even when no single number was ever
    actually wrong — a live system has to look confident, not just be technically correct
    frame by frame. So label and size get the same median-of-recent-samples treatment
    height already had, and the displayed value only moves when the EVIDENCE moves, not
    every time the detector's guess wobbles.
    """

    def __init__(self, max_len=30, max_missed_frames=15, locked_grace_frames=60,
                 height_samples=15, size_samples=8, label_votes=8):
        self.max_len = max_len
        self.max_missed = max_missed_frames
        # A locked target gets more patience than an ordinary trail — it must survive
        # a real occlusion (passing behind another object) — but not forever, or a
        # genuinely lost object (picked up, left the frame) freezes the arm on a ghost.
        self.locked_grace = locked_grace_frames
        self.trails = defaultdict(lambda: deque(maxlen=max_len))
        # Height can only be measured when the object shows a usable vertical edge, which
        # depends on where it sits relative to the camera — so it succeeds on some frames
        # and declines on others. Remembering the accepted estimates per tracked object
        # turns intermittent measurements into a stable one, instead of falling back to a
        # guessed height every time a single frame's geometry is unfavourable.
        self.heights = defaultdict(lambda: deque(maxlen=height_samples))
        self.sizes = defaultdict(lambda: deque(maxlen=size_samples))
        self.labels = defaultdict(lambda: deque(maxlen=label_votes))
        self.missed = defaultdict(int)
        self.locked_id = None

    def update(self, tracked_objects):
        """tracked_objects: list of measure_scene()-style dicts, each with a 'track_id'."""
        seen = set()
        for o in tracked_objects:
            tid = o.get("track_id")
            if tid is None:
                continue
            self.trails[tid].append(o["center_cm"])
            if o.get("height_measured"):
                self.heights[tid].append(float(o["height_cm"]))
            if o.get("label"):
                self.labels[tid].append(o["label"])
            self.missed[tid] = 0
            seen.add(tid)
        for tid in list(self.trails):
            if tid in seen:
                continue
            self.missed[tid] += 1
            limit = self.locked_grace if tid == self.locked_id else self.max_missed
            if self.missed[tid] > limit:
                del self.trails[tid]
                del self.missed[tid]
                self.heights.pop(tid, None)
                self.sizes.pop(tid, None)
                self.labels.pop(tid, None)
                if tid == self.locked_id:
                    self.locked_id = None    # genuinely lost — release the lock

    def height_of(self, track_id):
        """Consensus height for a tracked object, or None if it has never been measured.

        The median of past accepted estimates, so one odd frame cannot drag it.
        """
        samples = self.heights.get(track_id)
        return float(np.median(samples)) if samples else None

    def apply_remembered_heights(self, tracked_objects):
        """Fill in a remembered height for objects whose estimate declined this frame."""
        for o in tracked_objects:
            if o.get("height_measured"):
                continue
            known = self.height_of(o.get("track_id"))
            if known is not None:
                o["height_cm"] = known
                o["height_measured"] = True
                o["height_from_memory"] = True
        return tracked_objects

    def stabilize_labels(self, tracked_objects):
        """Replace this frame's label with the majority vote of recent frames.

        Called AFTER update() (so this frame's own vote already counts) and BEFORE
        anything keys off the label — a height hint, a colour default — so a track that
        is genuinely a bottle but got called "cup" for one uncertain frame doesn't also
        get the cup's height hint for that one frame.
        """
        for o in tracked_objects:
            votes = self.labels.get(o.get("track_id"))
            if votes:
                o["label"] = Counter(votes).most_common(1)[0][0]
        return tracked_objects

    def stabilize_sizes(self, tracked_objects):
        """Record this frame's final length/width and replace it with the running median.

        Called LAST, after any height-hint correction has already produced this frame's
        real answer — that answer both feeds the display AND becomes part of the history
        future frames smooth against, so a single bad frame's influence fades over the
        next few samples instead of vanishing (and reappearing) instantly.
        """
        for o in tracked_objects:
            tid = o.get("track_id")
            if tid is None:
                continue
            self.sizes[tid].append((float(o["length_cm"]), float(o["width_cm"])))
            samples = np.asarray(self.sizes[tid])
            o["length_cm"] = float(np.median(samples[:, 0]))
            o["width_cm"] = float(np.median(samples[:, 1]))
            o["area_cm2"] = o["length_cm"] * o["width_cm"]   # keep it consistent with L x W,
            # since the dashboard sorts the table by area and a smoothed size paired with a
            # stale, unsmoothed area would sort inconsistently with what's on screen
        return tracked_objects

    def trail_of(self, track_id):
        return np.asarray(self.trails.get(track_id, []), dtype=float)

    def lock(self, track_id):
        """Commit to following one specific object regardless of shape/label re-picks."""
        self.locked_id = track_id

    def unlock(self):
        self.locked_id = None

    def find_locked(self, tracked_objects):
        """Return the locked object this frame, or None if it isn't currently visible."""
        if self.locked_id is None:
            return None
        for o in tracked_objects:
            if o.get("track_id") == self.locked_id:
                return o
        return None


if __name__ == "__main__":
    # Trail buffer behaviour: accumulates, caps length, drops stale IDs, keeps a locked one.
    t = TrailTracker(max_len=3)
    for x in range(5):
        t.update([{"center_cm": (x, 0.0), "track_id": 1}, {"center_cm": (0.0, x), "track_id": 2}])
    assert len(t.trail_of(1)) == 3, "trail should cap at max_len"
    assert list(t.trail_of(1)[-1]) == [4.0, 0.0], "trail should keep the MOST RECENT points"

    t.lock(2)
    t.update([{"center_cm": (9.0, 9.0), "track_id": 1}])   # id 2 vanished this frame (occluded)
    assert 2 in t.trails, "a locked id's trail must survive one missed frame"
    assert t.find_locked([{"center_cm": (9, 9), "track_id": 1}]) is None, \
        "locked id not present this frame -> None, not a wrong object"
    assert t.find_locked([{"center_cm": (1, 1), "track_id": 2}])["center_cm"] == (1, 1)

    # survives a short gap (occlusion) up to max_missed...
    for _ in range(t.max_missed - 1):
        t.update([{"center_cm": (9.0, 9.0), "track_id": 1}])
    assert t.locked_id == 2, "locked id must survive an occlusion shorter than max_missed"
    # ...but a genuinely lost object eventually releases the lock, not frozen forever
    for _ in range(t.locked_grace):
        t.update([{"center_cm": (9.0, 9.0), "track_id": 1}])
    assert t.locked_id is None, "a locked id absent past locked_grace must auto-release"
    assert 2 not in t.trails

    t2 = TrailTracker(max_len=3, max_missed_frames=15)
    for _ in range(t2.max_missed + 2):
        t2.update([{"center_cm": (0.0, 0.0), "track_id": 1}])
    t2.trails[2] = deque([(0.0, 0.0)], maxlen=3)
    t2.missed[2] = 0
    for _ in range(t2.max_missed + 2):
        t2.update([{"center_cm": (0.0, 0.0), "track_id": 1}])
    assert 2 not in t2.trails, "unlocked id with no recent sightings should eventually drop"

    # Label flicker: a track called "bottle" on 7 of 8 frames and "cup" once (a momentary
    # low-confidence guess) must read as "bottle" throughout, not toggle on the odd frame.
    t3 = TrailTracker()
    labels = ["bottle"] * 4 + ["cup"] + ["bottle"] * 3
    for lbl in labels:
        objs = [{"center_cm": (0.0, 0.0), "track_id": 9, "label": lbl}]
        t3.update(objs)
        t3.stabilize_labels(objs)
        assert objs[0]["label"] == "bottle", \
            f"a single odd frame ({lbl!r}) should not flip the displayed label"

    # A track that GENUINELY changes identity (new object gets this ID after the old one
    # left) must not stay stuck parroting the old label forever — the vote has to catch up.
    t4 = TrailTracker(label_votes=4)
    for lbl in ["cup"] * 4 + ["bottle"] * 4:
        objs = [{"center_cm": (0.0, 0.0), "track_id": 5, "label": lbl}]
        t4.update(objs)
        t4.stabilize_labels(objs)
    assert objs[0]["label"] == "bottle", "a sustained label change must eventually win"

    # Size jitter: one wildly wrong frame among many consistent ones must not swing the
    # displayed size — this is what a "68 cm sliver flashing for one frame" looks like here.
    t5 = TrailTracker()
    sizes = [(6.0, 6.0)] * 4 + [(68.0, 3.0)] + [(6.0, 6.0)] * 3
    for length, width in sizes:
        objs = [{"track_id": 7, "length_cm": length, "width_cm": width}]
        t5.stabilize_sizes(objs)
        assert objs[0]["length_cm"] < 10.0, \
            f"one outlier frame ({length}x{width}) should not spike the displayed size"
    assert abs(objs[0]["length_cm"] - 6.0) < 0.5 and abs(objs[0]["area_cm2"] - 36.0) < 6.0, \
        "settles back to the true size, with area kept consistent with L x W"

    # A real, sustained size change (the object was swapped, or genuinely re-measured
    # better) must still be reflected, just not on the very first differing frame.
    t6 = TrailTracker(size_samples=5)
    for length, width in [(6.0, 6.0)] * 3 + [(12.0, 12.0)] * 6:
        objs = [{"track_id": 8, "length_cm": length, "width_cm": width}]
        t6.stabilize_sizes(objs)
    assert abs(objs[0]["length_cm"] - 12.0) < 0.5, "a sustained size change must settle in"

    print("OK")
