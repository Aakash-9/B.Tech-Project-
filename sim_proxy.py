"""Parent-side handle on the simulation process. Everything here is non-blocking.

The point is that no dashboard thread ever waits on PyBullet: commands are posted, frames
arrive on their own, and the arm's state is whatever the worker last reported.
"""
import multiprocessing as mp
import threading

from sim_worker import to_payload


def _entry(conn):
    import sim_worker
    sim_worker.main(conn)


class SimProxy:
    def __init__(self, on_event=None):
        ctx = mp.get_context("spawn")            # the only option on Windows
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_entry, args=(child,), daemon=True)
        self.proc.start()
        child.close()

        self.jpeg = None
        self.fps = 0.0
        self.held = None
        self.busy = False
        self.on_event = on_event or (lambda kind, arg: None)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            try:
                kind, arg = self.conn.recv()
            except (EOFError, OSError):
                return
            if kind == "frame":
                self.jpeg, self.fps = arg
            elif kind == "busy":
                self.busy = arg
            elif kind == "held":
                self.held = arg
            elif kind == "picked":
                ok, tid = arg
                self.held = tid
            self.on_event(kind, arg)

    def _send(self, kind, arg=None):
        try:
            self.conn.send((kind, arg))
        except (BrokenPipeError, OSError):
            pass

    def sync(self, objects):
        self._send("sync", to_payload(objects))

    def workspace(self, size):
        self._send("workspace", size)

    def reset(self, size):
        self.held = None
        self._send("reset", size)

    def release(self):
        self.held = None
        self._send("release")

    def pick(self, x, y, z, track_id):
        self.busy = True                  # locally, so the UI disables Send immediately
        self._send("pick", (x, y, z, track_id))

    def close(self):
        self._send("quit")
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
