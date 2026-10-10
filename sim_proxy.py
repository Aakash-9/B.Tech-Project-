"""Parent-side handle on the simulation process. Everything here is non-blocking.

The point is that no dashboard thread ever waits on PyBullet: commands are posted, frames
arrive on their own, and the arm's state is whatever the worker last reported.
"""
import multiprocessing as mp
import queue
import threading
import time

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
        self.status = 'Looking at the table'
        self._closed = threading.Event()
        self._outbox = queue.Queue()
        self._scene_lock = threading.Lock()
        self._latest_scene = None
        self._wake = threading.Event()
        self._pick_lock = threading.Lock()
        self.on_event = on_event or (lambda kind, arg: None)
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._sender_thread = threading.Thread(target=self._sender, daemon=True)
        self._reader_thread.start()
        self._sender_thread.start()

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
            elif kind == 'status':
                self.status = arg
            self.on_event(kind, arg)

    def _sender(self):
        """The only pipe writer. A full pipe never blocks Flask or capture."""
        while not self._closed.is_set():
            self._wake.wait(.1)
            self._wake.clear()
            with self._scene_lock:
                scene, self._latest_scene = self._latest_scene, None
            try:
                if scene is not None:
                    self.conn.send(('sync', scene))
                while not self._outbox.empty():
                    msg = self._outbox.get_nowait()
                    self.conn.send(msg)
                    if msg[0] == 'quit': return
            except (BrokenPipeError, OSError, ValueError):
                return

    def _send(self, kind, arg=None):
        if not self._closed.is_set():
            self._outbox.put_nowait((kind, arg))
            self._wake.set()

    def sync(self, objects, observed_at=None):
        with self._scene_lock:
            self._latest_scene = (to_payload(objects), time.perf_counter()
                                  if observed_at is None else observed_at)
        self._wake.set()

    def workspace(self, size):
        self._send("workspace", size)

    def reset(self, size):
        self.held = None
        with self._scene_lock: self._latest_scene = None
        self._send("reset", size)

    def release(self):
        self.held = None
        self._send("release")

    def pick(self, x, y, z, track_id):
        with self._pick_lock:
            if self.busy or self.held is not None: return False
            self.busy = True
            self._send("pick", (x, y, z, track_id))
            return True

    def close(self):
        if self._closed.is_set(): return
        self._send("quit")
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=3)
        self._closed.set()
        self._wake.set()
        self.conn.close()
        self._sender_thread.join(timeout=1)
        self._reader_thread.join(timeout=1)
