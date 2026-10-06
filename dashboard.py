"""Web dashboard: type a command, watch the arm do it, see what it measured.

    venv/Scripts/python.exe dashboard.py              simulated scene (accuracy provable)
    venv/Scripts/python.exe dashboard.py --camera     live webcam

then open http://127.0.0.1:5000

Left panel is what the camera sees, with every object outlined, colour-named and measured.
Right panel is the physics simulation the arm works in.

In simulated mode the table also shows each object's TRUE size, so accuracy is visible
rather than claimed. A real camera has no ground truth, so live mode shows detector
confidence and calibration status instead of inventing a "true" column.
"""
import argparse
import threading

from flask import Flask, Response, jsonify, render_template, request

from dashboard_engine import Engine

app = Flask(__name__)
engine = None          # built in __main__, once the source is known


def _stream(getter):
    """MJPEG: the browser shows a live image without any client-side video code.

    Sends a frame only when the producer has made a NEW one. The old version slept a flat
    50 ms, which both capped the stream at 20 fps and re-sent unchanged frames, spending
    bandwidth on a picture the browser already had.
    """
    import time
    last = None
    while True:
        frame = getter()
        if frame is not None and frame is not last:
            last = frame
            yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.004)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video/camera")
def video_camera():
    return Response(_stream(lambda: engine.camera_jpeg),
                    mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/video/sim")
def video_sim():
    return Response(_stream(lambda: engine.sim_jpeg),
                    mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/api/state")
def api_state():
    return jsonify(engine.state())


@app.route("/api/command", methods=["POST"])
def api_command():
    engine.command((request.get_json(silent=True) or {}).get("text", ""))
    return jsonify({"ok": True})


@app.route("/api/source", methods=["POST"])
def api_source():
    """Switch between the simulated scene and the live webcam from the page itself."""
    kind = (request.get_json(silent=True) or {}).get("kind", "sim")
    threading.Thread(target=engine.set_source, args=(kind,), daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="4-DOF arm vision dashboard")
    ap.add_argument("--camera", action="store_true",
                    help="use the live webcam instead of the simulated scene")
    ap.add_argument("--cam-index", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    engine = Engine(source="camera" if args.camera else "sim", cam_index=args.cam_index)
    threading.Thread(target=engine.run_forever, daemon=True).start()
    print(f"\n  {engine.source.name}  ->  http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
