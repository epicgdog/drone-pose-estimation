"""Web dashboard for main.py: camera, 2d + 3d pose, state and rc commands.

main.py runs the loop and calls LiveServer.publish() every frame.
Each frame goes to the page as a server-sent event: jpeg + 2d keypoints + latest 3d poses.
"""

import base64
import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
NODE_DIR = Path(__file__).parent / "node_modules"
# three.js from npm (npm install), served locally: the tello's wifi has no internet
VENDOR = {
    "/vendor/three.module.js": NODE_DIR / "three/build/three.module.js",
    "/vendor/OrbitControls.js": NODE_DIR / "three/examples/jsm/controls/OrbitControls.js",
}
JPEG_QUALITY = 70


class LiveState:
    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0  # bumps on every new frame
        self.frame = None  # dict sent to the page
        self.running = True

    def publish_frame(self, payload):
        with self.cond:
            self.frame = payload
            self.seq += 1
            self.cond.notify_all()


def frame_payload(bgr, kpts=None, scores=None, poses3d=None, **extra):
    # poses3d = {lifter name: (17, 3) pose in that lifter's skeleton, or None}
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    k = None
    if kpts is not None:
        k = [[float(x), float(y), float(s)] for (x, y), s in zip(kpts, scores)]
    p = {name: None if pose is None else np.asarray(pose).round(4).tolist()
         for name, pose in (poses3d or {}).items()}
    return {"img": base64.b64encode(jpg.tobytes()).decode("ascii"),
            "w": bgr.shape[1], "h": bgr.shape[0], "kpts2d": k, "poses3d": p, **extra}


class LiveServer:
    # dashboard server on a background thread. publish() each frame, close() at the end.
    def __init__(self, skeletons, port: int = 8000, open_browser: bool = False):
        self.state = LiveState()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.state, skeletons))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="web", daemon=True).start()
        url = f"http://127.0.0.1:{port}"  # not "localhost": it can resolve to ipv6, the server is ipv4
        logger.info("dashboard at %s", url)
        if open_browser:
            webbrowser.open(url)

    def publish(self, bgr, **kw):
        payload = frame_payload(bgr, **kw)
        if payload is not None:
            self.state.publish_frame(payload)

    def close(self):
        self.state.running = False
        with self.state.cond:
            self.state.cond.notify_all()
        self.server.shutdown()
        self.server.server_close()


def make_handler(state: LiveState, skeletons):
    lifters_meta = {
        name: {"skeleton": s.NAME, "bones": s.BONES,
               "left": sorted(s.JOINTS_LEFT), "right": sorted(s.JOINTS_RIGHT)}
        for name, s in skeletons.items()
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the console for the pipeline

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (WEB_DIR / "index.html").read_bytes()
                self._send(200, "text/html; charset=utf-8", body)
            elif self.path in VENDOR:
                if not VENDOR[self.path].exists():
                    return self._send(500, "text/plain", b"three.js missing, run npm install")
                self._send(200, "text/javascript; charset=utf-8", VENDOR[self.path].read_bytes())
            elif self.path == "/meta":
                self._json(200, {"lifters": lifters_meta})
            elif self.path == "/events":
                self._events()
            else:
                self._send(404, "text/plain", b"not found")

        def _json(self, code, obj):
            self._send(code, "application/json", json.dumps(obj).encode())

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while state.running:
                    with state.cond:
                        state.cond.wait_for(lambda: state.seq != seen or not state.running,
                                            timeout=1.0)
                        if state.seq == seen:
                            continue
                        seen, payload = state.seq, state.frame
                    if payload is None:
                        continue  # page connected before the first frame exists
                    self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # page closed

    return Handler
