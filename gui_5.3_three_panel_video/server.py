#!/usr/bin/env python3
"""
Local server backing index.html: the three-panel video (render.py) behind a
form. Fill in the inputs, "Check inputs" reports which frames every input
covers, "Render video" renders in a background thread with progress, and the
finished video plays in the page.

Inputs (saved to session_config.json next to this file):
  calib_json_path     rig JSON from GUI 1 (projects the head into the camera)
  camera_video_path   the raw camera video, cam_N.mp4 (N = rig camera number)
  masks_root          folder with cam_N/masks_raw (optional: tints the rats)
  head_tracking_dir   a GUI 5.2 head tracking export (head_pose_world.csv), or a
                      two-rat run folder with rat1/ and rat2/ (both rats drawn)
  skull_video_path    GUI 5's skull video export (<recording>_skull_<a>-<b>_...mp4;
                      for two rats GUI 5.1's side-by-side "both" export)
  skull_first_frame   recording frame of the skull video's first frame
                      (default: <a> from its file name)
  frame_start/end     frames to render (default: every frame all inputs cover)
  output_path         the .mp4 to write (default: on the Desktop)
plus the look: height_px, trail_seconds, tint_alpha, floor_z_m,
orbit_deg_per_s, elevation_deg.
"""

import json
import re
import sys
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
try:
    import render  # noqa: E402
except ImportError as e:
    sys.exit(f"cannot import render.py ({e}); it needs numpy, opencv-python, Pillow and matplotlib.")

CONFIG_PATH = TOOL_DIR / "session_config.json"
PORT = 8981

JOB = {"running": False, "done": 0, "total": 0, "log": [], "error": None, "result": None}
LOCK = threading.Lock()


def load_config():
    saved = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    return {**render.DEFAULTS, **saved}


def clean(payload):
    """Form values -> config: numbers as numbers, blanks as None/''."""
    cfg = {}
    for k, default in render.DEFAULTS.items():
        v = payload.get(k, default)
        if isinstance(v, str):
            v = v.strip()
        if k in ("skull_first_frame", "frame_start", "frame_end"):
            cfg[k] = int(v) if v not in (None, "") else None
        elif isinstance(default, float):
            cfg[k] = float(v) if v not in (None, "") else default
        elif isinstance(default, int) and not isinstance(default, bool):
            cfg[k] = int(v) if v not in (None, "") else default
        else:
            cfg[k] = v or ""
    return cfg


def log(msg):
    with LOCK:
        JOB["log"].append(f"[{datetime.now():%H:%M:%S}] {msg}")
    print(msg, flush=True)


def run(cfg):
    try:
        def progress(d, n):
            with LOCK:
                JOB["done"], JOB["total"] = d, n
        result = render.render_video(cfg, progress=progress, log=log)
        log(f"done: {result['output_path']}")
        with LOCK:
            JOB["result"] = result
    except Exception as e:  # reported to the page, not swallowed
        traceback.print_exc()
        with LOCK:
            JOB["error"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video(self, path):
        """The finished video, with byte ranges so the browser can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                data = (TOOL_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/config":
                self._json(load_config())
            elif path == "/api/progress":
                with LOCK:
                    self._json(dict(JOB, log=JOB["log"][-40:]))
            elif path == "/api/video":
                with LOCK:
                    result = JOB["result"]
                p = Path(result["output_path"]) if result else None
                if not p or not p.exists():
                    self._json({"error": "no finished video yet"}, 404)
                else:
                    self._video(p)
            else:
                self._json({"error": f"not found: {path}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            cfg = clean(json.loads(body or b"{}"))
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
            if path == "/api/inspect":
                self._json({"ok": True, **render.inspect(cfg)})
            elif path == "/api/render":
                with LOCK:
                    if JOB["running"]:
                        self._json({"ok": False, "error": "a render is already running"}, 409)
                        return
                    JOB.update(running=True, done=0, total=0, log=[], error=None, result=None)
                threading.Thread(target=run, args=(cfg,), daemon=True).start()
                self._json({"ok": True})
            else:
                self._json({"error": f"not found: {path}"}, 404)
        except Exception as e:
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)


def main():
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on port {PORT} ({e}); is another server.py still running?")
    print(f"three-panel video running at http://127.0.0.1:{PORT}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
