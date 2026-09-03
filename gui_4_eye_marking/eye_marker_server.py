#!/usr/bin/env python3
"""
Local server backing index.html — GUI 4: eye marking + FOV.

Generalized from the original eye_marker_server.py: instead of scanning one
fixed ROOT directory for frame_XXXX/mesh_preview.ply subfolders, the set of
frames to work on is set dynamically via POST /api/set_source, either a
single mesh path or a root folder + frame range (still following the
existing frame_NNNNNN/mesh_preview.ply convention used everywhere else in
this project).
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

TOOL_DIR = Path(__file__).resolve().parent
PORT = 8974

FRAME_RE = re.compile(r"^frame_(\d+)$")

STATE = {
    "frames": {},          # frame_id -> Path to mesh_preview.ply
    "object_label": "default",
}


def eye_positions_path():
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", STATE["object_label"] or "default")
    return TOOL_DIR / f"eye_positions_{safe}.json"


def load_json(path):
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    return {}


def resolve_single(mesh_path):
    p = Path(mesh_path)
    if p.is_file() and p.suffix == ".ply":
        frame_id = p.parent.name if FRAME_RE.match(p.parent.name) else p.stem
        return {frame_id: p}
    if p.is_dir():
        ply = p / "mesh_preview.ply"
        if ply.exists():
            frame_id = p.name if FRAME_RE.match(p.name) else "frame_0000"
            return {frame_id: ply}
    raise FileNotFoundError(f"No mesh_preview.ply found at or under {mesh_path}")


def resolve_range(root_dir, frame_start, frame_end):
    root = Path(root_dir)
    frames = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if not (frame_start <= num <= frame_end):
            continue
        ply = p / "mesh_preview.ply"
        if ply.exists():
            frames[p.name] = ply
    if not frames:
        raise FileNotFoundError(f"No frame_XXXX/mesh_preview.ply found in [{frame_start},{frame_end}] under {root_dir}")
    return frames


def sorted_frame_names():
    return sorted(STATE["frames"].keys(), key=lambda n: int(FRAME_RE.match(n).group(1)) if FRAME_RE.match(n) else 0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type="text/html"):
        if not path.exists():
            self.send_response(404); self._cors(); self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self._send_file(TOOL_DIR / "index.html", "text/html")
            return
        if path == "/api/frames":
            self._send_json(sorted_frame_names())
            return
        if path == "/api/eye_positions":
            self._send_json(load_json(eye_positions_path()))
            return
        if path == "/api/eye_positions_auto":
            self._send_json({})
            return

        m = re.match(r"^/(frame_\d+|frame_0000)/mesh_preview\.ply$", path)
        if m:
            frame_id = m.group(1)
            ply_path = STATE["frames"].get(frame_id)
            if not ply_path or not ply_path.exists():
                self.send_response(404); self._cors(); self.end_headers()
                return
            data = ply_path.read_bytes()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404); self._cors(); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/set_source":
            payload = json.loads(body)
            try:
                if payload["mode"] == "single":
                    STATE["frames"] = resolve_single(payload["mesh_path"])
                    STATE["object_label"] = payload.get("object_label") or "default"
                else:
                    STATE["frames"] = resolve_range(
                        payload["root_dir"], int(payload["frame_start"]), int(payload["frame_end"])
                    )
                    STATE["object_label"] = payload.get("object_label") or "default"
                self._send_json({"ok": True, "frames": sorted_frame_names()})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        if path == "/api/save_eye":
            payload = json.loads(body)
            frame = payload["frame"]
            eye_positions = load_json(eye_positions_path())
            eye_positions[frame] = {
                "eyeL": payload["eyeL"],
                "eyeR": payload["eyeR"],
                "headForward": payload["headForward"],
                "headRight": payload["headRight"],
                "headUp": payload["headUp"],
                "source": payload.get("source", "manual"),
            }
            with eye_positions_path().open("w") as f:
                json.dump(eye_positions, f, indent=2)
            self._send_json({"ok": True})
            return

        self.send_response(404); self._cors(); self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"GUI 4 (eye marking) running at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
