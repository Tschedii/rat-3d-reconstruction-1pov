#!/usr/bin/env python3
"""
Local server backing index.html — GUI 4: eye marking + FOV.

Generalized from the original eye_marker_server.py: instead of scanning one
fixed ROOT directory for frame_XXXX/mesh_preview.ply subfolders, the set of
frames to work on is set dynamically via POST /api/set_source, either a
single mesh path or a root folder + frame range. Each frame's mesh can be
either the original mesh_preview.ply convention or a mesh.obj (+ .mtl +
texture) as written by GUI 2's default "atlas" texture mode -- whichever
is found is preferred in that order... actually mesh.obj is tried first,
since it's GUI 2's current default output; mesh_preview.ply is the
fallback for the older convention.
"""

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

TOOL_DIR = Path(__file__).resolve().parent
PORT = 8974

FRAME_RE = re.compile(r"^frame_(\d+)$")
MESH_FILENAMES = ("mesh.obj", "mesh_preview.ply")  # preference order

STATE = {
    "frames": {},          # frame_id -> Path to mesh.obj or mesh_preview.ply
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
    if p.is_file() and p.suffix in (".obj", ".ply"):
        frame_id = p.parent.name if FRAME_RE.match(p.parent.name) else p.stem
        return {frame_id: p}
    if p.is_dir():
        for name in MESH_FILENAMES:
            mesh = p / name
            if mesh.exists():
                frame_id = p.name if FRAME_RE.match(p.name) else "frame_0000"
                return {frame_id: mesh}
    raise FileNotFoundError(f"No {' or '.join(MESH_FILENAMES)} found at or under {mesh_path}")


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
        for name in MESH_FILENAMES:
            mesh = p / name
            if mesh.exists():
                frames[p.name] = mesh
                break
    if not frames:
        raise FileNotFoundError(
            f"No frame_XXXX/({' or '.join(MESH_FILENAMES)}) found in [{frame_start},{frame_end}] under {root_dir}"
        )
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
            self._send_json([
                {"id": name, "filename": STATE["frames"][name].name}
                for name in sorted_frame_names()
            ])
            return
        if path == "/api/eye_positions":
            self._send_json(load_json(eye_positions_path()))
            return
        if path == "/api/eye_positions_auto":
            self._send_json({})
            return

        # Serves a frame's mesh file itself, and -- since GUI 2's "atlas"
        # texture mode writes a mesh.mtl + mesh_texture.png alongside
        # mesh.obj -- any other file living next to it (the .mtl and the
        # texture image the client discovers by parsing the .obj/.mtl
        # text), by filename, from that same directory.
        m = re.match(r"^/(frame_\d+|frame_0000)/([^/]+)$", path)
        if m:
            frame_id, filename = m.group(1), m.group(2)
            if ".." in filename:
                self.send_response(400); self._cors(); self.end_headers()
                return
            mesh_path = STATE["frames"].get(frame_id)
            if not mesh_path:
                self.send_response(404); self._cors(); self.end_headers()
                return
            file_path = mesh_path if filename == mesh_path.name else mesh_path.parent / filename
            if not file_path.exists():
                self.send_response(404); self._cors(); self.end_headers()
                return
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._send_file(file_path, content_type)
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
