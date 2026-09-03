#!/usr/bin/env python3
"""
Local server backing index.html — GUI 3: scene placement.

Generalized from the original env_marker_server.py: instead of hardcoded
ENV_DIR / EXTRA_RATS / a precomputed initial_rig.json, everything is driven
by a session_config.json (environment mesh path, calibration JSON path, rat
1/2 mesh + eye-position paths) set via POST /api/config, and the camera-rig
seed geometry is computed on the fly from whichever calibration JSON was
configured (build_initial_rig), so this works with any calibration output
from GUI 1 -- not just one fixed rig.
"""

import json
import mimetypes
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TOOL_DIR / "session_config.json"
STATE_PATH = TOOL_DIR / "scene_placements.json"
EXPORTS_DIR = TOOL_DIR / "exports"
PORT = 8973


def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            return json.load(f)
    return {}


# =============================================================================
# Rig geometry — table-aligned display frame, derived from the calibration
# JSON's own average viewing direction (same approach used for the rig plots
# in GUI 1's calibration_lib.py). Produces the same schema the three.js
# frontend already expects: {table_center, cameras:[{name,is_reference,
# position,forward,up,right}]}.
# =============================================================================

def build_initial_rig(calib_json_path):
    with open(calib_json_path) as f:
        data = json.load(f)

    cams = []
    for cam in data["cameras"]:
        R = np.array(cam["rotation_matrix"], dtype=np.float64)
        t = np.array(cam["translation_m"], dtype=np.float64)
        cam_pos = -R.T @ t
        cams.append({
            "name": cam["name"], "is_reference": cam.get("is_reference", False),
            "pos": cam_pos,
            "forward": R.T[:, 2],
            "up": -R.T[:, 1],
            "right": R.T[:, 0],
        })

    # table-aligned basis from the average viewing direction
    forwards = np.array([c["forward"] / np.linalg.norm(c["forward"]) for c in cams])
    avg_forward = forwards.mean(axis=0)
    avg_forward /= np.linalg.norm(avg_forward)

    up_disp = -avg_forward
    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up_disp, world_ref)) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])
    x_disp = np.cross(up_disp, world_ref)
    x_disp /= np.linalg.norm(x_disp)
    y_disp = np.cross(up_disp, x_disp)

    def to_display(v):
        return np.array([np.dot(v, x_disp), np.dot(v, y_disp), np.dot(v, up_disp)])

    # table center: least-squares intersection of all cameras' optical axes
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c["forward"] / np.linalg.norm(c["forward"])
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ c["pos"]
    try:
        table_center_raw = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        table_center_raw = np.mean([c["pos"] for c in cams], axis=0)

    rig = {
        "table_center": to_display(table_center_raw).tolist(),
        "cameras": [],
    }
    for c in cams:
        rig["cameras"].append({
            "name": c["name"],
            "is_reference": c["is_reference"],
            "position": to_display(c["pos"]).tolist(),
            "forward": to_display(c["forward"] / np.linalg.norm(c["forward"])).tolist(),
            "up": to_display(c["up"] / np.linalg.norm(c["up"])).tolist(),
            "right": to_display(c["right"] / np.linalg.norm(c["right"])).tolist(),
        })
    return rig


def resolve_url_to_path(url, config):
    path = urlparse(url).path
    if path.startswith("/env/"):
        return Path(config["env_mesh_path"]).parent / path[len("/env/"):]
    if path.startswith("/extra1/"):
        return Path(config["rat1_mesh_path"]).parent / path[len("/extra1/"):]
    if path.startswith("/extra2/"):
        return Path(config["rat2_mesh_path"]).parent / path[len("/extra2/"):]
    return None


def load_eye_entry(eyes_path):
    if not eyes_path:
        return None
    p = Path(eyes_path)
    if not p.exists():
        return None
    with p.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "eyeL" in data:
        return data
    if isinstance(data, dict) and data:
        return next(iter(data.values()))
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_file(self, path: Path, content_type=None):
        if not path.exists():
            self.send_response(404); self._cors(); self.end_headers()
            return
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self._send_file(TOOL_DIR / "index.html", "text/html")
            return
        if path == "/bundle.js":
            self._send_file(TOOL_DIR / "bundle.js", "application/javascript")
            return
        if path == "/api/config":
            self._send_json(load_config())
            return
        if path == "/api/initial_rig":
            config = load_config()
            if not config.get("calib_json_path"):
                self._send_json({"error": "not configured"}, status=400)
                return
            try:
                self._send_json(build_initial_rig(config["calib_json_path"]))
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return
        if path == "/api/state":
            if STATE_PATH.exists():
                self._send_file(STATE_PATH, "application/json")
            else:
                self._send_json({})
            return
        if path == "/api/extra_eyes":
            config = load_config()
            out = {}
            e1 = load_eye_entry(config.get("rat1_eyes_path"))
            if e1:
                out["extra1"] = e1
            e2 = load_eye_entry(config.get("rat2_eyes_path"))
            if e2:
                out["extra2"] = e2
            self._send_json(out)
            return

        config = load_config()
        if path.startswith("/env/") and config.get("env_mesh_path"):
            rel = path[len("/env/"):]
            self._send_file(Path(config["env_mesh_path"]).parent / rel)
            return
        if path.startswith("/extra1/") and config.get("rat1_mesh_path"):
            rel = path[len("/extra1/"):]
            self._send_file(Path(config["rat1_mesh_path"]).parent / rel)
            return
        if path.startswith("/extra2/") and config.get("rat2_mesh_path"):
            rel = path[len("/extra2/"):]
            self._send_file(Path(config["rat2_mesh_path"]).parent / rel)
            return

        self.send_response(404); self._cors(); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/config":
            payload = json.loads(body)
            with CONFIG_PATH.open("w") as f:
                json.dump(payload, f, indent=2)
            self._send_json({"ok": True})
            return

        if path == "/api/state":
            payload = json.loads(body)
            with STATE_PATH.open("w") as f:
                json.dump(payload, f, indent=2)
            self._send_json({"ok": True})
            return

        if path == "/api/export_obj_textured":
            config = load_config()
            payload = json.loads(body)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = EXPORTS_DIR / f"scene_{stamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            obj_bytes = payload["obj"].encode("utf-8")
            (out_dir / "scene.obj").write_bytes(obj_bytes)
            (out_dir / "scene.mtl").write_text(payload["mtl"])

            copied, missing = 0, []
            for tex in payload.get("textures", []):
                src = resolve_url_to_path(tex["url"], config)
                dst = out_dir / tex["fileName"]
                if src and src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    missing.append(tex["url"])

            self._send_json({
                "ok": True, "dir": str(out_dir), "obj_size": len(obj_bytes),
                "textures_copied": copied, "textures_missing": missing,
            })
            return

        self.send_response(404); self._cors(); self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"GUI 3 (scene placement) running at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
