#!/usr/bin/env python3
"""Local server backing index.html — GUI 2: rat mesh reconstruction."""

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import reconstruction_lib as recon

TOOL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOOL_DIR / "output"
PORT = 8972

STATE_LOCK = threading.Lock()
STATE = {
    "status": "idle",   # idle | running | done | error
    "log": [],
    "results": [],       # [{frame_id, verts, faces, output_path}]
    "error": None,
    "total_frames": 0,
    "completed_frames": 0,
    "current_frame": None,
}


def log(msg):
    print(msg)
    with STATE_LOCK:
        STATE["log"].append(msg)


def run_pipeline(cfg):
    with STATE_LOCK:
        STATE["status"] = "running"
        STATE["log"] = []
        STATE["results"] = []
        STATE["error"] = None
        STATE["total_frames"] = 0
        STATE["completed_frames"] = 0
        STATE["current_frame"] = None

    try:
        cameras = recon.load_calibration(cfg["calibration_json"])
        log(f"loaded calibration: {len(cameras)} cameras")

        object_id = int(cfg["object_id"])
        mode = cfg["mode"]
        if mode == "single":
            selection = {"frame_id": cfg["frame_id"]}
        else:
            selection = {"start": int(cfg["start"]), "end": int(cfg["end"]), "step": int(cfg.get("step", 1))}

        frames = recon.find_frames(cfg["masks_root"], object_id, mode, selection)
        log(f"resolved {len(frames)} frame(s) to reconstruct")
        with STATE_LOCK:
            STATE["total_frames"] = len(frames)

        params = {
            "voxel_size": float(cfg["voxel_size"]),
            "min_cameras": int(cfg["min_cameras"]),
            "half_xy": float(cfg["half_xy"]),
            "depth_below": float(cfg["depth_below"]),
            "depth_above": float(cfg["depth_above"]),
            "color_mesh": bool(cfg.get("color_mesh", True)),
        }

        output_root = Path(cfg.get("output_root") or OUTPUT_DIR)
        obj_dir = output_root / f"object_{object_id}"

        for frame_id, mask_paths in frames:
            log(f"--- {frame_id} ---")
            with STATE_LOCK:
                STATE["current_frame"] = frame_id
            try:
                ordered = recon.match_masks_to_cameras(cameras, mask_paths)
                verts, faces, centre, vertex_colors = recon.reconstruct_frame(cameras, ordered, params, log=log)
                out_path = obj_dir / frame_id / "mesh.obj"
                recon.save_obj(out_path, verts, faces, colors=vertex_colors)
                log(f"    saved {out_path}  ({len(verts):,} verts, {len(faces):,} faces)")
                with STATE_LOCK:
                    STATE["results"].append({
                        "frame_id": frame_id, "verts": len(verts), "faces": len(faces),
                        "output_path": str(out_path),
                        "centre": centre.tolist(),
                    })
            except Exception as fe:
                log(f"    FAILED: {fe}")
                with STATE_LOCK:
                    STATE["results"].append({"frame_id": frame_id, "error": str(fe)})
            finally:
                with STATE_LOCK:
                    STATE["completed_frames"] += 1

        with STATE_LOCK:
            STATE["status"] = "done"
            STATE["current_frame"] = None
        log("=== DONE ===")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"ERROR: {e}\n{tb}")
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["error"] = str(e)


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
        if path == "/api/progress":
            with STATE_LOCK:
                self._send_json(dict(STATE))
            return
        if path == "/api/defaults":
            self._send_json({
                "voxel_size": 0.003, "min_cameras": 4,
                "half_xy": 0.20, "depth_below": 0.05, "depth_above": 0.20,
                "color_mesh": True,
                "output_root": str(OUTPUT_DIR),
            })
            return
        self.send_response(404); self._cors(); self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(length))
            with STATE_LOCK:
                already_running = STATE["status"] == "running"
            if already_running:
                self._send_json({"ok": False, "error": "already running"}, status=409)
                return
            threading.Thread(target=run_pipeline, args=(cfg,), daemon=True).start()
            self._send_json({"ok": True})
            return
        self.send_response(404); self._cors(); self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"GUI 2 (rat reconstruction) running at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
