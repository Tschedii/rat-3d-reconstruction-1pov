#!/usr/bin/env python3
"""
Local server backing index.html — GUI 1: checkerboard calibration.

Runs the full pipeline (synced-frame extraction -> per-camera intrinsic
calibration -> extrinsic calibration -> rig visualization) in a background
thread so the frontend can poll progress without blocking.
"""

import json
import threading
import traceback
from datetime import datetime, date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import calibration_lib as cal

TOOL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOOL_DIR / "output"
WORK_DIR = TOOL_DIR / "work"
PORT = 8971

STATE_LOCK = threading.Lock()
STATE = {
    "status": "idle",     # idle | running | done | error
    "log": [],
    "result": None,       # {json_path, perspective_png, top_view_png}
    "error": None,
}


def log(msg):
    print(msg)
    with STATE_LOCK:
        STATE["log"].append(msg)


def run_pipeline(cfg):
    with STATE_LOCK:
        STATE["status"] = "running"
        STATE["log"] = []
        STATE["result"] = None
        STATE["error"] = None

    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        board_w = int(cfg["board_w"])
        board_h = int(cfg["board_h"])
        square_size = float(cfg["square_size"])
        if board_w < 2 or board_h < 2:
            raise ValueError(
                f"board_w/board_h must each be >= 2 (count INNER corners, not squares) -- got {board_w}x{board_h}"
            )
        if not (square_size > 0):
            raise ValueError(
                f"square_size must be a positive number in meters -- got {square_size!r}. A zero or blank "
                f"value collapses every checkerboard corner onto the same 3D point, which is what causes "
                f"OpenCV's cryptic 'initIntrinsicParams2D: matH0.size() == Size(3, 3)' assertion failure "
                f"deep inside calibrateCamera."
            )
        sensor_specs = {
            "focal_length_mm": cfg.get("focal_length_mm"),
            "sensor_width_mm": cfg.get("sensor_width_mm"),
            "sensor_height_mm": cfg.get("sensor_height_mm"),
        }
        reference_name = cfg["reference_camera"]
        cameras_cfg = cfg["cameras"]  # list of {name, video_path}

        video_paths = {c["name"]: c["video_path"] for c in cameras_cfg}

        log("=== STAGE 1: extracting diverse frames per camera (intrinsics) ===")
        intrinsics_by_cam = {}
        for c in cameras_cfg:
            name = c["name"]
            log(f"[{name}] extracting diverse frames from {c['video_path']}")
            frame_dir = WORK_DIR / "internal_parameter" / name
            cal.extract_diverse_frames(
                c["video_path"], frame_dir, board_w, board_h,
                target_frames=int(cfg.get("target_frames", 300)),
                min_pose_distance=float(cfg.get("min_pose_distance", 0.15)),
                min_trans_distance=float(cfg.get("min_trans_distance", 0.05)),
                sample_every_n=int(cfg.get("sample_every_n", 10)),
                log=log,
            )
            log(f"[{name}] calibrating intrinsics")
            intr = cal.calibrate_intrinsics(
                frame_dir, board_w, board_h, square_size,
                sensor_specs=sensor_specs,
                outlier_threshold_px=float(cfg.get("outlier_threshold_px", 0.8)),
                log=log,
            )
            intrinsics_by_cam[name] = intr

        log("=== STAGE 2: extracting synchronized frames (extrinsics) ===")
        synced_dir = WORK_DIR / "external_parameter"
        cal.extract_synced_frames(
            video_paths, synced_dir, board_w, board_h,
            sample_every_n=int(cfg.get("synced_sample_every_n", 5)),
            max_frames=int(cfg.get("synced_max_frames", 60)),
            log=log,
        )

        log("=== STAGE 3: extrinsic calibration ===")
        cameras_for_extrinsics = []
        for c in cameras_cfg:
            name = c["name"]
            cameras_for_extrinsics.append({
                "name": name,
                "frames_dir": synced_dir / name,
                "intrinsics": {"K": intrinsics_by_cam[name]["K"], "dist": intrinsics_by_cam[name]["dist"]},
            })
        summary = cal.calibrate_extrinsics(
            cameras_for_extrinsics, board_w, board_h, square_size, reference_name, log=log
        )

        log("=== STAGE 4: rig visualization ===")
        perspective_png, top_png = cal.plot_rig(summary, OUTPUT_DIR, log=log)

        date_str = cfg.get("output_name") or date.today().strftime("%Y_%m_%d")
        if not date_str.endswith(".json"):
            date_str += ".json"
        out_json = OUTPUT_DIR / date_str
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"  saved {out_json}")

        with STATE_LOCK:
            STATE["status"] = "done"
            STATE["result"] = {
                "json_path": str(out_json),
                "perspective_png": Path(perspective_png).name,
                "top_view_png": Path(top_png).name,
            }
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

    def _send_file(self, path: Path, content_type="application/octet-stream"):
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
                "board_w": 9, "board_h": 6, "square_size": 0.023,
                "focal_length_mm": 8.0, "sensor_width_mm": 6.6, "sensor_height_mm": 4.1,
                "target_frames": 300, "min_pose_distance": 0.15, "min_trans_distance": 0.05,
                "sample_every_n": 10, "synced_sample_every_n": 5, "synced_max_frames": 60,
                "outlier_threshold_px": 0.8,
                "output_name": date.today().strftime("%Y_%m_%d"),
                "camera_names": ["camera_001", "camera_002", "camera_003", "camera_004", "camera_005", "camera_006"],
                "reference_camera": "camera_004",
            })
            return
        if path.startswith("/files/"):
            self._send_file(OUTPUT_DIR / path[len("/files/"):], "image/png")
            return
        self.send_response(404); self._cors(); self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            cfg = json.loads(body)
            with STATE_LOCK:
                already_running = STATE["status"] == "running"
            if already_running:
                self._send_json({"ok": False, "error": "already running"}, status=409)
                return
            t = threading.Thread(target=run_pipeline, args=(cfg,), daemon=True)
            t.start()
            self._send_json({"ok": True})
            return
        self.send_response(404); self._cors(); self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"GUI 1 (camera calibration) running at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
