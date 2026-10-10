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
            "head_min_cameras": int(cfg["head_min_cameras"]) if cfg.get("head_min_cameras") not in (None, "") else None,
            "head_zone_frac": float(cfg.get("head_zone_frac") or 0),
            "smooth_iterations": int(cfg.get("smooth_iterations", 10)),
            "texture_mode": cfg.get("texture_mode", "atlas"),
            "atlas_size": int(cfg.get("atlas_size", 1536)),
            "completion_mode": cfg.get("completion_mode") or "none",
            "template_path": (cfg.get("template_path") or "").strip() or None,
            "fit_iterations": int(cfg.get("fit_iterations") or 600),
            "fit_nodes": int(cfg.get("fit_nodes") or 200),
            "fit_sil_cameras": int(cfg["fit_sil_cameras"]) if cfg.get("fit_sil_cameras") not in (None, "") else None,
        }
        if params["completion_mode"] not in ("none", "template"):
            raise ValueError(f"completion_mode must be 'none' or 'template' (got {params['completion_mode']!r})")
        if params["completion_mode"] == "template":
            if params["template_path"] and not Path(params["template_path"]).is_file():
                raise ValueError(f"template mesh not found: {params['template_path']}")
            if params["fit_iterations"] < 1:
                raise ValueError(f"fit_iterations must be >= 1 (got {params['fit_iterations']})")
            if params["fit_nodes"] < 2:
                raise ValueError(f"fit_nodes must be >= 2 (got {params['fit_nodes']})")
            if params["fit_sil_cameras"] is not None and params["fit_sil_cameras"] < 1:
                raise ValueError(f"fit_sil_cameras must be >= 1 (got {params['fit_sil_cameras']})")
            try:
                import torch  # noqa: F401
            except ImportError:
                raise ValueError("template completion needs PyTorch in this environment: pip install torch")
        if params["voxel_size"] <= 0:
            raise ValueError(f"voxel_size must be > 0 (got {params['voxel_size']})")
        if params["half_xy"] <= 0:
            raise ValueError(f"half_xy must be > 0 (got {params['half_xy']})")
        if params["depth_below"] + params["depth_above"] <= 0:
            raise ValueError(
                "depth_below + depth_above must be > 0 (got depth_below="
                f"{params['depth_below']}, depth_above={params['depth_above']})"
            )
        if params["min_cameras"] < 1:
            raise ValueError(f"min_cameras must be >= 1 (got {params['min_cameras']})")
        if not (0 <= params["head_zone_frac"] <= 1):
            raise ValueError(f"head_zone_frac must be between 0 and 1 (got {params['head_zone_frac']})")
        if params["head_min_cameras"] is not None:
            if params["head_min_cameras"] < 1:
                raise ValueError(f"head_min_cameras must be >= 1 (got {params['head_min_cameras']})")
            if params["head_min_cameras"] > params["min_cameras"]:
                raise ValueError(
                    "head_min_cameras should be <= min_cameras (it's meant to relax the threshold "
                    f"for the head zone, got head_min_cameras={params['head_min_cameras']} > "
                    f"min_cameras={params['min_cameras']})"
                )

        output_root = Path(cfg.get("output_root") or OUTPUT_DIR)
        obj_dir = output_root / f"object_{object_id}"
        fit_context = {}

        for frame_id, mask_paths in frames:
            log(f"--- {frame_id} ---")
            with STATE_LOCK:
                STATE["current_frame"] = frame_id
            try:
                ordered = recon.match_masks_to_cameras(cameras, mask_paths)
                result = recon.reconstruct_frame(cameras, ordered, params, log=log, fit_context=fit_context)
                verts, faces = result["verts"], result["faces"]
                out_path = obj_dir / frame_id / "mesh.obj"
                if result["atlas"] is not None:
                    recon.save_textured_obj(out_path, verts, faces, result["corner_uv"], result["atlas"])
                else:
                    recon.save_obj(out_path, verts, faces, colors=result["vertex_colors"])
                log(f"    saved {out_path}  ({len(verts):,} verts, {len(faces):,} faces)")
                entry = {
                    "frame_id": frame_id, "verts": len(verts), "faces": len(faces),
                    "output_path": str(out_path),
                    "centre": result["centre"].tolist(),
                }
                if result["hull_verts"] is not None:
                    hull_path = obj_dir / frame_id / "mesh_hull.obj"
                    recon.save_obj(hull_path, result["hull_verts"], result["hull_faces"])
                    log(f"    saved raw visual hull alongside: {hull_path.name}  "
                        f"({len(result['hull_verts']):,} verts)")
                    entry["completion"] = "template"
                    entry["supported_frac"] = result["fit_report"]["supported_frac"]
                with STATE_LOCK:
                    STATE["results"].append(entry)
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
                "voxel_size": 0.002, "min_cameras": 4,
                "half_xy": 0.20, "depth_below": 0.05, "depth_above": 0.20,
                "head_min_cameras": "", "head_zone_frac": 0,
                "smooth_iterations": 10, "texture_mode": "atlas", "atlas_size": 1536,
                "completion_mode": "none",
                # rat_brown.obj is not redistributed (third-party model); fall back to the procedural one
                "template_path": str(next(p for p in (TOOL_DIR / "templates" / "rat_brown.obj",
                                                      TOOL_DIR / "templates" / "rat_template.obj")
                                          if p.is_file() or p.name == "rat_template.obj")),
                "fit_iterations": 600, "fit_nodes": 200, "fit_sil_cameras": "",
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
