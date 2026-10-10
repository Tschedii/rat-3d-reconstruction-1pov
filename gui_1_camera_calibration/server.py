#!/usr/bin/env python3
"""
Local server backing index.html — GUI 1: checkerboard calibration.

Three run modes, all executed in a background thread so the frontend can
poll progress without blocking:

  intrinsics : per-camera diverse-frame extraction + intrinsic solve from the
               *intrinsic* video folder -> <name>_intrinsics.json
  extrinsics : synced-frame extraction + rig solve from the *extrinsic* video
               folder, using intrinsics loaded from a JSON -> <name>.json + plots
  both       : intrinsics followed by extrinsics (the original pipeline)

Videos are discovered from a folder: any video file whose name contains a
camera number (cam1.mp4, cam_01.mov, camera_001.mp4, ...) maps to camera_NNN.
"""

import concurrent.futures
import json
import os
import re
import threading
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

import calibration_lib as cal

TOOL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOOL_DIR / "output"
WORK_DIR = TOOL_DIR / "work"
PORT = 8971

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}
# "cam1", "cam_01", "camera-001", "Cam 3" ... -> the trailing number is the camera id
CAM_RE = re.compile(r"cam(?:era)?[\s_\-]?0*(\d+)", re.IGNORECASE)

STATE_LOCK = threading.Lock()
STATE = {
    "status": "idle",     # idle | running | done | error
    "mode": None,         # intrinsics | extrinsics | both
    "log": [],
    "result": None,
    "error": None,
}


def log(msg):
    print(msg)
    with STATE_LOCK:
        STATE["log"].append(msg)


# =============================================================================
# video discovery
# =============================================================================

def discover_camera_videos(folder):
    """Map every video in `folder` to a camera name. Returns
    ({camera_name: path}, [skipped file names]). Raises on a missing folder,
    on two files claiming the same camera, or when nothing is found."""
    folder = Path(str(folder).strip()).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"video folder not found: {folder}")
    found, skipped = {}, []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        m = CAM_RE.search(p.stem)
        if m:
            idx = int(m.group(1))
        elif p.stem.isdigit():
            idx = int(p.stem)
        else:
            skipped.append(p.name)
            continue
        name = f"camera_{idx:03d}"
        if name in found:
            raise ValueError(f"{folder}: both '{Path(found[name]).name}' and '{p.name}' map to {name}")
        found[name] = str(p)
    if not found:
        raise FileNotFoundError(
            f"no camera videos found in {folder} -- expected files like cam1.mp4 ... cam6.mp4 "
            f"(extensions {', '.join(sorted(VIDEO_EXTENSIONS))})"
        )
    return dict(sorted(found.items())), skipped


def split_folders(text):
    """'a; b' / 'a\\nb' -> ['a', 'b']. Several intrinsic folders = several
    recording sessions whose frames are pooled per camera."""
    return [p.strip() for p in re.split(r"[;\n]", text or "") if p.strip()]


def discover_camera_videos_multi(folders):
    """Same as discover_camera_videos for a list of folders. Every folder must
    contain the same cameras. Returns ({camera_name: [path, ...]}, [skipped])."""
    per_folder = [discover_camera_videos(f) for f in folders]
    names = set(per_folder[0][0])
    for f, (found, _) in zip(folders, per_folder):
        if set(found) != names:
            raise ValueError(
                f"intrinsic folders must contain the same cameras -- {folders[0]} has "
                f"{sorted(names)}, {f} has {sorted(found)}"
            )
    videos = {n: [found[n] for found, _ in per_folder] for n in sorted(names)}
    skipped = sorted({s for _, sk in per_folder for s in sk})
    return videos, skipped


def _scan_folder_for_api(folder, multi=False):
    if not folder:
        return {"ok": True, "cameras": [], "skipped": [], "empty": True}
    try:
        if multi:
            found, skipped = discover_camera_videos_multi(split_folders(folder))
            cams = [{"name": n, "file": " | ".join(Path(p).name for p in ps), "path": ps[0],
                     "paths": ps} for n, ps in found.items()]
            return {"ok": True, "cameras": cams, "skipped": skipped, "sessions": len(split_folders(folder))}
        found, skipped = discover_camera_videos(folder)
        return {"ok": True, "cameras": [{"name": n, "file": Path(p).name, "path": p} for n, p in found.items()],
                "skipped": skipped}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =============================================================================
# pipeline
# =============================================================================

def _parse_board(cfg):
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
    return board_w, board_h, square_size


def _output_path(cfg, suffix=""):
    name = (cfg.get("output_name") or "").strip() or date.today().strftime("%Y_%m_%d")
    if name.endswith(".json"):
        name = name[:-5]
    return OUTPUT_DIR / f"{name}{suffix}.json"


def run_pipeline(cfg):
    mode = (cfg.get("mode") or "both").strip().lower()
    with STATE_LOCK:
        STATE["status"] = "running"
        STATE["mode"] = mode
        STATE["log"] = []
        STATE["result"] = None
        STATE["error"] = None

    try:
        if mode not in ("intrinsics", "extrinsics", "both"):
            raise ValueError(f"unknown mode '{mode}' -- expected intrinsics, extrinsics or both")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        board_w, board_h, square_size = _parse_board(cfg)
        sensor_specs = {
            "focal_length_mm": cfg.get("focal_length_mm"),
            "sensor_width_mm": cfg.get("sensor_width_mm"),
            "sensor_height_mm": cfg.get("sensor_height_mm"),
        }

        # ---- video folders --------------------------------------------------
        intr_folders = split_folders(cfg.get("intrinsic_folder"))
        intr_folder = "; ".join(intr_folders)
        ext_folder = (cfg.get("extrinsic_folder") or "").strip()
        if not ext_folder:
            if len(intr_folders) == 1:
                ext_folder = intr_folders[0]
            elif len(intr_folders) > 1 and mode in ("extrinsics", "both"):
                raise ValueError(
                    "several intrinsic folders were given (pooled sessions), so the extrinsic "
                    "folder must be set explicitly -- the rig geometry has to come from ONE synchronous recording"
                )
        intr_videos = ext_videos = None

        if mode in ("intrinsics", "both"):
            if not intr_folders:
                raise ValueError("intrinsic video folder is required for this mode")
            intr_videos, skipped = discover_camera_videos_multi(intr_folders)
            log(f"=== intrinsic videos ({len(intr_folders)} session(s): {intr_folder}) ===")
            for n, ps in intr_videos.items():
                log(f"  {n} <- {' | '.join(Path(p).name for p in ps)}")
            if skipped:
                log(f"  ignored (no camera number in name): {', '.join(skipped)}")

        if mode in ("extrinsics", "both"):
            if not ext_folder:
                raise ValueError("extrinsic video folder is required for this mode")
            ext_videos, skipped = discover_camera_videos(ext_folder)
            log(f"=== extrinsic videos ({ext_folder}) ===")
            for n, p in ext_videos.items():
                log(f"  {n} <- {Path(p).name}")
            if skipped:
                log(f"  ignored (no camera number in name): {', '.join(skipped)}")

        if intr_videos is not None and ext_videos is not None and set(intr_videos) != set(ext_videos):
            only_i = sorted(set(intr_videos) - set(ext_videos))
            only_e = sorted(set(ext_videos) - set(intr_videos))
            raise ValueError(
                "intrinsic and extrinsic folders must contain the same cameras -- "
                f"only in intrinsic folder: {only_i or '-'}; only in extrinsic folder: {only_e or '-'}"
            )

        camera_names = sorted(intr_videos or ext_videos)
        result = {"mode": mode}
        intrinsics_warnings = []

        # ---- STAGE 1: intrinsics -------------------------------------------
        intrinsics_json = (cfg.get("intrinsics_json") or "").strip()
        if mode in ("intrinsics", "both"):
            cameras_cfg = [{"name": n, "video_paths": ps} for n, ps in intr_videos.items()]
            intrinsics_by_cam = run_intrinsics_stage(cfg, cameras_cfg, board_w, board_h, square_size, sensor_specs)
            log("=== intrinsics sanity check across cameras ===")
            intrinsics_warnings = cal.check_intrinsics_consistency(intrinsics_by_cam, sensor_specs, log=log)

            intr_summary = cal.intrinsics_summary(intrinsics_by_cam, board_w, board_h, square_size,
                                                  warnings=intrinsics_warnings, source_videos=intr_videos)
            intr_summary["source_folder"] = intr_folder
            out_intr = _output_path(cfg, "_intrinsics")
            with open(out_intr, "w") as f:
                json.dump(intr_summary, f, indent=2)
            log(f"  saved {out_intr}")
            result["intrinsics_json_path"] = str(out_intr)
            result["intrinsics"] = intrinsics_table(intr_summary)
            intrinsics_source = f"stage 1 (this run, {intr_folder})"
        else:
            if not intrinsics_json:
                raise ValueError(
                    "extrinsics-only mode needs an intrinsics JSON (run 'intrinsics' first, or point the "
                    "field at a previous *_intrinsics.json / full calibration json)"
                )
            if not Path(intrinsics_json).exists():
                raise FileNotFoundError(f"intrinsics_json not found: {intrinsics_json}")
            log(f"=== STAGE 1: skipped -- loading intrinsics from {intrinsics_json} ===")
            intrinsics_by_cam = cal.load_intrinsics_from_json(intrinsics_json, camera_names, log=log)
            log("=== intrinsics sanity check across cameras ===")
            intrinsics_warnings = cal.check_intrinsics_consistency(intrinsics_by_cam, sensor_specs, log=log)
            intrinsics_source = intrinsics_json

        if mode == "intrinsics":
            log("=== QUALITY SUMMARY (intrinsics) ===")
            for row in result["intrinsics"]:
                log(f"  {row['name']:12s} fx {row['fx']:.1f}  k1 {row['k1']:+.4f}  rms {row['rms_px']:.3f}px  "
                    f"images {row['num_images_used']}  model {row['model']}")
            for w in intrinsics_warnings:
                log(f"  WARNING: {w}")
            if not intrinsics_warnings:
                log("  no warnings")
            result["warnings"] = intrinsics_warnings
            with STATE_LOCK:
                STATE["status"] = "done"
                STATE["result"] = result
            log("=== DONE ===")
            return

        # ---- STAGE 2: synced frames ----------------------------------------
        reference_name = cfg.get("reference_camera") or camera_names[0]
        if reference_name not in camera_names:
            raise ValueError(f"reference camera '{reference_name}' is not among the discovered cameras {camera_names}")

        synced_dir = WORK_DIR / "external_parameter"
        reuse = bool(cfg.get("reuse_synced_frames", False))
        existing = [len(cal._get_frame_stems(synced_dir / n)) if (synced_dir / n).is_dir() else 0
                    for n in camera_names]
        if reuse and all(existing):
            log(f"=== STAGE 2: skipped -- reusing {min(existing)} already extracted synced frame set(s) "
                f"in {synced_dir} (no video decoding) ===")
        else:
            if reuse:
                log("  (reuse requested but no extracted frames found for every camera -- extracting)")
            log(f"=== STAGE 2: extracting synchronized frames from {ext_folder} (extrinsics) ===")
            cal.extract_synced_frames(
                ext_videos, synced_dir, board_w, board_h,
                sample_every_n=int(cfg.get("synced_sample_every_n", 5)),
                max_frames=int(cfg.get("synced_max_frames", 60)),
                min_frame_gap=int(cfg.get("synced_min_frame_gap", 30)),
                min_pose_distance=float(cfg.get("synced_min_pose_distance", 0.15)),
                min_trans_distance=float(cfg.get("synced_min_trans_distance", 0.05)),
                square_size=square_size,
                diversity_camera=reference_name,
                log=log,
            )

        # ---- STAGE 3: extrinsics -------------------------------------------
        bundle_adjust = bool(cfg.get("bundle_adjust", True))
        refine_focal = bool(cfg.get("refine_focal", False))
        log(f"=== STAGE 3: extrinsic calibration (bundle adjustment {'on' if bundle_adjust else 'off'}"
            f"{', fx refined' if bundle_adjust and refine_focal else ''}) ===")
        cameras_for_extrinsics = [
            {"name": n, "frames_dir": synced_dir / n, "intrinsics": intrinsics_by_cam[n]}
            for n in camera_names
        ]
        summary = cal.calibrate_extrinsics(
            cameras_for_extrinsics, board_w, board_h, square_size, reference_name,
            frame_outlier_px=float(cfg.get("frame_outlier_px", 3.0)),
            bundle_adjust=bundle_adjust,
            refine_focal=refine_focal,
            log=log,
        )
        summary["warnings"] = intrinsics_warnings + summary.get("warnings", [])
        summary["intrinsics_source"] = intrinsics_source
        summary["extrinsic_videos"] = {n: str(p) for n, p in ext_videos.items()}

        # ---- STAGE 4: plots + save -----------------------------------------
        log("=== STAGE 4: rig visualization ===")
        perspective_png, top_png = cal.plot_rig(summary, OUTPUT_DIR, log=log)

        out_json = _output_path(cfg)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"  saved {out_json}")

        log("=== QUALITY SUMMARY (RMS per corner) ===")
        quality = quality_table(summary)
        for row in quality:
            log(f"  {row['name']:12s} reproj averaged {row['reproj_avg_mean_px']:.3f}px  "
                f"{summary['extrinsics_method']} {row['reproj_mean_px']:.3f}px (max {row['reproj_max_px']:.3f})  "
                f"t_std {row['translation_std_mm']:.1f}mm  rot {row['rotation_spread_deg']:.3f}deg  "
                f"fx {row['fx']:.1f}"
                + (f" ({row['fx_change_pct']:+.2f}%)" if row['fx_change_pct'] is not None else "")
                + (f"  intr rms {row['intrinsic_rms_px']:.3f}px" if row['intrinsic_rms_px'] is not None else ""))
        if summary.get("dropped_frames"):
            log(f"  dropped {len(summary['dropped_frames'])} outlier frame set(s): "
                + ", ".join(d["stem"] for d in summary["dropped_frames"]))
        for w in summary["warnings"]:
            log(f"  WARNING: {w}")
        if not summary["warnings"]:
            log("  no warnings")

        result.update({
            "json_path": str(out_json),
            "perspective_png": Path(perspective_png).name,
            "top_view_png": Path(top_png).name,
            "quality": quality,
            "warnings": summary["warnings"],
            "num_frames_used": summary["num_frames_used"],
            "dropped_frames": summary.get("dropped_frames", []),
            "extrinsics_method": summary["extrinsics_method"],
            "bundle_adjustment": summary.get("bundle_adjustment"),
            "frame_diversity": summary["frame_diversity"],
            "overall_reprojection_mean_px": summary["validation"]["overall_mean_reprojection_error_px"],
            "overall_reprojection_averaged_px": summary["validation"]["averaged_rig_overall_mean_reprojection_error_px"],
            "intrinsics_source": intrinsics_source,
        })
        with STATE_LOCK:
            STATE["status"] = "done"
            STATE["result"] = result
        log("=== DONE ===")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"ERROR: {e}\n{tb}")
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["error"] = str(e)


def intrinsics_table(intr_summary):
    """Flat per-camera rows for an intrinsics-only result."""
    rows = []
    for cam in intr_summary["cameras"]:
        i = cam["intrinsics"]
        rows.append({
            "name": cam["name"],
            "fx": i["fx"], "fy": i["fy"], "cx": i["cx"], "cy": i["cy"],
            "k1": i["k1"], "k2": i.get("k2", 0.0),
            "rms_px": i.get("rms_reprojection_error_px"),
            "num_images_used": i.get("num_images_used"),
            "model": i.get("model", ""),
            "source_video": (" | ".join(Path(p).name for p in cam["source_videos"])
                             if cam.get("source_videos")
                             else (Path(cam["source_video"]).name if cam.get("source_video") else "")),
        })
    return rows


def quality_table(summary):
    """Flat per-camera quality rows for the log and the frontend."""
    rows = []
    fx_change = (summary.get("bundle_adjustment") or {}).get("fx_change_pct") or {}
    for cam in summary["cameras"]:
        cons = cam["consistency"]
        intr = cam["intrinsics"]
        avg = cons.get("reprojection_error_averaged_rig_px") or cons["reprojection_error_px"]
        rows.append({
            "name": cam["name"],
            "is_reference": cam["is_reference"],
            "reproj_avg_mean_px": avg["mean_px"],
            "reproj_mean_px": cons["reprojection_error_px"]["mean_px"],
            "reproj_max_px": cons["reprojection_error_px"]["max_px"],
            "worst_frame": cons["reprojection_error_px"].get("worst_frame"),
            "translation_std_mm": float(np.linalg.norm(cons["translation_std_m"]) * 1000.0),
            "rotation_spread_deg": cons["rotation_spread_deg"],
            "fx": intr["fx"],
            "fx_change_pct": fx_change.get(cam["name"]),
            "k1": intr["k1"],
            "intrinsic_rms_px": intr.get("rms_reprojection_error_px"),
        })
    return rows


def run_intrinsics_stage(cfg, cameras_cfg, board_w, board_h, square_size, sensor_specs):
    """STAGE 1: per-camera diverse-frame extraction + intrinsic calibration,
    fanned out across cameras. Each camera may have several videos (one per
    recording session); their frames are pooled in session_N/ sub-folders and
    calibrated together. Returns {name: intrinsics dict}."""
    log("=== STAGE 1: extracting diverse frames per camera (intrinsics) ===")

    def calibrate_one_camera(c):
        name = c["name"]
        paths = c.get("video_paths") or [c["video_path"]]
        frame_dir = WORK_DIR / "internal_parameter" / name
        stale = cal.clear_frame_tree(frame_dir)
        if stale:
            log(f"[{name}] removed {stale} stale frame(s) from a previous run")
        for k, video in enumerate(paths, start=1):
            log(f"[{name}] extracting diverse frames from {video}" + (f" (session {k}/{len(paths)})" if len(paths) > 1 else ""))
            cal.extract_diverse_frames(
                video, frame_dir / f"session_{k}", board_w, board_h,
                target_frames=int(cfg.get("target_frames", 300)),
                min_pose_distance=float(cfg.get("min_pose_distance", 0.15)),
                min_trans_distance=float(cfg.get("min_trans_distance", 0.05)),
                sample_every_n=int(cfg.get("sample_every_n", 10)),
                log=log,
            )
        log(f"[{name}] calibrating intrinsics" + (f" on {len(paths)} pooled sessions" if len(paths) > 1 else ""))
        return name, cal.calibrate_intrinsics(
            frame_dir, board_w, board_h, square_size,
            sensor_specs=sensor_specs,
            outlier_threshold_px=float(cfg.get("outlier_threshold_px", 1.0)),
            log=log,
        )

    # Each camera's extraction + intrinsic calibration is fully
    # independent (own video, own output folder) -- running them
    # sequentially wastes 5/6 of the machine's cores for the slowest
    # part of the whole pipeline, so fan them out instead.
    max_workers = min(len(cameras_cfg), os.cpu_count() or len(cameras_cfg))
    log(f"  running {len(cameras_cfg)} camera(s) in parallel (up to {max_workers} at once) -- "
        f"log lines below will interleave across cameras")

    intrinsics_by_cam = {}
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(calibrate_one_camera, c): c["name"] for c in cameras_cfg}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                _, intr = future.result()
                intrinsics_by_cam[name] = intr
            except Exception as e:
                failures[name] = e
                log(f"[{name}] FAILED: {e}")
                for pending in future_to_name:
                    pending.cancel()  # no-op for already-running work, skips any not yet started

    if failures:
        names = ", ".join(sorted(failures))
        raise RuntimeError(f"intrinsic calibration failed for camera(s): {names} -- see log above for details")
    return intrinsics_by_cam


# =============================================================================
# HTTP
# =============================================================================

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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._send_file(TOOL_DIR / "index.html", "text/html")
            return
        if path == "/api/progress":
            with STATE_LOCK:
                self._send_json(dict(STATE))
            return
        if path == "/api/scan":
            q = parse_qs(parsed.query)
            self._send_json({
                "intrinsic": _scan_folder_for_api((q.get("intrinsic") or [""])[0].strip(), multi=True),
                "extrinsic": _scan_folder_for_api((q.get("extrinsic") or [""])[0].strip()),
            })
            return
        if path == "/api/defaults":
            self._send_json({
                "board_w": 9, "board_h": 6, "square_size": 0.023,
                "focal_length_mm": 8.0, "sensor_width_mm": 6.6, "sensor_height_mm": 4.1,
                "target_frames": 300, "min_pose_distance": 0.15, "min_trans_distance": 0.05,
                "sample_every_n": 10, "synced_sample_every_n": 5, "synced_max_frames": 60,
                "synced_min_frame_gap": 30, "synced_min_pose_distance": 0.15, "synced_min_trans_distance": 0.05,
                "outlier_threshold_px": 1.0, "frame_outlier_px": 3.0,
                "bundle_adjust": True, "refine_focal": False, "reuse_synced_frames": False,
                "intrinsic_folder": "", "extrinsic_folder": "", "intrinsics_json": "",
                "output_name": date.today().strftime("%Y_%m_%d"),
                "reference_camera": "camera_004",
                "video_extensions": sorted(VIDEO_EXTENSIONS),
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
