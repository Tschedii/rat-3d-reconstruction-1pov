#!/usr/bin/env python3
"""
Local server backing index.html: head tracking of one or two rats (rat 1 =
mask label 1 / object 1, rat 2 = mask label 2 / object 2). Each tracked rat's
body position, heading and head position all come from every rig camera's
GUI 2 segmentation mask, fused per frame (analysis_lib.fuse_frame /
analyse_from_masks, one run per rat); no head-mounted IMU is used. Each rat's
own mesh is only a fallback for its head where the masks lose it for several
frames. Roll (rotation about the nose axis) cannot be seen in a silhouette and
is always reported as 0.

Serves two reconstructed rat meshes, each from any frame of the recording, in
their true relative position (same calibration-frame -> table-frame
transform as GUI 4), for visual context.

Everything is driven by session_config.json (POST /api/config):
  calib_json_path   rig JSON from GUI 1 (defines the world / table frame)
  masks_root        folder with cam_N/masks_raw (every camera found there is used)
  mask_label        older single-rat setting: the one label tracked when an
                    /api/analyze request names no "rats" (default 1); the panel's
                    rat checkboxes always send "rats"
  rat1_root         folder holding frame_XXXXXX/ mesh folders for object 1 (optional:
                    3D view, and rat 1's head fallback)
  rat2_root         same for object 2 (rat 2)
  fps               video frame rate (rows per second)
  camera_image_root optional: root of the raw per-camera frames, laid out as
                    cam_<n>/all_objects/frame_XXXXXX.png, used only to render
                    the head-path re-projection check onto the real footage
"""

import json
import mimetypes
import re
import sys
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    import numpy as np
except ImportError:  # the system python3 on macOS has no numpy
    sys.exit(f"{sys.executable} has no numpy. Run this server with the python that has numpy/scipy/matplotlib "
             f"(e.g. ~/opt/anaconda3/bin/python), or `pip install numpy scipy matplotlib`.")

TOOL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TOOL_DIR / "session_config.json"
STATE_PATH = TOOL_DIR / "view_state.json"
EXPORTS_DIR = TOOL_DIR / "exports"
CACHE_DIR = TOOL_DIR / "cache"
PORT = 8978

# The table-aligned display frame and the OBJ inspector are GUI 4's; importing
# them keeps one definition of the world frame across both GUIs, so a position
# written by this tool means the same thing as a placement in GUI 4.
GUI4_DIR = TOOL_DIR.parent / "gui_4_scene_placement"
sys.path.insert(0, str(GUI4_DIR))
try:
    from env_marker_server import build_initial_rig, inspect_obj  # noqa: E402
except ImportError:
    # running outside the repo (GUI 4 not next to us): use the vendored copy
    from gui4_rig import build_initial_rig, inspect_obj  # noqa: E402
    print(f"note: {GUI4_DIR} not found, using the local gui4_rig.py copy of GUI 4's rig helpers", flush=True)

try:
    import analysis_lib as al  # noqa: E402
except ImportError as e:
    sys.exit(f"cannot import analysis_lib from {TOOL_DIR} ({e}); it needs numpy, scipy and matplotlib.")

BUNDLE_PATH = TOOL_DIR / "bundle.js"
# Served instead of a silent 404 when the frontend was never built: index.html
# calls window.__enterScene from the bundle, so without it "Enter scene" posts
# the config, hides the setup form and then dies on an undefined function.
BUNDLE_MISSING_JS = (
    "(function(){var m='bundle.js is missing next to server.py: the three.js frontend has not been built. "
    "Run  npm install && npm run build  in head_tracking/ and reload.';"
    "console.error(m);window.__enterScene=function(){alert(m);};"
    "window.addEventListener('DOMContentLoaded',function(){"
    "var b=document.getElementById('enterBtn');if(b){b.textContent='frontend not built (see console)';}"
    "var o=document.getElementById('setupOverlay');if(o){var d=document.createElement('p');"
    "d.style.color='#f88';d.textContent=m;o.appendChild(d);}});})();"
)

FRAME_RE = re.compile(r"^frame_(\d+)$")

PROGRESS = {"running": False, "log": [], "error": None, "result": None}
PROGRESS_LOCK = threading.Lock()


def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            return json.load(f)
    return {}


def load_state():
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _has_frame_dirs(d: Path):
    try:
        return d.is_dir() and any(FRAME_RE.match(p.name) and p.is_dir() for p in d.iterdir())
    except OSError:
        return False


def root_for(which, config):
    key = {"rat1": "rat1_root", "rat2": "rat2_root"}.get(which)
    if not key or not config.get(key):
        return None
    root = Path(config[key]).expanduser()
    # GUI 2 writes object_N/output/frame_XXXXXX/; accept the parent folder too
    if not _has_frame_dirs(root) and _has_frame_dirs(root / "output"):
        return root / "output"
    return root


def frames_for(which, config):
    root = root_for(which, config)
    if not root or not root.is_dir():
        return {}
    return al.scan_mesh_frames(root)


def frame_id(n):
    return f"frame_{int(n):06d}"


def listing(d):
    """Names in an export folder for the results panel (subfolders with a
    trailing /), without the ._ AppleDouble files macOS leaves on network drives."""
    return sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir() if not p.name.startswith("._"))


def log(msg):
    with PROGRESS_LOCK:
        PROGRESS["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(msg, flush=True)


def rats_to_track(payload, config):
    """Mask labels to track this run, sorted: the request's "rats" (the panel's
    rat 1 / rat 2 checkboxes). A request without "rats" (the single-rat API
    before two-rat tracking) tracks its "mask_label", else session_config.json's
    mask_label, else 1, as it always did."""
    rats = payload.get("rats")
    if rats is None:
        rats = [payload.get("mask_label") or config.get("mask_label") or 1]
    rats = sorted({int(r) for r in rats})
    if not rats:
        raise ValueError("no rat selected: tick rat 1 and/or rat 2")
    if any(r not in al.RAT_COLOR for r in rats):
        raise ValueError(f"rats must be mask labels 1 and/or 2, got {rats}")
    return rats


def run_analysis(payload):
    """Background thread: everything after the user hits 'Run analysis'.
    Each selected rat is its own analyse_from_masks run (analysis_lib.py) on
    its own mask label, written to its own rat<N>/ subfolder with exactly the
    files a single-rat run has; position, heading and head all come from the
    masks, and that rat's mesh root (rat1_root / rat2_root) is only the head
    fallback during long gaps in the mask head detection. The run folder
    itself gets the plots and summary that put the rats side by side."""
    try:
        config = load_config()
        rig = build_initial_rig(config["calib_json_path"])
        B, tc = al.table_frame_from_rig(rig)
        cams = al.load_camera_extrinsics(config["calib_json_path"])

        fps = float(payload.get("fps") or config.get("fps") or 30.0)
        rats = rats_to_track(payload, config)
        masks_root = config.get("masks_root")
        if not masks_root:
            raise ValueError("masks_root is not configured (folder with cam_N/masks_raw)")
        params = dict(payload.get("params") or {})
        min_confidence = float(params.pop("min_confidence", 0.15) or 0.15)

        frame_start, frame_end = payload.get("frame_start"), payload.get("frame_end")
        if frame_start is None or frame_end is None:
            raise ValueError("frame_start and frame_end are required (there is no anchor frame to default a window from)")
        frame_start, frame_end = int(frame_start), int(frame_end)
        if frame_end - frame_start + 1 < 3:
            raise ValueError(f"analysis window [frame {frame_start}, frame {frame_end}] has fewer than 3 frames")

        ref = params.get("reference_frame")
        ref = int(ref) if ref is not None else frame_start
        ref = max(frame_start, min(frame_end, ref))

        mesh_frame = config.get("mesh_frame") or "calib"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = EXPORTS_DIR / f"masks_{stamp}_rats{''.join(map(str, rats))}_{frame_start:06d}-{frame_end:06d}"
        results, per_rat = {}, {}
        for mask_label in rats:
            def rlog(msg, n=mask_label):
                log(f"rat {n}: {msg}")
            mesh_root = root_for(f"rat{mask_label}", config)
            mesh_root = str(mesh_root) if mesh_root else None
            rlog(f"head tracking: frames {frame_start}..{frame_end} "
                 f"({(frame_end - frame_start + 1) / fps:.1f} s), label {mask_label}, masks root {masks_root}, "
                 f"cameras {', '.join(sorted(cams))}, head fallback mesh root {mesh_root or '(none)'}")
            res = al.analyse_from_masks(cams, masks_root, mask_label, fps, frame_start, frame_end, B, tc,
                                         params=params, log=rlog, min_confidence=min_confidence,
                                         mesh_root=mesh_root, mesh_frame=mesh_frame)
            res["summary"]["reference_frame"] = ref
            mf = res["summary"]["mask_fusion"]
            rlog(f"mask fusion: {mf['fraction_with_heading'] * 100:.0f}% of frames got a fused heading; head from masks "
                 f"in {mf['frames_head_from_masks']}/{len(res['t'])} frames, from the mesh in {mf['frames_head_from_mesh']}, "
                 f"interpolated in {mf['frames_head_interpolated']}, missing in {mf['frames_head_missing']}")

            # rat<N>/ holds exactly what a single-rat run wrote, so a tool that
            # reads one rat's export (e.g. GUI 5.3) can be pointed at it
            rat_dir = out_dir / f"rat{mask_label}"
            rat_dir.mkdir(parents=True, exist_ok=True)
            rlog(f"writing CSVs to {rat_dir}")
            al.write_csvs(res, rat_dir, B, tc, mesh_frame)
            rlog("rendering plots")
            plots = al.make_plots(res, rat_dir, reference_frame=ref, color=al.RAT_COLOR[mask_label], name=f"rat {mask_label}")

            alignment = {
                "source": "masks_only", "mask_label": mask_label, "reference_frame": ref, "fps": fps,
                "window": {"start_frame": frame_start, "end_frame": frame_end},
                "min_confidence": min_confidence,
                "table_frame": {"display_basis_rows": B.tolist(), "table_center_display": tc.tolist(),
                                "note": "p_table = display_basis @ p_calib - table_center_display"},
                "masks_root": masks_root,
                "head_fallback_mesh_root": mf["mesh_fallback_root"],
                "calib_json_path": config["calib_json_path"],
                "camera_image_root": config.get("camera_image_root") or None,
            }
            (rat_dir / "alignment.json").write_text(json.dumps(alignment, indent=2))
            (rat_dir / "summary.json").write_text(json.dumps(res["summary"], indent=2))
            results[mask_label] = res
            per_rat[f"rat{mask_label}"] = {
                "label": mask_label,
                "color": al.RAT_COLOR[mask_label],
                "rel_dir": f"{out_dir.name}/{rat_dir.name}",
                "plots": plots,
                "summary": res["summary"],
                "viewer": al.downsample_for_viewer(res),
                "files": listing(rat_dir),
            }

        log("rendering the combined plots")
        plots = al.make_two_rat_plots(results, out_dir, reference_frame=ref)
        cam_png = al.make_camera_overlays(results, out_dir, B, tc, mesh_frame,
                                          config["calib_json_path"], config.get("camera_image_root"),
                                          reference_frame=ref, log=log)
        if cam_png:
            plots["camera_overlays"] = cam_png
        summary = al.run_summary(results, reference_frame=ref)
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        between = (summary.get("between_rats_m") or {}).get("head_to_head") or {}
        if between.get("median") is not None:
            log(f"rat 1 to rat 2 head distance: median {between['median'] * 100:.1f} cm, "
                f"closest {between['min'] * 100:.1f} cm over {between['n_frames']} frames")

        result = {
            "out_dir": str(out_dir),
            "rel_dir": out_dir.name,
            "rats": rats,
            "plots": plots,
            "summary": summary,
            "per_rat": per_rat,
            "files": listing(out_dir),
        }
        with PROGRESS_LOCK:
            PROGRESS["result"] = result
        log("done")
    except Exception as e:  # report to the UI, keep the server alive
        traceback.print_exc()
        with PROGRESS_LOCK:
            PROGRESS["error"] = str(e)
        log(f"ERROR: {e}")
    finally:
        with PROGRESS_LOCK:
            PROGRESS["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_file(self, path: Path, content_type=None):
        if not path.exists() or not path.is_file():
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
        try:
            self._get()
        except Exception as e:  # keep the connection answered so the UI can show the reason
            traceback.print_exc()
            try:
                self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)
            except Exception:
                pass

    def _get(self):
        url = urlparse(self.path)
        path = url.path
        qs = parse_qs(url.query)
        config = load_config()

        if path in ("/", "/index.html"):
            self._send_file(TOOL_DIR / "index.html", "text/html"); return
        if path == "/bundle.js":
            if BUNDLE_PATH.is_file():
                self._send_file(BUNDLE_PATH, "application/javascript")
            else:
                body = BUNDLE_MISSING_JS.encode("utf-8")
                self.send_response(200); self._cors()
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)
            return
        if path == "/api/config":
            self._send_json(config); return
        if path == "/api/state":
            self._send_json(load_state()); return

        if path == "/api/initial_rig":
            if not config.get("calib_json_path"):
                self._send_json({"error": "not configured"}, status=400); return
            try:
                self._send_json(build_initial_rig(config["calib_json_path"]))
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/frames":
            which = qs.get("which", [""])[0]
            root = root_for(which, config)
            if not root:
                self._send_json({"configured": False, "frames": []}); return
            try:
                frames = al.scan_mesh_frames(root)
            except FileNotFoundError as e:
                self._send_json({"configured": True, "error": str(e), "frames": []}); return
            self._send_json({"configured": True, "root": str(root),
                             "frames": [{"frame": n, "id": frame_id(n),
                                         "mesh": "mesh.obj" if (d / "mesh.obj").exists() else "mesh_preview.ply"}
                                        for n, d in sorted(frames.items())]})
            return

        if path == "/api/mesh_info":
            which = qs.get("which", [""])[0]
            frame = qs.get("frame", [""])[0]
            frames = frames_for(which, config)
            try:
                n = int(frame)
            except ValueError:
                self._send_json({"error": "frame must be an integer"}, status=400); return
            d = frames.get(n)
            if not d:
                self._send_json({"exists": False}); return
            try:
                info = al.mesh_centroid(d)
            except Exception as e:
                self._send_json({"exists": True, "error": str(e)}); return
            info["exists"] = True
            info["frame"] = n
            info["dir"] = str(d)
            if (d / "mesh.obj").exists():
                info["obj"] = inspect_obj(d / "mesh.obj")
            self._send_json(info)
            return

        if path == "/api/sequence_info":
            which = qs.get("which", [""])[0]
            frames = frames_for(which, config)
            if not frames:
                self._send_json({"configured": bool(root_for(which, config)), "n_frames": 0}); return
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cents = al.collect_centroids(frames, cache_path=CACHE_DIR / f"centroids_{which}.json")
                info = al.sequence_provenance(cents)
            except Exception as e:
                self._send_json({"configured": True, "n_frames": len(frames), "error": str(e)}); return
            info["configured"] = True
            self._send_json(info)
            return

        if path == "/api/progress":
            with PROGRESS_LOCK:
                self._send_json(dict(PROGRESS))
            return

        # mesh files: /rat1/frame_000045/mesh.obj (+ .mtl, texture next to it)
        m = re.match(r"^/(rat1|rat2)/(frame_\d+)/([^/]+)$", path)
        if m:
            which, fid, filename = m.groups()
            if ".." in filename:
                self.send_response(400); self._cors(); self.end_headers(); return
            root = root_for(which, config)
            if not root:
                self.send_response(404); self._cors(); self.end_headers(); return
            # the frontend asks for frame_%06d; the folder on disk may use fewer digits
            d = root / fid
            if not d.is_dir():
                n = int(fid.split("_", 1)[1])
                d = frames_for(which, config).get(n, d)
            self._send_file(d / filename)
            return

        if path.startswith("/exports/"):
            rel = path[len("/exports/"):]
            if ".." in rel:
                self.send_response(400); self._cors(); self.end_headers(); return
            self._send_file(EXPORTS_DIR / rel)
            return

        self.send_response(404); self._cors(); self.end_headers()

    def do_POST(self):
        try:
            self._post()
        except Exception as e:
            traceback.print_exc()
            try:
                self._send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
            except Exception:
                pass

    def _post(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/config":
            payload = json.loads(body)
            if not isinstance(payload, dict):
                self._send_json({"ok": False, "error": "config must be a JSON object"}, status=400); return
            config = load_config()
            config.update(payload)  # keep keys the setup form does not know about
            with CONFIG_PATH.open("w") as f:
                json.dump(config, f, indent=2)
            missing = [k for k in ("calib_json_path", "rat1_root", "rat2_root", "masks_root", "camera_image_root")
                       if config.get(k) and not Path(config[k]).expanduser().exists()]
            for k in missing:
                print(f"config: {k} does not exist: {config[k]}", flush=True)
            self._send_json({"ok": True, "missing_paths": missing}); return

        if path == "/api/state":
            payload = json.loads(body)
            with STATE_PATH.open("w") as f:
                json.dump(payload, f, indent=2)
            self._send_json({"ok": True, "path": str(STATE_PATH)}); return

        if path == "/api/analyze":
            payload = json.loads(body)
            with PROGRESS_LOCK:
                if PROGRESS["running"]:
                    self._send_json({"ok": False, "error": "analysis already running"}, status=409); return
                PROGRESS.update({"running": True, "log": [], "error": None, "result": None})
            threading.Thread(target=run_analysis, args=(payload,), daemon=True).start()
            self._send_json({"ok": True}); return

        self.send_response(404); self._cors(); self.end_headers()


def main():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not BUNDLE_PATH.is_file():
        print(f"WARNING: {BUNDLE_PATH} is missing; run `npm install && npm run build` here or the page "
              f"cannot enter the scene", flush=True)
    if not (TOOL_DIR / "index.html").is_file():
        print(f"WARNING: {TOOL_DIR / 'index.html'} is missing", flush=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on port {PORT} ({e}); is another server.py still running?")
    print(f"head tracking running at http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
