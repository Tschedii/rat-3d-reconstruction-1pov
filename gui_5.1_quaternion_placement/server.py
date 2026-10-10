#!/usr/bin/env python3
"""
GUI 5 -- Trajectory Tracing: top-down video of the IMU rat with its head
orientation, computed here from scratch (analysis_core.py). Position comes
from multi-camera segmentation masks (triangulated, fused across views), not
a single mesh reconstruction's centroid, so it survives close contact between
the two animals; orientation comes from the head IMU's quaternions. GUI 5.1
is the earlier, mesh-centroid-based approach kept as a sibling tool.

The background is the RAW camera video (cam_N.mp4, decoded with OpenCV), not
GUI 2's segmented cutouts, warped to a top-down view with topview.py's
ground-plane homography, from one camera or as the per-pixel median of all
cameras with a video. Segmentation masks (all six cameras) are used only for
tracking: triangulated position and fused body axis. The browser draws the head marker / nose arrow and
the 3D inset (cap mesh + axes + trail) from the series computed once here.

Two rats, each with its own head IMU: rat 1 is object 1 in the masks (drawn
red, skull and trajectory), rat 2 is object 2 (blue), the same colours the
mask tint uses. Each rat is analysed on its own (analysis_core.compute with
that rat's settings), so each head is placed and adjusted separately.

session_config.json (POST /api/config):
  calib_json_path     rig JSON (GUI 1) -- table frame + camera geometry
  video_root          folder with the raw videos cam_N.mp4 (N = rig camera number)
  view_mode           "median" (per-pixel median of every camera with a video, which
                      removes each camera's parallax smear of the rat) or one
                      camera name, e.g. "camera_004"
  masks_root          folder with cam_N/masks_raw (default: video_root); every
                      camera found there is used for tracking
  anchor_every_frames re-align the IMU heading every N frames (0 = eyes only)
  fps                 (default 30)
  rats                [rat 1, rat 2], each:
    enabled             analyse this rat
    quat_csv_path       head IMU quaternions, one row per video frame (qw,qx,qy,qz or
                        qc_w,qc_x,qc_y,qc_z columns, or the MATLAB .mat with qc_video)
    anchor_mesh_dir     GUI 2 frame_XXXXXX/ folder (mesh.obj) of this rat
    anchor_frame        (default: from anchor_mesh_dir's name)
    eyes_json_path      GUI 3 eye positions for that frame
    sensor_forward/up   IMU axes pointing to the nose / up (default -x / +z: on this
                        rig the IMU's +x points to the TAIL, found from the mirrored
                        roll and nose-up-while-sniffing pitch with +x)
    imu_roll_sign       "as_recorded" (default) or "flip" (roll-only mirror, not physical)
    imu_offset_frames   video frame = CSV row + this (default 0)
    mount_tilt_mode, mount_roll_deg, mount_pitch_deg, head_clicks, head_click_reach_frames
  (mask_label is the rat's number. A single-rat config from before, with these
  keys at the top level, is read as that mask_label's rat.)
"""

import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit(f"{sys.executable} is missing numpy/Pillow/opencv. Install them with: pip install numpy Pillow opencv-python")

import analysis_core as ac
import topview as tv

TOOL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TOOL_DIR / "session_config.json"
CACHE_DIR = TOOL_DIR / "cache"
PORT = 8975
OUT_SIZE = (800, 800)
VIEW_MARGIN_M = 0.12
FRAME_CACHE_SIZE = 240
DEFAULTS = {"anchor_every_frames": 100, "fps": 30.0, "view_mode": "median"}
RAT_DEFAULTS = {"sensor_forward": "-x", "sensor_up": "+z", "imu_offset_frames": 0, "head_click_reach_frames": 30,
                "mount_tilt_mode": "auto_roll", "imu_roll_sign": "as_recorded"}
RAT_KEYS = ("quat_csv_path", "anchor_mesh_dir", "anchor_frame", "eyes_json_path", "sensor_forward", "sensor_up",
            "imu_offset_frames", "imu_roll_sign", "mount_tilt_mode", "mount_roll_deg", "mount_pitch_deg",
            "head_clicks", "head_click_reach_frames")
N_RATS = 2                       # rat n = object n in the masks

STATE = {"config": {}, "meta": None,
         "rats": [],             # per analysed rat: {"id", "series", "summary", "ctx", "frame_index", ...}
         "views": [],            # [{"name", "cap", "lock", "matrix", "coverage"}] used by the current view
         "n_video_frames": 0, "frame_cache": {}, "frame_cache_order": []}
PROGRESS = {"running": False, "stage": "", "done": 0, "total": 0, "error": None, "log": []}
PROGRESS_LOCK = threading.Lock()


def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            return json.load(f)
    return {}


def log(msg):
    with PROGRESS_LOCK:
        PROGRESS["log"].append(f"[{datetime.now():%H:%M:%S}] {msg}")
    print(msg, flush=True)


def video_path_for(video_root, cam_name):
    return Path(video_root) / f"cam_{ac.cam_number(cam_name)}.mp4"


def cameras_with_video(cfg):
    """Rig cameras whose cam_N.mp4 exists in video_root and opens."""
    out = []
    for c in sorted(tv.load_camera_extrinsics(cfg["calib_json_path"])):
        p = video_path_for(cfg["video_root"], c)
        if p.is_file():
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                out.append(c)
            cap.release()
    return out


def fill_defaults(config):
    """Derive what can be derived from the paths, so the setup form only needs
    the files: masks next to the videos, anchor frame from the mesh folder.
    Always returns N_RATS rats (cfg["rats"][n - 1] = rat n = object n)."""
    keep = lambda d, skip=(): {k: v for k, v in d.items() if v not in (None, "") and k not in skip}
    cfg = {**DEFAULTS, **keep(config, RAT_KEYS + ("rats", "mask_label"))}
    rats = list(config.get("rats") or [])
    if not rats and config.get("quat_csv_path"):
        # single-rat config from before: its IMU belongs to the rat with its mask label
        rats = [None] * N_RATS
        rats[min(max(int(config.get("mask_label") or 1), 1), N_RATS) - 1] = {
            **{k: config[k] for k in RAT_KEYS if k in config}, "enabled": True}
    out = []
    for n in range(1, N_RATS + 1):
        r = {**RAT_DEFAULTS, **keep((rats[n - 1] if n <= len(rats) else None) or {})}
        r["id"] = r["mask_label"] = n
        r["enabled"] = bool(r.get("enabled", bool(r.get("quat_csv_path"))))
        if not r.get("anchor_frame") and r.get("anchor_mesh_dir"):
            fm = re.search(r"frame_(\d+)", Path(r["anchor_mesh_dir"]).name)
            if fm:
                r["anchor_frame"] = int(fm.group(1))
        out.append(r)
    cfg["rats"] = out
    if cfg.get("video_root") and not cfg.get("masks_root"):
        cfg["masks_root"] = cfg["video_root"]
    return cfg


def active_rats(cfg):
    return [r for r in cfg["rats"] if r.get("enabled")]


def rat_config(cfg, r):
    """The flat config analysis_core.compute reads: shared keys + this rat's."""
    return {**{k: v for k, v in cfg.items() if k != "rats"}, **r}


REQUIRED = ("calib_json_path", "video_root", "masks_root")
RAT_REQUIRED = ("quat_csv_path", "anchor_mesh_dir", "eyes_json_path")


def validate(cfg):
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        return f"missing: {', '.join(missing)}"
    absent = [k for k in REQUIRED if not Path(cfg[k]).expanduser().exists()]
    if absent:
        return "not found: " + ", ".join(f"{k} ({cfg[k]})" for k in absent)
    if not active_rats(cfg):
        return "no rat to track: give rat 1 and/or rat 2 an IMU file and tick it"
    for r in active_rats(cfg):
        missing = [k for k in RAT_REQUIRED if not r.get(k)]
        if missing:
            return f"rat {r['id']}: missing {', '.join(missing)}"
        absent = [k for k in RAT_REQUIRED if not Path(r[k]).expanduser().exists()]
        if absent:
            return f"rat {r['id']}: not found: " + ", ".join(f"{k} ({r[k]})" for k in absent)
        if not r.get("anchor_frame") and r.get("anchor_frame") != 0:
            return f"rat {r['id']}: anchor_frame missing (could not read it from the anchor mesh folder name)"
    vids = cameras_with_video(cfg)
    if not vids:
        return f"no cam_N.mp4 for any rig camera in {cfg['video_root']}"
    if cfg["view_mode"] != "median" and cfg["view_mode"] not in vids:
        return f"view '{cfg['view_mode']}' has no video (cameras with video: {', '.join(vids)})"
    return None


def open_view(cam_name, cfg, cam, B, tc, view_box):
    """One camera of the top-down view: its video, its floor-plane remap (lens
    model included) and coverage (the part of the output its image actually
    reaches, so the median ignores what lies outside its field of view)."""
    cap = cv2.VideoCapture(str(video_path_for(cfg["video_root"], cam_name)))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open {video_path_for(cfg['video_root'], cam_name)}")
    mx, my = tv.plane_remap(cam, B, tc, view_box, OUT_SIZE, float(cfg["floor_z_m"]))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cov = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return {"name": cam_name, "cap": cap, "lock": threading.Lock(), "map": (mx, my), "coverage": cov,
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}


FLOOR_BG_FRAMES = 25


def floor_height(cfg, cams, B, tc, log):
    """The floor's height in the table frame. The calibration's z=0 is not the
    arena floor, and a top view warped onto the wrong plane never lines the
    cameras up. Unless floor_z_m is given, it is estimated from rat-free
    backgrounds (per-camera median of FLOOR_BG_FRAMES frames spread over the
    video) as the plane where the cameras agree best; cached per video set."""
    if cfg.get("floor_z_m") not in (None, ""):
        return float(cfg["floor_z_m"]), "set in the form"
    vids = cameras_with_video(cfg)
    paths = [str(video_path_for(cfg["video_root"], c)) for c in vids]
    key = hashlib.sha1(json.dumps([cfg["calib_json_path"], paths,
                                   [Path(q).stat().st_size for q in paths]]).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"floor_{key}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["z"], f"estimated from {len(vids)} cameras (cached, agreement {d['best_score']:.2f})"
    if len(vids) < 2:
        return 0.0, "calibration z=0 (need 2 videos to estimate it)"
    bgs = {}
    for i, c in enumerate(vids):
        with PROGRESS_LOCK:
            PROGRESS.update(stage="estimating floor height", done=i, total=len(vids))
        cap = cv2.VideoCapture(str(video_path_for(cfg["video_root"], c)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stack = []
        for f in np.linspace(0, max(n - 2, 0), FLOOR_BG_FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            if ok:
                stack.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        cap.release()
        if stack:
            bgs[c] = np.median(np.stack(stack), 0)
    z, info = tv.estimate_floor_z(bgs, cams, B, tc)
    if z is None:
        return 0.0, "calibration z=0 (estimate failed)"
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"z": z, **info}))
    return z, f"estimated from {len(bgs)} cameras (agreement {info['best_score']:.2f})"


def run_analysis(cfg):
    try:
        def progress(d, n, stage="reading masks"):
            with PROGRESS_LOCK:
                PROGRESS.update(stage=stage, done=d, total=n)
        B, tc = tv.build_initial_rig_basis(cfg["calib_json_path"])
        cams = tv.load_camera_extrinsics(cfg["calib_json_path"])
        z_floor, how = floor_height(cfg, cams, B, tc, log)
        cfg = {**cfg, "floor_z_m": round(z_floor, 4)}
        log(f"floor at z = {z_floor * 100:+.1f} cm in the table frame ({how})")
        rats = []
        for r in active_rats(cfg):
            rid = r["id"]
            log(f"rat {rid} (object {r['mask_label']}, IMU {Path(r['quat_csv_path']).name})")
            series, summary, ctx = ac.compute(
                rat_config(cfg, r), cache_dir=CACHE_DIR, log=lambda m, rid=rid: log(f"rat {rid}: {m}"),
                progress=lambda d, n, stage="reading masks", rid=rid: progress(d, n, f"rat {rid}: {stage}"),
                return_ctx=True)
            series["rat"] = rid
            rats.append({"id": rid, "series": series, "summary": summary, "ctx": ctx, "q_vision": None,
                         "frame_index": {f: i for i, f in enumerate(series["frame"])},
                         "head_clicks": r.get("head_clicks") or [],
                         "head_click_reach_frames": int(r.get("head_click_reach_frames") or 0)})

        with PROGRESS_LOCK:
            PROGRESS.update(stage="opening videos", done=0, total=0)
        pts = [p for rs in rats for p in rs["series"]["head_top"] if p]
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + VIEW_MARGIN_M
        else:
            cx = cy = 0.0; half = 0.5
        view_box = (cx - half, cx + half, cy - half, cy + half)
        vids = cameras_with_video(cfg)
        used = vids if cfg["view_mode"] == "median" else [cfg["view_mode"]]
        for v in STATE["views"]:
            v["cap"].release()
        views = [open_view(c, cfg, cams[c], B, tc, view_box) for c in used]
        n_video = min(v["n_frames"] for v in views)

        # playable: every video frame at least one rat's IMU covers
        lo = max(min(rs["series"]["frame"][0] for rs in rats), 0)
        hi = min(max(rs["series"]["frame"][-1] for rs in rats), n_video - 1)
        rat_meta = []
        for rs, r in zip(rats, active_rats(cfg)):
            s = rs["series"]
            with_pos = [f for f, ok in zip(s["frame"], s["has_pos"]) if ok]
            rat_meta.append({
                "id": rs["id"], "mask_label": r["mask_label"], "quat_file": Path(r["quat_csv_path"]).name,
                "frame_range": [s["frame"][0], s["frame"][-1]], "anchor_frame": int(r["anchor_frame"]),
                "position_frame_range": [min(with_pos), max(with_pos)] if with_pos else None,
                "summary": rs["summary"],
                "head_clicks": rs["head_clicks"], "head_click_reach_frames": rs["head_click_reach_frames"]})
        spans = [m["position_frame_range"] for m in rat_meta if m["position_frame_range"]]
        STATE.update({
            "config": cfg, "rats": rats, "views": views, "n_video_frames": n_video, "rig": (B, tc, cams),
            "frame_cache": {}, "frame_cache_order": [],
            "meta": {
                "frame_min": lo, "frame_max": hi, "fps": float(cfg["fps"]),
                "position_frame_range": [min(a for a, _ in spans), max(b for _, b in spans)] if spans else None,
                "view_box": list(view_box), "out_size": list(OUT_SIZE),
                "view_mode": cfg["view_mode"], "view_cameras": used, "cameras_with_video": vids,
                "video_root": cfg["video_root"], "n_video_frames": n_video, "floor_z_m": cfg["floor_z_m"],
                "cap_mesh_url": "/assets/cap_head.stl" if (TOOL_DIR / "assets" / "cap_head.stl").exists() else None,
                # default guess, not a measurement: -90 deg about local X sends
                # the scan's +Z (dome/stalk axis) onto the inset's "up"
                "cap_align_deg": cfg.get("cap_align_deg", [-90, 0, -90]),
                "rats": rat_meta,
                "recording": Path(cfg["video_root"]).name, "export_dir": str(EXPORT_DIR_DEFAULT),
            },
        })
        log(f"ready: rats {', '.join(str(rs['id']) for rs in rats)}; view {cfg['view_mode']} from {', '.join(used)} "
            f"({n_video} frames), playable frames {lo}..{hi}")
    except Exception as e:
        traceback.print_exc()
        with PROGRESS_LOCK:
            PROGRESS["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {e}")
    finally:
        with PROGRESS_LOCK:
            PROGRESS["running"] = False


def start_analysis(cfg):
    with PROGRESS_LOCK:
        if PROGRESS["running"]:
            return False
        PROGRESS.update(running=True, stage="starting", done=0, total=0, error=None, log=[])
    STATE["meta"] = None
    threading.Thread(target=run_analysis, args=(cfg,), daemon=True).start()
    return True


_POOL = ThreadPoolExecutor(max_workers=4)


def combine(warps, covs):
    """Per-pixel median over the cameras that cover each pixel. On the floor
    every camera agrees; the rat, which stands above it, is smeared by each
    camera away from itself in a different direction, so a median of 3 keeps
    the view shared by the majority. 2 cameras: mean; 1: that camera."""
    if len(warps) == 3:
        # median of three with uint8 min/max only, then the pixels fewer
        # cameras reach: pair means / single cameras, copied through masks
        a, b, c = warps
        out = cv2.min(cv2.max(cv2.min(a, b), c), cv2.max(a, b))
        for idx, mask in _coverage_masks(covs):
            if len(idx) == 2:
                src = cv2.addWeighted(warps[idx[0]], 0.5, warps[idx[1]], 0.5, 0)
            elif len(idx) == 1:
                src = warps[idx[0]]
            else:
                src = np.zeros_like(a)
            cv2.copyTo(src, mask, out)
        return out
    M = np.stack(covs)                                      # (k,H,W)
    cnt = M.sum(0)
    wts = (M / np.maximum(cnt, 1)).astype(np.float32)[..., None]
    mean = sum(w * wt for w, wt in zip(warps, wts))
    if len(warps) < 3:
        return (mean + 0.5).astype(np.uint8)
    Ws = np.where(M[..., None], np.stack(warps).astype(np.float32), np.inf)
    Ws.sort(axis=0)
    mid = np.clip((cnt - 1) // 2, 0, len(warps) - 1)
    med = np.clip(np.take_along_axis(Ws, mid[None, ..., None].repeat(3, -1), axis=0)[0], 0, 255)
    return np.where((cnt >= 3)[..., None], med, mean + 0.5).astype(np.uint8)


_MASK_CACHE = {}


def _coverage_masks(covs):
    """[(camera indices, uint8 mask)] for every proper subset of the 3 cameras
    that is exactly the set covering some pixels. Cached for the fixed full-view
    coverages (the rat patch's change every frame and are cheap anyway)."""
    key = tuple(id(c) for c in covs)
    hit = _MASK_CACHE.get(key)
    if hit is not None and all(h is c for h, c in zip(hit[0], covs)):
        return hit[1]
    code = sum(c.astype(np.uint8) << i for i, c in enumerate(covs))
    masks = []
    for v in range(7):                                   # 7 = all three
        m = code == v
        if m.any():
            masks.append(([i for i in range(3) if v >> i & 1], m.astype(np.uint8)))
    if len(_MASK_CACHE) > 8:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = (list(covs), masks)
    return masks


RAT_PATCH_RADIUS_M = 0.11     # disk around the tracked rat drawn from its own body plane
RAT_PATCH_FEATHER_M = 0.03


def rat_patch(frame, images, arr):
    """Redraw a disk around each tracked rat from the plane at its triangulated
    body height instead of the floor: on the floor plane each camera smears the
    rat away from itself, on its own plane the cameras line its body up."""
    if len(images) < 2:
        return arr
    for rs in STATE["rats"]:
        i = rs["frame_index"].get(frame)
        s = rs["series"]
        if i is not None and s["has_pos"][i] and s["body"][i] is not None:
            arr = _body_patch(s["body"][i], images, arr)
    return arr


def _body_patch(body, images, arr):
    meta = STATE["meta"]
    bx, by, bz = body
    xmin, xmax, ymin, ymax = meta["view_box"]
    W, H = OUT_SIZE
    sx, sy = (xmax - xmin) / W, (ymax - ymin) / H
    r_px = int(np.ceil(RAT_PATCH_RADIUS_M / sx))
    cx, cy = int(round((bx - xmin) / sx)), int(round((ymax - by) / sy))
    x0, x1, y0, y1 = max(cx - r_px, 0), min(cx + r_px, W), max(cy - r_px, 0), min(cy + r_px, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return arr
    sub_box = (xmin + x0 * sx, xmin + x1 * sx, ymax - y1 * sy, ymax - y0 * sy)
    size = (x1 - x0, y1 - y0)
    B, tc, cams = STATE["rig"]
    warps, covs = [], []
    for view, img in images:
        mx, my = tv.plane_remap(cams[view["name"]], B, tc, sub_box, size, float(bz))
        h, w = img.shape[:2]
        warps.append(cv2.cvtColor(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_BGR2RGB))
        covs.append((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1))
    patch = combine(warps, covs).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot((xx - cx) * sx, (yy - cy) * sy)
    alpha = np.clip((RAT_PATCH_RADIUS_M - d) / RAT_PATCH_FEATHER_M, 0, 1)[..., None]
    out = arr.copy()
    out[y0:y1, x0:x1] = (alpha * patch + (1 - alpha) * arr[y0:y1, x0:x1]).astype(np.uint8)
    return out


def read_view(view, frame):
    """This camera's raw BGR frame, or None."""
    with view["lock"]:
        cap = view["cap"]
        if not 0 <= frame < view["n_frames"]:
            return None
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
    return img if ok else None


MASK_TINT_ALPHA = 0.16                      # fill strength: low, so fur and cap details stay visible
MASK_EDGE_ALPHA = 0.45                      # thin outline, so the translucent fill still reads
MASK_COLORS = {1: (255, 40, 40), 2: (40, 110, 255)}   # object 1 red, object 2 blue


def read_mask_bgr(view, frame):
    """This camera's GUI 2 mask as a BGR image the top-view warp can treat like
    a video frame: object 1 in the red channel, object 2 in the blue one
    (255 = inside). Zeros if the mask file is missing."""
    p = ac.masks_dir_for(STATE["config"]["masks_root"], view["name"]) / f"frame_{int(frame):06d}.png"
    if not p.exists():
        return None
    lab = np.asarray(Image.open(p))
    out = np.zeros(lab.shape + (3,), np.uint8)
    out[..., 2] = (lab == 1) * 255
    out[..., 0] = (lab == 2) * 255
    return out


def tint_masks(frame, images, arr):
    """Tint both rats in the composed top view, warped exactly like the video
    (floor-plane median, each tracked rat's disk re-warped at its body height)
    so the colour lands on the rat as it is drawn."""
    masks = list(_POOL.map(lambda vi: read_mask_bgr(vi[0], frame), images))
    mimgs = [(v, m if m is not None else np.zeros_like(r)) for (v, r), m in zip(images, masks)]
    warps = [cv2.cvtColor(cv2.remap(m, *v["map"], cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_BGR2RGB) for v, m in mimgs]
    if len(warps) == 1:
        mk = warps[0]
    else:
        mk = rat_patch(frame, mimgs, combine(warps, [v["coverage"] for v, _ in mimgs]))
    out = arr.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label, ch in ((1, 0), (2, 2)):
        m = mk[..., ch]
        if not m.any():
            continue
        a = (m.astype(np.float32) / 255.0 * MASK_TINT_ALPHA)[..., None]
        edge = cv2.morphologyEx((m > 127).astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
        a[edge] = MASK_EDGE_ALPHA
        out = out * (1 - a) + np.float32(MASK_COLORS[label]) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def render_topview_jpeg(frame, masks=False):
    cache = STATE["frame_cache"]
    key = (frame, bool(masks))
    if key in cache:
        return cache[key]
    views = STATE["views"]
    raws = list(_POOL.map(lambda v: read_view(v, frame), views))
    images = [(v, r) for v, r in zip(views, raws) if r is not None]
    if images:
        warps = [cv2.cvtColor(cv2.remap(r, *v["map"], cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_BGR2RGB)
                 for v, r in images]
        if len(images) == 1:
            arr = warps[0]
        else:
            arr = combine(warps, [v["coverage"] for v, _ in images])
            arr = rat_patch(frame, images, arr)
        if masks:
            arr = tint_masks(frame, images, arr)
        im = Image.fromarray(arr)
    else:
        im = Image.new("RGB", OUT_SIZE, (17, 17, 17))
        ImageDraw.Draw(im).text((16, 16), f"frame {frame} not in the video", fill=(200, 200, 200))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    cache[key] = data
    STATE["frame_cache_order"].append(key)
    if len(STATE["frame_cache_order"]) > FRAME_CACHE_SIZE:
        cache.pop(STATE["frame_cache_order"].pop(0), None)
    return data


# =============================================================================
# Head position from clicks
# =============================================================================

HEAD_LOCK = threading.Lock()


def rat_state(rid):
    rs = next((r for r in STATE["rats"] if r["id"] == int(rid)), None)
    if rs is None:
        raise ValueError(f"rat {rid} is not analysed (tick it in the setup)")
    return rs


def save_rat_settings(rid, **kv):
    """Write some of one rat's settings to session_config.json (the file is
    rewritten in the two-rat layout, also if it was a single-rat one)."""
    cfg = fill_defaults(load_config())
    cfg["rats"][int(rid) - 1].update(kv)
    with CONFIG_PATH.open("w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def apply_head_clicks(rid, clicks, reach):
    """Refit one rat's head placement from its clicks, update its served
    series and summary, persist the clicks. Returns what the page needs to
    redraw."""
    with HEAD_LOCK:
        rs = rat_state(rid)
        def norm(c):
            out = {"frame": int(c["frame"])}
            if c.get("x") is not None:
                out["x"], out["y"] = float(c["x"]), float(c["y"])
            if c.get("q") is not None:                  # full orientation (3D inset) wins over a heading
                out["q"] = [float(v) for v in c["q"]]
            elif c.get("heading_deg") is not None:
                out["heading_deg"] = float(c["heading_deg"])
            return out
        clicks = sorted((norm(c) for c in clicks), key=lambda c: c["frame"])
        hp = ac.place_head(rs["ctx"], clicks, int(reach))
        clean = lambda a: [None if np.isnan(r[0]) else [float(v) for v in r] for r in a]
        series = rs["series"]
        series["head"], series["head_top"], series["nose_top"] = clean(hp["head"]), clean(hp["head_top"]), clean(hp["nose_top"])
        if rs.get("q_vision") is None:
            rs["q_vision"], rs["fwd_vision"] = rs["ctx"]["q_vision"], rs["ctx"]["fwd"].tolist()
        series["q"] = hp["q"].tolist() if hp["q"] is not None else rs["q_vision"]
        series["fwd"] = hp["fwd"].tolist() if hp["q"] is not None else rs["fwd_vision"]
        summary = rs["summary"]
        summary["offset_head_frame_m"] = hp["offset_H"].tolist()
        summary["head_fit"] = hp["fit"]
        rs["head_clicks"], rs["head_click_reach_frames"] = clicks, int(reach)
        rm = next(m for m in STATE["meta"]["rats"] if m["id"] == rs["id"])
        rm["head_clicks"], rm["head_click_reach_frames"] = clicks, int(reach)
        save_rat_settings(rs["id"], head_clicks=clicks, head_click_reach_frames=int(reach))
        log(f"rat {rs['id']} head position: {len(clicks)} clicked frames ({hp['fit']['n_heading_clicks']} with a direction), "
            f"reach {reach} frames"
            + (f", rigid-offset click error {hp['fit']['rms_before_cm']:.1f} -> {hp['fit']['rms_after_cm']:.1f} cm"
               if hp["fit"]["n_clicks"] else ""))
        return {"ok": True, "rat": rs["id"], "saved_to": str(CONFIG_PATH), "head_clicks": clicks,
                "head_click_reach_frames": int(reach), "fit": hp["fit"], "offset_H": hp["offset_H"].tolist(),
                "head": series["head"], "head_top": series["head_top"], "nose_top": series["nose_top"], "q": series["q"]}


def adjust_mount(rid, frame, q_target):
    """One rat's head was rotated in its 3D inset to q_target at this frame and
    that is meant for EVERY frame: fold the rotation into that rat's IMU
    mounting tilt (R_W<-H = ... R_S<-H C, so C -> C * R^T dR R, which gives
    exactly q_target here), keep its roll and pitch in manual mode and re-run
    the analysis. Turning about the head's own up axis is dropped: the heading
    anchors re-align the heading from the video anyway."""
    from scipy.spatial.transform import Rotation
    import math
    rs = rat_state(rid)
    i = rs["frame_index"].get(frame)
    if i is None:
        raise ValueError(f"frame {frame} is not in rat {rs['id']}'s analysis")
    w, x, y, z = rs["series"]["q"][i]
    R_cur = Rotation.from_quat([x, y, z, w])
    w, x, y, z = q_target
    dR = Rotation.from_quat([x, y, z, w]) * R_cur.inv()
    t = rs["summary"].get("mount_tilt") or {"roll_deg": 0.0, "pitch_deg": 0.0}
    C_old = Rotation.from_euler("YX", [math.radians(t["pitch_deg"]), -math.radians(t["roll_deg"])])
    C_new = C_old * (R_cur.inv() * dR * R_cur)
    pitch, roll_c, yaw = C_new.as_euler("YXZ", degrees=True)
    roll = -roll_c
    cfg = save_rat_settings(rs["id"], mount_tilt_mode="manual", mount_roll_deg=round(float(roll), 2),
                            mount_pitch_deg=round(float(pitch), 2))
    log(f"rat {rs['id']} IMU mounting set from the 3D view at frame {frame}: roll {roll:+.1f}, pitch {pitch:+.1f} deg "
        f"(turn about the head's up axis {yaw:+.1f} deg left to the heading anchors); re-running")
    started = start_analysis(cfg)
    return {"ok": started, "rat": rs["id"], "rerun": True, "mount_roll_deg": float(roll), "mount_pitch_deg": float(pitch),
            "dropped_yaw_deg": float(yaw), "error": None if started else "an analysis is already running"}


def output_px_to_table(px, py):
    xmin, xmax, ymin, ymax = STATE["meta"]["view_box"]
    W, H = OUT_SIZE
    return [xmin + px / W * (xmax - xmin), ymax - py / H * (ymax - ymin)]


# =============================================================================
# Export: trajectory CSV/JSON and the video (frames composited by the page)
# =============================================================================

EXPORT_DIR_DEFAULT = TOOL_DIR / "exports"
EXPORT_JOBS = {}
EXPORT_LOCK = threading.Lock()


def ffmpeg_exe():
    import shutil
    for c in (shutil.which("ffmpeg"), str(Path(sys.executable).parent / "ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if c and Path(c).exists():
            return c
    return None


_ENCODER = {}
ENCODER_CANDIDATES = [                  # first one that really encodes wins
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]),       # most Linux / Homebrew builds
    ("libopenh264", ["-c:v", "libopenh264", "-b:v", "10M"]),                   # conda-forge LGPL builds
    ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-b:v", "10M"]),       # macOS
    ("mpeg4", ["-c:v", "mpeg4", "-q:v", "3"]),                                 # always there, not H.264
]


def video_encoder_args(exe):
    """Encoder arguments for this ffmpeg: each candidate test-encodes one tiny
    frame (a listed encoder can still be unusable, e.g. VideoToolbox off macOS)."""
    if exe in _ENCODER:
        return _ENCODER[exe]
    for name, args in ENCODER_CANDIDATES:
        try:
            r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                                "-frames:v", "1", "-pix_fmt", "yuv420p", *args, "-f", "null", "-"],
                               capture_output=True, timeout=30)
        except Exception:
            continue
        if r.returncode == 0:
            _ENCODER[exe] = args
            log(f"export: video encoder {name}")
            return args
    raise ValueError(f"none of {', '.join(n for n, _ in ENCODER_CANDIDATES)} works with {exe}")


def export_frame_range(req):
    meta = STATE["meta"]
    f0 = int(req.get("frame_from", meta["frame_min"]))
    f1 = int(req.get("frame_to", meta["frame_max"]))
    f0, f1 = max(f0, meta["frame_min"]), min(f1, meta["frame_max"])
    if f1 < f0:
        raise ValueError(f"empty frame range {f0}..{f1}")
    return f0, f1


def export_base(req, f0, f1, kind, tag=None):
    """<out_dir>/<recording>_<kind>_<first>-<last>[_<tag>]_<stamp> (GUI 5.3
    reads the first frame of a skull video from this name)."""
    out_dir = Path(req.get("out_dir") or EXPORT_DIR_DEFAULT).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{STATE['meta']['recording']}_{kind}_{f0}-{f1}_{tag + '_' if tag else ''}{stamp}"


def write_trajectory(base, f0, f1):
    """<base>_rat<n>.csv per rat (one row per frame) + <base>.json (what produced them)."""
    from scipy.spatial.transform import Rotation
    meta = STATE["meta"]
    fps = meta["fps"]
    xmin, xmax, ymin, ymax = meta["view_box"]
    W, H = OUT_SIZE
    zf = float(meta["floor_z_m"])
    cols = ["frame", "time_s", "has_position", "state",
            "body_x_m", "body_y_m", "body_z_m", "head_x_m", "head_y_m", "head_z_m", "head_height_above_floor_m",
            "heading_deg", "pitch_nose_up_deg", "roll_right_ear_down_deg", "qw", "qx", "qy", "qz",
            "head_video_px_x", "head_video_px_y", "head_clicked"]
    fmt = lambda v, d=5: "" if v is None else f"{v:.{d}f}"
    files = []
    for rs in STATE["rats"]:
        series, idx = rs["series"], rs["frame_index"]
        clicked = {c["frame"] for c in rs["head_clicks"]}
        lines = [",".join(cols)]
        for f in range(f0, f1 + 1):
            i = idx.get(f)
            if i is None:
                continue
            q = series["q"][i]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            yaw, pitch_down, roll = np.degrees(R.as_euler("ZYX"))
            has = bool(series["has_pos"][i])
            b = series["body"][i] if has else None
            h = series["head"][i] if has else None
            ht = series["head_top"][i] if has else None
            px = [(ht[0] - xmin) / (xmax - xmin) * W, (ymax - ht[1]) / (ymax - ymin) * H] if ht else [None, None]
            lines.append(",".join([
                str(f), f"{f / fps:.4f}", "1" if has else "0", series["state"][i],
                *(fmt(v) for v in (b or [None] * 3)), *(fmt(v) for v in (h or [None] * 3)),
                fmt(h[2] - zf if h else None),
                f"{yaw:.2f}", f"{-pitch_down:.2f}", f"{roll:.2f}", *(f"{v:.6f}" for v in q),
                fmt(px[0], 1), fmt(px[1], 1), "1" if f in clicked else "0"]))
        p = base.parent / f"{base.name}_rat{rs['id']}.csv"
        p.write_text("\n".join(lines) + "\n")
        files.append(str(p))
    cfg = STATE["config"]
    info = {
        "frames": [f0, f1], "fps": fps, "recording": meta["recording"],
        "coordinates": "table frame of the calibration (x, y horizontal, z up, metres); "
                       f"the arena floor is at z = {zf:.4f} m",
        "angles": "head frame x = nose, y = left, z = up; heading = atan2 of the nose direction in the table frame; "
                  "pitch positive = nose up; roll positive = right ear down (ZYX Euler angles of the quaternion); "
                  "the IMU mounting tilt in summary.mount_tilt is already removed",
        "quaternion": "qw,qx,qy,qz rotating head-frame vectors into the table frame (IMU pitch/roll, heading re-aligned)",
        "head_video_px": "where the head marker is drawn in the exported video (800x800 top view, origin top-left)",
        "view_box_table_m": meta["view_box"], "video_size_px": list(OUT_SIZE), "view_mode": meta["view_mode"],
        "view_cameras": meta["view_cameras"],
        "rats": [{"id": rs["id"], "csv": Path(fp).name,
                  "colour": f"rgb{MASK_COLORS[rs['id']]} (object {rs['id']} in the masks)",
                  "head_clicks": rs["head_clicks"], "head_click_reach_frames": rs["head_click_reach_frames"],
                  "summary": rs["summary"],
                  "config": {k: v for k, v in cfg["rats"][rs["id"] - 1].items() if k != "head_clicks"}}
                 for rs, fp in zip(STATE["rats"], files)],
        "config": {k: v for k, v in cfg.items() if k != "rats"},
    }
    base.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return files + [str(base.with_suffix(".json"))]


EXPORT_KINDS = {"video": ("video", None), "skull": ("skull", "both"),
                "skull_rat1": ("skull", "rat1"), "skull_rat2": ("skull", "rat2")}


def export_video_start(req):
    exe = ffmpeg_exe()
    if exe is None:
        raise ValueError("ffmpeg not found (needed to write the video)")
    f0, f1 = export_frame_range(req)
    kind = req.get("kind") if req.get("kind") in EXPORT_KINDS else "video"
    base = export_base(req, f0, f1, *EXPORT_KINDS[kind])
    out = base.with_suffix(".mp4")
    fps = float(req.get("fps") or STATE["meta"]["fps"])
    proc = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:g}",
                             "-c:v", "mjpeg", "-i", "-", *video_encoder_args(exe),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    files = write_trajectory(base, f0, f1) if req.get("trajectory", True) else []
    job = uuid.uuid4().hex[:12]
    with EXPORT_LOCK:
        EXPORT_JOBS[job] = {"proc": proc, "path": str(out), "n": 0, "lock": threading.Lock(), "files": files}
    log(f"export: writing {out} (frames {f0}..{f1})")
    return {"ok": True, "job": job, "kind": kind, "frame_from": f0, "frame_to": f1, "video": str(out), "trajectory": files}


def export_video_frame(job_id, data):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise ValueError("unknown export job")
    with job["lock"]:
        job["proc"].stdin.write(data)
        job["n"] += 1


def export_video_finish(job_id):
    with EXPORT_LOCK:
        job = EXPORT_JOBS.pop(job_id, None)
    if job is None:
        raise ValueError("unknown export job")
    with job["lock"]:
        job["proc"].stdin.close()
        err = job["proc"].stderr.read().decode(errors="replace").strip()
        code = job["proc"].wait()
    if code != 0:
        raise ValueError(f"ffmpeg failed ({code}): {err[-400:]}")
    log(f"export: {job['n']} frames -> {job['path']}")
    return {"ok": True, "video": job["path"], "frames": job["n"], "trajectory": job["files"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_bytes(self, data, content_type, status=200):
        self.send_response(status); self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type=None):
        if not path.is_file():
            self.send_response(404); self._cors(); self.end_headers(); return
        self._send_bytes(path.read_bytes(), content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")

    def _send_json(self, obj, status=200):
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json", status)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        try:
            self._get()
        except Exception as e:
            traceback.print_exc()
            try:
                self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)
            except Exception:
                pass

    def _get(self):
        url = urlparse(self.path)
        path, qs = url.path, parse_qs(url.query)
        if path in ("/", "/index.html"):
            self._send_file(TOOL_DIR / "index.html", "text/html"); return
        if path == "/bundle.js":
            bp = TOOL_DIR / "bundle.js"
            if bp.is_file():
                self._send_file(bp, "application/javascript")
            else:
                self._send_bytes(b"console.error('bundle.js missing: run npm install && npm run build');", "application/javascript")
            return
        if path == "/api/config":
            self._send_json(fill_defaults(load_config())); return
        if path == "/api/progress":
            with PROGRESS_LOCK:
                p = dict(PROGRESS); p["log"] = p["log"][-12:]
            p["ready"] = STATE["meta"] is not None
            self._send_json(p); return
        if path == "/api/videos":
            cfg = fill_defaults({**load_config(), "video_root": qs.get("root", [""])[0],
                                 "calib_json_path": qs.get("calib", [""])[0]})
            ok = cfg["video_root"] and Path(cfg["video_root"]).is_dir() and Path(cfg["calib_json_path"]).is_file()
            self._send_json({"cameras": cameras_with_video(cfg) if ok else []}); return
        if path == "/api/cameras":
            calib = qs.get("calib", [""])[0]
            if not calib or not Path(calib).expanduser().exists():
                self._send_json({"cameras": []}); return
            self._send_json({"cameras": sorted(tv.load_camera_extrinsics(calib))}); return
        if path in ("/api/meta", "/api/series"):
            if STATE["meta"] is None:
                self._send_json({"error": "analysis not ready"}, status=400); return
            self._send_json(STATE["meta"] if path == "/api/meta" else {"rats": [rs["series"] for rs in STATE["rats"]]})
            return
        if path == "/api/topview_frame.jpg":
            if STATE["meta"] is None:
                self._send_json({"error": "analysis not ready"}, status=400); return
            try:
                frame = int(qs.get("frame", ["0"])[0])
            except ValueError:
                self._send_json({"error": "frame must be an integer"}, status=400); return
            masks = qs.get("masks", ["0"])[0] not in ("0", "", "false")
            self._send_bytes(render_topview_jpeg(frame, masks), "image/jpeg"); return
        if path.startswith("/assets/"):
            rel = path[len("/assets/"):]
            if ".." in rel:
                self.send_response(400); self._cors(); self.end_headers(); return
            self._send_file(TOOL_DIR / "assets" / rel); return
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
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/config":
            payload = json.loads(body)
            if not isinstance(payload, dict):
                self._send_json({"ok": False, "error": "config must be a JSON object"}, status=400); return
            old = fill_defaults(load_config())
            new_video = payload.get("video_root") not in (None, "", old.get("video_root"))
            rats = []
            for n, o in enumerate(old["rats"], start=1):
                p = (payload.get("rats") or [None] * N_RATS)[n - 1] or {}
                r = {**o, **p}
                if new_video or p.get("quat_csv_path") not in (None, "", o.get("quat_csv_path")):
                    r.pop("head_clicks", None)    # clicked head positions belong to the old recording / IMU
                if p.get("anchor_mesh_dir") not in (None, "", o.get("anchor_mesh_dir")) and "anchor_frame" not in p:
                    r.pop("anchor_frame", None)   # re-read it from the new folder's name
                rats.append(r)
            cfg = fill_defaults({**old, **{k: v for k, v in payload.items() if k != "rats"}, "rats": rats})
            err = validate(cfg)
            if err:
                self._send_json({"ok": False, "error": err, "config": cfg}, status=400); return
            with CONFIG_PATH.open("w") as f:
                json.dump(cfg, f, indent=2)
            if not start_analysis(cfg):
                self._send_json({"ok": False, "error": "an analysis is already running"}, status=409); return
            self._send_json({"ok": True, "config": cfg}); return
        if path.startswith("/api/export/frame"):
            export_video_frame(parse_qs(urlparse(self.path).query).get("job", [""])[0], body)
            self._send_json({"ok": True}); return
        if path in ("/api/head_click", "/api/head_clicks", "/api/head_orient", "/api/export/start", "/api/export/finish",
                    "/api/export/trajectory") and STATE["meta"] is None:
            self._send_json({"ok": False, "error": "analysis not ready"}, status=400); return
        if path in ("/api/head_click", "/api/head_orient", "/api/head_clicks"):
            # every head edit is for one rat: {rat: 1 | 2, ...} (default 1)
            payload = json.loads(body)
            try:
                rs = rat_state(payload.get("rat") or 1)
            except ValueError as e:
                self._send_json({"ok": False, "error": str(e)}, status=400); return
            rid = rs["id"]
        if path == "/api/head_click":
            # {rat, frame, px, py[, px2, py2]}: a click on the top view at output
            # pixel (px, py), dragged to (px2, py2) = towards the nose
            frame = int(payload["frame"])
            to = (output_px_to_table(float(payload["px2"]), float(payload["py2"]))
                  if payload.get("px2") is not None else None)
            try:
                xy, heading = ac.click_to_head_xy(rs["ctx"], frame,
                                                  output_px_to_table(float(payload["px"]), float(payload["py"])), to)
            except ValueError as e:
                self._send_json({"ok": False, "error": f"rat {rid}: {e}"}, status=400); return
            old = next((c for c in rs["head_clicks"] if c["frame"] == frame), {})
            clicks = [c for c in rs["head_clicks"] if c["frame"] != frame]
            new = {"frame": frame, "x": xy[0], "y": xy[1]}
            if heading is not None:
                new["heading_deg"] = heading            # a dragged direction replaces an earlier orientation
            else:                                       # a plain click keeps the frame's orientation fix
                new.update({k: old[k] for k in ("q", "heading_deg") if k in old})
            clicks.append(new)
            self._send_json(apply_head_clicks(rid, clicks, rs["head_click_reach_frames"])); return
        if path == "/api/head_orient":
            # {rat, frame, q, scope}: the head orientation set by rotating it in
            # the rat's 3D inset (wxyz, table frame). scope "frame": re-align the
            # gyroscope at this frame (+- reach); "mount": change this rat's IMU
            # mounting for all frames
            frame = int(payload["frame"])
            q = [float(v) for v in payload["q"]]
            if payload.get("scope") == "mount":
                self._send_json(adjust_mount(rid, frame, q)); return
            old = next((c for c in rs["head_clicks"] if c["frame"] == frame), {})
            clicks = [c for c in rs["head_clicks"] if c["frame"] != frame]
            new = {"frame": frame, "q": q}
            if "x" in old:
                new["x"], new["y"] = old["x"], old["y"]
            clicks.append(new)
            self._send_json(apply_head_clicks(rid, clicks, rs["head_click_reach_frames"])); return
        if path == "/api/head_clicks":
            # {rat, clicks?, reach?}: replace the rat's list (delete / clear) and/or its reach
            clicks = payload.get("clicks", rs["head_clicks"])
            reach = int(payload.get("reach", rs["head_click_reach_frames"]) or 0)
            self._send_json(apply_head_clicks(rid, clicks, max(reach, 0))); return
        if path == "/api/export/trajectory":
            req = json.loads(body or b"{}")
            f0, f1 = export_frame_range(req)
            files = write_trajectory(export_base(req, f0, f1, "trajectory"), f0, f1)
            log(f"export: trajectory frames {f0}..{f1} -> {files[0]}")
            self._send_json({"ok": True, "trajectory": files}); return
        if path == "/api/export/start":
            self._send_json(export_video_start(json.loads(body or b"{}"))); return
        if path == "/api/export/finish":
            self._send_json(export_video_finish(json.loads(body)["job"])); return
        if path == "/api/align":
            # separate from /api/config: called live while the cap alignment
            # sliders move, must not restart the analysis
            payload = json.loads(body)
            cfg = load_config()
            cfg["cap_align_deg"] = payload.get("cap_align_deg", [0, 0, 0])
            with CONFIG_PATH.open("w") as f:
                json.dump(cfg, f, indent=2)
            if STATE["meta"] is not None:
                STATE["meta"]["cap_align_deg"] = cfg["cap_align_deg"]
            self._send_json({"ok": True}); return
        self.send_response(404); self._cors(); self.end_headers()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="GUI 5: Trajectory Tracing (top-down video + head IMU)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GUI5_PORT", PORT)),
                    help=f"port to serve on (default {PORT}, or $GUI5_PORT)")
    port = ap.parse_args().port
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on port {port} ({e}); is another server.py still running?\n"
                 f"  see what holds it:  ss -ltnp | grep :{port}   (Linux)   or   lsof -iTCP:{port} -sTCP:LISTEN   (macOS)\n"
                 f"  or use another port: python server.py --port {port + 1}")
    cfg = fill_defaults(load_config())
    if validate(cfg) is None:
        print("saved config is complete; starting the analysis", flush=True)
        start_analysis(cfg)
    print(f"GUI 5 (Trajectory Tracing) running at http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
