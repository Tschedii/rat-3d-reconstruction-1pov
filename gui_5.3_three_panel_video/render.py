#!/usr/bin/env python3
"""
Three-panel video of one tracked rat:

  left         the raw camera video, both rats tinted from their GUI 2 masks
               (tracked rat red, the other blue, as in GUI 5), with the tracked
               rat's head marked and its recent head path drawn
  top right    the rotating IMU skull, taken frame by frame from GUI 5's skull
               video export
  bottom right the head path in 3D (table frame), drawn up to the current
               frame, with the current head, its height above the floor and its
               heading

The head path and heading come from a head tracking export
(head_pose_world.csv, GUI 5.2 head tracking). Frames are matched by their
recording frame number: camera video frame n is recording frame n, the skull
video's first frame is skull_first_frame.

Two rats: head_tracking_dir may be a GUI 5.2 two-rat run folder (rat1/ and
rat2/ subfolders, each a single-rat export) and skull_video_path GUI 5.1's
side-by-side skull export (kind "skull": rat 1 | rat 2). Then rat n is object
n in the masks and is drawn in its GUI 5.1 / 5.2 colour (rat 1 red, rat 2
blue): its tint, head and head path on the camera view, its skull panel (the
two skulls side by side on top) and its path in the 3D panel (below them,
twice as wide).

Usable from the command line (python render.py config.json) or through
server.py's GUI.
"""

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

DEFAULTS = {
    "calib_json_path": "",
    "camera_video_path": "",
    "masks_root": "",
    "head_tracking_dir": "",
    "skull_video_path": "",
    "skull_first_frame": None,
    "frame_start": None,
    "frame_end": None,
    "output_path": "",
    "height_px": 1080,
    "trail_seconds": 4.0,
    "tint_alpha": 0.45,
    "floor_z_m": 0.055,
    "orbit_deg_per_s": 4.0,
    "elevation_deg": 28.0,
}

HEAD_BGR = (30, 99, 224)        # orange, the head everywhere (#e0631e)
TRACKED_BGR = (48, 48, 226)     # red tint, the tracked rat (GUI 5: "IMU rat ... (red)")
OTHER_BGR = (214, 120, 40)      # blue tint, the other rat
PANEL_BG = "#14120e"            # matches GUI 5's skull video background
TEXT = "#e8e4da"
MUTED = "#8a867c"
# two-rat videos: rat n = object n, in GUI 5.1 / 5.2's colours rgb(255,40,40) / rgb(40,110,255)
RAT_BGR = {1: (40, 40, 255), 2: (255, 110, 40)}
RAT_HEX = {1: "#ff2828", 2: "#286eff"}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def camera_number(path):
    m = re.search(r"cam_(\d+)", Path(path).name)
    if not m:
        raise ValueError(f"cannot tell the rig camera from {Path(path).name}: expected a name like cam_1.mp4")
    return int(m.group(1))


def load_camera(calib_json_path, cam_no):
    data = json.loads(Path(calib_json_path).read_text())
    name = f"camera_{cam_no:03d}"
    for c in data.get("cameras", []):
        if c["name"] == name:
            return c
    raise ValueError(f"{name} is not in {calib_json_path}")


def project(points_C, cam):
    """(N,3) calibration-frame points -> (N,2) pixels; pinhole + k1, as the rig JSON."""
    P = np.atleast_2d(np.asarray(points_C, dtype=np.float64))
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    i = cam["intrinsics"]
    X = P @ R.T + t
    z = np.where(np.abs(X[:, 2]) < 1e-9, 1e-9, X[:, 2])
    x, y = X[:, 0] / z, X[:, 1] / z
    d = 1.0 + float(i.get("k1", 0.0)) * (x * x + y * y)
    return np.stack([i["fx"] * x * d + i["cx"], i["fy"] * y * d + i["cy"]], axis=1)


def load_head_tracking(export_dir):
    """head_pose_world.csv -> dict of per-frame arrays, indexed by recording frame."""
    path = Path(export_dir) / "head_pose_world.csv"
    if not path.exists():
        raise FileNotFoundError(f"no head_pose_world.csv in {export_dir}")

    def num(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else np.nan

    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {"frame": np.array([int(r["frame"]) for r in rows])}
    for key, cols in {
        "head_T": ("head_x_m", "head_y_m", "head_z_m"),
        "body_T": ("body_x_m", "body_y_m", "body_z_m"),
        "fwd_T": ("fwd_x", "fwd_y", "fwd_z"),
        "head_C": ("head_calib_x_m", "head_calib_y_m", "head_calib_z_m"),
    }.items():
        out[key] = np.array([[num(r, c) for c in cols] for r in rows])
    out["state"] = [r.get("state", "") for r in rows]
    summary = Path(export_dir) / "summary.json"
    out["fps"] = float(json.loads(summary.read_text()).get("fps", 30.0)) if summary.exists() else 30.0
    out["mask_label"] = 1
    alignment = Path(export_dir) / "alignment.json"
    if alignment.exists():
        out["mask_label"] = int(json.loads(alignment.read_text()).get("mask_label", 1))
    return out


def load_tracks(export_dir):
    """One load_head_tracking dict per rat, in mask-label order: a single-rat
    export (head_pose_world.csv in the folder) or a GUI 5.2 two-rat run folder
    (rat1/, rat2/ subfolders, each a single-rat export)."""
    d = Path(export_dir)
    if (d / "head_pose_world.csv").exists():
        return [load_head_tracking(d)]
    subs = [p for p in sorted(d.glob("rat[0-9]")) if (p / "head_pose_world.csv").exists()]
    if not subs:
        raise FileNotFoundError(f"no head_pose_world.csv in {d} or in rat<N>/ subfolders of it")
    return sorted((load_head_tracking(p) for p in subs), key=lambda t: t["mask_label"])


def aligned(ht, key, f0, f1):
    """ht[key] on the rendered frames f0..f1 (row k = frame f0 + k), NaN / ''
    where the export has no row."""
    fr = ht["frame"]
    m = (fr >= f0) & (fr <= f1)
    if key == "state":
        out = [""] * (f1 - f0 + 1)
        for f, s in zip(fr[m], np.asarray(ht["state"], dtype=object)[m]):
            out[int(f) - f0] = s
        return out
    out = np.full((f1 - f0 + 1, ht[key].shape[1]), np.nan)
    out[fr[m] - f0] = ht[key][m]
    return out


def skull_slots(path, size, labels):
    """{mask label: which square of the skull video shows that rat (None: not
    in it)}. GUI 5.1's side-by-side export has one square per rat in label
    order; a single-skull video shows the rat in its name (_rat<n>_), else the
    first rat."""
    w, h = size
    n = max(1, int(round(w / max(h, 1))))
    if n > 1:
        return {lab: (lab - 1 if lab - 1 < n else None) for lab in labels}
    m = re.search(r"_rat(\d)_", Path(path).name)
    shown = int(m.group(1)) if m else labels[0]
    return {lab: (0 if lab == shown else None) for lab in labels}


def skull_first_frame_from_name(path):
    """GUI 5 names its exports <recording>_skull_<first>-<last>_<stamp>.mp4."""
    m = re.search(r"_skull_(\d+)-(\d+)_", Path(path).name)
    return int(m.group(1)) if m else None


def video_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return n, size


def inspect(cfg):
    """What the inputs cover, and the frame range a render would use."""
    cfg = {**DEFAULTS, **cfg}
    for key, what in (("camera_video_path", "camera video"), ("calib_json_path", "rig calibration JSON"),
                      ("head_tracking_dir", "head tracking export folder"), ("skull_video_path", "skull video")):
        if not cfg[key]:
            raise ValueError(f"the {what} is not set")
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"the {what} does not exist: {cfg[key]}")
    tracks = load_tracks(cfg["head_tracking_dir"])
    ht = tracks[0]
    labels = [t["mask_label"] for t in tracks]
    cam_n, cam_size = video_frame_count(cfg["camera_video_path"])
    sk_n, sk_size = video_frame_count(cfg["skull_video_path"])
    sk0 = cfg["skull_first_frame"]
    if sk0 in (None, ""):
        sk0 = skull_first_frame_from_name(cfg["skull_video_path"])
    sk0 = int(sk0 or 0)
    ranges = {
        "head_tracking": [max(int(t["frame"].min()) for t in tracks), min(int(t["frame"].max()) for t in tracks)],
        "camera_video": [0, cam_n - 1],
        "skull_video": [sk0, sk0 + sk_n - 1],
    }
    lo = max(r[0] for r in ranges.values())
    hi = min(r[1] for r in ranges.values())
    f0 = int(cfg["frame_start"]) if cfg["frame_start"] not in (None, "") else lo
    f1 = int(cfg["frame_end"]) if cfg["frame_end"] not in (None, "") else hi
    if f0 < lo or f1 > hi or f1 < f0:
        raise ValueError(f"frames {f0}..{f1} are not covered by every input (common range {lo}..{hi}: {ranges})")
    cam_no = camera_number(cfg["camera_video_path"])
    load_camera(cfg["calib_json_path"], cam_no)
    slots = skull_slots(cfg["skull_video_path"], sk_size, labels)
    return {"ranges": ranges, "frame_start": f0, "frame_end": f1, "skull_first_frame": sk0,
            "camera": f"camera_{cam_no:03d}", "camera_size": cam_size, "skull_size": sk_size,
            "fps": ht["fps"], "mask_label": ht["mask_label"], "rats": labels,
            "skull_slots": {str(k): v for k, v in slots.items()}}


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def camera_panel(img, mask, heads, tracked_label, alpha, out_w, out_h, label, items):
    """heads: [(trail px (k,2), current px (2,), colour, rim colour)] per rat;
    items: the legend, [(colour, text)]."""
    if mask is not None:
        for lab_mask, color in ((mask == tracked_label, TRACKED_BGR), ((mask > 0) & (mask != tracked_label), OTHER_BGR)):
            if lab_mask.any():
                img[lab_mask] = ((1 - alpha) * img[lab_mask] + alpha * np.array(color)).astype(np.uint8)
    for head_px_trail, head_px_now, head_bgr, rim_bgr in heads:
        n = len(head_px_trail)
        for k in range(n - 1):
            p, q = head_px_trail[k], head_px_trail[k + 1]
            if np.isnan(p).any() or np.isnan(q).any():
                continue
            a = 0.25 + 0.75 * (k + 1) / max(1, n - 1)
            col = tuple(int(c * a) for c in head_bgr)
            cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), col, 3, cv2.LINE_AA)
        if head_px_now is not None and not np.isnan(head_px_now).any():
            c = tuple(np.round(head_px_now).astype(int))
            cv2.circle(img, c, 11, head_bgr, -1, cv2.LINE_AA)
            cv2.circle(img, c, 11, rim_bgr, 3, cv2.LINE_AA)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    s = out_h / 1080
    x = int(16 * s)
    for k, (col, text) in enumerate(items):
        y = int((34 + 30 * k) * s)
        cv2.rectangle(img, (x, y - int(14 * s)), (x + int(18 * s), y + int(4 * s)), col, -1)
        cv2.putText(img, text, (x + int(28 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, max(1, int(2 * s)))
    y = out_h - int(18 * s)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), (20, 18, 14), -1)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (232, 228, 218), max(1, int(2 * s)), cv2.LINE_AA)
    return img


class PathPanel:
    """The 3D head path(s), drawn with matplotlib. The figure and its artists
    are built once and only updated per frame, which is what keeps this fast.

    paths: one dict per rat with "head_T", "body_T", "fwd_T" on the rendered
    frames (row k = the k-th rendered frame, see aligned()) and its "color"
    (the head, matplotlib colour) and "body_rgba"."""

    def __init__(self, paths, size_px, floor_z, elev, azim0, orbit_deg_s, fps, footer, title="head path in 3D"):
        self.fps = fps
        self.elev, self.azim0, self.orbit = elev, azim0, orbit_deg_s
        self.floor = floor_z
        self.paths = []
        for p in paths:
            H, Bd = p["head_T"].copy(), p["body_T"].copy()
            # heights shown in cm above the arena floor
            H[:, 2] = (H[:, 2] - floor_z) * 100.0
            Bd[:, 2] = (Bd[:, 2] - floor_z) * 100.0
            H[:, :2] *= 100.0
            Bd[:, :2] *= 100.0
            self.paths.append({**p, "H": H, "Bd": Bd, "fwd": p["fwd_T"]})
        ok = [~np.isnan(p["H"][:, 0]) for p in self.paths]
        xy = np.concatenate([np.concatenate([p["H"][k, :2], p["Bd"][~np.isnan(p["Bd"][:, 0]), :2]])
                             for p, k in zip(self.paths, ok)])
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        c, half = (lo + hi) / 2, max(hi - lo) / 2 + 4.0
        zmax = max([10.0] + [float(np.nanmax(p["H"][:, 2])) + 2.0 for p, k in zip(self.paths, ok) if k.any()])

        dpi = 100
        w, h = size_px if isinstance(size_px, (tuple, list)) else (size_px, size_px)
        self.fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=PANEL_BG)
        ax = self.fig.add_axes([-0.04, 0.03, 1.08, 0.93], projection="3d")
        self.ax = ax
        ax.set_facecolor(PANEL_BG)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.10, 0.09, 0.07, 1.0))
            a._axinfo["grid"]["color"] = (0.30, 0.28, 0.24, 1.0)
            a._axinfo["grid"]["linewidth"] = 0.6
            a.label.set_color(MUTED)
            a.set_tick_params(colors=MUTED, labelsize=7, pad=0)
            a.set_major_locator(MaxNLocator(4))
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_zlim(0, zmax)
        # a wide panel (two rats) has room to draw the box larger
        ax.set_box_aspect((1, 1, 0.32), zoom=min(1.0 + 0.4 * (w / h - 1), 1.4))
        ax.set_xlabel("x (cm)", fontsize=8, labelpad=-2)
        ax.set_ylabel("y (cm)", fontsize=8, labelpad=-2)
        ax.set_zlabel("height (cm)", fontsize=8, labelpad=-4)

        # seeded with one degenerate segment: some matplotlib versions cannot
        # add an empty 3D collection
        p0 = np.array([c[0], c[1], 0.0])
        seed = [np.stack([p0, p0])]
        for p in self.paths:
            H = p["H"]
            p["future"], = ax.plot(H[:, 0], H[:, 1], H[:, 2], color=p["color"], alpha=0.12, lw=1.0)
            p["shadow"] = Line3DCollection(seed, colors=[(0.55, 0.52, 0.46, 0.35)], linewidths=1.0)
            p["body"] = Line3DCollection(seed, colors=[p["body_rgba"]], linewidths=0.8)
            p["past"] = Line3DCollection(seed, linewidths=2.0)
            for col in (p["shadow"], p["body"], p["past"]):
                ax.add_collection3d(col)
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_zlim(0, zmax)
        for p in self.paths:
            p["drop"], = ax.plot([], [], [], color="#e8e4da", lw=0.8, ls=":")
            p["arrow"], = ax.plot([], [], [], color="#f2c14e", lw=2.2)
            p["head"], = ax.plot([], [], [], "o", color=p["color"], markersize=7, markeredgecolor="#111", markeredgewidth=1.2)
            p["rgb"] = np.array(matplotlib.colors.to_rgb(p["color"]))
        self.fig.text(0.03 * h / w, 0.955, title, color=TEXT, fontsize=11, family="monospace")
        self.fig.text(0.03 * h / w, 0.02, footer, color=MUTED, fontsize=7, family="monospace")
        self.label = self.fig.text(1 - 0.03 * h / w, 0.955, "", color=MUTED, fontsize=9, ha="right", family="monospace")

    @staticmethod
    def _segments(P):
        seg = np.stack([P[:-1], P[1:]], axis=1)
        return seg[~np.isnan(seg).any(axis=(1, 2))]

    def render(self, i_local, t_s):
        for p in self.paths:
            self._render_path(p, i_local)
        self.ax.view_init(elev=self.elev, azim=self.azim0 + self.orbit * t_s)
        self.label.set_text(f"{t_s:6.2f} s")
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _render_path(self, p, i_local):
        H, Bd = p["H"], p["Bd"]
        past = H[: i_local + 1]
        seg = self._segments(past)
        n = len(seg)
        if n:
            age = np.linspace(0.25, 1.0, n)
            p["past"].set_segments(seg)
            p["past"].set_color(np.column_stack([np.tile(p["rgb"], (n, 1)), age]))
        else:
            p["past"].set_segments([])
        sh = past.copy()
        sh[:, 2] = 0.0
        p["shadow"].set_segments(self._segments(sh))
        p["body"].set_segments(self._segments(Bd[: i_local + 1]))
        h = H[i_local]
        if np.isnan(h).any():
            p["head"].set_data_3d([], [], [])
            p["drop"].set_data_3d([], [], [])
            p["arrow"].set_data_3d([], [], [])
        else:
            p["head"].set_data_3d([h[0]], [h[1]], [h[2]])
            p["drop"].set_data_3d([h[0], h[0]], [h[1], h[1]], [0.0, h[2]])
            f = p["fwd"][i_local]
            if not np.isnan(f).any():
                e = h + 5.0 * np.array([f[0], f[1], f[2]])  # 5 cm nose direction
                p["arrow"].set_data_3d([h[0], e[0]], [h[1], e[1]], [h[2], e[2]])
            else:
                p["arrow"].set_data_3d([], [], [])


def blank_panel(size, text):
    img = np.full((size, size, 3), (14, 18, 20), np.uint8)
    cv2.putText(img, text, (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def ffmpeg_exe():
    for c in (shutil.which("ffmpeg"), str(Path(sys.executable).parent / "ffmpeg"),
              "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("ffmpeg not found (needed to write the video)")


def encoder_args(exe):
    """The first H.264 encoder this ffmpeg can actually use (test-encodes a tiny clip)."""
    for args in (["-c:v", "libx264", "-crf", "18", "-preset", "medium"],
                 ["-c:v", "h264_videotoolbox", "-b:v", "16M"],
                 ["-c:v", "libopenh264", "-b:v", "16M"],
                 ["-c:v", "mpeg4", "-q:v", "2"]):
        r = subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2",
                            *args, "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True)
        if r.returncode == 0:
            return args
    raise RuntimeError("this ffmpeg has no usable video encoder")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def render_video(cfg, progress=None, log=print):
    cfg = {**DEFAULTS, **cfg}
    info = inspect(cfg)
    f0, f1, fps = info["frame_start"], info["frame_end"], info["fps"]
    tracks = load_tracks(cfg["head_tracking_dir"])
    two = len(tracks) > 1
    # one rat: its label tinted red, any other blue. Two rats: object 1 red
    # (labels sorted, so the first is 1), object 2 blue, as in GUI 5.1 / 5.2
    tracked = tracks[0]["mask_label"]
    cam_no = camera_number(cfg["camera_video_path"])
    cam = load_camera(cfg["calib_json_path"], cam_no)

    H = int(cfg["height_px"]) // 2 * 2
    S = H // 2
    cw, ch = info["camera_size"]
    W_left = int(round(H * cw / ch)) // 2 * 2
    W = W_left + S * len(tracks)            # two rats: two skull squares side by side
    out = Path(cfg["output_path"]) if cfg["output_path"] else \
        Path.home() / "Desktop" / f"three_panel_{Path(cfg['camera_video_path']).stem}_{f0}-{f1}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    trail = max(1, int(round(float(cfg["trail_seconds"]) * fps)))
    rats = []
    for t in tracks:
        lab = t["mask_label"]
        r = {k: aligned(t, k, f0, f1) for k in ("head_T", "body_T", "fwd_T", "head_C", "state")}
        r["label"], r["px"] = lab, np.full((f1 - f0 + 1, 2), np.nan)
        ok = ~np.isnan(r["head_C"][:, 0])
        r["px"][ok] = project(r["head_C"][ok], cam)
        if two:
            r["bgr"], r["rim"], r["color"] = RAT_BGR.get(lab, HEAD_BGR), (255, 255, 255), RAT_HEX.get(lab, "#e0631e")
            r["body_rgba"] = (*matplotlib.colors.to_rgb(r["color"]), 0.45)
        else:
            r["bgr"], r["rim"], r["color"], r["body_rgba"] = HEAD_BGR, (15, 15, 15), "#e0631e", (0.72, 0.70, 0.65, 0.55)
        rats.append(r)
    if two:
        legend = [(RAT_BGR.get(r["label"], HEAD_BGR), f"rat {r['label']} (object {r['label']}): head + path") for r in rats]
        footer = "   ".join(f"{'red' if r['label'] == 1 else 'blue'}: rat {r['label']} head" for r in rats) + \
            "   light: body centroids   yellow: nose direction   dotted: height"
        path = PathPanel(rats, (S * len(tracks), S), float(cfg["floor_z_m"]), float(cfg["elevation_deg"]), -60.0,
                         float(cfg["orbit_deg_per_s"]), fps, footer, title="head paths in 3D")
    else:
        legend = [(TRACKED_BGR, "tracked rat"), (OTHER_BGR, "other rat"), (HEAD_BGR, "head")]
        path = PathPanel(rats, S, float(cfg["floor_z_m"]), float(cfg["elevation_deg"]), -60.0,
                         float(cfg["orbit_deg_per_s"]), fps,
                         "orange: head   grey: body centroid   yellow: nose direction   dotted: height")
    slots = {int(k): v for k, v in info["skull_slots"].items()}

    cam_cap = cv2.VideoCapture(str(cfg["camera_video_path"]))
    cam_cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    sk_cap = cv2.VideoCapture(str(cfg["skull_video_path"]))
    sk_cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - info["skull_first_frame"])
    masks_dir = Path(cfg["masks_root"]) / f"cam_{cam_no}" / "masks_raw" if cfg["masks_root"] else None

    exe = ffmpeg_exe()
    enc = encoder_args(exe)
    log(f"rendering frames {f0}..{f1} ({f1 - f0 + 1} frames, {W}x{H}) with {enc[1]} -> {out}")
    proc = subprocess.Popen([exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", f"{fps:g}", "-i", "-", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(out)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        n = f1 - f0 + 1
        for k, f in enumerate(range(f0, f1 + 1)):
            ok_c, img = cam_cap.read()
            if not ok_c:
                raise RuntimeError(f"camera video ended before frame {f}")
            mask = None
            if masks_dir is not None:
                mp = masks_dir / f"frame_{f:06d}.png"
                if mp.exists():
                    mask = np.asarray(Image.open(mp))
            states = [r["state"][k].replace("_", " ") for r in rats]
            label = f"frame {f}   {f / fps:6.2f} s   " + ("   ".join(f"rat {r['label']} {s}" for r, s in zip(rats, states))
                                                          if two else states[0])
            heads = [(r["px"][max(0, k - trail): k + 1], r["px"][k], r["bgr"], r["rim"]) for r in rats]
            left = camera_panel(img, mask, heads, tracked, float(cfg["tint_alpha"]), W_left, H, label, legend)
            ok_s, sk = sk_cap.read()
            canvas = np.empty((H, W, 3), np.uint8)
            canvas[:, :W_left] = left
            for j, r in enumerate(rats):
                slot = slots.get(r["label"])
                if ok_s and slot is not None:
                    sh = sk.shape[0]
                    top = cv2.resize(sk[:, slot * sh:(slot + 1) * sh], (S, S), interpolation=cv2.INTER_AREA)
                else:
                    top = blank_panel(S, "no skull frame" if not ok_s else f"rat {r['label']} is not in the skull video")
                canvas[:S, W_left + j * S:W_left + (j + 1) * S] = top
                if j:
                    canvas[:S, W_left + j * S - 1:W_left + j * S + 1] = (60, 60, 60)
            canvas[S:2 * S, W_left:] = path.render(k, f / fps)
            canvas[:, W_left:W_left + 2] = (60, 60, 60)
            canvas[S - 1:S + 1, W_left:] = (60, 60, 60)
            proc.stdin.write(canvas.tobytes())
            if progress:
                progress(k + 1, n)
    finally:
        cam_cap.release()
        sk_cap.release()
        plt.close(path.fig)
        if proc.stdin:
            proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {err[-400:]}")
    return {"output_path": str(out), "frames": [f0, f1], "size": [W, H], "encoder": enc[1]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python render.py config.json")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    last = [0]

    def prog(d, n):
        if d - last[0] >= 100 or d == n:
            print(f"{d}/{n} frames", flush=True)
            last[0] = d
    print(json.dumps(render_video(cfg, progress=prog), indent=2))
