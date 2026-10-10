#!/usr/bin/env python3
"""
Analysis backing the head tracking GUI: every rig camera's GUI 2 segmentation
mask -> the tracked rat's body position, heading and head position in the
world frame, trajectory, kinematics and behavioural states. No head-mounted
IMU is used; the rat's GUI 2 mesh is only a fallback for the head when the
masks lose it for several frames in a row. Tracking two rats is one
analyse_from_masks run per mask label (rat 1 = label 1, rat 2 = label 2);
rat_distances, run_summary and make_two_rat_plots put the two side by side.

Frames used throughout (all right-handed, metres, radians unless _deg):

  calibration frame  (C)  the frame the camera poses (rig JSON) and every
                          reconstructed mesh live in.
  table frame        (W)  the table-aligned display frame from GUI 4's
                          build_initial_rig: z up, origin at the table
                          centre. This is the "world frame" all outputs are
                          reported in (plus the raw calibration-frame
                          coordinates for completeness).
  head frame         (H)  x = nose direction, y = left, z = up.

See analyse_from_masks for the per-frame computation and the module section
above it for what is and is not measurable from silhouettes.
"""

import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

FRAME_RE = re.compile(r"^frame_(\d+)$")

# Reference categorical palette (dataviz skill, light surface), fixed order.
COLOR_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATE_ORDER = ["rest", "head_movement", "locomotion"]
STATE_LABEL = {"rest": "rest", "head_movement": "head movement", "locomotion": "locomotion"}
STATE_COLOR = dict(zip(STATE_ORDER, COLOR_SERIES[:3]))
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
# Rat identity colours, the same as GUI 5.1's mask tint: rat 1 = mask label 1
# (object 1) red, rat 2 = mask label 2 (object 2) blue.
RAT_COLOR = {1: "#ff2828", 2: "#286eff"}


def _mix(color, other, frac):
    """Hex colour blended frac of the way towards another hex colour (e.g.
    towards white for the lighter body-centroid / future-path version)."""
    a = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(other[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return "#" + "".join(f"{int(round(v)):02x}" for v in (1 - frac) * a + frac * b)


# =============================================================================
# Meshes (3D view, and the head fallback)
# =============================================================================

def scan_mesh_frames(root):
    """frame_XXXXXX/ subfolders holding mesh.obj or mesh_preview.ply -> {int: Path}."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"mesh root is not a folder: {root}")
    frames = {}
    for p in root.iterdir():
        m = FRAME_RE.match(p.name)
        if not m or not p.is_dir():
            continue
        if (p / "mesh.obj").exists() or (p / "mesh_preview.ply").exists():
            frames[int(m.group(1))] = p
    return frames


def _ply_vertices(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            line = line.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next((h.split()[1] for h in header if h.startswith("format")), "ascii")
        n = int(next(h for h in header if h.startswith("element vertex")).split()[-1])
        props = []
        in_vertex = False
        for h in header:
            if h.startswith("element"):
                in_vertex = h.startswith("element vertex")
            elif in_vertex and h.startswith("property") and not h.startswith("property list"):
                props.append(h.split()[1:3])
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            xyz = data[:, :3]
        else:
            endian = "<" if "little" in fmt else ">"
            tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
                    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}
            dt = np.dtype([(name, endian + tmap[t]) for t, name in props])
            v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    return xyz


def _obj_vertices(path):
    pts = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass
    return np.asarray(pts, dtype=np.float64)


def _mesh_vertices(frame_dir):
    """Prefers the decimated mesh_preview.ply (same vertices, binary, ~30x
    faster to read) and falls back to parsing mesh.obj text."""
    frame_dir = Path(frame_dir)
    ply = frame_dir / "mesh_preview.ply"
    xyz = _ply_vertices(ply) if ply.exists() else _obj_vertices(frame_dir / "mesh.obj")
    if len(xyz) == 0:
        raise ValueError(f"{frame_dir}: mesh has no vertices")
    return xyz


def mesh_head(frame_dir, ref_T, B, tc, mesh_frame="calib", max_dist_m=0.10, tip_frac=0.08):
    """Head point (table frame) of one frame's reconstructed mesh, or None.

    The meshes have no fixed vertex layout, so the head is found by shape:
    the mean of the vertices in the outer tip_frac of the mesh's long
    horizontal axis, at whichever end is closer to ref_T (where the
    neighbouring mask detections put the head). The mesh's own ends cannot
    tell the snout from its tail stub reliably, so that choice is left to the
    masks. Rejected (None) if that end is more than max_dist_m from ref_T,
    which also rejects a mesh that has latched onto the other rat."""
    P = calib_to_table(_mesh_vertices(frame_dir), B, tc, mesh_frame)
    c = P.mean(axis=0)
    ev, evec = np.linalg.eigh(np.cov((P[:, :2] - c[:2]).T))
    axis = evec[:, -1]
    proj = (P[:, :2] - c[:2]) @ axis
    lo, hi = proj.min(), proj.max()
    ends = [P[proj > hi - tip_frac * (hi - lo)].mean(axis=0), P[proj < lo + tip_frac * (hi - lo)].mean(axis=0)]
    best = min(ends, key=lambda e: np.linalg.norm(e - ref_T))
    return best if np.linalg.norm(best - ref_T) <= max_dist_m else None


def mesh_centroid(frame_dir):
    """Vertex centroid + bbox of a frame's mesh, in the calibration frame."""
    frame_dir = Path(frame_dir)
    ply = frame_dir / "mesh_preview.ply"
    obj = frame_dir / "mesh.obj"
    xyz = _mesh_vertices(frame_dir)
    return {
        "centroid": xyz.mean(axis=0).tolist(),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "vertex_count": int(len(xyz)),
        "source": ply.name if ply.exists() else obj.name,
    }


def collect_centroids(frames, cache_path=None, log=None):
    """{frame:int -> centroid info} for every reconstructed frame, cached by mtime."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
    out = {}
    dirty = False
    items = sorted(frames.items())
    for i, (fr, d) in enumerate(items):
        key = str(d)
        mesh = d / "mesh_preview.ply" if (d / "mesh_preview.ply").exists() else d / "mesh.obj"
        mtime = mesh.stat().st_mtime
        entry = cache.get(key)
        if not entry or abs(entry.get("mtime", -1) - mtime) > 1e-6:
            entry = mesh_centroid(d)
            entry["mtime"] = mtime
            cache[key] = entry
            dirty = True
        out[fr] = entry
        if log and (i % 50 == 0 or i == len(items) - 1):
            log(f"centroids: {i + 1}/{len(items)} meshes")
    if cache_path and dirty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache))
    return out


def sequence_provenance(centroids):
    """Do the meshes of a sequence still carry world coordinates?

    A GUI 2 mesh sits wherever the animal was, so its centroid moves from
    frame to frame. A sequence that has been re-centred by a later step (every
    mesh normalised to its own origin, or to one fixed spot) has centroids that
    barely move on at least one axis while the animal itself is tens of cm
    long. Positions, paths and speeds from such a sequence are meaningless, so
    we measure rather than assume."""
    if len(centroids) < 2:
        return {"n_frames": len(centroids), "recentred": False, "reason": "single frame, cannot tell"}
    C = np.array([c["centroid"] for c in centroids.values()])
    sizes = np.array([np.subtract(c["bbox_max"], c["bbox_min"]) for c in centroids.values()])
    spread = C.max(axis=0) - C.min(axis=0)
    std = C.std(axis=0)
    mean_size = float(sizes.mean())
    # an axis whose centroid range is < 2% of the body size over the whole
    # sequence is pinned; two or more pinned axes means the sequence was
    # re-centred (one could in theory be a rat glued to a wall).
    pinned = int((spread < 0.02 * max(mean_size, 1e-3)).sum())
    recentred = pinned >= 2
    # a crouching rat is ~7 cm tall and >15 cm long, so the axis with the
    # smallest median bbox extent is (almost always) the vertical one
    med_ext = np.median(sizes, axis=0)
    thin = int(np.argmin(med_ext))
    return {
        "thin_axis": "xyz"[thin],
        "thin_extent_m": float(med_ext[thin]),
        "median_extent_m": med_ext.tolist(),
        "n_frames": int(len(C)),
        "centroid_range_m": spread.tolist(),
        "centroid_std_m": std.tolist(),
        "mean_body_size_m": mean_size,
        "pinned_axes": pinned,
        "recentred": bool(recentred),
        "reason": ("centroid is fixed on %d axes while the body is %.0f cm: meshes were re-centred, positions are not real"
                   % (pinned, mean_size * 100)) if recentred else
                  ("centroid moves %.0f x %.0f x %.0f cm over the sequence: looks like world coordinates"
                   % tuple(spread * 100)),
    }


# =============================================================================
# Frames
# =============================================================================

def table_frame_from_rig(rig):
    B = np.asarray(rig["display_basis"], dtype=np.float64)  # rows: x_disp, y_disp, up_disp
    tc = np.asarray(rig["table_center"], dtype=np.float64)   # already in display coordinates
    return B, tc


def direction_calib_to_table(d, B):
    """A DIRECTION (not a point) from the calibration frame to the table frame:
    rotation only, no origin shift -- unlike calib_to_table below. NaN rows
    (no fused heading that frame) pass through as NaN."""
    return np.asarray(d, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T


def calib_to_table(p, B, tc, mesh_frame="calib"):
    """mesh coordinates -> table frame. 'calib': the calibration frame in
    metres (GUI 2 output). 'yup': an already upright y-up export, taken as is
    (x, -z, y) so that z is up, measured from the mesh origin."""
    p = np.asarray(p, dtype=np.float64)
    if mesh_frame == "yup":
        return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)
    return (p @ B.T) - tc


def table_to_calib(p, B, tc, mesh_frame="calib"):
    p = np.asarray(p, dtype=np.float64)
    if mesh_frame == "yup":
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)
    return (p + tc) @ B


# =============================================================================
# Kinematics
# =============================================================================

def _odd_window(fps, seconds, n, minimum=5):
    w = int(round(fps * seconds))
    w = max(minimum, w | 1)
    if w >= n:
        w = (n - 1) | 1 if n > minimum else 0
    return w


def _smooth(x, window, poly=2, deriv=0, dt=1.0):
    if window < 5 or len(x) <= window:
        if deriv == 0:
            return x
        return np.gradient(x, dt, axis=0)
    return savgol_filter(x, window, poly, deriv=deriv, delta=dt, axis=0, mode="interp")


def _runs(labels):
    """[(start, end_exclusive, label)] for a 1-D label array."""
    runs = []
    if len(labels) == 0:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def _enforce_min_duration(labels, min_frames, max_iter=8):
    labels = np.array(labels, dtype=object)
    for _ in range(max_iter):
        runs = _runs(labels)
        short = [(s, e, l) for s, e, l in runs if e - s < min_frames]
        if not short or len(runs) == 1:
            break
        for s, e, l in short:
            i = next(k for k, r in enumerate(runs) if r[0] == s)
            prev_len = runs[i - 1][1] - runs[i - 1][0] if i > 0 else -1
            next_len = runs[i + 1][1] - runs[i + 1][0] if i + 1 < len(runs) else -1
            if prev_len < 0 and next_len < 0:
                continue
            take = runs[i - 1][2] if prev_len >= next_len else runs[i + 1][2]
            labels[s:e] = take
    return labels


# =============================================================================
# Trajectory from the masks
#
# Body position and heading come from every rig camera's GUI 2 segmentation
# mask via fuse_frame (mask_heading's centroid and PCA long axis per camera,
# triangulated into a 3D point and, from the confident cameras, a 3D heading
# direction; see triangulate_direction's own docstring for that fusion). The
# head comes from the masks too (mask_head / fuse_head), with the rat's mesh
# as a fallback over long gaps (mesh_head). This recovers position, yaw and
# pitch (nose up/down) purely from silhouettes. Roll, rotation about the nose
# axis, is NOT recoverable this way: a silhouette does not distinguish the
# animal being upright from being rolled onto its side around its own long
# axis. Roll is therefore always reported as exactly 0, with "up" taken as the
# closest table-vertical direction perpendicular to the fused heading, not a
# measurement.
# =============================================================================

def analyse_from_masks(cams, masks_root, mask_label, fps, frame_start, frame_end, B, tc,
                        params=None, log=None, min_confidence=0.15, max_workers=8,
                        mesh_root=None, mesh_frame="calib"):
    """Core computation. cams: {name -> cam dict} from load_camera_extrinsics,
    in the calibration frame the masks were segmented in; B, tc convert that
    frame to the table frame (table_frame_from_rig). frame_start/frame_end:
    inclusive global frame numbers to analyse (every frame in this range is
    attempted independently). mesh_root (optional): this rat's GUI 2 mesh
    folder, used only as the head fallback during long gaps in the mask head
    detection. Returns the per-frame arrays and summary that write_csvs,
    make_plots, downsample_for_viewer and make_camera_overlays read."""
    from concurrent.futures import ThreadPoolExecutor

    params = dict(params or {})
    ang_thr = float(params.get("rest_ang_speed_dps", 20.0))
    lin_thr = float(params.get("rest_lin_speed_mps", 0.05))
    min_dur = float(params.get("min_state_duration_s", 0.5))
    smooth_s = float(params.get("smoothing_s", 0.25))
    max_gap = int(params.get("max_interp_gap_frames", 60))

    frame_start, frame_end = int(frame_start), int(frame_end)
    N = frame_end - frame_start + 1
    if N < 3:
        raise ValueError(f"need at least 3 frames, got the range [{frame_start}, {frame_end}]")
    frame = np.arange(N) + frame_start
    t = frame / fps
    dt = 1.0 / fps

    min_head_cams = int(params.get("min_head_cams", 2))      # per candidate; track_head weighs the support
    max_jump = float(params.get("max_head_jump_m", 0.08))
    mesh_min_gap = int(params.get("mesh_fallback_min_gap_frames", 3))
    mesh_max_dist = float(params.get("mesh_head_max_dist_m", 0.10))

    body_C = np.full((N, 3), np.nan)
    fwd_C = np.full((N, 3), np.nan)
    head_C = np.full((N, 3), np.nan)
    n_cams_pos = np.zeros(N, dtype=int)
    n_cams_dir = np.zeros(N, dtype=int)
    n_cams_head = np.zeros(N, dtype=int)

    def _one(i):
        return i, fuse_frame(masks_root, cams, int(frame[i]), label=mask_label, min_confidence=min_confidence)

    cands = [[] for _ in range(N)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for k, (i, r) in enumerate(pool.map(_one, range(N))):
            if r["centroid"] is not None:
                body_C[i] = r["centroid"]
                n_cams_pos[i] = r["n_cams"]
            if r["direction"] is not None:
                fwd_C[i] = r["direction"]
                n_cams_dir[i] = r["n_cams_used_for_direction"]
            cands[i] = [h for h in r["head_candidates"] if h["n"] >= min_head_cams]
            if log and (k % 100 == 0 or k == N - 1):
                log(f"mask fusion: {k + 1}/{N} frames")

    # the head: each frame's candidates chosen jointly over time (track_head),
    # not each frame's best guess on its own, which jumps to the rump or
    # between the rats while they are in contact
    chosen = track_head(cands, [None if np.isnan(b[0]) else b for b in body_C],
                        **{k[len("head_track_"):]: v for k, v in params.items() if k.startswith("head_track_")})
    for i, j in enumerate(chosen):
        if j is not None:
            head_C[i] = cands[i][j]["X"]
            n_cams_head[i] = cands[i][j]["n"]
    if log:
        log(f"head track: {sum(j is not None for j in chosen)}/{N} frames from the masks")

    body = calib_to_table(body_C, B, tc)
    fwd = direction_calib_to_table(fwd_C, B)
    pos_ok = ~np.isnan(body[:, 0])
    dir_ok = ~np.isnan(fwd[:, 0])

    pos_source = np.array([""] * N, dtype=object)
    pos_source[pos_ok] = "mask"
    idx_pos = np.where(pos_ok)[0]
    for a, b in zip(idx_pos[:-1], idx_pos[1:]):
        if b - a <= 1 or b - a > max_gap:
            continue
        w = (np.arange(a + 1, b) - a) / (b - a)
        body[a + 1:b] = (1 - w)[:, None] * body[a] + w[:, None] * body[b]
        pos_source[a + 1:b] = "interp"

    # head from the masks (track_head): kept where at least min_head_cams
    # cameras agreed. A head cannot move max_jump in one frame's worth of
    # neighbours, so a detection far from the median of the detections within
    # 2 frames either side is dropped -- seen in practice when three cameras
    # agreed on the rump for a single frame.
    head = calib_to_table(head_C, B, tc)
    head_det = ~np.isnan(head[:, 0])
    idx_head = np.where(head_det)[0]
    rejected = []
    for k, i in enumerate(idx_head):
        nb = [j for j in idx_head[max(0, k - 2):k + 3] if j != i and abs(j - i) <= 2]
        if len(nb) >= 2 and np.linalg.norm(head[i] - np.median(head[nb], axis=0)) > max_jump:
            rejected.append(i)
    head[rejected] = np.nan
    head_det[rejected] = False

    # triangulate_direction only recovers the heading axis, not which end is
    # the nose: D and -D satisfy its linear system equally, and its per-frame
    # sign pick (agreement with each camera's wide-end = head call) can be
    # wrong, either as a sudden flip (typically during contact) or as a slow
    # drift over a second or so. Where the masks found the head, the sign is
    # simply whichever points from the body centroid towards it. Elsewhere,
    # in time order:
    #  1. Continuity: a rat cannot flip its heading 180 degrees in one video
    #     frame, so prefer whichever sign is closer to the last resolved frame.
    #  2. Direction of travel: while the body centroid is clearly moving
    #     across the table (vertical motion ignored -- a rat dropping from a
    #     rear says nothing about which end is the nose), the heading should
    #     not point backwards against that direction, since rats move
    #     nose-first. This catches a slow drift, which continuity cannot.
    # None of this changes which frames have a known heading, only which of
    # the two equally-valid signs is reported for them.
    idx_dir = np.where(dir_ok)[0]
    body_speed_thr = float(params.get("min_speed_for_heading_check_mps", 0.06))
    vel_span = max(1, int(round(0.1 * fps)))  # ~0.1s central-difference window
    last_good = None
    for i in idx_dir:
        candidate = fwd[i]
        if head_det[i] and pos_ok[i]:
            if np.dot(candidate, head[i] - body[i]) < 0:
                candidate = -candidate
        else:
            if last_good is not None and np.dot(candidate, last_good) < 0:
                candidate = -candidate
            lo, hi = i - vel_span, i + vel_span
            if lo >= 0 and hi < N and pos_source[lo] != "" and pos_source[hi] != "":
                vel_xy = (body[hi, :2] - body[lo, :2]) / (2 * vel_span * dt)
                speed = np.linalg.norm(vel_xy)
                fwd_xy = np.linalg.norm(candidate[:2])
                if speed > body_speed_thr and fwd_xy > 1e-6 and candidate[:2] @ vel_xy / (speed * fwd_xy) < -0.3:
                    candidate = -candidate
        fwd[i] = candidate
        last_good = candidate

    dir_known = dir_ok.copy()
    for a, b in zip(idx_dir[:-1], idx_dir[1:]):
        if b - a <= 1 or b - a > max_gap:
            continue
        w = (np.arange(a + 1, b) - a) / (b - a)
        blended = (1 - w)[:, None] * fwd[a] + w[:, None] * fwd[b]
        fwd[a + 1:b] = blended / np.linalg.norm(blended, axis=1, keepdims=True)
        dir_known[a + 1:b] = True

    # gaps in the mask head: shorter than mesh_min_gap frames are interpolated;
    # longer ones take the mesh head (mesh_head) at every frame there is a
    # mesh for, placed near where the detections either side of the gap put
    # the head, then the remaining frames are interpolated between all of
    # these anchors (up to max_gap).
    head_source = np.array([""] * N, dtype=object)
    head_source[head_det] = "mask"
    mesh_frames = {}
    if mesh_root:
        try:
            mesh_frames = scan_mesh_frames(mesh_root)
        except FileNotFoundError as e:
            if log:
                log(f"mesh fallback off: {e}")
    if mesh_frames:
        s_ = 0
        while s_ < N:
            if head_det[s_]:
                s_ += 1
                continue
            e_ = s_
            while e_ + 1 < N and not head_det[e_ + 1]:
                e_ += 1
            prev_i = s_ - 1 if s_ > 0 else None
            next_i = e_ + 1 if e_ + 1 < N else None
            if e_ - s_ + 1 >= mesh_min_gap and (prev_i is not None or next_i is not None):
                for i in range(s_, e_ + 1):
                    d = mesh_frames.get(int(frame[i]))
                    if d is None:
                        continue
                    if prev_i is not None and next_i is not None:
                        w = (i - prev_i) / (next_i - prev_i)
                        ref = (1 - w) * head[prev_i] + w * head[next_i]
                    else:
                        ref = head[prev_i if prev_i is not None else next_i]
                    try:
                        mh = mesh_head(d, ref, B, tc, mesh_frame, mesh_max_dist)
                    except (OSError, ValueError) as err:
                        if log:
                            log(f"mesh fallback: skipped {d}: {err}")
                        continue
                    if mh is not None:
                        head[i] = mh
                        head_source[i] = "mesh"
            s_ = e_ + 1
    anchors = np.where(head_source != "")[0]
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a <= 1 or b - a > max_gap:
            continue
        w = (np.arange(a + 1, b) - a) / (b - a)
        head[a + 1:b] = (1 - w)[:, None] * head[a] + w[:, None] * head[b]
        head_source[a + 1:b] = "interp"
    has_pos = ~np.isnan(head[:, 0])

    yaw = np.full(N, np.nan)
    pitch = np.full(N, np.nan)
    roll = np.zeros(N)  # not recoverable from a silhouette; always exactly 0 (see module note)
    up = np.full((N, 3), np.nan)
    world_up = np.array([0.0, 0.0, 1.0])
    if dir_known.any():
        fk = fwd[dir_known]
        yaw[dir_known] = np.degrees(np.arctan2(fk[:, 1], fk[:, 0]))
        pitch[dir_known] = np.degrees(np.arcsin(np.clip(fk[:, 2], -1, 1)))
        zru = np.cross(fk, np.cross(world_up, fk))
        nrm = np.linalg.norm(zru, axis=1, keepdims=True)
        up[dir_known] = np.divide(zru, nrm, out=np.tile(world_up, (len(fk), 1)), where=nrm > 1e-9)

    R_WH = np.full((N, 3, 3), np.nan)
    q_WH = np.full((N, 4), np.nan)
    if dir_known.any():
        left = np.cross(up[dir_known], fwd[dir_known])
        R_WH[dir_known] = np.stack([fwd[dir_known], left, up[dir_known]], axis=2)
        q_WH[dir_known] = Rotation.from_matrix(R_WH[dir_known]).as_quat()[:, [3, 0, 1, 2]]

    # angular velocity between consecutive frames that both have a known
    # fused heading (never bridged across a gap: unlike position, a rotation
    # rate implied across a multi-frame gap is not a measurement worth making)
    omega = np.full((N, 3), np.nan)
    for i in range(N - 1):
        if dir_known[i] and dir_known[i + 1]:
            rel = Rotation.from_matrix(R_WH[i]).inv() * Rotation.from_matrix(R_WH[i + 1])
            omega[i] = rel.as_rotvec() * fps
    ang_known = ~np.isnan(omega[:, 0])
    if ang_known.any():
        last = np.where(ang_known)[0][-1]
        omega[last + 1 if last + 1 < N else last] = omega[last]
        ang_known = ~np.isnan(omega[:, 0])

    win = _odd_window(fps, smooth_s, N)
    omega_deg = np.full((N, 3), np.nan)
    ang_speed = np.full(N, np.nan)
    ang_acc = np.full(N, np.nan)
    for s_, e_, ok in _runs(ang_known):
        if not ok or e_ - s_ < 3:
            continue
        seg = omega[s_:e_]
        w = win if win < e_ - s_ else ((e_ - s_ - 1) | 1)
        om = _smooth(seg, w) if w >= 5 else seg
        omega_deg[s_:e_] = np.degrees(om)
        ang_speed[s_:e_] = np.degrees(np.linalg.norm(om, axis=1))
        acc = _smooth(om, w, deriv=1, dt=dt) if w >= 5 else np.gradient(om, dt, axis=0)
        ang_acc[s_:e_] = np.degrees(np.linalg.norm(acc, axis=1))

    lin_speed = np.full(N, np.nan)
    lin_acc = np.full(N, np.nan)
    vel = np.full((N, 3), np.nan)
    if has_pos.any():
        win_lin = _odd_window(fps, float(params.get("position_smoothing_s", 1.0)), N)
        for s_, e_, ok in _runs(has_pos):
            if not ok or e_ - s_ < 3:
                continue
            seg = head[s_:e_]
            w = win_lin if win_lin < e_ - s_ else ((e_ - s_ - 1) | 1)
            v = _smooth(seg, w, deriv=1, dt=dt) if w >= 5 else np.gradient(seg, dt, axis=0)
            vel[s_:e_] = v
            lin_speed[s_:e_] = np.linalg.norm(v, axis=1)
            a = _smooth(v, w, deriv=1, dt=dt) if w >= 5 else np.gradient(v, dt, axis=0)
            lin_acc[s_:e_] = np.linalg.norm(a, axis=1)

    ang_speed_known = ~np.isnan(ang_speed)
    lin_known = ~np.isnan(lin_speed)
    moving_body = lin_known & (lin_speed >= lin_thr)
    still_body = ~lin_known | (lin_speed < lin_thr)
    still_head = ~ang_speed_known | (ang_speed < ang_thr)
    state = np.where(moving_body, "locomotion", np.where(still_head & still_body, "rest", "head_movement"))
    state = _enforce_min_duration(state, max(1, int(round(min_dur * fps))))

    bouts = []
    for s_, e_, lab in _runs(state):
        bouts.append({"state": str(lab), "start_frame": int(s_ + frame_start), "end_frame": int(e_ - 1 + frame_start),
                      "start_s": float((s_ + frame_start) / fps), "end_s": float((e_ - 1 + frame_start) / fps),
                      "duration_s": float((e_ - s_) / fps)})

    def path_length(P, mask):
        Q = P[mask]
        Q = Q[~np.isnan(Q[:, 0])]  # head needs a resolved heading too, so it can be gappier than mask's own detections
        return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()) if len(Q) > 1 else 0.0

    meas_mask = pos_source == "mask"
    fractions = {s_: float(np.mean(state == s_)) for s_ in STATE_ORDER}
    rest_bouts = [b for b in bouts if b["state"] == "rest"]
    frames_meas = np.where(meas_mask)[0]
    summary = {
        "n_frames": int(N), "fps": float(fps), "duration_s": float(N / fps),
        "anchor_frame": int(frame_start),  # = window start; the default present / reference frame for plots
        "window": {"start_frame": int(frame[0]), "end_frame": int(frame[-1])},
        "position_coverage": {
            "frames_with_mesh": int(meas_mask.sum()),
            "frames_with_position": int(has_pos.sum()),
            "fraction_with_position": float(has_pos.mean()),
            "first_frame": int(frames_meas[0] + frame_start) if len(frames_meas) else None,
            "last_frame": int(frames_meas[-1] + frame_start) if len(frames_meas) else None,
        },
        "path_length_m": {"head": path_length(head, meas_mask), "body_centroid": path_length(body, meas_mask)},
        "linear_speed_mps": {
            "mean": float(np.nanmean(lin_speed)) if lin_known.any() else None,
            "median": float(np.nanmedian(lin_speed)) if lin_known.any() else None,
            "max": float(np.nanmax(lin_speed)) if lin_known.any() else None,
        },
        "angular_speed_dps": {
            "mean": float(np.nanmean(ang_speed)) if ang_speed_known.any() else None,
            "median": float(np.nanmedian(ang_speed)) if ang_speed_known.any() else None,
            "max": float(np.nanmax(ang_speed)) if ang_speed_known.any() else None,
        },
        "state_fractions": fractions,
        "state_time_s": {s_: float(fractions[s_] * N / fps) for s_ in STATE_ORDER},
        "rest_bouts": {
            "count": len(rest_bouts),
            "mean_duration_s": float(np.mean([b["duration_s"] for b in rest_bouts])) if rest_bouts else 0.0,
            "longest_s": float(max((b["duration_s"] for b in rest_bouts), default=0.0)),
        },
        "thresholds": {"rest_ang_speed_dps": ang_thr, "rest_lin_speed_mps": lin_thr,
                       "min_state_duration_s": min_dur, "smoothing_s": smooth_s,
                       "position_smoothing_s": float(params.get("position_smoothing_s", 1.0)),
                       "max_interp_gap_frames": max_gap},
        "offset_head_frame_m": None,
        "position_valid": bool(has_pos.any()),
        "mask_fusion": {
            "label": int(mask_label), "min_confidence": float(min_confidence),
            "mean_n_cams_position": float(n_cams_pos[pos_ok].mean()) if pos_ok.any() else 0.0,
            "mean_n_cams_heading": float(n_cams_dir[dir_ok].mean()) if dir_ok.any() else 0.0,
            "fraction_with_heading": float(dir_ok.mean()),
            "min_head_cams": min_head_cams,
            "mean_n_cams_head": float(n_cams_head[head_det].mean()) if head_det.any() else 0.0,
            "frames_head_from_masks": int(head_det.sum()),
            "frames_head_from_mesh": int((head_source == "mesh").sum()),
            "frames_head_interpolated": int((head_source == "interp").sum()),
            "frames_head_missing": int((head_source == "").sum()),
            "head_jumps_rejected": [int(frame[i]) for i in rejected],
            "mesh_fallback_root": str(mesh_root) if mesh_frames else None,
        },
        "notes": [
            "no head-mounted IMU: body position and heading come from every rig camera's segmentation mask "
            "(mask_heading + triangulate_point / triangulate_direction)",
            "roll (rotation about the nose axis) is not recoverable from a silhouette and is always reported as 0; "
            "yaw and pitch come from the fused 3D heading direction",
            "position comes from mask triangulation at every frame with >=2 cameras seeing the label; frames between "
            "detections are linearly interpolated up to max_interp_gap_frames (pos_source = interp)",
            "head comes from the masks (mask_head + head_candidates + track_head): per camera, the tail is the "
            "long thin part of the silhouette that does not survive a morphological opening (compact pieces such as "
            "the headstage are not taken for it), and the head is the body point farthest from the tail base "
            "measured through the body; cameras without a visible tail offer both body ends. Every 3D head at least "
            "min_head_cams cameras agree on is a candidate; track_head picks one per frame (or none) jointly over "
            "all frames, weighing supporting and contradicting cameras against how far the head would have to "
            "move (head_source = mask); single-frame jumps beyond max_head_jump_m from the neighbouring detections "
            "are dropped (head_jumps_rejected)",
            "gaps in the mask head shorter than mesh_fallback_min_gap_frames are interpolated; longer gaps use the "
            "head of this rat's GUI 2 mesh wherever a mesh exists for the frame (head_source = mesh; the mesh end "
            "nearest the neighbouring mask detections, rejected if more than mesh_head_max_dist_m away), and are "
            "interpolated between those anchors otherwise (head_source = interp)",
            "the fused heading's sign (which end is the nose) points from the body centroid to the head wherever "
            "the masks found the head; elsewhere it is kept continuous with the previous frame and, while the body "
            "centroid is clearly moving across the table, not allowed to point backwards against the direction of "
            "travel",
            "angular velocity is only computed between directly adjacent frames that both have a fused heading; "
            "it is never bridged across a gap",
        ],
    }
    return {
        "t": t, "frame": frame, "q_WH": q_WH, "fwd": fwd, "up": up,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "omega_deg": omega_deg, "ang_speed": ang_speed, "ang_acc": ang_acc,
        "body": body, "head": head, "pos_source": pos_source, "has_pos": has_pos,
        "vel": vel, "lin_speed": lin_speed, "lin_acc": lin_acc,
        "state": state, "bouts": bouts, "summary": summary,
        "frames_meas": frames_meas + frame_start,
        "n_cams_position": n_cams_pos, "n_cams_heading": n_cams_dir,
        "head_source": head_source, "n_cams_head": n_cams_head,
    }


# =============================================================================
# Outputs
# =============================================================================

def _fmt(x, nd=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def write_csvs(res, out_dir, B, tc, mesh_frame="calib"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(res["t"])
    head_C = np.full((N, 3), np.nan)
    body_C = np.full((N, 3), np.nan)
    ok = res["has_pos"]
    if ok.any():
        head_C[ok] = table_to_calib(res["head"][ok], B, tc, mesh_frame)
        body_C[ok] = table_to_calib(res["body"][ok], B, tc, mesh_frame)

    cols = ["frame", "time_s", "pos_source",
            "body_x_m", "body_y_m", "body_z_m", "head_x_m", "head_y_m", "head_z_m",
            "qw", "qx", "qy", "qz", "yaw_deg", "pitch_deg", "roll_deg",
            "fwd_x", "fwd_y", "fwd_z", "up_x", "up_y", "up_z",
            "omega_x_dps", "omega_y_dps", "omega_z_dps", "ang_speed_dps", "ang_acc_dps2",
            "vel_x_mps", "vel_y_mps", "vel_z_mps", "lin_speed_mps", "lin_acc_mps2", "state",
            "body_calib_x_m", "body_calib_y_m", "body_calib_z_m",
            "head_calib_x_m", "head_calib_y_m", "head_calib_z_m", "head_source"]
    head_source = res.get("head_source", res["pos_source"])
    with (out_dir / "head_pose_world.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(N):
            w.writerow([
                int(res["frame"][i]), _fmt(res["t"][i], 4), res["pos_source"][i],
                *[_fmt(v) for v in res["body"][i]], *[_fmt(v) for v in res["head"][i]],
                *[_fmt(v, 8) for v in res["q_WH"][i]],
                _fmt(res["yaw"][i], 3), _fmt(res["pitch"][i], 3), _fmt(res["roll"][i], 3),
                *[_fmt(v) for v in res["fwd"][i]], *[_fmt(v) for v in res["up"][i]],
                *[_fmt(v, 3) for v in res["omega_deg"][i]], _fmt(res["ang_speed"][i], 3), _fmt(res["ang_acc"][i], 2),
                *[_fmt(v, 4) for v in res["vel"][i]], _fmt(res["lin_speed"][i], 4), _fmt(res["lin_acc"][i], 4),
                res["state"][i],
                *[_fmt(v) for v in body_C[i]], *[_fmt(v) for v in head_C[i]],
                head_source[i],
            ])
    with (out_dir / "state_bouts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "state", "start_frame", "end_frame", "start_s", "end_s", "duration_s"])
        w.writeheader()
        for i, b in enumerate(res["bouts"]):
            w.writerow({"index": i, **b})
    with (out_dir / "rest_bouts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "start_frame", "end_frame", "start_s", "end_s", "duration_s"])
        w.writeheader()
        for i, b in enumerate(b for b in res["bouts"] if b["state"] == "rest"):
            w.writerow({"index": i, **{k: b[k] for k in w.fieldnames if k != "index"}})


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _shade_rest(ax, bouts, fps=None):
    for b in bouts:
        if b["state"] == "rest":
            ax.axvspan(b["start_s"], b["end_s"] + (1.0 / fps if fps else 0), color=STATE_COLOR["rest"], alpha=0.12, lw=0)


def make_plots(res, out_dir, table_size_m=1.6, reference_frame=None, color=None, name=None):
    """One rat's plots. color (hex, e.g. RAT_COLOR[label]) replaces the default
    blue wherever this rat's path or a single time series is drawn, and name
    (e.g. "rat 1") prefixes the titles; the state and past/future colourings
    keep their own palettes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = res["summary"]
    t = res["t"]
    fps = s["fps"]
    frame0 = int(res["frame"][0])  # global frame number of array position 0 (the window may not start at frame 0)
    written = {}
    series_color = color or COLOR_SERIES[0]
    prefix = f"{name}: " if name else ""
    seq_blue = (LinearSegmentedColormap.from_list("seq_rat", [_mix(color, "#ffffff", 0.75), color, _mix(color, "#000000", 0.55)])
                if color else LinearSegmentedColormap.from_list("seq_blue", ["#d6e6f8", "#2a78d6", "#0d2f5e"]))
    PAST_COLOR, FUTURE_COLOR, PRESENT_COLOR = "#2a78d6", "#eb6834", "#0b0b0b"

    # ---- 1. path, top view --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), facecolor=SURFACE)
    meas = res["has_pos"]  # frames with a head position (from masks, mesh or interpolation)
    P = res["head"]
    Bc = res["body"]
    for ax in axes:
        _style_axis(ax)
        half = table_size_m / 2
        ax.add_patch(Rectangle((-half, -half), table_size_m, table_size_m, facecolor="#f1efe9", edgecolor=GRID, lw=1, zorder=0))
        ax.set_aspect("equal")
        ax.set_xlabel("x (m), table frame")
        ax.set_ylabel("y (m)")
    if meas.sum() > 1:
        idx = np.where(meas)[0]
        pts = P[idx]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)[:, :, :2]
        lc = LineCollection(segs, cmap=seq_blue, norm=Normalize(t[idx[0]], t[idx[-1]]), linewidths=2)
        lc.set_array(t[idx[:-1]])
        axes[0].add_collection(lc)
        axes[0].plot(Bc[idx, 0], Bc[idx, 1], color="#b9b7ae", lw=1, zorder=1, label="body centroid")
        axes[0].scatter(pts[0, 0], pts[0, 1], s=60, color="#1baf7a", zorder=5, edgecolor=SURFACE, linewidth=1.5)
        axes[0].scatter(pts[-1, 0], pts[-1, 1], s=60, color="#e34948", zorder=5, edgecolor=SURFACE, linewidth=1.5)
        axes[0].annotate("start", pts[0, :2], xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_PRIMARY)
        axes[0].annotate("end", pts[-1, :2], xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=TEXT_PRIMARY)
        cb = fig.colorbar(lc, ax=axes[0], fraction=0.046, pad=0.03)
        cb.set_label("time (s)", color=TEXT_SECONDARY)
        cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        cb.outline.set_edgecolor(GRID)
        axes[0].legend(handles=[Line2D([], [], color=series_color, lw=2, label="head path"),
                                Line2D([], [], color="#b9b7ae", lw=1, label="body centroid path")],
                       loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
        # state-coloured copy
        colors = [STATE_COLOR[str(res["state"][i])] for i in idx[:-1]]
        lc2 = LineCollection(segs, colors=colors, linewidths=2)
        axes[1].add_collection(lc2)
        # bout frame numbers are global; convert to local array positions
        rest_starts = [b["start_frame"] - frame0 for b in res["bouts"] if b["state"] == "rest"]
        rs = [f for f in rest_starts if 0 <= f < len(res["has_pos"]) and res["has_pos"][f]]
        if rs:
            axes[1].scatter(P[rs, 0], P[rs, 1], s=34, color=STATE_COLOR["rest"], edgecolor=SURFACE, linewidth=1, zorder=5)
        axes[1].legend(handles=[Patch(color=STATE_COLOR[k], label=STATE_LABEL[k]) for k in STATE_ORDER]
                       + [Line2D([], [], marker="o", ls="", color=STATE_COLOR["rest"], label="rest bout start")],
                       loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
        # reference-frame coloured copy: past vs future of a chosen "present" frame
        # (reference_frame, like anchor_frame, is a GLOBAL frame number; convert to
        # a local array position to index into this window's arrays)
        ref_global = int(reference_frame) if reference_frame is not None else int(s["anchor_frame"])
        ref_local = max(0, min(len(t) - 1, ref_global - frame0))
        ref_global = frame0 + ref_local  # reflect any clamping in the label below
        is_future = idx[:-1] >= ref_local
        colors3 = [FUTURE_COLOR if f else PAST_COLOR for f in is_future]
        lc3 = LineCollection(segs, colors=colors3, linewidths=2)
        axes[2].add_collection(lc3)
        if res["has_pos"][ref_local]:
            rp = P[ref_local]
            axes[2].scatter([rp[0]], [rp[1]], s=110, marker="*", color=PRESENT_COLOR, zorder=6, edgecolor=SURFACE, linewidth=1)
            axes[2].annotate(f"present (frame {ref_global}, t={t[ref_local]:.1f}s)", rp[:2], xytext=(6, 6),
                             textcoords="offset points", fontsize=8.5, color=TEXT_PRIMARY)
        axes[2].legend(handles=[Line2D([], [], color=PAST_COLOR, lw=2, label="past"),
                                Line2D([], [], color=FUTURE_COLOR, lw=2, label="future"),
                                Line2D([], [], marker="*", ls="", color=PRESENT_COLOR, label="present")],
                       loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
        # frame the path itself (the table rectangle is context, not the subject:
        # a 10 cm path on a 1.6 m table would otherwise be an unreadable dot)
        allp = np.vstack([pts[:, :2], Bc[idx, :2]])
        lo = np.nanmin(allp, axis=0); hi = np.nanmax(allp, axis=0)
        span = max(np.max(hi - lo) * 0.6 + 0.05, 0.2)
        cen = (lo + hi) / 2
        for ax in axes:
            ax.set_xlim(cen[0] - span, cen[0] + span)
            ax.set_ylim(cen[1] - span, cen[1] + span)
    else:
        for ax in axes:
            import textwrap
            msg = s.get("notes", [""])[0] if not s.get("position_valid") and s.get("notes") else ""
            body = "\n".join(textwrap.wrap(msg or "no head positions in the analysed window", 48))
            ax.text(0.5, 0.5, "no position data\n\n" + body, transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_SECONDARY, fontsize=8.5)
            ax.set_xlim(-half - 0.1, half + 0.1); ax.set_ylim(-half - 0.1, half + 0.1)
    axes[0].set_title(f"{prefix}Head path, coloured by time ({s['path_length_m']['head']:.2f} m over {s['duration_s']:.0f} s)", fontsize=11, loc="left")
    axes[1].set_title(f"{prefix}Head path, coloured by behavioural state", fontsize=11, loc="left")
    axes[2].set_title(f"{prefix}Head path, past / future of the present frame", fontsize=11, loc="left")
    fig.tight_layout()
    p = out_dir / "path_top_view.png"; fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    written["path_top_view"] = p.name

    # ---- 2. head orientation ------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 6.8), sharex=True, facecolor=SURFACE)
    for ax, key, lab in zip(axes, ("yaw", "pitch", "roll"), ("yaw (deg, heading in table frame)", "pitch (deg, + = nose up)", "roll (deg, about nose axis)")):
        _style_axis(ax)
        _shade_rest(ax, res["bouts"], fps)
        y = res[key]
        if key == "yaw":
            # break the line at +-180 wraps instead of drawing a vertical jump
            y = y.copy(); jumps = np.where(np.abs(np.diff(y)) > 180)[0]; y[jumps] = np.nan
        ax.plot(t, y, color=series_color, lw=1.2)
        ax.set_ylabel(lab, fontsize=9)
    axes[0].set_title(f"{prefix}Head orientation over time (rest bouts shaded)", fontsize=11, loc="left")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_ylim(-185, 185); axes[0].set_yticks([-180, -90, 0, 90, 180])
    fig.tight_layout()
    p = out_dir / "head_orientation.png"; fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    written["head_orientation"] = p.name

    # ---- 3. kinematics ------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 9), sharex=True, facecolor=SURFACE)
    thr = s["thresholds"]
    series = [
        ("ang_speed", "head angular speed (deg/s)", thr["rest_ang_speed_dps"]),
        ("ang_acc", "head angular acceleration (deg/s²)", None),
        ("lin_speed", "head linear speed (m/s)", thr["rest_lin_speed_mps"]),
        ("lin_acc", "head linear acceleration (m/s²)", None),
    ]
    for ax, (key, lab, th) in zip(axes, series):
        _style_axis(ax)
        _shade_rest(ax, res["bouts"], fps)
        y = res[key]
        if np.all(np.isnan(y)):
            ax.text(0.5, 0.5, "no position data (no meshes in range)", transform=ax.transAxes, ha="center", va="center", color=TEXT_SECONDARY)
        else:
            ax.plot(t, y, color=series_color, lw=1.1)
            if th is not None:
                ax.axhline(th, color=TEXT_SECONDARY, lw=1, ls="--")
                ax.annotate(f"rest threshold {th:g}", (t[-1], th), xytext=(-4, 4), textcoords="offset points",
                            ha="right", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].set_title(f"{prefix}Head kinematics (rest bouts shaded)", fontsize=11, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    p = out_dir / "head_kinematics.png"; fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    written["head_kinematics"] = p.name

    # ---- 4. states ----------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.5, 4.6), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.6, 1]})
    _style_axis(ax0); _style_axis(ax1)
    for k, st in enumerate(STATE_ORDER):
        spans = [(b["start_s"], b["duration_s"]) for b in res["bouts"] if b["state"] == st]
        if spans:
            ax0.broken_barh(spans, (k - 0.35, 0.7), facecolors=STATE_COLOR[st], edgecolor="none")
    ax0.set_yticks(range(len(STATE_ORDER))); ax0.set_yticklabels([STATE_LABEL[k] for k in STATE_ORDER])
    ax0.set_ylim(-0.7, len(STATE_ORDER) - 0.3); ax0.invert_yaxis()
    ax0.set_xlim(0, t[-1]); ax0.set_xlabel("time (s)")
    ax0.set_title(f"{prefix}Behavioural states ({s['rest_bouts']['count']} rest bouts, mean {s['rest_bouts']['mean_duration_s']:.1f} s)", fontsize=11, loc="left")
    ax0.grid(False, axis="y")
    fr = s["state_fractions"]
    left = 0
    for st in STATE_ORDER:
        wdt = fr[st] * 100
        ax1.barh([0], [wdt], left=left, color=STATE_COLOR[st], height=0.6, edgecolor=SURFACE, linewidth=2)
        if wdt > 4:
            ax1.text(left + wdt / 2, 0, f"{STATE_LABEL[st]}\n{wdt:.0f}%", ha="center", va="center", fontsize=8.5, color=TEXT_PRIMARY)
        left += wdt
    ax1.set_xlim(0, 100); ax1.set_yticks([]); ax1.set_xlabel("share of recording (%)")
    ax1.grid(False, axis="y"); ax1.spines["left"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "states.png"; fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    written["states"] = p.name
    return written


def downsample_for_viewer(res, max_points=6000):
    """Compact per-frame arrays for the 3D viewer: head path, nose direction
    (for the current-frame marker) + state, NaN -> null."""
    N = len(res["t"])
    step = max(1, N // max_points)
    idx = np.arange(0, N, step)
    def clean(a):
        return [None if np.isnan(v) else float(v) for v in a]
    return {
        "frame": [int(res["frame"][i]) for i in idx],
        "head": [clean(res["head"][i]) for i in idx],
        "body": [clean(res["body"][i]) for i in idx],
        "fwd": [clean(res["fwd"][i]) for i in idx],
        "state": [str(res["state"][i]) for i in idx],
        "pos_source": [str(res["pos_source"][i]) for i in idx],
    }


# =============================================================================
# Two rats: each rat is its own analyse_from_masks run (one mask label each);
# these only put the runs side by side.
# =============================================================================

def rat_distances(results):
    """Per-frame distance between the two rats' heads and between their body
    centroids (table frame, metres; NaN where either is unknown), plus their
    statistics. results: {mask label: analyse_from_masks result} over the same
    window; None unless it holds exactly two rats."""
    if len(results) != 2:
        return None
    a, b = (results[k] for k in sorted(results))
    out = {"frame": a["frame"]}
    for key in ("head", "body"):
        d = np.linalg.norm(a[key] - b[key], axis=1)
        ok = ~np.isnan(d)
        out[key] = d
        out[key + "_stats"] = {
            "n_frames": int(ok.sum()),
            "median": float(np.median(d[ok])) if ok.any() else None,
            "mean": float(np.mean(d[ok])) if ok.any() else None,
            "min": float(np.min(d[ok])) if ok.any() else None,
            "max": float(np.max(d[ok])) if ok.any() else None,
        }
    return out


def run_summary(results, reference_frame=None):
    """The run folder's run_summary.json: which rats were tracked, where each
    one's files are, a few headline numbers per rat, and (two rats) how far
    apart they were. Each rat's own summary.json holds everything else."""
    first = results[min(results)]["summary"]
    out = {
        "rats": [int(k) for k in sorted(results)],
        "colors": {f"rat{k}": RAT_COLOR.get(k) for k in sorted(results)},
        "fps": first["fps"], "window": first["window"], "reference_frame": reference_frame,
        "per_rat": {},
    }
    for k in sorted(results):
        s = results[k]["summary"]
        mf = s["mask_fusion"]
        out["per_rat"][f"rat{k}"] = {
            "folder": f"rat{k}", "mask_label": int(k),
            "frames_head_from_masks": mf["frames_head_from_masks"], "frames_head_from_mesh": mf["frames_head_from_mesh"],
            "frames_head_interpolated": mf["frames_head_interpolated"], "frames_head_missing": mf["frames_head_missing"],
            "fraction_with_heading": mf["fraction_with_heading"],
            "head_path_length_m": s["path_length_m"]["head"], "linear_speed_mps": s["linear_speed_mps"],
            "state_fractions": s["state_fractions"],
        }
    dist = rat_distances(results)
    if dist:
        out["between_rats_m"] = {"head_to_head": dist["head_stats"], "body_to_body": dist["body_stats"]}
    return out


def make_two_rat_plots(results, out_dir, table_size_m=1.6, reference_frame=None):
    """path_top_view_both.png: both rats' head paths in one top view (rat 1
    red, rat 2 blue, body-centroid paths in a lighter version of the same
    hue, each rat's head at the present / reference frame marked with a nose
    tick), and the head-to-head / body-to-body distance between them over
    time. Returns {key: filename}, empty unless results holds two rats."""
    dist = rat_distances(results)
    if dist is None:
        return {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    first = results[min(results)]
    t = first["t"]
    frame0 = int(first["frame"][0])
    ref_global = int(reference_frame) if reference_frame is not None else int(first["summary"]["anchor_frame"])
    ref_local = max(0, min(len(t) - 1, ref_global - frame0))
    ref_global = frame0 + ref_local

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.35]})
    _style_axis(ax); _style_axis(axd)
    half = table_size_m / 2
    ax.add_patch(Rectangle((-half, -half), table_size_m, table_size_m, facecolor="#f1efe9", edgecolor=GRID, lw=1, zorder=0))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), table frame")
    ax.set_ylabel("y (m)")
    handles, framed = [], []
    for k in sorted(results):
        res, col = results[k], RAT_COLOR.get(k, COLOR_SERIES[0])
        light = _mix(col, "#ffffff", 0.55)
        P, Bc, ok = res["head"], res["body"], res["has_pos"]
        handles += [Line2D([], [], color=col, lw=2, label=f"rat {k} head path"),
                    Line2D([], [], color=light, lw=1.2, label=f"rat {k} body centroid path")]
        if ok.sum() < 2:
            continue
        idx = np.where(ok)[0]
        ax.plot(Bc[idx, 0], Bc[idx, 1], color=light, lw=1.2, zorder=1)
        ax.plot(P[idx, 0], P[idx, 1], color=col, lw=2, zorder=2)
        ax.scatter(P[idx[0], 0], P[idx[0], 1], s=40, color=SURFACE, edgecolor=col, linewidth=1.8, zorder=5)
        ax.annotate(f"rat {k} start", P[idx[0], :2], xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=TEXT_PRIMARY)
        if ok[ref_local]:
            rp = P[ref_local]
            ax.scatter([rp[0]], [rp[1]], s=80, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=6)
            f = res["fwd"][ref_local]
            if not np.isnan(f[0]) and np.linalg.norm(f[:2]) > 1e-6:
                tip = rp[:2] + 0.04 * f[:2] / np.linalg.norm(f[:2])
                ax.annotate("", xy=tip, xytext=rp[:2], arrowprops=dict(arrowstyle="->", color=TEXT_PRIMARY, lw=1.4), zorder=7)
        framed.append(np.vstack([P[idx, :2], Bc[idx, :2]]))
    handles.append(Line2D([], [], marker="o", ls="", color=TEXT_SECONDARY, label=f"head at the present frame {ref_global}, nose tick"))
    if framed:
        allp = np.vstack(framed)
        lo = np.nanmin(allp, axis=0); hi = np.nanmax(allp, axis=0)
        span = max(np.max(hi - lo) * 0.6 + 0.05, 0.2)
        cen = (lo + hi) / 2
        ax.set_xlim(cen[0] - span, cen[0] + span); ax.set_ylim(cen[1] - span, cen[1] + span)
    else:
        ax.set_xlim(-half - 0.1, half + 0.1); ax.set_ylim(-half - 0.1, half + 0.1)
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
    ax.set_title("Head paths, rat 1 (red) and rat 2 (blue)", fontsize=11, loc="left")

    hs = dist["head_stats"]
    axd.plot(t, dist["head"] * 100, color=TEXT_PRIMARY, lw=1.4, label="head to head")
    axd.plot(t, dist["body"] * 100, color=TEXT_SECONDARY, lw=1.1, ls="--", label="body centroid to body centroid")
    axd.axvline(t[ref_local], color=TEXT_SECONDARY, lw=1, ls=":")
    axd.annotate(f"present (frame {ref_global})", (t[ref_local], 0), xycoords=("data", "axes fraction"), xytext=(4, 4),
                 textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    axd.set_xlabel("time (s)"); axd.set_ylabel("distance between the rats (cm)")
    dmax = np.nanmax(np.concatenate([dist["head"], dist["body"]])) * 100 if hs["median"] is not None else 1.0
    axd.set_ylim(0, dmax * 1.25)  # headroom for the legend
    axd.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY)
    axd.set_title("Distance between the rats" + (f" (heads: median {hs['median'] * 100:.1f} cm, closest {hs['min'] * 100:.1f} cm)"
                                                   if hs["median"] is not None else ""), fontsize=11, loc="left")
    fig.tight_layout()
    p = out_dir / "path_top_view_both.png"; fig.savefig(p, dpi=130, facecolor=SURFACE); plt.close(fig)
    return {"path_top_view_both": p.name}


# =============================================================================
# Re-projection onto the raw camera frames -- a visual check that the head
# pose above lands where the animal actually is in the footage, using the
# same rig JSON's per-camera extrinsics/intrinsics that placed the meshes.
# =============================================================================

def load_camera_extrinsics(calib_json_path):
    """{camera name -> raw calibration-frame rotation_matrix/translation_m/intrinsics},
    straight from the rig JSON. Distinct from build_initial_rig's direction-only,
    display-frame camera info: this keeps exactly what a pixel projection needs,
    X_cam = R @ X_calib + t, followed by the intrinsics."""
    data = json.loads(Path(calib_json_path).read_text())
    return {c["name"]: c for c in data.get("cameras", [])}


def _camera_matrix(cam):
    """3x4 projection matrix K[R|t]: homogeneous calibration-frame point -> pixel,
    ignoring lens distortion. Used only by the mask-triangulation helpers below,
    where a small distortion-induced bias near the image edge is immaterial
    next to the triangulation itself; project_to_camera (below) is the
    distortion-aware projection used for the reprojection-check overlays."""
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    intr = cam["intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["cx"]],
                  [0.0, intr["fy"], intr["cy"]],
                  [0.0, 0.0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


def _projection_jacobian(cam, P):
    """2x3 Jacobian d(u,v)/dP at the 3D point P (calibration frame), ignoring
    lens distortion: maps a calibration-frame direction to the pixel-space
    direction it would produce at P."""
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    intr = cam["intrinsics"]
    X, Y, Z = R @ np.asarray(P, dtype=np.float64) + t
    d_xy_dXc = np.array([[1.0 / Z, 0.0, -X / Z**2],
                         [0.0, 1.0 / Z, -Y / Z**2]])
    dK = np.diag([intr["fx"], intr["fy"]])
    return dK @ d_xy_dXc @ R


def mask_path(masks_root, cam_name, frame, kind="masks_raw"):
    """Rig camera names are 'camera_001'.. ; this assumes a same-index 'cam_1'..
    folder per camera, laid out as cam_<n>/<kind>/frame_<frame:06d>.png (GUI 2's
    mask export layout)."""
    n = int(re.sub(r"\D", "", cam_name) or 0)
    return Path(masks_root) / f"cam_{n}" / kind / f"frame_{int(frame):06d}.png"


def mask_heading(mask, label, nose_frac=0.22, min_pixels=80):
    """PCA long axis of one label's pixels in a mask array, disambiguated into a
    nose direction by comparing perpendicular spread near each end of the axis
    (the end that stays wider is the head; the tail tapers to a thin line).

    Returns None if the label has too few pixels (rat not in this mask), else
    {centroid (x,y), heading_deg (0..360, 0=+x/right, ccw), axis_length_px,
    confidence (0..1, how much wider the chosen head end is vs the tail end --
    low values mean the nose/tail call is uncertain)}."""
    ys, xs = np.where(mask == label)
    if len(xs) < min_pixels:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]  # unit vector, sign arbitrary
    proj = centered @ axis                  # position of every pixel along the axis
    perp = centered @ np.array([-axis[1], axis[0]])  # position across the axis

    lo, hi = proj.min(), proj.max()
    span = hi - lo
    if span < 1e-6:
        return None
    near_lo = proj < (lo + nose_frac * span)
    near_hi = proj > (hi - nose_frac * span)
    width_lo = perp[near_lo].std() if near_lo.any() else 0.0
    width_hi = perp[near_hi].std() if near_hi.any() else 0.0
    # the wider end is the head; point the returned heading that way
    sign = 1.0 if width_hi >= width_lo else -1.0
    heading_vec = axis * sign
    heading_deg = float(np.degrees(np.arctan2(heading_vec[1], heading_vec[0])) % 360)
    w_head, w_tail = max(width_lo, width_hi), min(width_lo, width_hi)
    confidence = float((w_head - w_tail) / w_head) if w_head > 1e-6 else 0.0

    return {
        "centroid": (float(centroid[0]), float(centroid[1])),
        "heading_deg": heading_deg,
        "axis_length_px": float(span),
        "confidence": confidence,
        "n_pixels": int(len(xs)),
    }


def _largest_component(m):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _geodesic_distance(body, sources):
    """Distance in pixels from the nearest source pixel to every body pixel,
    moving only through the body (8-connected, diagonal steps cost sqrt 2);
    inf outside the body."""
    idx = -np.ones(body.shape, int)
    ys, xs = np.nonzero(body)
    idx[ys, xs] = np.arange(len(ys))
    H, W = body.shape
    rows, cols, w = [], [], []
    for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        ok[ok] = body[y2[ok], x2[ok]]
        rows.append(idx[ys[ok], xs[ok]])
        cols.append(idx[y2[ok], x2[ok]])
        w.append(np.full(int(ok.sum()), c))
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(ys), len(ys))).tocsr()
    d = dijkstra(g, directed=False, indices=[idx[y, x] for y, x in sources], min_only=True)
    out = np.full(body.shape, np.inf)
    out[ys, xs] = d
    return out


def _farthest(d):
    return np.unravel_index(np.argmax(np.where(np.isfinite(d), d, -1)), d.shape)


TAIL_MIN_ELONGATION = 3.0     # a tail piece is at least this many times longer than wide
TAIL_MIN_LENGTH_FRAC = 0.15   # and at least this fraction of the body's through-the-body length


def _elongation(piece):
    """(length / width, length) of a pixel blob from its second moments."""
    ys, xs = np.nonzero(piece)
    if len(ys) < 3:
        return 0.0, 0.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(float))))
    length, width = 4 * np.sqrt(max(ev[1], 1e-9)), 4 * np.sqrt(max(ev[0], 1e-9))
    return length / max(width, 1.0), length


def mask_head(mask, label, min_pixels=500, tip_frac=0.12):
    """Where the head is in one camera's silhouette of one label.

    The tail is the part of the silhouette too thin to survive a morphological
    opening; what survives is the body (torso + head). Only a long, thin piece
    counts as the tail (TAIL_MIN_ELONGATION, TAIL_MIN_LENGTH_FRAC): the
    headstage sticking out above the head, a paw or an ear is trimmed off by
    the opening too, and taking one of those for the tail puts the "head" on
    the rump (seen on rat 2 during contact). Distances are measured
    through the body rather than straight across the image, so a curled or
    turning rat is followed along its own shape. If a tail is attached, the
    head is the body point farthest (through the body) from the tail base, and
    'head_px' is set. If no tail is visible (occluded, merged into the other
    rat's mask, curled under), the two body ends -- the pair of points farthest
    apart through the body -- are returned in 'ends_px' as unlabelled
    candidates, and fuse_head decides which one (if either) is the head from
    the cameras that did see a tail. Each point is the mean of the body pixels
    within tip_frac of the body's through-the-body length from that end (a
    small snout / rump tip region), in full-resolution image pixels.

    Works on a block-downsampled copy (~70 px across the body) for speed.
    Returns None if the label has too few pixels in this camera."""
    m = mask == label
    area = int(m.sum())
    if area < min_pixels:
        return None
    f = max(1, int(round(np.sqrt(area) / 70)))
    H, W = m.shape
    small = _largest_component(m[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f).mean((1, 3)) > 0.5)
    ys, xs = np.nonzero(small)
    if len(ys) < 20:
        return None
    y0, x0 = max(0, ys.min() - 3), max(0, xs.min() - 3)
    small = small[y0:ys.max() + 4, x0:xs.max() + 4]
    r = max(2, int(round(0.05 * np.sqrt(small.sum()))))
    body = _largest_component(ndimage.binary_opening(small, _disk(r)))
    if body.sum() < 20:
        return None
    # one body end and the body length (through the body), also the two
    # candidates when no tail is seen
    by, bx = np.nonzero(body)
    j = np.argmin((by - by.mean()) ** 2 + (bx - bx.mean()) ** 2)
    end_a = _farthest(_geodesic_distance(body, [(by[j], bx[j])]))
    d_a = _geodesic_distance(body, [end_a])
    body_len = float(np.max(d_a[np.isfinite(d_a)]))

    tail_base = None
    rest = small & ~ndimage.binary_dilation(body, _disk(1))
    lab, n = ndimage.label(rest, structure=np.ones((3, 3)))
    tail, tail_len = None, 0.0
    for k in range(1, n + 1):
        piece = lab == k
        if piece.sum() < 0.01 * body.sum():
            continue
        elong, length = _elongation(piece)
        if elong >= TAIL_MIN_ELONGATION and length >= TAIL_MIN_LENGTH_FRAC * body_len and length > tail_len:
            tail, tail_len = piece, length
    if tail is not None:
        touch = tail & ndimage.binary_dilation(body, _disk(2))
        if touch.any():
            ty, tx = np.nonzero(touch)
            j = np.argmin((by - ty.mean()) ** 2 + (bx - tx.mean()) ** 2)
            tail_base = (int(by[j]), int(bx[j]))

    if tail_base is not None:
        head_end, rump_end = _farthest(_geodesic_distance(body, [tail_base])), tail_base
    else:
        head_end, rump_end = end_a, _farthest(d_a)
    d_head = _geodesic_distance(body, [head_end])
    length = float(np.max(d_head[np.isfinite(d_head)]))
    d_rump = _geodesic_distance(body, [rump_end])

    def tip(d):
        ty, tx = np.nonzero(body & (d <= tip_frac * length))
        return (float((x0 + tx.mean() + 0.5) * f), float((y0 + ty.mean() + 0.5) * f))

    p_head = tip(d_head)
    return {
        "tail_seen": tail_base is not None,
        "head_px": p_head if tail_base is not None else None,
        "ends_px": (p_head, tip(d_rump)),
        "body_length_px": length * f,
    }


def _reprojection_px(cam, X):
    p = _camera_matrix(cam) @ np.append(X, 1.0)
    return p[:2] / p[2]


def fuse_head(cams, per_cam, tol_frac=0.15):
    """3D head (calibration frame) from every camera's mask_head result.

    Hypotheses are triangulated from every pair of cameras that saw a tail
    (so knew which body end is the head). Each hypothesis is scored by how
    many cameras agree with it: a tail-seeing camera agrees if its head point
    reprojects within tol_frac of its own body length; a camera without a
    tail agrees if either of its two body ends does (which settles, for that
    camera, which end is the head). The hypothesis with the most agreeing
    cameras wins and is re-triangulated from all of them. Returns
    (head or None, {camera: reprojection error px}) -- None if fewer than two
    cameras saw a tail or fewer than two agree."""
    seen = [n for n in per_cam if per_cam[n]["tail_seen"]]
    best = None
    for a, b in itertools.combinations(seen, 2):
        X = triangulate_point([(cams[a], per_cam[a]["head_px"]), (cams[b], per_cam[b]["head_px"])])
        support = {}
        for n, r in per_cam.items():
            options = [r["head_px"]] if r["tail_seen"] else list(r["ends_px"])
            errs = [np.linalg.norm(_reprojection_px(cams[n], X) - np.asarray(o)) for o in options]
            k = int(np.argmin(errs))
            if errs[k] < tol_frac * r["body_length_px"]:
                support[n] = options[k]
        score = (len(support), sum(per_cam[n]["tail_seen"] for n in support))
        if best is None or score > best[0]:
            best = (score, support)
    if best is None or len(best[1]) < 2:
        return None, {}
    support = best[1]
    X = triangulate_point([(cams[n], support[n]) for n in support])
    return X, {n: float(np.linalg.norm(_reprojection_px(cams[n], X) - np.asarray(support[n]))) for n in support}


def head_candidates(cams, per_cam, tol_frac=0.15, contra_frac=0.5, merge_m=0.015):
    """Every 3D head (calibration frame) at least two cameras agree on, for
    track_head to choose from over time; fuse_head's single best guess per
    frame is what goes wrong when the cameras split (two rats in contact).

    Every pair of per-camera options (the head point of a camera that saw the
    tail, either body end of one that did not) is triangulated; a camera
    supports the point if one of its options reprojects within tol_frac of its
    body length (the point is re-triangulated from its supporters), and
    contradicts it if none is within contra_frac (a camera that saw the tail
    counts twice either way). Near-duplicates are merged. Returns
    [{"X", "cams", "n", "n_tail", "contra"}], best supported first."""
    opts = {n: ([r["head_px"]] if r["tail_seen"] else list(r["ends_px"])) for n, r in per_cam.items()}
    names = list(opts)

    def errors(X):
        return {n: min(np.linalg.norm(_reprojection_px(cams[n], X) - np.asarray(o)) for o in opts[n])
                / per_cam[n]["body_length_px"] for n in names}

    def nearest(n, X):
        return min(opts[n], key=lambda o: np.linalg.norm(_reprojection_px(cams[n], X) - np.asarray(o)))

    hyps = []
    for a, b in itertools.combinations(names, 2):
        for pa in opts[a]:
            for pb in opts[b]:
                X = triangulate_point([(cams[a], pa), (cams[b], pb)])
                for _ in range(2):
                    sup = [n for n, e in errors(X).items() if e < tol_frac]
                    if len(sup) < 2:
                        break
                    X = triangulate_point([(cams[n], nearest(n, X)) for n in sup])
                e = errors(X)
                sup = sorted(n for n in names if e[n] < tol_frac)
                if len(sup) < 2:
                    continue
                w = {n: 2 if per_cam[n]["tail_seen"] else 1 for n in names}
                hyps.append({"X": X, "cams": sup, "n": len(sup), "n_tail": sum(per_cam[n]["tail_seen"] for n in sup),
                             "contra": sum(w[n] for n in names if e[n] > contra_frac)})
    hyps.sort(key=lambda h: (-(h["n"] + h["n_tail"] - h["contra"]), -h["n"]))
    kept = []
    for h in hyps:
        if all(np.linalg.norm(h["X"] - k["X"]) > merge_m for k in kept):
            kept.append(h)
    return kept


HEAD_TRACK = {"sigma_m": 0.015, "lookback": 10, "skip": 1.5, "restart": 40.0, "w_tail": 2.0, "w_cam": 1.0,
              "w_contra": 1.0, "body_min_m": 0.04, "body_max_m": 0.15, "w_body_per_cm": 8.0}


def track_head(cands, body, **kw):
    """The head over time: one of each frame's head_candidates, or none, chosen
    jointly for all frames (Viterbi). A candidate scores w_tail per supporting
    camera that saw the tail, w_cam per other supporter, minus w_contra per
    contradicting camera and w_body_per_cm per cm closer to the body centroid
    than body_min_m or farther than body_max_m. Moving from a head at frame
    t-k to one at frame t costs (distance / (sigma_m sqrt k))^2, each frame
    without a head costs `skip` (up to `lookback` in a row keep the track),
    and after a longer loss the track restarts anywhere for `restart`.
    cands: per frame a list of head_candidates (calibration frame), body: per
    frame the body centroid or None. Returns per frame the chosen index or None."""
    p = {**HEAD_TRACK, **kw}
    N = len(cands)
    E = []
    for t in range(N):
        e = []
        for h in cands[t]:
            s = p["w_tail"] * h["n_tail"] + p["w_cam"] * (h["n"] - h["n_tail"]) - p["w_contra"] * h["contra"]
            if body[t] is not None:
                d = float(np.linalg.norm(np.asarray(h["X"]) - np.asarray(body[t])))
                s -= p["w_body_per_cm"] * 100 * max(0.0, p["body_min_m"] - d, d - p["body_max_m"])
            e.append(-s)
        E.append(np.array(e))
    INF = 1e18
    cost = [np.full(len(c), INF) for c in cands]
    back = [[None] * len(c) for c in cands]
    pos = [np.array([h["X"] for h in c]).reshape(-1, 3) for c in cands]
    null = np.full(N + 1, INF)            # best cost with no head at frame t-1 (position forgotten)
    null_back = [None] * (N + 1)
    null[0] = 0.0
    for t in range(N):
        best, arg = null[t] + p["skip"], ("null", t)
        if t > 0 and len(cost[t - 1]):
            j = int(np.argmin(cost[t - 1]))
            if cost[t - 1][j] + p["skip"] < best:
                best, arg = cost[t - 1][j] + p["skip"], ("c", t - 1, j)
        null[t + 1], null_back[t + 1] = best, arg
        for i in range(len(cands[t])):
            best, b = null[t] + p["restart"], ("null", t)
            for k in range(1, p["lookback"] + 1):
                s = t - k
                if s < 0:
                    break
                if not len(cost[s]):
                    continue
                d = np.linalg.norm(pos[s] - pos[t][i], axis=1)
                c = cost[s] + (d / (p["sigma_m"] * np.sqrt(k))) ** 2 + p["skip"] * (k - 1)
                j = int(np.argmin(c))
                if c[j] < best:
                    best, b = c[j], ("c", s, j)
            cost[t][i] = E[t][i] + best
            back[t][i] = b
    best, state = null[N], ("null", N)
    for t in range(max(0, N - p["lookback"]), N):
        if len(cost[t]):
            j = int(np.argmin(cost[t]))
            c = cost[t][j] + p["skip"] * (N - 1 - t)
            if c < best:
                best, state = c, ("c", t, j)
    choice = [None] * N
    while state is not None:
        if state[0] == "c":
            _, t, j = state
            choice[t] = j
            state = back[t][j]
        else:
            if state[1] == 0:
                break
            state = null_back[state[1]]
    return choice


def triangulate_point(cams_uv):
    """cams_uv: [(cam_dict, (u, v)), ...] from >=2 cameras -> the 3D point (calib
    frame) whose reprojection best matches all of them, by linear DLT. Ignores
    lens distortion (small near image centre, where these rats mostly are)."""
    rows = []
    for cam, (u, v) in cams_uv:
        Pm = _camera_matrix(cam)
        rows.append(u * Pm[2] - Pm[0])
        rows.append(v * Pm[2] - Pm[1])
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def reproject_direction(cam, P, D):
    """The heading angle (deg, 0..360) that 3D direction D at 3D point P would
    produce in `cam`'s image, via the same linear model triangulate_direction
    fits against -- used both to resolve D's sign and to check the fused
    direction against each camera's raw 2D observation."""
    proj = _projection_jacobian(cam, P) @ np.asarray(D, dtype=np.float64)
    return float(np.degrees(np.arctan2(proj[1], proj[0])) % 360)


def triangulate_direction(cams_heading, P, min_confidence=0.15):
    """cams_heading: [(cam_dict, heading_deg, confidence), ...] -> (D, n_used):
    the unit 3D direction (calibration frame) at 3D point P best explaining
    every camera's observed 2D heading, or (None, n) if fewer than 2 cameras
    clear min_confidence.

    Fusing several cameras' 2D headings into one 3D heading is a different
    problem from the usual multi-view point triangulation: a 2D heading alone
    does not pin down a 3D direction the way a 2D point pins down a ray,
    because the observed angle is a projection of the direction's *effect on
    the image* at a specific 3D location. For a small step P + t*D, the pixel
    position moves as d(u,v)/dt = J_c(P) @ D (J_c = _projection_jacobian);
    observing a 2D heading means J_c(P) @ D is parallel to the observed unit
    vector d_c, i.e. perp(d_c)^T @ J_c(P) @ D = 0, one linear homogeneous
    equation in D's 3 unknowns per camera. Two or more cameras give the
    least-squares solution (smallest singular vector of the stacked
    constraint matrix); the sign is resolved afterwards by checking which one
    actually points the same way as the observed d_c in each camera."""
    used = [(c, h, w) for c, h, w in cams_heading if w >= min_confidence]
    if len(used) < 2:
        return None, len(used)

    rows = []
    for cam, heading_deg, w in used:
        d = np.array([np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))])
        perp = np.array([-d[1], d[0]])
        J = _projection_jacobian(cam, P)
        rows.append(np.sqrt(w) * (perp @ J))
    _, _, Vt = np.linalg.svd(np.stack(rows))
    D = Vt[-1]
    D /= np.linalg.norm(D)

    agreement = sum(w * np.cos(np.radians(reproject_direction(cam, P, D) - heading_deg))
                    for cam, heading_deg, w in used)
    if agreement < 0:
        D = -D
    return D, len(used)


def fuse_frame(masks_root, cams, frame, label=1, min_confidence=0.15):
    """cams: {camera name -> cam dict, from load_camera_extrinsics}. For every
    camera's mask at this frame: mask_heading (silhouette centroid and long
    axis) and mask_head (where the head is). Triangulates the 3D body
    centroid from all detections, the 3D heading axis from the confident ones
    (triangulate_direction), and the 3D head from the cameras that agree on it
    (fuse_head).

    Returns a dict; 'centroid' is None if fewer than 2 cameras saw the label,
    'direction' is None if fewer than 2 cleared min_confidence, 'head' is None
    if fewer than 2 cameras agreed on a head ('n_cams_head' is how many did;
    the caller decides how many it needs to trust it)."""
    from PIL import Image
    per_cam, per_cam_head = {}, {}
    for name in cams:
        p = mask_path(masks_root, name, frame)
        if not p.exists():
            continue
        mask = np.asarray(Image.open(p))
        r = mask_heading(mask, label)
        if r:
            per_cam[name] = r
        h = mask_head(mask, label)
        if h:
            per_cam_head[name] = h
    head, head_err = fuse_head(cams, per_cam_head)
    out = {"n_cams": len(per_cam), "n_cams_used_for_direction": 0, "centroid": None, "direction": None,
           "head": head.tolist() if head is not None else None, "n_cams_head": len(head_err),
           "n_cams_tail_seen": sum(h["tail_seen"] for h in per_cam_head.values()),
           "head_reprojection_error_px": head_err,
           "head_candidates": [{**h, "X": h["X"].tolist()} for h in head_candidates(cams, per_cam_head)]}
    if len(per_cam) < 2:
        return out

    P = triangulate_point([(cams[n], per_cam[n]["centroid"]) for n in per_cam])
    cams_heading = [(cams[n], per_cam[n]["heading_deg"], per_cam[n]["confidence"]) for n in per_cam]
    D, n_used = triangulate_direction(cams_heading, P, min_confidence)

    reproj_err = {}
    if D is not None:
        for n in per_cam:
            est = reproject_direction(cams[n], P, D)
            reproj_err[n] = float((est - per_cam[n]["heading_deg"] + 180) % 360 - 180)

    out.update({
        "n_cams_used_for_direction": n_used,
        "centroid": P.tolist(), "direction": D.tolist() if D is not None else None,
        "per_camera": per_cam, "reprojection_error_deg": reproj_err,
    })
    return out


def project_to_camera(points_C, cam):
    """points_C: (N,3) in the calibration frame -> (uv (N,2) pixels, in_front (N,) bool).

    Pinhole model with a single radial term (k1), matching the rig JSON's intrinsics."""
    points_C = np.atleast_2d(np.asarray(points_C, dtype=np.float64))
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    intr = cam["intrinsics"]
    Xc = points_C @ R.T + t
    z = Xc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = Xc[:, 0] / safe_z
    y = Xc[:, 1] / safe_z
    r2 = x * x + y * y
    k1 = float(intr.get("k1", 0.0))
    d = 1.0 + k1 * r2
    u = intr["fx"] * (x * d) + intr["cx"]
    v = intr["fy"] * (y * d) + intr["cy"]
    return np.stack([u, v], axis=1), z > 0


def _camera_image_path(image_root, cam_name, frame):
    """Rig camera names are 'camera_001'.. ; this assumes a same-index 'cam_1'..
    folder per camera under image_root, laid out as
    cam_<n>/all_objects/frame_<frame:06d>.png (matches GUI 2's raw-frame export).
    Adjust here if a given dataset uses a different layout or numbering."""
    n = int(re.sub(r"\D", "", cam_name) or 0)
    return Path(image_root) / f"cam_{n}" / "all_objects" / f"frame_{int(frame):06d}.png"


def make_camera_overlays(results, out_dir, B, tc, mesh_frame, calib_json_path, image_root,
                          reference_frame=None, image_size=(1920, 1200), log=None):
    """Project every tracked rat's head path onto every camera's raw frame at the
    reference frame, to visually check the alignment against the actual footage.
    results: {mask label: analyse_from_masks result} (a single result is taken as
    its own label); each rat is drawn in its RAT_COLOR, past in full colour,
    future lighter, present as a dot with a nose tick. One combined PNG, one
    panel per camera. Returns the filename, or None if no image_root is
    configured or no camera has an image."""
    if "t" in results:
        results = {results["summary"]["mask_fusion"]["label"]: results}
    if not image_root or not any(r["has_pos"].any() for r in results.values()):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.collections import LineCollection

    cams = load_camera_extrinsics(calib_json_path)
    if not cams:
        return None
    out_dir = Path(out_dir)
    first = results[min(results)]  # every rat covers the same window
    t = first["t"]
    N = len(t)
    frame0 = int(first["frame"][0])  # global frame number of array position 0
    # reference_frame, like anchor_frame, is a GLOBAL frame number; ref_local
    # indexes into this window's arrays, ref_global names the real video frame
    # (and is what the raw camera image filenames are keyed on)
    ref_global = int(reference_frame) if reference_frame is not None else int(first["summary"]["anchor_frame"])
    ref_local = max(0, min(N - 1, ref_global - frame0))
    ref_global = frame0 + ref_local

    tracks = []  # (label, colour, head_C (N,3), nose_C (N,3) with a 3cm nose tick)
    for k in sorted(results):
        res = results[k]
        tracks.append((k, RAT_COLOR.get(k, "#eda100"), table_to_calib(res["head"], B, tc, mesh_frame),
                       table_to_calib(res["head"] + res["fwd"] * 0.03, B, tc, mesh_frame)))

    names = list(cams.keys())
    cols = min(3, len(names))
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.7 * rows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    any_image = False
    for ax, name in zip(axes, names):
        ax.set_facecolor("#1a1a1a")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        cam = cams[name]
        img_path = _camera_image_path(image_root, name, ref_global)
        if img_path.exists():
            ax.imshow(mpimg.imread(img_path), extent=(0, image_size[0], image_size[1], 0))
            any_image = True
        else:
            ax.text(0.5, 0.5, "image not found:\n" + str(img_path), transform=ax.transAxes,
                    ha="center", va="center", color="#ddd", fontsize=7, wrap=True)

        for k, col, head_C, nose_C in tracks:
            uv, front = project_to_camera(head_C, cam)
            margin = 300
            inb = front & (uv[:, 0] > -margin) & (uv[:, 0] < image_size[0] + margin) \
                         & (uv[:, 1] > -margin) & (uv[:, 1] < image_size[1] + margin)
            if inb.sum() > 1:
                seg_ok = inb[:-1] & inb[1:]
                segs = np.stack([uv[:-1], uv[1:]], axis=1)[seg_ok]
                is_future = np.arange(N - 1) >= ref_local
                colors = [(_mix(col, "#ffffff", 0.55) if f else col) for f, ok in zip(is_future, seg_ok) if ok]
                if len(segs):
                    ax.add_collection(LineCollection(segs, colors=colors, linewidths=1.6, alpha=0.9))
            if front[ref_local] and inb[ref_local]:
                u0, v0 = uv[ref_local]
                ax.scatter([u0], [v0], s=70, facecolor=col, edgecolor="#ffffff", linewidth=1.2, zorder=6)
                uv_nose, nose_front = project_to_camera(nose_C[ref_local:ref_local + 1], cam)
                if nose_front[0]:
                    u1, v1 = uv_nose[0]
                    ax.annotate("", xy=(u1, v1), xytext=(u0, v0),
                                arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.6))

        ax.set_xlim(0, image_size[0]); ax.set_ylim(image_size[1], 0)
        title = name + (" (reference cam)" if cam.get("is_reference") else "")
        ax.set_title(title, fontsize=9, color=TEXT_PRIMARY, loc="left")
    for ax in axes[len(names):]:
        ax.axis("off")
    who = " and ".join(f"rat {k} ({'red' if k == 1 else 'blue' if k == 2 else 'gold'})" for k, *_ in tracks)
    fig.suptitle(f"Head path of {who} re-projected onto the {len(names)} camera views at frame {ref_global} "
                 f"(past = full colour, future = lighter, present = dot + white nose tick)",
                 fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "camera_overlays.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE)
    plt.close(fig)
    if log:
        log(("camera overlay written" if any_image else f"camera overlay written, but no image found under {image_root}") + f" -> {p.name}")
    return p.name
