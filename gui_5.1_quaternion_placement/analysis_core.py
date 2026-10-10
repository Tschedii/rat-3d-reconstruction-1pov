#!/usr/bin/env python3
"""
GUI 5's head-pose analysis (formerly GUI 5.2, promoted to GUI 5; the earlier
mesh-centroid-based approach lives on as GUI 5.1). Inputs: the rig
calibration, the head IMU quaternion CSV, every rig camera's GUI 2
segmentation masks, and ONE anchor frame's reconstruction of the IMU rat (GUI
2 mesh.obj + GUI 3 marked eyes). compute() analyses one rat; with two rats
wearing IMUs, server.py calls it once per rat with that rat's settings.

Position (multi-view). Per camera and frame, the IMU rat's mask gives a
silhouette centroid (tail stripped by a morphological opening) and long axis.
The centroids of all cameras that see the rat are triangulated (DLT on
undistorted points, worst camera dropped while its reprojection error is
large and >= 3 remain), giving the body's 3D position including height, so
rearing is no longer squashed onto a fixed plane. The 2D axis lines are fused
into one 3D body axis: each camera's line constrains the axis direction D to
the plane back-projected through it, perp(d_c)^T J_c(P) D = 0 with J_c the
camera's projection Jacobian at the triangulated point; the SVD null vector of
the stacked, elongation-weighted rows is the axis (a line: its sign is picked
later, per walking bout, from the direction of travel).

Orientation. The IMU's reference frame E is gravity-referenced (sensor +z
along reference +z for 95% of frames on Orientation_for_video.csv), so pitch
and roll are real and only the heading of E is unknown:

    R_W<-E(j) = Rz(psi_j),   R_W<-H(j) = R_W<-E(j)  R(q_j)  R_S<-H

psi drifts on this recording (hundreds of degrees over minutes), so it is
re-aligned at anchor frames every anchor_every_frames frames: at each anchor,
psi = circular mean over that anchor's frames of (fused body heading - IMU
heading), then linear interpolation between anchors. Small spacing follows
drift closely but forces the head's heading towards the BODY's (head-vs-body
turns shorter than the spacing are averaged away); large spacing keeps them
but lets drift build up between anchors. anchor_every_frames = 0: one offset
from the marked eyes at the anchor frame only.

Head point: body position + the anchor's body-to-eyes vector carried in the
head frame (the same approximation GUI 5.1 made):

    offset_H = R_W<-H(k)^T (eyes(k) - body(k)),   head(j) = body(j) + R_W<-H(j) offset_H

Checked on this data: the IMU is on object 1, CSV row 0 is video frame 0
(within ~0.3 s), label 1 = object 1 in all six cameras, and the frame-240
meshes land on the rats in all three raw videos (cam_1, cam_2, cam_4). Of
the two Day 7 IMU files, rPGE_007_... (= Orientation_for_video.csv) is on
object 1 and rPGE008_... on object 2: held-out heading-vs-walking R 0.84 and
0.77 for those pairings, 0.54 and 0.48 swapped.
"""

import csv
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import topview as tv

STATE_THRESHOLDS = {"rest_ang_speed_dps": 20.0, "rest_lin_speed_mps": 0.05}
NOSE_TICK_M = 0.04
TRIANG_REJECT_PX = 30.0


# =============================================================================
# Inputs
# =============================================================================

def _quat_columns(cols):
    """Indices of the w, x, y, z columns in a header: qw,qx,qy,qz, or any
    common prefix ending in _w/_x/_y/_z (qc_w,... from the .mat export).
    None if there is no such set."""
    if all(k in cols for k in ("qw", "qx", "qy", "qz")):
        return [cols.index(k) for k in ("qw", "qx", "qy", "qz")]
    for c in cols:
        if c.endswith("_w"):
            p = c[:-1]
            if all(p + a in cols for a in "xyz"):
                return [cols.index(p + a) for a in "wxyz"]
    return None


def parse_orientation_csv(path):
    """Rows are IMU samples, one per video frame. (N,4) wxyz, unit, sign-continuous.
    A CSV with qw,qx,qy,qz (or qc_w,qc_x,... ) columns, or the 4 first columns
    without a header; or a MATLAB .mat with one (N,4) wxyz array (qc_video)."""
    if Path(path).suffix.lower() == ".mat":
        from scipy.io import loadmat
        m = {k: np.asarray(v) for k, v in loadmat(path).items() if not k.startswith("__")}
        cand = [k for k, v in m.items() if v.ndim == 2 and v.shape[1] == 4 and np.issubdtype(v.dtype, np.number)]
        if not cand:
            raise ValueError(f"{path}: no (N,4) quaternion array (variables: {', '.join(m)})")
        rows = m["qc_video" if "qc_video" in cand else cand[0]]
    else:
        rows = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx = _quat_columns([h.strip().lower() for h in header])
            if idx is None:
                idx = [0, 1, 2, 3]
                rows.append([float(header[i]) for i in idx])
            for r in reader:
                if r and r[0].strip():
                    rows.append([float(r[i]) for i in idx])
    q = np.asarray(rows, dtype=np.float64)
    n = np.linalg.norm(q, axis=1)
    q[n < 1e-9] = [1.0, 0.0, 0.0, 0.0]
    q /= np.linalg.norm(q, axis=1)[:, None]
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def mirror_roll(R_ES, R_SH):
    """Flip the sign of the head's roll (rotation about the nose axis), keeping
    heading and pitch: ZYX Euler angles of the head frame, roll negated. Not a
    physical mounting (a reflection, not a rotation), kept only as an option:
    on Orientation_for_video.csv the mirrored roll really comes from the IMU
    being mounted backwards (sensor +x to the tail, sensor_forward = -x), which
    flips roll AND pitch; this flip fixed the roll but left nose-down
    sequences drawn nose-up (e.g. frames 390-460, rat pressed to the wall)."""
    e = Rotation.from_matrix(R_ES @ R_SH).as_euler("ZYX")
    e[:, 2] *= -1
    return Rotation.from_euler("ZYX", e).as_matrix() @ R_SH.T


def head_from_sensor_axes(forward_axis, up_axis):
    """R_S<-H with head x = nose, y = left, z = up, from names like '+x', '-z'."""
    def vec(name):
        name = name.strip().lower()
        v = np.zeros(3)
        v["xyz".index(name.lstrip("+-"))] = -1.0 if name.startswith("-") else 1.0
        return v
    f, u = vec(forward_axis), vec(up_axis)
    if abs(np.dot(f, u)) > 1e-6:
        raise ValueError("sensor forward and up axes must be different axes")
    return np.stack([f, np.cross(u, f), u], axis=1)


def obj_vertices(path):
    pts = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(pts, dtype=np.float64)


def calib_to_table(p, B, tc):
    return np.asarray(p, dtype=np.float64) @ np.asarray(B).T - np.asarray(tc)


def cam_number(name):
    return int(re.sub(r"\D", "", name) or 0)


def masks_dir_for(masks_root, cam_name):
    return Path(masks_root) / f"cam_{cam_number(cam_name)}" / "masks_raw"


# =============================================================================
# Per-camera mask features (the slow, cached step)
# =============================================================================

def mask_features_px(mask_path, label, step=2, open_px=9, axis_len_px=40.0):
    """(cx, cy, ex, ey, elong) in full-resolution pixels for one label in a
    GUI 2 mask PNG: silhouette centroid (tail stripped by an opening, ~18 px
    at full resolution), a point axis_len_px along its long axis (a line, sign
    arbitrary), and elongation 1 - l_min/l_max of the second moments (0 for a
    round blob, whose axis means nothing). None if the label is absent."""
    p = Path(mask_path)
    if not p.exists():
        return None
    b = (np.asarray(Image.open(p))[::step, ::step] == label).astype(np.uint8)
    if b.sum() < 30:
        return None
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if opened.sum() >= 30:
        b = opened
    mo = cv2.moments(b, binaryImage=True)
    cx, cy = mo["m10"] / mo["m00"] * step, mo["m01"] / mo["m00"] * step
    a, c, bb = mo["mu20"], mo["mu02"], mo["mu11"]
    theta = 0.5 * math.atan2(2 * bb, a - c)
    root = math.sqrt(((a - c) / 2) ** 2 + bb * bb)
    l_max, l_min = (a + c) / 2 + root, (a + c) / 2 - root
    elong = 1.0 - l_min / l_max if l_max > 0 else 0.0
    return cx, cy, cx + axis_len_px * math.cos(theta), cy + axis_len_px * math.sin(theta), elong


def mask_features(masks_dir, label, frames, cache_dir=None, workers=8, progress=None):
    """(len(frames),5) per-frame mask_features_px, NaN where the rat isn't in
    the mask. Cached on disk per masks folder."""
    masks_dir = Path(masks_dir)
    key = hashlib.sha1(f"{masks_dir.resolve()}|{label}|{frames[0]}|{frames[-1]}|v3".encode()).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"mask_features_{key}.npy" if cache_dir else None
    if cache_path and cache_path.exists():
        out = np.load(cache_path)
        if len(out) == len(frames):
            if progress:
                progress(len(frames), len(frames))
            return out
    out = np.full((len(frames), 5), np.nan)
    paths = [masks_dir / f"frame_{int(f):06d}.png" for f in frames]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, c in enumerate(ex.map(lambda p: mask_features_px(p, label), paths)):
            if c is not None:
                out[i] = c
            done += 1
            if progress and (done % 200 == 0 or done == len(frames)):
                progress(done, len(frames))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out


# =============================================================================
# Multi-view geometry
# =============================================================================

def _K_dist(cam):
    k = cam["intrinsics"]
    K = np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])
    return K, np.array([k.get("k1", 0.0), k.get("k2", 0.0), k.get("p1", 0.0), k.get("p2", 0.0), k.get("k3", 0.0)])


def undistort_normalized(px, cam):
    """(n,2) pixels (NaN allowed) -> (n,2) undistorted normalized coordinates."""
    out = np.full_like(px, np.nan)
    ok = ~np.isnan(px[:, 0])
    if ok.any():
        K, dist = _K_dist(cam)
        out[ok] = cv2.undistortPoints(px[ok].reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    return out


def triangulate(obs, Ps, fx, reject_px=TRIANG_REJECT_PX):
    """Batched DLT. obs: (n,C,2) normalized coords (NaN = not seen), Ps: (C,3,4)
    [R|t], fx: (C,) focal lengths (to express errors in pixels). Returns
    X (n,3), active (n,C) bool, residual_px (n,) (max over active cameras)."""
    n, C, _ = obs.shape
    active = ~np.isnan(obs[:, :, 0])
    o = np.nan_to_num(obs)
    X = np.full((n, 3), np.nan)
    for _ in range(C):
        A = np.concatenate([o[:, :, 0:1] * Ps[None, :, 2, :] - Ps[None, :, 0, :],
                            o[:, :, 1:2] * Ps[None, :, 2, :] - Ps[None, :, 1, :]], axis=1)   # (n,2C,4)
        A *= np.concatenate([active, active], axis=1)[:, :, None]
        ok = active.sum(1) >= 2
        Xh = np.full((n, 4), np.nan)
        if ok.any():
            Xh[ok] = np.linalg.svd(A[ok])[2][:, -1, :]
        X = Xh[:, :3] / Xh[:, 3:4]
        Xc = np.einsum("cij,nj->nci", Ps[:, :, :3], X) + Ps[None, :, :, 3]
        proj = Xc[:, :, :2] / Xc[:, :, 2:3]
        err = np.linalg.norm(proj - o, axis=2) * fx[None, :]
        err = np.where(active, err, -1.0)
        worst = np.nanargmax(np.nan_to_num(err, nan=-1.0), axis=1)
        worst_err = err[np.arange(n), worst]
        drop = ok & (active.sum(1) >= 4) & (worst_err > reject_px)
        if not drop.any():
            break
        active[np.where(drop)[0], worst[drop]] = False
    resid = np.where(active, err, np.nan)
    resid = np.nanmax(np.where(np.isnan(resid), -np.inf, resid), axis=1)
    resid[~np.isfinite(resid)] = np.nan
    X[active.sum(1) < 2] = np.nan
    return X, active, resid


def fuse_axes(obs_c, obs_e, weight, active, Ps, X):
    """3D axis line per frame from each camera's 2D axis line. obs_c / obs_e:
    (n,C,2) normalized centroid / axis-endpoint, weight (n,C), active (n,C),
    X (n,3) triangulated point. Returns D (n,3) unit (sign arbitrary) and a
    quality in [0,1] (1 - s_min/s_mid of the stacked constraints: 1 = the
    cameras' planes intersect in one clean line)."""
    n, C, _ = obs_c.shape
    R = Ps[:, :, :3]; t = Ps[:, :, 3]
    Xc = np.einsum("cij,nj->nci", R, X) + t[None]                      # (n,C,3)
    Z = Xc[:, :, 2]
    J = np.zeros((n, C, 2, 3))
    J[:, :, 0, 0] = 1 / Z; J[:, :, 0, 2] = -Xc[:, :, 0] / Z ** 2
    J[:, :, 1, 1] = 1 / Z; J[:, :, 1, 2] = -Xc[:, :, 1] / Z ** 2
    J = np.einsum("ncij,cjk->ncik", J, R)                              # d(normalized)/dX, (n,C,2,3)
    d = obs_e - obs_c
    perp = np.stack([-d[:, :, 1], d[:, :, 0]], axis=2)
    rows = np.einsum("nci,ncij->ncj", perp, J)
    norm = np.linalg.norm(rows, axis=2, keepdims=True)
    w = np.where(active & ~np.isnan(weight), np.nan_to_num(weight), 0.0)
    rows = np.nan_to_num(rows / np.maximum(norm, 1e-12)) * w[:, :, None]
    ok = ((w > 0.15).sum(1) >= 2) & ~np.isnan(X[:, 0])
    D = np.full((n, 3), np.nan)
    qual = np.zeros(n)
    if ok.any():
        _, S, Vt = np.linalg.svd(rows[ok])
        D[ok] = Vt[:, -1, :]
        # with only 2 constraint rows the two planes meet in an exact line: no
        # redundancy to measure consistency with
        qual[ok] = 1.0 - S[:, -1] / np.maximum(S[:, -2], 1e-12) if S.shape[1] >= 3 else 1.0
    return D, qual


# =============================================================================
# Time-series helpers
# =============================================================================

def _runs(mask):
    """[(start, end_exclusive)] of True runs in a boolean array."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _smooth_runs(x, valid, window):
    y = x.copy()
    for s, e in _runs(valid):
        w = min(window, (e - s) if (e - s) % 2 == 1 else (e - s) - 1)
        if w >= 5:
            y[s:e] = savgol_filter(x[s:e], w, 2, axis=0, mode="interp")
    return y


def _deriv_runs(x, valid, window, dt):
    d = np.full_like(x, np.nan)
    for s, e in _runs(valid):
        if e - s < 3:
            continue
        w = min(window, (e - s) if (e - s) % 2 == 1 else (e - s) - 1)
        d[s:e] = savgol_filter(x[s:e], w, 2, deriv=1, delta=dt, axis=0, mode="interp") if w >= 5 \
            else np.gradient(x[s:e], dt, axis=0)
    return d


def branch_by_travel(line_ang, valid, yaw_travel, walk):
    """Body-axis LINE angles -> headings: unwrapped in doubled-angle space, the
    nose/tail branch re-chosen at every walking bout by majority vote against
    the direction of travel (rats essentially never walk backwards); the
    timeline is split halfway between bouts. A single unwrap is not enough: a
    round silhouette gives an arbitrary axis, and the branch can stay flipped
    for minutes (it did, 60-220 s, on this recording)."""
    n = len(line_ang)
    out = np.full(n, np.nan)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out
    out[idx] = np.unwrap(2 * line_ang[idx]) / 2
    bouts = [(s, e) for s, e in _runs(walk) if e - s >= 5]
    if not bouts:
        return out
    edges = [0] + [(bouts[i][1] + bouts[i + 1][0]) // 2 for i in range(len(bouts) - 1)] + [n]
    for (s, e), a, b in zip(bouts, edges[:-1], edges[1:]):
        agree = np.cos(out[s:e] - yaw_travel[s:e])
        agree = agree[~np.isnan(agree)]
        if len(agree) and agree.mean() < 0:
            out[a:b] += np.pi
    return out


def anchor_psi(ref_yaw, use, yaw_imu, every):
    """Heading offset psi per frame from anchors every `every` frames: at each
    anchor, the circular mean of (ref_yaw - yaw_imu) over that anchor's frames
    where `use`; linear interpolation (unwrapped) between anchors, constant
    beyond the first/last. Returns (psi, anchor_centres, anchor_psis)."""
    n = len(yaw_imu)
    centres, vals = [], []
    for a in range(0, n, every):
        b = min(n, a + every)
        m = use[a:b]
        if m.sum() >= max(2, int(0.2 * (b - a))):
            z = np.exp(1j * (ref_yaw[a:b][m] - yaw_imu[a:b][m])).mean()
            centres.append((a + b - 1) / 2); vals.append(np.angle(z))
    if not centres:
        return None, [], []
    vals = np.unwrap(vals)
    return np.interp(np.arange(n), centres, vals), centres, vals


# =============================================================================
# The analysis
# =============================================================================

MOUNT_TILT_MODES = ("auto_roll", "auto_roll_pitch", "manual", "off")
MOUNT_TILT_MIN_FRAMES = 100


def mount_correction(R_ES, R_SH, level_mask, mode="auto_roll", roll_deg=0.0, pitch_deg=0.0):
    """Constant rotation C (head frame) for an IMU mounted tilted on the
    headstage: R_W<-H = R_W<-E R_E<-S R_S<-H C. The tilt is removed by
    assuming the head is level ON AVERAGE while walking: the mean gravity
    direction seen in the measured head frame, g = mean_j (R_E<-S R_S<-H)^T z,
    is rotated onto head up (the heading of E does not enter, so this is known
    before the heading anchors). auto_roll: only the roll part (rotation about
    the nose axis); auto_roll_pitch: the full tilt; manual: C removes the
    given tilt (roll positive = right ear down, pitch positive = nose up).
    Returns (C, info) with the removed tilt reported in the manual convention."""
    mode = mode if mode in MOUNT_TILT_MODES else "auto_roll"
    info = {"mode": mode, "frames_used": 0, "source": None}
    if mode == "off":
        C = np.eye(3)
    elif mode == "manual":
        C = Rotation.from_euler("YX", [math.radians(float(pitch_deg or 0)), -math.radians(float(roll_deg or 0))]).as_matrix()
        info["source"] = "set in the form"
    else:
        use = level_mask if level_mask.sum() >= MOUNT_TILT_MIN_FRAMES else np.ones(len(R_ES), bool)
        info["frames_used"] = int(use.sum())
        info["source"] = "walking frames" if use is level_mask else "all frames (too little walking)"
        g = np.einsum("ji,nkj,k->ni", R_SH, R_ES[use], [0.0, 0.0, 1.0]).mean(0)
        if mode == "auto_roll":
            C = Rotation.from_euler("X", math.atan2(-g[1], g[2])).as_matrix()
        else:
            C = Rotation.align_vectors([g / np.linalg.norm(g)], [[0.0, 0.0, 1.0]])[0].as_matrix()
    pitch_c, roll_c, _ = Rotation.from_matrix(C).as_euler("YXZ", degrees=True)
    info["roll_deg"], info["pitch_deg"] = float(-roll_c), float(pitch_c)
    return C, info


HEAD_CLICK_PRIOR = 0.05      # pull of the eyes-based offset in the refit (rows of R are unit vectors)
HEAD_CLICK_MAX_M = 0.20      # a click farther than this from the tracked body is on the wrong rat


def _local_correction(ci, vals, n, reach):
    """(n,d) correction that equals vals[k] at frame index ci[k], is linear
    between clicks up to 2*reach apart and fades to zero over reach frames
    otherwise (zero everywhere for reach <= 0 except at the clicks)."""
    ci = np.asarray(ci); vals = np.asarray(vals, dtype=np.float64).reshape(len(ci), -1)
    out = np.zeros((n, vals.shape[1]))
    if not len(ci):
        return out
    order = np.argsort(ci); ci, vals = ci[order], vals[order]
    if reach <= 0:
        out[ci] = vals
        return out
    K = len(ci)
    j = np.arange(n)
    p = np.searchsorted(ci, j, side="right") - 1          # last click at or before j
    q = p + 1
    hasp, hasq = p >= 0, q < K
    pc, qc = np.clip(p, 0, K - 1), np.clip(q, 0, K - 1)
    dp, dq = j - ci[pc], ci[qc] - j
    wp = np.where(hasp, np.clip(1 - dp / reach, 0, 1), 0.0)
    wq = np.where(hasq, np.clip(1 - dq / reach, 0, 1), 0.0)
    out = wp[:, None] * vals[pc] + wq[:, None] * vals[qc]
    gap = ci[qc] - ci[pc]
    bridge = hasp & hasq & (gap > 0) & (gap <= 2 * reach)
    t = np.where(bridge, dp / np.maximum(gap, 1), 0.0)
    out[bridge] = ((1 - t)[:, None] * vals[pc] + t[:, None] * vals[qc])[bridge]
    return out


def place_head(ctx, clicks=(), reach_frames=0):
    """Head pose per frame from the user's clicks {frame[, x, y][, heading_deg | q]}
    (true table xy of the head; nose heading in the table frame, or the full
    head orientation q = wxyz set by rotating the head in the 3D inset).

    Orientation: a click with q re-aligns the gyroscope to exactly that
    orientation there; one with heading_deg only turns it about vertical so
    the nose points that way. The correction (a rotation vector, world frame)
    is exact at the clicked frame and carried over reach_frames (see
    _local_correction). Position: body + R_W<-H offset_H, with offset_H refit
    from the clicked positions by least squares, regularised towards the marked
    eyes' offset (one click fixes the two horizontal directions it sees), plus
    the remaining error removed locally like the heading. Returns the corrected
    orientation too (R_WH, q wxyz, fwd)."""
    frames, body, valid = ctx["frames"], ctx["body"], ctx["valid"]
    R_WH, fwd = ctx["R_WH"], ctx["fwd"]
    o0 = np.asarray(ctx["offset0_H"], dtype=np.float64)
    n = len(frames)
    first = int(frames[0])
    fit = {"n_clicks": 0, "rms_before_cm": None, "rms_after_cm": None, "per_click_cm": [],
           "n_heading_clicks": 0, "heading_change_deg": []}

    # --- orientation clicks: world-frame rotation corrections ------------------
    oc = []
    for c in clicks:
        i = int(c["frame"]) - first
        if not 0 <= i < n:
            continue
        if c.get("q") is not None:
            w, x, y, z = (float(v) for v in c["q"])
            dR = Rotation.from_quat([x, y, z, w]) * Rotation.from_matrix(R_WH[i]).inv()
            oc.append((i, dR.as_rotvec()))
            fit["n_orientation_clicks"] = fit.get("n_orientation_clicks", 0) + 1
        elif c.get("heading_deg") is not None:
            yaw_i = math.atan2(fwd[i, 1], fwd[i, 0])
            d = (math.radians(float(c["heading_deg"])) - yaw_i + np.pi) % (2 * np.pi) - np.pi
            oc.append((i, np.array([0.0, 0.0, d])))
            fit["n_heading_clicks"] += 1
            fit["heading_change_deg"].append(float(np.degrees(d)))
    fit.setdefault("n_orientation_clicks", 0)
    if oc:
        corr = _local_correction([i for i, _ in oc], [v for _, v in oc], n, reach_frames)
        R_WH = np.einsum("nij,njk->nik", Rotation.from_rotvec(corr).as_matrix(), R_WH)
        fwd = R_WH[:, :, 0]

    # --- position clicks -------------------------------------------------------
    used = []
    for c in clicks:
        i = int(c["frame"]) - first
        if 0 <= i < n and valid[i] and c.get("x") is not None:
            used.append((i, np.array([float(c["x"]), float(c["y"])])))
    offset = o0.copy()
    fit["n_clicks"] = len(used)
    if used:
        A = np.vstack([R_WH[i][:2, :] for i, _ in used])
        b = np.concatenate([xy - body[i, :2] for i, xy in used])
        offset = np.linalg.solve(A.T @ A + HEAD_CLICK_PRIOR * np.eye(3), A.T @ b + HEAD_CLICK_PRIOR * o0)
        err = lambda o: np.linalg.norm((A @ o - b).reshape(-1, 2), axis=1) * 100
        fit["rms_before_cm"] = float(np.sqrt(np.mean(err(o0) ** 2)))
        fit["rms_after_cm"] = float(np.sqrt(np.mean(err(offset) ** 2)))
        fit["per_click_cm"] = [float(e) for e in err(offset)]
    head = body + np.einsum("nij,j->ni", R_WH, offset)
    if used and reach_frames > 0:
        head[:, :2] += _local_correction([i for i, _ in used], [xy - head[i, :2] for i, xy in used], n, reach_frames)
    elif used:
        for i, xy in used:
            head[i, :2] = xy
    nose = head + NOSE_TICK_M * fwd

    head_top = np.full((n, 2), np.nan); nose_top = np.full((n, 2), np.nan)
    if ctx["cam"] is not None:
        # a single camera's floor-plane view shows the head where its ray
        # through the head meets the floor, not above its true xy
        head_top[valid] = tv.apparent_on_plane(head[valid], ctx["cam"], ctx["B"], ctx["tc"], ctx["floor_z_m"])
        nose_top[valid] = tv.apparent_on_plane(nose[valid], ctx["cam"], ctx["B"], ctx["tc"], ctx["floor_z_m"])
    else:
        # the median view lines the tracked rat up on its own body plane, so it
        # shows up near its true xy: draw the head there
        head_top[valid] = head[valid, :2]; nose_top[valid] = nose[valid, :2]
    q = Rotation.from_matrix(R_WH).as_quat()[:, [3, 0, 1, 2]] if oc else None
    return {"head": head, "nose": nose, "head_top": head_top, "nose_top": nose_top, "offset_H": offset, "fit": fit,
            "R_WH": R_WH, "fwd": fwd, "q": q}


def view_to_true_xy(ctx, frame, xy_view, height=None):
    """Table xy of a point picked on the top view -> its true table xy. Median
    view: the same. Single camera: the pick is where the point appears on the
    floor, so walk back up that camera's ray to the head's height."""
    xy = np.asarray(xy_view, dtype=np.float64)
    if ctx["cam"] is None:
        return xy
    i = int(frame) - int(ctx["frames"][0])
    hz = float(place_head(ctx)["head"][i, 2]) if height is None else height
    C = tv.camera_centre_table(ctx["cam"], ctx["B"], ctx["tc"])
    A = np.array([xy[0], xy[1], ctx["floor_z_m"]])
    return (C + (A - C) * (hz - C[2]) / (A[2] - C[2]))[:2]


def click_to_head_xy(ctx, frame, xy_view, xy_view_to=None):
    """A click on the top view (table xy) -> the head's true table xy at that
    frame, plus the nose heading in degrees if the pointer was dragged to
    xy_view_to (else None). Raises ValueError for a frame without tracking or
    a click too far from the tracked rat."""
    i = int(frame) - int(ctx["frames"][0])
    if not (0 <= i < len(ctx["frames"]) and ctx["valid"][i]):
        raise ValueError(f"frame {frame} has no tracked body position")
    xy = view_to_true_xy(ctx, frame, xy_view)
    d = float(np.linalg.norm(xy - ctx["body"][i, :2]))
    if d > HEAD_CLICK_MAX_M:
        raise ValueError(f"that click is {d * 100:.0f} cm from this rat's tracked body; click the head of the rat "
                         f"selected under 'head' (red = rat 1, blue = rat 2)")
    heading = None
    if xy_view_to is not None:
        to = view_to_true_xy(ctx, frame, xy_view_to)
        v = to - xy
        if np.linalg.norm(v) > 1e-4:
            heading = float(math.degrees(math.atan2(v[1], v[0])))
    return [float(xy[0]), float(xy[1])], heading


def compute(config, cache_dir=None, log=print, progress=None, return_ctx=False):
    """Full analysis -> (series for the frontend, summary[, ctx for place_head]).
    See module docstring."""
    fps = float(config.get("fps") or 30.0)
    dt = 1.0 / fps
    offset = int(config.get("imu_offset_frames") or 0)
    anchor = int(config["anchor_frame"])
    label = int(config.get("mask_label") or 1)
    every = int(config.get("anchor_every_frames") or 0)
    B, tc = tv.build_initial_rig_basis(config["calib_json_path"])
    cams_all = tv.load_camera_extrinsics(config["calib_json_path"])
    R_SH = head_from_sensor_axes(config.get("sensor_forward") or "+x", config.get("sensor_up") or "+z")

    quats = parse_orientation_csv(config["quat_csv_path"])
    n = len(quats)
    k = anchor - offset
    if not 0 <= k < n:
        raise ValueError(f"anchor frame {anchor} maps to IMU row {k}, outside the CSV's 0..{n - 1}")
    frames = np.arange(n) + offset
    log(f"{n} IMU rows -> video frames {frames[0]}..{frames[-1]} (offset {offset})")

    # --- anchor geometry: reconstructed body centroid + marked eyes ---------
    V = obj_vertices(Path(config["anchor_mesh_dir"]) / "mesh.obj")
    centroid_W = calib_to_table(V.mean(axis=0), B, tc)
    eyes_all = json.loads(Path(config["eyes_json_path"]).read_text())
    eyes = eyes_all.get(f"frame_{anchor:06d}") or next(iter(eyes_all.values()))
    eye_mid_W = calib_to_table((np.asarray(eyes["eyeL"]) + np.asarray(eyes["eyeR"])) / 2, B, tc)
    d = eye_mid_W - centroid_W
    yaw_vis = math.atan2(d[1], d[0])
    log(f"anchor {anchor}: mesh body centroid {np.round(centroid_W, 3)} m, eyes {np.round(eye_mid_W, 3)} m, "
        f"heading {math.degrees(yaw_vis):.1f} deg")

    # --- per-camera mask features ---------------------------------------------
    track = config.get("tracking_cameras") or [c for c in sorted(cams_all)
                                                if masks_dir_for(config["masks_root"], c).is_dir()]
    if not track:
        raise ValueError(f"no cam_N/masks_raw folders under {config['masks_root']}")
    log(f"tracking with {len(track)} cameras: {', '.join(track)}")
    feats = []
    for i, c in enumerate(track):
        def prog(dn, tot, i=i):
            if progress:
                progress(i * tot + dn, len(track) * tot, f"masks {c}")
        feats.append(mask_features(masks_dir_for(config["masks_root"], c), label, frames, cache_dir=cache_dir, progress=prog))
    feats = np.stack(feats, axis=1)                                       # (n,C,5)
    C = len(track)
    Ps = np.stack([np.hstack([np.asarray(cams_all[c]["rotation_matrix"]),
                              np.asarray(cams_all[c]["translation_m"]).reshape(3, 1)]) for c in track])
    fx = np.array([cams_all[c]["intrinsics"]["fx"] for c in track])
    obs_c = np.stack([undistort_normalized(feats[:, i, 0:2], cams_all[c]) for i, c in enumerate(track)], axis=1)
    obs_e = np.stack([undistort_normalized(feats[:, i, 2:4], cams_all[c]) for i, c in enumerate(track)], axis=1)

    # --- 3D body position (triangulated) --------------------------------------
    X_C, active, resid = triangulate(obs_c, Ps, fx)
    body = calib_to_table(X_C, B, tc)
    valid = ~np.isnan(body[:, 0])
    n_used = active.sum(1)
    log(f"triangulated on {valid.sum()}/{n} frames; cameras used median {np.median(n_used[valid]):.0f}, "
        f"reprojection residual median {np.nanmedian(resid):.1f} px")
    tri_vs_mesh_cm = float(np.linalg.norm(body[k] - centroid_W) * 100) if valid[k] else None
    if tri_vs_mesh_cm is not None:
        log(f"check: triangulated body at frame {anchor} is {tri_vs_mesh_cm:.1f} cm from the reconstructed mesh centroid")
    body = _smooth_runs(body, valid, int(round(0.2 * fps)) | 1)

    vel = _deriv_runs(body[:, :2], valid, int(round(0.5 * fps)) | 1, dt)
    lin_speed = np.linalg.norm(vel, axis=1)
    known = ~np.isnan(lin_speed)
    walk = known & (lin_speed > 0.10)
    yaw_travel = np.arctan2(vel[:, 1], vel[:, 0])
    bouts = [(s, e) for s, e in _runs(walk) if e - s >= 5]

    # --- 3D body axis (fused) -> horizontal body heading ---------------------
    D_C, qual = fuse_axes(obs_c, obs_e, feats[:, :, 4], active, Ps, X_C)
    D_W = D_C @ np.asarray(B).T                                            # rotation only (a direction)
    horiz = np.linalg.norm(D_W[:, :2], axis=1)
    axis_ok = valid & ~np.isnan(D_W[:, 0]) & (qual > 0.5) & (horiz > 0.5)
    line_ang = np.arctan2(D_W[:, 1], D_W[:, 0])
    log(f"fused 3D body axis usable on {axis_ok.sum()}/{n} frames (quality > 0.5, not near-vertical)")

    # --- orientation: gravity-referenced IMU + heading anchors ----------------
    R_ES = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    roll_sign = config.get("imu_roll_sign") or "as_recorded"
    if roll_sign == "flip":
        R_ES = mirror_roll(R_ES, R_SH)
        log("IMU roll mirrored (imu_roll_sign = flip)")
    C_mount, tilt = mount_correction(R_ES, R_SH, walk, config.get("mount_tilt_mode") or "auto_roll",
                                     config.get("mount_roll_deg"), config.get("mount_pitch_deg"))
    R_SH = R_SH @ C_mount
    if tilt["mode"] != "off":
        log(f"IMU mounting tilt removed ({tilt['mode']}, {tilt['source']}"
            + (f", {tilt['frames_used']} frames" if tilt["frames_used"] else "")
            + f"): roll {tilt['roll_deg']:+.1f} deg (right ear down), pitch {tilt['pitch_deg']:+.1f} deg (nose up)")
    fwd_E = R_ES @ R_SH[:, 0]
    yaw_imu_all = np.unwrap(np.arctan2(fwd_E[:, 1], fwd_E[:, 0]))
    yaw_imu = float(yaw_imu_all[k])

    def psi_from(branch_walk, use_mask):
        ref = branch_by_travel(line_ang, axis_ok, yaw_travel, branch_walk)
        return anchor_psi(ref, use_mask & axis_ok, yaw_imu_all, every)

    anchors = []
    use_anchors = every > 0 and len(bouts) >= 2
    if use_anchors:
        psi_t, centres, vals = psi_from(walk, np.ones(n, bool))
        if psi_t is None:
            use_anchors = False
        else:
            anchors = [{"frame": int(frames[0] + round(c)), "psi_deg": float(np.degrees(v))} for c, v in zip(centres, vals)]
    if not use_anchors:
        psi_t = np.full(n, yaw_vis - yaw_imu)
    c_, s_ = np.cos(psi_t), np.sin(psi_t)
    R_WE = np.zeros((n, 3, 3))
    R_WE[:, 0, 0], R_WE[:, 0, 1], R_WE[:, 1, 0], R_WE[:, 1, 1], R_WE[:, 2, 2] = c_, -s_, s_, c_, 1.0
    R_WH = np.einsum("nij,njk,kl->nil", R_WE, R_ES, R_SH)
    rot = Rotation.from_matrix(R_WH)
    q_WH = rot.as_quat()[:, [3, 0, 1, 2]]
    fwd = R_WH[:, :, 0]
    omega = (rot[:-1].inv() * rot[1:]).as_rotvec() * fps
    omega = np.vstack([omega, omega[-1:]])
    ang_speed = np.degrees(np.linalg.norm(savgol_filter(omega, 9, 2, axis=0, mode="interp"), axis=1))
    corr = np.degrees(psi_t - psi_t[k])
    log(f"IMU heading at anchor {math.degrees(yaw_imu):.1f} deg; "
        + (f"{len(anchors)} heading anchors every {every} frames, correction {corr.min():+.0f}..{corr.max():+.0f} deg"
           if use_anchors else "single anchor from the eyes (no re-alignment)"))

    # --- head point (eyes-based offset, refit from clicked head positions) ----
    body_k = body[k] if valid[k] else centroid_W
    offset0_H = R_WH[k].T @ (eye_mid_W - body_k)
    log(f"body-to-eyes offset in the head frame {np.round(offset0_H, 3)} m (from the marked eyes)")
    view = config.get("view_mode") or "median"
    ctx = {"frames": frames, "body": body, "valid": valid, "R_WH": R_WH, "fwd": fwd, "offset0_H": offset0_H,
           "q_vision": q_WH.tolist(),
           "view": view, "cam": cams_all.get(view), "B": B, "tc": tc, "floor_z_m": float(config.get("floor_z_m") or 0.0)}
    hp = place_head(ctx, config.get("head_clicks") or [], int(config.get("head_click_reach_frames") or 0))
    head, head_top, nose_top = hp["head"], hp["head_top"], hp["nose_top"]
    if hp["q"] is not None:        # clicked headings re-aligned the gyroscope there
        q_WH, fwd = hp["q"], hp["fwd"]
        log(f"gyroscope re-aligned at {hp['fit']['n_heading_clicks'] + hp['fit']['n_orientation_clicks']} clicked frames")
    if hp["fit"]["n_clicks"]:
        log(f"head position refit from {hp['fit']['n_clicks']} clicked frames: offset {np.round(hp['offset_H'], 3)} m, "
            f"click error {hp['fit']['rms_before_cm']:.1f} -> {hp['fit']['rms_after_cm']:.1f} cm (rigid offset)")

    # --- behavioural state ---------------------------------------------------
    vel_state = _deriv_runs(body[:, :2], valid, int(round(1.0 * fps)) | 1, dt)
    speed_state = np.linalg.norm(vel_state, axis=1)
    known_state = ~np.isnan(speed_state)
    moving = known_state & (speed_state >= STATE_THRESHOLDS["rest_lin_speed_mps"])
    still = ~known_state | (speed_state < STATE_THRESHOLDS["rest_lin_speed_mps"])
    state = np.where(moving, "locomotion",
                     np.where(still & (ang_speed < STATE_THRESHOLDS["rest_ang_speed_dps"]), "rest", "head_movement"))

    # --- validation ---------------------------------------------------------
    # R = mean resultant length of (travel direction - head heading) during
    # walking (1 aligned, ~0 unrelated). Anchors are scored held-out: the test
    # bouts' vision data is excluded from BOTH the nose/tail branch choice and
    # the anchor means, so their heading comes from neighbouring anchors and
    # the IMU alone; even/odd bouts swap roles and the two scores are averaged.
    def align_R(yaw, mask):
        return float(abs(np.mean(np.exp(1j * (yaw_travel[mask] - yaw[mask]))))) if mask.sum() > 20 else None
    held_out = None
    if use_anchors and len(bouts) >= 4:
        scores = []
        for par in (0, 1):
            train = np.zeros(n, bool); test = np.zeros(n, bool)
            for i, (s, e) in enumerate(bouts):
                (train if i % 2 == par else test)[s:e] = True
            p_ho, _, _ = psi_from(train & walk, ~test)
            if p_ho is not None:
                r = align_R(p_ho + yaw_imu_all, test & walk)
                if r is not None:
                    scores.append(r)
        held_out = float(np.mean(scores)) if scores else None
    yaw_head = psi_t + yaw_imu_all
    anchor_resid = math.degrees(math.remainder(float(yaw_head[k]) - yaw_vis, 2 * math.pi))
    validation = {
        "walking_frames": int(walk.sum()), "walking_bouts": len(bouts),
        "heading_vs_travel_R_single_anchor": align_R(yaw_imu_all + (yaw_vis - yaw_imu), walk),
        "heading_vs_travel_R_anchors_heldout": held_out,
        "anchor_heading_minus_eyes_deg": anchor_resid,
        "triangulated_vs_mesh_centroid_cm": tri_vs_mesh_cm,
        "triangulation_residual_px_median": float(np.nanmedian(resid)),
        "cameras_used_median": float(np.median(n_used[valid])) if valid.any() else 0.0,
    }
    log(f"heading vs walking direction R: single anchor {validation['heading_vs_travel_R_single_anchor']:.2f}"
        + (f", anchors every {every} frames (held-out) {held_out:.2f}" if held_out is not None else "")
        + f"; at frame {anchor} the heading is {anchor_resid:+.0f} deg off the eyes")

    def clean(a):
        return [None if np.isnan(a[i, 0]) else [float(v) for v in a[i]] for i in range(len(a))]

    series = {
        "frame": frames.tolist(), "fps": fps, "view_mode": view,
        "q": q_WH.tolist(), "fwd": fwd.tolist(),
        "body": clean(body), "head": clean(head), "head_top": clean(head_top), "nose_top": clean(nose_top),
        "has_pos": valid.tolist(), "state": state.tolist(),
    }
    summary = {
        "anchor_frame": anchor, "imu_offset_frames": offset, "mask_label": label,
        "tracking_cameras": track, "view_mode": view,
        "anchor_every_frames": every if use_anchors else 0, "heading_anchors": anchors,
        "heading_vision_deg": math.degrees(yaw_vis), "heading_imu_deg": math.degrees(yaw_imu),
        "heading_correction_range_deg": [float(corr.min()), float(corr.max())],
        "validation": validation,
        "offset_head_frame_m": hp["offset_H"].tolist(), "offset_head_frame_eyes_m": offset0_H.tolist(),
        "head_fit": hp["fit"], "mount_tilt": tilt, "imu_roll_sign": roll_sign, "frames_with_position": int(valid.sum()), "n_frames": int(n),
        "body_height_m_median": float(np.nanmedian(body[:, 2])) if valid.any() else None,
        "floor_z_m": float(config.get("floor_z_m") or 0.0),
        "mean_ang_speed_dps": float(ang_speed.mean()),
        "mean_lin_speed_mps": float(np.nanmean(lin_speed)) if known.any() else None,
        "state_fractions": {st: float(np.mean(state == st)) for st in ("rest", "head_movement", "locomotion")},
    }
    return (series, summary, ctx) if return_ctx else (series, summary)
