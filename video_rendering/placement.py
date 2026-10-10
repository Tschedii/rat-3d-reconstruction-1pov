"""Where the IMU skulls and rat 1's eye camera go (numpy only, so Blender's Python can import it too).

Position comes from the head the cameras tracked (GUI 5.2, triangulated from the six masks, smoothed lightly
over time), orientation from each rat's IMU (GUI 5.1). GUI 5.1's own head point is body + an IMU-rotated
offset, which swings below the arena floor on nose-down poses (down to -5 cm), putting the skull under the
table; the tracked head never goes below the floor.

  skull (the headstage scan): the tracked head point sits on the real headstage in the raw camera views (checked
  in all six on frames 280, 314, 450 with overlay_check.py), so the scan goes just above and behind it,
  CAP_OFFSET in the head frame (x nose, y left, z up), kept clear of the floor
  eye (rat 1's first-person camera): just below the headstage, EYE_OFFSET
"""
import csv
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DRIVE = Path(__file__).resolve().parents[1]  # repository root
HEADS = DRIVE / "gui_5.2_head_tracking/exports/masks_20261008_224517_rats12_000000-001400"
S = json.load(open(HERE / "data/scene.json"))
POSES = json.load(open(HERE / "data/skull_poses.json"))
FLOOR_Z = S["floor_z"]
CAP_OFFSET = np.array([-0.010, 0.0, 0.020])      # tracked head -> centre of the headstage scan
EYE_OFFSET = np.array([0.000, 0.0, -0.012])      # tracked head -> eyes, under the front of the headstage
CAP_HALF_HEIGHT = 0.025
SMOOTH_SIGMA = 1.5                               # frames


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _smooth_runs(frames, P, sigma):
    """Gaussian smoothing along time within runs of consecutive frames."""
    out = P.copy()
    k = np.arange(-4, 5); w = np.exp(-0.5 * (k / sigma) ** 2)
    idx = {f: i for i, f in enumerate(frames)}
    for i, f in enumerate(frames):
        js = [(idx[f + d], wt) for d, wt in zip(k, w) if (f + d) in idx]
        ws = np.array([wt for _, wt in js]); out[i] = (P[[j for j, _ in js]] * ws[:, None]).sum(0) / ws.sum()
    return out


_TRACK = {}


def _track(r):
    if r not in _TRACK:
        fr, P = [], []
        for row in csv.DictReader(open(HEADS / f"rat{r}" / "head_pose_world.csv")):
            if row["head_x_m"]:
                fr.append(int(row["frame"])); P.append([float(row[k]) for k in ("head_x_m", "head_y_m", "head_z_m")])
        P = _smooth_runs(fr, np.array(P), SMOOTH_SIGMA)
        _TRACK[r] = {f: p for f, p in zip(fr, P)}
    return _TRACK[r]


def tracked_head(r, f):
    t = _track(r)
    if f in t:
        return t[f]
    near = [g for g in (f - 1, f + 1, f - 2, f + 2) if g in t]
    return t[near[0]] if near else None


def head_rotation(r, f):
    p = POSES[str(r)].get(str(f))
    return _quat_matrix(p["q"]) if p else None


def skull_pose(r, f):
    """(centre, R) of the headstage scan in the table frame, or None."""
    h, R = tracked_head(r, f), head_rotation(r, f)
    if h is None or R is None:
        return None
    c = h + R @ CAP_OFFSET
    c[2] = max(c[2], FLOOR_Z + CAP_HALF_HEIGHT)
    return c, R


def skull_matrix(r, f):
    sp = skull_pose(r, f)
    if sp is None:
        return None
    M = np.eye(4); M[:3, :3] = sp[1]; M[:3, 3] = sp[0]
    return M


def eye_pose(r, f):
    h, R = tracked_head(r, f), head_rotation(r, f)
    if h is None or R is None:
        return None
    e = h + R @ EYE_OFFSET
    e[2] = max(e[2], FLOOR_Z + 0.015)
    return e, R


def cap_points(n=20000):
    """n vertices of the headstage scan, centred, mm -> m, in the head frame (as the renderer draws it)."""
    p = Path(S["cap"]["stl"])
    with open(p, "rb") as fh:
        fh.read(80); count = struct.unpack("<I", fh.read(4))[0]
        data = np.frombuffer(fh.read(50 * count), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    V = data["v"].reshape(-1, 3).astype(np.float64)
    c = (V.min(0) + V.max(0)) / 2
    sel = np.linspace(0, len(V) - 1, n).astype(int)
    return ((V[sel] - c) / 1000.0) @ np.asarray(S["cap"]["fixed_rotation"]).T


# --- rat 1's first-person camera, stabilised like a gimbal ------------------------------------------------------
# The raw view (tracked head + IMU rotation per frame) turns a median 4.9 deg per frame and rolls between -40 and
# +63 deg. A camera gimbal follows the intended heading and pitch but removes the shake and keeps the horizon
# level: here yaw and pitch are low-passed (zero-phase Gaussian, GIMBAL_SIGMA frames), roll is low-passed and
# reduced to GIMBAL_ROLL of itself, the eye position low-passed with GIMBAL_POS_SIGMA.
GIMBAL_SIGMA = 4.5            # frames (~0.15 s)
GIMBAL_ROLL = 0.25
GIMBAL_POS_SIGMA = 3.0
GIMBAL_FRAMES = (740, 1000)   # smoothed over these (the first-person view shows 750..1000)
_GIMBAL = {}


def _gauss(x, sigma):
    """Zero-phase Gaussian low-pass along axis 0, weights renormalised at the ends."""
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1); w = np.exp(-0.5 * (k / sigma) ** 2)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        j = i + k; ok = (j >= 0) & (j < len(x))
        out[i] = (x[j[ok]] * w[ok].reshape(-1, *([1] * (x.ndim - 1)))).sum(0) / w[ok].sum()
    return out


def _rot_from(yaw, pitch, roll):
    n = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    lev = np.cross([0.0, 0.0, 1.0], n); lev /= np.linalg.norm(lev)          # level "left"
    lup = np.cross(n, lev)                                                 # level up
    up = np.cos(roll) * lup - np.sin(roll) * lev
    return np.column_stack([n, np.cross(up, n), up])                       # nose, left, up


def gimbal_eye(r, f):
    """(eye, R) of rat r's stabilised first-person camera at frame f (table frame), or None outside the range."""
    if r not in _GIMBAL:
        fr = [g for g in range(GIMBAL_FRAMES[0], GIMBAL_FRAMES[1] + 1)
              if tracked_head(r, g) is not None and head_rotation(r, g) is not None]
        R = np.array([head_rotation(r, g) for g in fr]); H = np.array([tracked_head(r, g) for g in fr])
        n, up = R[:, :, 0], R[:, :, 2]
        yaw = np.unwrap(np.arctan2(n[:, 1], n[:, 0])); pitch = np.arcsin(np.clip(n[:, 2], -1, 1))
        lev = np.cross([0.0, 0.0, 1.0], n); lev /= np.linalg.norm(lev, axis=1)[:, None]
        lup = np.cross(n, lev)
        roll = np.arctan2(np.einsum("ij,ij->i", up, -lev), np.einsum("ij,ij->i", up, lup))
        yaw, pitch = _gauss(yaw, GIMBAL_SIGMA), _gauss(pitch, GIMBAL_SIGMA)
        roll = GIMBAL_ROLL * _gauss(roll, GIMBAL_SIGMA)
        H = _gauss(H, GIMBAL_POS_SIGMA)
        out = {}
        for i, g in enumerate(fr):
            Rs = _rot_from(yaw[i], pitch[i], roll[i])
            e = H[i] + Rs @ EYE_OFFSET
            e[2] = max(e[2], FLOOR_Z + 0.015)
            out[g] = (e, Rs)
        _GIMBAL[r] = out
    t = _GIMBAL[r]
    if f in t:
        return t[f]
    near = [g for g in (f - 1, f + 1, f - 2, f + 2) if g in t]
    return t[near[0]] if near else None
