"""Camera-rig helpers shared by the BEAST3D conversion scripts.

Calibration: GUI 1 JSON (OpenCV convention, p_cam = R @ p_world + t, metres).
Session folder: <src>/cam_N/{all_objects,object_1,object_2,masks_raw}/frame_XXXXXX.png,
where masks_raw are palette PNGs with 0 = background, 1 = object_1, 2 = object_2.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CAMS = [f'cam_{i}' for i in range(1, 7)]


def load_rig(calib_path: str | Path) -> dict[str, dict]:
    """Return {'cam_N': {K, dist, R, t, size}} from a GUI 1 calibration JSON."""
    calib = json.loads(Path(calib_path).read_text())
    rig = {}
    for cam in calib['cameras']:
        idx = int(cam['name'].split('_')[-1])
        intr = cam['intrinsics']
        K = np.array([
            [intr['fx'], 0.0, intr['cx']],
            [0.0, intr['fy'], intr['cy']],
            [0.0, 0.0, 1.0],
        ])
        dist = np.array([intr['k1'], intr['k2'], intr['p1'], intr['p2'], intr['k3']], float)
        rig[f'cam_{idx}'] = {
            'K': K,
            'dist': dist,
            'R': np.array(cam['rotation_matrix'], float),
            't': np.array(cam['translation_m'], float),
            'size': tuple(intr['image_size_wh']),  # (width, height)
        }
    return rig


def frame_path(src: Path, cam: str, kind: str, frame: int) -> Path:
    return src / cam / kind / f'frame_{frame:06d}.png'


def list_frames(src: Path, cam: str, kind: str) -> set[int]:
    return {int(p.stem.split('_')[1]) for p in (src / cam / kind).glob('frame_*.png')}


def load_mask(src: Path, cam: str, frame: int, ids: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Boolean foreground mask from masks_raw (palette indices, not colours)."""
    labels = np.array(Image.open(frame_path(src, cam, 'masks_raw', frame)))
    return np.isin(labels, ids)


def project(points: np.ndarray, cam: dict, distort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N, 3) to pixels; returns (uv (N, 2), depth (N,))."""
    p_cam = points @ cam['R'].T + cam['t']
    depth = p_cam[:, 2]
    rvec, _ = cv2.Rodrigues(cam['R'])
    dist = cam['dist'] if distort else np.zeros(5)
    uv, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, cam['t'], cam['K'], dist)
    return uv.reshape(-1, 2), depth


def undistort_pixels(uv: np.ndarray, cam: dict) -> np.ndarray:
    """Distorted pixel coords (N, 2) to undistorted pixel coords with the same K."""
    pts = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), cam['K'], cam['dist'], P=cam['K'])
    return pts.reshape(-1, 2)


def triangulate(uv_by_cam: dict[str, np.ndarray], rig: dict[str, dict]) -> np.ndarray:
    """Linear (DLT) triangulation of one point from undistorted pixels in >= 2 cameras."""
    rows = []
    for cam_name, uv in uv_by_cam.items():
        cam = rig[cam_name]
        P = cam['K'] @ np.hstack([cam['R'], cam['t'][:, None]])
        u, v = uv
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def carve_hull(
    masks: dict[str, np.ndarray],
    rig: dict[str, dict],
    lo: np.ndarray,
    hi: np.ndarray,
    voxel: float,
    distort: bool = True,
) -> np.ndarray:
    """Visual hull: voxel centres in [lo, hi] that fall inside every given mask."""
    axes = [np.arange(a, b, voxel) for a, b in zip(lo, hi)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    for cam_name, mask in masks.items():
        if len(pts) == 0:
            break
        uv, depth = project(pts, rig[cam_name], distort=distort)
        h, w = mask.shape
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        keep = np.zeros(len(pts), bool)
        keep[ok] = mask[v[ok], u[ok]]
        pts = pts[keep]
    return pts


def render_points(pts: np.ndarray, cam: dict, shape: tuple[int, int], voxel: float, distort: bool = True) -> np.ndarray:
    """Splat 3D points into a binary image, each as a square of its projected voxel size."""
    out = np.zeros(shape, np.uint8)
    if len(pts) == 0:
        return out.astype(bool)
    uv, depth = project(pts, cam, distort=distort)
    h, w = shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out[v[ok], u[ok]] = 1
    footprint = cam['K'][0, 0] * voxel / np.median(depth[ok]) if ok.any() else 1.0
    k = max(1, int(np.ceil(footprint)))
    out = cv2.dilate(out, np.ones((k, k), np.uint8))
    return out.astype(bool)
