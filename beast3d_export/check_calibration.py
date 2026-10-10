"""Check that a rig calibration fits a recorded session, using only the SAM masks.

Two tests per sampled frame:
1. Centroid reprojection: triangulate each rat's mask centroid from all cameras and
   reproject it. Silhouette centroids are not exactly the same 3D point in every view,
   so expect errors of tens of pixels when the calibration fits and hundreds when not.
2. Leave-one-out visual hull: carve a voxel hull from five cameras, project it into the
   sixth and compare with that camera's mask. The hull contains the true shape, so with
   a good calibration recall (mask pixels covered) should be close to 1.

Usage:
    python check_calibration.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 --calib ../data/calibration/output/2026_09_21/2026_09_21.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rig import CAMS, carve_hull, list_frames, load_mask, load_rig, project, render_points, triangulate, undistort_pixels

MIN_AREA = 300  # px; ignore a rat in a camera if its mask is smaller than this


def centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_AREA:
        return None
    return np.array([xs.mean(), ys.mean()])


def check_frame(src: Path, rig: dict, frame: int, voxel: float, pad: float) -> tuple[list[dict], dict]:
    union = {cam: load_mask(src, cam, frame, (1, 2)) for cam in CAMS}
    rows = []

    # 1. per-rat centroid triangulation
    rat_points = []
    for rat in (1, 2):
        cents = {}
        for cam in CAMS:
            c = centroid(load_mask(src, cam, frame, (rat,)))
            if c is not None:
                cents[cam] = c
        if len(cents) < 3:
            continue
        und = {cam: undistort_pixels(c[None], rig[cam])[0] for cam, c in cents.items()}
        X = triangulate(und, rig)
        rat_points.append(X)
        for cam, c in cents.items():
            uv, _ = project(X[None], rig[cam])
            rows.append({'frame': frame, 'cam': cam, 'test': f'centroid_rat{rat}',
                         'value': float(np.linalg.norm(uv[0] - c))})

    if not rat_points:
        return rows, {}
    lo = np.min(rat_points, 0) - pad
    hi = np.max(rat_points, 0) + pad

    # 2. leave-one-out hull
    renders = {}
    for held in CAMS:
        others = {cam: m for cam, m in union.items() if cam != held}
        hull = carve_hull(others, rig, lo, hi, voxel)
        proj = render_points(hull, rig[held], union[held].shape, voxel)
        mask = union[held]
        inter = (proj & mask).sum()
        rows.append({'frame': frame, 'cam': held, 'test': 'loo_recall', 'value': inter / max(mask.sum(), 1)})
        rows.append({'frame': frame, 'cam': held, 'test': 'loo_precision', 'value': inter / max(proj.sum(), 1)})
        rows.append({'frame': frame, 'cam': held, 'test': 'loo_iou', 'value': inter / max((proj | mask).sum(), 1)})
        renders[held] = (mask, proj)
    full = carve_hull(union, rig, lo, hi, voxel)
    rows.append({'frame': frame, 'cam': 'all', 'test': 'hull_volume_cm3', 'value': len(full) * (voxel * 100) ** 3})
    return rows, renders


def plot_frame(frame: int, renders: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, cam in zip(axes.ravel(), CAMS):
        mask, proj = renders[cam]
        ys, xs = np.nonzero(mask | proj)
        if len(xs):
            m = 40
            y0, y1 = max(ys.min() - m, 0), ys.max() + m
            x0, x1 = max(xs.min() - m, 0), xs.max() + m
        else:
            y0, y1, x0, x1 = 0, mask.shape[0], 0, mask.shape[1]
        img = np.zeros(mask.shape + (3,))
        img[mask] = (0.75, 0.75, 0.75)
        img[proj & ~mask] = (0.85, 0.25, 0.2)
        img[proj & mask] = (0.2, 0.55, 0.85)
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(cam)
        ax.axis('off')
    fig.suptitle(f'frame {frame}: blue = hull from the other 5 cameras covers the mask, '
                 f'grey = mask missed, red = hull outside mask')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'calib_check')
    ap.add_argument('--n-frames', type=int, default=24)
    ap.add_argument('--voxel', type=float, default=0.005, help='voxel size in metres')
    ap.add_argument('--pad', type=float, default=0.25, help='hull box margin around the rats, metres')
    ap.add_argument('--n-plots', type=int, default=3)
    args = ap.parse_args()

    src = args.src.expanduser()
    rig = load_rig(args.calib.expanduser())
    args.out.mkdir(parents=True, exist_ok=True)

    common = sorted(set.intersection(*(list_frames(src, cam, 'masks_raw') for cam in CAMS)))
    frames = [common[i] for i in np.linspace(0, len(common) - 1, args.n_frames).round().astype(int)]
    plot_frames = set(frames[:: max(1, len(frames) // args.n_plots)][: args.n_plots])

    rows = []
    for frame in frames:
        frame_rows, renders = check_frame(src, rig, frame, args.voxel, args.pad)
        rows += frame_rows
        if frame in plot_frames and renders:
            plot_frame(frame, renders, args.out / f'loo_hull_frame_{frame:06d}.png')
        rec = [r['value'] for r in frame_rows if r['test'] == 'loo_recall']
        print(f'frame {frame:6d}: LOO recall per cam {np.round(rec, 2).tolist()}')

    with open(args.out / 'calib_check.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'cam', 'test', 'value'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n{len(frames)} frames, medians per camera')
    print(f'{"cam":6s} {"recall":>7s} {"precis":>7s} {"IoU":>6s} {"centroid px":>12s}')
    for cam in CAMS:
        def med(test_prefix: str) -> float:
            vals = [r['value'] for r in rows if r['cam'] == cam and r['test'].startswith(test_prefix)]
            return float(np.median(vals)) if vals else float('nan')
        print(f'{cam:6s} {med("loo_recall"):7.3f} {med("loo_precision"):7.3f} {med("loo_iou"):6.3f} {med("centroid"):12.1f}')
    vol = [r['value'] for r in rows if r['test'] == 'hull_volume_cm3']
    print(f'6-camera hull volume (both rats): median {np.median(vol):.0f} cm3')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
