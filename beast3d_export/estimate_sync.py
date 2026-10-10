"""Estimate per-camera frame offsets over a session from the SAM masks.

For each time window and each camera c, each rat's mask centroid is triangulated from the
other cameras at frame t and projected into c. That prediction is compared with c's own
centroid at t + d for d in [-max_lag, max_lag]. Silhouette centroids are not the same 3D
point in every view, but that bias is nearly constant within a window, so the d with the
smallest median error is c's lag. Windows where the rats barely move give flat curves and
are flagged.

Two passes: the second triangulates from lag-corrected reference cameras. Offsets are
reported relative to cam_4 (the calibration's reference camera): a value of +2 for cam_1
means cam_1's frame t + 2 shows the moment that cam_4's frame t shows.

Usage:
    python estimate_sync.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 \\
        --calib ../data/calibration/output/2026_09_21/2026_09_21.json
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from rig import CAMS, frame_path, load_rig, project, triangulate, undistort_pixels

MIN_AREA = 300
MAX_FG_FRACTION = 0.15
REF = 'cam_4'


def centroids(task: tuple[Path, str, int]) -> tuple[str, int, np.ndarray]:
    src, cam, frame = task
    out = np.full((2, 2), np.nan)
    try:
        labels = np.array(Image.open(frame_path(src, cam, 'masks_raw', frame)))
    except FileNotFoundError:
        return cam, frame, out
    if (labels > 0).mean() > MAX_FG_FRACTION:
        return cam, frame, out
    for i, rat in enumerate((1, 2)):
        ys, xs = np.nonzero(labels == rat)
        if len(xs) >= MIN_AREA:
            out[i] = xs.mean(), ys.mean()
    return cam, frame, out


def lag_curves(cent: dict, und: dict, rig: dict, ts: np.ndarray, lags: dict[str, int],
               D: np.ndarray) -> dict[str, np.ndarray]:
    """Median reprojection error per candidate offset, per held-out camera."""
    curves = {}
    for held in CAMS:
        pred, ok = [], []
        for t in ts:
            for r in range(2):
                obs = {}
                for c in CAMS:
                    if c == held:
                        continue
                    p = und[c].get(t + lags[c])
                    if p is not None and np.isfinite(p[r]).all():
                        obs[c] = p[r]
                if len(obs) < 3:
                    continue
                X = triangulate(obs, rig)
                pred.append((t, r, project(X[None], rig[held])[0][0]))
        errs = np.full(len(D), np.nan)
        for k, d in enumerate(D):
            e = []
            for t, r, uv in pred:
                q = cent[held].get(t + lags[held] + d)
                if q is not None and np.isfinite(q[r]).all():
                    e.append(np.linalg.norm(uv - q[r]))
            if len(e) >= 20:
                errs[k] = np.median(e)
        curves[held] = errs
    return curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'sync_offsets.csv')
    ap.add_argument('--window', type=int, default=300, help='frames per window')
    ap.add_argument('--every', type=int, default=1000, help='window spacing in frames')
    ap.add_argument('--first', type=int, default=300)
    ap.add_argument('--last', type=int, default=17700)
    ap.add_argument('--max-lag', type=int, default=10)
    ap.add_argument('--step', type=int, default=2, help='use every Nth frame inside a window')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    src = args.src.expanduser()
    rig = load_rig(args.calib.expanduser())
    D = np.arange(-args.max_lag, args.max_lag + 1)
    pad = 3 * args.max_lag
    centres = list(range(args.first + args.window // 2, args.last - args.window // 2 + 1, args.every))

    needed = set()
    for c0 in centres:
        needed.update(range(c0 - args.window // 2 - pad, c0 + args.window // 2 + pad + 1))
    tasks = [(src, cam, f) for cam in CAMS for f in sorted(needed)]
    print(f'{len(centres)} windows, reading {len(tasks)} masks')
    cent = {cam: {} for cam in CAMS}
    with Pool(args.workers) as pool:
        for cam, f, c in pool.imap_unordered(centroids, tasks, chunksize=64):
            cent[cam][f] = c
    und = {cam: {} for cam in CAMS}
    for cam in CAMS:
        for f, c in cent[cam].items():
            u = np.full((2, 2), np.nan)
            for r in range(2):
                if np.isfinite(c[r]).all():
                    u[r] = undistort_pixels(c[r][None], rig[cam])[0]
            und[cam][f] = u

    rows = []
    print(f'{"window":>13s} ' + ' '.join(f'{c:>6s}' for c in CAMS) + '   (offset vs cam_4; * = flat curve)')
    for c0 in centres:
        ts = np.arange(c0 - args.window // 2, c0 + args.window // 2, args.step)
        lags = {c: 0 for c in CAMS}
        for _ in range(2):
            curves = lag_curves(cent, und, rig, ts, lags, D)
            new = {}
            for c in CAMS:
                e = curves[c]
                new[c] = lags[c] + (int(D[np.nanargmin(e)]) if np.isfinite(e).any() else 0)
            lags = {c: new[c] - new[REF] for c in CAMS}
        flat = {}
        for c in CAMS:
            e = curves[c]
            if not np.isfinite(e).any():
                flat[c] = True
                continue
            far = np.nanmin(np.r_[e[:max(1, args.max_lag - 3)], e[args.max_lag + 4:]])
            flat[c] = far - np.nanmin(e) < 3.0  # px; under 3 px gain, the window carries no timing signal
        rows.append({'window_start': int(ts[0]), 'window_end': int(ts[-1]),
                     **{c: lags[c] for c in CAMS}, **{f'{c}_flat': flat[c] for c in CAMS}})
        print(f'{ts[0]:6d}-{ts[-1]:6d} ' + ' '.join(f'{lags[c]:+5d}{"*" if flat[c] else " "}' for c in CAMS), flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
