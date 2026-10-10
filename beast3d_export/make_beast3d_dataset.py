"""Convert a segmented six-camera rat session into a BEAST3D training dataset.

Writes the layout that BEAST3D's MultiViewDataset reads (the output of `beast extract_3d`):

    <out>/info.json
    <out>/<session>/selected_frames.csv      one img########.png name per line, no header
    <out>/<session>/cam_N/img########.png     RGB crop, out_size x out_size
    <out>/<session>/cam_N/img########.npy     {intrinsics, extrinsics (w2c 4x4, metres),
                                               distortions, width, height, + provenance}
    <out>/<session>/cam_N/mask########.png    foreground, 0/255

Three things differ from feeding full frames to BEAST3D, each because of how its loader works:
- The loader squashes every image to 256x256 and ignores the stored bbox, so each view is
  cropped to a square around the rats here and the intrinsics are shifted and scaled to match.
- The loader ignores lens distortion, so crops are resampled from the undistorted image and
  the stored distortions are zero.
- The loader thresholds grey(mask) > 0.5; the palette masks_raw would map rat 1 to 125/255
  and drop it, so masks are written as 0/255.

Frames are chosen as in BEAST (k-means for visual diversity), here on downsampled masks of
all six cameras, from frames where every chosen rat is visible in every camera.

The Day 7 cameras drift apart by several frames during the session, so either restrict to the
synced start (--frames) or pass the offsets from estimate_sync.py (--sync-csv). With offsets,
file names use the cam_4 frame number and each camera's .npy records the source frame used.

Usage:
    python make_beast3d_dataset.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 \\
        --calib ../data/calibration/output/2026_09_21/2026_09_21.json --out /path/to/beast3d_day7
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rig import CAMS, carve_hull, frame_path, list_frames, load_rig, project, render_points, undistort_pixels

MIN_AREA = 300          # px; a rat counts as visible in a camera above this mask area
MAX_FG_FRACTION = 0.15  # larger foreground = failed mask (Day 7 frame 1, cam_6 segments the floor)
DESC_HW = (20, 32)      # per-camera mask descriptor size for k-means (keeps the 1200:1920 aspect)


def animal_ids(animals: str) -> tuple[int, ...]:
    return (1, 2) if animals == 'both' else (int(animals),)


def image_kind(animals: str) -> str:
    return 'all_objects' if animals == 'both' else f'object_{animals}'


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------

def describe_frame(item: tuple[int, dict[str, int]], src: Path, ids: tuple[int, ...]) -> tuple[int, np.ndarray | None]:
    """Downsampled masks of all cameras, or None if a chosen rat is missing in any camera
    or a camera's mask is implausibly large. item = (frame, {cam: source frame})."""
    frame, source = item
    parts = []
    for cam in CAMS:
        labels = np.array(Image.open(frame_path(src, cam, 'masks_raw', source[cam])))
        for i in ids:
            if (labels == i).sum() < MIN_AREA:
                return frame, None
        if (labels > 0).mean() > MAX_FG_FRACTION:
            return frame, None
        fg = np.isin(labels, ids).astype(np.float32)
        parts.append(cv2.resize(fg, DESC_HW[::-1], interpolation=cv2.INTER_AREA).ravel())
    return frame, np.concatenate(parts)


def select_frames(src: Path, candidates: list[tuple[int, dict[str, int]]], ids: tuple[int, ...], n: int,
                  method: str, workers: int, seed: int) -> tuple[list[int], int]:
    with Pool(workers) as pool:
        results = pool.map(partial(describe_frame, src=src, ids=ids), candidates, chunksize=32)
    valid = [(f, d) for f, d in results if d is not None]
    frames = np.array([f for f, _ in valid])
    if len(frames) <= n:
        return frames.tolist(), len(frames)
    if method == 'uniform':
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return frames[idx].tolist(), len(frames)
    X = np.stack([d for _, d in valid])
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    emb = X @ vt[:32].T
    labels, centers = kmeans(emb, n, seed)
    chosen = set()
    for k in range(n):
        members = np.nonzero(labels == k)[0]
        if len(members):
            d = np.linalg.norm(emb[members] - centers[k], axis=1)
            chosen.add(int(frames[members[d.argmin()]]))
    return sorted(chosen), len(frames)


def kmeans(X: np.ndarray, k: int, seed: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ seeding (numpy only)."""
    rng = np.random.default_rng(seed)
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(len(X))]
    d2 = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        C[j] = X[rng.choice(len(X), p=p)]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
    sq = (X ** 2).sum(1)[:, None]
    labels = np.full(len(X), -1)
    for _ in range(iters):
        new = (sq - 2 * X @ C.T + (C ** 2).sum(1)[None]).argmin(1)
        if (new == labels).all():
            break
        labels = new
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)
        filled = counts > 0
        C[filled] = sums[filled] / counts[filled, None]
    return labels, C


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def distort_pixels(x: np.ndarray, y: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted pixel grid -> distorted (source image) pixel grid, OpenCV model."""
    K, (k1, k2, p1, p2, k3) = cam['K'], cam['dist']
    xn = (x - K[0, 2]) / K[0, 0]
    yn = (y - K[1, 2]) / K[1, 1]
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]


def crop_box(fg: np.ndarray, cam: dict, margin: float, min_side: int) -> tuple[int, int, int]:
    """Square (x0, y0, side) in undistorted pixels around the foreground."""
    ys, xs = np.nonzero(fg)
    step = max(1, len(xs) // 5000)
    und = undistort_pixels(np.stack([xs[::step], ys[::step]], 1).astype(np.float64), cam)
    (xmin, ymin), (xmax, ymax) = und.min(0), und.max(0)
    side = max((xmax - xmin), (ymax - ymin)) * (1 + 2 * margin)
    side = int(np.ceil(max(side, min_side)))
    x0 = int(round((xmin + xmax) / 2 - side / 2))
    y0 = int(round((ymin + ymax) / 2 - side / 2))
    return x0, y0, side


def export_view(task: dict) -> dict:
    cam, frame, src, out_dir = task['cam'], task['frame'], task['src'], task['out_dir']
    rigcam, size, ids = task['rigcam'], task['out_size'], task['ids']

    source = task['source_frame']  # this camera's frame showing the same moment (sync-corrected)
    img = np.array(Image.open(frame_path(src, cam, task['image_kind'], source)).convert('RGB'))
    labels = np.array(Image.open(frame_path(src, cam, 'masks_raw', source)))
    fg = np.isin(labels, ids)

    x0, y0, side = crop_box(fg, rigcam, task['margin'], task['min_side'])
    gx, gy = np.meshgrid(np.arange(side, dtype=np.float64) + x0, np.arange(side, dtype=np.float64) + y0)
    mx, my = distort_pixels(gx, gy, rigcam)
    mx, my = mx.astype(np.float32), my.astype(np.float32)
    border = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    crop = cv2.remap(img, mx, my, cv2.INTER_LINEAR, **border)
    crop_fg = cv2.remap(fg.astype(np.float32), mx, my, cv2.INTER_LINEAR, **border)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop_fg = cv2.resize(crop_fg, (size, size), interpolation=cv2.INTER_AREA) > 0.5

    s = size / side
    K = rigcam['K']
    K_out = np.array([
        [K[0, 0] * s, 0.0, (K[0, 2] - x0 + 0.5) * s - 0.5],
        [0.0, K[1, 1] * s, (K[1, 2] - y0 + 0.5) * s - 0.5],
        [0.0, 0.0, 1.0],
    ])
    w2c = np.eye(4)
    w2c[:3, :3] = rigcam['R']
    w2c[:3, 3] = rigcam['t']

    view_dir = out_dir / cam
    stem = f'{frame:08d}'
    Image.fromarray(crop).save(view_dir / f'img{stem}.png')
    Image.fromarray((crop_fg * 255).astype(np.uint8), mode='L').save(view_dir / f'mask{stem}.png')
    np.save(view_dir / f'img{stem}.npy', {
        'intrinsics': K_out,
        'extrinsics': w2c,
        'distortions': np.zeros(5),
        'width': size,
        'height': size,
        'source_frame': source,
        'crop_xys_undistorted': (x0, y0, side),
    })
    return {'cam': cam, 'frame': frame, 'side': side}


# ---------------------------------------------------------------------------
# self-check on the written dataset
# ---------------------------------------------------------------------------

def verify(session_dir: Path, frames: list[int], voxel: float = 0.005, pad: float = 0.25) -> dict:
    """Leave-one-out visual hull on the exported crops and their intrinsics.

    If crop, resize and intrinsics are consistent, recall (mask pixels covered by the hull
    of the other five crops) stays close to the value on the raw masks.
    """
    recalls, depths = [], []
    for frame in frames:
        stem = f'{frame:08d}'
        views, masks = {}, {}
        for cam in CAMS:
            info = np.load(session_dir / cam / f'img{stem}.npy', allow_pickle=True).item()
            views[cam] = {'K': info['intrinsics'], 'dist': np.zeros(5),
                          'R': info['extrinsics'][:3, :3], 't': info['extrinsics'][:3, 3]}
            masks[cam] = np.array(Image.open(session_dir / cam / f'mask{stem}.png')) > 127
        # box from the six-view hull of a generous region around the crop rays' meeting point
        centre = _ray_centre(views, masks)
        hull_all = carve_hull(masks, views, centre - 0.6, centre + 0.6, 0.01, distort=False)
        if len(hull_all) == 0:
            recalls.append({cam: 0.0 for cam in CAMS})
            continue
        lo, hi = hull_all.min(0) - pad / 2, hull_all.max(0) + pad / 2
        rec = {}
        for held in CAMS:
            others = {c: m for c, m in masks.items() if c != held}
            hull = carve_hull(others, views, lo, hi, voxel, distort=False)
            proj = render_points(hull, views[held], masks[held].shape, voxel, distort=False)
            rec[held] = float((proj & masks[held]).sum() / max(masks[held].sum(), 1))
            _, d = project(hull_all, views[held], distort=False)
            depths.append((d.min(), d.max()))
        recalls.append(rec)
    return {'recall': recalls, 'depth_m': (float(min(d[0] for d in depths)), float(max(d[1] for d in depths)))}


def _ray_centre(views: dict, masks: dict) -> np.ndarray:
    """Least-squares point closest to the rays through each crop's mask centroid."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for cam, mask in masks.items():
        ys, xs = np.nonzero(mask)
        v = views[cam]
        ray_cam = np.linalg.inv(v['K']) @ np.array([xs.mean(), ys.mean(), 1.0])
        d = v['R'].T @ ray_cam
        d /= np.linalg.norm(d)
        o = -v['R'].T @ v['t']
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def load_offsets(sync_csv: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per camera (window centres, offsets) from estimate_sync.py, skipping flat windows."""
    with open(sync_csv) as f:
        rows = list(csv.DictReader(f))
    centres = np.array([(int(r['window_start']) + int(r['window_end'])) / 2 for r in rows])
    offsets = {}
    for cam in CAMS:
        keep = np.array([r[f'{cam}_flat'] != 'True' for r in rows])
        offsets[cam] = (centres[keep], np.array([float(r[cam]) for r in rows])[keep])
    return offsets


def source_frames(frame: int, offsets: dict | None) -> dict[str, int]:
    """Each camera's frame showing the moment of reference frame `frame` (cam_4 time)."""
    if offsets is None:
        return {cam: frame for cam in CAMS}
    return {cam: frame + int(round(np.interp(frame, *offsets[cam]))) for cam in CAMS}


def normalised_scale(rig: dict) -> float:
    """BEAST's camera normalisation: cam_1 at the origin, mean camera distance = 1."""
    pos = np.stack([-rig[c]['R'].T @ rig[c]['t'] for c in CAMS])
    return float(np.linalg.norm(pos - pos[0], axis=1).mean())


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='session folder with cam_1..cam_6')
    ap.add_argument('--calib', type=Path, required=True, help='GUI 1 calibration JSON')
    ap.add_argument('--out', type=Path, required=True, help='dataset root (BEAST data_dir)')
    ap.add_argument('--session', default=None, help='session folder name (default: src folder name)')
    ap.add_argument('--animals', choices=['both', '1', '2'], default='both')
    ap.add_argument('--frames', default=None, metavar='START:END',
                    help='only use frames in [START, END) (e.g. the part of the session where cameras are in sync)')
    ap.add_argument('--sync-csv', type=Path, default=None,
                    help='per-camera frame offsets from estimate_sync.py (linear between windows)')
    ap.add_argument('--n-frames', type=int, default=2000)
    ap.add_argument('--select', choices=['kmeans', 'uniform'], default='kmeans')
    ap.add_argument('--pool-step', type=int, default=3, help='consider every Nth frame for selection')
    ap.add_argument('--out-size', type=int, default=256, help='must equal image_tokenizer.image_size')
    ap.add_argument('--margin', type=float, default=0.15, help='crop margin per side, fraction of the box')
    ap.add_argument('--min-side', type=int, default=256, help='smallest crop side in source pixels')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--verify-frames', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    session = args.session or src.name
    ids = animal_ids(args.animals)
    kind = image_kind(args.animals)
    rig = load_rig(args.calib.expanduser())

    # reference frames whose (sync-corrected) source frame exists in every camera, as image and mask
    offsets = load_offsets(args.sync_csv.expanduser()) if args.sync_csv else None
    have = {cam: list_frames(src, cam, 'masks_raw') & list_frames(src, cam, kind) for cam in CAMS}
    sources = {}
    for f in range(min(min(h) for h in have.values()), max(max(h) for h in have.values()) + 1):
        sf = source_frames(f, offsets)
        if all(sf[cam] in have[cam] for cam in CAMS):
            sources[f] = sf
    common = sorted(sources)
    if args.frames:
        start, end = (int(x) for x in args.frames.split(':'))
        common = [f for f in common if start <= f < end]
    candidates = [(f, sources[f]) for f in common[:: args.pool_step]]
    print(f'{len(common)} frames have {kind} + masks_raw in all 6 cameras'
          f'{" after sync correction" if offsets else ""}; scoring {len(candidates)}')

    frames, n_valid = select_frames(src, candidates, ids, args.n_frames, args.select, args.workers, args.seed)
    print(f'{n_valid} candidates show rat(s) {ids} in every camera; selected {len(frames)} ({args.select})')

    session_dir = out / session
    for cam in CAMS:
        (session_dir / cam).mkdir(parents=True, exist_ok=True)
    tasks = [{
        'cam': cam, 'frame': f, 'source_frame': sources[f][cam], 'src': src, 'out_dir': session_dir,
        'rigcam': rig[cam],
        'ids': ids, 'image_kind': kind, 'out_size': args.out_size,
        'margin': args.margin, 'min_side': args.min_side,
    } for f in frames for cam in CAMS]
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(export_view, tasks, chunksize=8)):
            results.append(r)
            if (i + 1) % 1200 == 0 or i + 1 == len(tasks):
                print(f'  exported {i + 1}/{len(tasks)} views')

    (session_dir / 'selected_frames.csv').write_text(''.join(f'img{f:08d}.png\n' for f in frames))
    sides = np.array([r['side'] for r in results])
    info = {
        'dataset': f'{session}_beast3d',
        'description': 'BEAST3D multi-view dataset built by make_beast3d_dataset.py (undistorted square crops)',
        'available_views': CAMS,
        'anchor_view': 'cam_1',
        'video_ids': [session],
        'number_of_videos': 1,
        'input_directory': str(src),
        'output_directory': str(out),
        'frames_per_video': len(frames),
        'n_digits': 8,
        'extension': 'png',
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'seed': args.seed,
        'total_ssl_frames': len(frames) * len(CAMS),
        'calibration': str(args.calib.expanduser().resolve()),
        'animals': args.animals,
        'frame_range': args.frames,
        'sync_csv': str(args.sync_csv.expanduser().resolve()) if args.sync_csv else None,
        'out_size': args.out_size,
        'crop_margin': args.margin,
        'crop_side_px': {'min': int(sides.min()), 'median': int(np.median(sides)), 'max': int(sides.max())},
    }
    (out / 'info.json').write_text(json.dumps(info, indent=2))
    print(f'crop side in source px: min {sides.min()}, median {int(np.median(sides))}, max {sides.max()} '
          f'(scale to {args.out_size}: median x{args.out_size / np.median(sides):.2f})')

    vf = [frames[i] for i in np.linspace(0, len(frames) - 1, min(args.verify_frames, len(frames))).round().astype(int)]
    check = verify(session_dir, vf)
    rec = np.array([[r[c] for c in CAMS] for r in check['recall']])
    print('self-check, leave-one-out hull recall on the exported crops (median per camera):')
    print('  ' + '  '.join(f'{c} {v:.3f}' for c, v in zip(CAMS, np.median(rec, 0))))
    scale = normalised_scale(rig)
    dmin, dmax = check['depth_m']
    print(f'rat depth from the cameras: {dmin:.2f} to {dmax:.2f} m; BEAST normalises 1 unit = {scale:.3f} m, '
          f'so {dmin / scale:.2f} to {dmax / scale:.2f} units')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
