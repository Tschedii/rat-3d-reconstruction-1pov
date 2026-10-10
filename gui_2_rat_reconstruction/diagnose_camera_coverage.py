#!/usr/bin/env python3
"""
diagnose_camera_coverage.py
----------------------------
Per-camera silhouette-coverage diagnostic for an already-reconstructed
mesh. For every vertex of mesh.obj, re-projects it into each calibrated
camera and checks whether it falls inside that camera's mask -- the same
test carve_voxels() uses during reconstruction. Reports, per camera, how
often it "votes yes" for the two extremal regions of the mesh along its
principal axis (typically head vs. tail/rear) versus the mesh as a whole,
plus the mean number of cameras agreeing on each region. A camera whose
miss-rate is much higher for one region than the rest, or a region whose
mean agreement count sits right at (or below) min_cameras, is the
under-constrained one -- exactly the kind of thing that produces a
carved-away head at a strict min_cameras threshold.

Also writes one overlay PNG per camera: the mask in grey, with the mesh's
projected vertices drawn on top (green = all vertices, magenta = one
axis-tail, yellow = the other) so you can see at a glance which tail is
the head and whether the silhouette actually covers it there.

Usage:
    python diagnose_camera_coverage.py \
        --calibration /path/to/2026_09_10.json \
        --masks-root /path/to/data/masks \
        --object-id 1 \
        --frame frame_000240 \
        --mesh /path/to/frame_000240/mesh.obj \
        --out-dir /path/to/coverage_frame_000240
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruction_lib as rl


def load_obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64)


def principal_axis_tails(verts, frac):
    centre = verts.mean(axis=0)
    centered = verts - centre
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    n = len(verts)
    k = max(1, int(n * frac))
    order = np.argsort(proj)
    tail_neg_idx = order[:k]
    tail_pos_idx = order[-k:]
    return tail_neg_idx, tail_pos_idx, proj


def per_camera_hits(verts, cam, mask):
    h, w = mask.shape
    px, in_front = rl.project_points(verts, cam["R"], cam["t"], cam["K"], cam["dist"])
    u = np.round(px[:, 0]).astype(np.int32)
    v = np.round(px[:, 1]).astype(np.int32)
    inside_frame = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hit = np.zeros(len(verts), dtype=bool)
    hit[inside_frame] = mask[v[inside_frame], u[inside_frame]] > 0
    return hit, u, v, inside_frame


def draw_overlay(mask, u, v, inside_frame, all_idx, tail_neg_idx, tail_pos_idx):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = (overlay * 0.5).astype(np.uint8)

    def draw(idx, color, radius):
        for i in idx:
            if not inside_frame[i]:
                continue
            cv2.circle(overlay, (int(u[i]), int(v[i])), radius, color, -1)

    draw(all_idx, (0, 180, 0), 1)
    draw(tail_neg_idx, (255, 0, 255), 2)
    draw(tail_pos_idx, (0, 255, 255), 2)
    return overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True, help="Path to the rig calibration JSON")
    ap.add_argument("--masks-root", required=True, help="Root folder containing per-camera masks (any of layouts A/B/C)")
    ap.add_argument("--object-id", required=True, help="e.g. 1 for rat_1")
    ap.add_argument("--frame", required=True, help="e.g. frame_000240")
    ap.add_argument("--mesh", required=True, help="Path to the reconstructed mesh.obj for this frame")
    ap.add_argument("--out-dir", required=True, help="Where to write the per-camera overlay PNGs")
    ap.add_argument("--tail-frac", type=float, default=0.1, help="Fraction of vertices to treat as each axis-tail region (default 0.1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = rl.load_calibration(args.calibration)
    frames = rl.find_frames(args.masks_root, args.object_id, "single", {"frame_id": args.frame})
    _, mask_paths = frames[0]
    ordered_paths = rl.match_masks_to_cameras(cameras, mask_paths)
    _, masks = rl.load_masks(ordered_paths)

    verts = load_obj_vertices(args.mesh)
    tail_neg_idx, tail_pos_idx, proj = principal_axis_tails(verts, args.tail_frac)
    all_idx = np.arange(len(verts))

    print(f"Loaded {len(verts):,} vertices, {len(cameras)} cameras "
          f"({sum(m is not None for m in masks)} with masks for {args.frame})")
    print(f"Principal-axis span: [{proj.min():+.4f}, {proj.max():+.4f}]  "
          f"tail groups: {len(tail_neg_idx):,} verts each end (frac={args.tail_frac})\n")

    votes = np.zeros(len(verts), dtype=np.int32)
    header = f"{'camera':<14}{'all miss%':>10}{'tail- miss%':>13}{'tail+ miss%':>13}"
    print(header)
    print("-" * len(header))

    for cam, mask, path in zip(cameras, masks, ordered_paths):
        if mask is None:
            print(f"{cam['name']:<14}{'NO MASK':>10}")
            continue
        hit, u, v, inside_frame = per_camera_hits(verts, cam, mask)
        votes += hit.astype(np.int32)

        all_miss = 100.0 * (~hit).mean()
        neg_miss = 100.0 * (~hit[tail_neg_idx]).mean()
        pos_miss = 100.0 * (~hit[tail_pos_idx]).mean()
        print(f"{cam['name']:<14}{all_miss:>9.1f}%{neg_miss:>12.1f}%{pos_miss:>12.1f}%")

        overlay = draw_overlay(mask, u, v, inside_frame, all_idx, tail_neg_idx, tail_pos_idx)
        cv2.imwrite(str(out_dir / f"{cam['name']}_coverage.png"), overlay)

    print(f"\nMean cameras agreeing per vertex: overall={votes.mean():.2f}  "
          f"tail-={votes[tail_neg_idx].mean():.2f}  tail+={votes[tail_pos_idx].mean():.2f}")
    print("(compare against the min_cameras used for this reconstruction -- a region "
          "whose mean sits at or just above that threshold is only marginally carved in, "
          "and any camera with a high miss% there is the one under-covering it)")
    print(f"\nOverlay images written to {out_dir}/ "
          "(green = all mesh vertices, magenta = one axis-tail, yellow = the other -- "
          "look at the images to tell which tail is the head)")


if __name__ == "__main__":
    main()
