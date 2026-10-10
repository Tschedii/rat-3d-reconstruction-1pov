#!/usr/bin/env python3
"""
make_rat_template.py
--------------------
Generates the default rat template mesh used by GUI 2's model-based
completion (templates/rat_template.obj): a smooth union of simple
primitives -- body, head, snout, ears, four legs and a tapered tail --
evaluated as a signed-distance field on a grid and meshed with marching
cubes. It is deliberately generic: an adult-rat-sized, neutral-pose,
watertight surface with all the parts the visual hull tends to lose. The
fitter deforms it per frame, so proportions only need to be in the right
ballpark. Swap in any better .obj (an external model, or one of your own
complete reconstructions) via the "template mesh" field in GUI 2.

Convention: metres; +X = nose direction, +Z = up, origin under the belly at
floor level.

    python make_rat_template.py [--resolution 0.0025] [--head-pitch 0] [--head-yaw 0]
                                [--tail-curl 0.4] [--out templates/rat_template.obj]

--head-pitch / --head-yaw (degrees) and --tail-curl exist so the same
generator can also produce differently-posed variants (used by the fitter's
synthetic self-test).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_lib import taubin_smooth, save_obj


def sd_ellipsoid(p, centre, radii):
    q = (p - centre) / radii
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / radii, axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_capsule(p, a, b, ra, rb):
    """Capsule from a to b with radius tapering linearly ra -> rb."""
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    r = ra + (rb - ra) * h
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def smooth_union(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rat_sdf(p, head_pitch=0.0, head_yaw=0.0, tail_curl=0.4):
    # body: ~18 cm long, 7 cm wide, 6.5 cm tall, belly ~1 cm above the floor
    d = sd_ellipsoid(p, np.array([0.0, 0.0, 0.042]), np.array([0.090, 0.036, 0.033]))

    # head assembly rotates about the neck joint (pitch = nose up/down, yaw = left/right)
    neck = np.array([0.075, 0.0, 0.052])
    Rh = rot_z(head_yaw) @ rot_y(-head_pitch)
    ph = (p - neck) @ Rh + neck   # inverse-rotate sample points into head frame
    head = sd_ellipsoid(ph, np.array([0.105, 0.0, 0.052]), np.array([0.033, 0.024, 0.023]))
    snout = sd_ellipsoid(ph, np.array([0.138, 0.0, 0.046]), np.array([0.020, 0.012, 0.011]))
    ear_l = sd_ellipsoid(ph, np.array([0.097, 0.021, 0.074]), np.array([0.006, 0.010, 0.011]))
    ear_r = sd_ellipsoid(ph, np.array([0.097, -0.021, 0.074]), np.array([0.006, 0.010, 0.011]))
    head_all = smooth_union(head, snout, 0.010)
    head_all = np.minimum(head_all, np.minimum(ear_l, ear_r))
    d = smooth_union(d, head_all, 0.020)

    # legs: front pair and hind pair, down to the floor
    for x, y, r in [(0.050, 0.026, 0.008), (0.050, -0.026, 0.008),
                    (-0.050, 0.030, 0.010), (-0.050, -0.030, 0.010)]:
        leg = sd_capsule(p, np.array([x, y, 0.030]), np.array([x, y * 1.15, 0.004]), r, r * 0.8)
        d = smooth_union(d, leg, 0.010)

    # tail: tapered chain of capsules curving down/back from the rump
    base = np.array([-0.085, 0.0, 0.040])
    n_seg = 14
    length = 0.19
    pts = [base]
    ang = 0.0
    for i in range(n_seg):
        ang += tail_curl * (i / n_seg) * 0.45
        step = np.array([-np.cos(ang), 0.0, -np.sin(ang) * 0.6]) * (length / n_seg)
        pts.append(pts[-1] + step)
    tail = None
    for i in range(n_seg):
        f0, f1 = i / n_seg, (i + 1) / n_seg
        seg = sd_capsule(p, pts[i], pts[i + 1], 0.0075 * (1 - f0) + 0.0025 * f0, 0.0075 * (1 - f1) + 0.0025 * f1)
        tail = seg if tail is None else np.minimum(tail, seg)
    d = smooth_union(d, tail, 0.008)
    return d


def build_template(resolution=0.0025, head_pitch=0.0, head_yaw=0.0, tail_curl=0.4, smooth_iterations=6):
    xs = np.arange(-0.30, 0.18, resolution)
    ys = np.arange(-0.08, 0.08, resolution)
    zs = np.arange(-0.01, 0.11, resolution)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    p = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    d = rat_sdf(p, head_pitch=head_pitch, head_yaw=head_yaw, tail_curl=tail_curl).reshape(gx.shape)
    verts_idx, faces, _, _ = measure.marching_cubes(d, level=0.0)
    verts = np.array([xs[0], ys[0], zs[0]]) + verts_idx * resolution
    if smooth_iterations > 0:
        verts = taubin_smooth(verts, faces, iterations=smooth_iterations)
    return verts, faces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", type=float, default=0.0025, help="SDF grid spacing in metres (default 2.5 mm)")
    ap.add_argument("--head-pitch", type=float, default=0.0, help="degrees, nose up (+) / down (-)")
    ap.add_argument("--head-yaw", type=float, default=0.0, help="degrees, nose left (+) / right (-)")
    ap.add_argument("--tail-curl", type=float, default=0.4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "templates" / "rat_template.obj"))
    args = ap.parse_args()
    verts, faces = build_template(args.resolution, args.head_pitch, args.head_yaw, args.tail_curl)
    save_obj(args.out, verts, faces)
    print(f"wrote {args.out}: {len(verts):,} verts, {len(faces):,} faces")


if __name__ == "__main__":
    main()
