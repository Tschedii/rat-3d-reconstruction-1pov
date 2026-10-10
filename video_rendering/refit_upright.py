"""Re-fit GUI 2's template completion for one rat with its orientation pinned down: the template's back up
(table up) and its nose towards the head the cameras tracked (GUI 5.2). GUI 2's fit keeps whichever of four
axis-sign starts fits the silhouettes best, and a belly-up rat casts nearly the same silhouettes, so rat 2's
fits came out belly-up in every frame (each frame warm-starts from the previous one, so one bad start sticks).

Optionally on the GPU (MPS): grid_sample's border padding is not available there, so image lookups clamp the
coordinates and use zero padding instead, which is the same thing for bilinear sampling with align_corners.

    python refit_upright.py <object> <first> <last> [--mps] [--iters N] [--only f1,f2,...]
        -> refit/object_<n>_template_fitted/frame_XXXXXX/{mesh.obj, mesh.mtl, mesh_texture.png, mesh_hull.obj}
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

DRIVE = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(DRIVE / "gui_2_rat_reconstruction"))
sys.path.insert(0, str(DRIVE / "gui_5.2_head_tracking"))
import reconstruction_lib as rl          # noqa: E402
import template_fit_lib as tfl           # noqa: E402
import analysis_lib as al                # noqa: E402
from gui4_rig import build_initial_rig   # noqa: E402

HERE = Path(__file__).resolve().parent
CALIB = DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json"
MASKS = DRIVE / "data/raw_data/Day_7_rPEG008_rPEG007_25-08-2026"
HEADS = DRIVE / "gui_5.2_head_tracking/exports/masks_20261008_224517_rats12_000000-001400"
TEMPLATE = DRIVE / "gui_2_rat_reconstruction/templates/rat_brown.obj"      # +x nose, +z up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", type=int); ap.add_argument("first", type=int); ap.add_argument("last", type=int)
    ap.add_argument("--mps", action="store_true"); ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--only", default=""); ap.add_argument("--out", default=str(HERE / "refit"))
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    if a.mps:
        torch.set_default_device("mps")

        def sample_image(img, uv, H, W):
            """grid_sample(bilinear, border, align_corners) by hand: its backward is not on MPS."""
            x = uv[:, 0].clamp(0, W - 1); y = uv[:, 1].clamp(0, H - 1)
            x0 = x.detach().floor().clamp(0, W - 2); y0 = y.detach().floor().clamp(0, H - 2)
            wx, wy = x - x0, y - y0
            i00 = (y0 * W + x0).long()
            flat = img.reshape(-1)
            return (flat[i00] * (1 - wx) * (1 - wy) + flat[i00 + 1] * wx * (1 - wy)
                    + flat[i00 + W] * (1 - wx) * wy + flat[i00 + W + 1] * wx * wy)
        tfl.sample_image = sample_image

    B, tc = al.table_frame_from_rig(build_initial_rig(str(CALIB)))
    up_C = np.asarray(B)[2]                                       # table up, in the calibration frame
    heads = {}
    for row in csv.DictReader(open(HEADS / f"rat{a.obj}" / "head_pose_world.csv")):
        if row["head_calib_x_m"]:
            heads[int(row["frame"])] = np.array([float(row[k]) for k in ("head_calib_x_m", "head_calib_y_m", "head_calib_z_m")])
    state = {"frame": None}
    orig = tfl.init_candidates

    def init_candidates(template_verts, hull_verts):
        cands = orig(template_verts, hull_verts)
        f = state["frame"]
        head = heads.get(f) if f is not None else None
        if head is None:                                  # nearest tracked head
            near = min(heads, key=lambda k: abs(k - f)) if heads and f is not None else None
            head = heads.get(near)
        hull_c = hull_verts.mean(0)
        keep = []
        for c in cands:
            R0 = c[1]
            up_ok = (R0 @ np.array([0.0, 0.0, 1.0])) @ up_C > 0
            nose_ok = head is None or (R0 @ np.array([1.0, 0.0, 0.0])) @ (head - hull_c) > 0
            if up_ok and nose_ok:
                keep.append(c)
        if not keep:
            keep = [c for c in cands if (c[1] @ np.array([0.0, 0.0, 1.0])) @ up_C > 0] or cands
        print(f"    frame {f}: {len(keep)} of {len(cands)} starts kept (back up, nose to the tracked head)", flush=True)
        return keep
    tfl.init_candidates = init_candidates

    cams = rl.load_calibration(str(CALIB))
    params = {"voxel_size": 0.002, "min_cameras": 4, "half_xy": 0.20, "depth_below": 0.05, "depth_above": 0.20,
              "head_min_cameras": None, "head_zone_frac": 0, "smooth_iterations": 10, "texture_mode": "atlas",
              "atlas_size": 1536, "completion_mode": "template", "template_path": str(TEMPLATE),
              "fit_iterations": a.iters, "fit_nodes": 200, "fit_sil_cameras": None}
    only = {int(x) for x in a.only.split(",") if x}
    frames = rl.find_frames(str(MASKS), a.obj, "range", {"start": a.first, "end": a.last})
    ctx = {}
    for fid, mp in frames:
        f = int(fid[6:])
        if only and f not in only:
            continue
        d = Path(a.out) / f"object_{a.obj}_template_fitted" / fid
        if (d / "mesh.obj").exists():
            continue
        state["frame"] = f
        t = time.time()
        try:
            r = rl.reconstruct_frame(cams, rl.match_masks_to_cameras(cams, mp), params, log=lambda m: None, fit_context=ctx)
        except Exception as e:
            print(fid, "FAILED", e, flush=True); ctx.clear(); continue
        rl.save_textured_obj(d / "mesh.obj", r["verts"], r["faces"], r["corner_uv"], r["atlas"])
        rl.save_obj(d / "mesh_hull.obj", r["hull_verts"], r["hull_faces"])
        V = np.asarray(r["verts"]); T0 = tfl.load_obj(str(TEMPLATE))[0]
        z0 = T0[:, 2]
        upv = V[z0 >= np.percentile(z0, 85)].mean(0) - V[z0 <= np.percentile(z0, 8)].mean(0)
        print(fid, f"{time.time() - t:.1f} s, supported {r['fit_report'].get('supported_frac', float('nan')):.2f}, "
                   f"back-up {float(upv @ up_C / np.linalg.norm(upv)):+.2f}", flush=True)


if __name__ == "__main__":
    main()
