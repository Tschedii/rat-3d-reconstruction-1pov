"""Template re-fit for the slide-11 video, upright and fast: GUI 2's template_fit_lib.fit_template on the visual
hull GUI 2 already carved for the frame (mesh_hull.obj, or the hull run's mesh.obj), so the 20 s carve is not
repeated, with the start pinned to the template's back up and nose towards the head the cameras tracked
(GUI 5.2), warm-started frame to frame like GUI 2 (cold again after a > 10 cm move). Then GUI 2's own texture
bake from the six camera cutouts.

    python refit_fast.py <object> <f1,f2,... | first-last> [--iters N] [--threads N]
        -> refit/object_<n>_template_fitted/frame_XXXXXX/{mesh.obj, mesh.mtl, mesh_texture.png, mesh_hull.obj}
"""
import argparse
import csv
import shutil
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
GUI2 = DRIVE / "gui_2_rat_reconstruction/output"
DL = Path.home() / "Downloads"


def hull_path(obj, f, near=0):
    for g in sorted(range(f - near, f + near + 1), key=lambda g: abs(g - f)):
        p = _hull_path(obj, g)
        if p is not None:
            return p
    return None


def _hull_path(obj, f):
    fid = f"frame_{f:06d}"
    for p in (GUI2 / f"object_{obj}_template_fitted" / fid / "mesh_hull.obj",
              DL / f"object_{obj}_template_fitted" / f"object_{obj}" / fid / "mesh_hull.obj",
              *([DL / fid / "mesh_hull.obj"] if obj == 2 else []),
              GUI2 / f"object_{obj}" / fid / "mesh.obj"):              # the hull-only run (same carve)
        if p.exists():
            return p
    return None


def read_v(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])


def parse_frames(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", type=int); ap.add_argument("frames")
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "refit"))
    ap.add_argument("--drop-bloated", action="store_true",
                    help="ignore a camera's mask where mask_stats.py flagged it as bloated (wall reflection, merged rats)")
    ap.add_argument("--hull-near", type=int, default=0, help="use the nearest carved hull within this many frames")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    B, tc = al.table_frame_from_rig(build_initial_rig(str(CALIB)))
    up_C = np.asarray(B)[2]
    heads = {}
    for row in csv.DictReader(open(HEADS / f"rat{a.obj}" / "head_pose_world.csv")):
        if row["head_calib_x_m"]:
            heads[int(row["frame"])] = np.array([float(row[k]) for k in ("head_calib_x_m", "head_calib_y_m", "head_calib_z_m")])
    cur = {"f": None}
    orig = tfl.init_candidates

    def init_candidates(template_verts, hull_verts):
        cands = orig(template_verts, hull_verts)
        f = cur["f"]
        head = heads.get(f, heads[min(heads, key=lambda k: abs(k - f))])
        hc = hull_verts.mean(0)
        keep = [c for c in cands if (c[1] @ [0, 0, 1.0]) @ up_C > 0 and (c[1] @ [1.0, 0, 0]) @ (head - hc) > 0]
        return keep or [c for c in cands if (c[1] @ [0, 0, 1.0]) @ up_C > 0] or cands
    tfl.init_candidates = init_candidates

    cams = rl.load_calibration(str(CALIB))
    tv, tf = tfl.load_obj(str(TEMPLATE))
    z0 = tv[:, 2]; dorsal, ventral = z0 >= np.percentile(z0, 85), z0 <= np.percentile(z0, 8)
    import json
    flags = json.load(open(HERE / "logs" / "mask_flags.json")).get(str(a.obj), {}) if a.drop_bloated else {}
    state, centre = None, None
    for f in parse_frames(a.frames):
        fid = f"frame_{f:06d}"
        d = Path(a.out) / f"object_{a.obj}_template_fitted" / fid
        if (d / "mesh.obj").exists():
            continue
        hp = hull_path(a.obj, f, a.hull_near)
        if hp is None:
            print(fid, "no hull", flush=True); state = None; continue
        t = time.time()
        hull = read_v(hp)
        (_, mp), = rl.find_frames(str(MASKS), a.obj, "range", {"start": f, "end": f})
        colors, masks = rl.load_masks(rl.match_masks_to_cameras(cams, mp))
        dropped = [i for i, cm in enumerate(cams) if int(cm["name"][-1]) in flags.get(str(f), [])]
        for i in dropped:
            masks[i] = None
        c = hull.mean(0)
        warm = state if (state is not None and centre is not None and np.linalg.norm(c - centre) <= 0.10) else None
        cur["f"] = f
        try:
            fit = tfl.fit_template(tv, tf, cams, masks, hull, params={"iterations": a.iters, "n_nodes": 200, "sil_cameras": None},
                                   log=lambda m: None, warm_start=warm)
        except Exception as e:
            print(fid, "FAILED", e, flush=True); state = None; continue
        state, centre = fit["state"], c
        V, Fc = fit["verts"], fit["faces"]
        uv, _ = rl.unwrap_cylindrical(V, Fc)
        atlas = rl.bake_texture_atlas(V, Fc, uv, cams, colors, masks, atlas_size=1536, log=lambda m: None)
        rl.save_textured_obj(d / "mesh.obj", V, Fc, uv, atlas)
        shutil.copy2(hp, d / "mesh_hull.obj")
        upv = V[dorsal].mean(0) - V[ventral].mean(0)
        print(fid, f"{time.time() - t:.1f} s{' (warm)' if warm is not None else ' (cold)'}, dropped {dropped}, hull {hp.parent.name}, supported "
                   f"{fit['report'].get('supported_frac', float('nan')):.2f}, back-up {float(upv @ up_C / np.linalg.norm(upv)):+.2f}",
              flush=True)


if __name__ == "__main__":
    main()
