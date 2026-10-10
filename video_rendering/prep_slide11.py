"""Data for the slide-11 video (render_slide11.py): everything in the table frame of GUI 4 / GUI 5
(build_initial_rig: z up, origin where the optical axes meet, metres; the arena floor at z = 5.5 cm).

  rig          the six cameras of the 2026_09_21 solve: centre, axes, frustum corners, colour
  skulls       both rats' IMU head pose per recording frame 0..1000 (GUI 5.1 two-rat export:
               head position + head quaternion, x = nose, y = left, z = up)
  recon        GUI 2's reconstruction of both rats at frame 500, step by step: the seed point and
               the per-camera rays it comes from, a display voxel grid (1 cm) with every camera's
               silhouette test, the 2 mm visual hull, the textured hull, the template fit and how
               far each template vertex lies outside the hull (what the template added)
  meshes       per frame 500..1000 and rat: the template-fitted textured mesh to show
  environment  Environment 4 (the room scan) and its placement (GUI 4, checked on the rats)

    python prep_slide11.py      -> slide11_video/data/
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DRIVE = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(DRIVE / "gui_2_rat_reconstruction"))
sys.path.insert(0, str(DRIVE / "gui_5.2_head_tracking"))
import reconstruction_lib as rl      # noqa: E402
import analysis_lib as al            # noqa: E402
from gui4_rig import build_initial_rig  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
CALIB = DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json"
MASKS = DRIVE / "data/raw_data/Day_7_rPEG008_rPEG007_25-08-2026"
GUI2 = DRIVE / "gui_2_rat_reconstruction/output"
EXTRA_RECON = HERE / "inputs/recon"           # extra GUI 2 runs at frame 500 (intermediate steps)
SKULL_CSV = DRIVE / "gui_5.1_quaternion_placement/exports/two_rats_skull_0-1400"
ENV_OBJ = DRIVE / "data/scenes/Environment_4/31.8.2026.obj"
ENV_TRANSFORM = HERE / "inputs/env_transform.json"   # GUI 4 placement of Environment 4
CAP_STL = DRIVE / "gui_5.1_quaternion_placement/assets/cap_head.stl"
CAP_ALIGN_DEG = [-90, 0, -90]          # GUI 5.1's cap drawing (three.js Euler XYZ on the scan)
COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#5FA8D3", "#6D6875"]   # the slide's rig colours
RECON_FRAME = 500
FRUSTUM_D = 0.2
DISPLAY_VOXEL = 0.01


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = build_initial_rig(str(CALIB))
    B, tc = al.table_frame_from_rig(rig)
    B, tc = np.asarray(B), np.asarray(tc)
    to_T = lambda p: np.asarray(p) @ B.T - tc               # calibration frame -> table frame
    calib_json = json.loads(CALIB.read_text())
    cams = rl.load_calibration(str(CALIB))
    meta = {"T_from_calib": np.vstack([np.hstack([B, -tc[:, None]]), [0, 0, 0, 1]]).tolist(), "floor_z": 0.055}

    # --- rig -------------------------------------------------------------------
    out_cams = []
    for c, cj in zip(cams, calib_json["cameras"]):
        assert c["name"] == cj["name"]
        K = cj["intrinsics"]
        w, h = K["image_size_wh"]
        corners = []
        for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
            d = np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
            corners.append(to_T(c["cam_pos"] + FRUSTUM_D * (c["R"].T @ d)).tolist())
        n = int(c["name"][-1])
        out_cams.append({"name": c["name"], "label": f"cam {n}", "color": COLORS[n - 1],
                         "centre": to_T(c["cam_pos"]).tolist(), "corners": corners,
                         "axes_T": (B @ c["R"].T).T.tolist()})        # rows: camera x, y, z (optical axis) in T
    meta["cameras"] = out_cams

    # --- skulls: GUI 5.1 IMU head pose per frame --------------------------------
    rats = {}
    for r in (1, 2):
        rows = list(csv.DictReader(open(glob.glob(str(SKULL_CSV / f"*_rat{r}.csv"))[0])))
        pose = {}
        for row in rows:
            f = int(row["frame"])
            if f > 1000:
                continue
            q = [float(row[k]) for k in ("qw", "qx", "qy", "qz")]
            has = row["has_position"] == "1"
            pose[f] = {"q": q, "head": [float(row[k]) for k in ("head_x_m", "head_y_m", "head_z_m")] if has else None,
                       "body": [float(row[k]) for k in ("body_x_m", "body_y_m", "body_z_m")] if has else None}
        rats[r] = pose
    E = Rotation.from_euler("XYZ", CAP_ALIGN_DEG, degrees=True).as_matrix()
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)          # three = M @ table
    meta["cap"] = {"stl": str(CAP_STL), "fixed_rotation": (M.T @ E).tolist(),
                   "note": "vertex_T = R_head(q) @ fixed_rotation @ (stl_mm - bbox_centre) / 1000 + head"}
    json.dump({str(r): {str(f): v for f, v in p.items()} for r, p in rats.items()}, open(OUT / "skull_poses.json", "w"))

    # --- reconstruction at frame 500 --------------------------------------------
    recon = {}
    for obj in (1, 2):
        (fid, mp), = rl.find_frames(str(MASKS), obj, "range", {"start": RECON_FRAME, "end": RECON_FRAME})
        ordered = rl.match_masks_to_cameras(cams, mp)
        colors, masks = rl.load_masks(ordered)
        centre = rl.estimate_rat_centre(cams, ordered, log=lambda m: None)
        rays = []
        for cam, mask in zip(cams, masks):
            if mask is None:
                continue
            px = rl._mask_centroid_px(mask)
            import cv2
            u, v = cv2.undistortPoints(np.array([[px]], np.float64), cam["K"], cam["dist"])[0, 0]
            d = cam["R"].T @ np.array([u, v, 1.0]); d /= np.linalg.norm(d)
            o = cam["cam_pos"]
            s = (centre - o) @ d                                   # closest approach to the seed
            rays.append({"camera": cam["name"], "from": to_T(o).tolist(), "to": to_T(o + d * (s + 0.12)).tolist()})
        pts, shape, origin = rl.build_voxel_grid(centre, 0.20, 0.05, 0.20, DISPLAY_VOXEL)
        hits = []
        for cam, mask in zip(cams, masks):
            hit = np.zeros(len(pts), bool)
            if mask is not None:
                hgt, wid = mask.shape
                px, front = rl.project_points(pts, cam["R"], cam["t"], cam["K"], cam["dist"])
                uu, vv = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
                ok = front & (uu >= 0) & (uu < wid) & (vv >= 0) & (vv < hgt)
                hit[ok] = mask[vv[ok], uu[ok]] > 0
            hits.append(hit)
        hits = np.array(hits)                                       # (6, N), camera order cam 1..6
        # GUI 2 keeps a voxel seen inside >= 4 of the 6 silhouettes: shown camera by camera, a voxel
        # outside a camera's silhouette is marked, and gone once it can no longer reach 4 (3rd miss)
        misses = np.cumsum(~hits, axis=0)
        alive = misses <= len(cams) - 4
        np.savez_compressed(OUT / f"voxels_rat{obj}.npz", centres_T=to_T(pts).astype(np.float32),
                            hits=hits, size=DISPLAY_VOXEL)
        # the upright re-fit (refit_fast.py), smoothed over time (smooth_fits.py)
        tdir = HERE / "refit_smooth" / f"object_{obj}" / fid
        # the hull the template was fitted to, textured by texture_hull.py when GUI 2's hull-only run carved
        # differently (rat 2: its hull-only mesh is an older, smaller carve)
        hull_tex = HERE / "data" / f"hull500_rat{obj}" / "mesh.obj"
        if not hull_tex.exists():
            hull_tex = GUI2 / f"object_{obj}" / fid / "mesh.obj"
        Vt = np.array([[float(x) for x in l.split()[1:4]] for l in open(tdir / "mesh.obj") if l.startswith("v ")])
        hull_file = HERE / "refit" / f"object_{obj}_template_fitted" / fid / "mesh_hull.obj"
        Vh = np.array([[float(x) for x in l.split()[1:4]] for l in open(hull_file) if l.startswith("v ")])
        added = cKDTree(Vh).query(Vt)[0]                            # template vertex -> nearest hull vertex (m)
        np.save(OUT / f"template_added_rat{obj}.npy", added.astype(np.float32))
        recon[obj] = {"seed_T": to_T(centre).tolist(), "rays": rays,
                      "voxels_alive": [int(a.sum()) for a in alive], "voxels_total": int(len(pts)),
                      "hull_textured": str(hull_tex), "hull_plain": str(hull_file),
                      "template_textured": str(tdir / "mesh.obj"),
                      "template_added_cm": {"median": float(np.median(added) * 100), "p90": float(np.percentile(added, 90) * 100)}}
        print(f"rat {obj}: seed {np.round(to_T(centre), 3)}; voxels alive per camera {recon[obj]['voxels_alive']} "
              f"of {len(pts)}; template vertices >1 cm outside the hull {float((added > 0.01).mean()) * 100:.0f}%")
    meta["recon"] = recon

    # --- per-frame meshes 500..1000 ---------------------------------------------
    meshes = {}
    for obj in (1, 2):
        lst = {}
        for f in range(RECON_FRAME, 1001):
            fid = f"frame_{f:06d}"
            # only the upright re-fits (smoothed where checked): GUI 2's earlier rat 2 fits are all belly-up.
            # Rat 1 is not shown after frame 750 (it is the viewer), so its earlier fits stay as a fallback there
            for p in (HERE / "refit_smooth" / f"object_{obj}" / fid / "mesh.obj",
                      HERE / "refit" / f"object_{obj}_template_fitted" / fid / "mesh.obj",
                      *([GUI2 / f"object_{obj}_template_fitted" / fid / "mesh.obj"] if obj == 1 and f > 750 else [])):
                if p.exists():
                    lst[f] = str(p)
                    break
        meshes[str(obj)] = {str(f): p for f, p in lst.items()}
        missing = [f for f in range(RECON_FRAME, 1001) if f not in lst]
        print(f"rat {obj}: template meshes for {len(lst)}/501 frames; missing {missing[:6]}{'...' if len(missing) > 6 else ''}")
    json.dump(meshes, open(OUT / "meshes.json", "w"), indent=0)

    meta["environment"] = {"obj": str(ENV_OBJ), "T_from_scan": json.load(open(ENV_TRANSFORM))["T_from_scan"]}
    json.dump(meta, open(OUT / "scene.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
