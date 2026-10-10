"""One rat, a few frames, re-fitted with GUI 2's template fit under changed settings, to fix fits whose shape is
wrong although their silhouettes match (rat 2 at frame 500: hunched, the template head pushed back into a bloated
torso). Settings:
  --w-reg      smoothness weight of the deformation graph (GUI 2: 0.5); higher keeps the template's anatomy
  --n-nodes    deformation-graph nodes (GUI 2: 200); fewer = stiffer
  --w-head     soft anchor of the template's head top to the head the cameras tracked (GUI 5.2), robust, in cm;
               the tracked point lies HEAD_UP below the head-top centroid on well-fitted frames
  --w-nose     soft anchor of the template's nose to the real nose: triangulated by hand at frame 500 from
               cams 2, 3, 5 (data/rat2_nose_f500_T.npy, agreeing to < 8 px), carried to other frames rigidly with
               the head (tracked headstage point + the IMU head rotation). Needed because the headstage is part of
               the rat's masks and the template has none: the silhouette terms pull the template head up into the
               headstage and the real face, below it, is filled with torso
  --erase-headstage  treat the headstage as unknown: the headstage scan placed as in the video
               (placement.skull_matrix: tracked head + IMU, checked on the raw views) projected into each camera;
               those pixels stay allowed for the template (containment) but no longer pull it (coverage), are left
               out of the texture, and visual-hull points inside the headstage no longer anchor the fit. In views
               where the rat faces the camera the headstage covers its own face, so the pixels cannot simply be
               deleted
Starts as refit_fast.py (back up, nose towards the tracked head), warm-started frame to frame, GUI 2's texture bake.

    python refit_variant.py <rat> <frames> --name N [--w-reg R] [--n-nodes K] [--w-head W]
                                                [--iters I] [--threads T]   -> variants/<N>/object_<rat>/frame_XXXXXX/
"""
import argparse
import csv
import json
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
sys.path.insert(0, str(HERE))
from refit_fast import CALIB, MASKS, HEADS, TEMPLATE, hull_path, read_v, parse_frames   # noqa: E402
HEAD_UP = 0.012


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", type=int); ap.add_argument("frames"); ap.add_argument("--name", required=True)
    ap.add_argument("--w-reg", type=float, default=0.5); ap.add_argument("--n-nodes", type=int, default=200)
    ap.add_argument("--w-head", type=float, default=0.0); ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--w-nose", type=float, default=0.0)
    ap.add_argument("--erase-headstage", action="store_true"); ap.add_argument("--debug-masks", action="store_true")
    ap.add_argument("--hs-box", action="store_true",
                    help="the headstage box (the scan above the dental-cement cap, head-frame z > HS_ZMIN) is solid: "
                         "template vertices inside it are penalised (--w-hsx); only the box (not the cement on the "
                         "skull) is used for the unknown pixels and the dropped hull points")
    ap.add_argument("--w-hsx", type=float, default=1.0)
    ap.add_argument("--real-hs", action="store_true",
                    help="use the headstage triangulated in the camera images (find_headstage.py, "
                         "data/rat<n>_headstage_track_T.npy) instead of the video's skull placement: a sphere of "
                         "HS_R around it is solid (--w-hsx), its pixels are unknown for coverage, hull points inside "
                         "it are dropped; the head top is pulled to HS_TOP below it and the nose target is carried "
                         "with it (offset from frame 500 in the IMU head frame)")
    ap.add_argument("--head-up", type=float, default=None, help="override HEAD_UP (m along table up; <0: below the tracked point)")
    ap.add_argument("--hs-texture-only", action="store_true",
                    help="leave the headstage out of the texture bake only (the white headstage otherwise bakes onto "
                         "the neck as a collar); the fit itself sees the full masks")
    ap.add_argument("--fix-scale", type=float, default=0.0,
                    help="hold the template's global scale at this value (a hunched rat otherwise shrinks the whole "
                         "template, the head kept full size by the masks, leaving too little torso)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    B, tc = al.table_frame_from_rig(build_initial_rig(str(CALIB)))
    up_C = np.asarray(B)[2]
    heads = {}
    for row in csv.DictReader(open(HEADS / f"rat{a.obj}" / "head_pose_world.csv")):
        if row["head_calib_x_m"]:
            heads[int(row["frame"])] = np.array([float(row[k]) for k in ("head_calib_x_m", "head_calib_y_m", "head_calib_z_m")])
    import placement as PL
    T = np.array(PL.S["T_from_calib"]); T_inv = np.linalg.inv(T)
    noses = {}
    if a.w_nose > 0 and not a.real_hs:
        n500 = np.load(HERE / "data" / f"rat{a.obj}_nose_f500_T.npy")
        off = PL.head_rotation(a.obj, 500).T @ (n500 - PL.tracked_head(a.obj, 500))     # in the head frame
        for f in parse_frames(a.frames):
            h, R = PL.tracked_head(a.obj, f), PL.head_rotation(a.obj, f)
            if h is not None and R is not None:
                nT = h + R @ off
                noses[f] = T_inv[:3, :3] @ nT + T_inv[:3, 3]                              # calib frame, as the fit
    HS_R, HS_TOP = 0.016, 0.033
    hs_c, hs_top = {}, {}
    if a.real_hs:
        tr = np.load(HERE / "data" / f"rat{a.obj}_headstage_track_T.npy", allow_pickle=True).item()
        trk = dict(zip(tr["frames"], tr["smooth"]))
        n500 = np.load(HERE / "data" / f"rat{a.obj}_nose_f500_T.npy")
        off_n = PL.head_rotation(a.obj, 500).T @ (n500 - trk[500])
        for f, P in trk.items():
            R = PL.head_rotation(a.obj, f)
            hs_c[f] = T_inv[:3, :3] @ P + T_inv[:3, 3]
            hs_top[f] = T_inv[:3, :3] @ (P - HS_TOP * R[:, 2]) + T_inv[:3, 3]
            noses[f] = T_inv[:3, :3] @ (P + R @ off_n) + T_inv[:3, 3]
    tv, tf = tfl.load_obj(str(TEMPLATE))
    x, z = tv[:, 0], tv[:, 2]
    reg = (x > np.percentile(x, 86)) & (x < np.percentile(x, 96))
    top_idx = torch.as_tensor(np.nonzero(reg & (z > np.percentile(z[reg], 70)))[0])
    nose = np.argsort(x)[-30:]
    nose_t = torch.as_tensor(nose)
    body = (x > np.percentile(x, 25)) & (x < np.percentile(x, 70))
    cur = {"f": None, "hs": {}}

    orig_init = tfl.init_candidates

    def init_candidates(template_verts, hull_verts):
        cands = orig_init(template_verts, hull_verts)
        head = heads.get(cur["f"], heads[min(heads, key=lambda k: abs(k - cur["f"]))])
        hc = hull_verts.mean(0)
        keep = [c for c in cands if (c[1] @ [0, 0, 1.0]) @ up_C > 0 and (c[1] @ [1.0, 0, 0]) @ (head - hc) > 0]
        keep = keep or [c for c in cands if (c[1] @ [0, 0, 1.0]) @ up_C > 0] or cands
        if a.fix_scale > 0:
            keep = [(a.fix_scale, *c[1:]) for c in keep]
        return keep
    tfl.init_candidates = init_candidates

    orig_loss = tfl._total_loss

    def total_loss(model, cams, prepped, hull_t, params, w_reg, corr=None, refresh=True):
        total, parts = orig_loss(model, cams, prepped, hull_t, params, w_reg, corr=corr, refresh=refresh)
        if a.w_head > 0 and (cur["f"] in hs_top if a.real_hs else cur["f"] in heads):
            pts = model()
            target = torch.as_tensor(hs_top[cur["f"]] if a.real_hs else heads[cur["f"]] + head_up * up_C, dtype=pts.dtype)
            d_cm = torch.linalg.norm(pts[top_idx].mean(0) - target) * 100.0
            h = 1.5 * (torch.sqrt(1 + (d_cm / 1.5) ** 2) - 1)          # pseudo-Huber, 1.5 cm scale
            total = total + a.w_head * h
            parts["head"] = float(d_cm)
        if a.real_hs and cur["f"] in hs_c:
            d = torch.linalg.norm(model() - torch.as_tensor(hs_c[cur["f"]], dtype=torch.float32), dim=1) * 100.0
            total = total + a.w_hsx * (torch.relu(HS_R * 100 + 0.3 - d) ** 2).sum() / 10.0
            parts["hs_inside"] = int((d < HS_R * 100).sum())
        if a.hs_box and cur.get("box") is not None:
            A, b = cur["box"]
            q = model() @ A.T + b
            lo, hi = torch.as_tensor(box_lo, dtype=q.dtype), torch.as_tensor(box_hi, dtype=q.dtype)
            depth = torch.minimum(q - lo, hi - q).min(dim=1).values * 100.0          # cm inside the box (>0)
            total = total + a.w_hsx * (torch.relu(depth + 0.3) ** 2).sum() / 10.0
            parts["hs_inside"] = int((depth > 0).sum())
        if a.w_nose > 0 and cur["f"] in noses:
            pts = model()
            d_cm = torch.linalg.norm(pts[nose_t].mean(0) - torch.as_tensor(noses[cur["f"]], dtype=pts.dtype)) * 100.0
            total = total + a.w_nose * 1.0 * (torch.sqrt(1 + d_cm ** 2) - 1)            # pseudo-Huber, 1 cm scale
            parts["nose"] = float(d_cm)
        return total, parts
    tfl._total_loss = total_loss

    cams = rl.load_calibration(str(CALIB))
    out_root = HERE / "variants" / a.name / f"object_{a.obj}"
    import cv2
    cap = PL.cap_points(20000)
    HS_ZMIN = -0.010
    if a.hs_box:
        cap = cap[cap[:, 2] > HS_ZMIN]                      # the box and connector, not the cement on the skull
    box_lo, box_hi = cap.min(0), cap.max(0)
    head_up = HEAD_UP if a.head_up is None else a.head_up

    from scipy import ndimage
    from scipy.spatial import cKDTree
    orig_prep = tfl.prepare_masks

    def prepare_masks(masks, coverage_samples=1500, seed=0):
        """GUI 2's preparation; coverage samples drawn from the mask minus the headstage (unknown pixels)."""
        out = orig_prep(masks, coverage_samples=coverage_samples, seed=seed)
        rng = np.random.default_rng(seed)
        for i, (m, pr) in enumerate(zip(masks, out)):
            hs = cur["hs"].get(i)
            if pr is None or hs is None:
                continue
            fg = (m > 0) & ~hs
            lab, n = ndimage.label(fg)
            if n > 1:
                fg = lab == (int(np.argmax(ndimage.sum(fg, lab, range(1, n + 1)))) + 1)
            ys, xs = np.nonzero(fg)
            if len(xs):
                pick = rng.choice(len(xs), size=min(coverage_samples, len(xs)), replace=False)
                pr["samples"] = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32); pr["area"] = int(fg.sum())
        return out
    if a.erase_headstage:
        tfl.prepare_masks = prepare_masks

    def headstage_points(f):
        M = PL.skull_matrix(a.obj, f)
        return None if M is None else (cap @ M[:3, :3].T + M[:3, 3]) @ T_inv[:3, :3].T + T_inv[:3, 3]

    def erase_headstage(colors, masks, f):
        cur["hs"] = {}
        if a.real_hs:
            if f not in hs_c:
                return []
            sph = np.random.default_rng(0).normal(size=(3000, 3)); sph /= np.linalg.norm(sph, axis=1)[:, None]
            P = hs_c[f] + HS_R * sph * np.random.default_rng(1).uniform(0, 1, (3000, 1)) ** (1 / 3)
            M = np.eye(4)
        else:
            M = PL.skull_matrix(a.obj, f)
        if M is None:
            return []
        if not a.real_hs:
            P = (cap @ M[:3, :3].T + M[:3, 3]) @ T_inv[:3, :3].T + T_inv[:3, 3]      # calib frame
        info, dbg = [], []
        for i, cam in enumerate(cams):
            if masks[i] is None:
                continue
            h, w = masks[i].shape[:2]
            px, ok = rl.project_points(P, cam["R"], cam["t"], cam["K"], cam["dist"])
            q = np.round(px[ok]).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)]
            hs = np.zeros((h, w), np.uint8); hs[q[:, 1], q[:, 0]] = 255
            hs = cv2.dilate(cv2.morphologyEx(cv2.dilate(hs, np.ones((7, 7), np.uint8)), cv2.MORPH_CLOSE,
                                             np.ones((15, 15), np.uint8)), np.ones((5, 5), np.uint8))
            before = int((masks[i] > 0).sum())
            if a.debug_masks:
                v = np.dstack([masks[i] > 0] * 3).astype(np.uint8) * 160; v[hs > 0] = (0, 0, 255)
                ys, xs = np.nonzero(masks[i] > 0)
                if len(xs):
                    y0, y1, x0, x1 = max(0, ys.min() - 40), ys.max() + 40, max(0, xs.min() - 40), xs.max() + 40
                    dbg.append(cv2.resize(v[y0:y1, x0:x1], (300, 300)))
            cur["hs"][i] = hs > 0
            if colors[i] is not None:
                colors[i][hs > 0] = 0
            info.append((cam["name"][-1], before, int(((masks[i] > 0) & (hs > 0)).sum())))
        if dbg:
            cv2.imwrite(str(HERE / "qa" / f"erase_headstage_{a.obj}_{f}.jpg"), np.hstack(dbg))
        return info
    state, report = None, {}
    for f in parse_frames(a.frames):
        t = time.time()
        hull = read_v(hull_path(a.obj, f, 2))
        if a.real_hs and f in hs_c:
            keep = np.linalg.norm(hull - hs_c[f], axis=1) > HS_R
            print(a.name, f, f"hull points inside the real headstage dropped: {int((~keep).sum())} of {len(hull)}", flush=True)
            hull = hull[keep]
        M = PL.skull_matrix(a.obj, f)
        if a.hs_box and M is not None:                     # calib -> headstage-box coordinates
            A = M[:3, :3].T @ T[:3, :3]; b = M[:3, :3].T @ (T[:3, 3] - M[:3, 3])
            cur["box"] = (torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))
        else:
            cur["box"] = None
        if a.hs_box and M is not None:
            q = hull @ cur["box"][0].numpy().T + cur["box"][1].numpy()
            keep = ~np.all((q > box_lo - 0.002) & (q < box_hi + 0.002), axis=1)
            print(a.name, f, f"hull points inside the headstage box dropped: {int((~keep).sum())} of {len(hull)}", flush=True)
            hull = hull[keep]
        elif a.erase_headstage and not a.real_hs and headstage_points(f) is not None:
            keep = cKDTree(headstage_points(f)).query(hull)[0] > 0.006
            print(a.name, f, f"hull points inside the headstage dropped: {int((~keep).sum())} of {len(hull)}", flush=True)
            hull = hull[keep]
        (_, mp), = rl.find_frames(str(MASKS), a.obj, "range", {"start": f, "end": f})
        colors, masks = rl.load_masks(rl.match_masks_to_cameras(cams, mp))
        if a.erase_headstage or a.hs_texture_only:
            info = erase_headstage(colors, masks, f)
            print(a.name, f, "headstage pixels removed per camera:", [(c, rem, f"{rem / max(b, 1) * 100:.0f}%") for c, b, rem in info], flush=True)
        cur["f"] = f
        extra = {"lr_scale": 1e-12} if a.fix_scale > 0 else {}
        fit = tfl.fit_template(tv, tf, cams, masks, hull, params={"iterations": a.iters, "n_nodes": a.n_nodes,
                                                                  "w_reg": a.w_reg, "sil_cameras": None, **extra},
                               log=lambda m: None, warm_start=state)
        state = fit["state"]
        V, Fc = fit["verts"], fit["faces"]
        uv, _ = rl.unwrap_cylindrical(V, Fc)
        bake_masks = [None if m is None else (np.where(cur.get("hs", {}).get(i, np.zeros(m.shape, bool)), 0, m).astype(m.dtype))
                      for i, m in enumerate(masks)]
        atlas = rl.bake_texture_atlas(V, Fc, uv, cams, colors, bake_masks, atlas_size=1536, log=lambda m: None)
        d = out_root / f"frame_{f:06d}"
        d.mkdir(parents=True, exist_ok=True)
        rl.save_textured_obj(d / "mesh.obj", V, Fc, uv, atlas)
        scale = np.median(np.linalg.norm(V - V.mean(0), axis=1)) / np.median(np.linalg.norm(tv - tv.mean(0), axis=1))
        ratio = np.linalg.norm(V[nose].mean(0) - V[body].mean(0)) / (np.linalg.norm(tv[nose].mean(0) - tv[body].mean(0)) * scale)
        head_d = float(np.linalg.norm(V[top_idx.numpy()].mean(0) - (heads[f] + HEAD_UP * up_C)) * 100) if f in heads else None
        report[f] = {"scale": round(float(scale), 3), "model_scale": round(float(torch.exp(state["log_scale"])), 3),
                     "head_body_ratio": round(float(ratio), 3), "head_top_to_tracked_cm": head_d,
                     "nose_to_target_cm": float(np.linalg.norm(V[nose].mean(0) - noses[f]) * 100) if f in noses else None,
                     "supported": fit["report"].get("supported_frac")}
        print(a.name, f, f"{time.time() - t:.0f} s", report[f], flush=True)
    json.dump(report, open(out_root / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
