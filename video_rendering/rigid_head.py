"""Rat 2's head as the template's own rigid head, posed from what the cameras see, joined to the fitted body.

The deformation-graph fit squeezes the head: rat 2's headstage (a ~4 cm box on the skull) is labelled as the other
rat in several masks and pulls the template's skull up. Here the template head (front 20 %, with ears, eyes, snout)
keeps its shape and is placed from measurements:
  * the box centre and the LED at its front, triangulated in the six images (find_headstage.py, find_led.py),
    give the head's forward direction (agrees with the IMU nose axis to ~6 deg); roll from the IMU up axis;
  * the skull top sits D_TOP below the box centre;
  * a small optimisation (shift along forward / sideways / down, pitch, yaw, roll, head scale, all with priors)
    keeps the head inside the rats' silhouettes in all six cameras, covers rat 2's head pixels, and meets the fitted
    body's neck.
The final mesh: body from the installed fit, head rigid, a neck zone (template x percentile 70..80) blended.

    python rigid_head.py <first> <last>    -> variants/rigid/object_2/frame_XXXXXX/ + logs/rigid_head.json
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import placement as PL          # noqa: E402
import smooth_fits as sf        # noqa: E402

T = np.array(PL.S["T_from_calib"]); T_inv = np.linalg.inv(T)
T0 = np.array([[float(x) for x in l.split()[1:4]] for l in open(Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])
x, z = T0[:, 0], T0[:, 2]
pc = lambda q: np.percentile(x, q)
HEAD = x > pc(80)
W_HEAD = np.clip((x - pc(70)) / (pc(80) - pc(70)), 0, 1)
NECK = (x > pc(72)) & (x < pc(80))
top = lambda a, b: (x > pc(a)) & (x < pc(b)) & (z > np.percentile(z[(x > pc(a)) & (x < pc(b))], 70))
o_t = T0[top(86, 96)].mean(0)
f_t = T0[top(97, 99.5)].mean(0) - T0[top(82, 86)].mean(0); f_t /= np.linalg.norm(f_t)
u_t = np.array([0, 0, 1.0]) - f_t[2] * f_t; u_t /= np.linalg.norm(u_t)
B_t = np.column_stack([f_t, np.cross(u_t, f_t), u_t])
LOCAL = (T0 - o_t) @ B_t                     # template vertices in the template's head frame
D_TOP = 0.020
MASK_DIR = sf.MASKS
CAMS = [(c["name"][-1], proj) for c, (_, proj) in zip(sf.CAL, sf.PROJ)]
HS = np.load(HERE / "data/rat2_headstage_track_T.npy", allow_pickle=True).item()
LED = np.load(HERE / "data/rat2_led_track_T.npy", allow_pickle=True).item()
BOX_C = dict(zip(HS["frames"], HS["smooth"])); FWD = dict(zip(LED["frames"], LED["forward"]))
H_SUB = np.nonzero(HEAD)[0][::3]


def to_calib(P):
    return P @ T_inv[:3, :3].T + T_inv[:3, 3]


def frame_data(f):
    m1, m2 = sf.load_masks(1, f), sf.load_masks(2, f)
    c = BOX_C[f]; F = FWD[f]
    U = PL.head_rotation(2, f)[:, 2]; U = U - (U @ F) * F; U /= np.linalg.norm(U)
    L = np.cross(U, F)
    per = []
    for (name, proj), a, b in zip(CAMS, m1, m2):
        if b is None:
            continue
        allowed = b | (a if a is not None else False)
        cc = proj(to_calib(c[None]))[0]
        edge = proj(to_calib((c + 0.022 * L)[None]))[0]
        rb = np.linalg.norm(edge - cc) * 1.15                          # projected box radius (px)
        yy, xx = np.mgrid[:b.shape[0], :b.shape[1]]
        box = (xx - cc[0]) ** 2 + (yy - cc[1]) ** 2 < rb ** 2
        allowed = allowed | box
        dt = cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 3)
        hc = proj(to_calib((c - 0.022 * U + 0.01 * F)[None]))[0]       # rough head centre
        rh = np.linalg.norm(proj(to_calib((c - 0.022 * U + 0.01 * F + 0.028 * L)[None]))[0] - hc)
        roi = b & ~box & ((xx - hc[0]) ** 2 + (yy - hc[1]) ** 2 < rh ** 2)
        ys, xs = np.nonzero(roi)
        samp = np.column_stack([xs, ys])[np.random.default_rng(0).choice(len(xs), min(300, len(xs)), replace=False)] if len(xs) else None
        per.append((name, proj, dt, samp))
    return c, F, L, U, per


def pose(params, c, F, L, U, scale):
    d_top, s_f, s_l, dp, dy, dr, ls = params
    Rs = Rotation.from_euler("zyx", [dy, dp, dr], degrees=True).as_matrix()
    B = np.column_stack([F, L, U]) @ Rs
    centre = c - d_top * U + s_f * F + s_l * L
    return centre + (scale * np.exp(ls)) * LOCAL @ B.T


def cost(params, data, body_T, scale, parts=False):
    c, F, L, U, per = data
    V = pose(params, c, F, L, U, scale)
    Vh = to_calib(V[H_SUB])
    out_c, cov_c = [], []
    for name, proj, dt, samp in per:
        uv = proj(Vh)
        h, w = dt.shape
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        d = np.full(len(uv), 30.0); q = np.round(uv[ok]).astype(int); d[ok] = dt[q[:, 1], q[:, 0]]
        out_c.append(np.mean(np.minimum(d, 30.0)))
        if samp is not None and len(samp) > 10:
            dd, _ = cKDTree(uv).query(samp)
            cov_c.append(np.mean(np.minimum(dd, 15.0)))
    neck = np.linalg.norm(V[NECK] - body_T[NECK], axis=1).mean() * 100
    d_top, s_f, s_l, dp, dy, dr, ls = params
    prior = ((d_top - D_TOP) / 0.007) ** 2 + (s_f / 0.007) ** 2 + (s_l / 0.005) ** 2 + (dp / 15) ** 2 + (dy / 12) ** 2 + (dr / 12) ** 2 + (ls / 0.1) ** 2
    terms = {"outside_px": float(np.mean(out_c)), "coverage_px": float(np.mean(cov_c)) if cov_c else 0.0, "neck_cm": float(neck), "prior": float(prior)}
    total = 1.0 * terms["outside_px"] + 0.4 * terms["coverage_px"] + 1.0 * terms["neck_cm"] + 0.5 * terms["prior"]
    return (total, terms) if parts else total


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    src = HERE / "refit" / "object_2_template_fitted"
    dst = HERE / "variants" / "rigid" / "object_2"
    x0 = np.array([D_TOP, 0, 0, 0, 0, 0, 0.0])
    report = {}
    for f in range(a, b + 1):
        Vc, Fc, rest = sf.read_obj(src / f"frame_{f:06d}" / "mesh.obj")
        body_T = Vc @ T[:3, :3].T + T[:3, 3]
        scale = np.median(np.linalg.norm(body_T - body_T.mean(0), axis=1)) / np.median(np.linalg.norm(T0 - T0.mean(0), axis=1))
        data = frame_data(f)
        before = cost(x0, data, body_T, scale, parts=True)
        r = minimize(cost, x0, args=(data, body_T, scale), method="Powell",
                     options={"xtol": 1e-4, "ftol": 1e-3, "maxfev": 3000})
        after = cost(r.x, data, body_T, scale, parts=True)
        V = pose(r.x, *data[:4], scale)
        Vout = (1 - W_HEAD[:, None]) * body_T + W_HEAD[:, None] * V
        d = dst / f"frame_{f:06d}"; d.mkdir(parents=True, exist_ok=True)
        sf.write_obj(d / "mesh.obj", to_calib(Vout), rest)
        for n in ("mesh.mtl", "mesh_texture.png"):
            shutil.copy2(src / f"frame_{f:06d}" / n, d / n)
        report[f] = {"params": [round(float(v), 4) for v in r.x], "before": before[1], "after": after[1]}
        print(f, "d_top %.1f cm, shift fwd %.1f side %.1f cm, pitch %+.0f yaw %+.0f roll %+.0f deg, scale x%.2f | outside %.1f -> %.1f px, coverage %.1f -> %.1f px, neck %.1f -> %.1f cm"
              % (r.x[0] * 100, r.x[1] * 100, r.x[2] * 100, r.x[3], r.x[4], r.x[5], np.exp(r.x[6]), before[1]["outside_px"], after[1]["outside_px"],
                 before[1]["coverage_px"], after[1]["coverage_px"], before[1]["neck_cm"], after[1]["neck_cm"]), flush=True)
        x0 = r.x
    json.dump(report, open(HERE / "logs" / "rigid_head.json", "w"), indent=1)


if __name__ == "__main__":
    main()
