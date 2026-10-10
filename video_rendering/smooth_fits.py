"""Temporal smoothing of the re-fitted rats, checked against the six camera masks.

Every re-fit is the same template (rat_brown) deformed, so vertex i is the same point of the rat in every frame:
smoothing each vertex along time (Gaussian, SIGMA frames, within runs of consecutive frames) removes the
frame-to-frame jitter of independent fits without blurring real movement much. Each frame keeps its own texture.

The check: every mesh projected into each camera and compared with that rat's GUI 2 mask (IoU), for the raw fit
and the smoothed one; a frame keeps the smoothed vertices only if they fit the masks about as well (IoU drop
< MAX_DROP), otherwise the raw fit.

    python smooth_fits.py <rat> [--sigma 1.2]   -> refit_smooth/object_<n>/frame_XXXXXX/mesh.obj
                                                                      + logs/iou_rat<n>.json
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DRIVE = Path(__file__).resolve().parents[1]  # repository root
MASKS = Path.home() / "Desktop" / "Day_7_rPEG008_rPEG007_25-08-2026"
CAL = json.loads((DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json").read_text())["cameras"]
MAX_DROP = 0.01
SCALE = 0.5                     # masks compared at half resolution


def read_obj(path):
    V, F, rest = [], [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        else:
            rest.append(line)
            if line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V), np.array(F), rest


def write_obj(path, V, rest):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = [l for l in rest if l.startswith("mtllib")]
    body = [l for l in rest if not l.startswith("mtllib")]
    with open(path, "w") as fh:
        fh.writelines(head)
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V)
        fh.writelines(body)


def projector(cam):
    R = np.array(cam["rotation_matrix"]); t = np.array(cam["translation_m"]); k = cam["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]]) * np.array([[SCALE], [SCALE], [1]])
    dist = np.array([k.get(n, 0.0) for n in ("k1", "k2", "p1", "p2", "k3")])
    rv = cv2.Rodrigues(R)[0]
    return lambda P: cv2.projectPoints(P.reshape(-1, 1, 3), rv, t, K, dist)[0].reshape(-1, 2)


PROJ = [(c["name"], projector(c)) for c in CAL]


def iou(V, F, masks):
    out = []
    for (name, proj), m in zip(PROJ, masks):
        if m is None:
            continue
        uv = np.round(proj(V) * 8).astype(np.int32)          # 3 bits of sub-pixel precision
        img = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(img, list(uv[F]), 1, lineType=cv2.LINE_8, shift=3)
        inter = np.logical_and(img, m).sum(); union = np.logical_or(img, m).sum()
        out.append(inter / union if union else 0.0)
    return float(np.mean(out)) if out else float("nan")


FLAGS = json.load(open(Path(__file__).resolve().parent / "logs" / "mask_flags.json")) \
    if (Path(__file__).resolve().parent / "logs" / "mask_flags.json").exists() else {}


def load_masks(rat, f):
    """The rat's mask in each camera at half resolution; None where missing or flagged as bloated (mask_stats.py)."""
    ms = []
    for c in CAL:
        p = MASKS / f"cam_{int(c['name'][-1])}" / "masks_raw" / f"frame_{f:06d}.png"
        if not p.exists() or int(c["name"][-1]) in FLAGS.get(str(rat), {}).get(str(f), []):
            ms.append(None); continue
        m = np.asarray(Image.open(p)) == rat
        ms.append(cv2.resize(m.astype(np.uint8), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST).astype(bool))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rat", type=int); ap.add_argument("--sigma", type=float, default=1.2)
    a = ap.parse_args()
    src = HERE / "refit" / f"object_{a.rat}_template_fitted"
    dst = HERE / "refit_smooth" / f"object_{a.rat}"
    frames = sorted(int(p.name[6:]) for p in src.glob("frame_*") if (p / "mesh.obj").exists())
    data = {f: read_obj(src / f"frame_{f:06d}" / "mesh.obj") for f in frames}
    F = data[frames[0]][1]
    # runs of consecutive frames whose fits do not jump (mean vertex move < 2 cm)
    runs, cur = [], [frames[0]]
    for f0, f1 in zip(frames[:-1], frames[1:]):
        if f1 == f0 + 1 and np.linalg.norm(data[f1][0] - data[f0][0], axis=1).mean() < 0.02:
            cur.append(f1)
        else:
            runs.append(cur); cur = [f1]
    runs.append(cur)
    k = np.arange(-4, 5); w = np.exp(-0.5 * (k / a.sigma) ** 2)
    report = {}
    for run in runs:
        idx = {f: i for i, f in enumerate(run)}
        Vs = np.stack([data[f][0] for f in run])
        for i, f in enumerate(run):
            js = [(idx[f + d], wt) for d, wt in zip(k, w) if (f + d) in idx]
            ws = np.array([wt for _, wt in js])
            Vsm = (Vs[[j for j, _ in js]] * ws[:, None, None]).sum(0) / ws.sum()
            masks = load_masks(a.rat, f)
            i_raw, i_sm = iou(data[f][0], F, masks), iou(Vsm, F, masks)
            use = i_sm >= i_raw - MAX_DROP
            out = dst / f"frame_{f:06d}"
            write_obj(out / "mesh.obj", Vsm if use else data[f][0], data[f][2])
            for name in ("mesh.mtl", "mesh_texture.png"):            # always: a re-fit brings its own texture
                shutil.copy2(src / f"frame_{f:06d}" / name, out / name)
            report[f] = {"iou_raw": round(i_raw, 4), "iou_smooth": round(i_sm, 4), "smoothed": bool(use), "run": [run[0], run[-1]]}
    (HERE / "logs").mkdir(exist_ok=True)
    json.dump(report, open(HERE / "logs" / f"iou_rat{a.rat}.json", "w"), indent=0)
    r = np.array([v["iou_raw"] for v in report.values()]); s = np.array([v["iou_smooth"] for v in report.values()])
    print(f"rat {a.rat}: {len(report)} frames in {len(runs)} runs; mask IoU raw median {np.median(r):.3f} (min {r.min():.3f}), "
          f"smoothed median {np.median(s):.3f}; smoothed kept in {sum(v['smoothed'] for v in report.values())} frames")


if __name__ == "__main__":
    main()
