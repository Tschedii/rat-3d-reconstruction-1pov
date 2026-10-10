"""Close-ups of a rat's head in the six raw camera views at one frame: the raw image (left) and, optionally, a fitted
mesh's head region projected on it (right; head vertices = template front 20 %), centred on the tracked head.

    python head_closeup.py <out.jpg> <rat> <frame> [mesh.obj] [--size 180]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_check as oc     # noqa: E402
import placement as pl         # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("rat", type=int); ap.add_argument("frame", type=int)
ap.add_argument("mesh", nargs="?"); ap.add_argument("--size", type=int, default=180)
a = ap.parse_args()
T0 = np.array([[float(x) for x in l.split()[1:4]] for l in open(Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])
head = T0[:, 0] > np.percentile(T0[:, 0], 80)
V = None
if a.mesh:
    V = np.array([[float(x) for x in l.split()[1:4]] for l in open(a.mesh) if l.startswith("v ")]) @ oc.T[:3, :3].T + oc.T[:3, 3]
hd = pl.tracked_head(a.rat, a.frame)
rows = []
for cam in oc.CAL:
    im = oc.frame_img(cam["name"], a.frame)
    c = oc.project(hd[None], cam)[0]
    C = a.size
    pad = cv2.copyMakeBorder(im, C, C, C, C, cv2.BORDER_CONSTANT)
    raw = pad[int(c[1]):int(c[1]) + 2 * C, int(c[0]):int(c[0]) + 2 * C].copy()
    tiles = [raw]
    if V is not None:
        ov = im.copy()
        oc.draw_points(ov, oc.project(V[head], cam), (255, 120, 40), 0.35)
        oc.draw_points(ov, oc.project(V[~head], cam), (180, 180, 180), 0.0)
        pad2 = cv2.copyMakeBorder(ov, C, C, C, C, cv2.BORDER_CONSTANT)
        tiles.append(pad2[int(c[1]):int(c[1]) + 2 * C, int(c[0]):int(c[0]) + 2 * C])
    t = np.hstack([cv2.resize(x, (360, 360)) for x in tiles])
    cv2.putText(t, f"cam {cam['name'][-1]}  frame {a.frame}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    rows.append(t)
grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, 6, 2)])
cv2.imwrite(a.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
