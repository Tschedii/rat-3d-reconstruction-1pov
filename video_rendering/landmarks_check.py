"""Template landmarks (tail base, hip, mid, shoulder, neck, nose) of a fitted rat drawn on the six raw camera images,
with the hull's outline, to see whether the fit covers the real body front to back.

    python landmarks_check.py <out.jpg> <rat> <frame> <mesh.obj> [<mesh2.obj> ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_check as oc     # noqa: E402

out, r, f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
meshes = sys.argv[4:]
T0 = np.array([[float(x) for x in l.split()[1:4]] for l in open(Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])
x = T0[:, 0]; pc = lambda a, b: (x >= np.percentile(x, a)) & (x < np.percentile(x, b))
G = [("tail", pc(18, 22), (0, 200, 255)), ("hip", pc(30, 36), (0, 255, 0)), ("mid", pc(48, 54), (255, 255, 0)),
     ("shoulder", pc(66, 72), (255, 0, 255)), ("neck", pc(78, 82), (0, 128, 255)), ("nose", x >= np.percentile(x, 99.7), (0, 0, 255))]
readv = lambda p: np.array([[float(v) for v in l.split()[1:4]] for l in open(p) if l.startswith("v ")])
hull = readv(json.load(open(HERE / "data/scene.json"))["recon"][str(r)]["hull_textured"])
rows = []
for mi, mp in enumerate(meshes):
    V = readv(mp)
    row = []
    for cam in oc.CAL:
        im = oc.frame_img(cam["name"], f)
        oc.draw_points(im, oc.project(hull @ oc.T[:3, :3].T + oc.T[:3, 3], cam), (200, 200, 200), 0.0)
        pts = [oc.project((V[m].mean(0) @ oc.T[:3, :3].T + oc.T[:3, 3])[None], cam)[0] for _, m, _ in G]
        for (a, _, _), (b, _, _), pa, pb in zip(G[:-1], G[1:], pts[:-1], pts[1:]):
            cv2.line(im, tuple(np.int32(pa)), tuple(np.int32(pb)), (255, 255, 255), 3)
        for (name, _, col), p in zip(G, pts):
            cv2.circle(im, tuple(np.int32(p)), 9, col, -1); cv2.circle(im, tuple(np.int32(p)), 9, (0, 0, 0), 2)
        c = np.mean(pts, axis=0)
        C = 260; pad = cv2.copyMakeBorder(im, C, C, C, C, cv2.BORDER_CONSTANT)
        crop = cv2.resize(pad[int(c[1]):int(c[1]) + 2 * C, int(c[0]):int(c[0]) + 2 * C], (300, 300))
        cv2.putText(crop, f"{Path(mp).parts[-3] if 'variants' in mp else 'now'} cam {cam['name'][-1]}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        row.append(crop)
    rows.append(np.hstack(row))
legend = np.full((26, rows[0].shape[1], 3), 30, np.uint8)
for i, (name, _, col) in enumerate(G):
    cv2.circle(legend, (20 + i * 150, 13), 7, col, -1); cv2.putText(legend, name, (32 + i * 150, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(out, np.vstack([legend] + rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
