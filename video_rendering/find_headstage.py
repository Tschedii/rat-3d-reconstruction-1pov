"""Locate a rat's real headstage (the beige plastic box) at one frame: in each camera, beige pixels (hue 15-40 deg,
saturation 0.15-0.6, bright) within RADIUS px of the tracked head's projection, largest blob -> centroid; then a
least-squares triangulation over the cameras where it was found, with reprojection residuals. Writes
data/rat<r>_headstage_f<frame>_T.npy (table frame) and a check sheet.

    python find_headstage.py <rat> <frame>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_check as oc     # noqa: E402
import placement as pl         # noqa: E402

r, f = int(sys.argv[1]), int(sys.argv[2])
QUIET = "--quiet" in sys.argv
RADIUS = 90
hd = pl.tracked_head(r, f)
obs, tiles = {}, []
for cam in oc.CAL:
    im = oc.frame_img(cam["name"], f)
    c = oc.project(hd[None], cam)[0]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    beige = (h > 15) & (h < 45) & (s > 0.12) & (s < 0.65) & (v > 0.45)
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    near = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < RADIUS ** 2
    m = (beige & near).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    vis = im.copy()
    cv2.circle(vis, tuple(np.int32(c)), RADIUS, (255, 255, 255), 1)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] > 40:
            obs[cam["name"][-1]] = cen[k]
            vis[lab == k] = (0.5 * vis[lab == k] + [0, 0, 127]).astype(np.uint8)
            cv2.drawMarker(vis, tuple(np.int32(cen[k])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    tiles.append((cam, vis, c))
cams = {c["name"][-1]: c for c in oc.CAL}


def res(P):
    return np.concatenate([oc.project(P[None], cams[k])[0] - v for k, v in obs.items()])


sol = least_squares(res, hd.copy(), loss="soft_l1", f_scale=5.0)
P = sol.x
rr = res(P).reshape(-1, 2)
print("found in cams", list(obs), "; reprojection residual px:", {k: round(float(np.linalg.norm(e)), 1) for k, e in zip(obs, rr)})
print(f"headstage centre (table frame): {np.round(P, 4)}; {np.linalg.norm(P - hd) * 100:.1f} cm from the tracked head; "
      f"{(P[2] - pl.FLOOR_Z) * 100:.1f} cm above the floor")
np.save(HERE / "data" / f"rat{r}_headstage_f{f}_T.npy", P)
rows = []
for cam, vis, c in tiles:
    q = oc.project(P[None], cam)[0]
    cv2.drawMarker(vis, tuple(np.int32(q)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    C = 130; pad = cv2.copyMakeBorder(vis, C, C, C, C, cv2.BORDER_CONSTANT)
    t = cv2.resize(pad[int(c[1]):int(c[1]) + 2 * C, int(c[0]):int(c[0]) + 2 * C], (300, 300))
    cv2.putText(t, f"cam {cam['name'][-1]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    rows.append(t)
if not QUIET:
    cv2.imwrite(str(HERE / "qa" / f"headstage_find_{r}_{f}.jpg"), np.hstack(rows))
