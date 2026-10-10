"""The headstage LED (bright red-white spot at the box's front end) of a rat at one frame: per camera the brightest
reddish/white-hot blob within RADIUS px of the triangulated box centre, then a robust triangulation.

    python find_led.py <rat> <frame>     -> data/rat<r>_led_f<frame>_T.npy, qa/led_<r>_<f>.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overlay_check as oc     # noqa: E402

r, f = int(sys.argv[1]), int(sys.argv[2])
tr = np.load(HERE / "data" / f"rat{r}_headstage_track_T.npy", allow_pickle=True).item()
C3 = dict(zip(tr["frames"], tr["smooth"]))[f]
RADIUS = 60
obs, tiles = {}, []
for cam in oc.CAL:
    im = oc.frame_img(cam["name"], f)
    c = oc.project(C3[None], cam)[0]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    red = ((h < 20) | (h > 340)) & (s > 0.35) & (v > 0.55)
    hot = (v > 0.97) & (s < 0.25)
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    near = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < RADIUS ** 2
    score = (red | hot) & near
    m = cv2.dilate(score.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    vis = im.copy()
    if n > 1:
        # the blob with the most red pixels next to white-hot ones
        best, bs = None, 0
        for k in range(1, n):
            blob = lab == k
            sc = (red & blob).sum() + 3 * (hot & blob).sum()
            if sc > bs:
                best, bs = k, sc
        if bs >= 6:
            obs[cam["name"][-1]] = cen[best]
            cv2.drawMarker(vis, tuple(np.int32(cen[best])), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
    tiles.append((cam, vis, c))
cams = {c["name"][-1]: c for c in oc.CAL}
res = lambda P: np.concatenate([oc.project(P[None], cams[k])[0] - v for k, v in obs.items()])
sol = least_squares(res, C3.copy(), loss="soft_l1", f_scale=4.0)
L = sol.x
rr = res(L).reshape(-1, 2)
print("LED found in cams", list(obs), "; residual px:", {k: round(float(np.linalg.norm(e)), 1) for k, e in zip(obs, rr)})
d = L - C3
print(f"LED - box centre: {np.round(d * 100, 2)} cm, |d| = {np.linalg.norm(d) * 100:.2f} cm, pitch {np.degrees(np.arcsin(d[2] / np.linalg.norm(d))):.0f} deg")
np.save(HERE / "data" / f"rat{r}_led_f{f}_T.npy", L)
rows = []
for cam, vis, c in tiles:
    q = oc.project(L[None], cam)[0]
    cv2.drawMarker(vis, tuple(np.int32(q)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.drawMarker(vis, tuple(np.int32(c)), (255, 255, 0), cv2.MARKER_SQUARE, 10, 2)
    C = 80; pad = cv2.copyMakeBorder(vis, C, C, C, C, cv2.BORDER_CONSTANT)
    t = cv2.resize(pad[int(c[1]):int(c[1]) + 2 * C, int(c[0]):int(c[0]) + 2 * C], (300, 300))
    cv2.putText(t, f"cam {cam['name'][-1]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    rows.append(t)
if "--quiet" not in sys.argv:
    cv2.imwrite(str(HERE / "qa" / f"led_{r}_{f}.jpg"), np.hstack(rows))
