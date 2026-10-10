"""Raw camera frame | the camera-view render laid over it, cropped around a rat's head, for every camera and every
tag given: rows = tags, column pairs = cameras.

    python camview_sheet.py <out.jpg> <rat> <frame> <tag1,tag2,...> [--size 170]
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
ap.add_argument("tags"); ap.add_argument("--size", type=int, default=170); ap.add_argument("--cams", default="1,2,3,4,5,6")
a = ap.parse_args()
hd = pl.tracked_head(a.rat, a.frame)
cams = [c for c in oc.CAL if c["name"][-1] in a.cams.split(",")]
raws = {c["name"][-1]: oc.frame_img(c["name"], a.frame) for c in cams}
rows = []
for tag in a.tags.split(","):
    row = []
    for c in cams:
        n = c["name"][-1]
        raw = raws[n]
        ren = cv2.imread(str(HERE / "variants" / f"camview_{tag}" / f"cam_{n}.png"), cv2.IMREAD_UNCHANGED)
        al = ren[:, :, 3:4].astype(np.float32) / 255
        comp = (raw * (1 - al) + ren[:, :, :3] * al).astype(np.uint8)
        ctr = oc.project(hd[None], c)[0] + np.array([0, 25])
        C = a.size
        crops = []
        for im in ((raw, comp) if tag == a.tags.split(",")[0] else (comp,)):
            pad = cv2.copyMakeBorder(im, C, C, C, C, cv2.BORDER_CONSTANT)
            crops.append(cv2.resize(pad[int(ctr[1]):int(ctr[1]) + 2 * C, int(ctr[0]):int(ctr[0]) + 2 * C], (300, 300)))
        t = np.hstack(crops)
        cv2.putText(t, f"{tag} cam {n}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        row.append(t)
    rows.append(row)
# first row has raw+comp per camera (600 px), later rows comp only (300 px): pad later rows to align under the comp
out_rows = [np.hstack(rows[0])]
for row in rows[1:]:
    out_rows.append(np.hstack([np.hstack([np.full((300, 300, 3), 255, np.uint8), t]) for t in row]))
cv2.imwrite(a.out, np.vstack(out_rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
