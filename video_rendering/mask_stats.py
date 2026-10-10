"""Per-camera mask areas of both rats (frames 480..1000) and outlier flags: a camera whose mask of a rat is much
larger than that camera's running median (a reflection in the arena wall or the other rat merged into it)
pulls the template fit there through the coverage term.  -> logs/mask_stats.json"""
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MASKS = Path.home() / "Desktop" / "Day_7_rPEG008_rPEG007_25-08-2026"
FR = range(480, 1001)


def stats(f):
    out = {}
    for c in range(1, 7):
        p = MASKS / f"cam_{c}" / "masks_raw" / f"frame_{f:06d}.png"
        if not p.exists():
            continue
        m = np.asarray(Image.open(p))
        for r in (1, 2):
            b = (m == r).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(b)
            areas = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
            hull_a = 0
            if b.any():
                pts = cv2.findNonZero(b); hull_a = float(cv2.contourArea(cv2.convexHull(pts)))
            out[f"{r}_{c}"] = [int(b.sum()), int(areas.max()), int(n - 1), hull_a]
    return f, out


if __name__ == "__main__":
    with Pool(8) as p:
        res = dict(p.map(stats, FR))
    json.dump(res, open("logs/mask_stats.json", "w"))
    flags = {}
    for r in (1, 2):
        for c in range(1, 7):
            A = np.array([res[f].get(f"{r}_{c}", [np.nan])[0] for f in FR], float)
            for i, f in enumerate(FR):
                w = A[max(0, i - 40): i + 41]
                med = np.nanmedian(w)
                if A[i] > 1.45 * med:
                    flags.setdefault(r, {}).setdefault(f, []).append(c)
    json.dump(flags, open("logs/mask_flags.json", "w"))
    import itertools
    for r in (1, 2):
        fl = flags.get(r, {})
        for c in range(1, 7):
            fs = [f for f in fl if c in fl[f]]
            runs = [[g[0][1], g[-1][1]] for _, g in ((k, list(g)) for k, g in itertools.groupby(enumerate(fs), lambda t: t[1] - t[0]))]
            print(f"rat {r} cam {c}: {len(fs)} bloated frames {runs}")
