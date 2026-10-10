"""One white balance for both rats' fur textures: the median, over textures sampled every 12th frame of both
rats, of the linear gains that bring the brightest 10 % of fur pixels to white. Under the arena's green light
the cameras record white fur yellow-green (blue suppressed about 4x); a single texture's gains vary from 2.4x
to 6.8x in blue, so per-texture balancing would flicker.

    python fur_gains.py   -> data/fur_gains.json
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
M = json.load(open(HERE / "data/meshes.json"))
lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
G = []
for r in (1, 2):
    for f in sorted(int(f) for f in M[str(r)] if 500 <= int(f) <= (750 if r == 1 else 1000))[::12]:
        im = np.asarray(Image.open(M[str(r)][str(f)].replace("mesh.obj", "mesh_texture.png")).convert("RGB").resize((384, 384))) / 255.0
        px = im.reshape(-1, 3); lum = px @ [0.2126, 0.7152, 0.0722]
        px, lum = px[lum > 0.05], lum[lum > 0.05]
        G.append(lin(0.92) / lin(px[lum >= np.percentile(lum, 90)]).mean(0))
g = np.median(np.array(G), 0)
json.dump({"gains": [round(float(x), 3) for x in g], "n_textures": len(G)}, open(HERE / "data/fur_gains.json", "w"))
print("fur gains", g.round(2), "from", len(G), "textures")
