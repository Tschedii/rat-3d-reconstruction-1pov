"""What the template adds to the hull at frame 500, as a clean per-vertex mask for the template stage: how far
each vertex of the template shown (scene.json recon template) lies from the visual hull shown (nearest hull
vertex), recomputed here so it always matches the meshes the renderer loads, smoothed over the mesh, kept where it
is more than THR, small islands (< MIN_VERTS) dropped, then one smoothing pass for a soft edge.

    python template_mask.py   -> data/template_mask_rat{1,2}.npy
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "data/scene.json"))
THR, MIN_VERTS = 0.006, 50


def read(p):
    V, F = [], []
    for l in open(p):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    return np.array(V), np.array(F)


import sys
# optional: template_mask.py <rat> <template.obj> <out.npy>   (a variant fit; scene.json's hull)
JOBS = [(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))] if len(sys.argv) > 3 else \
    [(r, S["recon"][str(r)]["template_textured"], HERE / f"data/template_mask_rat{r}.npy") for r in (1, 2)]
for r, tpath, out in JOBS:
    V, F = read(tpath)
    Vh, _ = read(S["recon"][str(r)]["hull_textured"])
    a = cKDTree(Vh).query(V)[0]
    if len(sys.argv) <= 3:
        np.save(HERE / f"data/template_added_rat{r}.npy", a.astype(np.float32))
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    n = len(V)
    A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    v = a.astype(float)
    for _ in range(4):
        v = 0.5 * v + 0.5 * (A @ v) / deg
    m = v > THR
    idx = np.nonzero(m)[0]
    nc, lab = connected_components(A[m][:, m], directed=False)
    sizes = np.bincount(lab)
    m[idx[sizes[lab] < MIN_VERTS]] = False
    soft = m.astype(float)
    soft = 0.5 * soft + 0.5 * (A @ soft) / deg
    np.save(out, soft.astype(np.float32))
    print(f"rat {r}: {m.mean() * 100:.1f}% of the template marked as added ({(sizes >= MIN_VERTS).sum()} regions)")
