# GUI 2 — Rat Mesh Reconstruction

Visual-hull (shape-from-silhouette) mesh reconstruction from 6 synchronized
segmentation masks + a rig calibration. Full walkthrough: see
[Stage 2 in the root README](../README.md#stage-2--rat-mesh-reconstruction).

```bash
python server.py   # http://127.0.0.1:8972
```

| File | Role |
|---|---|
| `server.py` | HTTP API (`/api/run`, `/api/progress`, `/api/defaults`) + background pipeline thread |
| `reconstruction_lib.py` | Mask loading, automatic centroid triangulation, voxel carving, marching cubes |
| `index.html` | Frontend — pick calibration + masks, choose frame(s), launch, watch progress |

Reads from `data/calibration/output/*.json` + `data/masks/`, writes to
`data/meshes/rat/`.
