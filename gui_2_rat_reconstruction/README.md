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
| `reconstruction_lib.py` | Mask loading, automatic centroid triangulation, voxel carving, marching cubes, Taubin smoothing, cylindrical UV unwrap + multi-view texture-atlas baking |
| `index.html` | Frontend — pick calibration + masks, choose frame(s), launch, watch progress |

Reads from `data/calibration/output/*.json` + `data/masks/`, writes to
`data/meshes/rat/`.

**Output formats** (`texture_mode`, chosen in the UI):
- `atlas` (default) — `mesh.obj` + `mesh.mtl` + `mesh_texture.png`: a standard
  image-textured mesh, sharper than per-vertex colour and viewable in any
  OBJ-capable tool, macOS Preview/Quick Look included.
- `vertex_color` — a single `mesh.obj` with colour baked into a
  non-standard `v x y z r g b` extension per vertex; faster, but needs a
  viewer that understands that extension (MeshLab, CloudCompare, Blender,
  Open3D) — plain OBJ viewers render it as untextured geometry.
- `none` — geometry only.
