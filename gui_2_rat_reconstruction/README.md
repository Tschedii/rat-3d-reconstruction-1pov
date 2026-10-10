# GUI 2 — Rat Mesh Reconstruction

Visual-hull (shape-from-silhouette) mesh reconstruction from 6 synchronized
segmentation masks + a rig calibration. Full walkthrough: see
[Stage 2 in the root README](../README.md#stage-2-rat-mesh-reconstruction).

```bash
python server.py   # http://127.0.0.1:8972
```

| File | Role |
|---|---|
| `server.py` | HTTP API (`/api/run`, `/api/progress`, `/api/defaults`) + background pipeline thread |
| `reconstruction_lib.py` | Mask loading, automatic centroid triangulation, voxel carving (with an optional relaxed-threshold head zone), marching cubes, Taubin smoothing, cylindrical UV unwrap + multi-view texture-atlas baking |
| `template_fit_lib.py` | Model-based completion: deforms a rat template (embedded deformation graph, PyTorch) to the silhouettes + hull so minority-visible parts (the head) aren't carved away |
| `make_rat_template.py` | Generates the procedural `templates/rat_template.obj` (SDF primitives → marching cubes); can also emit posed variants |
| `prepare_template.py` | Converts any rat model (GLB/GLTF/OBJ/PLY) into a fitter-ready template: welds seams, keeps body + tail, strips whisker strands (tube-vs-sheet test, so ears survive), optionally replaces a curled tail by a straight tapered tube of the same length and taper, decimates, re-orients to +X nose / +Z up and scales to rat size (needs `trimesh` + `open3d`) |
| `templates/` | `rat_template.obj` (procedural, made by `make_rat_template.py`). The thesis runs used `rat_brown.obj`, a third-party brown-rat model prepared with `prepare_template.py` (body + straight tail, ~11k verts); it is not redistributed here. Put your own prepared model at `templates/rat_brown.obj` and it becomes the default again |
| `diagnose_camera_coverage.py` | Per-camera diagnostic: how much of a reconstruction's two ends (head/tail) each mask covers, with overlay images |
| `index.html` | Frontend — pick calibration + masks, choose frame(s), launch, watch progress |

Reads from `data/calibration/output/*.json` + `data/masks/`, writes to
`data/meshes/rat/`.

**Completion** (`completion_mode`, chosen in the UI): `none` writes the visual
hull; `template` fits a complete rat mesh to the same data and writes that as
`mesh.obj`, keeping the raw hull alongside as `mesh_hull.obj`. Needs
`pip install torch` (CPU build is enough).

**Output formats** (`texture_mode`, chosen in the UI):
- `atlas` (default) — `mesh.obj` + `mesh.mtl` + `mesh_texture.png`: a standard
  image-textured mesh, sharper than per-vertex colour and viewable in any
  OBJ-capable tool, macOS Preview/Quick Look included.
- `vertex_color` — a single `mesh.obj` with colour baked into a
  non-standard `v x y z r g b` extension per vertex; faster, but needs a
  viewer that understands that extension (MeshLab, CloudCompare, Blender,
  Open3D) — plain OBJ viewers render it as untextured geometry.
- `none` — geometry only.
