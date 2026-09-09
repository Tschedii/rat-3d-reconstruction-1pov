# `data/`

This folder is the shared workspace between the four GUIs. It's empty on
a fresh clone (only `.gitkeep` files are tracked) — everything inside is
generated locally or dropped in by you, and stays out of git via
[`.gitignore`](../.gitignore).

```
data/
├── raw_videos/               → input:  Step 1
│   ├── camera_001/ … camera_006/    one synchronized video per rig camera
├── calibration/               ⇄ Step 1
│   ├── checkerboard_frames/  → intermediate: extracted checkerboard frames (auto-generated)
│   └── output/                → output: <name>.json (camera rig) + rig_perspective.png + rig_top_view.png
├── masks/                     → input:  Step 2 (from your segmentation tool of choice)
│   ├── rat_1/frame_000000/camera_001.png …                (layout A)
│   ├── frame_000000/cutouts/rat_2/camera_001.png …        (layout B, alternative)
│   └── camera_001/cutouts/rat_1/frame_000000.png …        (layout C, alternative)
├── meshes/
│   ├── rat/                   ⇄ Step 2 output → Step 4 input
│   │   └── object_<id>/frame_000000/mesh.obj
│   └── environment/           → input:  Step 3 (arena/box .obj + .mtl + textures)
├── eye_positions/              → output: Step 4 (eyeL/eyeR/head basis per frame, per object)
└── scenes/                     → output: Step 3 (exported scene placements / textured .obj)
```

## Camera videos — `raw_videos/`
Six sub-folders, one per camera, named to match the rig (`camera_001` …
`camera_006` by default — configurable in GUI 1). Drop one synchronized
video per camera into its folder.

## Calibration — `calibration/`
Populated by **GUI 1**. `checkerboard_frames/` holds the diverse/synced
frames it extracts for intrinsic and extrinsic calibration (safe to delete
between runs). `output/` holds the final rig file
(`{camera name, rotation_matrix, translation_m, intrinsics}` per camera)
plus two rig-visualization PNGs. This JSON is the calibration input every
other GUI depends on.

## Segmentation masks — `masks/`
Per-frame, per-camera silhouette masks for each tracked animal, produced by
your segmentation step (e.g. SAM). **GUI 2** auto-detects any of three
layouts:

- **A** — `masks/rat_<id>/frame_XXXXXX/<camera_name>.png`
- **B** — `masks/frame_XXXXXX/cutouts/rat_<id>/<camera_name>.png`
- **C** — `masks/<camera_name>/cutouts/rat_<id>/frame_XXXXXX.png`
  (camera folder outer, frame as the filename — e.g. `cam_1`, `cam_2`, …)

For layout C, camera folder names don't need to match the calibration
JSON's camera names exactly — `cam_1` matches a calibration camera named
`camera_001` by trailing index number if an exact/substring match isn't
found.

## Meshes — `meshes/`
- `rat/object_<id>/frame_XXXXXX/mesh.obj` — visual-hull mesh written by
  **GUI 2**. For **GUI 4** (eye marking), point it at a per-frame preview
  named `mesh_preview.ply` (a lightweight/decimated version of the same
  mesh works well for smooth interactive marking).
- `environment/` — your static scene mesh (arena, box, table — `.obj` /
  `.mtl` / textures), placed manually. Loaded by **GUI 3**.

## Eye positions — `eye_positions/`
`eye_positions_<object_label>.json`, written by **GUI 4**:
`{ frame_id: { eyeL, eyeR, headForward, headRight, headUp, source } }` —
the geometry behind the rat's-eye / first-person view.

## Scenes — `scenes/`
**GUI 3** writes textured scene exports to its own
`gui_3_scene_placement/exports/scene_<timestamp>/` folder (`scene.obj` +
`scene.mtl` + copied textures) — move the ones you want to keep into
`data/scenes/` for a tidy, versioned record of finalized placements.
