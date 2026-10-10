# GUI 4 — Scene Placement

Interactive three.js viewer for composing the camera rig, environment mesh,
and reconstructed animal(s) into one placed 3D scene. Full walkthrough: see
[Stage 4 in the root README](../README.md#stage-4-scene-placement).

```bash
npm install && npm run build   # bundles src/app.js -> bundle.js
python env_marker_server.py    # http://127.0.0.1:8973
```

| File | Role |
|---|---|
| `env_marker_server.py` | HTTP API — session config, rig geometry from calibration JSON, mesh inspection, eye storage, textured OBJ + screenshot export |
| `src/app.js` | three.js frontend source (orbit/transform controls, mesh loading, rig rendering, first-person view) |
| `bundle.js` | built frontend bundle served to the browser (`npm run build` regenerates it) |
| `index.html` | Page shell |

Rebuild `bundle.js` after any change to `src/app.js`. Reads from
`data/calibration/`, `data/meshes/`, `data/eye_positions/`; writes scene
exports to `exports/` and first-person renders to `exports/fpv/`
(git-ignored).

## Workflow

The panel unlocks in three stages, each gated by a confirm button.

1. **Camera rig.** Place the whole rig (`0`) coarsely, fine-tune individual
   cameras (`1`-`6`), then *Confirm camera position*.
2. **Meshes.** Tick object 1 / object 2. Each mesh loads with or without a
   sidecar `.mtl` (a geometry-only export is fine, it just renders untextured).
   The readout under the checkboxes reports, for the actual files configured,
   the vertex count, real-world size, and bounding-box centre, plus whether the
   mesh still carries reconstruction-frame world coordinates or has been
   re-centred on its own origin. Adjust with the gizmo if needed, then
   *Confirm rat placement*.
3. **First-person view.** Pick a rat and an eye, set the render field of view,
   *Preview through eye* to look through it live (Esc to leave), *Screenshot*
   to write a 1600x1200 PNG plus a sidecar JSON recording rat, eye, FOV, mesh
   path, eye source, and placement mode.

### Placement frame

- **as reconstructed** (default) applies the full calibration-frame to
  rig-frame transform, so a mesh that is still in world coordinates lands in
  its true position and the two animals keep their real separation. Use this
  whenever the readout says *looks like world coordinates*.
- **separate side by side** drops each mesh on the table origin with a
  display-only lateral offset. Use it when the meshes have been re-centred, in
  which case only their size is meaningful and their relative position is not.

Neither mode registers the animals against the environment mesh: the arena is
modelled independently, so aligning it to the rig stays a manual step.

### Eye positions

The field of view and the first-person cameras need the two eye positions in
the mesh's own frame. They come from the eye-marking tool via the
*eye positions JSON* fields in setup. If none exists yet, *Mark eyes on mesh*
lets you click the left then the right eye directly on the surface; the result
is written to `eye_positions/<objectN>.json` here and reused on reload. The
head-forward axis is seeded from the longer horizontal bounding-box extent and
re-orthogonalised against the inter-eye axis, the same heuristic the
eye-marking tool uses, so it is a coarse estimate rather than a measurement.

The rendered first-person image is an ordinary pinhole view. A rat's real
monocular field is roughly 200°, which no single perspective projection can
represent, so the FOV slider tops out well below that and the saved sidecar
JSON records the value used.
