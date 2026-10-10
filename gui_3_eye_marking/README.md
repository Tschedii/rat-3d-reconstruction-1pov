# GUI 3 — Eye Marking + First-Person View

Manual 3D eye marking on a reconstructed mesh, producing the animal's head
basis (forward/right/up) needed for a first-person view. Full walkthrough:
see [Stage 3 in the root README](../README.md#stage-3-eye-marking-and-field-of-view).

```bash
python eye_marker_server.py   # http://127.0.0.1:8974
```

| File | Role |
|---|---|
| `eye_marker_server.py` | HTTP API — set source mesh(es), save marked eye positions |
| `index.html` | Frontend — a hand-rolled WebGL viewer (no three.js): parses PLY or OBJ, loads a texture atlas if the OBJ has one, click left/right eye, save |

Reads `frame_XXXXXX/mesh.obj` (GUI 2's default output) or the older
`frame_XXXXXX/mesh_preview.ply` from `data/meshes/rat/` — mesh.obj is
preferred when both exist for a frame. An atlas-textured OBJ (`mesh.obj` +
`mesh.mtl` + `mesh_texture.png`) is rendered with the real baked texture;
a vertex-colour OBJ or a PLY falls back to per-vertex colour; a plain OBJ
with neither renders as flat-shaded grey geometry. Writes
`eye_positions_<object_label>.json` to `data/eye_positions/`.
