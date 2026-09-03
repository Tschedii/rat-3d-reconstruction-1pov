# GUI 4 — Eye Marking + First-Person View

Manual 3D eye marking on a reconstructed mesh, producing the animal's head
basis (forward/right/up) needed for a first-person view. Full walkthrough:
see [Stage 4 in the root README](../README.md#stage-4--eye-marking--first-person-view).

```bash
python eye_marker_server.py   # http://127.0.0.1:8974
```

| File | Role |
|---|---|
| `eye_marker_server.py` | HTTP API — set source mesh(es), save marked eye positions |
| `index.html` | Frontend — load a mesh, click left/right eye, save |

Reads `frame_XXXXXX/mesh_preview.ply` from `data/meshes/rat/`, writes
`eye_positions_<object_label>.json` to `data/eye_positions/`.
