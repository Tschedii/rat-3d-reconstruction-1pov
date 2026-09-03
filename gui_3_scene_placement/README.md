# GUI 3 — Scene Placement

Interactive three.js viewer for composing the camera rig, environment mesh,
and reconstructed animal(s) into one placed 3D scene. Full walkthrough: see
[Stage 3 in the root README](../README.md#stage-3--scene-placement).

```bash
npm install && npm run build   # bundles src/app.js -> bundle.js
python env_marker_server.py    # http://127.0.0.1:8973
```

| File | Role |
|---|---|
| `env_marker_server.py` | HTTP API — session config, rig geometry from calibration JSON, textured OBJ export |
| `src/app.js` | three.js frontend source (orbit/transform controls, mesh loading, rig rendering) |
| `bundle.js` | built frontend bundle served to the browser (`npm run build` regenerates it) |
| `index.html` | Page shell |

Rebuild `bundle.js` after any change to `src/app.js`. Reads from
`data/calibration/`, `data/meshes/`, `data/eye_positions/`; writes scene
exports to `exports/` (git-ignored).
