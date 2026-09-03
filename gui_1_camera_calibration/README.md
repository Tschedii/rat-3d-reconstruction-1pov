# GUI 1 — Camera Calibration

Checkerboard-based intrinsic + extrinsic calibration for a 6-camera rig.
Full walkthrough: see [Stage 1 in the root README](../README.md#stage-1--camera-calibration).

```bash
python server.py   # http://127.0.0.1:8971
```

| File | Role |
|---|---|
| `server.py` | HTTP API (`/api/run`, `/api/progress`, `/api/defaults`) + background pipeline thread |
| `calibration_lib.py` | Frame extraction, intrinsic/extrinsic solving, rig plotting (OpenCV) |
| `index.html` | Frontend — configure cameras, launch the run, watch live progress |

Reads from `data/raw_videos/`, writes to `data/calibration/`.
