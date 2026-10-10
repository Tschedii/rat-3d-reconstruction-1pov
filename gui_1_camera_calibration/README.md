# GUI 1 — Camera Calibration

Checkerboard-based intrinsic + extrinsic calibration for a 6-camera rig.
Full walkthrough: see [Stage 1 in the root README](../README.md#stage-1-camera-calibration).

```bash
python server.py   # http://127.0.0.1:8971
```

| File | Role |
|---|---|
| `server.py` | HTTP API (`/api/run`, `/api/progress`, `/api/scan`, `/api/defaults`) + background pipeline thread |
| `calibration_lib.py` | Frame extraction, intrinsic solving, extrinsic averaging + bundle adjustment, rig plotting (OpenCV / SciPy) |
| `index.html` | Frontend — pick video folders, launch a run, watch live progress |

Writes to `output/` next to `server.py`; intermediate frames go to `work/`.

## Inputs: two video folders

Instead of six file paths you give one **folder** per video stream. Every
video file in it whose name contains a camera number is picked up:
`cam1.mp4`, `cam_01.mov`, `camera_001.mp4`, `Cam 3.avi` … all map to
`camera_001`, `camera_003`, etc. Files without a number are ignored and listed.

| Folder | Recording | Used by |
|---|---|---|
| intrinsic videos | one camera at a time, board swept across the whole image, many tilts and distances | Run intrinsics / Run both |
| extrinsic videos | all cameras **synchronously**, board visible to every camera, moved to several distances and tilts | Run extrinsics / Run both |

Leave the extrinsic folder empty to reuse the intrinsic videos for both (the
original single-stream workflow). Both folders must contain the same cameras.

**Pooling several intrinsic sessions.** The intrinsic field accepts several
folders separated by `;` (`.../Calibration_10_SEP; .../Calibration_18_SEP`).
Each camera's frames from all sessions are extracted into
`work/internal_parameter/<camera>/session_N/` and calibrated together, which
gives the solver more board coverage without a new recording. This is valid
as long as focus and aperture were not touched between sessions. With more
than one intrinsic folder the extrinsic folder must be set explicitly, since
the rig geometry has to come from one synchronous recording.

## Three run modes

| Button | Does | Writes |
|---|---|---|
| **Run intrinsics** | stage 1 only: diverse-frame extraction + per-camera intrinsic solve + cross-camera sanity check | `output/<name>_intrinsics.json` (path is filled into the "intrinsics json" field) |
| **Run extrinsics** | stages 2–4 only: synced frames from the extrinsic folder + rig solve + plots, intrinsics from the intrinsics json | `output/<name>.json`, rig PNGs |
| **Run both** | intrinsics then extrinsics | both of the above |

The intrinsics json field accepts either a `*_intrinsics.json` or a full
calibration json from an earlier run (older files that only stored `k1` load
with the other distortion terms as zero).

## Extrinsic solve

Stage 3 works in three steps, all on the same synced frame detections:

1. **Averaging** (the original method). For every synced frame set the board
   pose is solved per camera and the relative transform to the reference
   camera is averaged over all frames. This gives the initialisation and the
   per-frame scatter (`translation_std_m`, `rotation_spread_deg`).
2. **Bundle adjustment** (option, default on). All non-reference camera poses
   and all board poses are refined jointly so that the projected corners match
   the detections in every camera at once, with a Huber loss (1 px) so a few
   bad corners cannot pull the solve. A plain least-squares pass runs first,
   then the robust pass from that solution; a 6 × 80 frame problem takes about
   a second. With **refine fx** ticked, one focal scale per camera (fx and fy
   together, principal point and distortion fixed) is added. Treat it as a
   diagnostic: a shift above 0.5 % means the fixed intrinsics were limiting
   the rig solve, and `fx_before_bundle_adjustment` is kept in the json.
3. **Outlier rejection.** Frame sets whose worst-camera RMS error exceeds
   `max(frame_outlier_px, median + 3 MAD)` are dropped and steps 1–2 are
   repeated (up to 3 passes). A sync slip or a misdetection in one camera
   therefore removes that whole frame set. Dropped stems and their errors are
   listed in `dropped_frames`.

**Re-running without a new video.** Tick *reuse already extracted synced
frames* to skip stage 2 and re-solve on the frames left in
`work/external_parameter/` from the previous run (different reference camera,
outlier threshold, bundle adjustment on/off, fx refinement). No video decoding
is needed.

## Error metric

Every reprojection error is the **RMS distance per corner in pixels**,
`sqrt(mean(dx² + dy²))`, the same quantity `cv2.calibrateCamera` reports.
Earlier versions divided the L2 norm over all corners by the corner count (the
OpenCV tutorial idiom), which under-reports the per-corner error by
`sqrt(n_corners)`, about 7.3× for a 9×6 board. Numbers in json files written
before this change are not comparable: a reported 1.2 px was really about
9 px per corner. Thresholds in the GUI and the defaults below already use the
corrected scale. Guide values after bundle adjustment: below 0.5 px is good,
above 1 px means intrinsics or detections are limiting.

## Quality controls

Extrinsic frame selection (stage 2) drops near-duplicate board poses so a
board resting in front of the cameras counts as one pose, not a run of
identical frames:

| Parameter | Default | Meaning |
|---|---|---|
| `synced_min_frame_gap` | 30 | minimum video frames between two kept synced frame sets |
| `synced_min_pose_distance` | 0.15 | board must have rotated by this much (rad) OR ... |
| `synced_min_trans_distance` | 0.05 | ... moved by this much (m) relative to every kept pose (checked in the reference camera) |
| `outlier_threshold_px` | 1.0 | intrinsics: images above this RMS per corner are dropped and the solve repeated |
| `frame_outlier_px` | 3.0 | extrinsics: synced frame sets whose worst-camera RMS per corner exceeds this (or median + 3 MAD) are dropped |
| `bundle_adjust` | on | joint refinement of rig + board poses after averaging |
| `refine_focal` | off | also refine fx per camera inside the bundle adjustment |
| `reuse_synced_frames` | off | skip stage 2, re-solve on the frames already in `work/external_parameter/` |

The output JSON keeps the original schema and adds:

- `extrinsics_method` : `bundle_adjusted` or `averaged`
- `validation` : overall mean / RMS reprojection error of the final solve and of the averaged rig
- `bundle_adjustment` : RMS before / after, evaluations, solver status, `fx_change_pct` per camera
- `dropped_frames` : outlier frame sets with their worst-camera error
- `cameras[].consistency.reprojection_error_px` : per-camera mean / RMS / max and the worst frame (final solve)
- `cameras[].consistency.reprojection_error_averaged_rig_px` : the same for the averaged rig, for comparison
- `cameras[].intrinsics` : full `k1..k3, p1, p2`, stage-1 `rms_reprojection_error_px`, `model`, `num_images_used`, `fx_before_bundle_adjustment` when fx was refined
- `frame_diversity` : board distance and tilt range (seen from the reference camera) across the used frames
- `warnings` : intrinsics that disagree across cameras (identical lenses should agree in fx within ~3 %),
  too little distance / tilt variation, excessive per-frame translation scatter, residual error above 1 px after
  the solve, fx moving by more than 0.5 % in the bundle adjustment

The GUI shows the same numbers as a per-camera table under the rig plots, with
the averaged-rig and the final error side by side.
