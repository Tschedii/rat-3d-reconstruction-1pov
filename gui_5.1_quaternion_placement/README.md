# GUI 5.1: IMU head pose (two rats)

Head pose of one or both rats over time, in the table frame: **position from
the cameras, orientation from the head IMU**. Shows a top-down video of the
arena with each rat's head marker, nose arrow and trail, a 3D inset with the
cap scan per rat, and exports both as MP4 plus one trajectory CSV per rat.
Rat 1 is object 1 in the masks (red), rat 2 is object 2 (blue).

```bash
python server.py               # http://127.0.0.1:8975  (--port or $GUI5_PORT to change)
```

`bundle.js` is already built. Only after changing `src/app.js`:

```bash
npm install && npm run build   # bundles src/app.js -> bundle.js (three.js)
```

Needs Python with numpy, opencv-python and Pillow, and ffmpeg for the video
exports (libx264, libopenh264, h264_videotoolbox or mpeg4, whichever works).

| File | Role |
|---|---|
| `server.py` | HTTP API: session config, top-down frames, per-rat analysis in the background, head clicks, video + trajectory export |
| `analysis_core.py` | the maths, one rat per call (`compute(config)` also runs headless) |
| `topview.py` | ground-plane homography from a rig camera to the top-down view |
| `gui4_rig.py` | copy of GUI 4's rig helpers (table frame); used when `gui_4_scene_placement` is not next to this folder |
| `src/app.js` | three.js frontend: top view, head markers and trails, skull insets, rat picker (keys 1 / 2) |
| `bundle.js`, `index.html` | built frontend and page shell |
| `assets/cap_head.stl` | scan of the head cap, **not in the repository** (128 MB). Without it the insets show axes only |

## Method

- **Position.** Per camera and frame, the rat's mask gives a silhouette
  centroid (tail removed by a morphological opening) and a long axis. The
  centroids of all cameras that see the rat are triangulated (DLT on
  undistorted points, the worst camera dropped while its reprojection error is
  large and at least three remain), so rearing keeps its height. The 2D axes
  are fused into one 3D body axis.
- **Orientation.** The IMU frame is gravity-referenced, so pitch and roll are
  real and only its heading drifts:
  `R_W<-H(j) = Rz(psi_j) R(q_j) R_S<-H`. Every `anchor_every_frames` frames
  (default 100) `psi` is re-estimated as the circular mean of body heading
  minus IMU heading, and interpolated in between. The body axis is a line;
  its nose/tail sign comes from the direction of travel in each walking bout.
- **Head point.** One anchor frame per rat (its GUI 2 mesh and GUI 3 eyes)
  gives the body-to-eyes vector in the head frame, carried to every frame.
- **Sensor mounting.** On this rig the IMU's +x points to the tail
  (`sensor_forward = -x`, `sensor_up = +z`), plus an optional mount tilt and
  manual head clicks that correct the heading where the cameras clearly
  disagree.

On Day 7, `rPGE_007` belongs to object 1 and `rPGE008` to object 2: the
held-out correlation of IMU heading and walking direction is 0.84 and 0.77,
against 0.54 and 0.48 with the files swapped.

## Inputs (`session_config.json`, written by the setup form)

| Key | Meaning |
|---|---|
| `calib_json_path` | rig JSON from GUI 1 |
| `video_root` | raw videos `cam_N.mp4` |
| `view_mode` | `median` (per-pixel median of all cameras' top-down warps) or one camera name |
| `masks_root` | `cam_N/masks_raw/frame_XXXXXX.png` (default: `video_root`) |
| `anchor_every_frames` | heading re-anchoring interval, 0 = eyes only |
| `rats[i].quat_csv_path` | head IMU quaternions, one row per video frame (`qw..qz` or `qc_w..qc_z`, or the MATLAB `.mat`) |
| `rats[i].anchor_mesh_dir`, `eyes_json_path` | GUI 2 `frame_XXXXXX/` folder and GUI 3 eyes of the anchor frame |
| `rats[i].sensor_forward`, `sensor_up`, `imu_offset_frames`, `mount_*`, `head_clicks` | mounting and timing corrections |

A single-rat config from an earlier version (keys at the top level) is read as
the rat with that `mask_label`.

## Exports (`exports/`)

`<recording>_video_<first>-<last>_<stamp>.mp4` (top-down video),
`<recording>_skull_<first>-<last>_{both,rat1,rat2}_<stamp>.mp4` (rotating
skulls, read by GUI 5.3) and one trajectory CSV per rat.
