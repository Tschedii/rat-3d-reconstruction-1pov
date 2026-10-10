# Head tracking

Tracks the head (and body position and heading) of one or both rats from
every rig camera's GUI 2 segmentation mask, with no head-mounted IMU. Rat 1 is
object 1 in the masks and is drawn in red (`#ff2828`), rat 2 is object 2 and
is drawn in blue (`#286eff`), the same colours GUI 5.1 tints the masks with.
Per camera and frame, the mask's silhouette gives a 2D centroid, a long axis,
and (from where the tail attaches) where the head is; triangulating these
across cameras gives the 3D body centroid, the 3D heading and the 3D head.
Each rat is tracked on its own (one run of the same computation per mask
label). A rat's own GUI 2 mesh is used only as a fallback for its head when
the masks lose it for several frames in a row. Roll (rotation about the nose
axis) cannot be seen in a silhouette and is always reported as exactly 0.

```bash
python server.py               # http://127.0.0.1:8978
```

`bundle.js` is already built. Only after changing `src/app.js`:

```bash
npm install && npm run build   # bundles src/app.js -> bundle.js (three.js)
```

Needs Python with numpy, scipy, matplotlib and Pillow. On first start the
setup form asks for the paths below and saves them to `session_config.json`
next to `server.py`.

| File | Role |
|---|---|
| `server.py` | HTTP API: session config, frame lists, mesh serving, background analysis (one run per tracked rat) with progress polling |
| `gui4_rig.py` | copy of GUI 4's rig helpers (table frame, camera rig); used when `gui_4_scene_placement` is not next to this folder |
| `analysis_lib.py` | the maths: `mask_heading` (silhouette centroid and long axis), `mask_head` / `head_candidates` / `track_head` (head from the masks, tracked over time), `mesh_head` (head fallback from a mesh), `triangulate_point` / `triangulate_direction` (multi-camera fusion), `analyse_from_masks` (kinematics, state segmentation, one rat), `rat_distances` / `run_summary` / `make_two_rat_plots` (the two rats side by side), CSV + PNG output |
| `src/app.js` | three.js frontend: two rats at any frame for visual context, tinted red / blue (no gizmo, no per-frame placement), a frame-range picker, each tracked rat's path and a head marker that follows the frame scrubber, results panel |
| `bundle.js` | built frontend served to the browser (`npm run build` regenerates it) |
| `index.html` | page shell + setup form |

Reads the rig JSON from GUI 1 (`data/calibration/output/`), every rig
camera's masks from `masks_root` (`cam_N/masks_raw/frame_XXXXXX.png`, label
values matching GUI 2's per-object numbering), and optionally GUI 2's
per-frame meshes (`rat1_root` / `rat2_root`: the 3D view, and each tracked
rat's own one as its head fallback) and the raw per-camera frames (for the
re-projection check). Writes one folder per run under `exports/`, with one
subfolder per tracked rat.

## Inputs

- **Calibration JSON**: defines the world frame. All positions are reported
  in the *table frame* (the table-aligned display frame from GUI 4: z up,
  origin at the table centre, metres), plus the raw calibration-frame
  coordinates for completeness.
- **Masks root**: folder holding `cam_N/masks_raw/frame_XXXXXX.png` for
  every rig camera that has one; every camera found there is used.
- **Rats to track**: ticked per run in the panel's analysis section (*track
  rat 1* / *track rat 2*, both on by default). Rat N is mask label N (GUI 2's
  per-object numbering: rat 1 = object 1, rat 2 = object 2). The older
  `mask_label` setting in `session_config.json` is kept: a request to
  `/api/analyze` that names no `rats` (as before two-rat tracking) tracks
  that one label, as it always did.
- **fps**: rows per second of video.
- **Object 1 / object 2 mesh roots**: optional. Folders holding
  `frame_XXXXXX/` subfolders. Both are shown in the 3D view, tinted in their
  rat's colour; each is also its own rat's head fallback described below
  (object 1 for rat 1, object 2 for rat 2). Without it, that rat's long head
  gaps are only interpolated.
- **Camera images root**: optional. Root of the raw per-camera frames, laid
  out as `cam_<n>/all_objects/frame_XXXXXX.png` (the `<n>` matching each rig
  camera's `camera_00<n>` name). When set, the analysis re-projects the head
  path onto each camera's frame at the *present / reference frame* as a
  visual check against the real footage (`camera_overlays.png`); a camera
  missing that frame's image is drawn as an empty panel rather than failing
  the whole run.

## Workflow

1. **Meshes (optional).** Pick the frame each rat is shown at, to orient
   yourself in the 3D view before choosing a frame range. The frame shown has
   no effect on the analysis.
2. **Analysis window.** Set the first/last frame to analyse (there is no
   anchor frame: every frame in the range gets its own independent mask
   fusion), a *present / reference frame* for the past/future path panel and
   the camera overlay (blank = the first frame), and the mask-fusion/state
   thresholds (minimum per-camera confidence to trust a heading, rest
   thresholds, smoothing windows, max gap to interpolate), and tick which
   rats to track. *Run analysis* runs `analyse_from_masks` once per ticked
   rat in a background thread (so two rats take about twice as long);
   progress streams into the panel and the plots appear on the right (click
   to enlarge): the two rats together first, then each tracked rat's own
   summary and plots, headed with its colour. Each rat's head path is also
   drawn in 3D in its colour, with its body centroid path in a lighter
   version of the same colour; *colour paths by* switches the head paths to
   behavioural state instead.
3. **Playback.** After an analysis the frame scrubber covers the analysed
   window and starts at the present / reference frame. Scrubbing or playing
   moves a marker per rat along its path: a sphere in the rat's colour at
   its head and an arrow in the nose direction (where the heading is known).
   The meshes stay at the frame picked in step 1.

## What the computation means

**Body and heading.** Per frame and camera, `mask_heading` finds the mask's
centroid and PCA long axis. Across cameras, `triangulate_point` gives the
body's 3D position (from every detection) and `triangulate_direction` the 3D
heading axis (from cameras clearing the confidence threshold); see that
function's docstring for the derivation, since fusing 2D headings into a 3D
direction is a different problem from ordinary point triangulation. Yaw and
pitch come from the fused direction; roll is not recoverable from a
silhouette and is always exactly 0, with "up" taken as the closest
table-vertical direction perpendicular to the heading rather than a
measurement.

**Head, from the masks.** Per camera, `mask_head` strips the tail off the
silhouette and takes the body point farthest from the tail base, measured
through the body rather than straight across the image, so a curled or turning
rat is followed along its own shape. The head point is the mean of a small
snout-tip region there. Only a long, thin piece of the silhouette counts as the
tail (at least 3 times longer than wide and 15% of the body's length): the
headstage sticking out above the head, a paw or an ear also falls off in the
opening, and taking one of those for the tail puts the head on the rump. A
camera that sees no tail (occluded, merged into the other rat's mask, curled
under) offers both body ends instead.

`head_candidates` triangulates every pair of per-camera options and keeps
every 3D point at least `min_head_cams` (default 2) cameras agree on, with how
many cameras support it (a camera that saw the tail counts double) and how
many contradict it (none of their options anywhere near). `track_head` then
picks one candidate per frame, or none, for all frames at once (a Viterbi
search): support counts for a candidate, contradicting cameras and a head
closer than 4 cm or farther than 15 cm from the body centroid count against
it, and moving the head costs more the farther it would jump in the time
available. So when the two rats are in contact and the cameras split, the
head stays with the candidate that continues the track instead of hopping to
whichever point most cameras happen to agree on in that one frame. A detection
more than `max_head_jump_m` (default 8 cm) from the median of the detections
within 2 frames either side is still dropped as a one-frame mistake.

Before this tracker (fuse_head's best guess per frame, at least 3 cameras),
rat 2 on Day 7 frames 220 to 1400 had 23 head jumps of more than 5 cm in one
frame, 30 frames with the head within 4 cm of the body centroid, 40 frames that
fewer than 2 cameras agreed with, and lost the head for 60 frames in a row
(1095 to 1154). With it: no jump over 5 cm, the head found in all but a few
frames, and no frame that 3 or more cameras contradict; rat 1, already clean,
also lost its two jumps over 5 cm.

**Head, when the masks lose it.** A gap of fewer than
`mesh_fallback_min_gap_frames` (default 3) frames is interpolated. For a
longer gap, every frame in it that has a mesh gets the mesh's head
(`mesh_head`): the meshes have no fixed vertex layout, so this is the tip of
the mesh's long horizontal axis at whichever end is nearer to where the mask
detections either side of the gap put the head, rejected if more than
`mesh_head_max_dist_m` (default 10 cm) away (which also rejects a mesh that
latched onto the other rat). The rest of the gap is interpolated between
these anchors, up to *max gap to interpolate*.

**Which end is the nose.** The fused heading is only an axis until a sign is
picked. Wherever the masks found the head, the heading points from the body
centroid towards it. Elsewhere, it is kept continuous with the previous frame
(no 180° flips within one frame) and, while the body centroid is clearly
moving across the table (horizontal speed above
`min_speed_for_heading_check_mps`, default 0.06 m/s, over a ~0.1 s window),
not allowed to point backwards against that direction of travel.

## Outputs (`exports/masks_<timestamp>_rats<n…>_<first>-<last>/`)

The run folder is named after the tracked rats (`rats12` for both, `rats1` or
`rats2` for one). It holds one subfolder per tracked rat plus the files that
put the rats side by side:

| File | Content |
|---|---|
| `rat1/`, `rat2/` | each tracked rat's own export: exactly the files of a single-rat run, listed below (same CSV columns, same `summary.json` / `alignment.json` keys, `mask_label` in `alignment.json`), so a tool that reads one rat's export (e.g. GUI 5.3) is pointed at the subfolder |
| `path_top_view_both.png` | (two rats only) both rats' head paths in one top view (rat 1 red, rat 2 blue, body centroid paths lighter, each head at the present / reference frame marked with a nose tick), and the head-to-head and body-to-body distance between the rats over time |
| `camera_overlays.png` | (only if `camera_image_root` is configured) every tracked rat's head path re-projected onto each raw camera frame at the reference frame, in the rat's colour (past in full colour, future lighter, present as a dot with a white nose tick), as a visual check against the real footage |
| `run_summary.json` | which rats were tracked and their colours, the window, fps and reference frame, a few headline numbers per rat (head sources, path length, speed, time share per state; the rest is in each rat's `summary.json`), and (two rats) `between_rats_m`: median / mean / min / max head-to-head and body-to-body distance |

Each `rat<n>/` subfolder:

| File | Content |
|---|---|
| `head_pose_world.csv` | per frame: time, `pos_source` (body: mask / interp / empty), body and head position (table frame), orientation quaternion in the world (roll always 0), yaw / pitch / roll, nose and up vectors, angular velocity, angular speed and acceleration, head velocity, linear speed and acceleration, behavioural state, the same positions in the calibration frame, and `head_source` (mask / mesh / interp / empty) |
| `state_bouts.csv`, `rest_bouts.csv` | one row per bout: state, start/end frame, start/end time, duration |
| `summary.json` | coverage, the analysed window, path length, speed statistics, time share per state, rest-bout statistics, thresholds, `mask_fusion` (mean cameras used for position/heading/head, fraction of frames with a heading, how many frames got the head from the masks / the mesh / interpolation / not at all, which head detections were dropped as jumps, the mesh root used), notes |
| `alignment.json` | how the run was produced: mask label, masks root, head fallback mesh root, min confidence, the analysed window, the reference frame, table-frame definition |
| `path_top_view.png` | path coloured by time (light to dark in the rat's colour), by behavioural state, and by past/future of the chosen *present / reference frame* |
| `head_orientation.png` | yaw / pitch / roll over time in the rat's colour (roll is always the zero line), rest bouts shaded |
| `head_kinematics.png` | angular speed, angular acceleration, linear speed, linear acceleration in the rat's colour, thresholds drawn |
| `states.png` | ethogram of rest / head movement / locomotion and the time share of each |

**States.** *Rest*: angular speed below the threshold and, when position is
available, linear speed below its threshold. *Locomotion*: linear speed
above threshold. *Head movement*: everything else. Runs shorter than *min
state duration* are merged into their neighbours. Angular quantities are
Savitzky–Golay smoothed over *orientation smoothing* (only across frames with
a known heading, never bridged across a gap); positions over *position
smoothing*.

## Where accuracy actually comes from

Every frame is independent: there is no anchor to drift away from. The head
depends on the tail being visible in at least two cameras (to know which end
is the head) and on enough cameras agreeing; on the Day 7 recording, frames
240 to 900, that held in about 98% of frames, with the head within roughly
1 to 2 cm across cameras. That figure is rat 1's; for rat 2, frames 240 to
540, the masks found the head in 77% of frames (231 of 301), the rest coming
from its mesh (53) or interpolation (17). The two rats touching or
overlapping is the main failure mode: masks lose the tail or merge, and the
mesh fallback is least reliable exactly then (the old mesh-centroid approach tracked the wrong rat
during contact), which is why a mesh head is only accepted near where the
mask detections either side of the gap put it. The two rats are tracked
independently (each from its own mask label, with its own mesh as fallback),
so one rat's gaps never borrow from the other's track; the head-to-head
distance in `path_top_view_both.png` shows when they were close enough for
this to matter.
