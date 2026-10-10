# Three-panel video

One video of the tracked rat (or of both rats, see *Two rats* below), three
panels side by side:

| panel | content | from |
|---|---|---|
| left | the raw camera view, both rats tinted from their masks (tracked rat red, the other blue, as in GUI 5), the tracked rat's head (orange dot) and its last few seconds of head path | the camera video, GUI 2 masks, head tracking export |
| top right | the rotating IMU skull | GUI 5's skull video export |
| bottom right | the head path in 3D (table frame, height above the arena floor), drawn up to the current frame, with the current head, a dotted drop line showing its height, the nose direction, the body centroid path and the path's shadow on the floor; the view orbits slowly | head tracking export |

```bash
python server.py               # http://127.0.0.1:8981
```

or without the page: `python render.py config.json`, with the same keys the
form saves to `session_config.json`. Needs Python with numpy, opencv-python,
Pillow and matplotlib, and ffmpeg (any H.264 encoder it has: libx264,
h264_videotoolbox or libopenh264 are tried in that order).

## Two rats

Give the head tracking field a GUI 5.2 two-rat run folder (the one holding
`rat1/` and `rat2/`) and the skull field GUI 5.1's side-by-side skull export
(kind *skulls only, side by side*, `..._skull_<first>-<last>_both_....mp4`).
Rat n is object n in the masks and keeps its GUI 5.1 / 5.2 colour everywhere,
rat 1 red and rat 2 blue:

| panel | content |
|---|---|
| left | both rats tinted (object 1 red, object 2 blue), each rat's head as a dot in its colour (white rim) and its last few seconds of head path |
| top right | the two skulls side by side, cut from the skull export (rat 1 left, rat 2 right) |
| bottom right | both head paths in 3D in their colours, body centroid paths in lighter versions of them; this panel is as wide as the two skulls |

A single-skull export (`..._skull_<a>-<b>_rat1_...`) also works: the other
rat's square then says it is not in the skull video. *Check inputs* shows
which square of the skull video each rat gets.

## Inputs

- **Camera video**: the raw `cam_N.mp4`; `N` picks the rig camera
  (`camera_00N` in the calibration). Video frame n is recording frame n.
- **Rig calibration JSON** (GUI 1): projects the head into that camera.
- **Masks root** (optional): the folder with `cam_N/masks_raw`; without it the
  rats are not tinted. The tracked rat is the head tracking export's mask
  label.
- **Head tracking export**: a GUI 5.2 head tracking export folder
  (`head_pose_world.csv`, plus `summary.json` for the fps and
  `alignment.json` for the mask label), or a two-rat run folder holding one
  such folder per rat (`rat1/`, `rat2/`).
- **Skull video**: GUI 5's `..._skull_<first>-<last>_....mp4` export. Its
  first frame is taken from the file name unless set.
- **Frames**: blank = every frame all three inputs cover (*Check inputs*
  shows each input's range).
- **Output**: default `~/Desktop/three_panel_<video>_<first>-<last>.mp4`.

The look: video height (the right column is two squares of half that
height; the camera panel keeps its aspect), head trail length, tint strength,
arena floor height in the table frame (default 5.5 cm above the table plane:
GUI 5 estimated 5.49 cm for this rig when building its top view), 3D view
orbit speed and elevation.

## Notes

- Frames without a head position (no masks for that frame, or the head
  tracking could not find it) show no head marker, and the 3D path has a gap
  there.
- The height axis of the 3D panel is stretched relative to x and y (the box
  is always 1 : 1 : 0.32, whatever the ranges; for frames 0 to 1400 of Day 7
  that is about 1.6 times), so that head height changes of a few cm stay
  visible next to a path tens of cm across.
