# Video rendering (Blender)

The pipeline behind [`rig_to_first_person.mp4`](https://raw.githubusercontent.com/Tschedii/rat-3d-reconstruction-1pov/master/docs/media/rig_to_first_person.mp4) (download),
a 55 s video of the whole 3D pipeline on Day 7 (frames 220 to 1000):

| Part | Shows |
|---|---|
| rig | the six calibrated cameras orbiting the arena, then a zoom onto the table |
| skulls | both rats' IMU skulls with their head paths, frames 220 to 500 (GUI 5.1 orientation, GUI 5.2 position) |
| reconstruction | GUI 2 at frame 500 step by step: seed point and rays, voxel carving per camera, surface, true colour, template fit, fur colours |
| motion | both rats moving, frames 500 to 650, with re-fitted upright templates |
| environment | the scanned room faded in at frame 650 (placement from GUI 4) |
| field of view | rat 1's eye cone at frame 750 |
| first person | rat 1's gimbal-stabilised first-person view, frames 750 to 1000, with rat 2 labelled when in view and a top-down map |

![Reconstruction steps](../docs/media/reconstruction_steps.gif)

## Run order

```bash
python prep_slide11.py              # rig, skulls, frame-500 reconstruction steps, meshes, room -> data/
python refit_fast.py <object> <first-last>   # upright template re-fits on GUI 2's hulls (~25 s/frame on CPU)
python smooth_fits.py <rat>         # temporal smoothing, checked against the six masks
python merge_refits.py              # keep the better of two re-fit rounds per frame (mask IoU)
python rigid_head.py <first> <last> # rat 2's head as a rigid template head posed from the cameras
blender -b -P render_slide11.py -- [first last | f1,f2,...]   # Eevee, frame by frame -> frames/
python finish_slide11.py            # captions, first-person crop, labels, encode -> MP4
```

`timeline.json` holds the frame ranges of the parts. `placement.py` (numpy
only, so Blender's Python can import it) places the IMU skulls and rat 1's
eye camera.

## Checks

Every geometric step was checked against the six raw camera views, not only
in the render:

| Script | Checks |
|---|---|
| `overlay_check.py` | projects the skull and/or a mesh into all six raw frames |
| `render_camviews.py` + `camview_sheet.py` | renders the textured rats through each calibrated camera over the raw frame |
| `landmarks_check.py`, `head_closeup.py` | template landmarks and head region on the raw images |
| `find_headstage.py`, `find_led.py` | triangulates the real headstage box and its LED (box to LED = head forward; agrees with the IMU nose axis to about 6°) |
| `mask_stats.py` | flags masks that are much larger than usual (wall reflections, merged rats), which pull a fit |
| `qa_jitter.py`, `qa_sheets.py` | shake of the first-person view, contact sheets of the finished frames |

`refit_upright.py`, `refit_variant.py`, `install_variant.py`, `render_variants.py`
and `template_mask.py` were used to try and install alternative fits;
`fur_gains.py` and `texture_hull.py` prepare the textures.

## Inputs

Paths are relative to the repository: the calibration
(`data/calibration/output/2026_09_21/`), the Day 7 masks
(`data/raw_data/Day_7_rPEG008_rPEG007_25-08-2026/`), GUI 2's output, the GUI 5.1
two-rat skull export, the room scan (`data/scenes/Environment_4/`), the cap
scan (`gui_5.1_quaternion_placement/assets/cap_head.stl`) and the brown-rat
template (`gui_2_rat_reconstruction/templates/rat_brown.obj`). The last two
are not in the repository (see the main README). Two inputs of
`prep_slide11.py` sit in `inputs/` next to the scripts: extra GUI 2 runs at
frame 500 (`inputs/recon/`) and the room placement (`inputs/env_transform.json`).

Lessons from building it, kept here because they cost time:

- Blender 5.1 Eevee: opaque materials use DITHERED, fades use BLENDED with
  `use_transparency_overlap = False`; blended discs drawn over blended paths
  give a green wash, so the arena disc stays opaque.
- Under the arena's green light white fur records yellow-green: white
  balance, green despill (`G <= (R+B)/2`) and a blue clamp in the shader.
- The room scan's cloth surface lies 1 to 2 cm above the calibrated floor, so
  rats and the eye camera are lifted by a ray-cast height.
