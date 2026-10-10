# BEAST3D on the six-camera rat rig (Day 7)

Scripts that turn a segmented rig session into a BEAST3D training dataset
(https://github.com/paninski-lab/beast, `docs/beast3d.md`), plus checks of the
calibration and of camera sync.

## Files

| File | What it does |
|---|---|
| `rig.py` | Shared helpers: reads the GUI 1 calibration JSON, projection, triangulation, visual-hull carving |
| `check_calibration.py` | Does the calibration fit the session? Leave-one-out visual hull + centroid reprojection |
| `estimate_sync.py` | Per-camera frame offsets over the session, from mask-centroid triangulation |
| `make_beast3d_dataset.py` | Writes the BEAST3D dataset folder (crops, masks, per-frame cameras, `info.json`) |
| `beast3d_day7.yaml` | The repo's `configs/multiview/beast3d.yaml` adapted to this rig (changes marked `Day 7`) |

## Why not `beast extract_3d`

It needs an anipose TOML per session and the raw video of every camera; only three of the
six Day 7 videos are on the analysis laptop, but SAM cutouts and masks exist for all six. More
importantly, the BEAST3D loader:

- resizes every image to 256 x 256 and never uses the stored bbox, so full 1920 x 1200 frames
  would shrink the rats about 8x;
- ignores lens distortion;
- keeps mask pixels with grey value > 0.5, which turns rat 1 in the palette `masks_raw`
  (grey 125/255) into background.

`make_beast3d_dataset.py` therefore crops each view to a square around the rats, resamples it
from the undistorted image, shifts and scales the intrinsics to match, and writes 0/255 masks.

## Day 7 findings (2026-10-07)

**Calibration `2026_09_21.json` fits Day 7** (recorded 25 Aug), from `check_calibration.py` on
24 frames spread over the session:

| | cam_1 | cam_2 | cam_3 | cam_4 | cam_5 | cam_6 |
|---|---|---|---|---|---|---|
| Leave-one-out hull recall | 0.87 | 0.90 | 0.88 | 0.90 | 0.87 | 0.88 |
| Leave-one-out hull IoU | 0.75 | 0.74 | 0.73 | 0.74 | 0.71 | 0.72 |
| Rat-centroid reprojection (px) | 16 | 14 | 15 | 12 | 17 | 13 |

Frame 1 fails only because cam_6's mask covers the floor there; the converter rejects frames
with a mask over 15% of the image. Misses sit mostly at heads and tails.

**The cameras drift out of sync.** Up to about frame 6000 (3.3 min) every test agrees within
about ±1 frame. Later, cam_2, cam_3 and cam_5 fall about 3 to 7 frames behind cam_1 and
cam_4 (up to about 330 ms between cameras). The original recordings had different lengths
(`masks_raw` runs past the 18,001-frame cutouts): cam_1, cam_4, cam_6 18,007 frames,
cam_2 and cam_5 18,002, cam_3 18,104. The two cameras with five fewer frames are the ones that
lag, which fits dropped frames. No cutout is a byte-identical copy of the previous frame, so
the drops were not padded with duplicates. `estimate_sync.py` is too noisy to locate the drops
(most windows have too little motion), so `sync_offsets.csv` should not be used for
correction yet. Per-frame timestamps from the acquisition software would settle it.

This also affects anything else that combines cameras frame by frame after about frame 6000
(visual hull, triangulated head tracking).

The dataset in `/path/to/beast3d_day7` is therefore built from frames 0 to 6000 only.

## Run

```bash
python check_calibration.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 --calib ../data/calibration/output/2026_09_21/2026_09_21.json
python estimate_sync.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 --calib ../data/calibration/output/2026_09_21/2026_09_21.json
python make_beast3d_dataset.py --src /path/to/Day_7_rPEG008_rPEG007_25-08-2026 \
    --calib ../data/calibration/output/2026_09_21/2026_09_21.json --out /path/to/beast3d_day7 \
    --frames 0:6000 --pool-step 2 --n-frames 2000
```

`--sync-csv` takes per-camera offsets (format of `sync_offsets.csv`) once reliable ones exist.
`--animals 1` or `--animals 2` builds a one-rat dataset from the `object_N` cutouts.

Then on a Linux GPU machine with BEAST installed (and access to
`facebook/dinov3-vitb16-pretrain-lvd1689m` granted on Hugging Face):

```bash
beast train --config beast3d_day7.yaml --data /path/to/beast3d_day7 --output runs/beast3d_day7
```

The validation split is the last 10% of frames in time (BEAST sorts frame ids).
