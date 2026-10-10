"""Texture the visual hull the frame-500 template fit actually used (refit/.../mesh_hull.obj) with GUI 2's own
unwrap and bake, so the reconstruction stage shows the hull the template was fitted to.

    python texture_hull.py <rat>   -> data/hull500_rat<n>/mesh.obj (+ mtl, texture)
"""
import sys
from pathlib import Path

DRIVE = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(DRIVE / "gui_2_rat_reconstruction"))
import reconstruction_lib as rl          # noqa: E402
import template_fit_lib as tfl           # noqa: E402

HERE = Path(__file__).resolve().parent
CALIB = DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json"
MASKS = DRIVE / "data/raw_data/Day_7_rPEG008_rPEG007_25-08-2026"
r = int(sys.argv[1])
V, F = tfl.load_obj(str(HERE / f"refit/object_{r}_template_fitted/frame_000500/mesh_hull.obj"))
cams = rl.load_calibration(str(CALIB))
(_, mp), = rl.find_frames(str(MASKS), r, "range", {"start": 500, "end": 500})
colors, masks = rl.load_masks(rl.match_masks_to_cameras(cams, mp))
uv, _ = rl.unwrap_cylindrical(V, F)
atlas = rl.bake_texture_atlas(V, F, uv, cams, colors, masks, atlas_size=1536, log=lambda m: None)
out = HERE / "data" / f"hull500_rat{r}" / "mesh.obj"
out.parent.mkdir(parents=True, exist_ok=True)
rl.save_textured_obj(out, V, F, uv, atlas)
print("wrote", out, len(V), "vertices")
