"""Contact sheets of the finished frames for review: qa/v2/<segment>_<n>.jpg, 3 x 3 frames each, labelled with the
video frame and recording frame.

    python qa_sheets.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SEG = {k: tuple(v) for k, v in json.load(open(HERE / "timeline.json"))["segments"].items()}
OUT = HERE / "qa" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {"a_rig_skulls": ["intro", "zoom", "skulls", "dim"],
          "b_reconstruction": ["seed", "block", "carve1", "carve2", "surface", "truecol", "template", "fur"],
          "c_motion_env": ["motion", "env"],
          "d_fov_flyin": ["fov", "flyin"],
          "e_fpv": ["fpv"]}
STEP = {"a_rig_skulls": 12, "b_reconstruction": 9, "c_motion_env": 6, "d_fov_flyin": 4, "e_fpv": 5}


def rec(vf):
    for k, (a, b) in SEG.items():
        if a <= vf < b:
            i = vf - a
            return {"skulls": 220 + i, "motion": 500 + i, "env": 650 + i, "fpv": 750 + i}.get(
                k, 220 if k in ("intro", "zoom") else (750 if k in ("fov", "flyin") else 500))
    return None


for g, segs in GROUPS.items():
    vfs = [vf for k in segs for vf in range(SEG[k][0], SEG[k][1], STEP[g])]
    for n in range(0, len(vfs), 9):
        sheet = Image.new("RGB", (3 * 640, 3 * 360), "white")
        for j, vf in enumerate(vfs[n:n + 9]):
            p = HERE / "frames_final" / f"vf_{vf:04d}.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB").resize((640, 360))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 340, 200, 360], fill=(0, 0, 0)); d.text((4, 344), f"vf {vf}  rec {rec(vf)}", fill=(255, 255, 0))
            sheet.paste(im, ((j % 3) * 640, (j // 3) * 360))
        sheet.save(OUT / f"{g}_{n // 9:02d}.jpg", quality=82)
print(sorted(p.name for p in OUT.glob("*.jpg")))
