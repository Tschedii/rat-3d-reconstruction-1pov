"""Captions, legend and frame counter on render_slide11.py's frames, the first-person part cut to GUI 4's
4:3 eye camera with rat 2 labelled whenever it is in view and a top-down map beside it, then the video.

    python finish_slide11.py [--preview f1,f2,...]   -> slide11_rig_to_first_person.mp4
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FR = HERE / "frames"
DONE = HERE / "frames_final"
DONE.mkdir(exist_ok=True)
TL = json.load(open(HERE / "timeline.json"))
SEG = {k: tuple(v) for k, v in TL["segments"].items()}
POSES = json.load(open(HERE / "data" / "skull_poses.json"))
W, H = 1920, 1080
FPV_W = int(round(H * 4 / 3))                     # 1440: GUI 4's 4:3 eye camera, 120 deg vertical
RAT = {1: (255, 40, 40), 2: (40, 110, 255)}
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
f_cap = ImageFont.truetype(FONT, 34)
f_sub = ImageFont.truetype(FONT, 25)
f_small = ImageFont.truetype(FONT, 22)


def seg_of(vf):
    for k, (a, b) in SEG.items():
        if a <= vf < b:
            return k, (vf - a) / max(1, b - a - 1), vf - a
    return "fpv", 1.0, 0


def rec_frame(vf):
    k, u, i = seg_of(vf)
    return {"intro": 220, "zoom": 220, "skulls": 220 + i, "motion": 500 + i, "env": 650 + i, "fov": 750, "flyin": 750,
            "fpv": 750 + i}.get(k, 500)


def caption(vf):
    k, u, i = seg_of(vf)
    f = rec_frame(vf)
    if k in ("intro", "zoom"):
        return "A calibrated six-camera rig", "solve of 21 September 2026, in the table frame"
    if k == "skulls":
        return "Head IMUs of both rats, frames 220 to 500", "head orientation from each rat's IMU, head position from the cameras"
    if k == "dim":
        return "Paused at frame 500", "reconstructing both rats from the six silhouettes (GUI 2)"
    if k == "seed":
        return "1  Seed point", "the rays through each camera's mask centroid meet at the rat"
    if k == "block":
        return "2  Voxel grid around the seed", "shown at 1 cm; the reconstruction uses 2 mm voxels"
    if k in ("carve1", "carve2"):
        r = 1 if k == "carve1" else 2
        cam = min(6, int(u * 6) + 1) if k == "carve1" else min(6, max(1, int((u - 0.15) / 0.85 * 6) + 1))
        return f"3  Voxel carving, rat {r}: camera {cam} of 6", "a voxel stays if at least 4 of the 6 silhouettes contain it"
    if k == "surface":
        return "4  Surface", "marching cubes on the 2 mm voxels (the visual hull)"
    if k == "truecol":
        return "5  True colours", "projected onto the surface from the camera images"
    if k == "template":
        return "6  Template fit", "red / blue: what the fitted rat template adds to the visual hull"
    if k == "fur":
        return "7  Fur colours", "the template-fitted rats, textured from the cameras"
    if k == "motion":
        return "Frames 500 to 650", "both rats reconstructed in every frame"
    if k == "env":
        return ("Frame 650: Environment 4 added", "the scanned room, placed around the rig as in GUI 4") if f < 700 else \
            ("Frames 650 to 750", "the rats moving inside the scanned room")
    if k == "fov":
        return "Paused at frame 750: rat 1's field of view", "GUI 4 eye camera, 120 deg vertical x 133 deg horizontal"
    if k == "flyin":
        return "Into rat 1's eyes", ""
    return "Rat 1, first-person view, frames 750 to 1000", "head orientation from rat 1's IMU; rat 2 in blue when in view"


def box_text(d, xy, lines, fonts, pad=14, fill=(20, 18, 14, 165), colors=None):
    x, y = xy
    widths = [d.textlength(t, font=fn) for t, fn in zip(lines, fonts) if t]
    hts = [fn.size + 8 for t, fn in zip(lines, fonts) if t]
    if not widths:
        return
    d.rounded_rectangle([x, y, x + max(widths) + 2 * pad, y + sum(hts) + 2 * pad - 6], 10, fill=fill)
    yy = y + pad
    for t, fn, c in zip(lines, fonts, colors or [(245, 242, 235)] * len(lines)):
        if t:
            d.text((x + pad, yy), t, font=fn, fill=c); yy += fn.size + 8


def legend(d, x, y):
    for k, r in enumerate((1, 2)):
        yy = y + k * 34
        d.rounded_rectangle([x, yy, x + 22, yy + 22], 4, fill=RAT[r])
        d.text((x + 32, yy - 2), f"rat {r}", font=f_sub, fill=(30, 30, 30))


def mini_map(f, size):
    """Top-down: arena, rat 1 (red, field of view wedge), rat 2 (blue)."""
    im = Image.new("RGBA", (size, size), (20, 18, 14, 210))
    d = ImageDraw.Draw(im)
    s = size / 1.0                                  # 1 m across
    c = size / 2
    to_px = lambda p: (c + p[0] * s, c - p[1] * s)
    d.ellipse([c - 0.45 * s, c - 0.45 * s, c + 0.45 * s, c + 0.45 * s], outline=(150, 190, 130), width=2)
    import placement as PL                       # same eye and heads as the renderer
    ep, h2 = (PL.gimbal_eye(1, f) or PL.eye_pose(1, f)), PL.tracked_head(2, f)
    if ep is not None:
        eye, R = ep
        nose = R[:, 0]
        a = math.atan2(nose[1], nose[0]); half = math.radians(133.4 / 2)
        e = to_px(eye); L = 0.22 * s
        pts = [e] + [(e[0] + L * math.cos(a + t), e[1] - L * math.sin(a + t)) for t in np.linspace(-half, half, 20)]
        d.polygon(pts, fill=(255, 40, 40, 70), outline=(255, 90, 90))
        d.ellipse([e[0] - 6, e[1] - 6, e[0] + 6, e[1] + 6], fill=RAT[1])
    if h2 is not None:
        b = to_px(h2)
        d.ellipse([b[0] - 7, b[1] - 7, b[0] + 7, b[1] + 7], fill=RAT[2])
    d.text((10, 8), "top view", font=f_small, fill=(220, 215, 205))
    return im


def finish(vf):
    src = FR / f"vf_{vf:04d}.png"
    if not src.exists():
        return False
    k, u, i = seg_of(vf)
    f = rec_frame(vf)
    im = Image.open(src).convert("RGB")
    noenv = HERE / "frames_noenv" / f"vf_{vf:04d}.png"
    if k == "env" and u < 0.4 and noenv.exists():           # Environment 4 crossfades in at frame 650
        a = u / 0.4; a = a * a * (3 - 2 * a)
        im = Image.blend(Image.open(noenv).convert("RGB"), im, a)
    if k == "fpv":
        x0 = (W - FPV_W) // 2
        view = im.crop((x0, 0, x0 + FPV_W, H))
        im = Image.new("RGB", (W, H), (16, 15, 12)); im.paste(view, (x0, 0))
        meta = FR / f"vf_{vf:04d}.json"
        if meta.exists():
            m = json.load(open(meta))
            q = m.get("rat2_body_img")
            if q and q[2] > 0:
                X, Y = q[0] * W, (1 - q[1]) * H
                if x0 + 20 < X < x0 + FPV_W - 20 and 20 < Y < H - 20:
                    d = ImageDraw.Draw(im, "RGBA")
                    d.ellipse([X - 9, Y - 9, X + 9, Y + 9], outline=RAT[2] + (255,), width=4)
                    box_text(d, (X + 16, Y - 52), ["rat 2"], [f_sub], pad=8, fill=(255, 255, 255, 215), colors=[RAT[2]])
        mm = mini_map(f, 230)
        im.paste(mm, (x0 + FPV_W + 5, H - 320), mm)
    d = ImageDraw.Draw(im, "RGBA")
    title, sub = caption(vf)
    box_text(d, (28, 26), [title, sub], [f_cap, f_sub])
    if k != "fpv":
        d.rounded_rectangle([24, H - 108, 150, H - 24], 10, fill=(255, 255, 255, 200))
        legend(d, 38, H - 96)
    if k not in ("intro", "zoom"):
        txt = f"frame {f}   {f / 30:5.2f} s"
        tw = d.textlength(txt, font=f_sub)
        d.rounded_rectangle([W - tw - 52, H - 70, W - 24, H - 24], 10, fill=(20, 18, 14, 165))
        d.text((W - tw - 38, H - 64), txt, font=f_sub, fill=(245, 242, 235))
    im.save(DONE / f"vf_{vf:04d}.png")
    return True


if __name__ == "__main__":
    if "--preview" in sys.argv:
        for vf in [int(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]:
            finish(vf)
        sys.exit()
    n = TL["n_video"]
    todo = range(n)
    if "--range" in sys.argv:                     # re-finish only these frames (others kept), then encode all
        a, b = (int(x) for x in sys.argv[sys.argv.index("--range") + 1].split("-"))
        todo = range(a, b + 1)
    from multiprocessing import Pool
    with Pool(8) as pool:
        pool.map(finish, todo)
    ok = [(DONE / f"vf_{vf:04d}.png").exists() for vf in range(n)]
    missing = [vf for vf, good in zip(range(n), ok) if not good]
    if missing:
        sys.exit(f"{len(missing)} frames not rendered yet, e.g. {missing[:10]}")
    out = HERE / "slide11_rig_to_first_person.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                    "-i", str(DONE / "vf_%04d.png"), "-c:v", "h264_videotoolbox", "-b:v", "20M", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out)
