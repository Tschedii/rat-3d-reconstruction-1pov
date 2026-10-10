"""Check the video's 3D placement against the six raw camera videos: for a recording frame, project the IMU
skull (cap scan, placed as in render_slide11.py) and/or a rat mesh into every camera's raw frame and draw them
over it (cap: yellow outline + translucent fill; rat mesh: its rat colour, outline), cropped around the rat.

    python overlay_check.py <out.jpg> <rat>:<frame>[:cap|mesh|both][,...] [--mesh-root DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
DRIVE = Path(__file__).resolve().parents[1]  # repository root
VIDEOS = DRIVE / "data/raw_videos/Day_7_rPEG008_rPEG007_25-08-2026"
sys.path.insert(0, str(HERE))
import placement as pl                    # noqa: E402  (skull / eye placement shared with the renderer)

CAL = json.loads((DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json").read_text())["cameras"]
S = json.load(open(HERE / "data/scene.json"))
T = np.array(S["T_from_calib"]); T_inv = np.linalg.inv(T)
RAT_BGR = {1: (40, 40, 255), 2: (255, 110, 40)}


def project(P_T, cam):
    P = P_T @ T_inv[:3, :3].T + T_inv[:3, 3]
    R = np.array(cam["rotation_matrix"]); t = np.array(cam["translation_m"]); k = cam["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    dist = np.array([k.get(n, 0.0) for n in ("k1", "k2", "p1", "p2", "k3")])
    uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, dist)
    return uv.reshape(-1, 2)


def frame_img(cam_name, f):
    cap = cv2.VideoCapture(str(VIDEOS / f"cam_{int(cam_name[-1])}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read(); cap.release()
    return im if ok else np.zeros((1200, 1920, 3), np.uint8)


def draw_points(im, uv, color, alpha=0.35):
    h, w = im.shape[:2]
    m = np.zeros((h, w), np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    p = np.minimum(np.round(uv[ok]).astype(int), [w - 1, h - 1])
    m[p[:, 1], p[:, 0]] = 255
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    over = im.copy(); over[m > 0] = color
    im[:] = cv2.addWeighted(over, alpha, im, 1 - alpha, 0)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, color, 2)
    return p.mean(0) if len(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("items"); ap.add_argument("--mesh-root", default=str(HERE / "refit_smooth"))
    a = ap.parse_args()
    rows = []
    cap_pts = pl.cap_points(20000)
    for item in a.items.split(","):
        parts = item.split(":")
        r, f = int(parts[0]), int(parts[1]); what = parts[2] if len(parts) > 2 else "both"
        row = []
        for cam in CAL:
            im = frame_img(cam["name"], f)
            centre = None
            if what in ("mesh", "both"):
                mp = next((q for q in (Path(a.mesh_root) / f"object_{r}_template_fitted" / f"frame_{f:06d}" / "mesh.obj",
                                       Path(a.mesh_root) / f"object_{r}" / f"frame_{f:06d}" / "mesh.obj") if q.exists()), None)
                if mp is not None:
                    V = np.array([[float(x) for x in l.split()[1:4]] for l in open(mp) if l.startswith("v ")])
                    centre = draw_points(im, project(V @ T[:3, :3].T + T[:3, 3], cam), RAT_BGR[r], 0.25)
            if what in ("cap", "both"):
                M = pl.skull_matrix(r, f)
                if M is not None:
                    c2 = draw_points(im, project(cap_pts @ M[:3, :3].T + M[:3, 3], cam), (0, 230, 255), 0.45)
                    centre = centre if centre is not None else c2
            if centre is None:
                hd = pl.tracked_head(r, f)
                centre = project(hd[None], cam)[0] if hd is not None else np.array([960, 600])
            C = 230
            pad = cv2.copyMakeBorder(im, C, C, C, C, cv2.BORDER_CONSTANT)
            x, y = int(centre[0]), int(centre[1])
            crop = cv2.resize(pad[y:y + 2 * C, x:x + 2 * C], (260, 260))
            cv2.putText(crop, f"rat {r} f{f} {cam['name'][-1]}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            row.append(crop)
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
