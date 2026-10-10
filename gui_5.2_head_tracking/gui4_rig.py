#!/usr/bin/env python3
"""Fallback copy of build_initial_rig / inspect_obj from GUI 4 (gui_4_scene_placement/env_marker_server.py),
used only when server.py runs outside the repo and GUI 4 is not next to it. Keep in sync with GUI 4.
"""

import json
from pathlib import Path

import numpy as np


# =============================================================================
# Rig geometry — table-aligned display frame, derived from the calibration
# JSON's own average viewing direction (same approach used for the rig plots
# in GUI 1's calibration_lib.py). Produces the same schema the three.js
# frontend already expects: {table_center, cameras:[{name,is_reference,
# position,forward,up,right}]}.
# =============================================================================

def build_initial_rig(calib_json_path):
    with open(calib_json_path) as f:
        data = json.load(f)

    cams = []
    for cam in data["cameras"]:
        R = np.array(cam["rotation_matrix"], dtype=np.float64)
        t = np.array(cam["translation_m"], dtype=np.float64)
        cam_pos = -R.T @ t
        cams.append({
            "name": cam["name"], "is_reference": cam.get("is_reference", False),
            "pos": cam_pos,
            "forward": R.T[:, 2],
            "up": -R.T[:, 1],
            "right": R.T[:, 0],
        })

    # table-aligned basis from the average viewing direction
    forwards = np.array([c["forward"] / np.linalg.norm(c["forward"]) for c in cams])
    avg_forward = forwards.mean(axis=0)
    avg_forward /= np.linalg.norm(avg_forward)

    up_disp = -avg_forward
    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up_disp, world_ref)) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])
    x_disp = np.cross(up_disp, world_ref)
    x_disp /= np.linalg.norm(x_disp)
    y_disp = np.cross(up_disp, x_disp)

    def to_display(v):
        return np.array([np.dot(v, x_disp), np.dot(v, y_disp), np.dot(v, up_disp)])

    # table center: least-squares intersection of all cameras' optical axes
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c["forward"] / np.linalg.norm(c["forward"])
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ c["pos"]
    try:
        table_center_raw = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        table_center_raw = np.mean([c["pos"] for c in cams], axis=0)

    rig = {
        "table_center": to_display(table_center_raw).tolist(),
        # The calibration -> display rotation itself, as three row vectors.
        # The camera poses below are already expressed in the display frame,
        # but a reconstructed mesh arrives in the RAW calibration frame, so the
        # frontend needs this basis to put the two in the same space. Without
        # it the only options are to guess (and land the animal metres off) or
        # to place it by hand.
        "display_basis": [x_disp.tolist(), y_disp.tolist(), up_disp.tolist()],
        "cameras": [],
    }
    for c in cams:
        rig["cameras"].append({
            "name": c["name"],
            "is_reference": c["is_reference"],
            "position": to_display(c["pos"]).tolist(),
            "forward": to_display(c["forward"] / np.linalg.norm(c["forward"])).tolist(),
            "up": to_display(c["up"] / np.linalg.norm(c["up"])).tolist(),
            "right": to_display(c["right"] / np.linalg.norm(c["right"])).tolist(),
        })
    return rig



# =============================================================================
# Mesh inspection. The frontend needs to know, for the actual files the user
# configured, whether the mesh still carries its reconstruction-frame world
# coordinates or has been re-centred on its own origin by some later step --
# that decision changes how the mesh must be placed, and guessing it wrong is
# exactly the "rats are in the wrong place" failure. So rather than assume,
# we measure: parse the vertices, report the bounding box and centroid in the
# calibration frame, and let the caller compare against the rig's own table
# centre.
# =============================================================================

def inspect_obj(obj_path):
    p = Path(obj_path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    acc = [0.0, 0.0, 0.0]
    n = 0
    with p.open("r", errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            for i in range(3):
                lo[i] = min(lo[i], xyz[i])
                hi[i] = max(hi[i], xyz[i])
                acc[i] += xyz[i]
            n += 1
    if n == 0:
        return {"exists": True, "path": str(p), "vertex_count": 0}
    return {
        "exists": True,
        "path": str(p),
        "vertex_count": n,
        "bbox_min": lo,
        "bbox_max": hi,
        "size": [hi[i] - lo[i] for i in range(3)],
        "centroid": [acc[i] / n for i in range(3)],
        "bbox_center": [(lo[i] + hi[i]) / 2 for i in range(3)],
        "has_mtl": p.with_suffix(".mtl").exists(),
        "mtl_name": p.with_suffix(".mtl").name,
    }
