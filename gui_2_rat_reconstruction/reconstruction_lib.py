"""
reconstruction_lib.py
----------------------
Visual-hull mesh reconstruction from 6 synchronized silhouette masks, ported
near-verbatim from reconstruct_3d.py (marching cubes over a carved voxel
grid). The one new piece is estimate_rat_centre(): the original script took
a hand-entered RAT_CENTRE world-space seed point (produced once by a
now-missing triangulate_both_rats.py); here it's derived automatically per
frame by triangulating the silhouette masks' own centroids across all
cameras, using the exact same p_cam = R @ p_world + t projection convention
already used everywhere in this codebase (just inverted).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from skimage import measure
from scipy import ndimage

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


# =============================================================================
# Calibration loading (unchanged convention from reconstruct_3d.py)
# =============================================================================

def load_calibration(json_path):
    with open(json_path) as f:
        data = json.load(f)

    conv = data.get("convention", {})
    rot_conv = conv.get("rotation_matrix", "")
    if rot_conv and "world_to_cam" not in rot_conv.lower():
        raise ValueError(f"Unexpected rotation convention in JSON: '{rot_conv}'")

    cameras = []
    for cam in data["cameras"]:
        R = np.array(cam["rotation_matrix"], dtype=np.float64)
        t = np.array(cam["translation_m"], dtype=np.float64)
        intr = cam["intrinsics"]
        K = np.array([[intr["fx"], 0.0, intr["cx"]],
                      [0.0, intr["fy"], intr["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([intr.get("k1", 0.0), intr.get("k2", 0.0),
                          intr.get("p1", 0.0), intr.get("p2", 0.0),
                          intr.get("k3", 0.0)], dtype=np.float64)
        cam_pos = (-R.T @ t).tolist()
        cameras.append({"name": cam["name"], "K": K, "dist": dist, "R": R, "t": t, "cam_pos": cam_pos})
    return cameras


# =============================================================================
# Frame discovery
# =============================================================================

def find_frames(masks_root, object_id, mode, selection):
    """
    masks_root is expected to contain, per frame, a mask per camera. Two
    layouts are supported (auto-detected):
      A) <masks_root>/rat_<object_id>/frame_XXXXXX/<camera_name>.png
      B) <masks_root>/frame_XXXXXX/cutouts/rat_<object_id>/<camera_name>.png
    (matches the cutouts/rat_N/ convention already produced by the existing
    rat_cutouts.py segmentation-stage tool)

    Returns list of (frame_id, {camera_name: mask_path}).
    """
    root = Path(masks_root)
    rat_dir_a = root / f"rat_{object_id}"

    def frame_ids_from(base_glob_dir, layout):
        ids = []
        for p in sorted(base_glob_dir.glob("frame_*")):
            if p.is_dir():
                ids.append(p.name)
        return ids

    if rat_dir_a.exists():
        layout = "A"
        base_dir = rat_dir_a
        all_frame_ids = frame_ids_from(base_dir, layout)
    else:
        layout = "B"
        base_dir = root
        all_frame_ids = [p.name for p in sorted(root.glob("frame_*"))
                          if (p / "cutouts" / f"rat_{object_id}").exists()]

    if not all_frame_ids:
        raise FileNotFoundError(
            f"No frames found for object {object_id} under {masks_root} "
            f"(expected rat_{object_id}/frame_XXXXXX/ or frame_XXXXXX/cutouts/rat_{object_id}/)"
        )

    if mode == "single":
        frame_id = selection["frame_id"]
        if frame_id not in all_frame_ids:
            raise FileNotFoundError(f"Frame {frame_id} not found under {masks_root}")
        chosen = [frame_id]
    else:
        start, end, step = selection["start"], selection["end"], selection.get("step", 1)
        chosen = []
        for fid in all_frame_ids:
            try:
                num = int(fid.replace("frame_", ""))
            except ValueError:
                continue
            if start <= num <= end and (num - start) % step == 0:
                chosen.append(fid)
        if not chosen:
            raise FileNotFoundError(f"No frames in range [{start},{end}] step {step} found under {masks_root}")

    result = []
    for fid in chosen:
        if layout == "A":
            frame_dir = base_dir / fid
        else:
            frame_dir = base_dir / fid / "cutouts" / f"rat_{object_id}"
        mask_paths = {}
        for p in frame_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                mask_paths[p.stem] = p
        if not mask_paths:
            continue
        result.append((fid, mask_paths))

    if not result:
        raise FileNotFoundError(f"Resolved frame ids but found no mask images inside them under {masks_root}")
    return result


def match_masks_to_cameras(cameras, mask_paths):
    """Order mask_paths (dict of arbitrary-key -> path) to match the
    cameras list order, matching by camera name substring."""
    ordered = []
    for cam in cameras:
        found = None
        for key, path in mask_paths.items():
            if cam["name"] in key or key in cam["name"]:
                found = path
                break
        ordered.append(found)
    return ordered


# =============================================================================
# RAT_CENTRE auto-estimation (new — replaces the missing triangulate script)
# =============================================================================

def _mask_centroid_px(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def estimate_rat_centre(cameras, mask_paths_ordered, log=print):
    """
    For each camera with a non-empty mask: undistort+un-project the mask's
    pixel centroid into a world-space ray (origin = camera position,
    direction = R.T @ K^-1 @ [u,v,1], normalized). Then solve the
    least-squares closest point to all rays (standard multi-ray
    triangulation via normal equations, same approach already used for the
    table-center estimate in the rig visualization).
    """
    origins, directions = [], []
    for cam, mpath in zip(cameras, mask_paths_ordered):
        if mpath is None:
            continue
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        centroid_px = _mask_centroid_px(mask)
        if centroid_px is None:
            log(f"    {cam['name']}: empty mask, skipping for centre estimate")
            continue

        pt = np.array([[centroid_px]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pt, cam["K"], cam["dist"])
        u, v = undistorted[0, 0]
        dir_cam = np.array([u, v, 1.0])
        dir_cam /= np.linalg.norm(dir_cam)
        dir_world = cam["R"].T @ dir_cam
        dir_world /= np.linalg.norm(dir_world)

        origins.append(np.array(cam["cam_pos"]))
        directions.append(dir_world)

    if len(origins) < 2:
        raise RuntimeError("Need at least 2 cameras with non-empty masks to estimate RAT_CENTRE")

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ o
    centre = np.linalg.solve(A, b)
    log(f"    estimated RAT_CENTRE from {len(origins)} cameras: [{centre[0]:+.4f}, {centre[1]:+.4f}, {centre[2]:+.4f}]")
    return centre


# =============================================================================
# Voxel carving + marching cubes (ported verbatim from reconstruct_3d.py)
# =============================================================================

def load_masks(mask_paths_ordered):
    masks = []
    for p in mask_paths_ordered:
        if p is None:
            masks.append(None)
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        masks.append(binary)
    return masks


def build_voxel_grid(centre, half_xy, depth_below, depth_above, voxel_size):
    cx, cy, cz = centre
    xs = np.arange(cx - half_xy, cx + half_xy, voxel_size)
    ys = np.arange(cy - half_xy, cy + half_xy, voxel_size)
    zs = np.arange(cz - depth_below, cz + depth_above, voxel_size)
    nx, ny, nz = len(xs), len(ys), len(zs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
    origin = np.array([xs[0], ys[0], zs[0]])
    return pts, (nx, ny, nz), origin


def project_points(pts_world, R, t, K, dist):
    pts_cam = (R @ pts_world.T).T + t
    in_front = pts_cam[:, 2] > 0
    px = np.full((len(pts_world), 2), -1.0)
    if in_front.any():
        sub = pts_cam[in_front]
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        proj, _ = cv2.projectPoints(sub.reshape(-1, 1, 3), rvec, tvec, K, dist)
        px[in_front] = proj.reshape(-1, 2)
    return px, in_front


def carve_voxels(pts, cameras, masks, min_cameras, log=print):
    N = len(pts)
    votes = np.zeros(N, dtype=np.int32)
    active = [(c, m) for c, m in zip(cameras, masks) if m is not None]
    for cam, mask in active:
        h, w = mask.shape
        px, in_front = project_points(pts, cam["R"], cam["t"], cam["K"], cam["dist"])
        u = np.round(px[:, 0]).astype(np.int32)
        v = np.round(px[:, 1]).astype(np.int32)
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = np.zeros(N, dtype=bool)
        hit[inside] = mask[v[inside], u[inside]] > 0
        votes += hit.astype(np.int32)
        log(f"    {cam['name']}: {hit.sum():,} voxels in mask")
    return votes >= min_cameras


def keep_largest_component(occupied, shape, log=print):
    grid = occupied.reshape(shape)
    labelled, n = ndimage.label(grid)
    if n <= 1:
        return occupied
    sizes = ndimage.sum(grid, labelled, range(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    clean = (labelled == largest_label).ravel()
    log(f"    found {n} components, kept largest ({int(clean.sum()):,} voxels)")
    return clean


def voxels_to_mesh(occupied, shape, origin, voxel_size):
    grid = occupied.reshape(shape).astype(np.float32)
    verts_idx, faces, _, _ = measure.marching_cubes(grid, level=0.5)
    verts_world = origin + verts_idx * voxel_size
    return verts_world, faces


def save_obj(path, verts, faces):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Visual hull reconstruction\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")


# =============================================================================
# Orchestration
# =============================================================================

def reconstruct_frame(cameras, mask_paths_ordered, params, log=print):
    masks = load_masks(mask_paths_ordered)
    centre = estimate_rat_centre(cameras, mask_paths_ordered, log=log)

    pts, shape, origin = build_voxel_grid(
        centre, params["half_xy"], params["depth_below"], params["depth_above"], params["voxel_size"]
    )
    log(f"    grid: {shape[0]}x{shape[1]}x{shape[2]} = {shape[0]*shape[1]*shape[2]:,} voxels")

    occupied = carve_voxels(pts, cameras, masks, params["min_cameras"], log=log)
    n_occ = int(occupied.sum())
    log(f"    occupied: {n_occ:,} / {len(occupied):,} voxels")
    if n_occ == 0:
        raise RuntimeError("Zero voxels survived carving -- check masks and min_cameras")

    occupied = keep_largest_component(occupied, shape, log=log)
    if occupied.sum() < 50:
        raise RuntimeError(f"Only {int(occupied.sum())} voxels remain after cleanup -- reconstruction failed")

    verts, faces = voxels_to_mesh(occupied, shape, origin, params["voxel_size"])
    return verts, faces, centre
