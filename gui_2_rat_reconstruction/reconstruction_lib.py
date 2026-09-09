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
already used everywhere in this codebase (just inverted). The other new
piece is color_vertices(): each mesh vertex is reprojected into every
camera and coloured from whichever cutouts see it as foreground, so the
output .obj carries real fur colour (as a "v x y z r g b" vertex-color
extension) rather than bare geometry.
"""

import json
import re
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
    masks_root is expected to contain, per frame, a mask per camera. Three
    layouts are supported (auto-detected):
      A) <masks_root>/rat_<object_id>/frame_XXXXXX/<camera_name>.png
      B) <masks_root>/frame_XXXXXX/cutouts/rat_<object_id>/<camera_name>.png
      C) <masks_root>/<camera_name>/cutouts/rat_<object_id>/frame_XXXXXX.png
    (A and B match the cutouts/rat_N/ convention already produced by the
    existing rat_cutouts.py segmentation-stage tool; C is the same tool run
    with the camera folder as the outer level instead of the frame)

    Returns list of (frame_id, {camera_name: mask_path}).
    """
    root = Path(masks_root)
    rat_dir_a = root / f"rat_{object_id}"

    def frame_ids_from(base_glob_dir):
        ids = []
        for p in sorted(base_glob_dir.glob("frame_*")):
            if p.is_dir():
                ids.append(p.name)
        return ids

    # Layout B's outer dirs are frame_XXXXXX/cutouts/rat_<id>/ too, so
    # exclude frame_*-named dirs here or every layout-B tree would also
    # look like a (single-frame-named) layout-C camera directory.
    cam_dirs_c = sorted(
        p for p in root.glob("*")
        if p.is_dir() and not p.name.startswith("frame_")
        and (p / "cutouts" / f"rat_{object_id}").is_dir()
    )

    if rat_dir_a.exists():
        layout = "A"
        base_dir = rat_dir_a
        all_frame_ids = frame_ids_from(base_dir)
    elif cam_dirs_c:
        layout = "C"
        all_frame_ids = sorted({
            p.stem
            for cam_dir in cam_dirs_c
            for p in (cam_dir / "cutouts" / f"rat_{object_id}").iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.startswith("frame_")
        })
    else:
        layout = "B"
        base_dir = root
        all_frame_ids = [p.name for p in sorted(root.glob("frame_*"))
                          if (p / "cutouts" / f"rat_{object_id}").exists()]

    if not all_frame_ids:
        raise FileNotFoundError(
            f"No frames found for object {object_id} under {masks_root} (expected "
            f"rat_{object_id}/frame_XXXXXX/, frame_XXXXXX/cutouts/rat_{object_id}/, "
            f"or <camera>/cutouts/rat_{object_id}/frame_XXXXXX.png)"
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
        mask_paths = {}
        if layout == "C":
            for cam_dir in cam_dirs_c:
                cam_frame_dir = cam_dir / "cutouts" / f"rat_{object_id}"
                for ext in IMAGE_EXTENSIONS:
                    fpath = cam_frame_dir / f"{fid}{ext}"
                    if fpath.exists():
                        mask_paths[cam_dir.name] = fpath
                        break
        else:
            frame_dir = base_dir / fid if layout == "A" else base_dir / fid / "cutouts" / f"rat_{object_id}"
            for p in frame_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    mask_paths[p.stem] = p
        if not mask_paths:
            continue
        result.append((fid, mask_paths))

    if not result:
        raise FileNotFoundError(f"Resolved frame ids but found no mask images inside them under {masks_root}")
    return result


def _trailing_number(s):
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else None


def match_masks_to_cameras(cameras, mask_paths):
    """Order mask_paths (dict of arbitrary-key -> path) to match the
    cameras list order. Matches by camera name substring first (e.g.
    calibration name "camera_001" vs mask key "camera_001"), falling back
    to matching by trailing camera index (e.g. calibration name
    "camera_001" vs a mask folder named "cam_1")."""
    ordered = []
    used_keys = set()
    for cam in cameras:
        found = None
        for key, path in mask_paths.items():
            if key in used_keys:
                continue
            if cam["name"] in key or key in cam["name"]:
                found = path
                used_keys.add(key)
                break
        if found is None:
            cam_num = _trailing_number(cam["name"])
            if cam_num is not None:
                for key, path in mask_paths.items():
                    if key in used_keys:
                        continue
                    if _trailing_number(key) == cam_num:
                        found = path
                        used_keys.add(key)
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
    """Load each per-camera cutout and derive its foreground silhouette.

    Cutouts are camera-space crops of the animal on a black background
    (the segmentation stage's own convention) -- any non-black pixel counts
    as foreground. That works identically for real RGB fur-colour cutouts
    and for plain white-on-black binary masks, and (unlike a mid-grey
    brightness threshold) doesn't drop dark fur pixels as background.

    Returns (colors, masks): colors[i] is the raw BGR cutout (for vertex
    colouring), masks[i] is the derived binary foreground mask -- both
    None for any camera with no mask for this frame.
    """
    colors, masks = [], []
    for p in mask_paths_ordered:
        if p is None:
            colors.append(None)
            masks.append(None)
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            colors.append(None)
            masks.append(None)
            continue
        mask = (img.max(axis=2) > 0).astype(np.uint8) * 255
        colors.append(img)
        masks.append(mask)
    return colors, masks


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


def color_vertices(verts, cameras, colors, masks, fallback=(200, 200, 200), log=print):
    """Sample fur colour for each output mesh vertex by reprojecting it
    into every camera (same p_cam = R @ p_world + t projection used for
    carving) and averaging the cutout pixel colour from every camera that
    sees it as foreground. Vertices no camera claims (e.g. fully
    self-occluded) fall back to a neutral grey.

    Returns an (N, 3) uint8 array in R, G, B order.
    """
    n = len(verts)
    accum = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)

    for cam, color, mask in zip(cameras, colors, masks):
        if color is None or mask is None:
            continue
        h, w = mask.shape
        px, in_front = project_points(verts, cam["R"], cam["t"], cam["K"], cam["dist"])
        u = np.round(px[:, 0]).astype(np.int32)
        v = np.round(px[:, 1]).astype(np.int32)
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible = np.zeros(n, dtype=bool)
        visible[inside] = mask[v[inside], u[inside]] > 0
        bgr = color[v[visible], u[visible]].astype(np.float64)
        accum[visible] += bgr[:, ::-1]  # BGR (cv2) -> RGB
        counts[visible] += 1

    has_color = counts > 0
    out = np.tile(np.array(fallback, dtype=np.float64), (n, 1))
    out[has_color] = accum[has_color] / counts[has_color, None]
    if not has_color.all():
        log(f"    {int((~has_color).sum()):,} / {n:,} vertices had no camera coverage, using fallback grey")
    return np.clip(out, 0, 255).astype(np.uint8)


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


def save_obj(path, verts, faces, colors=None):
    """colors, if given, is an (N, 3) uint8 RGB array -- written as the
    widely-supported (MeshLab / CloudCompare / Open3D / Blender) "v x y z
    r g b" vertex-color extension, with r/g/b normalized to [0, 1]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Visual hull reconstruction\n")
        if colors is not None:
            rgb = colors.astype(np.float64) / 255.0
            for v, c in zip(verts, rgb):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        else:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")


# =============================================================================
# Orchestration
# =============================================================================

def reconstruct_frame(cameras, mask_paths_ordered, params, log=print):
    colors, masks = load_masks(mask_paths_ordered)
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

    vertex_colors = None
    if params.get("color_mesh", True):
        vertex_colors = color_vertices(verts, cameras, colors, masks, log=log)

    return verts, faces, centre, vertex_colors
