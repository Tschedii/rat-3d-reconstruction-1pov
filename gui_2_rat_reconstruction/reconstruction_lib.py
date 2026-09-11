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
from scipy import ndimage, sparse

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


def _vertex_adjacency_matrix(n_verts, faces):
    """Symmetric 0/1 sparse adjacency matrix from triangle edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    data = np.ones(len(edges), dtype=np.float64)
    A = sparse.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0  # collapse any duplicate edges back to a plain 0/1 adjacency
    return A


def taubin_smooth(verts, faces, iterations=10, lambda_=0.5, mu=-0.53):
    """Taubin (1995) lambda/mu mesh smoothing.

    Plain (Laplacian-only) smoothing removes the staircase artefacts of
    marching cubes but visibly shrinks the mesh over successive iterations,
    since every step moves each vertex toward the centroid of its
    neighbours. Taubin smoothing alternates that shrinking step (factor
    lambda_ > 0) with a second, oppositely-signed "inflating" step (factor
    mu < 0, |mu| > lambda_) that undoes the shrinkage's low-frequency
    component while leaving the high-frequency (staircase) noise removed --
    a low-pass filter on the mesh surface rather than a pure contraction.
    """
    n = len(verts)
    A = _vertex_adjacency_matrix(n, faces)
    degree = np.asarray(A.sum(axis=1)).flatten()
    isolated = degree == 0
    degree[isolated] = 1.0  # guard only; a valid marching-cubes mesh has no isolated vertices
    inv_degree = (1.0 / degree)[:, None]

    v = verts.astype(np.float64).copy()
    for _ in range(iterations):
        for factor in (lambda_, mu):
            neighbour_mean = A @ v * inv_degree
            v = v + factor * (neighbour_mean - v)
    return v


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
# Texture-atlas baking -- a real image texture sampled from the source
# camera footage, rather than the flat "v x y z r g b" per-vertex colour of
# color_vertices()/save_obj(). Per-vertex colour is bounded by mesh vertex
# spacing no matter how sharp the source images are, and (being a
# non-standard OBJ extension) isn't rendered at all by many simple viewers,
# including macOS Preview/Quick Look. An image texture referenced through a
# standard .mtl file has neither limitation.
# =============================================================================

def unwrap_cylindrical(verts, faces):
    """Per-face-corner cylindrical UV unwrap. Finds the mesh's principal
    axis via PCA (a reasonable proxy for the animal's nose-to-tail axis for
    a visual-hull body) and parameterises each vertex by its angle around
    that axis (u) and position along it (v). UV is computed per FACE CORNER
    rather than per vertex specifically to handle the seam where the angle
    wraps from 1 back to 0: a face straddling the seam gets its low-u
    corner(s) shifted up by +1 for that face only, so its UV footprint is
    contiguous instead of spanning almost the whole atlas width -- the same
    vertex can therefore carry different UV coordinates in different faces,
    which is exactly what independent v/vt face-corner indices in the OBJ
    format are for.

    Returns (corner_uv, axis): corner_uv has shape (F, 3, 2), values in
    [0, ~2) for u (wrap with `% 1.0` when addressing actual texture pixels)
    and [0, 1] for v.
    """
    centre = verts.mean(axis=0)
    centered = verts - centre
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, world_ref)) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, world_ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    along = centered @ axis
    x1 = centered @ e1
    x2 = centered @ e2
    angle = np.arctan2(x2, x1)

    u_vertex = (angle + np.pi) / (2.0 * np.pi)
    span = along.max() - along.min()
    v_vertex = (along - along.min()) / (span if span > 1e-9 else 1.0)

    corner_uv = np.empty((len(faces), 3, 2), dtype=np.float64)
    us = u_vertex[faces]  # (F, 3)
    seam = (us.max(axis=1) - us.min(axis=1)) > 0.5
    us_fixed = np.where(seam[:, None] & (us < 0.5), us + 1.0, us)
    corner_uv[:, :, 0] = us_fixed
    corner_uv[:, :, 1] = v_vertex[faces]
    return corner_uv, axis


def bake_texture_atlas(verts, faces, corner_uv, cameras, colors, masks, atlas_size=1536, log=print):
    """Bake a texture atlas: for each face, pick whichever camera views it
    the most frontally (highest alignment between the face normal and the
    direction to that camera) among cameras where all three of its vertices
    are foreground, then warp that camera's corresponding image triangle
    into the face's UV footprint via an affine transform -- the standard
    per-triangle projective-texturing approach, with view (not multi-band)
    selection: no blending is performed across the boundary between two
    faces textured from different cameras, so a faint seam can appear
    there. Faces with no fully-visible camera fall back to a flat fill
    from color_vertices()'s per-vertex average.

    Returns an (atlas_size, atlas_size, 3) uint8 RGB image.
    """
    n_verts = len(verts)
    n_cams = len(cameras)

    proj_px = np.zeros((n_cams, n_verts, 2))
    vis = np.zeros((n_cams, n_verts), dtype=bool)
    for ci, cam in enumerate(cameras):
        if colors[ci] is None or masks[ci] is None:
            continue
        px, in_front = project_points(verts, cam["R"], cam["t"], cam["K"], cam["dist"])
        h, w = masks[ci].shape
        u_px = np.round(px[:, 0]).astype(np.int32)
        v_px = np.round(px[:, 1]).astype(np.int32)
        inside = in_front & (u_px >= 0) & (u_px < w) & (v_px >= 0) & (v_px < h)
        ok = np.zeros(n_verts, dtype=bool)
        ok[inside] = masks[ci][v_px[inside], u_px[inside]] > 0
        proj_px[ci] = px
        vis[ci] = ok

    tri_v = verts[faces]  # (F, 3, 3)
    normals = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len
    centroids = tri_v.mean(axis=1)
    cam_pos_arr = np.array([cam["cam_pos"] for cam in cameras], dtype=np.float64)

    fallback_colors = color_vertices(verts, cameras, colors, masks, log=lambda *_: None)

    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    painted = np.zeros((atlas_size, atlas_size), dtype=bool)
    n_baked, n_fallback = 0, 0

    for fi, face in enumerate(faces):
        face_vis = vis[:, face].all(axis=1)  # (C,) -- True where all 3 verts are foreground
        cam_candidates = np.nonzero(face_vis)[0]

        # Wrap the whole face into [0, 1) with a SINGLE shared offset -- not
        # each corner's own `% 1.0` -- so a face straddling the seam (whose
        # corner_uv values from unwrap_cylindrical span e.g. [0.98, 1.02])
        # keeps its three corners together instead of being torn back apart
        # across the cut, which would otherwise paint a huge, wrong-content
        # triangle over unrelated parts of the atlas.
        u_corners = corner_uv[fi, :, 0] - np.floor(corner_uv[fi, :, 0].min())
        dst = np.stack([u_corners * atlas_size,
                         (1.0 - corner_uv[fi, :, 1]) * atlas_size], axis=1).astype(np.float32)
        x0, y0 = np.floor(dst.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, atlas_size), min(y1, atlas_size)
        if x1 <= x0 or y1 <= y0:
            continue  # degenerate UV triangle (zero footprint in the atlas): nothing to paint

        dst_local = dst - [x0, y0]
        mask_local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask_local, dst_local.astype(np.int32), 255)
        region = mask_local > 0
        if not region.any():
            continue

        if len(cam_candidates) > 0:
            view_dirs = cam_pos_arr[cam_candidates] - centroids[fi]
            view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
            best_ci = cam_candidates[np.argmax(view_dirs @ normals[fi])]
            src_tri = proj_px[best_ci, face].astype(np.float32)
            M = cv2.getAffineTransform(src_tri, dst_local.astype(np.float32))
            warped_bgr = cv2.warpAffine(
                colors[best_ci], M, (x1 - x0, y1 - y0),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            atlas[y0:y1, x0:x1][region] = warped_bgr[region][:, ::-1]  # BGR -> RGB
            n_baked += 1
        else:
            flat_rgb = fallback_colors[face].mean(axis=0).astype(np.uint8)
            atlas[y0:y1, x0:x1][region] = flat_rgb
            n_fallback += 1
        painted[y0:y1, x0:x1] |= region

    log(f"    texture atlas: {n_baked:,} faces baked from camera views, {n_fallback:,} used a flat fallback fill")

    unpainted = (~painted).astype(np.uint8) * 255
    if unpainted.any() and painted.any():
        atlas = cv2.inpaint(atlas, unpainted, 3, cv2.INPAINT_TELEA)
    return atlas


def save_textured_obj(path, verts, faces, corner_uv, atlas_rgb):
    """Write an OBJ + MTL + PNG texture triple: a standard image-textured
    mesh, viewable (unlike the "v x y z r g b" extension of save_obj) in
    essentially any OBJ-capable tool, including macOS Preview/Quick Look."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    tex_path = path.with_name(path.stem + "_texture.png")

    cv2.imwrite(str(tex_path), atlas_rgb[:, :, ::-1])  # RGB -> BGR for cv2.imwrite

    with open(mtl_path, "w") as f:
        f.write(f"newmtl material0\nKd 1.0 1.0 1.0\nmap_Kd {tex_path.name}\n")

    with open(path, "w") as f:
        f.write("# Visual hull reconstruction (textured)\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl material0\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv_tri in corner_uv.reshape(-1, 2):
            f.write(f"vt {uv_tri[0] % 1.0:.6f} {uv_tri[1]:.6f}\n")
        for fi, face in enumerate(faces):
            vt0, vt1, vt2 = 3 * fi + 1, 3 * fi + 2, 3 * fi + 3
            f.write(f"f {face[0]+1}/{vt0} {face[1]+1}/{vt1} {face[2]+1}/{vt2}\n")


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
    log(f"    raw mesh: {len(verts):,} verts, {len(faces):,} faces")

    smooth_iterations = int(params.get("smooth_iterations", 10))
    if smooth_iterations > 0:
        verts = taubin_smooth(verts, faces, iterations=smooth_iterations)
        log(f"    smoothed surface ({smooth_iterations} Taubin iterations)")

    texture_mode = params.get("texture_mode", "atlas")  # "atlas" | "vertex_color" | "none"
    result = {"verts": verts, "faces": faces, "centre": centre,
              "vertex_colors": None, "corner_uv": None, "atlas": None}

    if texture_mode == "atlas":
        corner_uv, _ = unwrap_cylindrical(verts, faces)
        atlas_size = int(params.get("atlas_size", 1536))
        atlas = bake_texture_atlas(verts, faces, corner_uv, cameras, colors, masks,
                                    atlas_size=atlas_size, log=log)
        result["corner_uv"] = corner_uv
        result["atlas"] = atlas
    elif texture_mode == "vertex_color":
        result["vertex_colors"] = color_vertices(verts, cameras, colors, masks, log=log)

    return result
