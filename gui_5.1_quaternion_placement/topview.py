#!/usr/bin/env python3
"""
Ground-plane (table, z=0) homography from a rig camera to a top-down view,
and warping a raw frame through it with PIL. Used by GUI 5 (server.py) for
the top-down video and by analysis_core.py to back-project mask centroids
onto a horizontal plane at the rat's body height.

Derivation. A point on the table plane (z=0 in the table frame) is
p_table(x,y) = [x, y, 0]. table_frame_from_rig's basis says the calibration-
frame point is (p_table + tc) @ B = x*B[0] + y*B[1] + tc@B -- affine in (x,y),
i.e. p_calib(x,y) = M @ [x,y,1] for a fixed 3x3 M (columns B[0], B[1], tc@B).
A camera's own (distortion-free) extrinsics are p_cam = R @ p_calib + t, also
affine, so p_cam(x,y) = N @ [x,y,1] for N = [R@M[:,0], R@M[:,1], R@M[:,2]+t].
Perspective division by p_cam's z is what turns this into a projective (not
affine) map from (x,y) to pixels -- composing the intrinsics K with N gives
exactly a planar homography H = K @ N: [u,v,1] ~ H @ [x,y,1] (up to scale).
Lens distortion (k1) is ignored, same approximation already used elsewhere in
this codebase for anything homography-shaped (see analysis_lib.py's camera
re-projection); it is small away from the image edges.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def build_initial_rig_basis(calib_json_path):
    """table_frame_from_rig, without importing the rest of GUI 4/5's stack:
    the table-aligned display basis + centre derived from the rig's average
    viewing direction (same maths as gui4_rig.build_initial_rig)."""
    with open(calib_json_path) as f:
        data = json.load(f)
    cams = []
    for cam in data["cameras"]:
        R = np.array(cam["rotation_matrix"], dtype=np.float64)
        t = np.array(cam["translation_m"], dtype=np.float64)
        cams.append({"pos": -R.T @ t, "forward": R.T[:, 2]})
    forwards = np.array([c["forward"] / np.linalg.norm(c["forward"]) for c in cams])
    avg_forward = forwards.mean(axis=0); avg_forward /= np.linalg.norm(avg_forward)
    up_disp = -avg_forward
    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up_disp, world_ref)) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])
    x_disp = np.cross(up_disp, world_ref); x_disp /= np.linalg.norm(x_disp)
    y_disp = np.cross(up_disp, x_disp)

    def to_display(v):
        return np.array([np.dot(v, x_disp), np.dot(v, y_disp), np.dot(v, up_disp)])

    A = np.zeros((3, 3)); b = np.zeros(3)
    for c in cams:
        d = c["forward"] / np.linalg.norm(c["forward"])
        proj = np.eye(3) - np.outer(d, d)
        A += proj; b += proj @ c["pos"]
    try:
        table_center_raw = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        table_center_raw = np.mean([c["pos"] for c in cams], axis=0)

    B = np.stack([x_disp, y_disp, up_disp], axis=0)
    tc = to_display(table_center_raw)
    return B, tc


def load_camera_extrinsics(calib_json_path):
    """{camera name -> raw rotation_matrix/translation_m/intrinsics}."""
    data = json.loads(Path(calib_json_path).read_text())
    return {c["name"]: c for c in data.get("cameras", [])}


def _intrinsics_matrix(cam):
    intr = cam["intrinsics"]
    return np.array([[intr["fx"], 0.0, intr["cx"]],
                     [0.0, intr["fy"], intr["cy"]],
                     [0.0, 0.0, 1.0]])


def table_plane_homography(cam, B, tc, z=0.0):
    """3x3 H: table (x,y,1) [homogeneous] -> pixel (u,v,1) [homogeneous, up to
    scale] for points on the horizontal plane at height z in the table frame
    (z=0: the table surface), ignoring lens distortion. A plane at height z is
    p_table = [x, y, z], i.e. p_calib = x*B[0] + y*B[1] + (z*B[2] + tc@B)."""
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Bx, By = B[0], B[1]
    C0 = z * B[2] + np.asarray(tc) @ B
    N = np.stack([R @ Bx, R @ By, R @ C0 + t], axis=1)
    return _intrinsics_matrix(cam) @ N


def project_table_points(P_table, cam, B, tc):
    """(N,3) table-frame points -> (N,2) pixels, pinhole only (no k1), so the
    result is consistent with the distortion-free top-view warp: a point
    projected here and pushed back through the z=0 homography lands exactly
    where the warped image shows it."""
    P = np.atleast_2d(np.asarray(P_table, dtype=np.float64))
    P_calib = (P + np.asarray(tc)) @ np.asarray(B)
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    Xc = P_calib @ R.T + t
    uvw = Xc @ _intrinsics_matrix(cam).T
    return uvw[:, :2] / uvw[:, 2:3]


def pixel_to_plane(uv, H):
    """(N,2) pixels -> (N,2) table xy on the plane that H maps from."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    h = np.hstack([uv, np.ones((len(uv), 1))]) @ np.linalg.inv(H).T
    return h[:, :2] / h[:, 2:3]


def output_to_table_affine(view_box, out_size):
    """3x3 S: output pixel (px,py,1) -> table (x,y,1), for a chosen table
    view_box=(xmin,xmax,ymin,ymax) mapped onto an out_size=(W,H) canvas, y
    increasing upward on screen (matches the top-view plots elsewhere)."""
    xmin, xmax, ymin, ymax = view_box
    W, H = out_size
    sx = (xmax - xmin) / W
    sy = (ymax - ymin) / H
    return np.array([[sx, 0.0, xmin],
                     [0.0, -sy, ymax],
                     [0.0, 0.0, 1.0]])


def full_warp_matrix(cam, B, tc, view_box, out_size):
    """Composed output-pixel -> source-pixel homography, ready for
    PIL's Image.transform(..., Image.PERSPECTIVE, coeffs)."""
    H = table_plane_homography(cam, B, tc)
    S = output_to_table_affine(view_box, out_size)
    full = H @ S
    full = full / full[2, 2]
    return full


def warp_coeffs(full_matrix):
    """The 8 coefficients PIL's PERSPECTIVE transform expects from a
    normalised (bottom-right = 1) 3x3 homography."""
    m = full_matrix
    return [m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1]]


def warp_topview(image, full_matrix, out_size):
    """A raw camera frame (file path, or an RGB uint8 array as decoded from the
    video), warped to the top-down view."""
    im = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image).convert("RGB")
    return im.transform(out_size, Image.PERSPECTIVE, warp_coeffs(full_matrix), resample=Image.BILINEAR)


def _cam_K_dist(cam):
    k = cam["intrinsics"]
    return _intrinsics_matrix(cam), np.array([k.get("k1", 0.0), k.get("k2", 0.0), k.get("p1", 0.0),
                                              k.get("p2", 0.0), k.get("k3", 0.0)])


def plane_remap(cam, B, tc, view_box, out_size, z=0.0):
    """(map_x, map_y) float32 for cv2.remap: every output pixel's table point
    on the plane at height z, projected into the camera WITH its lens model.
    Unlike the homography warp this lines the cameras up across the whole
    image, not only near their centres."""
    import cv2
    W, H = out_size
    S = output_to_table_affine(view_box, out_size)
    px, py = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xy = np.stack([px.ravel(), py.ravel(), np.ones(px.size)], 1) @ S.T
    P_table = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z)])
    P_calib = (P_table + np.asarray(tc)) @ np.asarray(B)
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    K, dist = _cam_K_dist(cam)
    Pc = P_calib @ R.T + t
    uv, _ = cv2.projectPoints(P_calib.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, dist)
    uv = uv.reshape(-1, 2)
    uv[Pc[:, 2] <= 0] = -1e6                      # behind the camera: no image
    return uv[:, 0].reshape(H, W).astype(np.float32), uv[:, 1].reshape(H, W).astype(np.float32)


def camera_centre_table(cam, B, tc):
    R = np.asarray(cam["rotation_matrix"], dtype=np.float64)
    t = np.asarray(cam["translation_m"], dtype=np.float64)
    return (-R.T @ t) @ np.asarray(B).T - np.asarray(tc)


def apparent_on_plane(P_table, cam, B, tc, z):
    """Where points above the floor APPEAR in this camera's floor-plane view:
    the camera ray through each point, intersected with the plane at height z."""
    C = camera_centre_table(cam, B, tc)
    P = np.atleast_2d(np.asarray(P_table, dtype=np.float64))
    s = (z - C[2]) / (P[:, 2] - C[2])
    return (C + (P - C) * s[:, None])[:, :2]


def estimate_floor_z(backgrounds, cams, B, tc, centre_xy=(0.0, 0.0), radius=0.2,
                     z_range=(-0.02, 0.15), step=0.005):
    """Height of the floor in the table frame: the plane on which the cameras'
    (rat-free, temporal-median) background images agree best. Score = mean
    pairwise correlation of the warped gradient magnitude inside a disk.
    Returns (z, scores dict)."""
    import cv2
    names = list(backgrounds)
    if len(names) < 2:
        return None, {}
    half = radius * 1.05
    vb = (centre_xy[0] - half, centre_xy[0] + half, centre_xy[1] - half, centre_xy[1] + half)
    out = (400, 400)
    yy, xx = np.mgrid[0:400, 0:400]
    disk = np.hypot(xx - 199.5, yy - 199.5) < 400 / 2 / 1.05
    blurred = {c: cv2.GaussianBlur(backgrounds[c].astype(np.float32), (0, 0), 1.5) for c in names}
    zs = np.arange(z_range[0], z_range[1] + 1e-9, step)
    scores = []
    for z in zs:
        grads = []
        for c in names:
            mx, my = plane_remap(cams[c], B, tc, vb, out, z)
            w = cv2.remap(blurred[c], mx, my, cv2.INTER_LINEAR, borderValue=0)
            grads.append(np.hypot(cv2.Sobel(w, cv2.CV_32F, 1, 0), cv2.Sobel(w, cv2.CV_32F, 0, 1))[disk])
        scores.append(np.mean([np.corrcoef(grads[i], grads[j])[0, 1]
                               for i in range(len(names)) for j in range(i + 1, len(names))]))
    scores = np.nan_to_num(np.asarray(scores), nan=-1.0)
    i = int(np.argmax(scores))
    z = float(zs[i])
    if 0 < i < len(zs) - 1:                      # parabola through the peak
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        if den < 0:
            z += 0.5 * (a - c) / den * step
    return z, {"z_grid": zs.tolist(), "score": scores.tolist(), "best_score": float(scores[i])}


def table_to_output_px(xy, view_box, out_size):
    """Inverse of output_to_table_affine, for drawing overlays: table (x,y)
    (N,2) array -> output pixel (px,py) (N,2)."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    xmin, xmax, ymin, ymax = view_box
    W, H = out_size
    px = (xy[:, 0] - xmin) / (xmax - xmin) * W
    py = H - (xy[:, 1] - ymin) / (ymax - ymin) * H
    return np.stack([px, py], axis=1)
