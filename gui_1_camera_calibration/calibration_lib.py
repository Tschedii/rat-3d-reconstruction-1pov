"""
calibration_lib.py
-------------------
Parameterized checkerboard calibration pipeline: frame extraction, intrinsic
calibration, extrinsic calibration, and rig visualization. Ported from the
original 0_camera_calibration scripts (frame_extraction_for_one_camera.py,
internal_parameters.py, extrinsic_parameters.py, visualize_cameras.py /
the rig-plot script), with hardcoded module-level constants turned into
function parameters so a GUI can drive them.

JSON convention (unchanged from the original scripts):
  rotation_matrix   : R_world_to_cam  (p_cam = R @ p_world + t)
  translation_m      : standard OpenCV t vector (NOT camera world position)
  cam_pos_world_m    : camera position in world = -R^T @ t
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


# =============================================================================
# STAGE 1 — frame extraction
# =============================================================================

def _corner_criteria():
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def extract_diverse_frames(video_path, out_dir, board_w, board_h,
                            target_frames=300, min_pose_distance=0.15,
                            min_trans_distance=0.05, sample_every_n=10,
                            log=print):
    """Extract pose-diverse chessboard frames from one camera's video.
    Ported from frame_extraction_for_one_camera.py. Used for intrinsic
    calibration (each camera calibrated independently)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    criteria = _corner_criteria()

    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    kept_poses = []
    saved_count = 0
    frame_idx = 0
    rough_K = None

    while cap.isOpened() and saved_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if rough_K is None:
            h, w = gray.shape
            rough_K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)

        found, corners = cv2.findChessboardCorners(
            gray, (board_w, board_h),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ok, rvec, tvec = cv2.solvePnP(objp, corners_refined, rough_K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue

        diverse = True
        for rv, tv in kept_poses:
            if np.linalg.norm(rvec - rv) < min_pose_distance and np.linalg.norm(tvec - tv) < min_trans_distance:
                diverse = False
                break
        if not diverse:
            continue

        filename = out_path / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(filename), frame)
        kept_poses.append((rvec, tvec))
        saved_count += 1

    cap.release()
    log(f"  extracted {saved_count} diverse frames -> {out_path}")
    return saved_count


def extract_synced_frames(video_paths, out_root, board_w, board_h,
                           sample_every_n=5, max_frames=60, log=print):
    """Walk N camera videos in lockstep, keep a frame index only when the
    checkerboard is detected in ALL cameras simultaneously. Saves into
    <out_root>/<camera_name>/frame_NNNNNN.png -- the folder structure
    calibrate_extrinsics() expects (one synchronized frame set per stem,
    common across every camera). This is new logic (no equivalent existed
    in the original scripts, which assumed synced frames were already on
    disk) but reuses the same corner-detection code as extract_diverse_frames.

    video_paths: dict camera_name -> video file path
    """
    names = list(video_paths.keys())
    caps = {name: cv2.VideoCapture(str(video_paths[name])) for name in names}
    for name, cap in caps.items():
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for {name}: {video_paths[name]}")

    out_dirs = {}
    for name in names:
        d = Path(out_root) / name
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d

    criteria = _corner_criteria()
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        frames = {}
        any_ended = False
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                any_ended = True
                break
            frames[name] = frame
        if any_ended:
            break
        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        found_all = True
        for name, frame in frames.items():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(
                gray, (board_w, board_h),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not found:
                found_all = False
                break

        if not found_all:
            continue

        stem = f"frame_{frame_idx:06d}"
        for name, frame in frames.items():
            cv2.imwrite(str(out_dirs[name] / f"{stem}.png"), frame)
        saved_count += 1
        log(f"  synced frame {stem}  ({saved_count}/{max_frames})")

    for cap in caps.values():
        cap.release()

    log(f"  extracted {saved_count} synchronized frame sets -> {out_root}")
    return saved_count


# =============================================================================
# STAGE 2 — intrinsic calibration (per camera)
# =============================================================================

def _load_images(folder):
    folder = Path(folder)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(sorted(folder.glob(f"*{ext}")))
        images.extend(sorted(folder.glob(f"*{ext.upper()}")))
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images


def _detect_corners(image_paths, board_w, board_h, log=print):
    criteria = _corner_criteria()
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    obj_points, img_points, used_files = [], [], []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, (board_w, board_h),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            continue
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp.copy())
        img_points.append(corners_refined)
        used_files.append(path.name)

    log(f"  detected corners in {len(used_files)}/{len(image_paths)} images")
    return obj_points, img_points, image_size, used_files


def _build_initial_K(image_size, sensor_specs):
    if not sensor_specs or None in (sensor_specs.get("focal_length_mm"),
                                     sensor_specs.get("sensor_width_mm"),
                                     sensor_specs.get("sensor_height_mm")):
        return None
    w, h = image_size
    fx = sensor_specs["focal_length_mm"] * (w / sensor_specs["sensor_width_mm"])
    fy = sensor_specs["focal_length_mm"] * (h / sensor_specs["sensor_height_mm"])
    f = (fx + fy) / 2
    return np.array([[f, 0., w / 2.], [0., f, h / 2.], [0., 0., 1.]], dtype=np.float64)


def _per_image_errors(obj_points, img_points, rvecs, tvecs, K, dist):
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        errors.append(cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected))
    return errors


def _distortion_is_stable(dist):
    k1, k2, p1, p2, k3 = dist.flatten()
    if abs(k2) > 0.8 or abs(k3) > 2.0 or abs(p1) > 0.2 or abs(p2) > 0.2:
        return False
    return True


def _principal_point_is_sane(K, image_size):
    w, h = image_size
    cx_off = abs(K[0, 2] - w / 2) / w * 100
    cy_off = abs(K[1, 2] - h / 2) / h * 100
    return cx_off <= 15 and cy_off <= 15


def _try_all_models(obj_points, img_points, image_size, K0, log=print):
    FIX_AR = cv2.CALIB_FIX_ASPECT_RATIO
    FIX_PP = cv2.CALIB_FIX_PRINCIPAL_POINT
    FIX_K2 = cv2.CALIB_FIX_K2
    FIX_K3 = cv2.CALIB_FIX_K3
    ZTD = cv2.CALIB_ZERO_TANGENT_DIST

    models = [
        ("k1 (fx=fy,cx=cy fixed)", FIX_AR | FIX_PP | FIX_K2 | FIX_K3 | ZTD),
        ("k1,k2 (fx=fy,cx=cy fixed)", FIX_AR | FIX_PP | FIX_K3 | ZTD),
        ("k1,k2,p1,p2 (fx=fy,cx=cy fixed)", FIX_AR | FIX_PP | FIX_K3),
        ("k1 (fx=fy fixed)", FIX_AR | FIX_K2 | FIX_K3 | ZTD),
        ("k1,k2 (fx=fy fixed)", FIX_AR | FIX_K3 | ZTD),
        ("k1,k2,p1,p2 (fx=fy fixed)", FIX_AR | FIX_K3),
        ("k1,k2,p1,p2,k3 (fx=fy fixed)", FIX_AR),
        ("k1,k2 (free)", FIX_K3 | ZTD),
        ("k1,k2,p1,p2 (free)", FIX_K3),
        ("k1,k2,p1,p2,k3 (free)", 0),
    ]

    results = []
    for label, flags in models:
        try:
            K_init = K0.copy() if K0 is not None else None
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, image_size, K_init, None, flags=flags
            )
            errors = _per_image_errors(obj_points, img_points, rvecs, tvecs, K, dist)
            ok = _distortion_is_stable(dist) and _principal_point_is_sane(K, image_size)
            results.append({"label": label, "rms": rms, "K": K, "dist": dist,
                             "rvecs": rvecs, "tvecs": tvecs, "errors": errors, "stable": ok})
        except cv2.error as e:
            log(f"    model '{label}' failed: {e}")
    return results


def _select_best_model(results):
    stable = [r for r in results if r["stable"]]
    if not stable:
        return min(results, key=lambda r: r["rms"])
    best_rms = min(r["rms"] for r in stable)
    candidates = [r for r in stable if r["rms"] <= best_rms + 0.05]
    return candidates[0]


def calibrate_intrinsics(frames_folder, board_w, board_h, square_size,
                          sensor_specs=None, outlier_threshold_px=0.8, log=print):
    """Full intrinsic calibration for one camera's frame folder.
    Ported from internal_parameters.py. Returns a dict with K/dist/rms/etc,
    JSON-serializable via the 'summary' key."""
    image_paths = _load_images(frames_folder)
    obj_points, img_points, image_size, used_files = _detect_corners(image_paths, board_w, board_h, log)

    if len(obj_points) < 10:
        raise RuntimeError(f"Only {len(obj_points)} usable images in {frames_folder} -- need >= 10")

    for objp in obj_points:
        objp *= square_size

    K0 = _build_initial_K(image_size, sensor_specs)

    results = _try_all_models(obj_points, img_points, image_size, K0, log)
    best = _select_best_model(results)

    errors = best["errors"]
    outliers = [f for f, e in zip(used_files, errors) if e > outlier_threshold_px]
    if outliers:
        keep = [(o, i, f) for o, i, f, e in zip(obj_points, img_points, used_files, errors) if e <= outlier_threshold_px]
        if keep:
            obj_points, img_points, used_files = map(list, zip(*keep))
            log(f"  removed {len(outliers)} outlier frame(s), re-running with {len(used_files)} frames")
            results = _try_all_models(obj_points, img_points, image_size, K0, log)
            best = _select_best_model(results)

    K, dist, rms = best["K"], best["dist"], best["rms"]
    log(f"  model={best['label']}  rms={rms:.4f}px  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    return {
        "K": K, "dist": dist, "rms": float(rms),
        "image_size": image_size, "num_images_used": len(used_files),
        "summary": {
            "model": best["label"],
            "rms_reprojection_error_px": float(rms),
            "image_size_wh": list(image_size),
            "num_images_used": len(used_files),
            "camera_matrix": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                               "cx": float(K[0, 2]), "cy": float(K[1, 2])},
            "distortion_coefficients": {
                "k1": float(dist.flatten()[0]), "k2": float(dist.flatten()[1]),
                "p1": float(dist.flatten()[2]), "p2": float(dist.flatten()[3]),
                "k3": float(dist.flatten()[4]),
            },
        },
    }


# =============================================================================
# STAGE 3 — extrinsic calibration (relative to a reference camera)
# =============================================================================

def _get_frame_stems(folder):
    folder = Path(folder)
    stems = set()
    for ext in IMAGE_EXTENSIONS:
        for p in folder.glob(f"*{ext}"):
            stems.add(p.stem)
        for p in folder.glob(f"*{ext.upper()}"):
            stems.add(p.stem)
    return stems


def _find_frame_path(folder, stem):
    folder = Path(folder)
    for ext in IMAGE_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
        p = folder / f"{stem}{ext.upper()}"
        if p.exists():
            return p
    return None


def _rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _matrix_to_rvec_tvec(T):
    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


def _reprojection_error(objp, imgp, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    imgp = imgp.astype(np.float64)
    return cv2.norm(imgp, projected, cv2.NORM_L2) / len(imgp)


def calibrate_extrinsics(cameras, board_w, board_h, square_size, reference_name, log=print):
    """
    cameras: list of {"name": str, "frames_dir": path, "intrinsics": {"K": ndarray, "dist": ndarray}}
    Returns the full summary dict, matching 2026_04_14.json's schema exactly.
    Ported from extrinsic_parameters.py steps 1-6.
    """
    names = [c["name"] for c in cameras]
    if reference_name not in names:
        raise ValueError(f"reference_name '{reference_name}' not among camera names {names}")
    ref_idx = names.index(reference_name)
    n_cams = len(cameras)

    # step2: match frames
    all_stems = [_get_frame_stems(c["frames_dir"]) for c in cameras]
    common = sorted(set.intersection(*all_stems))
    log(f"  common synchronized frames across all {n_cams} cameras: {len(common)}")
    if len(common) < 5:
        raise RuntimeError(f"Only {len(common)} common frames found -- need at least 5")

    # step3: detect + solvePnP
    criteria = _corner_criteria()
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp = (objp * square_size).astype(np.float64)

    per_cam_rvecs = [[] for _ in range(n_cams)]
    per_cam_tvecs = [[] for _ in range(n_cams)]
    per_cam_imgpts = [[] for _ in range(n_cams)]
    used_stems = []

    for stem in common:
        corners_all = []
        for c in cameras:
            fpath = _find_frame_path(c["frames_dir"], stem)
            if fpath is None:
                corners_all.append(None)
                continue
            img = cv2.imread(str(fpath))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (board_w, board_h),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            corners_all.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) if found else None)

        if any(c is None for c in corners_all):
            continue

        frame_rvecs, frame_tvecs, frame_ok = [], [], True
        for ci, c in enumerate(cameras):
            K, dist = c["intrinsics"]["K"], c["intrinsics"]["dist"]
            ok, rvec, tvec = cv2.solvePnP(objp, corners_all[ci], K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                frame_ok = False
                break
            frame_rvecs.append(rvec)
            frame_tvecs.append(tvec)
        if not frame_ok:
            continue

        for ci in range(n_cams):
            per_cam_rvecs[ci].append(frame_rvecs[ci])
            per_cam_tvecs[ci].append(frame_tvecs[ci])
            per_cam_imgpts[ci].append(corners_all[ci])
        used_stems.append(stem)

    log(f"  used {len(used_stems)} frames for extrinsic solve")
    if len(used_stems) < 5:
        raise RuntimeError(f"Only {len(used_stems)} frames had the board detected in all cameras")

    # step4: relative transforms
    n_frames = len(used_stems)
    rel_transforms = [[] for _ in range(n_cams)]
    for fi in range(n_frames):
        T_ref = _rvec_tvec_to_matrix(per_cam_rvecs[ref_idx][fi], per_cam_tvecs[ref_idx][fi])
        T_ref_inv = np.linalg.inv(T_ref)
        for ci in range(n_cams):
            T_ci = _rvec_tvec_to_matrix(per_cam_rvecs[ci][fi], per_cam_tvecs[ci][fi])
            rel_transforms[ci].append(T_ci @ T_ref_inv)

    extrinsics = []
    for ci in range(n_cams):
        transforms = rel_transforms[ci]
        t_all = np.array([T[:3, 3] for T in transforms])
        t_mean = t_all.mean(axis=0)
        t_std = t_all.std(axis=0)

        rots = Rscipy.from_matrix([T[:3, :3] for T in transforms])
        r_mean = rots.mean()
        angle_errors = [(r_mean.inv() * r).magnitude() * 180.0 / np.pi for r in rots]
        rot_spread = float(np.mean(angle_errors))

        R_mean = r_mean.as_matrix()
        rvec_mean, _ = cv2.Rodrigues(R_mean)
        tvec_mean = t_mean.reshape(3, 1)

        extrinsics.append({
            "name": cameras[ci]["name"], "rvec": rvec_mean, "tvec": tvec_mean,
            "R": R_mean, "t_std": t_std, "rot_spread_deg": rot_spread,
        })
        cam_pos = (-R_mean.T @ t_mean)
        log(f"  {cameras[ci]['name']:15s} cam_pos=[{cam_pos[0]:+.4f},{cam_pos[1]:+.4f},{cam_pos[2]:+.4f}]m "
            f"rot_spread={rot_spread:.3f}deg")

    # step5: validation
    all_errors = [[] for _ in range(n_cams)]
    for fi in range(n_frames):
        T_board_to_ref = _rvec_tvec_to_matrix(per_cam_rvecs[ref_idx][fi], per_cam_tvecs[ref_idx][fi])
        for ci in range(n_cams):
            T_rel = _rvec_tvec_to_matrix(extrinsics[ci]["rvec"], extrinsics[ci]["tvec"])
            T_board_to_ci = T_rel @ T_board_to_ref
            rvec_ci, tvec_ci = _matrix_to_rvec_tvec(T_board_to_ci)
            err = _reprojection_error(objp, per_cam_imgpts[ci][fi], rvec_ci, tvec_ci,
                                       cameras[ci]["intrinsics"]["K"], cameras[ci]["intrinsics"]["dist"])
            all_errors[ci].append(err)
    overall_mean = float(np.mean([e for errs in all_errors for e in errs]))
    log(f"  overall validation reprojection error: {overall_mean:.4f} px")

    # step6: build JSON summary (exact schema of 2026_04_14.json)
    summary = {
        "convention": {
            "rotation_matrix": "R_world_to_cam: p_cam = R @ p_world + t",
            "translation_m": "standard OpenCV t vector (NOT camera world position)",
            "cam_pos_world_m": "camera position in world coords = -R^T @ t",
        },
        "reference_camera": reference_name,
        "num_frames_used": len(used_stems),
        "used_stems": used_stems,
        "board": {"inner_corners_wh": [board_w, board_h], "square_size_m": square_size},
        "cameras": [],
    }
    for ci, (e, c) in enumerate(zip(extrinsics, cameras)):
        R = e["R"]
        t = e["tvec"].flatten()
        cam_pos = (-R.T @ t).tolist()
        euler_deg = Rscipy.from_matrix(R).as_euler("xyz", degrees=True).tolist()
        K, dist = c["intrinsics"]["K"], c["intrinsics"]["dist"]
        summary["cameras"].append({
            "name": e["name"],
            "is_reference": ci == ref_idx,
            "rotation_matrix": R.tolist(),
            "translation_m": t.tolist(),
            "cam_pos_world_m": cam_pos,
            "euler_angles_xyz_deg": euler_deg,
            "consistency": {
                "translation_std_m": e["t_std"].tolist(),
                "rotation_spread_deg": e["rot_spread_deg"],
            },
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "k1": float(dist.flatten()[0]),
            },
        })

    return summary


# =============================================================================
# STAGE 4 — rig visualization (table-aligned display frame)
# =============================================================================

def _load_cams_for_plot(summary):
    cams = []
    for cam in summary["cameras"]:
        cams.append({
            "name": cam["name"],
            "pos": np.array(cam["cam_pos_world_m"], dtype=np.float64),
            "R": np.array(cam["rotation_matrix"], dtype=np.float64),
            "fx": cam["intrinsics"]["fx"], "fy": cam["intrinsics"]["fy"],
            "cx": cam["intrinsics"]["cx"], "cy": cam["intrinsics"]["cy"],
        })
    return cams


def _table_aligned_basis(cams):
    """Derive a physically-meaningful 'up' from the average viewing direction
    of all cameras (since the raw calibration world frame = an arbitrary
    reference camera orientation, not necessarily vertical)."""
    forwards = np.array([c["R"].T[:, 2] for c in cams])
    avg_forward = forwards.mean(axis=0)
    avg_forward /= np.linalg.norm(avg_forward)

    up_disp = -avg_forward  # cameras look down and inward -> "up" is opposite avg view dir
    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up_disp, world_ref)) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])
    x_disp = np.cross(up_disp, world_ref)
    x_disp /= np.linalg.norm(x_disp)
    y_disp = np.cross(up_disp, x_disp)
    return x_disp, y_disp, up_disp


def _estimate_table_center(cams):
    """Least-squares intersection of all cameras' optical axes."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c["R"].T[:, 2]
        d = d / np.linalg.norm(d)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ c["pos"]
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        center = np.mean([c["pos"] for c in cams], axis=0)
    return center


def _to_display_frame(p, center, x_disp, y_disp, up_disp):
    rel = p - center
    return np.array([np.dot(rel, x_disp), np.dot(rel, y_disp), np.dot(rel, up_disp)])


def plot_rig(summary, out_dir, arrow_len=0.15, log=print):
    """Produces a perspective view and a strict top-down view of the camera
    rig in the table-aligned display frame. Ported from the rig-plot script
    used to make camera_rig_3d_scene.png / camera_rig_top_view.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = _load_cams_for_plot(summary)
    x_disp, y_disp, up_disp = _table_aligned_basis(cams)
    center = _estimate_table_center(cams)

    disp = []
    for c in cams:
        pos_d = _to_display_frame(c["pos"], center, x_disp, y_disp, up_disp)
        forward_world = c["R"].T[:, 2]
        forward_d = np.array([np.dot(forward_world, x_disp), np.dot(forward_world, y_disp), np.dot(forward_world, up_disp)])
        forward_d /= np.linalg.norm(forward_d)
        disp.append({"name": c["name"], "pos": pos_d, "forward": forward_d})

    colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#a8dadc", "#6d6875"]

    def draw(ax, hide_z_ticks, elev, azim, title):
        positions = np.array([d["pos"] for d in disp])
        # checkered table plane at z=0
        span = max(np.abs(positions[:, :2]).max() * 1.4, 0.3)
        n_tiles = 8
        tile = span * 2 / n_tiles
        for i in range(n_tiles):
            for j in range(n_tiles):
                if (i + j) % 2 == 0:
                    x0, y0 = -span + i * tile, -span + j * tile
                    xs = [x0, x0 + tile, x0 + tile, x0]
                    ys = [y0, y0, y0 + tile, y0 + tile]
                    ax.plot_trisurf(xs, ys, [0, 0, 0, 0], color="#d8cbb0", alpha=0.25, shade=False)

        for i, d in enumerate(disp):
            col = colors[i % len(colors)]
            pos = d["pos"]
            ax.scatter(*pos, color=col, s=110, zorder=5)
            ax.quiver(pos[0], pos[1], pos[2],
                      d["forward"][0] * arrow_len, d["forward"][1] * arrow_len, d["forward"][2] * arrow_len,
                      color=col, linewidth=2, arrow_length_ratio=0.25)
            ax.text(pos[0], pos[1], pos[2] + 0.03, d["name"].replace("camera_", "cam"),
                    fontsize=8, color=col, ha="center")

        order = list(range(len(disp))) + [0]
        for a, b in zip(order[:-1], order[1:]):
            pa, pb = disp[a]["pos"], disp[b]["pos"]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], "k--", alpha=0.15, linewidth=1)

        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-0.1, span * 1.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if not hide_z_ticks:
            ax.set_zlabel("height above table (m)")
        else:
            ax.set_zticks([])
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    fig1 = plt.figure(figsize=(9, 8))
    ax1 = fig1.add_subplot(111, projection="3d")
    draw(ax1, False, elev=28, azim=-55, title="Camera rig — perspective view")
    perspective_path = out_dir / "camera_rig_perspective.png"
    fig1.tight_layout()
    fig1.savefig(str(perspective_path), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 8))
    ax2 = fig2.add_subplot(111, projection="3d")
    draw(ax2, True, elev=89.9, azim=-90, title="Camera rig — top-down view")
    top_path = out_dir / "camera_rig_top_view.png"
    fig2.tight_layout()
    fig2.savefig(str(top_path), dpi=150)
    plt.close(fig2)

    log(f"  saved {perspective_path}")
    log(f"  saved {top_path}")
    return str(perspective_path), str(top_path)
