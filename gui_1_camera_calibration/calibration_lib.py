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

Error metric: every reprojection error in this module is the RMS distance per
corner in pixels, sqrt(mean(dx^2 + dy^2)). Earlier versions divided the L2
norm over ALL corners by the corner count (the OpenCV tutorial idiom), which
under-reports the per-corner error by sqrt(n_corners) (~7.3x for a 9x6 board).
Numbers from files written before this change are therefore not comparable.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
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


def _unit_objp(board_w, board_h):
    """Board corner coordinates in units of one square (z = 0)."""
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    return objp


def _rough_K(gray):
    """Crude pinhole guess (f = image width, principal point at centre) used
    only to judge board-pose diversity before real intrinsics exist."""
    h, w = gray.shape
    return np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)


def _clear_images(folder):
    """Delete every image file in `folder` (not recursive). Returns the count."""
    folder = Path(folder)
    n = 0
    for ext in IMAGE_EXTENSIONS:
        for old in list(folder.glob(f"*{ext}")) + list(folder.glob(f"*{ext.upper()}")):
            old.unlink()
            n += 1
    return n


def clear_frame_tree(folder):
    """Delete every image below `folder` (recursive) and remove the emptied
    sub-folders. Used before pooling several intrinsic sessions into one
    camera folder, so frames of an earlier run never leak into the solve."""
    folder = Path(folder)
    if not folder.exists():
        return 0
    n = 0
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            p.unlink()
            n += 1
    for d in sorted((d for d in folder.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        try:
            d.rmdir()
        except OSError:
            pass
    return n


def _is_pose_diverse(rvec, tvec, kept_poses, min_pose_distance, min_trans_distance):
    """A pose is rejected only when it is close to an already-kept pose in
    BOTH rotation (rad, rvec norm) AND translation (same units as the objp
    that produced tvec)."""
    for rv, tv in kept_poses:
        if (np.linalg.norm(rvec - rv) < min_pose_distance
                and np.linalg.norm(tvec - tv) < min_trans_distance):
            return False
    return True


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

    # calibrate_intrinsics() uses every image in this folder, so frames from
    # an earlier run (possibly of a different video) must not mix in.
    stale = _clear_images(out_path)
    if stale:
        log(f"  removed {stale} stale frame(s) from a previous run in {out_path}")

    criteria = _corner_criteria()
    objp = _unit_objp(board_w, board_h)

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
            rough_K = _rough_K(gray)

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

        if not _is_pose_diverse(rvec, tvec, kept_poses, min_pose_distance, min_trans_distance):
            continue

        filename = out_path / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(filename), frame)
        kept_poses.append((rvec, tvec))
        saved_count += 1

    cap.release()
    log(f"  extracted {saved_count} diverse frames -> {out_path}")
    return saved_count


def extract_synced_frames(video_paths, out_root, board_w, board_h,
                           sample_every_n=5, max_frames=60,
                           min_frame_gap=30, min_pose_distance=0.15,
                           min_trans_distance=0.05, square_size=None,
                           diversity_camera=None, log=print):
    """Walk N camera videos in lockstep, keep a frame index only when the
    checkerboard is detected in ALL cameras simultaneously. Saves into
    <out_root>/<camera_name>/frame_NNNNNN.png -- the folder structure
    calibrate_extrinsics() expects (one synchronized frame set per stem,
    common across every camera). This is new logic (no equivalent existed
    in the original scripts, which assumed synced frames were already on
    disk) but reuses the same corner-detection code as extract_diverse_frames.

    Two filters keep the kept frame sets from being near-duplicates (a run of
    consecutive frames of a resting board counts as ONE board pose for the
    extrinsic solve, but would otherwise eat several slots of max_frames and
    bias the pose average):
      min_frame_gap      : minimum number of video frames between two kept sets
      min_pose_distance /
      min_trans_distance : same pose-diversity test as extract_diverse_frames,
                           evaluated on `diversity_camera` (default: first
                           camera). Translation is in metres when square_size
                           is given, otherwise in board squares.

    video_paths: dict camera_name -> video file path
    """
    names = list(video_paths.keys())
    if diversity_camera is None or diversity_camera not in names:
        diversity_camera = names[0]
    objp_div = _unit_objp(board_w, board_h)
    if square_size:
        objp_div = objp_div * float(square_size)
    rough_K = None
    kept_poses = []
    last_saved_idx = None
    skipped_gap = 0
    skipped_similar = 0
    caps = {name: cv2.VideoCapture(str(video_paths[name])) for name in names}
    for name, cap in caps.items():
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for {name}: {video_paths[name]}")

    # calibrate_extrinsics() uses every stem it finds in these folders, so
    # frames left over from an earlier run (possibly extracted with a looser
    # gap / diversity setting) must not leak into this one.
    out_dirs = {}
    stale = 0
    for name in names:
        d = Path(out_root) / name
        d.mkdir(parents=True, exist_ok=True)
        stale += _clear_images(d)
        out_dirs[name] = d
    if stale:
        log(f"  removed {stale} stale frame(s) from a previous run in {out_root}")

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
        if last_saved_idx is not None and frame_idx - last_saved_idx < min_frame_gap:
            skipped_gap += 1
            continue

        found_all = True
        div_corners = None
        div_gray = None
        for name, frame in frames.items():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (board_w, board_h),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not found:
                found_all = False
                break
            if name == diversity_camera:
                div_corners, div_gray = corners, gray

        if not found_all:
            continue

        # pose-diversity gate (same test as the intrinsic frame extractor)
        if div_corners is not None:
            if rough_K is None:
                rough_K = _rough_K(div_gray)
            div_corners = cv2.cornerSubPix(div_gray, div_corners, (11, 11), (-1, -1), criteria)
            ok, rvec, tvec = cv2.solvePnP(objp_div, div_corners, rough_K, None, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                if not _is_pose_diverse(rvec, tvec, kept_poses, min_pose_distance, min_trans_distance):
                    skipped_similar += 1
                    continue
                kept_poses.append((rvec, tvec))

        stem = f"frame_{frame_idx:06d}"
        for name, frame in frames.items():
            cv2.imwrite(str(out_dirs[name] / f"{stem}.png"), frame)
        saved_count += 1
        last_saved_idx = frame_idx
        log(f"  synced frame {stem}  ({saved_count}/{max_frames})")

    for cap in caps.values():
        cap.release()

    log(f"  extracted {saved_count} synchronized frame sets -> {out_root}  "
        f"(skipped {skipped_gap} within {min_frame_gap} frames of the previous set, "
        f"{skipped_similar} with a near-identical board pose)")
    return saved_count


# =============================================================================
# STAGE 2 — intrinsic calibration (per camera)
# =============================================================================

def _load_images(folder):
    """All images in `folder`, recursively, so several recording sessions can
    be pooled as sub-folders (session_1/, session_2/, ...) of one camera."""
    folder = Path(folder)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(sorted(folder.rglob(f"*{ext}")))
        images.extend(sorted(folder.rglob(f"*{ext.upper()}")))
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
        # parent/name keeps pooled sessions apart (session_1/frame_000030.png)
        used_files.append(f"{path.parent.name}/{path.name}")

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


def _rms_per_corner(imgp, projected):
    """RMS reprojection distance per corner in px: sqrt(mean(dx^2 + dy^2)).
    This is the same quantity cv2.calibrateCamera reports as its RMS."""
    d = (np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
         - np.asarray(projected, dtype=np.float64).reshape(-1, 2))
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def _per_image_errors(obj_points, img_points, rvecs, tvecs, K, dist):
    errors = []
    for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        errors.append(_rms_per_corner(imgp, projected))
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


def check_intrinsics_consistency(intrinsics_by_cam, sensor_specs=None,
                                 rel_tol_median=0.03, rel_tol_prior=0.08,
                                 log=print):
    """Cross-camera plausibility check for a rig of identical lenses.

    Flags a camera whose focal length deviates from the rig median by more
    than rel_tol_median, or from the sensor-spec prior by more than
    rel_tol_prior, and a camera whose k1 is ~0 while the others agree on a
    clearly non-zero value (the classic fx/k1 trade-off of an under-
    constrained solve). Returns a list of warning strings (empty = fine)."""
    warnings = []
    names = sorted(intrinsics_by_cam)
    if len(names) < 2:
        return warnings
    fx = {n: float(intrinsics_by_cam[n]["K"][0, 0]) for n in names}
    k1 = {n: float(np.asarray(intrinsics_by_cam[n]["dist"]).flatten()[0]) for n in names}
    fx_med = float(np.median(list(fx.values())))
    k1_med = float(np.median(list(k1.values())))

    for n in names:
        dev = (fx[n] - fx_med) / fx_med
        if abs(dev) > rel_tol_median:
            warnings.append(
                f"{n}: fx={fx[n]:.1f}px deviates {dev*100:+.1f}% from the rig median "
                f"({fx_med:.1f}px); identical lenses should agree within ~{rel_tol_median*100:.0f}%. "
                f"Check board coverage / re-record this camera's intrinsic frames."
            )
        if abs(k1_med) > 0.005 and abs(k1[n]) < 0.2 * abs(k1_med):
            warnings.append(
                f"{n}: k1={k1[n]:+.4f} is near zero while the rig median is {k1_med:+.4f}; "
                f"distortion is probably absorbed into fx for this camera."
            )

    image_size = None
    for n in names:
        if intrinsics_by_cam[n].get("image_size") is not None:
            image_size = intrinsics_by_cam[n]["image_size"]
            break
    K0 = _build_initial_K(image_size, sensor_specs) if image_size is not None else None
    if K0 is not None:
        f_prior = float(K0[0, 0])
        for n in names:
            dev = (fx[n] - f_prior) / f_prior
            if abs(dev) > rel_tol_prior:
                warnings.append(
                    f"{n}: fx={fx[n]:.1f}px is {dev*100:+.1f}% off the lens/sensor prior "
                    f"({f_prior:.1f}px). Either the sensor specs are wrong or the solve drifted."
                )

    for w in warnings:
        log(f"  WARNING intrinsics: {w}")
    if not warnings:
        log(f"  intrinsics consistent across cameras (fx median {fx_med:.1f}px, "
            f"spread {max(fx.values())-min(fx.values()):.1f}px)")
    return warnings


def load_intrinsics_from_json(json_path, camera_names, log=print):
    """Reuse intrinsics from an earlier pipeline output (the extrinsics
    summary JSON) so stage 1 can be skipped. Missing distortion terms are
    taken as zero (older files only stored k1)."""
    with open(json_path) as f:
        data = json.load(f)
    by_name = {c["name"]: c for c in data.get("cameras", [])}
    missing = [n for n in camera_names if n not in by_name]
    if missing:
        raise ValueError(f"{json_path} has no intrinsics for camera(s): {', '.join(missing)}")

    out = {}
    for n in camera_names:
        intr = by_name[n]["intrinsics"]
        K = np.array([[intr["fx"], 0.0, intr["cx"]],
                      [0.0, intr["fy"], intr["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([[intr.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")]], dtype=np.float64)
        out[n] = {
            "K": K, "dist": dist,
            "rms": intr.get("rms_reprojection_error_px"),
            "image_size": tuple(intr["image_size_wh"]) if intr.get("image_size_wh") else None,
            "num_images_used": intr.get("num_images_used"),
            "summary": {"model": intr.get("model", "reused"), "source": str(json_path)},
        }
        log(f"  [{n}] reusing intrinsics from {Path(json_path).name}: fx={K[0,0]:.1f} k1={dist[0,0]:+.4f}")
    return out


def intrinsics_summary(intrinsics_by_cam, board_w, board_h, square_size,
                       warnings=None, source_videos=None):
    """JSON-serializable result of an intrinsics-only run. Uses the same
    per-camera "intrinsics" block as the full rig JSON, so the file can be
    fed back through load_intrinsics_from_json() for an extrinsics-only run."""
    cams = []
    for name in sorted(intrinsics_by_cam):
        entry = {"name": name, "intrinsics": _intrinsics_block(intrinsics_by_cam[name])}
        if source_videos and name in source_videos:
            src = source_videos[name]
            if isinstance(src, (list, tuple)):
                # several pooled sessions
                entry["source_videos"] = [str(s) for s in src]
                entry["source_video"] = str(src[0]) if len(src) == 1 else " | ".join(Path(s).name for s in src)
            else:
                entry["source_video"] = str(src)
        cams.append(entry)
    return {
        "type": "intrinsics",
        "board": {"inner_corners_wh": [board_w, board_h], "square_size_m": square_size},
        "warnings": list(warnings or []),
        "cameras": cams,
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
    return _rms_per_corner(imgp, projected)


def _intrinsics_block(intr):
    """JSON block for one camera's intrinsics. fx/fy/cx/cy/k1 keep the
    original schema; the remaining keys are additive (full distortion vector
    and the stage-1 quality figures) so downstream readers keep working."""
    K = np.asarray(intr["K"], dtype=np.float64)
    d = np.asarray(intr["dist"], dtype=np.float64).flatten()
    d = np.concatenate([d, np.zeros(max(0, 5 - d.size))])[:5]
    block = {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "k1": float(d[0]), "k2": float(d[1]), "p1": float(d[2]), "p2": float(d[3]), "k3": float(d[4]),
    }
    if intr.get("rms") is not None:
        block["rms_reprojection_error_px"] = float(intr["rms"])
    if intr.get("fx_before_bundle_adjustment") is not None:
        block["fx_before_bundle_adjustment"] = float(intr["fx_before_bundle_adjustment"])
    if intr.get("num_images_used") is not None:
        block["num_images_used"] = int(intr["num_images_used"])
    if intr.get("image_size") is not None:
        block["image_size_wh"] = [int(v) for v in intr["image_size"]]
    summ = intr.get("summary") or {}
    if summ.get("model"):
        block["model"] = summ["model"]
    if summ.get("source"):
        block["source"] = summ["source"]
    return block


def _average_rig(per_cam_rvecs, per_cam_tvecs, frame_ids, ref_idx):
    """Per-frame relative transforms T_cam <- T_ref, averaged over `frame_ids`.
    Returns one dict per camera: T (4x4 ref->cam), t_std (m), rot_spread_deg."""
    out = []
    for ci in range(len(per_cam_rvecs)):
        transforms = []
        for fi in frame_ids:
            T_ref = _rvec_tvec_to_matrix(per_cam_rvecs[ref_idx][fi], per_cam_tvecs[ref_idx][fi])
            T_ci = _rvec_tvec_to_matrix(per_cam_rvecs[ci][fi], per_cam_tvecs[ci][fi])
            transforms.append(T_ci @ np.linalg.inv(T_ref))
        t_all = np.array([T[:3, 3] for T in transforms])
        rots = Rscipy.from_matrix([T[:3, :3] for T in transforms])
        r_mean = rots.mean()
        angle_errors = [(r_mean.inv() * r).magnitude() * 180.0 / np.pi for r in rots]
        T = np.eye(4)
        T[:3, :3] = r_mean.as_matrix()
        T[:3, 3] = t_all.mean(axis=0)
        out.append({"T": T, "t_std": t_all.std(axis=0), "rot_spread_deg": float(np.mean(angle_errors))})
    return out


def _rig_errors(rel_T, board_T, per_cam_imgpts, frame_ids, objp, intrinsics):
    """(n_cams, n_frames) RMS-per-corner error of board poses board_T[j]
    (board->ref) pushed through the rig transforms rel_T[ci] (ref->cam)."""
    n_cams = len(rel_T)
    E = np.zeros((n_cams, len(frame_ids)))
    for j, fi in enumerate(frame_ids):
        for ci in range(n_cams):
            rvec, tvec = _matrix_to_rvec_tvec(rel_T[ci] @ board_T[j])
            E[ci, j] = _reprojection_error(objp, per_cam_imgpts[ci][fi], rvec, tvec,
                                           intrinsics[ci]["K"], intrinsics[ci]["dist"])
    return E


def _error_stats(E_row, frame_ids, used_stems):
    errs = np.asarray(E_row, dtype=np.float64)
    return {
        "mean_px": float(errs.mean()),
        "rms_px": float(np.sqrt((errs ** 2).mean())),
        "max_px": float(errs.max()),
        "worst_frame": used_stems[frame_ids[int(errs.argmax())]],
    }


def _project_points_np(P_cam, K, dist):
    """Vectorised OpenCV pinhole + Brown distortion (k1,k2,p1,p2,k3), zero skew.
    P_cam: (..., 3) points in the camera frame -> (..., 2) pixels."""
    x = P_cam[..., 0] / P_cam[..., 2]
    y = P_cam[..., 1] / P_cam[..., 2]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=-1)


def bundle_adjust_rig(objp, imgpts, rel_T0, board_T0, intrinsics, ref_idx,
                      refine_focal=False, huber_px=1.0, max_nfev=100, tol=1e-6,
                      verbose=0, log=print):
    """Joint refinement of the rig: all non-reference camera poses (ref->cam)
    and all board poses (board->ref) are optimised together so that the
    projected corners match the detections in EVERY camera at once. The
    averaged per-frame solve only satisfies each frame separately and then
    blurs the result; here the inconsistencies are minimised globally, with a
    Huber loss (scale `huber_px`) so a few bad corners do not pull the solve.

    refine_focal=True adds one log-scale factor per camera applied to fx AND
    fy (aspect ratio preserved, principal point and distortion untouched).
    Use it as a diagnostic: a shift of more than ~0.5 % says the fixed
    intrinsics were limiting the solve.

    objp: (N,3) board corners (m); imgpts[ci][j]: detections of used frame j
    in camera ci; rel_T0[ci], board_T0[j]: 4x4 initial transforms;
    intrinsics[ci]: {"K", "dist"}.

    Returns dict(rel_T, board_T, K, focal_scale, errors (C,F) RMS per corner,
                 rms_before_px, rms_after_px, n_iterations, status).
    """
    n_cams, n_frames, n_pts = len(rel_T0), len(board_T0), objp.shape[0]
    cam_ids = [ci for ci in range(n_cams) if ci != ref_idx]
    objp = np.asarray(objp, dtype=np.float64)
    obs = np.stack([np.stack([np.asarray(imgpts[ci][j], dtype=np.float64).reshape(-1, 2)
                              for j in range(n_frames)]) for ci in range(n_cams)])     # (C,F,N,2)
    K0 = [np.asarray(intrinsics[ci]["K"], dtype=np.float64) for ci in range(n_cams)]
    dist = [np.concatenate([np.asarray(intrinsics[ci]["dist"], dtype=np.float64).flatten(), np.zeros(5)])[:5]
            for ci in range(n_cams)]

    n_cam_par, n_frame_par = 6 * len(cam_ids), 6 * n_frames

    def pack():
        x = []
        for ci in cam_ids:
            rv, tv = _matrix_to_rvec_tvec(rel_T0[ci])
            x += [rv.flatten(), tv.flatten()]
        for j in range(n_frames):
            rv, tv = _matrix_to_rvec_tvec(board_T0[j])
            x += [rv.flatten(), tv.flatten()]
        if refine_focal:
            x.append(np.zeros(n_cams))
        return np.concatenate(x)

    def unpack(x):
        cam_par = x[:n_cam_par].reshape(-1, 6)
        frame_par = x[n_cam_par:n_cam_par + n_frame_par].reshape(-1, 6)
        scales = np.exp(x[n_cam_par + n_frame_par:]) if refine_focal else np.ones(n_cams)
        R_rel = np.tile(np.eye(3), (n_cams, 1, 1))
        t_rel = np.zeros((n_cams, 3))
        if cam_ids:
            R_rel[cam_ids] = Rscipy.from_rotvec(cam_par[:, :3]).as_matrix()
            t_rel[cam_ids] = cam_par[:, 3:]
        R_b = Rscipy.from_rotvec(frame_par[:, :3]).as_matrix()
        t_b = frame_par[:, 3:]
        return R_rel, t_rel, R_b, t_b, scales

    def residuals(x):
        R_rel, t_rel, R_b, t_b, scales = unpack(x)
        P_ref = np.einsum("fij,nj->fni", R_b, objp) + t_b[:, None, :]        # (F,N,3) in the ref camera
        res = np.empty((n_cams, n_frames, n_pts, 2))
        for ci in range(n_cams):
            P_c = np.einsum("ij,fnj->fni", R_rel[ci], P_ref) + t_rel[ci]
            K = K0[ci].copy()
            K[0, 0] *= scales[ci]
            K[1, 1] *= scales[ci]
            res[ci] = _project_points_np(P_c, K, dist[ci]) - obs[ci]
        return res.ravel()

    # Jacobian sparsity: block (ci, j) depends only on camera ci's pose, board
    # pose j and (optionally) camera ci's focal scale. Lets least_squares
    # build the finite-difference Jacobian in ~13 evaluations instead of ~500.
    n_par = n_cam_par + n_frame_par + (n_cams if refine_focal else 0)
    block = 2 * n_pts
    S = lil_matrix((n_cams * n_frames * block, n_par), dtype=np.int8)
    for ci in range(n_cams):
        for j in range(n_frames):
            r0 = (ci * n_frames + j) * block
            rows = slice(r0, r0 + block)
            if ci != ref_idx:
                k = cam_ids.index(ci)
                S[rows, 6 * k:6 * k + 6] = 1
            S[rows, n_cam_par + 6 * j:n_cam_par + 6 * j + 6] = 1
            if refine_focal:
                S[rows, n_cam_par + n_frame_par + ci] = 1

    def rms_all(r):
        return float(np.sqrt((r.reshape(-1, 2) ** 2).sum(axis=1).mean()))

    x0 = pack()
    rms_before = rms_all(residuals(x0))
    # Two stages: (1) plain least squares converges in a handful of Gauss-Newton
    # steps from the averaged initialisation; (2) the Huber pass starts from
    # that solution and only has to down-weight the bad corners. Starting the
    # robust loss from far away is very slow because every residual is in the
    # linear regime and gets down-weighted.
    # tol: residuals are pixels and the cost is O(1e3..1e5), so the default
    # 1e-8 relative tolerances only stop at max_nfev; 1e-6 stops once the RMS
    # changes by well under 0.001 px.
    common = dict(jac_sparsity=S, x_scale="jac", method="trf", ftol=tol, xtol=tol, gtol=tol, verbose=verbose)
    sol_lin = least_squares(residuals, x0, loss="linear", max_nfev=max_nfev, **common)
    # the robust pass creeps slowly while a gross outlier frame is still in the
    # set (it gets dropped by the caller afterwards), so cap it
    sol = least_squares(residuals, sol_lin.x, loss="huber", f_scale=huber_px,
                        max_nfev=min(max_nfev, 30), **common)
    nfev = int(sol_lin.nfev + sol.nfev)
    R_rel, t_rel, R_b, t_b, scales = unpack(sol.x)
    res = residuals(sol.x).reshape(n_cams, n_frames, n_pts, 2)   # sol.fun is loss-scaled; recompute raw
    errors = np.sqrt((res ** 2).sum(axis=-1).mean(axis=-1))                # (C,F)

    rel_T, board_T, K_out = [], [], []
    for ci in range(n_cams):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_rel[ci], t_rel[ci]
        rel_T.append(T)
        K = K0[ci].copy()
        K[0, 0] *= scales[ci]
        K[1, 1] *= scales[ci]
        K_out.append(K)
    for j in range(n_frames):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_b[j], t_b[j]
        board_T.append(T)

    rms_after = rms_all(res)
    log(f"  bundle adjustment: {n_cams} cameras x {n_frames} frames x {n_pts} corners, "
        f"{n_par} parameters, {nfev} evaluations -> RMS per corner "
        f"{rms_before:.3f}px -> {rms_after:.3f}px"
        + (f"  (focal scales {', '.join(f'{(s-1)*100:+.2f}%' for s in scales)})" if refine_focal else ""))
    return {
        "rel_T": rel_T, "board_T": board_T, "K": K_out, "focal_scale": scales,
        "errors": errors, "rms_before_px": rms_before, "rms_after_px": rms_after,
        "n_evaluations": nfev, "status": int(sol.status), "message": str(sol.message),
    }


def calibrate_extrinsics(cameras, board_w, board_h, square_size, reference_name,
                         frame_outlier_px=3.0, bundle_adjust=True, refine_focal=False,
                         huber_px=1.0, log=print):
    """
    cameras: list of {"name": str, "frames_dir": path, "intrinsics": {"K": ndarray, "dist": ndarray, ...}}
             (extra intrinsics keys rms / num_images_used / image_size / summary are copied
             into the output when present -- pass the dict returned by calibrate_intrinsics)
    frame_outlier_px : synced frame sets whose RMS error (over all cameras) exceeds
                       max(frame_outlier_px, median + 3 MAD) are dropped and the solve is
                       repeated (up to 3 passes). 0 disables.
    bundle_adjust    : refine the averaged rig jointly (see bundle_adjust_rig).
    refine_focal     : also refine fx=fy per camera inside the bundle adjustment.
    Returns the full summary dict: the 2026_04_14.json schema plus additive
    keys "validation", "frame_diversity", "warnings", "dropped_frames",
    "bundle_adjustment", "extrinsics_method" and per-camera
    consistency.reprojection_error_px / full distortion coefficients.
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
    objp = (_unit_objp(board_w, board_h) * square_size).astype(np.float64)

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

    # step4+5: average the per-frame relative transforms (initial rig), validate
    # it by reprojection, optionally refine it by bundle adjustment, then drop
    # frame sets that stay outliers and repeat. All errors: RMS per corner (px).
    intr = [c["intrinsics"] for c in cameras]
    n_frames_all = len(used_stems)
    kept = list(range(n_frames_all))
    dropped = []

    def solve(frame_ids):
        avg = _average_rig(per_cam_rvecs, per_cam_tvecs, frame_ids, ref_idx)
        rel_T0 = [a["T"] for a in avg]
        board_T0 = [_rvec_tvec_to_matrix(per_cam_rvecs[ref_idx][fi], per_cam_tvecs[ref_idx][fi])
                    for fi in frame_ids]
        E_avg = _rig_errors(rel_T0, board_T0, per_cam_imgpts, frame_ids, objp, intr)
        ba = None
        if bundle_adjust:
            imgpts = [[per_cam_imgpts[ci][fi] for fi in frame_ids] for ci in range(n_cams)]
            ba = bundle_adjust_rig(objp, imgpts, rel_T0, board_T0, intr, ref_idx,
                                   refine_focal=refine_focal, huber_px=huber_px, log=log)
        return avg, rel_T0, E_avg, ba

    max_passes = 4
    for it in range(max_passes):
        avg, rel_T0, E_avg, ba = solve(kept)
        E_final = ba["errors"] if ba is not None else E_avg
        # worst camera per frame: a sync slip or misdetection in ONE camera
        # must flag the whole synced frame set
        frame_err = E_final.max(axis=0)
        log(f"  pass {it + 1}: {len(kept)} frames, RMS per corner "
            f"averaged {float(np.sqrt((E_avg ** 2).mean())):.3f}px"
            + (f", bundle adjusted {float(np.sqrt((E_final ** 2).mean())):.3f}px" if ba is not None else ""))
        if frame_outlier_px <= 0 or it == max_passes - 1:
            break
        med = float(np.median(frame_err))
        mad = float(np.median(np.abs(frame_err - med))) * 1.4826
        cut = max(float(frame_outlier_px), med + 3.0 * mad)
        bad = [j for j, e in enumerate(frame_err) if e > cut]
        if not bad or len(kept) - len(bad) < 5:
            break
        for j in bad:
            dropped.append({"stem": used_stems[kept[j]], "error_px": float(frame_err[j])})
            log(f"  dropping {used_stems[kept[j]]}: worst camera RMS {frame_err[j]:.2f}px "
                f"(cut-off {cut:.2f}px)")
        bad_set = set(bad)
        kept = [fi for j, fi in enumerate(kept) if j not in bad_set]

    n_frames = len(kept)
    if ba is not None:
        method = "bundle_adjusted"
        rel_T = ba["rel_T"]
        E_final = ba["errors"]
    else:
        method = "averaged"
        rel_T = rel_T0
        E_final = E_avg

    # final intrinsics (fx/fy refined only when asked)
    final_intr = []
    fx_change_pct = {}
    for ci in range(n_cams):
        d = dict(intr[ci])
        if ba is not None and refine_focal:
            fx0 = float(np.asarray(intr[ci]["K"])[0, 0])
            d["K"] = ba["K"][ci]
            d["fx_before_bundle_adjustment"] = fx0
            summ = dict(d.get("summary") or {})
            summ["model"] = f"{summ.get('model', 'reused')} + fx refined in bundle adjustment"
            d["summary"] = summ
            fx_change_pct[cameras[ci]["name"]] = (float(ba["K"][ci][0, 0]) - fx0) / fx0 * 100.0
        final_intr.append(d)

    extrinsics = []
    for ci in range(n_cams):
        T = rel_T[ci]
        R_mean, t_mean = T[:3, :3], T[:3, 3]
        rvec_mean, _ = cv2.Rodrigues(R_mean)
        extrinsics.append({
            "name": cameras[ci]["name"], "rvec": rvec_mean, "tvec": t_mean.reshape(3, 1),
            "R": R_mean, "t_std": avg[ci]["t_std"], "rot_spread_deg": avg[ci]["rot_spread_deg"],
        })
        cam_pos = (-R_mean.T @ t_mean)
        log(f"  {cameras[ci]['name']:15s} cam_pos=[{cam_pos[0]:+.4f},{cam_pos[1]:+.4f},{cam_pos[2]:+.4f}]m "
            f"rot_spread={avg[ci]['rot_spread_deg']:.3f}deg")

    per_cam_reproj, per_cam_reproj_avg = [], []
    for ci in range(n_cams):
        stats = _error_stats(E_final[ci], kept, used_stems)
        stats_avg = _error_stats(E_avg[ci], kept, used_stems)
        per_cam_reproj.append(stats)
        per_cam_reproj_avg.append(stats_avg)
        log(f"  {cameras[ci]['name']:15s} reprojection RMS/corner: averaged rig mean={stats_avg['mean_px']:.3f}px"
            f"  {method} mean={stats['mean_px']:.3f}px max={stats['max_px']:.3f}px ({stats['worst_frame']})")
    overall_mean = float(E_final.mean())
    overall_rms = float(np.sqrt((E_final ** 2).mean()))
    overall_mean_avg = float(E_avg.mean())
    overall_rms_avg = float(np.sqrt((E_avg ** 2).mean()))
    log(f"  overall reprojection error ({method}): mean={overall_mean:.4f}px rms={overall_rms:.4f}px"
        f"  (averaged rig: mean={overall_mean_avg:.4f}px rms={overall_rms_avg:.4f}px)")

    # step5b: how well do the used frames constrain the solve? Board distance
    # and tilt as seen from the reference camera. A narrow distance range
    # leaves depth (the z-component of translation_std) poorly determined; a
    # narrow tilt range leaves rotation weakly constrained.
    dists, tilts = [], []
    for fi in kept:
        tvec = per_cam_tvecs[ref_idx][fi].flatten()
        Rb, _ = cv2.Rodrigues(per_cam_rvecs[ref_idx][fi])
        dists.append(float(np.linalg.norm(tvec)))
        tilts.append(float(np.degrees(np.arccos(np.clip(abs(Rb[2, 2]), 0.0, 1.0)))))
    dists, tilts = np.array(dists), np.array(tilts)
    frame_diversity = {
        "evaluated_in": reference_name,
        "board_distance_m": {"min": float(dists.min()), "max": float(dists.max()),
                             "range": float(dists.max() - dists.min())},
        "board_tilt_deg": {"min": float(tilts.min()), "max": float(tilts.max()),
                           "range": float(tilts.max() - tilts.min())},
    }
    warnings = []
    if frame_diversity["board_distance_m"]["range"] < 0.20:
        warnings.append(
            f"board distance to {reference_name} only varies by "
            f"{frame_diversity['board_distance_m']['range']*100:.0f}cm across the used frames; "
            f"depth is weakly constrained -- record the board at several distances."
        )
    if frame_diversity["board_tilt_deg"]["range"] < 20.0:
        warnings.append(
            f"board tilt relative to {reference_name} only varies by "
            f"{frame_diversity['board_tilt_deg']['range']:.0f}deg across the used frames; "
            f"rotation is weakly constrained -- tilt the board more."
        )
    worst_t = max(range(n_cams), key=lambda ci: float(np.linalg.norm(extrinsics[ci]["t_std"])))
    if float(np.linalg.norm(extrinsics[worst_t]["t_std"])) > 0.015:
        warnings.append(
            f"{cameras[worst_t]['name']}: per-frame translation scatter is "
            f"{np.linalg.norm(extrinsics[worst_t]['t_std'])*1000:.1f}mm (std) before bundle adjustment; "
            f"expect ~<10mm for a good solve."
        )
    if overall_mean > 1.0:
        warnings.append(
            f"reprojection error is still {overall_mean:.2f}px RMS per corner after the {method} solve "
            f"(target <0.7px). " +
            ("Try 'refine fx' to test whether the fixed intrinsics are the limit."
             if not refine_focal else
             "Even with fx refined; the principal point / distortion model or the corner "
             "detections are probably the limit -- recalibrate intrinsics with better board coverage.")
        )
    for name, pct in fx_change_pct.items():
        if abs(pct) > 0.5:
            warnings.append(
                f"{name}: fx moved {pct:+.2f}% in the bundle adjustment; the fixed intrinsics were "
                f"limiting the solve -- consider recalibrating intrinsics (pool sessions / larger board)."
            )
    if dropped:
        log(f"  dropped {len(dropped)} outlier frame set(s): "
            + ", ".join(f"{d['stem']} ({d['error_px']:.1f}px)" for d in dropped))
    log(f"  board distance {dists.min():.2f}-{dists.max():.2f}m, tilt {tilts.min():.0f}-{tilts.max():.0f}deg "
        f"(seen from {reference_name})")
    for w in warnings:
        log(f"  WARNING extrinsics: {w}")

    # step6: build JSON summary (exact schema of 2026_04_14.json + additive keys)
    summary = {
        "convention": {
            "rotation_matrix": "R_world_to_cam: p_cam = R @ p_world + t",
            "translation_m": "standard OpenCV t vector (NOT camera world position)",
            "cam_pos_world_m": "camera position in world coords = -R^T @ t",
        },
        "reference_camera": reference_name,
        "extrinsics_method": method,
        "num_frames_used": n_frames,
        "used_stems": [used_stems[fi] for fi in kept],
        "dropped_frames": dropped,
        "board": {"inner_corners_wh": [board_w, board_h], "square_size_m": square_size},
        "validation": {
            "description": "per-frame error = RMS corner distance in px (sqrt(mean(dx^2+dy^2))). "
                           "'averaged' = board pose from the reference camera pushed through the averaged rig "
                           "transform; 'bundle_adjusted' = residuals of the joint rig + board-pose refinement.",
            "method": method,
            "overall_mean_reprojection_error_px": overall_mean,
            "overall_rms_reprojection_error_px": overall_rms,
            "averaged_rig_overall_mean_reprojection_error_px": overall_mean_avg,
            "averaged_rig_overall_rms_reprojection_error_px": overall_rms_avg,
        },
        "bundle_adjustment": None if ba is None else {
            "refine_focal": bool(refine_focal),
            "huber_px": float(huber_px),
            "rms_before_px": ba["rms_before_px"],
            "rms_after_px": ba["rms_after_px"],
            "n_evaluations": ba["n_evaluations"],
            "status": ba["status"],
            "message": ba["message"],
            "fx_change_pct": fx_change_pct,
        },
        "frame_diversity": frame_diversity,
        "warnings": warnings,
        "cameras": [],
    }
    for ci, (e, c) in enumerate(zip(extrinsics, cameras)):
        R = e["R"]
        t = e["tvec"].flatten()
        cam_pos = (-R.T @ t).tolist()
        euler_deg = Rscipy.from_matrix(R).as_euler("xyz", degrees=True).tolist()
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
                "reprojection_error_px": per_cam_reproj[ci],
                "reprojection_error_averaged_rig_px": per_cam_reproj_avg[ci],
            },
            "intrinsics": _intrinsics_block(final_intr[ci]),
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
