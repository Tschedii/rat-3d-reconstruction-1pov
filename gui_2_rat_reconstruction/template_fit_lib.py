"""
template_fit_lib.py
--------------------
Model-based shape completion for the visual-hull reconstruction.

Voxel carving can only keep geometry that enough cameras agree on. When a
body part (typically the head: ears, snout) is seen well by only a minority
of the cameras in a given frame, the hull loses it -- and lowering
min_cameras globally just lets under-constrained noise through elsewhere.
Here instead a complete rat template mesh (any .obj: an external model, or
one of your own well-reconstructed frames) is deformed to explain the same
six silhouettes and the carved hull. Where the data is good, the fitted
template follows the data; where it is weak, the template's own anatomy
fills in. The deformation is an embedded deformation graph (Sumner et al.
2007): a sparse set of control nodes sampled over the template, each with
its own rotation + translation, blended into every vertex, with a
smoothness term between neighbouring nodes so the neck/spine/tail can bend
without the surface tearing or folding.

Losses (all optimised together with PyTorch autograd, no differentiable
rasteriser needed):
  * silhouette containment -- every template vertex, projected into each
    camera, should land inside that camera's mask; measured with the mask's
    distance transform, robustified, and *trimmed* to the best K cameras per
    vertex so a minority of masks that miss a body part don't veto it.
  * mask coverage -- pixels of each mask should be near some projected
    template vertex (pulls the template out to fill the head in the cameras
    that do see it).
  * hull anchoring -- every carved-hull vertex should be near the template
    surface (keeps the well-observed body where the data says it is).
  * deformation smoothness -- neighbouring graph nodes should agree.

Everything is expressed in mm (3D) or pixels (2D) so the loss weights are
comparable across datasets.
"""

import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # surfaced with a clear message when fitting is requested
    torch = None
    F = None


# =============================================================================
# Mesh I/O + geometry helpers
# =============================================================================

def load_obj(path):
    """Minimal OBJ reader: vertices (ignoring any trailing colour columns) and
    faces (any polygon, fan-triangulated; texture/normal indices dropped)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise ValueError(f"{path}: no vertices/faces found -- is this a triangle/polygon OBJ?")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def farthest_point_sampling(pts, k, seed=0):
    n = len(pts)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = rng.integers(n)
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(pts - pts[idx[i]], axis=1))
    return idx


def build_deformation_graph(verts, n_nodes=120, skin_k=4, node_k=6):
    """Sample control nodes over the template (farthest-point sampling) and
    skin every vertex to its skin_k nearest nodes with Sumner's weights
    w_j = (1 - d_j / d_{k+1})^2, normalised. Node-to-node edges connect each
    node to its node_k nearest nodes (used by the smoothness term)."""
    n_nodes = max(2, min(n_nodes, len(verts)))
    skin_k = max(1, min(skin_k, n_nodes - 1))
    node_k = max(1, min(node_k, n_nodes - 1))
    node_idx = farthest_point_sampling(verts, n_nodes)
    nodes = verts[node_idx]
    tree = cKDTree(nodes)
    d, skin_idx = tree.query(verts, k=skin_k + 1)
    d = d.reshape(len(verts), -1)
    skin_idx = skin_idx.reshape(len(verts), -1)
    w = (1.0 - d[:, :skin_k] / (d[:, skin_k:skin_k + 1] + 1e-12)) ** 2
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    _, nbr = tree.query(nodes, k=node_k + 1)
    nbr = nbr.reshape(n_nodes, -1)[:, 1:]
    edges = np.stack([np.repeat(np.arange(n_nodes), node_k), nbr.ravel()], axis=1)
    return {"nodes": nodes, "skin_idx": skin_idx[:, :skin_k], "skin_w": w, "edges": edges}


def pca_frame(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    extent = (pts - c) @ vt.T
    lengths = extent.max(axis=0) - extent.min(axis=0)
    return c, vt, lengths


def init_candidates(template_verts, hull_verts):
    """Rigid+scale initialisations aligning the template's principal axes to
    the hull's. The axis directions are sign-ambiguous (head/tail, up/down),
    so all four proper rotations are returned and the caller keeps the one
    that fits best after a short global-only optimisation."""
    ct, At, Lt = pca_frame(template_verts)
    ch, Ah, Lh = pca_frame(hull_verts)
    s0 = float(Lh[0] / max(Lt[0], 1e-9))
    cands = []
    for s1, s2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        Ah_s = Ah * np.array([[s1], [s2], [1.0]])
        if np.linalg.det(Ah_s.T @ At) < 0:
            Ah_s[2] *= -1.0
        R0 = Ah_s.T @ At
        cands.append((s0, R0, ct, ch))
    return cands


# =============================================================================
# Camera / mask preprocessing
# =============================================================================

def prepare_masks(masks, coverage_samples=1500, seed=0):
    """Per camera: the mask's outside-distance transform (0 inside the
    silhouette, distance in px to the nearest silhouette pixel outside) and a
    random sample of silhouette pixels for the coverage term. Only the
    largest connected component of each mask is used for coverage so stray
    segmentation specks don't pull the template around; the distance
    transform keeps the full mask (a speck can only make containment
    *easier* there, which is harmless)."""
    rng = np.random.default_rng(seed)
    out = []
    for mask in masks:
        if mask is None:
            out.append(None)
            continue
        fg = mask > 0
        dt = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        labelled, n = ndimage.label(fg)
        if n > 1:
            sizes = ndimage.sum(fg, labelled, range(1, n + 1))
            fg_main = labelled == (int(np.argmax(sizes)) + 1)
        else:
            fg_main = fg
        ys, xs = np.nonzero(fg_main)
        if len(xs) == 0:
            out.append(None)
            continue
        pick = rng.choice(len(xs), size=min(coverage_samples, len(xs)), replace=False)
        samples = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)
        out.append({"dt": dt, "samples": samples, "H": mask.shape[0], "W": mask.shape[1],
                    "area": int(fg_main.sum())})
    return out


# =============================================================================
# Torch pieces
# =============================================================================

def _require_torch():
    if torch is None:
        raise ImportError("template fitting needs PyTorch: pip install torch")


def safe_norm(x, dim=-1, keepdim=False, eps=1e-18):
    # torch.norm has a NaN gradient at exactly zero, which every rotation
    # parameter starts at; sqrt(sum^2 + eps) is finite there.
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def so3_exp(omega):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3), Rodrigues."""
    theta = safe_norm(omega, dim=-1, keepdim=True)
    k = omega / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], -1),
        torch.stack([kz, zero, -kx], -1),
        torch.stack([-ky, kx, zero], -1),
    ], -2)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(K.shape)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1.0 - c) * (K @ K)


def cameras_to_torch(cameras, dtype):
    out = []
    for cam in cameras:
        out.append({
            "R": torch.as_tensor(cam["R"], dtype=dtype),
            "t": torch.as_tensor(cam["t"], dtype=dtype),
            "K": torch.as_tensor(cam["K"], dtype=dtype),
            "dist": torch.as_tensor(np.asarray(cam["dist"], dtype=np.float64).ravel()[:5], dtype=dtype),
            "name": cam["name"],
        })
    return out


def project_torch(pts, cam):
    """Same pinhole + Brown-Conrady model as cv2.projectPoints (k1 k2 p1 p2 k3),
    differentiable. Returns pixel coords (N, 2) and an in-front-of-camera mask."""
    pc = pts @ cam["R"].T + cam["t"]
    z = pc[:, 2]
    in_front = z > 1e-6
    zc = z.clamp_min(1e-6)
    x = pc[:, 0] / zc
    y = pc[:, 1] / zc
    k1, k2, p1, p2, k3 = cam["dist"]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    u = cam["K"][0, 0] * xd + cam["K"][0, 2]
    v = cam["K"][1, 1] * yd + cam["K"][1, 2]
    return torch.stack([u, v], dim=-1), in_front


def sample_image(img, uv, H, W):
    """Bilinear lookup of a (1,1,H,W) image at pixel coords (N,2)."""
    gx = 2.0 * uv[:, 0] / (W - 1) - 1.0
    gy = 2.0 * uv[:, 1] / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None, None]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]


def robust(d, c):
    """Geman-McClure: ~d^2 for d << c, saturates at c^2 for d >> c."""
    d2 = d * d
    return (c * c) * d2 / (c * c + d2)


class TemplateDeformer(torch.nn.Module if torch is not None else object):
    """Global similarity (scale, rotation, translation) on top of an embedded
    deformation graph. forward() returns the deformed template vertices in
    world coordinates."""

    def __init__(self, verts, graph, s0, R0, c_template, c_hull, dtype):
        super().__init__()
        self.rest = torch.as_tensor(verts, dtype=dtype)
        self.nodes = torch.as_tensor(graph["nodes"], dtype=dtype)
        self.skin_idx = torch.as_tensor(graph["skin_idx"], dtype=torch.long)
        self.skin_w = torch.as_tensor(graph["skin_w"], dtype=dtype)
        self.edges = torch.as_tensor(graph["edges"], dtype=torch.long)
        # buffers (not plain attributes) so a saved state carries the base
        # alignment too and a warm-started model reproduces the same pose
        self.register_buffer("R0", torch.as_tensor(R0, dtype=dtype))
        self.register_buffer("c_template", torch.as_tensor(c_template, dtype=dtype))
        self.register_buffer("c_hull", torch.as_tensor(c_hull, dtype=dtype))
        n_nodes = len(graph["nodes"])
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(s0), dtype=dtype))
        self.g_rot = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.g_t = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.n_rot = torch.nn.Parameter(torch.zeros(n_nodes, 3, dtype=dtype))
        self.n_t = torch.nn.Parameter(torch.zeros(n_nodes, 3, dtype=dtype))

    def global_params(self):
        return [self.log_scale, self.g_rot, self.g_t]

    def node_params(self):
        return [self.n_rot, self.n_t]

    def scale(self):
        return torch.exp(self.log_scale)

    def deformed_local(self):
        R = so3_exp(self.n_rot)                                   # (K,3,3)
        g = self.nodes[self.skin_idx]                             # (N,m,3)
        Rk = R[self.skin_idx]                                     # (N,m,3,3)
        tk = self.n_t[self.skin_idx]                              # (N,m,3)
        v = self.rest[:, None, :] - g                             # (N,m,3)
        local = torch.einsum("nmij,nmj->nmi", Rk, v) + g + tk
        return (self.skin_w[..., None] * local).sum(dim=1)

    def forward(self):
        Rg = so3_exp(self.g_rot[None])[0] @ self.R0
        local = self.deformed_local()
        return self.scale() * (local - self.c_template) @ Rg.T + self.c_hull + self.g_t

    def smoothness(self):
        """Sumner's regulariser: each node's transform, applied to its
        neighbour's rest position, should land where the neighbour actually
        moved to. Reported in world-mm^2."""
        R = so3_exp(self.n_rot)
        k, l = self.edges[:, 0], self.edges[:, 1]
        gk, gl = self.nodes[k], self.nodes[l]
        pred = torch.einsum("eij,ej->ei", R[k], gl - gk) + gk + self.n_t[k]
        actual = gl + self.n_t[l]
        d = (pred - actual) * (self.scale() * 1000.0)
        return (d * d).sum(dim=1).mean()


# =============================================================================
# Loss terms
# =============================================================================

def silhouette_loss(pts, cams, prepped, keep_k, c_px, outlier_weight=0.2):
    """Per vertex, the K cameras with the smallest containment cost count in
    full; the remaining (trimmed-away) cameras count with outlier_weight.
    Full trimming (weight 0) lets a minority-visible part exist, but also lets
    the whole surface inflate to the looser K-camera hull; the small residual
    weight keeps the pressure on everywhere while the coverage term can still
    win where a part is genuinely seen by only K cameras."""
    costs = []
    for cam, prep in zip(cams, prepped):
        if prep is None:
            continue
        uv, in_front = project_torch(pts, cam)
        d = sample_image(prep["dt_t"], uv, prep["H"], prep["W"])
        d = torch.where(in_front, d, torch.full_like(d, 3.0 * c_px))
        costs.append(robust(d, c_px))
    if not costs:
        return pts.sum() * 0.0
    C = torch.stack(costs, dim=1)
    k = max(1, min(keep_k, C.shape[1]))
    sorted_costs = torch.sort(C, dim=1).values
    kept = sorted_costs[:, :k].sum(dim=1)
    trimmed = sorted_costs[:, k:].sum(dim=1)
    return (kept + outlier_weight * trimmed).mean()


def hull_support(hull_verts, cams, prepped, tol_px, dtype):
    """Number of cameras in whose mask each hull vertex projects (within
    tol_px of the silhouette). Hull surface points that are far outside a
    camera's silhouette are the visual hull's phantom volume -- carving kept
    them only because *other* cameras agreed -- and shouldn't anchor the fit."""
    pts = torch.as_tensor(hull_verts, dtype=dtype)
    support = np.zeros(len(hull_verts), dtype=np.int32)
    with torch.no_grad():
        for cam, prep in zip(cams, prepped):
            if prep is None:
                continue
            uv, in_front = project_torch(pts, cam)
            d = sample_image(prep["dt_t"], uv, prep["H"], prep["W"]).cpu().numpy()
            support += (in_front.cpu().numpy() & (d <= tol_px)).astype(np.int32)
    return support


def coverage_loss(pts, cams, prepped, c_px, corr=None, refresh=True):
    """corr is an optional dict caching nearest-vertex correspondences per
    camera; when refresh is False the cached indices are reused (the
    distances are still recomputed differentiably), which is the usual
    ICP-style trick -- KD-tree builds dominated the per-iteration cost."""
    total, n = 0.0, 0
    for ci, (cam, prep) in enumerate(zip(cams, prepped)):
        if prep is None:
            continue
        uv, in_front = project_torch(pts, cam)
        if corr is None or refresh or ci not in corr:
            front = in_front.detach().cpu().numpy()
            if front.sum() < 3:
                continue
            full_idx = np.nonzero(front)[0]
            tree = cKDTree(uv.detach().cpu().numpy()[front])
            _, idx = tree.query(prep["samples"], k=1)
            idx_t = torch.as_tensor(full_idx[idx], dtype=torch.long)
            if corr is not None:
                corr[ci] = idx_t
        else:
            idx_t = corr[ci]
        d = safe_norm(uv[idx_t] - prep["samples_t"], dim=1)
        total = total + robust(d, c_px).mean()
        n += 1
    return total / max(n, 1)


def hull_loss(pts, hull_t, c_mm, corr=None, refresh=True):
    if corr is None or refresh or "hull" not in corr:
        tree = cKDTree(pts.detach().cpu().numpy())
        _, idx = tree.query(hull_t.cpu().numpy(), k=1)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        if corr is not None:
            corr["hull"] = idx_t
    else:
        idx_t = corr["hull"]
    d = safe_norm(pts[idx_t] - hull_t, dim=1) * 1000.0
    return robust(d, c_mm).mean()


# =============================================================================
# Fitting driver
# =============================================================================

DEFAULT_FIT_PARAMS = {
    "n_nodes": 200,            # deformation-graph control nodes
    "iterations": 600,         # full (non-rigid) optimisation steps
    "init_iterations": 60,     # global-only steps per initial-orientation candidate
    "sil_cameras": None,       # containment must hold in the best K cameras per vertex;
                               # None -> half the active cameras (min 2). Unlike voxel
                               # carving the template can't grow new blobs, so a low K
                               # is safe: it just lets a part be supported by a minority
                               # of cameras when the coverage term pulls it there.
    "sil_outlier_weight": 0.05, # residual weight of the trimmed-away cameras
    "w_sil": 1.0,
    "w_cov": 2.0,
    "w_hull": 0.5,
    "w_reg": 0.5,
    "c_px": 20.0,              # robust-kernel scale for pixel terms
    "c_mm": 15.0,              # robust-kernel scale for the hull term (mm)
    "hull_samples": 4000,
    "corr_every": 4,           # rebuild nearest-neighbour correspondences every N steps
    "hull_support_tol_px": 3.0,
    "hull_min_support": None,  # anchor only to hull verts inside >= this many masks;
                               # None -> all active cameras but one
    "dtype": "float32",
    "coverage_samples": 1500,
    "lr_rot": 0.01,
    "lr_trans": 0.001,         # metres
    "lr_scale": 0.005,
    "seed": 0,
}


def _make_optimizer(model, params, include_nodes):
    groups = [
        {"params": [model.g_rot], "lr": params["lr_rot"]},
        {"params": [model.g_t], "lr": params["lr_trans"]},
        {"params": [model.log_scale], "lr": params["lr_scale"]},
    ]
    if include_nodes:
        groups.append({"params": [model.n_rot], "lr": params["lr_rot"]})
        groups.append({"params": [model.n_t], "lr": params["lr_trans"]})
    return torch.optim.Adam(groups)


def _total_loss(model, cams, prepped, hull_t, params, w_reg, corr=None, refresh=True):
    pts = model()
    sil = silhouette_loss(pts, cams, prepped, params["sil_cameras"], params["c_px"],
                          outlier_weight=params["sil_outlier_weight"])
    cov = coverage_loss(pts, cams, prepped, params["c_px"], corr=corr, refresh=refresh)
    hul = hull_loss(pts, hull_t, params["c_mm"], corr=corr, refresh=refresh)
    reg = model.smoothness()
    total = params["w_sil"] * sil + params["w_cov"] * cov + params["w_hull"] * hul + w_reg * reg
    return total, {"sil": float(sil), "cov": float(cov), "hull": float(hul), "reg": float(reg)}


def fit_report(pts_np, cameras, masks, prepped, hull_verts, min_cameras):
    """Human-readable fit quality: per-camera mean containment distance (px,
    untrimmed) and coverage distance (px), hull residual (mm), and how much of
    the fitted surface is actually supported by >= min_cameras masks versus
    filled in by the template prior."""
    dtype = next(pr["dt_t"].dtype for pr in prepped if pr is not None)
    pts = torch.as_tensor(pts_np, dtype=dtype)
    cams = cameras_to_torch(cameras, dtype)
    per_cam = []
    support = np.zeros(len(pts_np), dtype=np.int32)
    with torch.no_grad():
        for cam, prep, mask in zip(cams, prepped, masks):
            if prep is None:
                continue
            uv, in_front = project_torch(pts, cam)
            d = sample_image(prep["dt_t"], uv, prep["H"], prep["W"]).cpu().numpy()
            front = in_front.cpu().numpy()
            inside = front & (d <= 0.5)
            support += inside.astype(np.int32)
            tree = cKDTree(uv.cpu().numpy()[front])
            cov_d, _ = tree.query(prep["samples"], k=1)
            per_cam.append({"camera": cam["name"],
                            "outside_px_mean": float(d[front].mean()) if front.any() else float("nan"),
                            "outside_frac": float(np.mean(~inside)),
                            "coverage_px_mean": float(cov_d.mean())})
    tree = cKDTree(pts_np)
    hd, _ = tree.query(hull_verts, k=1)
    return {
        "per_camera": per_cam,
        "hull_residual_mm_mean": float(hd.mean() * 1000.0),
        "hull_residual_mm_p95": float(np.percentile(hd, 95) * 1000.0),
        "support_count": support,
        "supported_frac": float(np.mean(support >= min_cameras)),
    }


def fit_template(template_verts, template_faces, cameras, masks, hull_verts, params=None, log=print,
                 warm_start=None):
    """Fit the template to (masks, hull). Returns dict with the fitted vertices
    (world coords), the template faces, the fit report, and a state dict that
    can be passed back as warm_start for the next frame of a sequence."""
    _require_torch()
    p = dict(DEFAULT_FIT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    torch.manual_seed(int(p["seed"]))
    dtype = torch.float64 if str(p["dtype"]) == "float64" else torch.float32

    prepped = prepare_masks(masks, coverage_samples=int(p["coverage_samples"]), seed=int(p["seed"]))
    n_active = sum(x is not None for x in prepped)
    if n_active < 2:
        raise RuntimeError("template fitting needs at least 2 cameras with non-empty masks")
    if not p["sil_cameras"]:
        p["sil_cameras"] = max(2, n_active // 2)
    p["sil_cameras"] = int(min(p["sil_cameras"], n_active))
    for prep in prepped:
        if prep is not None:
            prep["dt_t"] = torch.as_tensor(prep["dt"], dtype=dtype)[None, None]
            prep["samples_t"] = torch.as_tensor(prep["samples"], dtype=dtype)
    cams = cameras_to_torch(cameras, dtype)

    rng = np.random.default_rng(int(p["seed"]))
    min_support = p["hull_min_support"] if p["hull_min_support"] else max(2, n_active - 1)
    support = hull_support(hull_verts, cams, prepped, float(p["hull_support_tol_px"]), dtype)
    anchored = hull_verts[support >= int(min_support)]
    if len(anchored) < 200:
        log(f"    warning: only {len(anchored)} hull verts are inside >= {int(min_support)} masks; "
            f"anchoring to the whole hull instead")
        anchored = hull_verts
    else:
        log(f"    hull anchoring: {len(anchored):,} / {len(hull_verts):,} hull verts inside >= {int(min_support)} masks "
            f"(the rest is unsupported phantom volume and is ignored)")
    hull_sub = anchored[rng.choice(len(anchored), size=min(int(p["hull_samples"]), len(anchored)), replace=False)]
    hull_t = torch.as_tensor(hull_sub, dtype=dtype)

    graph = build_deformation_graph(template_verts, n_nodes=int(p["n_nodes"]))
    log(f"    template: {len(template_verts):,} verts, {len(template_faces):,} faces, "
        f"{len(graph['nodes'])} deformation nodes; {n_active} cameras")

    # --- stage 0: pick the best rigid initialisation ------------------------
    if warm_start is not None:
        model = TemplateDeformer(template_verts, graph, 1.0, np.eye(3), template_verts.mean(0),
                                 hull_verts.mean(0), dtype)
        model.load_state_dict(warm_start)
        log("    warm-started from previous frame's fit")
    else:
        best = None
        for ci, (s0, R0, ct, ch) in enumerate(init_candidates(template_verts, hull_verts)):
            cand = TemplateDeformer(template_verts, graph, s0, R0, ct, ch, dtype)
            opt = _make_optimizer(cand, p, include_nodes=False)
            corr = {}
            for it in range(int(p["init_iterations"])):
                opt.zero_grad()
                total, _parts = _total_loss(cand, cams, prepped, hull_t, p, w_reg=0.0,
                                            corr=corr, refresh=(it % int(p["corr_every"]) == 0))
                total.backward()
                opt.step()
            with torch.no_grad():
                total, parts = _total_loss(cand, cams, prepped, hull_t, p, w_reg=0.0)
            log(f"    init candidate {ci}: loss={float(total):.2f} "
                f"(sil {parts['sil']:.1f}, cov {parts['cov']:.1f}, hull {parts['hull']:.1f})")
            if best is None or float(total) < best[0]:
                best = (float(total), cand)
        model = best[1]

    # --- stage 1: non-rigid fit -------------------------------------------
    opt = _make_optimizer(model, p, include_nodes=True)
    n_iter = int(p["iterations"])
    corr = {}
    for it in range(n_iter):
        # heavier smoothness early (find the pose) then relax (fit the details)
        w_reg = p["w_reg"] * (3.0 if it < n_iter * 0.4 else 1.0)
        opt.zero_grad()
        total, parts = _total_loss(model, cams, prepped, hull_t, p, w_reg=w_reg,
                                   corr=corr, refresh=(it % int(p["corr_every"]) == 0))
        total.backward()
        opt.step()
        if it % max(1, n_iter // 8) == 0 or it == n_iter - 1:
            log(f"    fit iter {it:4d}/{n_iter}: loss={float(total):.2f} "
                f"(sil {parts['sil']:.1f}, cov {parts['cov']:.1f}, hull {parts['hull']:.1f}, "
                f"reg {parts['reg']:.1f}, scale {float(model.scale()):.3f})")

    with torch.no_grad():
        fitted = model().cpu().numpy()
    report = fit_report(fitted, cameras, masks, prepped, hull_verts, int(p["sil_cameras"]))
    log(f"    fit report: hull residual {report['hull_residual_mm_mean']:.1f} mm mean / "
        f"{report['hull_residual_mm_p95']:.1f} mm p95; {report['supported_frac']*100:.1f}% of the fitted "
        f"surface is inside >= {int(p['sil_cameras'])} masks (rest completed from the template prior)")
    for row in report["per_camera"]:
        log(f"      {row['camera']}: {row['outside_frac']*100:.1f}% of verts outside mask, "
            f"coverage gap {row['coverage_px_mean']:.1f} px")

    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"verts": fitted, "faces": np.asarray(template_faces, dtype=np.int64),
            "report": report, "state": state}
