#!/usr/bin/env python3
"""
prepare_template.py
-------------------
Turns an arbitrary rat model (GLB/GLTF/OBJ/PLY/STL -- e.g. a downloaded scan
or artist model) into a template the completion fitter of
template_fit_lib.py can use well:

  1. loads the model with its scene transforms applied and welds coincident
     vertices (scans are usually split at UV/normal seams);
  2. keeps only the largest connected components (body and tail, typically),
     discarding separately-modelled eyes, teeth, whiskers, claws, props;
  3. optionally strips whisker-like strands that are part of the body mesh:
     a vertex is treated as a strand if the surface is thin along its normal
     AND thin in the tangent plane (1-D tube), which keeps ears (thin sheets:
     thin along the normal only) while removing whiskers; thin geometry
     hurts the silhouette terms and carries no information the masks have;
  4. decimates to a fitter-friendly face count (quadric decimation);
  5. re-orients to the pipeline's convention (+X = nose, +Z = up), scales the
     nose-to-tail-tip length to a given size, and puts the belly at z = 0.

Requires trimesh (loading) and open3d (decimation, ray casting):
    pip install trimesh open3d

Example (the bundled brown-rat template was produced with):
    python prepare_template.py --input "Brown Rat in Profile.glb" \
        --output templates/rat_brown.obj --forward-axis +z --up-axis +y \
        --keep-components 2 --strip-whiskers --thin-normal 0.003 --thin-tangent 0.003 \
        --straighten-tail --target-faces 24000 --length 0.42

  6. optionally straightens a curled tail (--straighten-tail): the tail
     component is parametrised by geodesic arclength from its base, its
     length and radius profile are measured, and it is re-generated as a
     straight tapered tube behind the body. A curled tail is a pose of the
     source model, not anatomy, and the fitter's smooth deformation cannot
     unbend a tight curl -- it folds it into a hairpin instead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_lib import save_obj

AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}


def load_any(path):
    import trimesh
    scene = trimesh.load(str(path), force="scene", process=False)
    parts_v, parts_f, offset = [], [], 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        v = trimesh.transform_points(np.asarray(g.vertices, dtype=np.float64), T)
        parts_v.append(v)
        parts_f.append(np.asarray(g.faces, dtype=np.int64) + offset)
        offset += len(v)
    if not parts_v:
        raise ValueError(f"{path}: no triangle geometry found")
    return np.vstack(parts_v), np.vstack(parts_f)


def weld(V, F, decimals=6):
    U, inv = np.unique(np.round(V, decimals), axis=0, return_inverse=True)
    F = inv.reshape(-1)[F] if inv.ndim > 1 else inv[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return U, F


def components(V, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    n, lab = connected_components(A, directed=False)
    return n, lab


def subset(V, F, keep_vertex_mask):
    remap = -np.ones(len(V), dtype=np.int64)
    remap[keep_vertex_mask] = np.arange(int(keep_vertex_mask.sum()))
    F2 = remap[F[keep_vertex_mask[F].all(axis=1)]]
    V2 = V[keep_vertex_mask]
    used = np.zeros(len(V2), dtype=bool)
    used[F2.ravel()] = True
    remap2 = -np.ones(len(V2), dtype=np.int64)
    remap2[used] = np.arange(int(used.sum()))
    return V2[used], remap2[F2]


def keep_largest(V, F, k, log):
    n, lab = components(V, F)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    log(f"  {n} connected components; sizes: {sizes[order][:10].tolist()}{' ...' if n > 10 else ''}")
    keep = np.isin(lab, order[:k])
    return subset(V, F, keep)


def strip_strands(V, F, thin_normal, thin_tangent, log):
    """Remove 1-D strand geometry (whiskers): thin along the normal and thin
    in at least one tangent direction. Sheets (ears) are thin along the
    normal only and are kept."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.compute_vertex_normals()
    N = np.asarray(m.vertex_normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    def cast(origins, direction):
        rays = o3d.core.Tensor(np.hstack([origins, direction]).astype(np.float32))
        t = scene.cast_rays(rays)["t_hit"].numpy()
        t[~np.isfinite(t)] = np.inf
        return t

    # thickness along the inward normal, from just below the surface
    t_n = cast(V - N * 1e-4, -N)
    # in-plane widths, measured from a point *inside* the surface (a tangent
    # ray from the surface itself never re-enters a convex tube): a whisker
    # is narrow across in some in-plane direction, an ear sheet is wide in
    # every in-plane direction (rim vertices included, since the ray towards
    # the interior of the sheet is long)
    inside = V - N * np.minimum(0.5 * t_n, 0.5 * thin_normal)[:, None]
    helper = np.where((np.abs(N[:, 0]) < 0.9)[:, None], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    T1 = np.cross(N, helper); T1 /= np.linalg.norm(T1, axis=1, keepdims=True) + 1e-12
    T2 = np.cross(N, T1)
    width_1 = cast(inside, T1) + cast(inside, -T1)
    width_2 = cast(inside, T2) + cast(inside, -T2)
    t_t = np.minimum(width_1, width_2)
    strand = (t_n < thin_normal) & (t_t < thin_tangent)
    # only strip strands from the largest component (the body): a separately
    # modelled tail is itself a tube and must survive
    n, lab = components(V, F)
    body = lab == np.argmax(np.bincount(lab))
    strand &= body
    # erode the removal set by one ring: a vertex goes only if all its
    # neighbours are strand-like too, so the rim of a thin sheet (an ear
    # edge, where in-plane rays exit quickly) is kept while whole whiskers go
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]], F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    non_strand_neighbours = A @ (~strand).astype(np.float64)
    remove = strand & (non_strand_neighbours == 0)
    log(f"  strand vertices removed: {int(remove.sum()):,} of {len(V):,} "
        f"(thin along normal < {thin_normal*1000:.0f} mm and in-plane < {thin_tangent*1000:.0f} mm, body component only)")
    return subset(V, F, ~remove)


def decimate(V, F, target_faces, log):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.remove_duplicated_vertices().remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    if len(m.triangles) > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        m = m.remove_degenerate_triangles().remove_unreferenced_vertices()
    V2, F2 = np.asarray(m.vertices), np.asarray(m.triangles)
    log(f"  decimated to {len(V2):,} vertices / {len(F2):,} faces")
    return V2, F2


def straighten_tail(V, F, log, bin_size=0.004, ring_verts=16):
    """Replace a curled tail with a straight tapered tube of the same
    length and radius profile (in the pipeline frame: +X nose, so the tail
    extends towards -X). A curled tail is a pose of the source model, not
    anatomy, and the fitter's smooth deformation graph cannot unbend a
    tight curl into a straight silhouette -- it folds it into a hairpin
    instead. Only the length and the taper carry information for
    silhouette fitting, so the tail is measured (geodesic arclength from
    its base, per-section centroid and radius) and re-generated as a clean
    tube rather than deformed, which avoids reconstructing a reliable
    medial frame from a thick, noisy surface. Works on a tail that is its
    own connected component (the second-largest); the body is untouched."""
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import dijkstra
    n, lab = components(V, F)
    sizes = np.bincount(lab)
    if n < 2:
        log("  straighten-tail: tail is not a separate component, skipped")
        return V, F
    body_lab, tail_lab = np.argsort(-sizes)[:2]
    tail = np.nonzero(lab == tail_lab)[0]
    body = np.nonzero(lab == body_lab)[0]
    T = V[tail]

    # (a) the detached tail piece: arclength from its base along its own
    # surface (mesh edges only -- bridging across the curl would short-circuit
    # the distance field), then per-section centroid and cross-section radius
    d_body, _ = cKDTree(V[body]).query(T)
    base = int(np.argmin(d_body))
    tf_ = F[(lab[F] == tail_lab).all(axis=1)]
    local = -np.ones(len(V), dtype=np.int64); local[tail] = np.arange(len(tail))
    e = local[np.concatenate([tf_[:, [0, 1]], tf_[:, [1, 2]], tf_[:, [2, 0]]])]
    w = np.linalg.norm(T[e[:, 0]] - T[e[:, 1]], axis=1)
    s = dijkstra(sparse.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(T), len(T))),
                 directed=False, indices=base)
    s[~np.isfinite(s)] = s[np.isfinite(s)].max()
    nb_t = max(4, int(np.ceil(s.max() / bin_size)))
    et = np.linspace(0, s.max() + 1e-9, nb_t + 1)
    it = np.clip(np.searchsorted(et, s, side="right") - 1, 0, nb_t - 1)
    dist_t, rad_t = [], []
    for b in range(nb_t):
        sel = it == b
        if sel.sum() >= 3:
            c = T[sel].mean(axis=0)
            dist_t.append(0.5 * (et[b] + et[b + 1]))
            rad_t.append(np.median(np.linalg.norm(T[sel] - c, axis=1)))

    # near-base radius: bin 0 is a small cap around the base vertex, not a
    # full ring, so take a robust radius over the first quarter of the tail
    r_base = float(np.median(rad_t[: max(1, len(rad_t) // 4)])) if rad_t else np.nan

    # (b) the body mesh also carries a tail (this source model duplicates it:
    # one copy in the body component, one as a separate component). Behind
    # the rump the body's cross-section collapses to tail thickness, so bin
    # the body along -X and cut where the radius jumps back up to trunk size.
    Bv = V[body]
    xb = np.arange(Bv[:, 0].min(), Bv[:, 0].max(), bin_size)
    ib = np.clip(np.searchsorted(xb, Bv[:, 0], side="right") - 1, 0, len(xb) - 1)
    rad_b = np.full(len(xb), np.nan); cen_b = np.full((len(xb), 3), np.nan)
    for b in range(len(xb)):
        sel = ib == b
        if sel.sum() >= 3:
            cen_b[b] = Bv[sel].mean(axis=0)
            rad_b[b] = np.median(np.linalg.norm(Bv[sel][:, 1:] - cen_b[b, 1:], axis=1))
    # the rump is where the body's cross-section jumps well above the thin
    # strand at its rear; if there is no such strand (no duplicated tail) the
    # rearmost bins are already the rump, nothing is cut from the body
    ok = np.nonzero(np.isfinite(rad_b))[0]
    if not np.isfinite(r_base) or len(ok) < 4:
        log("  straighten-tail: could not measure the body's rear, skipped")
        return V, F
    r_strand = float(np.median(rad_b[ok[: min(10, len(ok))]]))
    over = np.nonzero(np.isfinite(rad_b) & (rad_b > 3.0 * r_strand))[0]
    cut = int(over[0]) if len(over) else int(ok[0])
    attach = cen_b[cut]

    # (c) the profile from the rump to the tip, resampled and made monotone.
    # The two copies describe the same tail, so its length is the arclength
    # measured on the detached piece, not the sum of the two.
    dist, rad = np.asarray(dist_t), np.asarray(rad_t)
    length = float(dist[-1])
    nb = max(4, int(np.ceil(length / bin_size)))
    arc = np.linspace(0.0, length, nb)
    prof = np.interp(arc, dist, rad)
    prof[0] = max(prof[0], r_base)
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    prof = np.minimum.accumulate(np.maximum(prof, prof[-1]))

    tail_sel = np.zeros(len(V), dtype=bool)
    tail_sel[tail] = True
    tail_sel[body[Bv[:, 0] < xb[cut]]] = True         # the body's copy of the tail

    # rebuild as a straight tapered tube: rings of ring_verts vertices along
    # -X from the attachment point, an apex at the tip, a fan cap at the base
    axis = np.array([-1.0, 0.0, 0.0])
    N0 = np.array([0.0, 1.0, 0.0]); B0 = np.cross(axis, N0)
    theta = np.linspace(0, 2 * np.pi, ring_verts, endpoint=False)
    ring = np.cos(theta)[:, None] * N0 + np.sin(theta)[:, None] * B0     # (ring_verts, 3)
    # the first sections start inside the body, plugging the opening left
    # where the model's own (curled) tail was removed from the rump
    arc = arc - 2.0 * bin_size
    rings = attach + arc[:, None, None] * axis + prof[:, None, None] * ring[None]
    new_V = np.vstack([rings.reshape(-1, 3), attach, attach + length * axis])
    base_c, tip_c = len(new_V) - 2, len(new_V) - 1
    tris = []
    for b in range(nb - 1):
        i0, i1 = b * ring_verts, (b + 1) * ring_verts
        for k in range(ring_verts):
            k2 = (k + 1) % ring_verts
            tris.append([i0 + k, i1 + k, i1 + k2])
            tris.append([i0 + k, i1 + k2, i0 + k2])
    for k in range(ring_verts):                       # caps
        k2 = (k + 1) % ring_verts
        tris.append([base_c, k2, k])
        tris.append([tip_c, (nb - 1) * ring_verts + k, (nb - 1) * ring_verts + k2])
    new_F = np.asarray(tris, dtype=np.int64)

    V2, F2 = subset(V, F, ~tail_sel)                  # drop the curled tail (root included)
    V3 = np.vstack([V2, new_V])
    F3 = np.vstack([F2, new_F + len(V2)])
    log(f"  tail rebuilt straight: {int(tail_sel.sum()):,} curled vertices replaced by a "
        f"{nb}-section tube ({len(new_V):,} vertices), length {length:.3f}, "
        f"radius {prof[0]:.4f} -> {prof[-1]:.4f} (model units)")
    return V3, F3


def reorient(V, forward, up, length, log):
    fi, fs = AXES[forward]
    ui, us = AXES[up]
    if fi == ui:
        raise ValueError("forward and up axes must differ")
    f = np.zeros(3); f[fi] = fs
    u = np.zeros(3); u[ui] = us
    left = np.cross(u, f)                       # right-handed: x=forward, y=left, z=up
    R = np.stack([f, left, u])                  # rows = new axes expressed in old coords
    P = V @ R.T
    if length is not None:
        s = length / (P[:, 0].max() - P[:, 0].min())
        P = P * s
        log(f"  scaled by {s:.4f} so that the nose-to-tail extent is {length:.3f} m")
    P[:, 0] -= (P[:, 0].max() + P[:, 0].min()) / 2
    P[:, 1] -= (P[:, 1].max() + P[:, 1].min()) / 2
    P[:, 2] -= P[:, 2].min()
    log(f"  final extents (m): {np.round(P.max(0) - P.min(0), 3).tolist()} (+X nose, +Z up, belly at z=0)")
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--forward-axis", default="+x", choices=sorted(AXES), help="model axis pointing from tail to nose")
    ap.add_argument("--up-axis", default="+z", choices=sorted(AXES))
    ap.add_argument("--length", type=float, default=0.42, help="nose-to-tail-tip length in metres after scaling (0 = keep)")
    ap.add_argument("--keep-components", type=int, default=2, help="largest connected components to keep (body + tail = 2)")
    ap.add_argument("--strip-whiskers", action="store_true")
    ap.add_argument("--thin-normal", type=float, default=0.006, help="strand test: thickness along the normal (m)")
    ap.add_argument("--thin-tangent", type=float, default=0.004, help="strand test: min in-plane thickness (m)")
    ap.add_argument("--target-faces", type=int, default=24000)
    ap.add_argument("--straighten-tail", action="store_true",
                    help="re-lay a curled tail (separate component) along a straight line behind the body")
    args = ap.parse_args()
    log = print

    log(f"loading {args.input}")
    V, F = load_any(args.input)
    log(f"  {len(V):,} vertices / {len(F):,} faces as loaded")
    V, F = weld(V, F)
    log(f"  {len(V):,} vertices after welding coincident vertices")
    V, F = keep_largest(V, F, args.keep_components, log)
    if args.strip_whiskers:
        # scale-aware thresholds: thresholds are given in the *output* metric
        # scale, so convert them to the model's current units first
        cur_len = float(np.ptp(V[:, AXES[args.forward_axis][0]]))
        k = cur_len / args.length if args.length else 1.0
        V, F = strip_strands(V, F, args.thin_normal * k, args.thin_tangent * k, log)
        V, F = keep_largest(V, F, args.keep_components, log)
    V, F = decimate(V, F, args.target_faces, log)
    V, F = keep_largest(V, F, args.keep_components, log)   # decimation can leave slivers
    V = reorient(V, args.forward_axis, args.up_axis, args.length if args.length else None, log)
    if args.straighten_tail:
        V, F = straighten_tail(V, F, log)
        V = reorient(V, "+x", "+z", args.length if args.length else None, log)   # re-centre/re-scale after unbending
    save_obj(args.output, V, F)
    log(f"wrote {args.output}: {len(V):,} vertices / {len(F):,} faces")


if __name__ == "__main__":
    main()
