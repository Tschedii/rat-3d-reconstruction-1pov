"""Slide-11 video, rendered with Blender (Eevee) from prep_slide11.py's data/, frame by frame:

  1. the calibrated six-camera rig (as on slide 11), orbiting, then a zoom onto the arena
  2. both rats' IMU skulls (rat 1 red, rat 2 blue) with their head paths, recording frames 220..500 (the
     segmentation masks start at 220)
  3. paused at frame 500: GUI 2's reconstruction of both rats, step by step - seed point (the rays
     through every camera's mask centroid), voxel carving camera by camera, the surface, its true
     colours, the template fit (red / blue: what it adds to the hull), then the fur colours
  4. frames 500..750 with the rats moving, Environment 4 added at frame 650
  5. paused at frame 750: rat 1's field of view, then its first-person view for frames 750..1000,
     rat 2 in view whenever rat 1 looks at it

    blender -b -P render_slide11.py -- [first last | f1,f2,...]   -> slide11_video/frames/vf_NNNN.png
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import placement as PL      # noqa: E402  skull and eye placement: tracked head (GUI 5.2) + IMU orientation (GUI 5.1)
D = HERE / "data"
OUT = HERE / "frames"
OUT.mkdir(exist_ok=True)
S = json.load(open(D / "scene.json"))
POSES = json.load(open(D / "skull_poses.json"))
MESHES = json.load(open(D / "meshes.json"))
T_CALIB = Matrix(S["T_from_calib"])
FLOOR_Z = S["floor_z"]
RAT_RGB = {1: (1.0, 40 / 255, 40 / 255), 2: (40 / 255, 110 / 255, 1.0)}     # GUI 5.1 / 5.2 colours
FPS = 30
FOV_V_DEG, FOV_ASPECT = 120.0, 4 / 3            # GUI 4's eye camera

# --------------------------------------------------------------------------- timeline
SEG = {}
_t = 0
for name, n in (("intro", 150), ("zoom", 75), ("skulls", 281), ("dim", 40), ("seed", 70), ("block", 30),
                ("carve1", 90), ("carve2", 90), ("surface", 45), ("truecol", 45), ("template", 75), ("fur", 45),
                ("motion", 150), ("env", 100), ("fov", 60), ("flyin", 45), ("fpv", 251)):
    SEG[name] = (_t, _t + n)
    _t += n
N_VIDEO = _t
SKULL_START = 220          # first recording frame shown (the segmentation masks start there)
import os
ENV_PASS = os.environ.get("ENV_PASS", "")     # "none": render without the room (for the crossfade at frame 650)
if ENV_PASS == "none":
    OUT = HERE / "frames_noenv"; OUT.mkdir(exist_ok=True)
if os.environ.get("OUT_DIR"):
    OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)


def seg_of(vf):
    for k, (a, b) in SEG.items():
        if a <= vf < b:
            return k, (vf - a) / max(1, b - a - 1), vf - a
    return "fpv", 1.0, SEG["fpv"][1] - SEG["fpv"][0] - 1


def rec_frame(vf):
    """The recording frame shown at video frame vf."""
    k, u, i = seg_of(vf)
    if k in ("intro", "zoom"):
        return SKULL_START
    if k == "skulls":
        return SKULL_START + i
    if k in ("dim", "seed", "block", "carve1", "carve2", "surface", "truecol", "template", "fur"):
        return 500
    if k == "motion":
        return 500 + i
    if k == "env":
        return 650 + i
    if k in ("fov", "flyin"):
        return 750
    return 750 + i


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.film_transparent = False
sc.view_settings.view_transform = 'Standard'
sc.eevee.taa_render_samples = 24
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (1, 1, 1, 1)
bg.inputs["Strength"].default_value = 1.0
_wn = world.node_tree.nodes; _wl = world.node_tree.links
_amb = _wn.new('ShaderNodeBackground'); _amb.inputs["Color"].default_value = (0.55, 0.55, 0.55, 1); _amb.inputs["Strength"].default_value = 0.8
_lp = _wn.new('ShaderNodeLightPath'); _mx = _wn.new('ShaderNodeMixShader')
_wl.new(_lp.outputs['Is Camera Ray'], _mx.inputs[0]); _wl.new(_amb.outputs[0], _mx.inputs[1]); _wl.new(bg.outputs[0], _mx.inputs[2])
_wl.new(_mx.outputs[0], _wn["World Output"].inputs['Surface'])

cam_data = bpy.data.cameras.new("view"); cam = bpy.data.objects.new("view", cam_data)
sc.collection.objects.link(cam); sc.camera = cam
cam_data.clip_start, cam_data.clip_end = 0.004, 50
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
sun.data.energy = 3.0; sun.rotation_euler = (math.radians(35), math.radians(10), math.radians(30))
sc.collection.objects.link(sun)


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def flat_material(name, rgb, alpha=1.0):
    """Unlit colour with an animatable alpha (node 'A')."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (*rgb, 1)
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix = nt.nodes.new('ShaderNodeMixShader'); mix.name = "A"; mix.inputs[0].default_value = alpha
    nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    set_method(m, alpha)
    return m


def lit_material(name, rgb, alpha=1.0, roughness=0.55):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs['Base Color'].default_value = (*rgb, 1); p.inputs['Roughness'].default_value = roughness
    p.inputs['Specular IOR Level'].default_value = 0.25
    p.inputs['Alpha'].default_value = alpha
    set_method(m, alpha)
    return m


def set_method(mat, a, partial=False):
    """Solid (alpha 1, nothing partial): opaque, depth-correct, hides what is behind it. Otherwise blended, with
    only the front-most layer of the object drawn (no back faces showing through, no dither grain)."""
    want = 'DITHERED' if a >= 0.999 and not partial else 'BLENDED'
    if mat.surface_render_method != want:
        mat.surface_render_method = want
    mat.use_transparency_overlap = False


def set_alpha(mat, a):
    n = mat.node_tree.nodes
    if "A" in n:
        n["A"].inputs[0].default_value = a
    else:
        n["Principled BSDF"].inputs['Alpha'].default_value = a
    set_method(mat, a)


def obj_with(mesh, mat, name):
    o = bpy.data.objects.new(name, mesh); o.data.materials.append(mat)
    sc.collection.objects.link(o)
    return o


def tube(name, pts_list, radius, mat):
    """Curve object, one poly spline per point list."""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = radius; cu.bevel_resolution = 2
    o = bpy.data.objects.new(name, cu); o.data.materials.append(mat); sc.collection.objects.link(o)
    set_tube(o, pts_list)
    return o


def set_tube(o, pts_list, taper=None):
    """taper: (radius factor at the first point, at the last point), linear in between."""
    cu = o.data
    cu.splines.clear()
    for pts in pts_list:
        if len(pts) < 2:
            continue
        sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
        sp.points.foreach_set("co", np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]).ravel())
        if taper:
            sp.points.foreach_set("radius", np.linspace(taper[0], taper[1], len(pts)))


def dashed(a, b, dash, gap):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a); d = (b - a) / L
    segs, s = [], 0.0
    while s < L:
        segs.append([a + d * s, a + d * min(s + dash, L)]); s += dash + gap
    return segs


def text_obj(name, body, rgb, size):
    cu = bpy.data.curves.new(name, 'FONT'); cu.body = body; cu.size = size; cu.align_x = 'CENTER'
    o = bpy.data.objects.new(name, cu); o.data.materials.append(flat_material(name + "_m", rgb)); sc.collection.objects.link(o)
    return o


# --- rig (as on slide 11) -------------------------------------------------------------
rig_objs, rig_mats = [], []
cams = S["cameras"]
ring = sorted(cams, key=lambda c: math.atan2(c["centre"][1], c["centre"][0]))
labels = []
for c in cams:
    rgb = hexrgb(c["color"])
    C = Vector(c["centre"]); K = [Vector(p) for p in c["corners"]]
    me = bpy.data.meshes.new(c["name"])
    me.from_pydata([C] + K, [], [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 2, 3, 4)])
    m_fill = flat_material(c["name"] + "_fill", rgb, 0.45); rig_mats.append((m_fill, 0.45))
    rig_objs.append(obj_with(me, m_fill, c["name"] + "_frustum"))
    m_edge = flat_material(c["name"] + "_edge", tuple(0.75 * v for v in rgb), 1.0); rig_mats.append((m_edge, 1.0))
    edges = [[C, k] for k in K] + [[K[i], K[(i + 1) % 4]] for i in range(4)]
    rig_objs.append(tube(c["name"] + "_edges", edges, 0.0025, m_edge))
    m_ax = flat_material(c["name"] + "_axis", rgb, 0.8); rig_mats.append((m_ax, 0.8))
    rig_objs.append(tube(c["name"] + "_axis", dashed(C, (0, 0, FLOOR_Z), 0.012, 0.012), 0.0012, m_ax))
    dot = bpy.data.meshes.new(c["name"] + "_dot")
    import bmesh
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.012); bm.to_mesh(dot); bm.free()
    o = obj_with(dot, m_edge, c["name"] + "_dotobj"); o.location = C; rig_objs.append(o)
    t = text_obj(c["name"] + "_label", c["label"], tuple(0.85 * v for v in rgb), 0.06)
    t.location = C + Vector((0, 0, 0.07)); labels.append(t); rig_mats.append((t.data.materials[0], 1.0))
hexm = flat_material("hexagon", (0.45, 0.45, 0.45), 0.9); rig_mats.append((hexm, 0.9))
hx = []
for i in range(6):
    hx += dashed(ring[i]["centre"], ring[(i + 1) % 6]["centre"], 0.03, 0.02)
rig_objs.append(tube("hexagon", hx, 0.0018, hexm))
# arena disc + floor rings
import bmesh
disc = bpy.data.meshes.new("arena"); bm = bmesh.new()
bmesh.ops.create_circle(bm, cap_ends=True, radius=0.45, segments=96); bm.to_mesh(disc); bm.free()
arena_m = flat_material("arena", tuple(0.85 * c + 0.15 for c in (0.62, 0.80, 0.52)), 1.0)
arena = obj_with(disc, arena_m, "arena"); arena.location = (0, 0, FLOOR_Z - 0.001)
gridm = flat_material("grid", (0.86, 0.86, 0.86), 1.0)
grid_lines = []
for r in (1.05,):
    grid_lines.append([(r * math.cos(a), r * math.sin(a), FLOOR_Z) for a in np.linspace(0, 2 * math.pi, 181)])
for x in np.arange(-1.0, 1.01, 0.2):
    h = math.sqrt(max(1.05 ** 2 - x * x, 0))
    grid_lines += [[(x, -h, FLOOR_Z), (x, h, FLOOR_Z)], [(-h, x, FLOOR_Z), (h, x, FLOOR_Z)]]
grid = tube("grid", grid_lines, 0.0012, gridm)
cross = tube("cross", [[(-0.04, 0, FLOOR_Z + 0.001), (0.04, 0, FLOOR_Z + 0.001)], [(0, -0.04, FLOOR_Z + 0.001), (0, 0.04, FLOOR_Z + 0.001)]],
             0.0015, flat_material("crossm", (0.25, 0.35, 0.2), 1.0))

# --- skulls + head paths -------------------------------------------------------------------
bpy.ops.wm.stl_import(filepath=S["cap"]["stl"])
cap_src = bpy.context.selected_objects[0]
V = np.empty(len(cap_src.data.vertices) * 3); cap_src.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
c0 = (V.min(0) + V.max(0)) / 2
V = (V - c0) / 1000.0 @ np.asarray(S["cap"]["fixed_rotation"]).T
cap_src.data.vertices.foreach_set("co", V.ravel()); cap_src.data.update()
skull, skull_mat, path_obj, path_mat = {}, {}, {}, {}
cap_mesh = cap_src.data
cap_mesh.materials.clear(); cap_mesh.materials.append(bpy.data.materials.new("cap_placeholder"))
for r in (1, 2):
    skull_mat[r] = lit_material(f"skull{r}", RAT_RGB[r], 1.0, 0.45)
    o = bpy.data.objects.new(f"skull{r}", cap_mesh); sc.collection.objects.link(o)
    o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = skull_mat[r]     # the two skulls share the mesh
    skull[r] = o
    path_mat[r] = flat_material(f"path{r}", RAT_RGB[r], 1.0)
    path_obj[r] = tube(f"path{r}", [], 0.0022, path_mat[r])
bpy.data.objects.remove(cap_src)


def pose(r, f):
    return POSES[str(r)].get(str(f))


def path_points(r, f_last, f_first=SKULL_START):
    """The head path: the head the cameras tracked (GUI 5.2), as in placement.py."""
    out, cur = [], []
    for f in range(f_first, f_last + 1):
        h = PL.tracked_head(r, f)
        if h is not None:
            cur.append(list(h))
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def place_skull(r, f):
    M = PL.skull_matrix(r, f)
    o = skull[r]
    if M is None:
        o.hide_render = True
        return
    o.hide_render = False
    o.matrix_world = Matrix(M.tolist())


# --- reconstruction at frame 500 ----------------------------------------------------------
REC = S["recon"]
if os.environ.get("TMPL2"):            # comparing alternative rat 2 fits at frame 500 (refit_variant.py)
    REC["2"]["template_textured"] = os.environ["TMPL2"]
ray_obj, ray_mat, seed_obj = {}, {}, {}
for r in (1, 2):
    rr = REC[str(r)]
    ray_mat[r] = flat_material(f"rays{r}", RAT_RGB[r], 0.9)
    ray_obj[r] = tube(f"rays{r}", [], 0.0012, ray_mat[r])
    me = bpy.data.meshes.new(f"seed{r}"); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.008); bm.to_mesh(me); bm.free()
    seed_obj[r] = obj_with(me, lit_material(f"seedm{r}", RAT_RGB[r], 1.0, 0.3), f"seed{r}")
    seed_obj[r].location = rr["seed_T"]

VOX = {r: np.load(D / f"voxels_rat{r}.npz") for r in (1, 2)}
vox_obj, vox_mat = {}, {}
CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
CUBE_F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def vox_material(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = "col"; at.attribute_type = 'GEOMETRY'
    nt.links.new(at.outputs['Color'], p.inputs['Base Color'])
    nt.links.new(at.outputs['Alpha'], p.inputs['Alpha'])
    p.inputs['Roughness'].default_value = 0.7
    m.surface_render_method = 'BLENDED'
    return m


def set_voxels(r, centres, rgba, size):
    """One mesh of cubes (centres (n,3)), per-cube colour + alpha in the 'col' attribute."""
    if r in vox_obj:
        old = vox_obj[r].data
        vox_obj[r].data = bpy.data.meshes.new(f"vox{r}")
        bpy.data.meshes.remove(old)
    else:
        vox_mat[r] = vox_material(f"voxm{r}")
        vox_obj[r] = obj_with(bpy.data.meshes.new(f"vox{r}"), vox_mat[r], f"vox{r}")
    me = vox_obj[r].data
    n = len(centres)
    if n == 0:
        return
    verts = (centres[:, None, :] + CUBE[None] * size / 2).reshape(-1, 3)
    faces = (np.arange(n)[:, None, None] * 8 + np.array(CUBE_F)[None]).reshape(-1, 4)
    me.vertices.add(len(verts)); me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(faces) * 4); me.loops.foreach_set("vertex_index", faces.ravel())
    me.polygons.add(len(faces)); me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 4, 4))
    me.update()
    if not vox_obj[r].data.materials:
        vox_obj[r].data.materials.append(vox_mat[r])
    attr = me.color_attributes.new("col", 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", np.repeat(rgba, 8, axis=0).ravel())


def _crop(r):
    """Display only the part of GUI 2's 40 x 40 x 25 cm grid around the rat (final voxels + 4 cm)."""
    v = VOX[r]
    c, hits = v["centres_T"], v["hits"]
    kept = (~hits).sum(0) <= 2
    lo, hi = np.percentile(c[kept], 1, axis=0) - 0.03, np.percentile(c[kept], 99, axis=0) + 0.03
    lo[2] = max(lo[2], FLOOR_Z)
    return np.all((c >= lo) & (c <= hi), axis=1)


CROP = {r: _crop(r) for r in (1, 2)}


def show_carve(r, k_float, fade=1.0):
    """Voxel display for rat r after k_float cameras (0..6, fractional = the current camera's step)."""
    v = VOX[r]
    sel = CROP[r]
    centres, hits, size = v["centres_T"][sel], v["hits"][:, sel], float(v["size"]) * 0.82
    k = int(math.floor(k_float))
    frac = k_float - k
    misses_done = (~hits[:k]).sum(0) if k else np.zeros(hits.shape[1], int)
    cur_miss = ~hits[k] if k < 6 else np.zeros(hits.shape[1], bool)
    # a voxel is gone at its third miss (it can no longer be inside 4 of the 6 silhouettes)
    gone_before = misses_done >= 3
    keep = ~gone_before
    misses_now = misses_done + (cur_miss & (frac > 0.35))
    vanishing = keep & (misses_now >= 3)
    base = np.array([*RAT_RGB[r], 1.0])
    rgba = np.tile(base, (len(centres), 1))
    rgba[:, 3] = np.where(misses_now == 0, 0.8, np.where(misses_now == 1, 0.42, 0.22))
    if k_float >= 6:
        rgba[:, 3] = 0.95                              # carved: every kept voxel solid
    grey = np.array([0.72, 0.72, 0.72])
    m1 = misses_now >= 1
    rgba[m1, :3] = 0.5 * rgba[m1, :3] + 0.5 * grey
    # the voxels this camera removes fade out over the second half of its step
    rgba[vanishing, 3] *= max(0.0, 1 - (frac - 0.35) / 0.65) if frac > 0.35 else 1.0
    rgba[:, 3] *= fade
    show = keep & (rgba[:, 3] > 0.01)
    set_voxels(r, centres[show], rgba[show], size)


def import_obj(path, name, floor=True):
    bpy.ops.wm.obj_import(filepath=path, forward_axis='Y', up_axis='Z')
    o = bpy.context.selected_objects[0]
    o.name = name
    o.data.transform(T_CALIB)
    o.matrix_world = Matrix.Identity(4)
    if floor:
        V = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        V[:, 2] = np.maximum(V[:, 2], FLOOR_Z + 0.002)
        o.data.vertices.foreach_set("co", V.ravel()); o.data.update()
    return o


WB = {1: (1.0, 1.0, 1.0), 2: (1.0, 1.0, 1.0)}      # per-rat white balance of the fur textures (set below)


def white_balance(img):
    """Linear per-channel gains that bring the brightest fur in a GUI 2 texture to white: the textures are
    baked from the camera images, where the arena's green light turns white fur yellow-green."""
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32); img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)[:, :3]
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    px = px[lum > 0.05]; lum = lum[lum > 0.05]
    ref = px[lum >= np.percentile(lum, 95)].mean(0)
    lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    gains = lin(np.float32(0.92)) / np.maximum(lin(ref), 1e-3)
    return tuple(float(g) for g in np.clip(gains, 0.6, 3.0))


FUR_SAT = {1: 0.8, 2: 0.35}      # rat 2 is black and white in the camera views, rat 1 has a brown hood
# rat 2's hood (head, neck, shoulders) is solid black in all six cameras; the baked texture smears white over the
# face where the head bends under the headstage, so the template's front part is darkened towards that black
_T0x = np.array([[float(v) for v in l.split()[1:4]] for l in open(Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])[:, 0]
HOOD_W = np.clip((_T0x - np.percentile(_T0x, 56)) / (np.percentile(_T0x, 70) - np.percentile(_T0x, 56)), 0, 1).astype(np.float32)
HOOD_BROWN, HOOD_MIX = (0.045, 0.030, 0.022, 1.0), 0.85      # warm dark brown as in the cameras; lit, so ears/eyes show


def textured_material(o, rgb, name, wb=None, sat=1.0, hood=False):
    hood_fac = None
    """The OBJ's texture, unlit (it is baked from the camera images), mixed with a flat lit rat colour by
    node 'C' (0 = texture, 1 = colour) and faded by node 'A' (alpha); 'W' scales the alpha by the
    'added' attribute when it exists (template additions)."""
    img = None
    if o.data.materials and o.data.materials[0] and o.data.materials[0].node_tree:
        img = next((n.image for n in o.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE'), None)
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    pr = nt.nodes.new('ShaderNodeBsdfPrincipled'); pr.inputs['Base Color'].default_value = (*rgb, 1)
    pr.inputs['Roughness'].default_value = 0.7; pr.inputs['Specular IOR Level'].default_value = 0.25
    pr.inputs['Emission Color'].default_value = (*rgb, 1); pr.inputs['Emission Strength'].default_value = 0.35
    if img:
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img
        col = tex.outputs['Color']
        if wb:
            mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = wb
            nt.links.new(col, mul.inputs[0]); col = mul.outputs[0]
        # despill: the arena's green light and green background baked into the fur, G <= (R + B) / 2
        sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(col, sep.inputs[0])
        rb = nt.nodes.new('ShaderNodeMath'); rb.operation = 'ADD'
        nt.links.new(sep.outputs[0], rb.inputs[0]); nt.links.new(sep.outputs[2], rb.inputs[1])
        half = nt.nodes.new('ShaderNodeMath'); half.operation = 'MULTIPLY'; half.inputs[1].default_value = 0.5
        nt.links.new(rb.outputs[0], half.inputs[0])
        g = nt.nodes.new('ShaderNodeMath'); g.operation = 'MINIMUM'
        nt.links.new(sep.outputs[1], g.inputs[0]); nt.links.new(half.outputs[0], g.inputs[1])
        # and no blue cast either (fur is never blue; the white balance tints faces no camera saw)
        rg = nt.nodes.new('ShaderNodeMath'); rg.operation = 'MAXIMUM'
        nt.links.new(sep.outputs[0], rg.inputs[0]); nt.links.new(g.outputs[0], rg.inputs[1])
        bb = nt.nodes.new('ShaderNodeMath'); bb.operation = 'MINIMUM'
        nt.links.new(sep.outputs[2], bb.inputs[0]); nt.links.new(rg.outputs[0], bb.inputs[1])
        comb = nt.nodes.new('ShaderNodeCombineColor')
        nt.links.new(sep.outputs[0], comb.inputs[0]); nt.links.new(g.outputs[0], comb.inputs[1]); nt.links.new(bb.outputs[0], comb.inputs[2])
        hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = sat
        nt.links.new(comb.outputs[0], hs.inputs['Color'])
        last = hs.outputs[0]
        if hood and len(o.data.vertices) == len(HOOD_W):
            if "hood" not in o.data.attributes:
                o.data.attributes.new("hood", 'FLOAT', 'POINT').data.foreach_set("value", HOOD_W)
            an = nt.nodes.new('ShaderNodeAttribute'); an.attribute_name = "hood"; an.attribute_type = 'GEOMETRY'
            hood_fac = nt.nodes.new('ShaderNodeMath'); hood_fac.operation = 'MULTIPLY'; hood_fac.inputs[1].default_value = HOOD_MIX
            nt.links.new(an.outputs['Fac'], hood_fac.inputs[0])
        nt.links.new(last, em.inputs['Color'])
    else:
        em.inputs['Color'].default_value = (*rgb, 1)
    tex_shader = em.outputs[0]
    if hood_fac is not None:
        # the hood is shaded (lit dark-brown fur) instead of the flat baked texture, so its form reads
        ph = nt.nodes.new('ShaderNodeBsdfPrincipled'); ph.inputs['Base Color'].default_value = HOOD_BROWN
        ph.inputs['Roughness'].default_value = 0.6; ph.inputs['Specular IOR Level'].default_value = 0.2
        mh = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(hood_fac.outputs[0], mh.inputs[0]); nt.links.new(em.outputs[0], mh.inputs[1]); nt.links.new(ph.outputs[0], mh.inputs[2])
        tex_shader = mh.outputs[0]
    mixc = nt.nodes.new('ShaderNodeMixShader'); mixc.name = "C"; mixc.inputs[0].default_value = 0.0
    nt.links.new(tex_shader, mixc.inputs[1]); nt.links.new(pr.outputs[0], mixc.inputs[2])
    if "added" in o.data.attributes:
        atc = nt.nodes.new('ShaderNodeAttribute'); atc.attribute_name = "added"; atc.attribute_type = 'GEOMETRY'
        cv = nt.nodes.new('ShaderNodeMath'); cv.operation = 'MULTIPLY'; cv.name = "CV"; cv.inputs[1].default_value = 0.0
        nt.links.new(atc.outputs['Fac'], cv.inputs[0]); nt.links.new(cv.outputs[0], mixc.inputs[0])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mixa = nt.nodes.new('ShaderNodeMixShader'); mixa.name = "A0"; mixa.inputs[0].default_value = 1.0
    if "added" in o.data.attributes:
        at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = "added"; at.attribute_type = 'GEOMETRY'
        w = nt.nodes.new('ShaderNodeMath'); w.operation = 'MAXIMUM'; w.name = "W"; w.inputs[1].default_value = 1.0
        nt.links.new(at.outputs['Fac'], w.inputs[0])
        mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.name = "AM"; mul.inputs[1].default_value = 1.0
        nt.links.new(w.outputs[0], mul.inputs[0])
        nt.links.new(mul.outputs[0], mixa.inputs[0])
    nt.links.new(tr.outputs[0], mixa.inputs[1]); nt.links.new(mixc.outputs[0], mixa.inputs[2])
    nt.links.new(mixa.outputs[0], out.inputs['Surface'])
    set_method(m, 1.0)
    old = list(o.data.materials)
    o.data.materials.clear(); o.data.materials.append(m)
    for om in old:
        if om and om.users == 0:
            bpy.data.materials.remove(om)
    return m


def mat_set(m, colour=None, alpha=None, added_only=None):
    n = m.node_tree.nodes
    if colour is not None:
        if "CV" in n:
            n["CV"].inputs[1].default_value = colour        # colour on the template's additions only
        else:
            n["C"].inputs[0].default_value = colour
    if alpha is not None:
        if "AM" in n:
            n["AM"].inputs[1].default_value = alpha
        else:
            n["A0"].inputs[0].default_value = alpha
    if added_only is not None and "W" in n:
        n["W"].inputs[1].default_value = 1.0 - added_only     # 1: alpha follows 'added' alone; 0: everywhere
    a = n["AM"].inputs[1].default_value if "AM" in n else n["A0"].inputs[0].default_value
    partial = "W" in n and n["W"].inputs[1].default_value < 0.999
    set_method(m, a, partial)


hull, hull_mat, tmpl, tmpl_mat = {}, {}, {}, {}
for r in (1, 2):
    tmpl[r] = import_obj(REC[str(r)]["template_textured"], f"tmpl{r}")
    WB[r] = tuple(json.load(open(D / "fur_gains.json"))["gains"])        # fur_gains.py: one balance for every texture
    print(f"rat {r}: fur white balance gains {np.round(WB[r], 2)}")
    hull[r] = import_obj(REC[str(r)]["hull_textured"], f"hull{r}")
    hull_mat[r] = textured_material(hull[r], RAT_RGB[r], f"hullm{r}", WB[r], FUR_SAT[r])
    added = np.load(os.environ["MASK2"] if (r == 2 and os.environ.get("MASK2")) else D / f"template_mask_rat{r}.npy")   # template_mask.py
    if len(added) == len(tmpl[r].data.vertices):
        a = tmpl[r].data.attributes.new("added", 'FLOAT', 'POINT')
        a.data.foreach_set("value", np.clip(added, 0, 1).astype(np.float32))
    else:
        print(f"rat {r}: template has {len(tmpl[r].data.vertices)} vertices, 'added' has {len(added)}; not highlighting")
    tmpl_mat[r] = textured_material(tmpl[r], RAT_RGB[r], f"tmplm{r}", WB[r], FUR_SAT[r], hood=(r == 2))

# --- per-frame rats (template fits, textured) -------------------------------------------------
live = {1: None, 2: None}
live_frame = {1: None, 2: None}
live_mat = {1: None, 2: None}


live_lift = {1: 0.0, 2: 0.0}


def show_live(r, f, visible=True, on_env=False):
    paths = MESHES[str(r)]
    if not visible:
        if live[r]:
            live[r].hide_render = True
        return
    fr = f if str(f) in paths else min((int(k) for k in paths), key=lambda k: abs(k - f))
    if live_frame[r] != fr:
        if live[r]:
            me, mats = live[r].data, list(live[r].data.materials)
            bpy.data.objects.remove(live[r]); bpy.data.meshes.remove(me)
            for m in mats:
                if m:
                    imgs = [n.image for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
                    bpy.data.materials.remove(m)
                    for im in imgs:
                        if im.users == 0:
                            bpy.data.images.remove(im)
        live[r] = import_obj(paths[str(fr)], f"live{r}")
        live_mat[r] = textured_material(live[r], RAT_RGB[r], f"livem{r}_{fr}", WB[r], FUR_SAT[r], hood=(r == 2))
        live_frame[r] = fr
        V = np.empty(len(live[r].data.vertices) * 3); live[r].data.vertices.foreach_get("co", V)
        live_lift[r] = env_lift(V.reshape(-1, 3).mean(0))
    live[r].hide_render = False
    live[r].location = (0, 0, live_lift[r] if on_env else 0.0)


# --- environment 4 ------------------------------------------------------------------------------
bpy.ops.wm.obj_import(filepath=S["environment"]["obj"], forward_axis='Y', up_axis='Z')
env = list(bpy.context.selected_objects)
TE = Matrix(S["environment"]["T_from_scan"])
env_mats = []
for o in env:
    o.data.transform(TE); o.matrix_world = Matrix.Identity(4)
    for i, m in enumerate(o.data.materials):
        img = next((n.image for n in m.node_tree.nodes if n.type == 'TEX_IMAGE'), None) if m and m.node_tree else None
        nm = bpy.data.materials.new(f"env_{o.name}_{i}"); nm.use_nodes = True
        nt = nm.node_tree; nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
        if img:
            tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; nt.links.new(tex.outputs['Color'], em.inputs['Color'])
        tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        mix = nt.nodes.new('ShaderNodeMixShader'); mix.name = "A"
        nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs['Surface'])
        nm.surface_render_method = 'DITHERED'
        o.data.materials[i] = nm; env_mats.append(nm)

from mathutils.bvhtree import BVHTree
_ev, _ep = [], []
for o in env:
    _off = len(_ev); _ev += [v.co.copy() for v in o.data.vertices]
    _ep += [tuple(_off + i for i in p.vertices) for p in o.data.polygons]
ENV_BVH = BVHTree.FromPolygons(_ev, _ep)
del _ev, _ep


def env_z(x, y):
    """Height of the scanned table under (x, y): the cloth of Environment 4 lies 1 to 2 cm above the arena
    floor the cameras were calibrated on."""
    h = ENV_BVH.ray_cast(Vector((x, y, 0.4)), Vector((0, 0, -1)))
    return h[0].z if h[0] is not None and h[0].z > FLOOR_Z - 0.03 else FLOOR_Z


def surface_below(xy, radius=0.015):
    """Highest point of the scanned table within radius of xy."""
    return max(env_z(xy[0] + dx, xy[1] + dy) for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)))


def env_lift(xy, radius=0.03):
    """How far the scanned table lies above the arena floor around xy (rats and eye are raised by it)."""
    zs = [env_z(xy[0] + dx, xy[1] + dy) for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius))]
    return max(0.0, float(np.median(zs)) - FLOOR_Z)


# --- rat 1's field of view ----------------------------------------------------------------------
FOV_DEPTH = 0.16
fov_mat = flat_material("fov", RAT_RGB[1], 0.25)
fov_edge_mat = flat_material("fov_edge", tuple(0.7 * v for v in RAT_RGB[1]), 1.0)
fov_obj = obj_with(bpy.data.meshes.new("fov"), fov_mat, "fov")
fov_edges = tube("fov_edges", [], 0.0012, fov_edge_mat)


from mathutils.bvhtree import BVHTree
_bvh = {"key": None, "tree": None}


def rat2_tree():
    """BVH of the rat 2 mesh on show (None when it is hidden)."""
    o = live[2]
    if o is None or o.hide_render:
        return None
    key = (live_frame[2], round(o.location.z, 5))
    if _bvh["key"] != key:
        me, off = o.data, o.location.copy()
        _bvh["tree"] = BVHTree.FromPolygons([v.co + off for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        _bvh["key"] = key
    return _bvh["tree"]


def inside(tree, p):
    """Ray parity, majority of three directions (the fitted meshes have small openings): is p inside?"""
    votes = 0
    for d in ((0.13, 0.31, 0.94), (0.71, -0.62, 0.33), (-0.55, -0.2, -0.81)):
        n, q, d = 0, Vector(p), Vector(d).normalized()
        for _ in range(60):
            hit = tree.ray_cast(q, d)
            if hit[0] is None:
                break
            n += 1; q = hit[0] + d * 1e-5
        votes += n % 2
    return votes >= 2


EYE_PUSH = {}      # cm the eye was moved back out of rat 2, per recording frame


def _base_eye(r, f):
    """Rat r's eye camera before keeping clear of rat 2: rat 1 in the first-person range uses the gimbal-stabilised
    pose (placement.gimbal_eye), raised onto the scanned table like the rats, never inside it."""
    g = PL.gimbal_eye(r, f) if r == 1 else None
    e, R = g if g is not None else PL.eye_pose(r, f)
    eye, R = Vector(e.tolist()), Matrix(R.tolist())
    eye.z += env_lift(eye)                               # standing on the scanned table, like the rats
    eye.z = max(eye.z, surface_below(eye) + 0.015)      # never inside the scanned table
    return eye, R


def _rat2_tree_at(f):
    """BVH of rat 2's mesh at frame f placed as show_live places it in the room (floor clamp, lift onto the scan)."""
    paths = MESHES["2"]
    fr = f if str(f) in paths else min((int(k) for k in paths), key=lambda k: abs(k - f))
    V, P = [], []
    for line in open(paths[str(fr)]):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            P.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    V = np.array(V) @ np.array(T_CALIB)[:3, :3].T + np.array(T_CALIB)[:3, 3]
    V[:, 2] = np.maximum(V[:, 2], FLOOR_Z + 0.002)
    V[:, 2] += env_lift(V.mean(0))
    return BVHTree.FromPolygons([Vector(v) for v in V], P)


_PUSH = {}


def fpv_push():
    """How far rat 1's eye moves back along its nose axis to stay outside rat 2 and 12 mm clear of its fur (nose
    to fur when sniffing), for every first-person frame; widened over +-3 frames and smoothed so it never
    stutters, never less than needed."""
    if not _PUSH:
        fs = list(range(750, 1001))
        raw = []
        for f in fs:
            eye, R = _base_eye(1, f)
            tree, push = _rat2_tree_at(f), 0
            for _ in range(20):
                loc, nrm, idx, dist = tree.find_nearest(eye)
                if loc is None or (dist >= 0.012 and not inside(tree, eye)):
                    break
                eye = eye - 0.005 * R.col[0]; push += 1
            raw.append(push * 0.005)
        raw = np.array(raw)
        wide = np.array([raw[max(0, i - 3): i + 4].max() for i in range(len(raw))])
        _PUSH.update(zip(fs, np.maximum(PL._gauss(wide, 2.0), raw)))
        print(f"eye push-back from rat 2: {int((raw > 0).sum())} frames, max {raw.max() * 100:.1f} cm")
    return _PUSH


def head_frame(r, f):
    eye, R = _base_eye(r, f)
    if r == 1 and 750 <= f <= 1000:
        p = float(fpv_push()[f])
        eye = eye - p * R.col[0]
        EYE_PUSH[f] = round(float(p) * 100, 2)
    return eye, R          # columns: nose, left, up


def build_fov(f):
    eye, R = head_frame(1, f)
    nose, left, up = R.col[0], R.col[1], R.col[2]
    tv = math.tan(math.radians(FOV_V_DEG / 2)); th = tv * FOV_ASPECT
    corners = [eye + FOV_DEPTH * (nose + sx * th * (-left) + sy * tv * up) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    me = fov_obj.data; me.clear_geometry()
    me.from_pydata([eye] + corners, [], [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 2, 3, 4)])
    set_tube(fov_edges, [[eye, k] for k in corners] + [[corners[i], corners[(i + 1) % 4]] for i in range(4)])


def eye_camera_matrix(f):
    eye, R = head_frame(1, f)
    nose, left, up = R.col[0], R.col[1], R.col[2]
    xc, yc, zc = -left, up, -nose            # Blender camera: looks down -z, x right, y up
    return Matrix.Translation(eye) @ Matrix((xc, yc, zc)).transposed().to_4x4()


# --------------------------------------------------------------------------- camera path
def look_at(eye, target, fov_v_deg):
    eye, target = Vector(eye), Vector(target)
    zc = (eye - target).normalized()
    xc = Vector((0, 0, 1)).cross(zc).normalized()
    yc = zc.cross(xc)
    cam.matrix_world = Matrix.Translation(eye) @ Matrix((xc, yc, zc)).transposed().to_4x4()
    cam_data.sensor_fit = 'VERTICAL'; cam_data.angle_y = math.radians(fov_v_deg)


def orbit(target, dist, elev_deg, az_deg):
    e, a = math.radians(elev_deg), math.radians(az_deg)
    t = Vector(target)
    return t + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def centre_of(frames):
    pts = [list(PL.tracked_head(r, f)) for r in (1, 2) for f in frames if PL.tracked_head(r, f) is not None]
    return list(np.mean(pts, axis=0)) if pts else [0, 0, 0.1]


C_SKULLS = centre_of(range(SKULL_START, 501))
C_RECON = list((np.array(REC["1"]["seed_T"]) + np.array(REC["2"]["seed_T"])) / 2)
C_MOTION = centre_of(range(500, 751))
AZ0 = -100.0
SHOTS = {   # (target, distance, elevation, azimuth, vertical fov)
    "rig": ((0, 0, 0.50), 2.55, 24.0, AZ0, 38.0),
    "skulls": (C_SKULLS, 0.75, 52.0, AZ0 + 60, 38.0),
    "recon": (C_RECON, 0.55, 35.0, AZ0 + 75, 38.0),
    "motion": (C_MOTION, 0.75, 40.0, AZ0 + 100, 38.0),
    "env": ((0, 0, 0.12), 1.0, 42.0, AZ0 + 125, 58.0),     # inside the green backdrop of the scan
}


def blend_shot(a, b, u):
    u = ease(u)
    ta, tb = Vector(a[0]), Vector(b[0])
    return (ta.lerp(tb, u), a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u, a[4] + (b[4] - a[4]) * u)


def camera_for(vf):
    k, u, i = seg_of(vf)
    if k == "intro":
        t, d, e, a, fv = SHOTS["rig"]; shot = (Vector(t), d, e, a + 60 * u, fv)
    elif k == "zoom":
        r = SHOTS["rig"]; shot = blend_shot((r[0], r[1], r[2], r[3] + 60, r[4]), SHOTS["skulls"], u)
    elif k == "skulls":
        t, d, e, a, fv = SHOTS["skulls"]; shot = (Vector(t), d, e, a + 15 * u, fv)
    elif k == "dim":
        s, rc = SHOTS["skulls"], SHOTS["recon"]
        shot = blend_shot((s[0], s[1], s[2], s[3] + 15, s[4]), (rc[0], rc[1] + 0.28, rc[2] + 6, rc[3], rc[4]), u)
    elif k in ("seed", "block", "carve1", "carve2", "surface", "truecol", "template", "fur"):
        a0, b0 = SEG["seed"][0], SEG["fur"][1]
        t, d, e, a, fv = SHOTS["recon"]
        far = 1.0 if k in ("seed", "block", "carve1", "carve2") else (1 - ease(u) if k == "surface" else 0.0)
        shot = (Vector(t), d + 0.28 * far, e + 6 * far, a + 40 * (vf - a0) / (b0 - a0), fv)
    elif k == "motion":
        rc = SHOTS["recon"]; shot = blend_shot((rc[0], rc[1], rc[2], rc[3] + 40, rc[4]), SHOTS["motion"], min(1, u * 2.5))
    elif k == "env":
        shot = blend_shot(SHOTS["motion"], SHOTS["env"], u)
    elif k == "fov":
        f = 750
        eye, R = head_frame(1, f)
        side = (eye + 0.30 * (-R.col[1]) + Vector((0, 0, 0.25)) - 0.20 * R.col[0])
        env_s = SHOTS["env"]
        start = orbit(env_s[0], env_s[1], env_s[2], env_s[3])
        pos = Vector(start).lerp(side, ease(u))
        tgt = Vector(env_s[0]).lerp(eye + 0.08 * R.col[0], ease(u))
        look_at(pos, tgt, env_s[4] + (45 - env_s[4]) * ease(u))
        return
    elif k == "flyin":
        eye, R = head_frame(1, 750)
        side = (eye + 0.30 * (-R.col[1]) + Vector((0, 0, 0.25)) - 0.20 * R.col[0])
        look_at(side, eye + 0.08 * R.col[0], 45)
        Ma = cam.matrix_world.copy()
        Mb = eye_camera_matrix(750)
        w = ease(u)
        loc = Ma.to_translation().lerp(Mb.to_translation(), w)
        rot = Ma.to_quaternion().slerp(Mb.to_quaternion(), w)
        cam.matrix_world = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        cam_data.sensor_fit = 'VERTICAL'; cam_data.angle_y = math.radians(45 + (FOV_V_DEG - 45) * w)
        return
    else:
        cam.matrix_world = eye_camera_matrix(rec_frame(vf))
        cam_data.sensor_fit = 'VERTICAL'; cam_data.angle_y = math.radians(FOV_V_DEG)
        return
    t, d, e, a, fv = shot
    look_at(orbit(t, d, e, a), t, fv)


# --------------------------------------------------------------------------- per-frame state
def set_rig_alpha(a):
    for m, base in rig_mats:
        set_alpha(m, base * a)
    for o in rig_objs + labels:
        o.hide_render = a < 0.01


def apply_state(vf):
    k, u, i = seg_of(vf)
    f = rec_frame(vf)
    order = list(SEG)
    at_or_after = lambda name: order.index(k) >= order.index(name)

    # moving rats (template fits): from the motion segment on; rat 1 fades out as the camera flies into its
    # head and is gone in its own first-person view (set first: the eye camera keeps clear of rat 2)
    for r in (1, 2):
        vis = k in ("motion", "env", "fov") or (k == "flyin" and (r == 2 or u < 0.6)) or (k == "fpv" and r == 2)
        show_live(r, f, vis, on_env=at_or_after("env") and ENV_PASS != "none")
        if vis:
            a = 1 - ease((u - 0.35) / 0.25) if (k == "flyin" and r == 1) else 1.0
            mat_set(live_mat[r], colour=0.0, alpha=a)
    camera_for(vf)

    # rig: full in the intro, fading out during the zoom, gone afterwards
    rig_a = 1.0 if k == "intro" else (1 - ease(u / 0.45) if k == "zoom" else 0.0)
    set_rig_alpha(rig_a)
    for t in labels:
        t.hide_render = k not in ("intro", "zoom")
        t.rotation_euler = cam.matrix_world.to_euler()
    # arena disc + floor grid until Environment 4 replaces them (crossfade in finish_slide11.py)
    floor_on = not at_or_after("env") or (ENV_PASS == "none" and k == "env")
    grid.hide_render = cross.hide_render = arena.hide_render = not floor_on

    # skulls: from the zoom until the reconstruction starts
    for r in (1, 2):
        if k in ("zoom", "skulls"):
            place_skull(r, f); set_alpha(skull_mat[r], 1.0 if k == "skulls" else ease(u))
        elif k == "dim":
            place_skull(r, 500); set_alpha(skull_mat[r], 1 - ease(u))
        else:
            skull[r].hide_render = True
    # head paths: grow with the skulls, then thin away during the pause at 500; no tracking once the rats are
    # reconstructed
    for r in (1, 2):
        w = 1.0 if k in ("zoom", "skulls") else (1 - ease(u) if k == "dim" else 0.0)
        path_obj[r].hide_render = w < 0.02
        if w < 0.02:
            continue
        set_tube(path_obj[r], path_points(r, f))
        path_obj[r].data.bevel_depth = 0.0022 * w
        set_alpha(path_mat[r], 1.0)

    # seed: rays drawn in, then the point
    for r in (1, 2):
        rays = REC[str(r)]["rays"]
        if k == "seed":
            g = ease(min(1, u * 1.6))
            # rays as 12-point lines, thin at the camera end (a ray passing near the viewpoint stays a line)
            set_tube(ray_obj[r], [[list(np.asarray(ry["from"]) + t * g * (np.asarray(ry["to"]) - np.asarray(ry["from"])))
                                   for t in np.linspace(0, 1, 12)] for ry in rays], taper=(0.12, 1.0))
            ray_obj[r].hide_render = False; set_alpha(ray_mat[r], 0.9)
            seed_obj[r].hide_render = u < 0.55
            seed_obj[r].scale = (1, 1, 1)
        elif k in ("block", "carve1", "carve2"):
            if k == "block":
                set_tube(ray_obj[r], [[list(np.asarray(ry["from"]) + t * (np.asarray(ry["to"]) - np.asarray(ry["from"])))
                                       for t in np.linspace(0, 1, 12)] for ry in rays], taper=(0.12, 1.0))
            ray_obj[r].hide_render = False; set_alpha(ray_mat[r], 0.9 * (1 - ease(u)) if k == "block" else 0.0)
            ray_obj[r].hide_render = k != "block"
            seed_obj[r].hide_render = False
        else:
            ray_obj[r].hide_render = True
            seed_obj[r].hide_render = True

    # voxels: both blocks appear, rat 1 is carved camera by camera, then rat 2
    for r in (1, 2):
        if k == "block":
            show_carve(r, 0.0, fade=ease(u)); vox_obj[r].hide_render = r != 1
        elif k == "carve1":
            show_carve(r, 6 * u if r == 1 else 0.0); vox_obj[r].hide_render = r != 1
        elif k == "carve2":
            if r == 1:
                show_carve(r, 6.0)
            else:
                show_carve(r, max(0.0, 6 * (u - 0.15) / 0.85), fade=ease(u / 0.15))
            vox_obj[r].hide_render = False
        elif k == "surface":
            show_carve(r, 6.0); vox_obj[r].hide_render = u > 0.6
        elif r in vox_obj:
            vox_obj[r].hide_render = True

    # surface (hull): flat colour -> true colours; template: additions in colour, then fur everywhere
    for r in (1, 2):
        h, t = hull[r], tmpl[r]
        if k == "surface":
            h.hide_render = False; mat_set(hull_mat[r], colour=1.0, alpha=ease(u / 0.7))
        elif k == "truecol":
            h.hide_render = False; mat_set(hull_mat[r], colour=1 - ease(u), alpha=1.0)
        elif k == "template":
            h.hide_render = False; mat_set(hull_mat[r], colour=0.0, alpha=1.0)
        elif k == "fur":
            # the hull fades off the (now solid) template, whose additions turn from colour to fur
            h.hide_render = u > 0.35; mat_set(hull_mat[r], colour=0.0, alpha=1 - ease(u / 0.35))
        else:
            h.hide_render = True
        if k == "template":
            t.hide_render = False; mat_set(tmpl_mat[r], colour=1.0, alpha=ease(u / 0.5), added_only=1.0)
        elif k == "fur":
            t.hide_render = False; mat_set(tmpl_mat[r], colour=1 - ease(u), alpha=1.0, added_only=0.0)
        else:
            t.hide_render = True

    # environment 4 from frame 650
    for o in env:
        o.hide_render = not at_or_after("env")
    for m in env_mats:      # the fade-in is a crossfade in finish_slide11.py (ENV_PASS: none / full)
        m.node_tree.nodes["A"].inputs[0].default_value = 1.0
    if ENV_PASS == "none":
        for o in env:
            o.hide_render = True
    bg.inputs["Color"].default_value = (1, 1, 1, 1) if not at_or_after("env") else (0.93, 0.93, 0.93, 1)

    # rat 1's field of view at frame 750
    if k in ("fov", "flyin"):
        build_fov(750)
        fov_obj.hide_render = fov_edges.hide_render = False
        a = ease(u / 0.5) if k == "fov" else 1 - ease(u)
        set_alpha(fov_mat, 0.25 * a); set_alpha(fov_edge_mat, a)
    else:
        fov_obj.hide_render = fov_edges.hide_render = True


def record_view(vf):
    """Where rat 2 (body centroid, head) lands in the image, for the label in finish_slide11.py."""
    from bpy_extras.object_utils import world_to_camera_view
    k, u, i = seg_of(vf)
    if k not in ("flyin", "fpv"):
        return
    f = rec_frame(vf)
    eye, R = head_frame(1, f)
    out = {"frame": f, "eye": list(eye), "nose": list(R.col[0]), "eye_push_cm": EYE_PUSH.get(f, 0), "rat2_body_img": None}
    # the label goes on the part of rat 2's mesh inside the 4:3 first-person crop (finish_slide11.py)
    o = live[2]
    if o is not None and not o.hide_render:
        V = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)[::15]
        x_lo, x_hi = (1920 - 1440) / 2 / 1920 + 0.02, 1 - (1920 - 1440) / 2 / 1920 - 0.02
        qs = [world_to_camera_view(sc, cam, Vector(v)) for v in V]
        vis = np.array([(q.x, q.y) for q in qs if q.z > 0 and x_lo < q.x < x_hi and 0.03 < q.y < 0.97])
        out["rat2_visible_frac"] = round(len(vis) / len(V), 3)
        if len(vis) >= 0.04 * len(V):
            out["rat2_body_img"] = [float(vis[:, 0].mean()), float(vis[:, 1].mean()), 1.0]
    json.dump(out, open(OUT / f"vf_{vf:04d}.json", "w"))


def render(vf):
    apply_state(vf)
    for name in filter(None, os.environ.get("DEBUG_HIDE", "").split(",")):
        bpy.data.objects[name].hide_render = True
    record_view(vf)
    sc.render.filepath = str(OUT / f"vf_{vf:04d}.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) == 1 and "," in argv[0]:
        todo = [int(x) for x in argv[0].split(",")]
    elif len(argv) >= 2:
        todo = list(range(int(argv[0]), min(int(argv[1]), N_VIDEO - 1) + 1))
    else:
        todo = list(range(N_VIDEO))
    json.dump({"segments": SEG, "n_video": N_VIDEO}, open(HERE / "timeline.json", "w"), indent=1)
    for vf in todo:
        if len(argv) >= 3 and (OUT / f"vf_{vf:04d}.png").exists():
            continue
        render(vf)
    print("rendered", len(todo), "frames")
