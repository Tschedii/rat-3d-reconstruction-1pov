"""Blender: the textured rat meshes of one frame seen through each of the six calibrated cameras (pinhole from the
calibration, distortion ignored), shaded like the video (white balance, despill, per-rat saturation), on a
transparent background -> variants/camview_<tag>/cam_N.png (1920 x 1200), to lay over the raw frames.

    blender -b -P render_camviews.py -- <frame> <tag> <rat1 mesh|-> <rat2 mesh|->
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

HERE = Path(__file__).resolve().parent
DRIVE = Path(__file__).resolve().parents[1]  # repository root
S = json.load(open(HERE / "data/scene.json"))
T = np.array(S["T_from_calib"])
CAL = json.loads((DRIVE / "data/calibration/output/2026_09_21/2026_09_21.json").read_text())["cameras"]
WB = json.load(open(HERE / "data/fur_gains.json"))["gains"]
SAT = {1: 0.8, 2: 0.35}
args = sys.argv[sys.argv.index("--") + 1:]
f, tag, m1, m2 = args[:4]
HOOD = "--hood" in args                    # rat 2's black hood: darken the template's head and shoulders
HEADSTAGE = "--headstage" in args          # the headstage scans on both heads (placement.skull_matrix)
sys.path.insert(0, str(HERE))
import placement as PL                     # noqa: E402
T0 = np.array([[float(x) for x in l.split()[1:4]] for l in open(DRIVE / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])
_x = T0[:, 0]
HOOD_W = np.clip((_x - np.percentile(_x, 62)) / (np.percentile(_x, 72) - np.percentile(_x, 62)), 0, 1).astype(np.float32)
out = HERE / "variants" / f"camview_{tag}"; out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'; sc.render.resolution_x, sc.render.resolution_y = 1920, 1200
sc.render.film_transparent = True; sc.view_settings.view_transform = 'Standard'
sc.world = bpy.data.worlds.new("w")


def textured(o, r):
    img = next(n.image for n in o.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE')
    m = bpy.data.materials.new(f"m{r}"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    outn = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img
    mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = WB
    sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(tex.outputs['Color'], mul.inputs[0]); nt.links.new(mul.outputs[0], sep.inputs[0])
    rb = nt.nodes.new('ShaderNodeMath'); rb.operation = 'ADD'; nt.links.new(sep.outputs[0], rb.inputs[0]); nt.links.new(sep.outputs[2], rb.inputs[1])
    half = nt.nodes.new('ShaderNodeMath'); half.operation = 'MULTIPLY'; half.inputs[1].default_value = 0.5; nt.links.new(rb.outputs[0], half.inputs[0])
    g = nt.nodes.new('ShaderNodeMath'); g.operation = 'MINIMUM'; nt.links.new(sep.outputs[1], g.inputs[0]); nt.links.new(half.outputs[0], g.inputs[1])
    rg = nt.nodes.new('ShaderNodeMath'); rg.operation = 'MAXIMUM'; nt.links.new(sep.outputs[0], rg.inputs[0]); nt.links.new(g.outputs[0], rg.inputs[1])
    bb = nt.nodes.new('ShaderNodeMath'); bb.operation = 'MINIMUM'; nt.links.new(sep.outputs[2], bb.inputs[0]); nt.links.new(rg.outputs[0], bb.inputs[1])
    comb = nt.nodes.new('ShaderNodeCombineColor')
    nt.links.new(sep.outputs[0], comb.inputs[0]); nt.links.new(g.outputs[0], comb.inputs[1]); nt.links.new(bb.outputs[0], comb.inputs[2])
    hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = SAT[r]
    nt.links.new(comb.outputs[0], hs.inputs['Color'])
    last = hs.outputs[0]
    if HOOD and r == 2 and len(o.data.vertices) == len(HOOD_W):
        at = o.data.attributes.new("hood", 'FLOAT', 'POINT'); at.data.foreach_set("value", HOOD_W)
        an = nt.nodes.new('ShaderNodeAttribute'); an.attribute_name = "hood"; an.attribute_type = 'GEOMETRY'
        k = nt.nodes.new('ShaderNodeMath'); k.operation = 'MULTIPLY'; k.inputs[1].default_value = 0.8
        nt.links.new(an.outputs['Fac'], k.inputs[0])
        mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.inputs['B'].default_value = (0.035, 0.035, 0.04, 1)
        nt.links.new(k.outputs[0], mx.inputs['Factor']); nt.links.new(last, mx.inputs['A'])
        last = mx.outputs['Result']
    nt.links.new(last, em.inputs['Color'])
    nt.links.new(em.outputs[0], outn.inputs['Surface'])
    o.data.materials.clear(); o.data.materials.append(m)


for r, mp in ((1, m1), (2, m2)):
    if mp == "-":
        continue
    bpy.ops.wm.obj_import(filepath=mp, forward_axis='Y', up_axis='Z')
    o = bpy.context.selected_objects[0]; o.data.transform(Matrix(T.tolist())); o.matrix_world = Matrix.Identity(4)
    textured(o, r)
if HEADSTAGE:
    cap = PL.cap_points(1)            # (only to share the centring with the video)
    bpy.ops.wm.stl_import(filepath=S["cap"]["stl"])
    src = bpy.context.selected_objects[0]
    V = np.empty(len(src.data.vertices) * 3); src.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    V = (V - (V.min(0) + V.max(0)) / 2) / 1000.0 @ np.asarray(S["cap"]["fixed_rotation"]).T
    src.data.vertices.foreach_set("co", V.ravel()); src.data.update()
    hm = bpy.data.materials.new("headstage"); hm.use_nodes = True
    pr = hm.node_tree.nodes["Principled BSDF"]; pr.inputs['Base Color'].default_value = (0.78, 0.74, 0.62, 1); pr.inputs['Roughness'].default_value = 0.4
    em2 = hm.node_tree.nodes.new('ShaderNodeEmission'); em2.inputs['Color'].default_value = (0.70, 0.67, 0.57, 1)
    hm.node_tree.links.new(em2.outputs[0], hm.node_tree.nodes["Material Output"].inputs['Surface'])
    src.data.materials.clear(); src.data.materials.append(hm)
    capv = PL.cap_points(20000); c_box = capv[capv[:, 2] > -0.010].mean(0)     # centre of the box part of the scan
    for r in (1, 2):
        M = PL.skull_matrix(r, int(f))
        real = HERE / "data" / f"rat{r}_headstage_track_T.npy"
        if "--hs-real" in args and real.exists():          # box centre on the headstage triangulated in the images
            tr = np.load(real, allow_pickle=True).item(); P = dict(zip(tr["frames"], tr["smooth"]))[int(f)]
            M = M.copy(); M[:3, 3] = P - M[:3, :3] @ c_box
        o = bpy.data.objects.new(f"hs{r}", src.data); sc.collection.objects.link(o); o.matrix_world = Matrix(M.tolist())
    bpy.data.objects.remove(src)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_start = 0.01
for c in CAL:
    R = np.array(c["rotation_matrix"]); t = np.array(c["translation_m"]); k = c["intrinsics"]
    C = T[:3, :3] @ (-R.T @ t) + T[:3, 3]
    Rw = T[:3, :3] @ R.T @ np.diag([1, -1, -1])                 # OpenCV camera -> Blender camera axes, in the table frame
    M = np.eye(4); M[:3, :3] = Rw; M[:3, 3] = C
    cam.matrix_world = Matrix(M.tolist())
    W, H = 1920, 1200
    cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36.0
    cam.data.lens = k["fx"] * 36.0 / W
    sc.render.pixel_aspect_y = k["fx"] / k["fy"] if abs(k["fx"] - k["fy"]) > 1e-6 else 1.0
    cam.data.shift_x = (W / 2 - k["cx"]) / W
    cam.data.shift_y = (k["cy"] - H / 2) / W
    sc.render.filepath = str(out / f"cam_{c['name'][-1]}.png")
    bpy.ops.render.render(write_still=True)
print("done", out)
