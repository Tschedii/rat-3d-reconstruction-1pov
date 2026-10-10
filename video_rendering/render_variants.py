"""Blender: each variant mesh of one rat and frame alone, textured (unlit, white-balanced like the video), from
three angles around its body axis (side, front three-quarter, top), labelled.

    blender -b -P render_variants.py -- <rat> <frame> <name1,name2,...>   -> variants/compare_<rat>_<frame>_<name>.png
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "data/scene.json"))
T = Matrix(S["T_from_calib"])
WB = json.load(open(HERE / "data/fur_gains.json"))["gains"]
r, f, names = sys.argv[sys.argv.index("--") + 1:][:3]
r, f = int(r), int(f)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'; sc.render.resolution_x, sc.render.resolution_y = 640, 480
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.angle = math.radians(30)
T0 = np.array([[float(x) for x in l.split()[1:4]] for l in open(Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/templates/rat_brown.obj") if l.startswith("v ")])
nose_i = np.argsort(T0[:, 0])[-30:]; tail_i = (T0[:, 0] > np.percentile(T0[:, 0], 20)) & (T0[:, 0] < np.percentile(T0[:, 0], 35))
for name in names.split(","):
    for o in list(bpy.data.objects):
        if o.type == 'MESH':
            bpy.data.objects.remove(o)
    p = HERE / ("refit_smooth" if name == "current" else f"variants/{name}") / f"object_{r}" / f"frame_{f:06d}" / "mesh.obj"
    bpy.ops.wm.obj_import(filepath=str(p), forward_axis='Y', up_axis='Z')
    o = bpy.context.selected_objects[0]; o.data.transform(T); o.matrix_world = Matrix.Identity(4)
    img = next(n.image for n in o.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE')
    m = bpy.data.materials.new("m"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img
    mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = WB
    hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = 0.35 if r == 2 else 0.8
    nt.links.new(tex.outputs['Color'], mul.inputs[0]); nt.links.new(mul.outputs[0], hs.inputs['Color'])
    nt.links.new(hs.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs['Surface'])
    o.data.materials.clear(); o.data.materials.append(m)
    V = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    c = V.mean(0); ax = V[nose_i].mean(0) - V[tail_i].mean(0); ax[2] = 0; ax /= np.linalg.norm(ax)
    side = np.cross([0, 0, 1], ax)
    for k, (d, lab) in enumerate(((side * 1.0 + [0, 0, 0.25], "side"), (ax * 0.8 + side * 0.6 + [0, 0, 0.45], "front 3/4"), (np.array([0, 0, 1.0]) + ax * 0.05, "top"))):
        d = np.asarray(d, float); d /= np.linalg.norm(d)
        eye = Vector(c + d * 0.55); tgt = Vector(c)
        zc = (eye - tgt).normalized(); upv = Vector((0, 0, 1)) if abs(d[2]) < 0.95 else Vector(ax.tolist())
        xc = upv.cross(zc).normalized(); yc = zc.cross(xc)
        cam.matrix_world = Matrix.Translation(eye) @ Matrix((xc, yc, zc)).transposed().to_4x4()
        sc.render.filepath = str(HERE / "variants" / f"_v_{name}_{k}.png")
        bpy.ops.render.render(write_still=True)
print("done")
