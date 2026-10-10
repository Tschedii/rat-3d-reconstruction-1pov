"""Install a re-fitted chain of one rat (refit_variant.py output) into refit/: frames up to BLEND_FROM - 1 replaced,
frames BLEND_FROM..last blended vertex by vertex into the existing fit (same template, so vertex i is the same point
of the rat in every fit) with the new weight falling linearly to 0 after the last frame, so the motion carries on
without a jump. Replaced fits are backed up to refit_backup_<name>/. Then smooth_fits.py and template_mask.py.

    python install_variant.py <rat> <name> <first> <last> <blend_from> [keep-texture]

keep-texture: only the shape changes; texture, UVs and faces stay those of the installed fit (same template, so
vertex i keeps its colour). Used when a re-fit bends the head into a pose the texture bake renders badly (a white
collar where the bent neck is filled from the chest fur).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smooth_fits as sf     # noqa: E402

r, name, first, last, blend_from = int(sys.argv[1]), sys.argv[2], *map(int, sys.argv[3:6])
keep_texture = len(sys.argv) > 6 and sys.argv[6] == "keep-texture"
new_root = HERE / "variants" / name / f"object_{r}"
old_root = HERE / "refit" / f"object_{r}_template_fitted"
bak_root = HERE / f"refit_backup_{name}" / f"object_{r}_template_fitted"
for f in range(first, last + 1):
    fid = f"frame_{f:06d}"
    w = 1.0 if f < blend_from else (last + 1 - f) / (last + 2 - blend_from)
    Vn, _, rest_n = sf.read_obj(new_root / fid / "mesh.obj")
    Vo, _, rest_o = sf.read_obj(old_root / fid / "mesh.obj")
    if not (bak_root / fid).exists():
        shutil.copytree(old_root / fid, bak_root / fid)
    V = w * Vn + (1 - w) * Vo
    src = new_root if (w >= 0.5 and not keep_texture) else bak_root     # texture, UVs and faces
    for n in ("mesh.mtl", "mesh_texture.png"):
        shutil.copy2(src / fid / n, old_root / fid / n)
    sf.write_obj(old_root / fid / "mesh.obj", V, rest_n if (w >= 0.5 and not keep_texture) else rest_o)
    print(fid, f"new weight {w:.2f}, mean move {np.linalg.norm(V - Vo, axis=1).mean() * 100:.2f} cm")
subprocess.run([sys.executable, str(HERE / "smooth_fits.py"), str(r)], check=True)
subprocess.run([sys.executable, str(HERE / "template_mask.py")], check=True)
