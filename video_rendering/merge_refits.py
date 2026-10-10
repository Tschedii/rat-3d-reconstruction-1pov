"""Second-round re-fits (refit2/: bloated masks dropped, missing frames filled) merged into refit/: for every frame
refit2 has, the fit that matches the six camera masks better (IoU, flagged masks left out of both) is kept; the
replaced first-round fit is moved to refit_v1/. Then smooth_fits.py is re-run for both rats and data/meshes.json
rebuilt.

    python merge_refits.py   -> logs/merge_rat{1,2}.json
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smooth_fits as sf     # noqa: E402

FLAGS = json.load(open(HERE / "logs" / "mask_flags.json"))


def masks_for(r, f):
    ms = sf.load_masks(r, f)
    for i, c in enumerate(sf.CAL):
        if int(c["name"][-1]) in FLAGS.get(str(r), {}).get(str(f), []):
            ms[i] = None
    return ms


def main():
    for r in (1, 2):
        new = HERE / "refit2" / f"object_{r}_template_fitted"
        old = HERE / "refit" / f"object_{r}_template_fitted"
        bak = HERE / "refit_v1" / f"object_{r}_template_fitted"
        report = {}
        for d in sorted(new.glob("frame_*")):
            if not (d / "mesh.obj").exists():
                continue
            f = int(d.name[6:])
            ms = masks_for(r, f)
            Vn, Fn, _ = sf.read_obj(d / "mesh.obj")
            i_new = sf.iou(Vn, Fn, ms)
            i_old = None
            if (old / d.name / "mesh.obj").exists():
                Vo, Fo, _ = sf.read_obj(old / d.name / "mesh.obj")
                i_old = sf.iou(Vo, Fo, ms)
            use_new = i_old is None or i_new >= i_old - 0.005
            report[f] = {"iou_old": None if i_old is None else round(i_old, 4), "iou_new": round(i_new, 4), "new": bool(use_new)}
            if use_new:
                if (old / d.name).exists():
                    bak.mkdir(parents=True, exist_ok=True)
                    if not (bak / d.name).exists():
                        shutil.move(str(old / d.name), str(bak / d.name))
                    else:
                        shutil.rmtree(old / d.name)
                shutil.copytree(d, old / d.name)
        json.dump(report, open(HERE / "logs" / f"merge_rat{r}.json", "w"), indent=0)
        n_new = sum(v["new"] for v in report.values())
        gains = [v["iou_new"] - v["iou_old"] for v in report.values() if v["iou_old"] is not None]
        print(f"rat {r}: {len(report)} second-round fits, {n_new} kept "
              f"({sum(v['iou_old'] is None for v in report.values())} filling gaps); "
              f"IoU change where both exist: median {np.median(gains) if gains else 0:+.3f}, max {max(gains) if gains else 0:+.3f}", flush=True)
    for r in (1, 2):
        subprocess.run([sys.executable, str(HERE / "smooth_fits.py"), str(r)], check=True)
    # meshes.json, as prep_slide11.py builds it
    GUI2 = Path(__file__).resolve().parents[1] / "gui_2_rat_reconstruction/output"
    meshes = {}
    for obj in (1, 2):
        lst = {}
        for f in range(500, 1001):
            fid = f"frame_{f:06d}"
            for p in (HERE / "refit_smooth" / f"object_{obj}" / fid / "mesh.obj",
                      HERE / "refit" / f"object_{obj}_template_fitted" / fid / "mesh.obj",
                      *([GUI2 / f"object_{obj}_template_fitted" / fid / "mesh.obj"] if obj == 1 and f > 750 else [])):
                if p.exists():
                    lst[f] = str(p)
                    break
        meshes[str(obj)] = {str(f): p for f, p in lst.items()}
        missing = [f for f in range(500, 1001) if f not in lst]
        print(f"rat {obj}: meshes for {len(lst)}/501 frames; missing {missing}")
    json.dump(meshes, open(HERE / "data" / "meshes.json", "w"), indent=0)


if __name__ == "__main__":
    main()
