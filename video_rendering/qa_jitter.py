"""Image-based shake of the first-person view: dense optical flow between consecutive finished frames (4:3 crop,
half resolution); per frame the median flow vector (the camera's apparent motion) and the median flow magnitude.
Shake = how much that motion changes from one frame to the next.

    python qa_jitter.py <frames dir> <first vf> <last vf>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

d, a, b = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
prev, vec, mag = None, [], []
for vf in range(a, b + 1):
    im = cv2.imread(str(d / f"vf_{vf:04d}.png"), cv2.IMREAD_GRAYSCALE)[60:1040, 300:1620]
    im = cv2.resize(im, None, fx=0.5, fy=0.5)
    if prev is not None:
        fl = cv2.calcOpticalFlowFarneback(prev, im, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        gx, gy = cv2.Sobel(prev, cv2.CV_32F, 1, 0), cv2.Sobel(prev, cv2.CV_32F, 0, 1)
        tex = np.hypot(gx, gy) > 20                        # textured pixels only (the cloth backdrop is flat green)
        f2 = fl[tex] if tex.sum() > 500 else fl.reshape(-1, 2)
        vec.append(np.median(f2, axis=0)); mag.append(np.median(np.linalg.norm(f2, axis=1)))
    prev = im
vec, mag = np.array(vec), np.array(mag)
acc = np.linalg.norm(np.diff(vec, axis=0), axis=1)
print(f"{d.name} vf {a}-{b}: image motion per frame median {np.median(mag):.1f} px, p90 {np.percentile(mag, 90):.1f} px; "
      f"shake (change of motion between frames) median {np.median(acc):.1f} px, p90 {np.percentile(acc, 90):.1f} px")
