"""调试: 高度图凸起区 -> 骨架 -> 直线段, 画指定区域的放大图(1px=1mm)。
用法: python -I dbg_skel.py <pcd> <out_png> <x0> <y0> <x1> <y1> [--mode auto]
图例: 灰=高度, 黄=骨架, 品红=分叉点, 红=骨架直线段, 绿=底板可见
"""
import sys, os, json
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import detect_frame, heightmap, skeleton_segments, crossing_number, disk

pcd, out = sys.argv[1], sys.argv[2]
X0, Y0, X1, Y1 = map(float, sys.argv[3:7])
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "auto"
P = read_xyz(pcd); P = P[np.isfinite(P).all(1)]
F = detect_frame(P, mode); Q = F["Q"]
hmp = heightmap(Q, 1.0, F["has_base"])
hm, occ, bv = hmp["hm"], hmp["occ"], hmp["base_vis"]
raised = ((hm > 8) if F["has_base"] else occ).astype(np.uint8)
raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
wide = cv2.morphologyEx(raised, cv2.MORPH_OPEN, disk(12))
thin = raised & (1 - cv2.dilate(wide, disk(2)))
sk = cv2.ximgproc.thinning(thin * 255) > 0
junc = sk & (crossing_number(sk) >= 3)
segs = skeleton_segments(thin, hmp)
c0, r0 = int((X0 - hmp["x0"])), int((Y0 - hmp["y0"]))
c1, r1 = int((X1 - hmp["x0"])), int((Y1 - hmp["y0"]))
sl = (slice(max(r0, 0), r1), slice(max(c0, 0), c1))
h = hm[sl]; o = occ[sl]
img = np.zeros(h.shape + (3,), np.uint8)
if o.any():
    lo, hi = np.percentile(h[o], [1, 99.5])
    g = (np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1) * 160 + 60).astype(np.uint8)
    img[o] = np.stack([g[o]] * 3, 1)
img[bv[sl] & (h < 8)] = (40, 90, 40)
img[thin[sl] > 0] = (110, 60, 60)
img[sk[sl]] = (0, 255, 255)
img[junc[sl]] = (255, 0, 255)
n_in = 0
for s in segs:
    a = s["c"] + s["d"] * s["s0"]; b = s["c"] + s["d"] * s["s1"]
    pa = (int(a[0] - hmp["x0"]) - sl[1].start, int(a[1] - hmp["y0"]) - sl[0].start)
    pb = (int(b[0] - hmp["x0"]) - sl[1].start, int(b[1] - hmp["y0"]) - sl[0].start)
    if max(pa[0], pb[0]) < 0 or min(pa[0], pb[0]) > img.shape[1] or max(pa[1], pb[1]) < 0 or min(pa[1], pb[1]) > img.shape[0]:
        continue
    n_in += 1
    cv2.line(img, pa, pb, (0, 0, 255), 1)
    cv2.circle(img, pa, 2, (0, 128, 255), -1); cv2.circle(img, pb, 2, (0, 128, 255), -1)
cv2.imwrite(out, img[::-1])
lens = sorted(round(s["s1"] - s["s0"]) for s in segs)
print(json.dumps(dict(segs=len(segs), segs_in_crop=n_in, skel_px=int(sk.sum()), junc_px=int(junc.sum()),
                      thin_px=int(thin.sum()), lens_top20=lens[-20:], n_short=sum(l < 60 for l in lens))))
