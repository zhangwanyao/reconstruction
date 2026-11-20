"""auto_px 稳定性: 网格起点平移 / 坐标系转 180° 时 occ(px)/(4*occ(2px)) 变多少。用法: python -I px_stability.py <pcd>..."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import detect_frame


def occ(xy, p, off):
    ij = np.floor((xy - xy.min(0) + off) / p).astype(np.int64)
    W = int(ij[:, 0].max()) + 1
    return len(np.unique(ij[:, 1] * W + ij[:, 0]))


for f in sys.argv[1:]:
    P = read_xyz(f); P = P[np.isfinite(P).all(1)]
    Q = detect_frame(P)["Q"]
    res = {}
    for p in (1.0, 1.5, 2.0, 2.5, 3.0):
        vals = []
        for flip in (1, -1):
            xy = Q[:, :2] * flip
            for ox in (0.0, 0.25, 0.5, 0.75):
                for oy in (0.0, 0.5):
                    o = np.array([ox, oy]) * p
                    o2 = np.array([ox, oy]) * 2 * p
                    vals.append(occ(xy, p, o) / (4.0 * occ(xy, 2 * p, o2)))
        res[p] = [round(min(vals), 3), round(max(vals), 3)]
    print(json.dumps(dict(f=os.path.basename(f), ratio_min_max=res)), flush=True)
