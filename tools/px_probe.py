"""测每份点云在不同像素尺寸下的高度图占用率。ratio(px) = 占用像素数(px) / (4 * 占用像素数(2px))，
点够密时 ~1，点稀时明显 <1（像素里出现大量空洞）。用法: python -I px_probe.py <pcd> [<pcd> ...]"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz


def occ_count(XY, px):
    mn = XY.min(0)
    ij = ((XY - mn) / px).astype(np.int64)
    W = int(ij[:, 0].max()) + 1; H = int(ij[:, 1].max()) + 1
    return int((np.bincount(ij[:, 1] * W + ij[:, 0], minlength=W * H) > 0).sum())


for f in sys.argv[1:]:
    t = time.time()
    P = read_xyz(f); P = P[np.isfinite(P).all(1)]; XY = P[:, :2].astype(np.float64)
    cands = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    cnt = {px: occ_count(XY, px) for px in sorted(set(cands + [2 * c for c in cands]))}
    ratio = {px: round(cnt[px] / (4.0 * cnt[2 * px]), 3) for px in cands}
    print(json.dumps(dict(f=os.path.basename(f), n=len(P), ratio=ratio, s=round(time.time() - t, 2))))
