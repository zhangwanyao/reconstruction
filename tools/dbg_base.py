"""诊断底板是否平：水平法向点 -> 10mm 格中位 z -> 平面/二次曲面拟合残差。用法: python -I dbg_base.py <pcd>"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import voxel_down, normals

P = read_xyz(sys.argv[1]); P = P[np.isfinite(P).all(1)]
V = voxel_down(P, 4.0); N, cv = normals(V)
N *= np.where(N[:, 2:3] < 0, -1, 1)
hz = (cv < 0.02) & (N[:, 2] > np.cos(np.radians(10)))
H = V[hz]
k = np.floor(H[:, :2] / 20).astype(np.int64); key = k[:, 0] * 10 ** 7 + k[:, 1]
u, inv = np.unique(key, return_inverse=True)
zmed = np.array([np.median(H[inv == i, 2]) for i in range(len(u))]) if len(u) < 40000 else None
cx = np.bincount(inv, H[:, 0]) / np.bincount(inv); cy = np.bincount(inv, H[:, 1]) / np.bincount(inv)
zmin_cell = np.full(len(u), np.inf); np.minimum.at(zmin_cell, inv, H[:, 2])
z = zmin_cell
# 取每格最低的水平点（底板），用最低 60% 的格子做稳健拟合
lo = z < np.percentile(z, 60)
for it in range(4):
    A = np.c_[cx[lo], cy[lo], np.ones(lo.sum())]
    c, *_ = np.linalg.lstsq(A, z[lo], rcond=None)
    r = z - np.c_[cx, cy, np.ones(len(cx))] @ c
    s = 1.4826 * np.median(np.abs(r[lo]))
    lo = np.abs(r) < max(1.5, 3 * s)
print("plane: slope x %.5f y %.5f (deg %.3f)  inlier cells %d/%d  rms %.2f  p95 %.2f" % (
    c[0], c[1], np.degrees(np.arctan(np.hypot(c[0], c[1]))), lo.sum(), len(z), np.sqrt(np.mean(r[lo] ** 2)), np.percentile(np.abs(r[lo]), 95)))
A2 = np.c_[cx, cy, np.ones(len(cx)), cx ** 2, cy ** 2, cx * cy]
c2, *_ = np.linalg.lstsq(A2[lo], z[lo], rcond=None)
r2 = z - A2 @ c2
print("quadratic: rms %.2f p95 %.2f" % (np.sqrt(np.mean(r2[lo] ** 2)), np.percentile(np.abs(r2[lo]), 95)))
# 残差沿长轴的分布
L = np.c_[cx, cy][lo]; rr = r[lo]
ax = np.argmax(np.ptp(L, 0))
bins = np.linspace(L[:, ax].min(), L[:, ax].max(), 12)
idx = np.digitize(L[:, ax], bins)
print("plane residual median along long axis:", [round(float(np.median(rr[idx == i])), 2) for i in range(1, 12) if (idx == i).any()])
print("bbox", np.round(P.min(0), 1), np.round(P.max(0), 1))
