"""参考 STEP 每个面相对点云的偏移: 点云点到最近参考面的有符号距离(沿外法向, 正 = 点在实体外), 按面统计中位数。
用法: python -I ref_offset.py <OCP_deps> <cloud.pcd> <ref.step> [min_pts=300] [max_dist=10]
用来判断"参考模型和点云对不齐"是哪些面、偏多少。
"""
import sys, os
import numpy as np
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
from read_any_pcd import read_xyz
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh


def _st(cls, name):
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def _cast(kind, s):
    return (getattr(TopoDS, kind + "_s", None) or getattr(TopoDS, kind))(s)


def explore(shape, kind, cast):
    ex = TopExp_Explorer(shape, kind); out = []
    while ex.More():
        out.append(_cast(cast, ex.Current())); ex.Next()
    return out


cloud, step = sys.argv[2], sys.argv[3]
min_pts = int(sys.argv[4]) if len(sys.argv) > 4 else 300
max_d = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
r = STEPControl_Reader()
if r.ReadFile(step) != IFSelect_RetDone:
    raise SystemExit("read fail " + step)
r.TransferRoots()
shape = r.OneShape()
rng = np.random.default_rng(0)
P, N, F, info = [], [], [], []
fid = 0
for ui, u in enumerate(explore(shape, TopAbs_SOLID, "Solid")):
    BRepMesh_IncrementalMesh(u, 0.3, False, 0.3, True)
    for f in explore(u, TopAbs_FACE, "Face"):
        loc = TopLoc_Location()
        tri = _st(BRep_Tool, "Triangulation")(f, loc)
        if tri is None:
            continue
        tr = loc.Transformation()
        nodes = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
        tris = np.array([[t.Value(1), t.Value(2), t.Value(3)] for t in (tri.Triangle(i) for i in range(1, tri.NbTriangles() + 1))]) - 1
        a, b, c = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
        cr = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(cr, axis=1)
        nrm = cr / np.maximum(2 * area[:, None], 1e-12)
        if f.Orientation() == TopAbs_REVERSED:
            nrm = -nrm
        k = np.floor(area / 1.0 + rng.random(len(area))).astype(int)
        if k.sum() == 0:
            continue
        idx = np.repeat(np.arange(len(area)), k)
        r1, r2 = rng.random(len(idx)), rng.random(len(idx)); s = np.sqrt(r1)
        pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
        nm = (nrm * area[:, None]).sum(0); nm /= max(np.linalg.norm(nm), 1e-12)
        P.append(pts); N.append(nrm[idx]); F.append(np.full(len(idx), fid))
        info.append((ui, fid, nm, area.sum()))
        fid += 1
P, N, F = np.vstack(P), np.vstack(N), np.concatenate(F)
X = read_xyz(cloud); X = X[np.isfinite(X).all(1)]
if len(X) > 400000:
    X = X[rng.choice(len(X), 400000, replace=False)]
d, j = cKDTree(P).query(X, distance_upper_bound=max_d)
ok = np.isfinite(d)
sd = np.einsum("ij,ij->i", X[ok] - P[j[ok]], N[j[ok]])
ff = F[j[ok]]
print("cloud pts used %d / %d within %.0f mm" % (ok.sum(), len(X), max_d))
print("unit face  normal(x,y,z)          area_cm2   n_pts  med_sd   p25    p75   |sd|<1  ")
rows = []
for ui, k, nm, ar in info:
    m = ff == k
    if m.sum() < min_pts:
        continue
    v = sd[m]
    rows.append((ui, k, nm, ar, m.sum(), np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.mean(np.abs(v) < 1)))
for ui, k, nm, ar, n, md, p25, p75, w1 in rows:
    print("%4d %4d  [%6.3f %6.3f %6.3f]  %9.1f %7d  %6.2f %6.2f %6.2f  %5.2f" % (ui, k, nm[0], nm[1], nm[2], ar / 100, n, md, p25, p75, w1))
up = [r for r in rows if r[2][2] > 0.9]
side = [r for r in rows if abs(r[2][2]) < 0.3]
for name, g in (("up faces", up), ("side faces", side)):
    if g:
        w = np.array([r[4] for r in g], float); md = np.array([r[5] for r in g])
        print("%-10s n_faces %3d  weighted median-of-medians %.2f  range [%.2f, %.2f]" % (name, len(g), np.average(md, weights=w), md.min(), md.max()))
