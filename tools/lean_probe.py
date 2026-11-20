"""参考 STEP 里每块竖向/倾斜板: 点云在横截面(v=水平法向坐标, z)里的样子 + 每个大面上的点数和高度跨度。
用法: python -I lean_probe.py <OCP_deps> <pcd> <ref.step> <out_prefix>
输出: <out_prefix>_lean.json, <out_prefix>_u<i>.png (每块非水平板一张横截面图, ASCII 标注)
"""
import sys, os, json
import numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = lambda c, n: getattr(c, n + "_s", None) or getattr(c, n)
cast = lambda k, s: (getattr(TopoDS, k + "_s", None) or getattr(TopoDS, k))(s)


def items(shape, kind, k):
    ex = TopExp_Explorer(shape, kind); out = []
    while ex.More():
        out.append(cast(k, ex.Current())); ex.Next()
    return out


r = STEPControl_Reader(); r.ReadFile(sys.argv[3]); r.TransferRoots(); shape = r.OneShape()
P = read_xyz(sys.argv[2]); P = P[np.isfinite(P).all(1)]
res = []
for i, so in enumerate(items(shape, TopAbs_SOLID, "Solid")):
    fs = []
    for f in items(so, TopAbs_FACE, "Face"):
        ad = BRepAdaptor_Surface(f)
        if ad.GetType() != GeomAbs_Plane:
            continue
        pl = ad.Plane(); ax = pl.Axis().Direction(); o = pl.Location()
        n = np.array([ax.X(), ax.Y(), ax.Z()]); gp = GProp_GProps(); S(BRepGProp, "SurfaceProperties")(f, gp)
        fs.append(dict(a=gp.Mass(), n=n, d=float(n @ np.array([o.X(), o.Y(), o.Z()]))))
    if not fs:
        continue
    big = max(fs, key=lambda x: x["a"])
    n = big["n"]
    if abs(n[2]) > 0.8:
        continue                                     # 水平板跳过
    V = np.array([[p.X(), p.Y(), p.Z()] for p in (S(BRep_Tool, "Pnt")(v) for v in items(so, TopAbs_VERTEX, "Vertex"))])
    nh = n.copy(); nh[2] = 0; nh /= np.linalg.norm(nh)
    a = np.cross([0, 0, 1.0], nh); a /= np.linalg.norm(a)
    sV, vV = V @ a, V @ nh
    lean = float(np.degrees(np.arcsin(min(1, abs(n[2])))))
    s_, v_ = P @ a, P @ nh
    m = (s_ > sV.min() + 10) & (s_ < sV.max() - 10) & (v_ > vV.min() - 30) & (v_ < vV.max() + 30) & \
        (P[:, 2] > V[:, 2].min() - 5) & (P[:, 2] < V[:, 2].max() + 5)
    Q = P[m]
    par = [f for f in fs if abs(abs(f["n"] @ n) - 1) < 1e-3]
    faces = []
    for f in par:
        dist = np.abs(Q @ f["n"] - f["d"])
        on = dist < 1.5
        up = float(f["n"][2] * np.sign(f["n"] @ n))
        faces.append(dict(nz=round(float(f["n"][2]), 3), area_cm2=round(f["a"] / 100, 1), pts=int(on.sum()),
                          zspan=round(float(np.ptp(Q[on, 2])) if on.sum() > 2 else 0.0, 1)))
    t = abs(par[0]["d"] * np.sign(par[0]["n"] @ n) - par[-1]["d"] * np.sign(par[-1]["n"] @ n)) if len(par) >= 2 else None
    item = dict(i=i, lean_deg=round(lean, 2), t=None if t is None else round(float(t), 2), L=round(float(np.ptp(sV)), 1),
                z=[round(float(V[:, 2].min()), 1), round(float(V[:, 2].max()), 1)], band_pts=int(m.sum()), faces=faces)
    res.append(item)
    print(json.dumps(item))
    fig, ax = plt.subplots(figsize=(5, 6))
    k = np.random.default_rng(0).choice(len(Q), min(len(Q), 40000), replace=False) if len(Q) else []
    ax.scatter(v_[m][k], Q[k, 2], s=0.3, c="k")
    zz = np.array([V[:, 2].min(), V[:, 2].max()])
    for f in par:
        nn = f["n"]; sg = np.hypot(nn[0], nn[1]) * np.sign(nn @ nh)
        ax.plot((f["d"] - nn[2] * zz - (nn[:2] @ a[:2]) * 0) / sg if abs(sg) > 1e-6 else zz * 0, zz, "r-", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("v (mm, horizontal normal)"); ax.set_ylabel("z (mm)")
    ax.set_title("unit %d lean %.1f deg t=%s" % (i, lean, item["t"]))
    fig.savefig("%s_u%d.png" % (sys.argv[4], i), dpi=110, bbox_inches="tight"); plt.close(fig)
json.dump(res, open(sys.argv[4] + "_lean.json", "w"), indent=1)
