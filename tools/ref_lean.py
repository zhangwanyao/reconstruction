"""参考 STEP 每个实体：主面（面积最大的平面）法向 -> 相对底板法向的倾角、厚度、顶边是否带坡。
用法: python -I ref_lean.py <OCP_deps> <ref.step> [<ref.step> ...]
"""
import sys, os
import numpy as np
sys.path.insert(0, sys.argv[1])
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

S = lambda c, n: getattr(c, n + "_s", None) or getattr(c, n)
cast = lambda k, s: (getattr(TopoDS, k + "_s", None) or getattr(TopoDS, k))(s)


def items(shape, kind, k):
    ex = TopExp_Explorer(shape, kind); out = []
    while ex.More():
        out.append(cast(k, ex.Current())); ex.Next()
    return out


def solid_info(so):
    faces = []
    for f in items(so, TopAbs_FACE, "Face"):
        ad = BRepAdaptor_Surface(f)
        if ad.GetType() != GeomAbs_Plane:
            continue
        pl = ad.Plane(); ax = pl.Axis().Direction(); o = pl.Location()
        n = np.array([ax.X(), ax.Y(), ax.Z()]); p0 = np.array([o.X(), o.Y(), o.Z()])
        gp = GProp_GProps(); S(BRepGProp, "SurfaceProperties")(f, gp)
        faces.append((gp.Mass(), n, float(n @ p0)))
    V = np.array([[p.X(), p.Y(), p.Z()] for p in (S(BRep_Tool, "Pnt")(v) for v in items(so, TopAbs_VERTEX, "Vertex"))])
    gp = GProp_GProps(); S(BRepGProp, "VolumeProperties")(so, gp)
    return faces, V, abs(gp.Mass())


for step in sys.argv[2:]:
    r = STEPControl_Reader(); r.ReadFile(step); r.TransferRoots(); shape = r.OneShape()
    sols = [solid_info(s) for s in items(shape, TopAbs_SOLID, "Solid")]
    # 底板法向：面积最大的近水平面
    big = max(((a, n) for fs, _, _ in sols for a, n, _ in fs if abs(n[2]) > 0.95), key=lambda x: x[0], default=(0, np.array([0, 0, 1.0])))
    zb = big[1] * np.sign(big[1][2])
    print("==", os.path.basename(step), "base normal", np.round(zb, 4), "tilt %.2f deg" % np.degrees(np.arccos(min(1, zb[2]))))
    for i, (fs, V, vol) in enumerate(sols):
        if not fs or vol < 1:
            print("#%d degenerate" % i); continue
        a, n, d = max(fs, key=lambda x: x[0])
        par = [(aa, dd * np.sign(nn @ n)) for aa, nn, dd in fs if abs(abs(nn @ n) - 1) < 1e-3]
        t = max(dd for _, dd in par) - min(dd for _, dd in par)
        lean = np.degrees(np.arcsin(min(1.0, abs(n @ zb))))
        tilt_from_vertical = lean          # 主面法向与底板法向的夹角余角：0=竖板，90=平板
        kind = "H" if tilt_from_vertical > 60 else ("V" if tilt_from_vertical < 2 else "SLANT")
        # 顶边坡度：沿主面内水平方向，顶部顶点高度的变化
        h = V @ zb
        dirs = np.cross(n, zb); dirs = dirs / (np.linalg.norm(dirs) + 1e-12)
        s = V @ dirs
        top_pts = sorted(set((round(float(si), 1), round(float(hi), 1)) for si, hi in zip(s, h) if hi > h.max() - 0.5 * (h.max() - h.min())))
        print("#%-2d %-5s plate_angle_from_vertical=%5.1f  t=%5.1f  faces=%2d  vol=%7.0fcm3  h=[%.1f, %.1f]  L_h=%.0f%s" % (
            i, kind, tilt_from_vertical, t, len(fs), vol / 1000, h.min(), h.max(), np.ptp(s),
            ("  top_pts(s,h)=" + str(top_pts[:6])) if kind != "H" and len(top_pts) <= 8 else ""))
