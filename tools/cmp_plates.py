"""对比两个 STEP 中长板（主面）的位置：法向、两主面偏移、沿轴范围、高度范围。
用法: python -I cmp_plates.py <OCP_deps> <a.step> <b.step> [min_len]
"""
import sys
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


def plates(path, min_len):
    r = STEPControl_Reader(); r.ReadFile(path); r.TransferRoots(); shape = r.OneShape(); out = []
    for i, so in enumerate(items(shape, TopAbs_SOLID, "Solid")):
        fs = []
        for f in items(so, TopAbs_FACE, "Face"):
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() != GeomAbs_Plane:
                continue
            pl = ad.Plane(); ax = pl.Axis().Direction(); o = pl.Location()
            n = np.array([ax.X(), ax.Y(), ax.Z()]); gp = GProp_GProps(); S(BRepGProp, "SurfaceProperties")(f, gp)
            fs.append((gp.Mass(), n, n @ np.array([o.X(), o.Y(), o.Z()])))
        if not fs:
            continue
        a, n, d = max(fs, key=lambda x: x[0])
        if abs(n[2]) > 0.5:
            continue
        n = n * np.sign(n[0] if abs(n[0]) > abs(n[1]) else n[1])
        offs = sorted(dd * np.sign(nn @ n) for aa, nn, dd in fs if abs(abs(nn @ n) - 1) < 1e-3)
        V = np.array([[p.X(), p.Y(), p.Z()] for p in (S(BRep_Tool, "Pnt")(v) for v in items(so, TopAbs_VERTEX, "Vertex"))])
        ax_ = np.cross([0, 0, 1.0], n); ax_ /= np.linalg.norm(ax_)
        s = V @ ax_
        if np.ptp(s) < min_len:
            continue
        out.append(dict(i=i, n=n, off=(offs[0], offs[-1]), s=(s.min(), s.max()), z=(V[:, 2].min(), V[:, 2].max()), ax=ax_,
                        mid=V.mean(0)))
    return out


def local(p, ref_pt):
    """平面常数换到参考点处（避免离原点很远时微小夹角把偏差放大）。"""
    return tuple(o - p["n"] @ ref_pt for o in p["off"])


A = plates(sys.argv[2], float(sys.argv[4]) if len(sys.argv) > 4 else 500)
B = plates(sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 500)
for a in A:
    ao = local(a, a["mid"])
    print("A#%d n=%s t=%.1f s=[%.0f, %.0f] z=[%.1f, %.1f]" % (a["i"], np.round(a["n"], 4), ao[1] - ao[0], *a["s"], *a["z"]))
    for b in B:
        if abs(abs(a["n"] @ b["n"]) - 1) > 0.01:
            continue
        sg = np.sign(a["n"] @ b["n"]); bo = sorted(o * sg for o in local(b, a["mid"]))
        if abs(np.mean(bo) - np.mean(ao)) > 40:
            continue
        ang = np.degrees(np.arccos(min(1, abs(a["n"] @ b["n"]))))
        print("   B#%d t=%.1f  face shift lo=%+.1f hi=%+.1f (at A mid)  angle=%.3f deg  s=[%.0f, %.0f] z=[%.1f, %.1f]" % (
            b["i"], bo[1] - bo[0], bo[0] - ao[0], bo[1] - ao[1], ang, *b["s"], *b["z"]))
