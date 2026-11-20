"""参考 STEP 逐实体检查：尺寸(OBB)、面数、面类型；BSpline 面是否其实是平面(拟合平面残差)。
用法: python -I inspect_ref.py <OCP_deps> <out.json> <step> [<step> ...]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, sys.argv[1])
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_OBB
from OCP.BRepBndLib import BRepBndLib

S = lambda c, n: getattr(c, n + "_s", None) or getattr(c, n)
cast = lambda k, s: (getattr(TopoDS, k + "_s", None) or getattr(TopoDS, k))(s)


def items(shape, kind, k):
    ex = TopExp_Explorer(shape, kind); out = []
    while ex.More():
        out.append(cast(k, ex.Current())); ex.Next()
    return out


res = {}
for path in sys.argv[3:]:
    r = STEPControl_Reader(); r.ReadFile(path); r.TransferRoots(); shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.2, False, 0.2, True)
    solids = []
    for si, so in enumerate(items(shape, TopAbs_SOLID, "Solid")):
        obb = Bnd_OBB(); S(BRepBndLib, "AddOBB")(so, obb, True, True, False)
        dims = sorted([2 * obb.XHSize(), 2 * obb.YHSize(), 2 * obb.ZHSize()])
        ax = [obb.XDirection(), obb.YDirection(), obb.ZDirection()]
        hs = [obb.XHSize(), obb.YHSize(), obb.ZHSize()]
        thin = ax[int(np.argmin(hs))]
        gp = GProp_GProps(); S(BRepGProp, "VolumeProperties")(so, gp)
        types, bs = {}, []
        for f in items(so, TopAbs_FACE, "Face"):
            t = str(BRepAdaptor_Surface(f).GetType()).split(".")[-1].replace("GeomAbs_", "")
            types[t] = types.get(t, 0) + 1
            if t == "BSplineSurface":
                loc = TopLoc_Location(); tri = S(BRep_Tool, "Triangulation")(f, loc)
                if tri is None:
                    continue
                tr = loc.Transformation()
                P = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
                c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T))
                dev = np.abs((P - c) @ v[:, 0])
                ga = GProp_GProps(); S(BRepGProp, "SurfaceProperties")(f, ga)
                bs.append(dict(area_cm2=round(ga.Mass() / 100, 1), nodes=len(P), max_dev=round(float(dev.max()), 3),
                               normal_z=round(float(abs(v[2, 0])), 3)))
        solids.append(dict(i=si, dims=[round(d, 1) for d in dims], thin_axis_z=round(abs(thin.Z()), 2),
                           faces=sum(types.values()), types=types, vol_cm3=round(abs(gp.Mass()) / 1000, 1), bspline=bs))
    res[os.path.basename(path)] = solids
    print("==", os.path.basename(path), "solids", len(solids))
    for s in solids:
        b = s["bspline"]
        bsum = "" if not b else " | BSpline %d: max_dev max=%.3f  normal_z=%s" % (
            len(b), max(x["max_dev"] for x in b), sorted({x["normal_z"] for x in b})[:4])
        print("  #%-2d dims=%-24s thinZ=%.2f faces=%-3d %s%s" % (s["i"], s["dims"], s["thin_axis_z"], s["faces"],
                                                                  {k: v for k, v in s["types"].items()}, bsum))
json.dump(res, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
