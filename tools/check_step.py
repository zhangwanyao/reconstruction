"""STEP 实体检查：数实体、逐个 BRepCheck_Analyzer、体积 > 0、面数。
用法: python -I check_step.py <model.step> --ocp <OCP 目录> [--json out.json]
退出码: 0 = 读得进且全部实体有效; 1 = 否则。
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("step")
ap.add_argument("--ocp", default=os.environ.get("OCP_DEPS", ""))
ap.add_argument("--json")
args = ap.parse_args()
if args.ocp:
    sys.path.insert(0, args.ocp)
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp


def S(cls, name):
    return getattr(cls, name + "_s", None) or getattr(cls, name)


res = dict(file=args.step, read_ok=False, solids=0, valid=0, faces=0, bad=[], volumes=[])
r = STEPControl_Reader()
if os.path.exists(args.step) and r.ReadFile(args.step) == IFSelect_RetDone:
    res["read_ok"] = True
    r.TransferRoots()
    shp = r.OneShape()
    ex = TopExp_Explorer(shp, TopAbs_SOLID)
    k = 0
    while ex.More():
        so = S(TopoDS, "Solid")(ex.Current())
        gp = GProp_GProps()
        S(BRepGProp, "VolumeProperties")(so, gp)
        nf = 0
        fx = TopExp_Explorer(so, TopAbs_FACE)
        while fx.More():
            nf += 1
            fx.Next()
        good = bool(BRepCheck_Analyzer(so).IsValid()) and gp.Mass() > 1e-6
        res["solids"] += 1
        res["faces"] += nf
        res["volumes"].append(round(gp.Mass(), 3))
        if good:
            res["valid"] += 1
        else:
            res["bad"].append(dict(i=k, volume=round(gp.Mass(), 3), faces=nf))
        k += 1
        ex.Next()
print(json.dumps(res, ensure_ascii=False))
if args.json:
    json.dump(res, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.exit(0 if res["read_ok"] and res["solids"] > 0 and res["solids"] == res["valid"] else 1)
