"""诊断 STEP 里无效实体的端面多边形: 顶点数、重复点、自交、最短边、最尖角。
用法: python -I diag_invalid.py <OCP_deps> <run_dir> [<run_dir> ...]
      run_dir 里要有 model.step、check_step.json、parts.json(实体顺序和 STEP 一致)。
"""
import json, os, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.BRep import BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer


def S(cls, name):
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def seg_cross(a, b, c, d):
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def on(p, q, r):
        return min(p[0], q[0]) - 1e-9 <= r[0] <= max(p[0], q[0]) + 1e-9 and min(p[1], q[1]) - 1e-9 <= r[1] <= max(p[1], q[1]) + 1e-9

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b))


def poly_report(P):
    n = len(P)
    dup = sum(1 for i in range(n) for j in range(i + 1, n) if np.hypot(*(P[i] - P[j])) < 1e-6)
    cross = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if seg_cross(P[i], P[(i + 1) % n], P[j], P[(j + 1) % n]):
                cross.append((i, j))
    e = [np.hypot(*(P[(i + 1) % n] - P[i])) for i in range(n)]
    ang = []
    for i in range(n):
        u, v = P[i - 1] - P[i], P[(i + 1) % n] - P[i]
        c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    area = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    return dict(n=n, dup=dup, cross=len(cross), cross_pairs=cross[:6], min_edge=round(min(e), 3), min_angle=round(min(ang), 2), area=round(area, 1))


for run in sys.argv[2:]:
    cj = json.load(open(os.path.join(run, "check_step.json"), encoding="utf-8"))
    pj = json.load(open(os.path.join(run, "parts.json"), encoding="utf-8"))
    bad = {b["i"] for b in cj["bad"]}
    r = STEPControl_Reader()
    r.ReadFile(os.path.join(run, "model.step"))
    r.TransferRoots()
    ex = TopExp_Explorer(r.OneShape(), TopAbs_SOLID)
    k = 0
    print("==", os.path.basename(run.rstrip("\\/")), "bad", sorted(bad))
    while ex.More():
        if k in bad:
            so = S(TopoDS, "Solid")(ex.Current())
            caps, nbadf = [], 0
            fx = TopExp_Explorer(so, TopAbs_FACE)
            while fx.More():
                f = S(TopoDS, "Face")(fx.Current())
                if not BRepCheck_Analyzer(f).IsValid():
                    nbadf += 1
                w = S(BRepTools, "OuterWire")(f)
                we = BRepTools_WireExplorer(w)
                pts = []
                while we.More():
                    p = BRep_Tool.Pnt_s(we.CurrentVertex()) if hasattr(BRep_Tool, "Pnt_s") else BRep_Tool.Pnt(we.CurrentVertex())
                    pts.append((p.X(), p.Y(), p.Z()))
                    we.Next()
                caps.append(np.array(pts))
                fx.Next()
            cap = max(caps, key=len)
            c0 = cap.mean(0)
            _, _, vt = np.linalg.svd(cap - c0)
            P2 = (cap - c0) @ vt[:2].T
            sd = pj["solids"][k] if k < len(pj["solids"]) else {}
            print(" solid", k, sd.get("kind"), sd.get("name"), "faces", len(caps), "bad_faces", nbadf, poly_report(P2))
        k += 1
        ex.Next()
