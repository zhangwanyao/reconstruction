"""参考 STEP 逐实体分类 + 俯视图（点云高度图打底，实体着色编号）。
用法: python -I ref_detail.py <OCP_deps> <cloud.pcd> <ref.step> <eval_ref.json或-> <out_prefix>
输出: <out_prefix>_parts.json, <out_prefix>_top.png
"""
import sys, os, json
import numpy as np
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from read_any_pcd import read_xyz
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_WIRE
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_OBB, Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepTools import BRepTools

S = lambda c, n: getattr(c, n + "_s", None) or getattr(c, n)
cast = lambda k, s: (getattr(TopoDS, k + "_s", None) or getattr(TopoDS, k))(s)


def items(shape, kind, k):
    ex = TopExp_Explorer(shape, kind); out = []
    while ex.More():
        out.append(cast(k, ex.Current())); ex.Next()
    return out


ocp, cloud_p, step_p, evj, outp = sys.argv[1:6]
r = STEPControl_Reader(); r.ReadFile(step_p); r.TransferRoots(); shape = r.OneShape()
BRepMesh_IncrementalMesh(shape, 0.3, False, 0.3, True)
cov = None
if evj != "-" and os.path.exists(evj):
    cov = [u["covered"] for u in json.load(open(evj, encoding="utf-8"))["ref"]["per_unit"]]

parts = []
tris_all = []
for si, so in enumerate(items(shape, TopAbs_SOLID, "Solid")):
    obb = Bnd_OBB(); S(BRepBndLib, "AddOBB")(so, obb, True, True, False)
    ax = [obb.XDirection(), obb.YDirection(), obb.ZDirection()]
    hs = np.array([obb.XHSize(), obb.YHSize(), obb.ZHSize()])
    order = np.argsort(hs)
    thin = ax[order[0]]; longd = ax[order[2]]
    bb = Bnd_Box(); S(BRepBndLib, "Add")(so, bb)
    lo, hi = bb.CornerMin(), bb.CornerMax()
    gp = GProp_GProps(); S(BRepGProp, "VolumeProperties")(so, gp)
    faces = items(so, TopAbs_FACE, "Face")
    inner_loops = 0
    for f in faces:
        nw = len(items(f, TopAbs_WIRE, "Wire"))
        inner_loops += max(0, nw - 1)
    tz = abs(thin.Z())
    kind = "H" if tz > 0.9 else ("V" if tz < 0.15 else "S")
    t, w, L = (2 * hs[order]).tolist()
    if gp.Mass() < 1.0:
        kind = "degenerate"
    P = []
    for f in faces:
        loc = TopLoc_Location(); tri = S(BRep_Tool, "Triangulation")(f, loc)
        if tri is None:
            continue
        tr = loc.Transformation()
        nodes = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
        tt = np.array([[q.Value(1), q.Value(2), q.Value(3)] for q in (tri.Triangle(i) for i in range(1, tri.NbTriangles() + 1))]) - 1
        P.append(nodes[tt])
    tris_all.append(np.concatenate(P) if P else np.zeros((0, 3, 3)))
    parts.append(dict(i=si, kind=kind, t=round(t, 1), w=round(w, 1), L=round(L, 1), faces=len(faces), inner_loops=inner_loops,
                      vol_cm3=round(abs(gp.Mass()) / 1000, 1), z=[round(lo.Z(), 1), round(hi.Z(), 1)],
                      xy=[round(lo.X(), 1), round(lo.Y(), 1), round(hi.X(), 1), round(hi.Y(), 1)],
                      long_dir=[round(longd.X(), 3), round(longd.Y(), 3), round(longd.Z(), 3)],
                      covered=(cov[si] if cov else None)))

# base = 最低的大 H 件
Hs = [p for p in parts if p["kind"] == "H"]
base = None
if Hs:
    big = max(p["w"] * p["L"] for p in Hs)
    cand = [p for p in Hs if p["w"] * p["L"] > 0.3 * big]
    base = min(cand, key=lambda p: p["z"][0])
    lowest_other = min([p["z"][0] for p in parts if p is not base and p["kind"] != "degenerate"] or [1e9])
    if lowest_other < base["z"][0] - 5:
        base = None
top = base["z"][1] if base else None
for p in parts:
    p["role"] = ("base" if p is base else
                 ("vertical_plate" if p["kind"] == "V" else
                  ("horizontal_plate" if p["kind"] == "H" else
                   ("slanted" if p["kind"] == "S" else "degenerate"))))
    if p["role"] == "vertical_plate" and p["t"] > 25:
        p["role"] = "block"
    if top is not None and p is not base:
        p["above_base"] = [round(p["z"][0] - top, 1), round(p["z"][1] - top, 1)]
json.dump(dict(base_top=top, parts=parts), open(outp + "_parts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- 俯视图 ----
pts = read_xyz(cloud_p)
allt = np.concatenate([t.reshape(-1, 3) for t in tris_all if len(t)])
x0, y0 = min(pts[:, 0].min(), allt[:, 0].min()) - 20, min(pts[:, 1].min(), allt[:, 1].min()) - 20
x1, y1 = max(pts[:, 0].max(), allt[:, 0].max()) + 20, max(pts[:, 1].max(), allt[:, 1].max()) + 20
long_x = (x1 - x0) >= (y1 - y0)
span_long = (x1 - x0) if long_x else (y1 - y0)
span_short = (y1 - y0) if long_x else (x1 - x0)
chunks = int(np.ceil(span_long / max(span_short * 3.0, 1400.0)))
chunk_len = span_long / chunks
scale = min(1500.0 / chunk_len, 2.0)


def to_uv(xy):
    a = (xy[:, 0] - x0) if long_x else (xy[:, 1] - y0)
    b = (y1 - xy[:, 1]) if long_x else (xy[:, 0] - x0)
    return a, b


W = int(chunk_len * scale) + 1
Hh = int(span_short * scale) + 1
canvas = np.full((chunks * (Hh + 30), W, 3), 255, np.uint8)
a, b = to_uv(pts[:, :2])
z = pts[:, 2]
zn = np.clip((z - np.percentile(z, 1)) / max(np.percentile(z, 99) - np.percentile(z, 1), 1e-6), 0, 1)
ci = np.minimum((a / chunk_len).astype(int), chunks - 1)
u = ((a - ci * chunk_len) * scale).astype(int).clip(0, W - 1)
v = (b * scale).astype(int).clip(0, Hh - 1) + ci * (Hh + 30) + 30
g = (230 - 200 * zn).astype(np.uint8)
order = np.argsort(z)
canvas[v[order], u[order]] = np.stack([g[order]] * 3, 1)
rng = np.random.default_rng(3)
cols = [tuple(int(c) for c in rng.integers(40, 230, 3)) for _ in parts]
overlay = canvas.copy()
labels = []
for p, T in zip(parts, tris_all):
    if len(T) == 0 or p["role"] == "base":
        continue
    flat = T.reshape(-1, 3)
    aa, bb_ = to_uv(flat[:, :2])
    for c in range(chunks):
        lo_, hi_ = c * chunk_len, (c + 1) * chunk_len
        if aa.max() < lo_ or aa.min() > hi_:
            continue
        uu = ((aa - lo_) * scale).reshape(-1, 3)
        vv = (bb_ * scale + c * (Hh + 30) + 30).reshape(-1, 3)
        polys = np.stack([uu, vv], -1).astype(np.int32)
        cv2.fillPoly(overlay, list(polys), cols[p["i"]])
        m = (aa >= lo_) & (aa <= hi_)
        if not m.any():
            continue
        labels.append((int(np.mean((aa[m] - lo_) * scale)), int(np.mean(bb_[m] * scale + c * (Hh + 30) + 30)), p))
canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
if base is not None:
    T = tris_all[base["i"]].reshape(-1, 3)
    aa, bb_ = to_uv(T[:, :2])
    for c in range(chunks):
        lo_ = c * chunk_len
        uu = ((aa - lo_) * scale).reshape(-1, 3); vv = (bb_ * scale + c * (Hh + 30) + 30).reshape(-1, 3)
        for tri in np.stack([uu, vv], -1).astype(np.int32):
            pass
        mask = np.zeros(canvas.shape[:2], np.uint8)
        cv2.fillPoly(mask, list(np.stack([uu, vv], -1).astype(np.int32)), 255)
        cs, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cs, -1, (0, 160, 0), 1)
for (uu, vv, p) in labels:
    txt = "%d" % p["i"]
    cv2.putText(canvas, txt, (uu - 6, vv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(canvas, txt, (uu - 6, vv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
for c in range(chunks):
    cv2.putText(canvas, "segment %d/%d  %.0f-%.0f mm  scale %.2f px/mm  green=base outline" % (c + 1, chunks, c * chunk_len, (c + 1) * chunk_len, scale),
                (5, c * (Hh + 30) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
cv2.imwrite(outp + "_top.png", canvas)
print("base_top", top)
for p in parts:
    print("#%-2d %-16s t=%-5s w=%-6s L=%-7s faces=%-3d holes=%-2d z=%s above=%s cov=%s" % (
        p["i"], p["role"], p["t"], p["w"], p["L"], p["faces"], p["inner_loops"], p["z"], p.get("above_base"), p["covered"]))
