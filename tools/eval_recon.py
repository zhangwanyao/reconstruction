"""逆向建模结果评估：点云 / 参考 STEP / 输出 STEP 三方对比。

用法:
  python -I eval_recon.py --cloud X.pcd --ref ref.step [--out out.step] [--json result.json]
  只给 --ref：统计参考模型，并算 点云<->参考 的偏差（基线）。
  再给 --out：统计输出模型，算 点云<->输出，以及 输出<->参考 的逐实体召回/精度。
依赖：numpy、scipy、OCP（可用 --ocp 指定 OCP 所在目录）。
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cloud", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out")
ap.add_argument("--json")
ap.add_argument("--ocp", default=os.environ.get("OCP_DEPS", ""))
ap.add_argument("--tau", type=float, default=3.0, help="判定'重合'的距离阈值 mm")
ap.add_argument("--cloud-voxel", type=float, default=2.0)
args = ap.parse_args()
if args.ocp:
    sys.path.insert(0, args.ocp)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.spatial import cKDTree
from read_any_pcd import read_xyz
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_OBB, Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def _cast(kind, s):
    f = getattr(TopoDS, kind + "_s", None) or getattr(TopoDS, kind)
    return f(s)


def _static(cls, name):
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def load_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEP 读取失败: " + path)
    r.TransferRoots()
    return r.OneShape()


def units_of(shape):
    """评估单元：优先 SOLID；没有 SOLID 时退回 SHELL，再退回整体。"""
    for kind, cast in ((TopAbs_SOLID, "Solid"), (TopAbs_SHELL, "Shell")):
        ex = TopExp_Explorer(shape, kind); out = []
        while ex.More():
            out.append(_cast(cast, ex.Current())); ex.Next()
        if out:
            return out, cast
    return [shape], "Shape"


def faces_of(shape):
    ex = TopExp_Explorer(shape, TopAbs_FACE); out = []
    while ex.More():
        out.append(_cast("Face", ex.Current())); ex.Next()
    return out


def unit_stats(u):
    fs = faces_of(u)
    types = [str(BRepAdaptor_Surface(f).GetType()).split(".")[-1].replace("GeomAbs_", "") for f in fs]
    planar = sum(t == "Plane" for t in types)
    gp = GProp_GProps(); _static(BRepGProp, "VolumeProperties")(u, gp)
    obb = Bnd_OBB(); _static(BRepBndLib, "AddOBB")(u, obb, True, True, False)
    dims = sorted([2 * obb.XHSize(), 2 * obb.YHSize(), 2 * obb.ZHSize()])
    return dict(faces=len(fs), planar=planar, types={t: types.count(t) for t in set(types)},
                volume=abs(gp.Mass()), obb=[round(d, 1) for d in dims])


def sample_shape(units, spacing, rng):
    """按面积均匀采样表面点，返回 (pts, normals, unit_id, plane_n, plane_d)。平面面给出精确平面。"""
    P, N, U, PN, PD = [], [], [], [], []
    for ui, u in enumerate(units):
        BRepMesh_IncrementalMesh(u, 0.3, False, 0.3, True)
        for f in faces_of(u):
            loc = TopLoc_Location()
            tri = _static(BRep_Tool, "Triangulation")(f, loc)
            if tri is None:
                continue
            tr = loc.Transformation()
            nodes = np.array([[p.X(), p.Y(), p.Z()] for p in
                              (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
            tris = np.array([[t.Value(1), t.Value(2), t.Value(3)] for t in
                             (tri.Triangle(i) for i in range(1, tri.NbTriangles() + 1))]) - 1
            a, b, c = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
            cr = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(cr, axis=1)
            nrm = cr / np.maximum(2 * area[:, None], 1e-12)
            if f.Orientation() == TopAbs_REVERSED:
                nrm = -nrm
            k = np.floor(area / spacing ** 2 + rng.random(len(area))).astype(int)
            if k.sum() == 0:
                continue
            idx = np.repeat(np.arange(len(area)), k)
            r1, r2 = rng.random(len(idx)), rng.random(len(idx))
            s = np.sqrt(r1)
            pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() == GeomAbs_Plane:
                pl = ad.Plane(); d = pl.Axis().Direction(); o = pl.Location()
                pn = np.array([d.X(), d.Y(), d.Z()]); pd = pn @ np.array([o.X(), o.Y(), o.Z()])
            else:
                pn = np.zeros(3); pd = np.nan
            P.append(pts); N.append(nrm[idx]); U.append(np.full(len(idx), ui))
            PN.append(np.tile(pn, (len(idx), 1))); PD.append(np.full(len(idx), pd))
    if not P:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0, int), z, np.zeros(0)
    return np.vstack(P), np.vstack(N), np.concatenate(U), np.vstack(PN), np.concatenate(PD)


def total_area(units):
    a = 0.0
    for u in units:
        gp = GProp_GProps(); _static(BRepGProp, "SurfaceProperties")(u, gp); a += gp.Mass()
    return a


def bbox(shape):
    b = Bnd_Box(); _static(BRepBndLib, "Add")(shape, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return [round(v, 1) for v in (lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z())]


def cloud_to_model(cloud, S, spacing):
    """点 -> 模型 距离：先找最近采样点，落在平面面附近时改用到该平面的精确距离。"""
    pts, _, _, pn, pd = S
    d, i = cKDTree(pts).query(cloud, workers=-1)
    planar = ~np.isnan(pd[i])
    exact = np.abs(np.einsum("ij,ij->i", cloud, pn[i]) - pd[i])
    use = planar & (d < 2 * spacing)
    d = d.copy(); d[use] = np.minimum(d[use], exact[use])
    return d


def pct(x, q):
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def model_report(path, cloud, ctree, spacing_hint, rng):
    shape = load_step(path)
    units, kind = units_of(shape)
    stats = [unit_stats(u) for u in units]
    area = total_area(units)
    spacing = max(spacing_hint, float(np.sqrt(area / 2.5e6)))
    S = sample_shape(units, spacing, rng)
    d_cm = cloud_to_model(cloud, S, spacing)
    d_mc, _ = ctree.query(S[0], workers=-1)
    cov = d_mc < args.tau
    up = S[1][:, 2] > 0.5
    side = np.abs(S[1][:, 2]) <= 0.5
    per_unit_cov = [round(float(cov[S[2] == k].mean()), 2) if (S[2] == k).any() else 0.0 for k in range(len(units))]
    rep = dict(
        file=path, unit_kind=kind, units=len(units), faces=sum(s["faces"] for s in stats),
        planar_faces=sum(s["planar"] for s in stats),
        face_types={t: sum(s["types"].get(t, 0) for s in stats) for t in set().union(*[s["types"] for s in stats])},
        area_m2=round(area / 1e6, 3),
        volume_dm3=round(sum(s["volume"] for s in stats) / 1e6, 3), bbox=bbox(shape),
        thickness_mm=sorted([s["obb"][0] for s in stats]), sample_spacing=round(spacing, 2),
        cloud_to_model=dict(p50=pct(d_cm, 50), p90=pct(d_cm, 90), p95=pct(d_cm, 95),
                            within_1mm=round(float((d_cm < 1).mean()), 3), within_3mm=round(float((d_cm < 3).mean()), 3),
                            within_10mm=round(float((d_cm < 10).mean()), 3)),
        model_covered_by_cloud=dict(all=round(float(cov.mean()), 3),
                                    up_faces=round(float(cov[up].mean()), 3) if up.any() else None,
                                    side_faces=round(float(cov[side].mean()), 3) if side.any() else None),
        per_unit=[dict(i=k, **stats[k], covered=per_unit_cov[k]) for k in range(len(units))])
    return rep, S


def compare(out_S, ref_S, n_out, n_ref, tau):
    po, pr = out_S[0], ref_S[0]
    if len(po) == 0 or len(pr) == 0:
        return dict(error="empty model")
    d_ro, _ = cKDTree(po).query(pr, workers=-1)
    d_or, _ = cKDTree(pr).query(po, workers=-1)
    rec_u = [float((d_ro[ref_S[2] == k] < tau).mean()) if (ref_S[2] == k).any() else 0.0 for k in range(n_ref)]
    pre_u = [float((d_or[out_S[2] == k] < tau).mean()) if (out_S[2] == k).any() else 0.0 for k in range(n_out)]
    return dict(
        tau_mm=tau,
        surface_recall=round(float((d_ro < tau).mean()), 3),
        surface_precision=round(float((d_or < tau).mean()), 3),
        ref_to_out=dict(p50=pct(d_ro, 50), p95=pct(d_ro, 95)),
        out_to_ref=dict(p50=pct(d_or, 50), p95=pct(d_or, 95)),
        ref_units_found=sum(r >= 0.5 for r in rec_u), ref_units=n_ref,
        ref_units_missing=[k for k, r in enumerate(rec_u) if r < 0.5],
        out_units_valid=sum(p >= 0.5 for p in pre_u), out_units=n_out,
        out_units_extra=[k for k, p in enumerate(pre_u) if p < 0.5],
        per_ref_recall=[round(r, 2) for r in rec_u], per_out_precision=[round(p, 2) for p in pre_u])


t0 = time.time()
rng = np.random.default_rng(0)
raw = read_xyz(args.cloud)
k = np.floor((raw - raw.min(0)) / args.cloud_voxel).astype(np.int64)
_, first = np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2], return_index=True)
cloud = raw[first]
ctree = cKDTree(cloud)
res = dict(cloud=args.cloud, cloud_points=len(raw), cloud_eval_points=len(cloud),
           cloud_bbox=[round(float(v), 1) for v in np.r_[raw.min(0), raw.max(0)]])
res["ref"], ref_S = model_report(args.ref, cloud, ctree, 1.5, rng)
if args.out:
    res["out"], out_S = model_report(args.out, cloud, ctree, res["ref"]["sample_spacing"], rng)
    res["out_vs_ref"] = compare(out_S, ref_S, res["out"]["units"], res["ref"]["units"], args.tau)
res["seconds"] = round(time.time() - t0, 1)
if args.json:
    json.dump(res, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
short = {k: v for k, v in res.items() if k not in ("ref", "out")}
for key in ("ref", "out"):
    if key in res:
        short[key] = {k: v for k, v in res[key].items() if k not in ("per_unit", "thickness_mm", "file")}
if "out_vs_ref" in short:
    short["out_vs_ref"] = {k: v for k, v in short["out_vs_ref"].items() if not k.startswith("per_")}
print(json.dumps(short, ensure_ascii=False))
