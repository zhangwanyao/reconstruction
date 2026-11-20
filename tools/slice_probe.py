"""切面探针：沿每个构件轴线做垂直切面，在每个切面里量 顶高 / 顶宽 / 侧面倾角 / 底边，
再把同一构件的切面串起来拟合 3D 侧面平面；并和"固定 Y 方向切"做对比。
用法: python -I slice_probe.py <pcd> <out_dir> [--mode auto|base|nobase] [--px 1] [--plots 8]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import detect_frame, heightmap, skeleton_segments, dp_split, fit_plane

ap = argparse.ArgumentParser()
ap.add_argument("pcd"); ap.add_argument("out")
ap.add_argument("--mode", default="auto", choices=["auto", "base", "nobase"])
ap.add_argument("--px", type=float, default=1.0)
ap.add_argument("--h-min", type=float, default=8.0)
ap.add_argument("--max-w", type=float, default=90.0)
ap.add_argument("--min-len", type=float, default=60.0)
ap.add_argument("--slab", type=float, default=2.0)
ap.add_argument("--margin", type=float, default=35.0)
ap.add_argument("--plots", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TM = {}; t0 = time.time()
P = read_xyz(args.pcd); P = P[np.isfinite(P).all(1)]
TM["read"] = round(time.time() - t0, 2); t1 = time.time()
F = detect_frame(P, args.mode)
Q, has_base = F["Q"], F["has_base"]
hmp = heightmap(Q, args.px, has_base)
hm, occ = hmp["hm"], hmp["occ"]
mask = ((hm > args.h_min) if has_base else occ).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
nl, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
small = st[:, cv2.CC_STAT_AREA] < max(20, int(150 / args.px ** 2))
mask[small[lab] & (lab > 0)] = 0
segs = [s for s in skeleton_segments(mask, hmp, min_len_mm=args.min_len) if s["w"] <= args.max_w]
TM["frame_layout"] = round(time.time() - t1, 2); t1 = time.time()
XY, H = Q[:, :2], Q[:, 2]
tree = cKDTree(XY)
zlo = 2.0 if has_base else -1e9
rng = np.random.default_rng(0)


def ransac_line(u, z, tol=1.0, iters=150, max_tan=np.tan(np.radians(40))):
    """侧面线 u = a + b*z（b = tan(倾角)）。"""
    n = len(u)
    if n < 6:
        return None
    ii = rng.integers(0, n, (iters, 2))
    best = None; bc = 0
    for i, j in ii:
        if abs(z[i] - z[j]) < 3:
            continue
        b = (u[i] - u[j]) / (z[i] - z[j])
        if abs(b) > max_tan:
            continue
        a = u[i] - b * z[i]
        inl = np.abs(u - (a + b * z)) < tol
        if inl.sum() > bc:
            best, bc = inl, inl.sum()
    if best is None or bc < 6 or np.ptp(z[best]) < 5:
        return None
    b, a = np.polyfit(z[best], u[best], 1)
    inl = np.abs(u - (a + b * z)) < tol
    return float(a), float(b), inl


def slice_member(sg):
    c, d = sg["c"], sg["d"]; nrm = np.array([-d[1], d[0]])
    s0, s1 = sg["s0"], sg["s1"]; L = s1 - s0
    mu = sg["w"] / 2 + args.margin
    mid = c + d * (s0 + s1) / 2
    idx = np.asarray(tree.query_ball_point(mid, float(np.hypot(L / 2 + 15, mu))))
    if len(idx) < 50:
        return None
    xy, z = XY[idx], H[idx]
    s = (xy - c) @ d; u = (xy - c) @ nrm
    sel = (np.abs(u) < mu) & (s > s0 - 5) & (s < s1 + 5) & (z > zlo)
    s, u, z = s[sel], u[sel], z[sel]
    step = float(np.clip(L / 30, 5, 20))
    rows, side = [], []
    for sc in np.arange(s0 + step / 2, s1, step):
        m = np.abs(s - sc) < args.slab
        r = dict(s=round(float(sc), 1), n=int(m.sum()))
        rows.append(r)
        if m.sum() < 12:
            continue
        uu, zz, ss = u[m], z[m], s[m]
        near = np.abs(uu) < sg["w"] / 2 + 6
        if near.sum() < 5:
            continue
        ztop = float(np.percentile(zz[near], 99))
        cap = near & (zz > ztop - 3)
        clo, chi = np.percentile(uu[cap], [3, 97]); uc = (clo + chi) / 2
        r.update(ztop=round(ztop, 2), uc=round(float(uc), 2), wcap=round(float(chi - clo), 2))
        sd = (zz < ztop - 4) & (np.abs(uu - uc) < (chi - clo) / 2 + 0.85 * np.maximum(ztop - zz, 0) + 4)
        res = ransac_line(uu[sd], zz[sd]) if sd.sum() >= 8 else None
        if res is None:
            continue
        a, b, inl = res
        zi = zz[sd][inl]
        r.update(a=round(a, 2), b=round(b, 4), lean=round(float(np.degrees(np.arctan(b))), 2), n_side=int(inl.sum()),
                 zlo=round(float(np.percentile(zi, 2)), 1), zhi=round(float(np.percentile(zi, 98)), 1))
        side.append(np.c_[ss[sd][inl], uu[sd][inl], zi])
        rest = ~inl
        if rest.sum() >= 8:
            r2 = ransac_line(uu[sd][rest], zz[sd][rest])
            if r2 is not None and abs(np.degrees(np.arctan(r2[1]) - np.arctan(b))) < 4 and abs(r2[0] - a) > 2:
                r["t2"] = round(float(abs(r2[0] - a) * np.cos(np.arctan(b))), 2)
    ok = [r for r in rows if "ztop" in r]
    if len(ok) < 3:
        return None
    S = np.array([r["s"] for r in ok]); Z = np.array([r["ztop"] for r in ok])
    prof = np.c_[S, Z]; kk = dp_split(prof, 3.0)
    leans = np.array([r["lean"] for r in ok if "lean" in r])
    info = dict(L=round(L, 1), w_top_view=round(sg["w"], 1), ang_x=round(float(np.degrees(np.arctan2(d[1], d[0]))) % 180, 1),
                slices=len(rows), slices_ok=len(ok), slices_side=int(len(leans)),
                ztop_p5_p95=[round(float(np.percentile(Z, 5)), 1), round(float(np.percentile(Z, 95)), 1)],
                top_poly=prof[kk].round(1).tolist(), wcap_med=round(float(np.median([r["wcap"] for r in ok])), 2),
                lean2d_med=round(float(np.median(leans)), 2) if len(leans) else None,
                t2_med=round(float(np.median([r["t2"] for r in ok if "t2" in r])), 2) if any("t2" in r for r in ok) else None)
    lean = info["lean2d_med"] or 0.0
    if side and sum(len(a) for a in side) >= 30:
        A = np.vstack(side); n, dd = fit_plane(A)
        if n[1] < 0:
            n, dd = -n, -dd
        info["lean3d"] = round(float(np.degrees(np.arcsin(min(1.0, abs(n[2]))))) * np.sign(-n[2] if n[1] > 0 else n[2]), 2)
        info["side_yaw_deg"] = round(float(np.degrees(np.arctan2(n[0], n[1]))), 2)
        info["side_rms"] = round(float(np.sqrt(np.mean((A @ n - dd) ** 2))), 2)
        info["side_pts"] = int(len(A))
        lean = abs(info["lean3d"])
    info["t_cap"] = round(info["wcap_med"] * np.cos(np.radians(lean)), 2)
    zl = [r["zlo"] for r in ok if "zlo" in r]
    info["bottom"] = 0.0 if has_base else (round(float(np.median(zl)), 1) if zl else None)
    info["side_reaches_bottom"] = round(float(np.mean(np.array(zl) < (6 if has_base else info["bottom"] + 6))), 2) if zl else 0.0
    return info, rows, (s, u, z)


results = []
for k, sg in enumerate(segs):
    out = slice_member(sg)
    if out is None:
        continue
    info, rows, pts = out
    info["id"] = len(results); info["c"] = [round(x, 1) for x in sg["c"]]; info["d"] = [round(x, 4) for x in sg["d"]]
    info["s0"], info["s1"] = round(sg["s0"], 1), round(sg["s1"], 1)
    results.append((info, rows, pts))
TM["slicing"] = round(time.time() - t1, 2); t1 = time.time()


def interest(info):
    tp = info["ztop_p5_p95"]
    return (abs(info.get("lean3d") or 0) > 3) * 2 + ((tp[1] - tp[0]) > 10) * 2 + min(info["L"], 1500) / 1500


order = sorted(range(len(results)), key=lambda i: -interest(results[i][0]))
plot_ids = order[:args.plots]
for i in plot_ids:
    info, rows, (s, u, z) = results[i]
    ok = [r for r in rows if "ztop" in r]
    sl = ok[:: max(1, len(ok) // 7)][:8]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for j, r in enumerate(sl):
        m = np.abs(s - r["s"]) < args.slab
        col = plt.cm.viridis(j / max(1, len(sl) - 1))
        ax[0].scatter(u[m], z[m], s=1.5, color=col, label="s=%.0f" % r["s"])
        if "a" in r:
            zz = np.array([r["zlo"], r["zhi"]]); ax[0].plot(r["a"] + r["b"] * zz, zz, "-", color="r", lw=0.8)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("u (across axis, mm)"); ax[0].set_ylabel("z (mm)")
    ax[0].legend(fontsize=7, markerscale=5, loc="best"); ax[0].set_title("perpendicular slices of member %d" % info["id"])
    S = np.array([r["s"] for r in ok]); Z = np.array([r["ztop"] for r in ok])
    ax[1].plot(S, Z, ".", ms=3, label="top z per slice")
    tp = np.array(info["top_poly"]); ax[1].plot(tp[:, 0], tp[:, 1], "r-", lw=1.2, label="top polyline (DP 3mm)")
    zl = [(r["s"], r["zlo"]) for r in ok if "zlo" in r]
    if zl:
        zl = np.array(zl); ax[1].plot(zl[:, 0], zl[:, 1], "g.", ms=3, label="lowest side point")
    ax[1].set_xlabel("s (along axis, mm)"); ax[1].set_ylabel("z (mm)"); ax[1].legend(fontsize=7)
    ax[1].set_title("L=%.0f  t_cap=%.1f  t_2side=%s  lean3d=%s  bottom=%s" % (info["L"], info["t_cap"], info["t2_med"],
                    info.get("lean3d"), info["bottom"]), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "member_%02d.png" % info["id"]), dpi=110); plt.close(fig)

# 对比图：最"斜"（走向离 X/Y 轴最远）的长构件，固定 Y 切 vs 垂直轴线切
diag = [i for i in range(len(results)) if results[i][0]["L"] >= 100]
if diag:
    def obliq(i):
        a = results[i][0]["ang_x"] % 90
        return min(a, 90 - a)
    i = max(diag, key=obliq)
    info, rows, (s, u, z) = results[i]
    c, d = np.array(info["c"]), np.array(info["d"]); nrm = np.array([-d[1], d[0]])
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for j, f in enumerate((0.2, 0.4, 0.6, 0.8)):
        sc = info["s0"] + f * (info["s1"] - info["s0"]); pc = c + d * sc
        m = np.abs(XY[:, 1] - pc[1]) < args.slab
        m &= np.abs(XY[:, 0] - pc[0]) < 70
        if has_base:
            m &= H > zlo
        ax[0, j].scatter(XY[m, 0] - pc[0], H[m], s=1.5)
        ax[0, j].set_title("fixed-Y slice  y=%.0f" % pc[1], fontsize=9); ax[0, j].set_aspect("equal")
        mm = np.abs(s - sc) < args.slab
        ax[1, j].scatter(u[mm], z[mm], s=1.5, color="C1")
        ax[1, j].set_title("perpendicular slice  s=%.0f" % sc, fontsize=9); ax[1, j].set_aspect("equal")
    fig.suptitle("member %d, axis %.1f deg from X: fixed-Y cut stretches width by 1/sin(angle) and skews the lean" % (info["id"], info["ang_x"]))
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "compare_fixedY_vs_perp.png"), dpi=100); plt.close(fig)

# 总览图：高度图 + 构件编号
img = np.zeros((hmp["Ht"], hmp["W"], 3), np.uint8)
v = hm[occ]
if v.size:
    lo, hi = np.percentile(v, [1, 99.5])
    g = (np.clip((hm - lo) / max(hi - lo, 1e-6), 0, 1) * 200 + 40).astype(np.uint8)
    img[occ] = np.stack([g[occ]] * 3, 1)
for info, _, _ in results:
    c, d = np.array(info["c"]), np.array(info["d"])
    a, b = c + d * info["s0"], c + d * info["s1"]
    pa = (int((a[0] - hmp["x0"]) / args.px), int((a[1] - hmp["y0"]) / args.px))
    pb = (int((b[0] - hmp["x0"]) / args.px), int((b[1] - hmp["y0"]) / args.px))
    col = (0, 0, 255) if info["id"] in [results[i][0]["id"] for i in plot_ids] else (0, 160, 255)
    cv2.line(img, pa, pb, col, max(1, int(2 / args.px)))
img = np.ascontiguousarray(img[::-1])
for info, _, _ in results:
    c, d = np.array(info["c"]), np.array(info["d"]); mpt = c + d * (info["s0"] + info["s1"]) / 2
    p = (int((mpt[0] - hmp["x0"]) / args.px), hmp["Ht"] - 1 - int((mpt[1] - hmp["y0"]) / args.px))
    cv2.putText(img, str(info["id"]), p, cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5 / args.px), (255, 255, 0), 1)
sc_ = min(1.0, 2400.0 / max(img.shape[:2]))
if sc_ < 1:
    img = cv2.resize(img, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(args.out, "overview.png"), img)
TM["plots"] = round(time.time() - t1, 2)
summary = dict(has_base=has_base, log=F["log"], segments=len(segs), members=len(results), time_s=TM,
               members_info=[r[0] for r in results], plotted=[results[i][0]["id"] for i in plot_ids])
json.dump(summary, open(os.path.join(args.out, "sections.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(has_base=has_base, segments=len(segs), members=len(results), time_s=TM, plotted=summary["plotted"])))
for i in plot_ids:
    m = results[i][0]
    print("#%d L=%.0f ang=%.0f t_cap=%.1f t2=%s lean2d=%s lean3d=%s ztop=%s bottom=%s side=%d/%d reach=%s rms=%s" % (
        m["id"], m["L"], m["ang_x"], m["t_cap"], m["t2_med"], m["lean2d_med"], m.get("lean3d"), m["ztop_p5_p95"],
        m["bottom"], m["slices_side"], m["slices"], m["side_reaches_bottom"], m.get("side_rms")))
