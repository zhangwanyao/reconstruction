"""交线特征逆向建模 —— Python 原型 v2（C++ 版的对照参考实现）。
流程: 读点云 -> 工件坐标系/判底板(protolib.detect_frame) -> 高度图 -> 凸起区 -> 细条/宽块
      -> 骨架直线段(crossing number) -> 每块板做"垂直于轴线的逐段切片"(顶高/顶面中心/顶面宽) -> 共线合并 -> 重拟合
      -> 端点接头(交线) -> 顶边剖面(跳过被别的板穿过的区间) -> 底板轮廓 -> 棱柱实体 -> STEP(纯文本) + parts.json + 调试图
相对 v1: 判底板改为网格顶面+足迹内判断(能量出底板厚); 分叉用 crossing number; 宽块只在外接框里算;
        板厚=逐切片顶面宽中位数, 方向=逐切片顶面中心直线拟合; 剖面跳过交叉区; STEP 不依赖 OCCT。
用法: python -I proto_v2.py <pcd> <out_dir> [--base-t 20] [--px 1.0] [--mode auto|base|nobase]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("pcd"); ap.add_argument("out")
ap.add_argument("--base-t", type=float, default=20.0, help="底板底面看不到时的默认厚度")
ap.add_argument("--px", type=float, default=0.0, help="高度图像素 mm; 0 = 按点密度自动选")
ap.add_argument("--h-min", type=float, default=8.0)
ap.add_argument("--w-thin", type=float, default=24.0)
ap.add_argument("--mode", default="auto", choices=["auto", "base", "nobase"])
ap.add_argument("--lean-min", type=float, default=3.0, help="侧面拟合倾角 >= 该值(度)才按倾斜板建模; 设 90 = 全部按竖直板")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import detect_frame, heightmap, skeleton_segments, dp_split, disk, fit_plane, simple_ring
from step_writer import write_step

os.makedirs(args.out, exist_ok=True)
TM, LOG = {}, {}
_t = [time.time()]
BIN = 5.0                                     # 切片间距(沿板轴线)


def tick(name):
    now = time.time(); TM[name] = round(now - _t[0], 3); _t[0] = now


def perp(d):
    return np.array([-d[1], d[0]])


def bin_pct(b, v, nb, pct, min_n=3):
    """按 bin 号分组取百分位; 点数不足的 bin 为 NaN。"""
    out = np.full(nb, np.nan)
    if len(b) == 0:
        return out
    order = np.lexsort((v, b)); bs = b[order]; vs = v[order]
    st = np.searchsorted(bs, np.arange(nb)); cnt = np.searchsorted(bs, np.arange(nb), 'right') - st
    ok = cnt >= min_n
    out[ok] = vs[st[ok] + ((cnt[ok] - 1) * pct / 100.0).astype(int)]
    return out


# ---------------------------------------------------------------- 1 读入 + 2 坐标系/判底板
P = read_xyz(args.pcd); P = P[np.isfinite(P).all(1)]
LOG["points"] = int(len(P)); tick("read")
F = detect_frame(P, args.mode)
Q, E, o, has_base = F["Q"], F["E"], F["o"], F["has_base"]
LOG.update(F["log"])
base_t = float(F["base_t"]) if F["base_t"] else args.base_t
LOG["base_t_used"] = round(base_t, 2)
tick("frame")

# ---------------------------------------------------------------- 3 高度图 + 凸起区 + 细条/宽块
def occ_count(xy, p):
    ij = ((xy - xy.min(0)) / p).astype(np.int64); W = int(ij[:, 0].max()) + 1
    return int((np.bincount(ij[:, 1] * W + ij[:, 0], minlength=W * (int(ij[:, 1].max()) + 1)) > 0).sum())


def auto_px(xy, cands=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0), thr=0.7):
    """占用率 ratio(px) = 占用像素数(px) / (4*占用像素数(2px)); 取 ratio>=thr 的最小像素(点稀时像素放大, 免得条带里全是空洞)。"""
    ratios = {}
    for p in cands:
        ratios[p] = round(occ_count(xy, p) / (4.0 * occ_count(xy, 2 * p)), 3)
        if ratios[p] >= thr:
            return p, ratios
    return cands[-1], ratios


def fill_small_holes(m, max_px):
    inv = (m == 0).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    sm = st[:, cv2.CC_STAT_AREA] <= max_px
    sm[0] = False
    m = m.copy(); m[sm[lb] & (inv > 0)] = 1
    return m


if args.px > 0:
    px, px_ratio = args.px, None
else:
    px, px_ratio = auto_px(Q[:, :2])
hmp = heightmap(Q, px, has_base)
hm, occ, base_vis = hmp["hm"], hmp["occ"], hmp["base_vis"]
x0, y0, Wd, Ht = hmp["x0"], hmp["y0"], hmp["W"], hmp["Ht"]
raised = ((hm > args.h_min) if has_base else occ).astype(np.uint8)
raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
raised = fill_small_holes(raised, max(4, int(40 / px ** 2)))
nl, lab, st, _ = cv2.connectedComponentsWithStats(raised, connectivity=8)
small = st[:, cv2.CC_STAT_AREA] < max(20, int(150 / px ** 2))
raised[small[lab] & (lab > 0)] = 0
wide = cv2.morphologyEx(raised, cv2.MORPH_OPEN, disk(args.w_thin / px / 2))
thin = raised & (1 - cv2.dilate(wide, disk(2.0 / px)))
rho_top = len(Q) / max(float(occ.sum()) * px * px, 1.0)        # 俯视每 mm² 的点数(倾斜面判定用)
LOG["heightmap"] = dict(px=px, size=[Wd, Ht], occ_ratio=px_ratio, rho_top=round(rho_top, 3))
tick("heightmap_masks")

# ---------------------------------------------------------------- 4 骨架直线段（细条）+ 宽块（只在外接框内计算）
segs = skeleton_segments(thin, hmp)
for s in segs:
    s["src"] = "thin"
tick("skeleton")
blobs = []


def quad_px(loc, s, w, ext=0.0):
    d = np.asarray(s["d"]); nrm = perp(d); c = np.asarray(s["c"])
    a, b = c + d * (s["s0"] - ext), c + d * (s["s1"] + ext)
    q = np.array([a - nrm * w / 2, b - nrm * w / 2, b + nrm * w / 2, a + nrm * w / 2])
    return np.round(np.c_[(q[:, 0] - loc["x0"]) / loc["px"] - 0.5, (q[:, 1] - loc["y0"]) / loc["px"] - 0.5]).astype(np.int32)


def strip_fill(m, loc, s, w):
    r = np.zeros_like(m); cv2.fillPoly(r, [quad_px(loc, s, w)], 1)
    return float((r & m).sum()) / max(int(r.sum()), 1)


hm_fill = np.where(occ, hm, cv2.dilate(np.where(occ, hm, -1e9).astype(np.float32), np.ones((3, 3), np.uint8)))
nl, lab, st, _ = cv2.connectedComponentsWithStats(wide, connectivity=8)
for k in range(1, nl):
    bx, by, bw, bh, area = st[k]
    if area * px * px < 400:
        continue
    comp = lab[by:by + bh, bx:bx + bw] == k
    hv = hm_fill[by:by + bh, bx:bx + bw][comp]           # 闭运算补上的空像素: 取邻格高度, 仍为空的取中位数
    if (hv > -1e8).sum() < 20:
        continue
    hv = np.where(hv > -1e8, hv, np.median(hv[hv > -1e8]))
    lv_hist = np.convolve(np.bincount(np.clip((hv - hv.min()).astype(int), 0, 5000)), np.ones(5), 'same')
    lvls = []
    for b in np.argsort(-lv_hist):
        if lv_hist[b] < 0.12 * comp.sum():
            break
        z = hv.min() + b + 0.5
        if all(abs(z - l) > 10 for l in lvls):
            lvls.append(z)
    lvls = sorted(lvls) or [float(np.median(hv))]
    dist_lv = np.abs(hv[:, None] - np.array(lvls)[None, :])
    which = np.argmin(dist_lv, 1)
    off = dist_lv.min(1) > 10.0                    # 不属于任何水平层的像素: 斜面(倾斜板露出的侧面)等
    # 斜坡带: 不做块, 按骨架取直线段交给逐段切片 + 侧面倾斜判定(src="slope", 不设 level)
    so = np.zeros(comp.shape, np.uint8); so[comp] = off
    so = cv2.morphologyEx(so, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n4, lab4, st4, _ = cv2.connectedComponentsWithStats(so, connectivity=8)
    for k4 in range(1, n4):
        sx, sy, sw, sh, a4 = st4[k4]
        if a4 * px * px < 1500:
            continue
        m4 = np.pad((lab4[sy:sy + sh, sx:sx + sw] == k4).astype(np.uint8), 3)
        loc = dict(px=px, x0=x0 + (bx + sx - 3) * px, y0=y0 + (by + sy - 3) * px)
        for s in skeleton_segments(m4, loc, min_len_mm=40.0):
            if s["s1"] - s["s0"] < max(60.0, 2.0 * s["w"]) or strip_fill(m4, loc, s, s["w"]) < 0.6:
                continue
            s.update(src="slope"); segs.append(s)
    for li, z in enumerate(lvls):
        sub = np.zeros(comp.shape, np.uint8); sub[comp] = (which == li) & ~off
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for k2 in range(1, n2):
            sx, sy, sw, sh, a2 = st2[k2]
            if a2 * px * px < 400:
                continue
            m2 = np.pad((lab2[sy:sy + sh, sx:sx + sw] == k2).astype(np.uint8), 3)
            r_off, c_off = by + sy - 3, bx + sx - 3
            loc = dict(px=px, x0=x0 + c_off * px, y0=y0 + r_off * px)
            # 宽区也按骨架拆成直线段(之字形腹杆、长条垫板); 矩形填充率>=0.8 的段当作杆/板, 剩下的面积再做块
            acc = np.zeros_like(m2)
            for s in skeleton_segments(m2, loc, min_len_mm=20.0):
                if s["s1"] - s["s0"] < 1.5 * s["w"] or strip_fill(m2, loc, s, s["w"]) < 0.8:
                    continue
                s.update(src="wide", level=float(z)); segs.append(s)
                cv2.fillPoly(acc, [quad_px(loc, s, s["w"] + 4, 0.5 * s["w"] + 5)], 1)
            rest = cv2.morphologyEx(m2 & (1 - acc), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n3, lab3, st3, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
            for k3 in range(1, n3):
                if st3[k3, cv2.CC_STAT_AREA] * px * px < 400:
                    continue
                cs, _ = cv2.findContours((lab3 == k3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                poly = simple_ring(max(cs, key=cv2.contourArea), max(2.0, 2.5 / px), "block %d" % len(blobs))
                blobs.append(dict(poly=np.c_[loc["x0"] + (poly[:, 0] + 0.5) * px, loc["y0"] + (poly[:, 1] + 0.5) * px],
                                  top=float(z), mask_px=int(st3[k3, cv2.CC_STAT_AREA])))
LOG["raw_segments"] = len(segs); LOG["blobs"] = len(blobs)
tick("wide_blocks")

# ---------------------------------------------------------------- 5 逐段切片拟合每块板
XY = Q[:, :2]; H = Q[:, 2]
tree2 = cKDTree(XY)


def strip_points(c, d, s0, s1, half):
    n = max(1, int(np.ceil((s1 - s0) / max(2 * half, 40.0))))
    ss = s0 + (np.arange(n) + 0.5) * (s1 - s0) / n
    lists = tree2.query_ball_point(c[None] + ss[:, None] * d[None], np.hypot((s1 - s0) / n / 2, half))
    return np.unique(np.concatenate([np.asarray(l, np.int64) for l in lists]))


def fit_strip(s):
    """每 BIN mm 一个横切片: 顶高(P95)、顶面带(顶高-4mm 以上)的中心与宽度。
    方向 = 切片中心的稳健直线拟合; 板厚 = 切片宽度中位数(P2.5~P97.5 / 0.95 还原均匀分布的全宽)。"""
    c = np.asarray(s["c"], float); d = np.asarray(s["d"], float); d = d / np.linalg.norm(d)
    half = s["w"] / 2 + 6; ext = max(15.0, 0.5 * s["w"] + 5)
    idx = strip_points(c, d, s["s0"] - ext, s["s1"] + ext, half + 2)
    if len(idx) < 30:
        return None
    xy = XY[idx]; h = H[idx]
    hlo = args.h_min / 2 if has_base else -1e9
    if "level" in s:
        hlo = max(hlo, s["level"] - 15)
    for it in range(3):
        nrm = perp(d)
        sa = (xy - c) @ d; va = (xy - c) @ nrm
        band = (np.abs(va) < half) & (sa > s["s0"] - ext) & (sa < s["s1"] + ext)
        sel = band & (h > hlo)
        if sel.sum() < 30:
            return None
        sa_, va_, h_ = sa[sel], va[sel], h[sel]
        smin = sa_.min()
        b = np.floor((sa_ - smin) / BIN).astype(int); nb = b.max() + 1
        top = bin_pct(b, h_, nb, 95)
        ok = ~np.isnan(top)
        if ok.sum() < 2:
            return None
        topi = np.interp(np.arange(nb), np.nonzero(ok)[0], top[ok])
        cap = h_ > topi[b] - 4
        lo = bin_pct(b[cap], va_[cap], nb, 2.5, 6); hi = bin_pct(b[cap], va_[cap], nb, 97.5, 6)
        good = ~np.isnan(lo)
        if good.sum() < 2:
            return None
        sc = smin + (np.arange(nb) + 0.5) * BIN
        S_, C_ = sc[good], ((lo + hi) / 2)[good]
        A_ = np.c_[np.ones_like(S_), S_]; keep = np.ones(len(S_), bool)
        for _ in range(3):
            coef = np.linalg.lstsq(A_[keep], C_[keep], rcond=None)[0]
            r = C_ - A_ @ coef
            keep = np.abs(r) < 3 * (1.4826 * np.median(np.abs(r[keep])) + 0.3)
        if it == 2 or good.sum() < 6 or abs(coef[1]) > np.tan(np.radians(5)) or abs(coef[1]) < 1e-4:
            break
        c = c + nrm * coef[0]; d = d + coef[1] * nrm; d = d / np.linalg.norm(d)
    wid = ((hi - lo) / 0.95)[good][keep]
    width = float(np.median(wid)); voff = float(np.median(C_[keep] - coef[1] * S_[keep]))
    s0, s1 = np.percentile(sa_[cap], [0.3, 99.7])
    hb = h[band]
    bottom = 0.0 if has_base else float(np.percentile(hb, 1))
    out = dict(c=c + nrm * voff, d=d, s0=float(s0), s1=float(s1), w=width, bins=np.c_[sc[ok], top[ok]], bottom=bottom,
               n_top=int(cap.sum()), src=s["src"], n_slices=int(good.sum()), w_iqr=float(np.subtract(*np.percentile(wid, [75, 25]))))
    if "level" in s:
        out["level"] = s["level"]
    return out


plates = [p for p in (fit_strip(s) for s in segs) if p is not None and p["s1"] - p["s0"] > 15 and p["w"] > 2.5]
LOG["fitted"] = len(plates)
tick("fit")


# ---------------------------------------------------------------- 6 共线合并（遮挡桥接）+ 重拟合
def gap_has_base(p, a, b):
    if not has_base or b - a < px:
        return 0.0
    ss = np.arange(a, b, px); vv = np.linspace(-p["w"] / 2, p["w"] / 2, 3)
    pts = (p["c"][None, None] + ss[:, None, None] * p["d"][None, None] + vv[None, :, None] * perp(p["d"])[None, None]).reshape(-1, 2)
    c = ((pts[:, 0] - x0) / px).astype(int).clip(0, Wd - 1); r = ((pts[:, 1] - y0) / px).astype(int).clip(0, Ht - 1)
    return float(base_vis[r, c].mean())


merged_any = set()
changed = True
while changed:
    changed = False
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            A, B = plates[i], plates[j]
            if abs(A["d"] @ B["d"]) < np.cos(np.radians(4)):
                continue
            nA = perp(A["d"])
            eB = [B["c"] + B["d"] * B["s0"], B["c"] + B["d"] * B["s1"]]
            if max(abs((e - A["c"]) @ nA) for e in eB) > max(3.0, 0.5 * max(A["w"], B["w"])):
                continue
            if max(A["w"], B["w"]) > 1.8 * min(A["w"], B["w"]) + 2:
                continue
            sb = sorted(((e - A["c"]) @ A["d"]) for e in eB)
            gap_lo, gap_hi = (A["s1"], sb[0]) if sb[0] >= A["s1"] else ((sb[1], A["s0"]) if sb[1] <= A["s0"] else (0, 0))
            if gap_hi - gap_lo > 400 or (gap_hi - gap_lo > 15 and gap_has_base(A, gap_lo, gap_hi) > 0.3):
                continue
            wA, wB = A["n_top"], B["n_top"]
            A.update(s0=float(min(A["s0"], sb[0])), s1=float(max(A["s1"], sb[1])), w=(A["w"] * wA + B["w"] * wB) / (wA + wB),
                     n_top=wA + wB, bottom=min(A["bottom"], B["bottom"]))
            merged_any.add(id(A)); plates.pop(j); changed = True
            break
        if changed:
            break
for i, p in enumerate(plates):
    if id(p) in merged_any:
        r = fit_strip(dict(c=p["c"], d=p["d"], s0=p["s0"], s1=p["s1"], w=p["w"], src=p["src"], **({"level": p["level"]} if "level" in p else {})))
        if r is not None:
            r["s0"], r["s1"] = min(r["s0"], p["s0"]), max(r["s1"], p["s1"])
            plates[i] = r
LOG["merged"] = len(plates)
tick("merge")

# ---------------------------------------------------------------- 7 端点接头（中线与对方中面求交 -> 顶到近侧面 / 伸到外侧面）
plates = [p for p in plates if p["s1"] - p["s0"] >= max(15.0, 2.0 * p["w"])]     # 交叉口残渣: 又短又宽
for A in plates:
    for end in ("s0", "s1"):
        best = None
        for B in plates:
            if B is A or abs(A["d"] @ B["d"]) > np.cos(np.radians(20)):
                continue
            nB = perp(B["d"]); den = A["d"] @ nB
            sA = ((B["c"] - A["c"]) @ nB) / den
            sB = (A["c"] + A["d"] * sA - B["c"]) @ B["d"]
            if sB < B["s0"] - 10 or sB > B["s1"] + 10:
                continue
            off = (B["w"] / 2) / abs(den)
            if abs(sA - A[end]) > off + A["w"] / 2 + 15:
                continue
            if best is None or abs(sA - A[end]) < abs(best[0] - A[end]):
                best = (sA, off, B, sB)
        if best is None:
            continue
        sA, off, B, sB = best
        inside = B["s0"] + 12 < sB < B["s1"] - 12
        sign = -1 if end == "s1" else 1
        A[end] = float(sA + sign * off) if (inside or (B["s1"] - B["s0"]) >= (A["s1"] - A["s0"])) else float(sA - sign * off)
        A.setdefault("joints", []).append(end)
plates = [p for p in plates if p["s1"] - p["s0"] >= max(10.0, 0.5 * p["w"])]
tick("junctions")


# ---------------------------------------------------------------- 8 顶边剖面（跳过交叉区）
def profile(p):
    bs, bt = p["bins"][:, 0], p["bins"][:, 1]
    m = (bs > p["s0"] - BIN) & (bs < p["s1"] + BIN)
    for q in plates:
        if q is p:
            continue
        sn = abs(p["d"][0] * q["d"][1] - p["d"][1] * q["d"][0])
        if sn < np.sin(np.radians(20)):
            continue
        nq = perp(q["d"]); sA = ((q["c"] - p["c"]) @ nq) / (p["d"] @ nq)
        sB = (p["c"] + p["d"] * sA - q["c"]) @ q["d"]
        if q["s0"] - 5 <= sB <= q["s1"] + 5:
            hw = (q["w"] / 2) / sn + (p["w"] / 2) * abs(p["d"] @ q["d"]) / sn + 4
            m &= ~((bs > sA - hw) & (bs < sA + hw))
    if m.sum() < 2:
        m = (bs > p["s0"] - BIN) & (bs < p["s1"] + BIN)
    if m.sum() < 1:
        m = np.ones(len(bs), bool)
    s_, z_ = bs[m], bt[m]
    z0, z1 = np.interp([p["s0"], p["s1"]], s_, z_)
    inner = (s_ > p["s0"]) & (s_ < p["s1"])
    pr = np.vstack([[p["s0"], z0], np.c_[s_[inner], z_[inner]], [p["s1"], z1]])
    if np.ptp(pr[:, 1]) < 6:
        pr = np.array([[p["s0"], np.median(z_)], [p["s1"], np.median(z_)]])
    else:
        pr = pr[dp_split(pr, 3.0)]
    pr[:, 1] = np.maximum(pr[:, 1], p["bottom"] + 2)
    return pr


for p in plates:
    p["prof"] = profile(p)
tick("profiles")

# ---------------------------------------------------------------- 8b 侧面平面 -> 倾斜板（门槛严格; 不过门槛一律按竖直板）
EZ = np.array([0, 0, 1.0])


def ransac_plane(X, d3, iters=200, tol=1.5, seed=0):
    """侧面候选: 法向近水平(|nz|<sin30°)且与板轴线近垂直(|n·d|<sin15°) 的内点最多平面。"""
    rng = np.random.default_rng(seed); best, bc = None, 0
    Xs = X if len(X) <= 5000 else X[rng.choice(len(X), 5000, replace=False)]
    for _ in range(iters):
        a, b, c_ = Xs[rng.choice(len(Xs), 3, replace=False)]
        n = np.cross(b - a, c_ - a); ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if abs(n[2]) > np.sin(np.radians(30)) or abs(n @ d3) > np.sin(np.radians(15)):
            continue
        cnt = int((np.abs((Xs - a) @ n) < tol).sum())
        if cnt > bc:
            best, bc = (n, float(n @ a)), cnt
    if best is None:
        return None
    n, dd = best
    inl = np.abs(X @ n - dd) < tol
    for _ in range(2):
        n, dd = fit_plane(X[inl]); inl = np.abs(X @ n - dd) < tol
    return n, dd, inl


def lean_check(p):
    """在扩宽的板带里(顶面以下、根部以上)找侧面平面。全部满足才算倾斜板:
    内点>=max(300,15%)、内点高度跨度>=max(30mm,板高50%)、rms<=1mm、该面在顶高处正好贴着顶面条带、
    倾角>=lean_min、该面的相对点密度>=0.8*sin(倾角)。"""
    pr = p["prof"]; Hp = float(pr[:, 1].max() - p["bottom"])
    if Hp < 40 or p["s1"] - p["s0"] < 40:
        return None
    half = p["w"] / 2 + 6 + min(80.0, 0.45 * Hp)
    idx = strip_points(p["c"], p["d"], p["s0"] + 5, p["s1"] - 5, half + 2)
    X = Q[idx]; nrm = perp(p["d"])
    sa = (X[:, :2] - p["c"]) @ p["d"]; va = (X[:, :2] - p["c"]) @ nrm
    zt = np.interp(sa, pr[:, 0], pr[:, 1])
    zlo = 6.0 if has_base else p["bottom"] + 6
    m = (np.abs(va) < half) & (sa > p["s0"] + 5) & (sa < p["s1"] - 5) & (X[:, 2] < zt - 6) & (X[:, 2] > zlo)
    if m.sum() < 300:
        return None
    d3 = np.array([*p["d"], 0.0])
    r = ransac_plane(X[m], d3)
    if r is None:
        return None
    n, dd, inl = r
    if n[2] < 0:
        n, dd = -n, -dd
    Xi = X[m][inl]
    info = dict(inl=int(inl.sum()), cand=int(m.sum()), zspan=round(float(np.subtract(*np.percentile(Xi[:, 2], [98, 2]))), 1),
                rms=round(float(np.sqrt(np.mean((Xi @ n - dd) ** 2))), 2), lean=round(float(np.degrees(np.arcsin(min(1.0, n[2])))), 2))
    nh = np.array([n[0], n[1], 0.0]); nh /= np.linalg.norm(nh)
    smid = (p["s0"] + p["s1"]) / 2; ztop = float(np.interp(smid, pr[:, 0], pr[:, 1]))
    cmid = np.array([*(p["c"] + p["d"] * smid), 0.0])
    v_face = (dd - cmid @ n - ztop * n[2]) / (nh @ n)            # 顶高处, 该面离顶面条带中心的水平距离
    info["face_to_cap"] = round(float(v_face), 1)
    info["ok"] = bool(info["inl"] >= max(300, 0.15 * info["cand"]) and info["zspan"] >= max(30.0, 0.5 * Hp)
                      and info["rms"] <= 1.0 and abs(v_face) <= p["w"] / 2 + 4)
    # 朝扫描仪倾斜 θ 的面, 单位面积点数约为水平面的 sinθ 倍; 竖直面被斜着扫到时远低于这个值(实测假倾斜 <=0.07, 真倾斜 >=0.11)
    info["rel_dens"] = round(info["inl"] / max((p["s1"] - p["s0"] - 10) * info["zspan"], 1.0) / rho_top, 3)
    info["dense"] = bool(info["rel_dens"] >= 0.8 * np.sin(np.radians(info["lean"])))
    p["lean_info"] = info
    if not info["ok"] or info["lean"] < args.lean_min or not info["dense"]:
        return None
    th = np.radians(info["lean"])
    t = max(0.5 * p["w"], (p["w"] - 4 * np.tan(th)) * np.cos(th))
    k = -1.0 if (np.array([*cmid[:2], ztop]) @ n - dd) < 0 else 1.0       # 板体在面的哪一侧: 顶面条带所在侧
    d3n = np.cross(EZ, n); d3n /= np.linalg.norm(d3n)
    if d3n[:2] @ p["d"] < 0:
        d3n = -d3n
    depth = k * (X @ n - dd)
    slab = (depth > -1.5) & (depth < t + 1.5) & (X[:, 2] > (2.0 if has_base else p["bottom"] + 2))
    if slab.sum() < 100:
        return None
    s_ax = X[slab, :2] @ d3n[:2]; zs = X[slab, 2]
    s0, s1 = np.percentile(s_ax, [0.3, 99.7])
    b = np.floor((s_ax - s0) / BIN).astype(int); nb = int(b.max()) + 1
    top = bin_pct(b, zs, nb, 95); ok = ~np.isnan(top)
    sc = s0 + (np.arange(nb) + 0.5) * BIN
    prf = np.vstack([[s0, top[ok][0]], np.c_[sc[ok], top[ok]], [s1, top[ok][-1]]])
    prf = np.array([[s0, np.median(top[ok])], [s1, np.median(top[ok])]]) if np.ptp(prf[:, 1]) < 6 else prf[dp_split(prf, 3.0)]
    prf[:, 1] = np.maximum(prf[:, 1], p["bottom"] + 2)
    return dict(n=n, dd=dd, k=k, t=float(t), d3=d3n, nh=nh, s0=float(s0), s1=float(s1), prof=prf, info=info)


LOG["lean"] = []
for p in plates:
    lp = lean_check(p)
    if lp is not None:
        p["lean"] = lp
    if "lean_info" in p:
        LOG["lean"].append(dict(c=[round(x, 1) for x in p["c"]], L=round(p["s1"] - p["s0"], 1), **p["lean_info"]))


def top_of(p):
    return float((p["lean"]["prof"] if "lean" in p else p["prof"])[:, 1].max())


# 倾斜板的可见斜面在俯视图里也会被认成一条"矮板": 顶面中点落在倾斜板板体里(或斜面上)的, 只留顶更高的那块
drop = set()
for A in [p for p in plates if "lean" in p]:
    lp = A["lean"]
    for B in plates:
        if B is A or id(B) in drop or id(A) in drop or abs(A["d"] @ B["d"]) < np.cos(np.radians(10)):
            continue
        sm = (B["s0"] + B["s1"]) / 2
        zb = top_of(B) if "lean" in B else float(np.interp(sm, B["prof"][:, 0], B["prof"][:, 1]))
        if zb > top_of(A) + 5:                     # 只删比倾斜板矮的"影子"; 更高的板不可能是它露出的侧面
            continue
        xm = np.array([*(B["c"] + B["d"] * sm), zb])
        depth = lp["k"] * (xm @ lp["n"] - lp["dd"])
        sB = xm[:2] @ lp["d3"][:2]
        if lp["s0"] - 10 < sB < lp["s1"] + 10 and -3 < depth < lp["t"] + 3:
            drop.add(id(B))
LOG["lean_dups_dropped"] = len(drop)
plates = [p for p in plates if id(p) not in drop]
tick("lean")

# ---------------------------------------------------------------- 9 底板轮廓
base_poly = None
if has_base:
    occb = (base_vis | (raised > 0)).astype(np.uint8)
    occb = cv2.morphologyEx(occb, cv2.MORPH_CLOSE, disk(6.0 / px))
    occb = ndimage.binary_fill_holes(occb).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(occb, connectivity=8)
    if nl > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_b = max(cs, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt_b)
        fill_b = cv2.contourArea(cnt_b) / max(rect[1][0] * rect[1][1], 1.0)
        LOG["base_fill_rect"] = round(float(fill_b), 3)
        # 底板多半是矩形: 轮廓占最小外接矩形 >=85% 就用矩形(补上没扫到的角落), 否则保留多边形
        poly = cv2.boxPoints(rect) if fill_b >= 0.85 else simple_ring(cnt_b, max(2.0, 3.0 / px), "base")
        base_poly = np.c_[x0 + (poly[:, 0] + 0.5) * px, y0 + (poly[:, 1] + 0.5) * px]
tick("base_outline")


# ---------------------------------------------------------------- 10 棱柱实体（工件坐标 -> 原坐标）+ STEP
def to_world(Qp):
    return Qp @ E + o


solids = []


def dedupe(ring):
    out = []
    for s, z in ring:
        if not out or np.hypot(s - out[-1][0], z - out[-1][1]) >= 0.5:
            out.append((s, z))
    return out


for k, p in enumerate(plates):
    if "lean" in p:
        # 倾斜板: 轮廓画在拟合出来的可见侧面上, 再沿水平法向推 t/cosθ (上下边保持水平, 两侧面间距正好 = t)
        lp = p["lean"]; n3, dd, nh, d3 = lp["n"], lp["dd"], lp["nh"], lp["d3"]
        cth = float(n3 @ nh)
        poly = dedupe([(lp["s0"], p["bottom"]), (lp["s1"], p["bottom"])] + [(s, z) for s, z in lp["prof"][::-1]])
        pts3 = np.array([[*(d3[:2] * s + nh[:2] * ((dd - z * n3[2]) / cth)), z] for s, z in poly])
        solids.append(dict(name="plate_%02d" % k, kind="lean_" + p["src"], ring=to_world(pts3), vec=(lp["k"] * lp["t"] / cth * nh) @ E,
                           t=round(lp["t"], 2), L=round(lp["s1"] - lp["s0"], 1), top=round(float(lp["prof"][:, 1].max()), 1),
                           lean_deg=lp["info"]["lean"], prof_pts=len(lp["prof"])))
        continue
    nrm = perp(p["d"]); pr = p["prof"]
    poly = dedupe([(p["s0"], p["bottom"]), (p["s1"], p["bottom"])] + [(s, z) for s, z in pr[::-1]])
    base0 = p["c"] - nrm * p["w"] / 2
    pts3 = np.array([[*(base0 + p["d"] * s), z] for s, z in poly])
    solids.append(dict(name="plate_%02d" % k, kind="strip_" + p["src"], ring=to_world(pts3), vec=np.array([*(nrm * p["w"]), 0.0]) @ E,
                       t=round(p["w"], 2), L=round(p["s1"] - p["s0"], 1), top=round(float(pr[:, 1].max()), 1),
                       prof_pts=len(pr), slices=p["n_slices"], w_iqr=round(p["w_iqr"], 2)))
for k, b in enumerate(blobs):
    bot = 0.0
    if not has_base:
        bb = b["poly"].min(0), b["poly"].max(0)
        inb = (XY[:, 0] > bb[0][0]) & (XY[:, 0] < bb[1][0]) & (XY[:, 1] > bb[0][1]) & (XY[:, 1] < bb[1][1])
        bot = float(np.percentile(H[inb], 1)) if inb.any() else 0.0
    pts3 = np.c_[b["poly"], np.full(len(b["poly"]), bot)]
    solids.append(dict(name="block_%02d" % k, kind="block", ring=to_world(pts3), vec=np.array([0, 0, b["top"] - bot]) @ E,
                       t=round(b["top"] - bot, 1), L=None, top=round(b["top"], 1)))
if base_poly is not None:
    pts3 = np.c_[base_poly, np.full(len(base_poly), -base_t)]
    solids.append(dict(name="base", kind="base", ring=to_world(pts3), vec=np.array([0, 0, base_t]) @ E, t=base_t, L=None, top=0.0))
tick("solids")
LOG["step_solids"] = write_step(os.path.join(args.out, "model.step"), [(s["name"], s["ring"], s["vec"]) for s in solids])
tick("step")

# ---------------------------------------------------------------- 11 调试图 + parts.json
img = np.zeros((Ht, Wd, 3), np.uint8)
vals = hm[occ]
if vals.size:
    lo, hi = np.percentile(vals, [1, 99.5])
    g = (np.clip((hm - lo) / max(hi - lo, 1e-6), 0, 1) * 200 + 40).astype(np.uint8)
    img[occ] = np.stack([g[occ]] * 3, 1)
img[base_vis & (hm < args.h_min)] = (60, 90, 60)


def mmpix(xy):
    return np.c_[(xy[:, 0] - x0) / px, (xy[:, 1] - y0) / px].astype(np.int32)


for p in plates:
    nrm = perp(p["d"]); a, b = p["c"] + p["d"] * p["s0"], p["c"] + p["d"] * p["s1"]
    quad = np.array([a - nrm * p["w"] / 2, b - nrm * p["w"] / 2, b + nrm * p["w"] / 2, a + nrm * p["w"] / 2])
    cv2.polylines(img, [mmpix(quad)], True, {"thin": (0, 0, 255), "slope": (255, 0, 255)}.get(p["src"], (255, 128, 0)), max(1, int(1.5 / px)))
for b in blobs:
    cv2.polylines(img, [mmpix(b["poly"])], True, (0, 200, 255), max(1, int(1.5 / px)))
if base_poly is not None:
    cv2.polylines(img, [mmpix(base_poly)], True, (0, 255, 0), max(1, int(2 / px)))
img = img[::-1]
sc = min(1.0, 2400.0 / max(img.shape[:2]))
if sc < 1:
    img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(args.out, "debug_top.png"), img)
LOG["time_s"] = TM; LOG["total_s"] = round(sum(TM.values()), 2)
LOG["solids"] = [{k: v for k, v in s.items() if k not in ("ring", "vec")} for s in solids]
json.dump(LOG, open(os.path.join(args.out, "parts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(has_base=has_base, base_t=base_t, segs=LOG["raw_segments"], fitted=LOG["fitted"], merged=LOG["merged"],
                      blobs=len(blobs), solids=len(solids), step=LOG["step_solids"], total_s=LOG["total_s"], t=TM)))
