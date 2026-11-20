"""交线特征逆向建模 —— 第一版 Python 原型 v1（已被 proto_v2.py 取代，只作历史参考；C++ 版照 proto_v2.py 写）。
流程: 读点云 -> 降采样+法向 -> 判底板/定基准面 -> 工件坐标系 -> 高度图
      -> 凸起区 -> 细条(立板)/宽块 分离 -> 骨架 -> 直线段 -> 用原始点拟合每块板(中线/厚度/顶高剖面/端点)
      -> 共线合并(遮挡桥接) -> 端点接头(T/L) -> 底板轮廓 -> 拉伸成实体 -> STEP + parts.json + 调试图
用法: python -I proto_rebuild.py <pcd> <out_dir> [--ocp DIR] [--base-t 20] [--px 1.0] [--mode auto|base|nobase]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("pcd"); ap.add_argument("out")
ap.add_argument("--ocp", default=os.environ.get("OCP_DEPS", ""))
ap.add_argument("--base-t", type=float, default=20.0)
ap.add_argument("--px", type=float, default=1.0)
ap.add_argument("--h-min", type=float, default=8.0)
ap.add_argument("--w-thin", type=float, default=24.0)
ap.add_argument("--mode", default="auto", choices=["auto", "base", "nobase"])
ap.add_argument("--no-step", action="store_true")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz

os.makedirs(args.out, exist_ok=True)
TM, LOG = {}, {}
_t = [time.time()]


def tick(name):
    now = time.time(); TM[name] = round(now - _t[0], 3); _t[0] = now


# ---------------------------------------------------------------- 基础工具
def voxel_down(p, s):
    k = np.floor((p - p.min(0)) / s).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel(); out = np.empty((len(cnt), 3))
    for d in range(3):
        out[:, d] = np.bincount(inv, p[:, d], len(cnt)) / cnt
    return out


def normals(p, k=12, chunk=200000):
    tree = cKDTree(p); n = np.empty_like(p); curv = np.empty(len(p))
    for s in range(0, len(p), chunk):
        _, idx = tree.query(p[s:s + chunk], k, workers=-1)
        c = p[idx] - p[idx].mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', c, c))
        n[s:s + chunk] = v[:, :, 0]; curv[s:s + chunk] = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    return n, curv


def fit_plane(P):
    c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T)); n = v[:, 0]
    return n, float(n @ c)


def robust_plane(P, iters=3, k=2.5, floor=0.8):
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        n, d = fit_plane(P[keep]); r = P @ n - d
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < max(floor, k * s)
    n, d = fit_plane(P[keep])
    return n, d, keep


def dp_split(pts, tol):
    """Douglas-Peucker 断点（返回保留点的下标）。pts: (n,2) 有序。"""
    keep = [0, len(pts) - 1]; stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = pts[b] - pts[a]; L = np.hypot(*d)
        seg = pts[a + 1:b] - pts[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i; keep.append(m); stack += [(a, m), (m, b)]
    return sorted(set(keep))


def disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ---------------------------------------------------------------- 1 读入
P = read_xyz(args.pcd); P = P[np.isfinite(P).all(1)]
LOG["points"] = int(len(P)); tick("read")

# ---------------------------------------------------------------- 2 降采样 + 法向
cell = 10.0
ij0 = np.floor((P[:, :2] - P[:, :2].min(0)) / cell).astype(np.int64)
foot_area = len(np.unique(ij0[:, 0] * 10 ** 7 + ij0[:, 1])) * cell ** 2
vox = max(3.0, float(np.sqrt(foot_area / 4e5)))
V = voxel_down(P, vox); Nv, curv = normals(V)
LOG["voxel_mm"] = round(vox, 2); LOG["voxel_points"] = int(len(V)); tick("voxel_normals")

# ---------------------------------------------------------------- 3 上方向 + 水平层 + 判底板
m = Nv * np.where(Nv[:, 2:3] < 0, -1.0, 1.0)
up = np.array([0, 0, 1.0])
for ang in (30, 15, 8):
    sel = (curv < 0.01) & (m @ up > np.cos(np.radians(ang)))
    up = m[sel].mean(0); up /= np.linalg.norm(up)
a0 = np.cross([0, 1.0, 0], up); a0 /= np.linalg.norm(a0)
R0 = np.vstack([a0, np.cross(up, a0), up])
q = V @ R0.T; mq = m @ R0.T
hz = (curv < 0.02) & (mq[:, 2] > np.cos(np.radians(10)))
zh = q[hz, 2]; zmin = zh.min()
hist = np.convolve(np.bincount(np.floor(zh - zmin).astype(int)), np.ones(5), 'same')
levels = []
for b in np.argsort(-hist):
    if hist[b] < max(150, 0.01 * hz.sum()):
        break
    z = zmin + b + 0.5
    if all(abs(z - l) > 8 for l in levels):
        levels.append(z)
levels.sort()
qa = q[:, :2]
allcells = np.unique(np.floor(qa / cell).astype(np.int64), axis=0)
hull_all = cv2.contourArea(cv2.convexHull(allcells.astype(np.float32)))
cand = []
for z in levels:
    selp = hz & (np.abs(q[:, 2] - z) < 2.5)
    cells = np.unique(np.floor(q[selp, :2] / cell).astype(np.int64), axis=0)
    if len(cells) < 20:
        continue
    area = len(cells) * cell ** 2
    hull = cv2.contourArea(cv2.convexHull(cells.astype(np.float32))) * cell ** 2
    far = float((q[:, 2] < z - 5).mean())
    cand.append(dict(z=round(float(z), 1), area_m2=round(area / 1e6, 3), solidity=round(area / max(hull, 1), 2),
                     hull_ratio=round(hull / max(hull_all * cell ** 2, 1), 2), far=round(far, 3)))
LOG["h_levels"] = cand
has_base = False
base_lv = None
if cand:
    big = max(c["area_m2"] for c in cand)
    for c in cand:  # 从低到高找第一个"大"水平层
        if c["area_m2"] >= 0.25 * big:
            base_lv = c; break
    has_base = bool(base_lv and base_lv["far"] < 0.03 and base_lv["solidity"] > 0.3 and base_lv["hull_ratio"] > 0.3)
if args.mode != "auto":
    has_base = args.mode == "base"
LOG["has_base"] = has_base
tick("base_detect")

# ---------------------------------------------------------------- 4 基准面 + 工件坐标系
Pr = P @ R0.T
if has_base:
    zb = base_lv["z"]
    bcells = set(map(tuple, np.unique(np.floor(q[hz & (np.abs(q[:, 2] - zb) < 2.5), :2] / cell).astype(np.int64), axis=0)))
    near = np.abs(Pr[:, 2] - zb) < 4
    ck = np.floor(Pr[near, :2] / cell).astype(np.int64)
    inb = np.array([tuple(c) in bcells for c in ck]) if len(ck) < 3e6 else np.ones(len(ck), bool)
    Bp = P[near][inb]
    if len(Bp) > 400000:
        Bp = Bp[np.random.default_rng(0).choice(len(Bp), 400000, replace=False)]
    n, d, keep = robust_plane(Bp)
    if n @ up < 0:
        n, d = -n, -d
    res = Bp[keep] @ n - d
    LOG["base_plane"] = dict(n=[round(x, 5) for x in n], tilt_deg=round(float(np.degrees(np.arccos(abs(n[2])))), 3),
                             rms=round(float(np.sqrt(np.mean(res ** 2))), 3),
                             p95_abs=round(float(np.percentile(np.abs(Bp @ n - d), 95)), 3), inliers=int(keep.sum()))
    ez = n; o = ez * d
    base_xy = (Bp[keep] - o)
else:
    ez = up; o = np.zeros(3)
    base_xy = P - o
# X 轴: 基准面内的主方向（最小外接矩形长边）
tmpx = np.cross([0, 1.0, 0], ez); tmpx /= np.linalg.norm(tmpx); tmpy = np.cross(ez, tmpx)
uv = np.c_[base_xy @ tmpx, base_xy @ tmpy].astype(np.float32)
if len(uv) > 200000:
    uv = uv[np.random.default_rng(1).choice(len(uv), 200000, replace=False)]
(cx_, cy_), (rw, rh), ang = cv2.minAreaRect(uv)
th = np.radians(ang if rw >= rh else ang + 90)
ex = np.cos(th) * tmpx + np.sin(th) * tmpy; ey = np.cross(ez, ex)
E = np.vstack([ex, ey, ez])
Q = (P - o) @ E.T
if not has_base:
    zf = float(np.percentile(Q[:, 2], 0.5)); o = o + ez * zf; Q[:, 2] -= zf
LOG["frame"] = dict(origin=[round(x, 3) for x in o], ex=[round(x, 5) for x in ex], ey=[round(x, 5) for x in ey], ez=[round(x, 5) for x in ez])
tick("frame")

# ---------------------------------------------------------------- 5 高度图
px = args.px
X, Y, H = Q[:, 0], Q[:, 1], Q[:, 2]
x0, y0 = X.min() - 10 * px, Y.min() - 10 * px
I = ((X - x0) / px).astype(np.int64); J = ((Y - y0) / px).astype(np.int64)
Wd, Ht = int(I.max()) + 11, int(J.max()) + 11
lin = J * Wd + I
hm = np.full(Wd * Ht, -1e9); np.maximum.at(hm, lin, H); hm = hm.reshape(Ht, Wd)
occ = hm > -1e8
base_vis = np.zeros(Wd * Ht, bool)
if has_base:
    base_vis[lin[np.abs(H) < 2.5]] = True
base_vis = base_vis.reshape(Ht, Wd)
LOG["heightmap"] = dict(px=px, size=[Wd, Ht])
tick("heightmap")


def to_mm(r, c):
    return np.c_[x0 + (np.asarray(c) + 0.5) * px, y0 + (np.asarray(r) + 0.5) * px]


# ---------------------------------------------------------------- 6 凸起区 + 细条/宽块
if has_base:
    raised = (hm > args.h_min).astype(np.uint8)
else:
    raised = occ.astype(np.uint8)
raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
nl, lab, st, _ = cv2.connectedComponentsWithStats(raised, connectivity=8)
small = st[:, cv2.CC_STAT_AREA] < max(20, int(150 / px ** 2))
raised[small[lab] & (lab > 0)] = 0
wide = cv2.morphologyEx(raised, cv2.MORPH_OPEN, disk(args.w_thin / px / 2))
thin = raised & (1 - cv2.dilate(wide, disk(2.0 / px)))
tick("masks")


# ---------------------------------------------------------------- 7 骨架 -> 直线段
def skeleton_segments(mask, min_len_mm=20.0):
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = sk & (nb >= 3)
    br = sk & ~(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    lb, n = ndimage.label(br, structure=np.ones((3, 3)))
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    segs = []
    objs = ndimage.find_objects(lb)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lb[sl] == k)
        rr = rr + sl[0].start; cc = cc + sl[1].start
        if len(rr) * px < min_len_mm * 0.5:
            continue
        # 沿骨架走一遍得到有序链
        pix = {(r, c) for r, c in zip(rr, cc)}
        deg = {p: sum((p[0] + a, p[1] + b) in pix for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b) for p in pix}
        start = min(pix, key=lambda p: (deg[p], p))
        chain = [start]; seen = {start}; cur = start
        while True:
            nxt = [(cur[0] + a, cur[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b)]
            nxt = [p for p in nxt if p in pix and p not in seen]
            if not nxt:
                break
            nxt.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
            cur = nxt[0]; chain.append(cur); seen.add(cur)
        ch = np.array(chain, float)
        xy = to_mm(ch[:, 0], ch[:, 1])
        brk = dp_split(xy, max(2.0, 2 * px))
        for a, b in zip(brk[:-1], brk[1:]):
            seg = xy[a:b + 1]
            if len(seg) < 3:
                continue
            c = seg.mean(0); w_, v_ = np.linalg.eigh(np.cov((seg - c).T)); d = v_[:, 1]
            s = (seg - c) @ d
            L = s.max() - s.min()
            if L < min_len_mm:
                continue
            wpx = dt[ch[a:b + 1, 0].astype(int), ch[a:b + 1, 1].astype(int)]
            segs.append(dict(c=c, d=d, s0=float(s.min()), s1=float(s.max()), w=float(2 * np.median(wpx) * px)))
    return segs


segs = skeleton_segments(thin)
for s in segs:
    s["src"] = "thin"
tick("skeleton")

# 宽块: 按高度分层 -> 长条用骨架段 / 紧凑块用轮廓多边形
blobs = []
nl, lab, st, _ = cv2.connectedComponentsWithStats(wide, connectivity=8)
for k in range(1, nl):
    if st[k, cv2.CC_STAT_AREA] * px * px < 400:
        continue
    comp = lab == k
    hv = hm[comp]
    lv_hist = np.bincount(np.clip((hv - hv.min()).astype(int), 0, None))
    lv_hist = np.convolve(lv_hist, np.ones(5), 'same')
    lvls = []
    for b in np.argsort(-lv_hist):
        if lv_hist[b] < 0.12 * comp.sum():
            break
        z = hv.min() + b + 0.5
        if all(abs(z - l) > 10 for l in lvls):
            lvls.append(z)
    lvls.sort()
    if not lvls:
        lvls = [float(np.median(hv))]
    # 每个像素归到最近的层
    which = np.argmin(np.abs(hv[:, None] - np.array(lvls)[None, :]), 1)
    rr, cc = np.nonzero(comp)
    for li, z in enumerate(lvls):
        sub = np.zeros_like(wide); sub[rr[which == li], cc[which == li]] = 1
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for k2 in range(1, n2):
            if st2[k2, cv2.CC_STAT_AREA] * px * px < 400:
                continue
            m2 = (lab2 == k2).astype(np.uint8)
            cs, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_ = max(cs, key=cv2.contourArea)
            (rcx, rcy), (rw_, rh_), _ = cv2.minAreaRect(cnt_)
            fill = cv2.contourArea(cnt_) / max(rw_ * rh_, 1)
            Lr, Wr = max(rw_, rh_) * px, min(rw_, rh_) * px
            if fill > 0.8 and Lr > 2.5 * Wr:
                ss = skeleton_segments(m2, min_len_mm=max(20.0, Wr))
                if ss:
                    best = max(ss, key=lambda s: s["s1"] - s["s0"])
                    best["w"] = Wr; best["src"] = "wide"; best["level"] = float(z)
                    segs.append(best); continue
            poly = cv2.approxPolyDP(cnt_, max(2.0, 2.5 / px), True)[:, 0, :]
            blobs.append(dict(poly=to_mm(poly[:, 1], poly[:, 0]), top=float(z), mask_px=int(m2.sum())))
LOG["raw_segments"] = len(segs); LOG["blobs"] = len(blobs)
tick("wide_blocks")

# ---------------------------------------------------------------- 8 用原始点拟合每条段
XY = Q[:, :2]
tree2 = cKDTree(XY)


def fit_strip(s):
    c, d = s["c"], s["d"]; nrm = np.array([-d[1], d[0]])
    L = s["s1"] - s["s0"]; mid = c + d * (s["s0"] + s["s1"]) / 2
    half = s["w"] / 2 + 6
    idx = tree2.query_ball_point(mid, np.hypot(L / 2 + 20, half))
    if len(idx) < 30:
        return None
    idx = np.asarray(idx); xy = XY[idx]; h = H[idx]
    sa = (xy - c) @ d; va = (xy - c) @ nrm
    hlo = args.h_min / 2 if has_base else -1e9
    sel = (np.abs(va) < half) & (sa > s["s0"] - 15) & (sa < s["s1"] + 15) & (h > hlo)
    if "level" in s:
        sel &= h > s["level"] - 15
    if sel.sum() < 30:
        return None
    sa, va, h, xyc = sa[sel], va[sel], h[sel], xy[sel]
    # 顶高剖面（5mm 一格取 P95）
    b = np.floor((sa - sa.min()) / 5).astype(int)
    nb_ = b.max() + 1
    top = np.full(nb_, np.nan)
    order = np.lexsort((h, b)); bs = b[order]; hs_ = h[order]
    starts = np.searchsorted(bs, np.arange(nb_)); ends = np.searchsorted(bs, np.arange(nb_), 'right')
    for i in range(nb_):
        if ends[i] - starts[i] >= 3:
            top[i] = hs_[starts[i] + int(0.95 * (ends[i] - starts[i] - 1))]
    ok = ~np.isnan(top)
    if ok.sum() < 2:
        return None
    topi = np.interp(np.arange(nb_), np.nonzero(ok)[0], top[ok])
    tsel = h > topi[b] - 4
    # 方向精修（顶面条带点）
    pts2 = xyc[tsel]
    if len(pts2) >= 20:
        cc = pts2.mean(0); w_, v_ = np.linalg.eigh(np.cov((pts2 - cc).T)); d2 = v_[:, 1]
        if d2 @ d < 0:
            d2 = -d2
        if np.degrees(np.arccos(np.clip(d2 @ d, -1, 1))) < 5:
            d, c = d2, cc; nrm = np.array([-d[1], d[0]])
    sa = (xyc - c) @ d; va = (xyc - c) @ nrm
    vt = va[tsel]
    if len(vt) < 10:
        return None
    lo_, hi_ = np.percentile(vt, [2, 98])
    width = float(hi_ - lo_); voff = float((lo_ + hi_) / 2)
    st_ = sa[tsel]
    s0, s1 = np.percentile(st_, [0.3, 99.7])
    # 剖面简化
    sb = sa.min() + (np.arange(nb_) + 0.5) * 5
    msk = (sb >= s0 - 3) & (sb <= s1 + 3)
    prof = np.c_[sb[msk], topi[msk]]
    if len(prof) < 2:
        prof = np.array([[s0, np.median(h[tsel])], [s1, np.median(h[tsel])]])
    if prof[:, 1].max() - prof[:, 1].min() < 6:
        prof = np.array([[s0, np.median(prof[:, 1])], [s1, np.median(prof[:, 1])]])
    else:
        kk = dp_split(prof, 3.0); prof = prof[kk]
        prof[0, 0], prof[-1, 0] = s0, s1
    bottom = 0.0 if has_base else float(np.percentile(H[tree2.query_ball_point(c + d * (s0 + s1) / 2, np.hypot((s1 - s0) / 2, width / 2 + 4))], 1))
    return dict(c=c + nrm * voff, d=d, s0=float(s0), s1=float(s1), w=width, prof=prof, bottom=bottom,
                n_top=int(tsel.sum()), src=s["src"])


plates = [p for p in (fit_strip(s) for s in segs) if p is not None and p["s1"] - p["s0"] > 15 and p["w"] > 2.5]
LOG["fitted"] = len(plates)
tick("fit")


# ---------------------------------------------------------------- 9 共线合并（遮挡桥接）
def gap_has_base(p, a, b):
    if not has_base:
        return 0.0
    ss = np.arange(a, b, px)
    if len(ss) == 0:
        return 0.0
    vv = np.linspace(-p["w"] / 2, p["w"] / 2, 3)
    pts = (p["c"][None, None, :] + ss[:, None, None] * p["d"][None, None, :] + vv[None, :, None] * np.array([-p["d"][1], p["d"][0]])[None, None, :]).reshape(-1, 2)
    c = ((pts[:, 0] - x0) / px).astype(int).clip(0, Wd - 1); r = ((pts[:, 1] - y0) / px).astype(int).clip(0, Ht - 1)
    return float(base_vis[r, c].mean())


changed = True
while changed:
    changed = False
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            A, B = plates[i], plates[j]
            if abs(A["d"] @ B["d"]) < np.cos(np.radians(4)):
                continue
            dB = B["d"] if A["d"] @ B["d"] > 0 else -B["d"]
            nA = np.array([-A["d"][1], A["d"][0]])
            eB = [B["c"] + dB * B["s0"] * (1 if dB is B["d"] else -1), B["c"] + B["d"] * B["s1"]]
            eB = [B["c"] + B["d"] * B["s0"], B["c"] + B["d"] * B["s1"]]
            if max(abs((e - A["c"]) @ nA) for e in eB) > max(3.0, 0.5 * max(A["w"], B["w"])):
                continue
            if max(A["w"], B["w"]) > 1.8 * min(A["w"], B["w"]) + 2:
                continue
            sb = sorted(((e - A["c"]) @ A["d"]) for e in eB)
            gap_lo, gap_hi = (A["s1"], sb[0]) if sb[0] >= A["s1"] else ((sb[1], A["s0"]) if sb[1] <= A["s0"] else (0, 0))
            g = gap_hi - gap_lo
            if g > 400:
                continue
            if g > 15 and gap_has_base(A, gap_lo, gap_hi) > 0.3:
                continue
            # 合并: 端点取并集，宽度按顶点数加权，剖面拼接
            s0 = min(A["s0"], sb[0]); s1 = max(A["s1"], sb[1])
            wA, wB = A["n_top"], B["n_top"]
            profB = np.c_[[((B["c"] + B["d"] * s) - A["c"]) @ A["d"] for s in B["prof"][:, 0]], B["prof"][:, 1]]
            prof = np.vstack([A["prof"], profB]); prof = prof[np.argsort(prof[:, 0])]
            A.update(s0=float(s0), s1=float(s1), w=(A["w"] * wA + B["w"] * wB) / (wA + wB), n_top=wA + wB, prof=prof,
                     bottom=min(A["bottom"], B["bottom"]))
            plates.pop(j); changed = True
            break
        if changed:
            break
for p in plates:
    pr = p["prof"]
    if pr[:, 1].max() - pr[:, 1].min() < 6:
        p["prof"] = np.array([[p["s0"], np.median(pr[:, 1])], [p["s1"], np.median(pr[:, 1])]])
    else:
        kk = dp_split(pr, 3.0); p["prof"] = pr[kk]; p["prof"][0, 0], p["prof"][-1, 0] = p["s0"], p["s1"]
LOG["merged"] = len(plates)
tick("merge")

# ---------------------------------------------------------------- 10 端点接头（交线定端点）
for A in plates:
    nA = np.array([-A["d"][1], A["d"][0]])
    for end in ("s0", "s1"):
        e = A["c"] + A["d"] * A[end]
        best = None
        for B in plates:
            if B is A:
                continue
            cosang = abs(A["d"] @ B["d"])
            if cosang > np.cos(np.radians(20)):
                continue
            nB = np.array([-B["d"][1], B["d"][0]])
            # A 中线与 B 中面的交点
            den = A["d"] @ nB
            sA = ((B["c"] - A["c"]) @ nB) / den
            hit = A["c"] + A["d"] * sA
            sB = (hit - B["c"]) @ B["d"]
            if sB < B["s0"] - 10 or sB > B["s1"] + 10:
                continue
            off = (B["w"] / 2) / abs(den)
            if abs(sA - A[end]) > off + A["w"] / 2 + 15:
                continue
            if best is None or abs(sA - A[end]) < abs(best[0] - A[end]):
                best = (sA, off, B)
        if best is None:
            continue
        sA, off, B = best
        inside = (best[2]["s0"] + 12 < (A["c"] + A["d"] * sA - B["c"]) @ B["d"] < best[2]["s1"] - 12)
        sign = -1 if end == "s1" else 1
        if inside or (B["s1"] - B["s0"]) >= (A["s1"] - A["s0"]):
            A[end] = float(sA + sign * off)          # T 型 / L 型短板: 顶到 B 的近侧面
        else:
            A[end] = float(sA - sign * off)          # L 型长板: 伸到 B 的外侧面
    pr = A["prof"]; pr[0, 0], pr[-1, 0] = A["s0"], A["s1"]
    if len(pr) > 2:
        pr = pr[(pr[:, 0] >= A["s0"]) & (pr[:, 0] <= A["s1"])]
        if len(pr) < 2 or pr[0, 0] > A["s0"] or pr[-1, 0] < A["s1"]:
            zz = np.interp([A["s0"], A["s1"]], A["prof"][:, 0], A["prof"][:, 1])
            pr = np.vstack([[A["s0"], zz[0]], pr[(pr[:, 0] > A["s0"]) & (pr[:, 0] < A["s1"])], [A["s1"], zz[1]]])
        A["prof"] = pr
tick("junctions")

# ---------------------------------------------------------------- 11 底板轮廓
base_poly = None
if has_base:
    occb = (base_vis | (raised > 0)).astype(np.uint8)
    occb = cv2.morphologyEx(occb, cv2.MORPH_CLOSE, disk(6.0 / px))
    occb = ndimage.binary_fill_holes(occb).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(occb, connectivity=8)
    if nl > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_ = max(cs, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt_, max(2.0, 3.0 / px), True)[:, 0, :]
        base_poly = to_mm(poly[:, 1], poly[:, 0])
tick("base_outline")


# ---------------------------------------------------------------- 12 实体（工件坐标系） -> 回原坐标
def to_world(Qp):
    return Qp @ E + o


solids = []
for k, p in enumerate(plates):
    nrm = np.array([-p["d"][1], p["d"][0]])
    pr = p["prof"]
    ring = [(p["s0"], p["bottom"]), (p["s1"], p["bottom"])] + [(s, z) for s, z in pr[::-1]]
    poly = []
    for s, z in ring:
        if poly and np.hypot(s - poly[-1][0], z - poly[-1][1]) < 0.5:
            continue
        poly.append((s, z))
    base0 = p["c"] - nrm * p["w"] / 2
    pts3 = np.array([[*(base0 + p["d"] * s), z] for s, z in poly])
    ext = np.array([*(nrm * p["w"]), 0.0])
    solids.append(dict(name="plate_%02d" % k, kind="strip_" + p["src"], ring=to_world(pts3), vec=ext @ E,
                       t=round(p["w"], 2), L=round(p["s1"] - p["s0"], 1), top=round(float(pr[:, 1].max()), 1)))
for k, b in enumerate(blobs):
    bot = 0.0 if has_base else float(np.percentile(H[tree2.query_ball_point(b["poly"].mean(0), np.ptp(b["poly"], 0).max() / 2 + 4)], 1))
    pts3 = np.c_[b["poly"], np.full(len(b["poly"]), bot)]
    solids.append(dict(name="block_%02d" % k, kind="block", ring=to_world(pts3), vec=np.array([0, 0, b["top"] - bot]) @ E,
                       t=round(b["top"] - bot, 1), L=None, top=round(b["top"], 1)))
if base_poly is not None:
    pts3 = np.c_[base_poly, np.full(len(base_poly), -args.base_t)]
    solids.append(dict(name="base", kind="base", ring=to_world(pts3), vec=np.array([0, 0, args.base_t]) @ E,
                       t=args.base_t, L=None, top=0.0))
tick("solids")

# ---------------------------------------------------------------- 13 STEP
if not args.no_step:
    if args.ocp:
        sys.path.insert(0, args.ocp)
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp); nok = 0
    for s in solids:
        try:
            mp = BRepBuilderAPI_MakePolygon()
            for x, y, z in s["ring"]:
                mp.Add(gp_Pnt(float(x), float(y), float(z)))
            mp.Close()
            f = BRepBuilderAPI_MakeFace(mp.Wire(), True)
            if not f.IsDone():
                s["error"] = "face"; continue
            pr = BRepPrimAPI_MakePrism(f.Face(), gp_Vec(*map(float, s["vec"])))
            bb.Add(comp, pr.Shape()); nok += 1
        except Exception as ex:
            s["error"] = str(ex)[:80]
    w = STEPControl_Writer(); w.Transfer(comp, STEPControl_AsIs); w.Write(os.path.join(args.out, "model.step"))
    LOG["step_solids"] = nok
tick("step")

# ---------------------------------------------------------------- 14 输出 + 调试图
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
    nrm = np.array([-p["d"][1], p["d"][0]])
    a, b = p["c"] + p["d"] * p["s0"], p["c"] + p["d"] * p["s1"]
    quad = np.array([a - nrm * p["w"] / 2, b - nrm * p["w"] / 2, b + nrm * p["w"] / 2, a + nrm * p["w"] / 2])
    cv2.polylines(img, [mmpix(quad)], True, (0, 0, 255) if p["src"] == "thin" else (255, 128, 0), max(1, int(1.5 / px)))
for b in blobs:
    cv2.polylines(img, [mmpix(b["poly"])], True, (0, 200, 255), max(1, int(1.5 / px)))
if base_poly is not None:
    cv2.polylines(img, [mmpix(base_poly)], True, (0, 255, 0), max(1, int(2 / px)))
img = img[::-1]  # y 向上
sc = min(1.0, 2400.0 / max(img.shape[:2]))
if sc < 1:
    img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(args.out, "debug_top.png"), img)
LOG["time_s"] = TM; LOG["total_s"] = round(sum(TM.values()), 2)
LOG["solids"] = [dict(name=s["name"], kind=s["kind"], t=s["t"], L=s["L"], top=s["top"], err=s.get("error")) for s in solids]
json.dump(LOG, open(os.path.join(args.out, "parts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(has_base=has_base, segs=LOG["raw_segments"], fitted=LOG["fitted"], merged=LOG["merged"], blobs=len(blobs),
                      solids=len(solids), step=LOG.get("step_solids"), total_s=LOG["total_s"], base=LOG.get("base_plane"), t=TM)))
