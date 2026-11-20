"""原型公共函数（Python 参考实现，供 C++ 版对照）。
detect_frame: 上方向 / 判底板 / 底板平面 / 工件坐标系；heightmap: 高度图；skeleton_segments: 骨架 -> 直线段。
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import cv2, sys


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
    """Douglas-Peucker：返回保留点下标。pts: (n,2) 有序。"""
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


def _key(a):
    return a[:, 0] * 10 ** 7 + a[:, 1]


def _ransac_up(H, up, tol, iters=300, max_dev_deg=5.0, seed=0):
    """近水平点里找内点最多的平面，返回其法向（朝 +up）。用于把"法向均值"得到的上方向精修到 0.01° 级。"""
    if len(H) < 50:
        return up, 0
    rng = np.random.default_rng(seed); best, bc = None, 0
    Hs = H if len(H) <= 60000 else H[rng.choice(len(H), 60000, replace=False)]
    for _ in range(iters):
        a, b, c = Hs[rng.choice(len(Hs), 3, replace=False)]
        n = np.cross(b - a, c - a); ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if abs(n @ up) < np.cos(np.radians(max_dev_deg)):
            continue
        cnt = int((np.abs((Hs - a) @ n) < tol).sum())
        if cnt > bc:
            best, bc = (n, a), cnt
    if best is None:
        return up, 0
    n, a = best
    inl = np.abs((H - a) @ n) < tol
    n2, _ = fit_plane(H[inl])
    if n2 @ up < 0:
        n2 = -n2
    return n2, int(inl.sum())


def frame_from_up(up):
    a0 = np.cross([0, 1.0, 0], up); a0 /= np.linalg.norm(a0)
    return np.vstack([a0, np.cross(up, a0), up])


def grid_top(q, cell, pct=95):
    """10mm 网格顶面: 每格取高度 P95（比最大值抗飞点）。返回 (格坐标 ij[n,2], 顶高[n], 点数[n], 每点所属格序号)。"""
    ij = np.floor(q[:, :2] / cell).astype(np.int64)
    key = _key(ij)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.lexsort((q[:, 2], inv))
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    ztop = q[order[starts + ((cnt - 1) * pct // 100)], 2]
    return ij[order[starts]], ztop, cnt, inv


def flat_cells(cij, ztop, tol=1.5):
    """3x3 邻格都存在且顶高相差 < tol 的格（平整的水平面）。"""
    k = _key(cij); lut = dict(zip(k.tolist(), range(len(k))))
    ok = np.ones(len(k), bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            nb = np.array([lut.get(x, -1) for x in _key(cij + [a, b]).tolist()])
            ok &= (nb >= 0)
            ok[nb >= 0] &= np.abs(ztop[nb[nb >= 0]] - ztop[nb >= 0]) < tol
    return ok


def detect_frame(P, mode="auto", cell=10.0):
    """返回 dict(Q=工件坐标点, E=3x3 行向量 ex/ey/ez, o=原点, has_base, base_t, log)。有底板时 z=0 为底板顶面。
    做法: 10mm 网格顶面高度(P95) -> 平整格 -> RANSAC 最大水平面精修上方向 -> 平整格顶高直方图找水平层。
    判底板: 最低的"大"水平层(面积>=最大层25%, 实心度>0.3, 凸包占比>0.3), 且其足迹内(不是全局)下方点<3%;
    若下方点集中在 3~40mm 深处的平行层 = 底板底面被扫到(仿真点云), 也接受, 并直接量出底板厚 base_t。"""
    log = {}
    up = np.array([0, 0, 1.0])                       # 传感器 +Z 朝相机
    for _ in range(2):
        R0 = frame_from_up(up); q = P @ R0.T
        cij, ztop, cnt, inv = grid_top(q, cell)
        fl = flat_cells(cij, ztop) & (cnt >= 3)
        Hc = np.c_[(cij[fl] + 0.5) * cell, ztop[fl]]
        n_loc, n_in = _ransac_up(Hc, np.array([0, 0, 1.0]), tol=1.5)
        up = R0.T @ n_loc; up /= np.linalg.norm(up)
    log["up_tilt_from_sensor_deg"] = round(float(np.degrees(np.arccos(abs(up[2])))), 3)
    R0 = frame_from_up(up); q = P @ R0.T
    cij, ztop, cnt, inv = grid_top(q, cell)
    fl = flat_cells(cij, ztop) & (cnt >= 3)
    log["grid_cells"] = int(len(cij)); log["flat_cells"] = int(fl.sum())
    zf_ = ztop[fl]; zmin = zf_.min()
    hist = np.convolve(np.bincount(np.floor(zf_ - zmin).astype(int)), np.ones(3), 'same')
    levels = []
    for b in np.argsort(-hist):
        if hist[b] < max(30, 0.01 * fl.sum()):
            break
        z = zmin + b + 0.5
        if all(abs(z - l) > 6 for l in levels):
            levels.append(z)
    levels.sort()
    hull_all = cv2.contourArea(cv2.convexHull(cij.astype(np.float32))) + 1.0
    nb9 = np.array([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)])
    cand = []
    for z in levels:
        lc = np.abs(ztop - z) < 2.5
        if lc.sum() < 20:
            continue
        cells = cij[lc]
        area = len(cells) * cell ** 2
        hull = (cv2.contourArea(cv2.convexHull(cells.astype(np.float32))) + 0.5 * cv2.arcLength(cv2.convexHull(cells.astype(np.float32)), True) + 1.0) * cell ** 2
        foot = np.unique((cells[:, None, :] + nb9[None]).reshape(-1, 2), axis=0)
        inf = np.isin(_key(cij), _key(foot))[inv]                 # 足迹内的点
        below = inf & (q[:, 2] < z - 4)
        c = dict(z=round(float(z), 1), area_m2=round(area / 1e6, 3), solidity=round(min(1.0, area / hull), 2),
                 hull_ratio=round(hull / (hull_all * cell ** 2), 2), far=round(float(below.sum() / max(1, inf.sum())), 3))
        if c["far"] >= 0.03 and below.sum() > 200:      # 底面检测: 下方点是否集中在一个平行层
            zb = q[below, 2]
            hb = np.bincount(np.floor(z - zb).astype(int))
            hb3 = np.convolve(hb, np.ones(5), 'same'); pk = int(np.argmax(hb3))
            inpk = np.abs((z - zb) - (pk + 0.5)) < 3
            cov = len(np.unique(_key(np.floor(q[below][inpk, :2] / cell).astype(np.int64)))) / max(1, len(cells))
            if 3 <= pk + 0.5 <= 40 and inpk.mean() > 0.7 and cov > 0.3:
                c["bottom_depth"] = round(pk + 0.5, 1)
                c["far_below_bottom"] = round(float((inf & (q[:, 2] < z - pk - 0.5 - 4)).sum() / max(1, inf.sum())), 3)
        cand.append(c)
    log["h_levels"] = cand
    base_lv = None
    if cand:
        big = max(c["area_m2"] for c in cand)
        base_lv = next((c for c in cand if c["area_m2"] >= 0.25 * big), None)
    far_ok = bool(base_lv) and (base_lv["far"] < 0.03 or base_lv.get("far_below_bottom", 1.0) < 0.03)
    has_base = bool(base_lv and far_ok and base_lv["solidity"] > 0.3 and base_lv["hull_ratio"] > 0.3)
    if mode != "auto":
        has_base = (mode == "base") and base_lv is not None
    base_t = None
    if has_base:
        zl = base_lv["z"]
        bc = cij[np.abs(ztop - zl) < 2.5]
        inb = np.isin(_key(cij), _key(bc))[inv]
        near = inb & (np.abs(q[:, 2] - zl) < 4)
        Bp = P[near]
        if len(Bp) > 400000:
            Bp = Bp[np.random.default_rng(0).choice(len(Bp), 400000, replace=False)]
        n, d, keep = robust_plane(Bp)
        if n @ up < 0:
            n, d = -n, -d
        res = Bp[keep] @ n - d
        log["base_plane"] = dict(n=[round(x, 5) for x in n], tilt_deg=round(float(np.degrees(np.arccos(abs(n[2])))), 3),
                                 rms=round(float(np.sqrt(np.mean(res ** 2))), 3), inliers=int(keep.sum()))
        if "bottom_depth" in base_lv:
            rb = P[inb] @ n - d
            rb = rb[np.abs(rb + base_lv["bottom_depth"]) < 3]
            base_t = round(float(-np.median(rb)), 2)
            log["base_plane"]["bottom_visible_t"] = base_t
        ez = n; o = ez * d; base_xy = Bp[keep] - o
    else:
        ez = up; o = np.zeros(3); base_xy = P
    tmpx = np.cross([0, 1.0, 0], ez); tmpx /= np.linalg.norm(tmpx); tmpy = np.cross(ez, tmpx)
    uv = np.c_[base_xy @ tmpx, base_xy @ tmpy].astype(np.float32)
    if len(uv) > 200000:
        uv = uv[np.random.default_rng(1).choice(len(uv), 200000, replace=False)]
    _, (rw, rh), ang = cv2.minAreaRect(uv)
    th = np.radians(ang if rw >= rh else ang + 90)
    ex = np.cos(th) * tmpx + np.sin(th) * tmpy; ey = np.cross(ez, ex)
    E = np.vstack([ex, ey, ez])
    Q = (P - o) @ E.T
    if not has_base:
        zf = float(np.percentile(Q[:, 2], 0.5)); o = o + ez * zf; Q[:, 2] -= zf
    log["has_base"] = has_base
    log["frame"] = dict(origin=[round(x, 3) for x in o], ex=[round(x, 5) for x in ex], ey=[round(x, 5) for x in ey],
                        ez=[round(x, 5) for x in ez])
    return dict(Q=Q, E=E, o=o, has_base=has_base, base_t=base_t, log=log)


def heightmap(Q, px, has_base, base_tol=2.5):
    X, Y, H = Q[:, 0], Q[:, 1], Q[:, 2]
    x0, y0 = X.min() - 10 * px, Y.min() - 10 * px
    I = ((X - x0) / px).astype(np.int64); J = ((Y - y0) / px).astype(np.int64)
    W, Ht = int(I.max()) + 11, int(J.max()) + 11
    lin = J * W + I
    hm = np.full(W * Ht, -1e9); np.maximum.at(hm, lin, H); hm = hm.reshape(Ht, W)
    bv = np.zeros(W * Ht, bool)
    if has_base:
        bv[lin[np.abs(H) < base_tol]] = True
    return dict(hm=hm, occ=hm > -1e8, base_vis=bv.reshape(Ht, W), x0=x0, y0=y0, px=px, W=W, Ht=Ht)


def to_mm(hmp, r, c):
    px = hmp["px"]
    return np.c_[hmp["x0"] + (np.asarray(c) + 0.5) * px, hmp["y0"] + (np.asarray(r) + 0.5) * px]


def crossing_number(sk):
    """8 邻域按环形顺序数 0->1 跳变次数。>=3 才是真分叉；用邻居个数判分叉会把斜线的台阶拐角误当分叉。"""
    s = np.pad(sk.astype(np.uint8), 1); h, w = sk.shape
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    nb = [s[1 + a:1 + a + h, 1 + b:1 + b + w] for a, b in offs]
    return sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))


def skeleton_segments(mask, hmp, min_len_mm=20.0):
    """细化 -> 去掉分叉点 -> 每条分支按 Douglas-Peucker 断成直线段；宽度取距离变换中位数 x2。"""
    px = hmp["px"]
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    junc = sk & (crossing_number(sk) >= 3)
    br = sk & ~(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    lb, n = ndimage.label(br, structure=np.ones((3, 3)))
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    segs = []
    for k, sl in enumerate(ndimage.find_objects(lb), 1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lb[sl] == k)
        rr = rr + sl[0].start; cc = cc + sl[1].start
        if len(rr) * px < min_len_mm * 0.5:
            continue
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
        xy = to_mm(hmp, ch[:, 0], ch[:, 1])
        brk = dp_split(xy, max(2.0, 2 * px))
        for a, b in zip(brk[:-1], brk[1:]):
            seg = xy[a:b + 1]
            if len(seg) < 3:
                continue
            c = seg.mean(0); _, v_ = np.linalg.eigh(np.cov((seg - c).T)); d = v_[:, 1]
            s = (seg - c) @ d
            if s.max() - s.min() < min_len_mm:
                continue
            wpx = dt[ch[a:b + 1, 0].astype(int), ch[a:b + 1, 1].astype(int)]
            segs.append(dict(c=c, d=d, s0=float(s.min()), s1=float(s.max()), w=float(2 * np.median(wpx) * px)))
    return segs


def ring_clean(P):
    """整数像素多边形: 去掉连续重复点(含首尾)和原路折返的尖刺顶点, 反复做到没有为止。"""
    P = [(int(p[0]), int(p[1])) for p in P]
    changed = True
    while changed and len(P) >= 3:
        changed = False
        Q = [p for k, p in enumerate(P) if p != P[k - 1]]
        if len(Q) != len(P):
            P, changed = Q, True
            continue
        for i in range(len(P)):
            a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            dt = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
            if cr == 0 and dt < 0:
                del P[i]; changed = True
                break
    return P


def ring_simple(P):
    """整数坐标精确判断: >= 3 点、面积不为 0、任意两条不相邻的边既不相交也不接触。"""
    n = len(P)
    if n < 3:
        return False
    A = np.array(P, np.int64); B = np.roll(A, -1, axis=0)
    if int(np.sum(A[:, 0] * B[:, 1] - B[:, 0] * A[:, 1])) == 0:
        return False

    def orient(p, q, r):  # p, q: 一个点; r: 一组点
        return np.sign((q[0] - p[0]) * (r[:, 1] - p[1]) - (q[1] - p[1]) * (r[:, 0] - p[0]))

    def on_seg(p, q, r):
        return ((np.minimum(p[0], q[0]) <= r[:, 0]) & (r[:, 0] <= np.maximum(p[0], q[0]))
                & (np.minimum(p[1], q[1]) <= r[:, 1]) & (r[:, 1] <= np.maximum(p[1], q[1])))

    for i in range(n - 2):
        j = np.arange(i + 2, n if i > 0 else n - 1)  # 不和相邻的边比
        if len(j) == 0:
            continue
        a, b, C, D = A[i], B[i], A[j], B[j]
        o1, o2 = orient(a, b, C), orient(a, b, D)
        o3 = np.sign((D[:, 0] - C[:, 0]) * (a[1] - C[:, 1]) - (D[:, 1] - C[:, 1]) * (a[0] - C[:, 0]))
        o4 = np.sign((D[:, 0] - C[:, 0]) * (b[1] - C[:, 1]) - (D[:, 1] - C[:, 1]) * (b[0] - C[:, 0]))
        hit = (o1 * o2 < 0) & (o3 * o4 < 0)
        hit |= (o1 == 0) & on_seg(a, b, C)
        hit |= (o2 == 0) & on_seg(a, b, D)
        cmin, cmax = np.minimum(C, D), np.maximum(C, D)
        hit |= (o3 == 0) & np.all((cmin <= a) & (a <= cmax), axis=1)
        hit |= (o4 == 0) & np.all((cmin <= b) & (b <= cmax), axis=1)
        if hit.any():
            return False
    return True


def convex_hull(P):
    """Andrew 单调链, 整数坐标; 去掉共线点; 从 (x, y) 最小的点开始, 按 x-y 坐标逆时针。"""
    pts = sorted({(int(p[0]), int(p[1])) for p in P})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def simple_ring(cnt, eps, tag=None):
    """外轮廓(findContours 的输出) -> 简单多边形(整数像素坐标, n x 2 数组)。
    DP(eps) 后清理; 还不是简单多边形就用 2eps、4eps 再试; 都不行用凸包。用了后备方案时往 stderr 打一行。"""
    for lvl, f in enumerate((1, 2, 4)):
        P = ring_clean(cv2.approxPolyDP(cnt, eps * f, True)[:, 0, :])
        if ring_simple(P):
            break
    else:
        lvl, P = 3, convex_hull(cnt[:, 0, :])
    if lvl and tag:
        print("ring_fix %s level %d (%d pts)" % (tag, lvl, len(P)), file=sys.stderr)
    return np.array(P)
