"""探针：降采样 -> 法向 -> 法向分组 + 偏移分层 + 连通分割，统计耗时、平面数和有无底板判断。
用法: python -I probe_planes.py <out.json> <pcd> [<pcd> ...]
"""
import sys, os, time, json
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz

VOX, CELL, MIN_PTS = 3.0, 10.0, 100


def voxel_down(p, s):
    k = np.floor((p - p.min(0)) / s).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.empty((len(cnt), 3))
    for d in range(3):
        out[:, d] = np.bincount(inv, p[:, d], len(cnt)) / cnt
    return out


def normals(p, k=12, chunk=200000):
    tree = cKDTree(p)
    n = np.empty_like(p); curv = np.empty(len(p))
    for s in range(0, len(p), chunk):
        _, idx = tree.query(p[s:s + chunk], k, workers=-1)
        nb = p[idx]
        c = nb - nb.mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', c, c))
        n[s:s + chunk] = v[:, :, 0]
        curv[s:s + chunk] = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    return n, curv


def datum_frame(n, curv):
    m = n[curv < 0.01]
    m = m * np.where(m[:, 2:3] < 0, -1.0, 1.0)
    z = np.array([0, 0, 1.0])
    for ang in (30, 15, 8):
        z = m[m @ z > np.cos(np.radians(ang))].mean(0); z /= np.linalg.norm(z)
    x = np.cross([0, 1.0, 0], z); x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def direction_groups(m, curv):
    planar = curv < 0.02
    hz = planar & (np.abs(m[:, 2]) > np.cos(np.radians(10)))
    groups = [('H', np.array([0, 0, 1.0]), hz)]
    vt = planar & (np.abs(m[:, 2]) < np.sin(np.radians(10)))
    idx = np.where(vt)[0]
    az = np.degrees(np.arctan2(m[vt, 1], m[vt, 0])) % 180
    taken = np.zeros(len(az), bool)
    while (~taken).sum() >= MIN_PTS:
        h = np.bincount(np.floor(az[~taken]).astype(int) % 180, minlength=180)
        hs = sum(np.roll(h, s) for s in range(-2, 3))
        pk = int(np.argmax(hs))
        if hs[pk] < MIN_PTS:
            break
        rel = (az - (pk + 0.5) + 90) % 180 - 90
        sel = ~taken & (np.abs(rel) < 5)
        a = np.radians(pk + 0.5 + np.median(rel[sel]))
        mask = np.zeros(len(m), bool); mask[idx[sel]] = True
        groups.append(('V', np.array([np.cos(a), np.sin(a), 0.0]), mask))
        taken |= sel
    oblique = int((planar & ~hz & ~vt).sum())
    return groups, oblique


def patches_of(q, groups):
    out = []
    for gi, (kind, d, mask) in enumerate(groups):
        P = q[mask]
        if len(P) < MIN_PTS:
            continue
        o = P @ d
        b = np.floor(o - o.min()).astype(int)
        hs = np.convolve(np.bincount(b), [1, 1, 1], 'same')
        runs, _ = ndimage.label(hs >= 30)
        lab = runs[b]
        U = np.array([1, 0, 0.]) if kind == 'H' else np.cross([0, 0, 1.], d)
        V = np.array([0, 1, 0.]) if kind == 'H' else np.array([0, 0, 1.])
        for L in np.unique(lab[lab > 0]):
            S = P[lab == L]
            if len(S) < MIN_PTS:
                continue
            uv = np.c_[S @ U, S @ V]
            ij = np.floor((uv - uv.min(0)) / CELL).astype(int)
            g = np.zeros(ij.max(0) + 1, bool); g[ij[:, 0], ij[:, 1]] = True
            gl, _ = ndimage.label(g, structure=np.ones((3, 3)))
            cells = np.bincount(gl.ravel())
            pl = gl[ij[:, 0], ij[:, 1]]
            for c in np.unique(pl):
                T = S[pl == c]
                if len(T) < MIN_PTS:
                    continue
                w = np.linalg.eigvalsh(np.cov((T - T.mean(0)).T))
                out.append(dict(kind=kind, carrier="%d_%d" % (gi, L), n=int(len(T)), off=float(np.median(T @ d)),
                                rms=float(np.sqrt(max(w[0], 0))), area_m2=float(cells[c] * CELL ** 2 / 1e6)))
    return out


res = []
for path in sys.argv[2:]:
    r = {"cloud": os.path.basename(path)}
    t = time.time(); p = read_xyz(path); r["t_read"] = time.time() - t; r["n_raw"] = len(p)
    t = time.time()
    ij0 = np.floor((p[:, :2] - p[:, :2].min(0)) / CELL).astype(np.int64)
    area0 = len(np.unique(ij0[:, 0] * 1000000 + ij0[:, 1])) * CELL ** 2
    vox = max(VOX, float(np.sqrt(area0 / 3e5))); r["voxel_mm"] = vox
    v = voxel_down(p, vox); r["t_voxel"] = time.time() - t; r["n_vox"] = len(v)
    t = time.time(); n, curv = normals(v); r["t_normal"] = time.time() - t
    t = time.time()
    R = datum_frame(n, curv); q = v @ R.T; m = n @ R.T
    groups, r["oblique_pts"] = direction_groups(m, curv)
    pt = patches_of(q, groups)
    r["t_segment"] = time.time() - t
    r["planes_H"] = sum(x["kind"] == "H" for x in pt); r["planes_V"] = sum(x["kind"] == "V" for x in pt)
    r["V_directions"] = len(groups) - 1
    r["coverage"] = sum(x["n"] for x in pt) / len(v)
    r["V_pts_frac"] = sum(x["n"] for x in pt if x["kind"] == "V") / len(v)
    r["rms_median"] = float(np.median([x["rms"] for x in pt])) if pt else None
    H = [x for x in pt if x["kind"] == "H"]
    ij = np.floor((q[:, :2] - q[:, :2].min(0)) / CELL).astype(int)
    foot = len(np.unique(ij[:, 0] * 100000 + ij[:, 1])) * CELL ** 2 / 1e6
    car = {}
    for x in pt:
        c = car.setdefault(x["carrier"], dict(kind=x["kind"], area_m2=0.0, n=0, off=x["off"]))
        c["area_m2"] += x["area_m2"]; c["n"] += x["n"]
    r["carriers_H"] = sum(c["kind"] == "H" for c in car.values())
    r["carriers_V"] = sum(c["kind"] == "V" for c in car.values())
    r["patch_largest_H_ratio"] = max(x["area_m2"] for x in H) / foot if H else 0
    H = [c for c in car.values() if c["kind"] == "H"]
    if H:
        big = max(H, key=lambda x: x["area_m2"])
        up = (q[:, 2] > big["off"] + 5).mean(); dn = (q[:, 2] < big["off"] - 5).mean()
        r["base_area_ratio"] = big["area_m2"] / foot
        r["minor_side_frac"] = float(min(up, dn))
        r["has_base"] = bool(r["base_area_ratio"] > 0.4 and r["minor_side_frac"] < 0.05)
    else:
        r["has_base"] = False
    r["footprint_m2"] = foot
    from scipy.spatial import ConvexHull
    hull_all = ConvexHull(q[:, :2]).volume / 1e6
    r["H_planes"] = []
    for key, c in car.items():
        if c["kind"] != "H" or c["n"] < 1000:
            continue
        sel = np.zeros(len(q), bool)
        gi, L = map(int, key.split("_"))
        Pm = np.abs(q[:, 2] - c["off"]) < 1.5
        hull_c = ConvexHull(q[Pm, :2]).volume / 1e6
        r["H_planes"].append(dict(off=round(c["off"], 1), area=round(c["area_m2"], 3),
                                  solidity=round(c["area_m2"] / hull_c, 2), cover_hull=round(hull_c / hull_all, 2),
                                  camera_side=round(float((q[:, 2] > c["off"] + 5).mean()), 2),
                                  far_side=round(float((q[:, 2] < c["off"] - 5).mean()), 2)))
    r["H_planes"].sort(key=lambda x: x["off"])
    r["t_total_after_read"] = r["t_voxel"] + r["t_normal"] + r["t_segment"]
    r = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    res.append(r)
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
