"""Synthetic top-down scans with known answers (for unit/regression tests of plate_recon).
Each case: <case>.pcd (binary float32 xyz, sensor-like pose), <case>.step (true solids), <case>.json (truth).
Scan model: orthographic z-buffer from above on a jittered grid; optional bottom-face points (like the sim clouds).
Usage: python -I synth_gen.py <out_dir> [case ...]"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_writer import write_step  # noqa: E402


def box(name, x0, y0, x1, y1, z0, z1, kind="block", **kw):
    """axis-aligned box. truth: plate -> t = narrow side, L = long side; block/base -> t = height."""
    ring = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], float)
    dx, dy = x1 - x0, y1 - y0
    dims = dict(t=min(dx, dy), L=max(dx, dy)) if kind == "plate" else dict(t=z1 - z0)
    return dict(name=name, kind=kind, ring=ring, vec=np.array([0, 0, z1 - z0], float), top=z1, **dims, **kw)


def vplate(name, p0, p1, t, ztop0, ztop1=None, z0=0.0, **kw):
    """vertical plate, centre line p0->p1 (xy), thickness t, top linear from ztop0 (at p0) to ztop1 (at p1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); ztop1 = ztop0 if ztop1 is None else ztop1
    d = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-d[1], d[0]])
    a, b = p0 - n * t / 2, p1 - n * t / 2
    ring = np.array([[*a, z0], [*b, z0], [*b, ztop1], [*a, ztop0]])
    return dict(name=name, kind="plate", ring=ring, vec=np.array([*(n * t), 0.0]), t=t, top=max(ztop0, ztop1),
                L=float(np.linalg.norm(p1 - p0)), dir_deg=float(np.degrees(np.arctan2(d[1], d[0]))), **kw)


def lplate(name, p0, p1, t, theta_deg, H, **kw):
    """plate leaning by theta towards +n (n = left normal of p0->p1); p0/p1 = foot of the visible (upper) face."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-d[1], d[0]]); th = np.radians(theta_deg)
    sh = n * H * np.tan(th)
    ring = np.array([[*p0, 0], [*p1, 0], [*(p1 + sh), H], [*(p0 + sh), H]])
    return dict(name=name, kind="lean", ring=ring, vec=np.array([*(n * t / np.cos(th)), 0.0]), t=t, top=H, lean_deg=theta_deg,
                L=float(np.linalg.norm(p1 - p0)), **kw)


def faces(s):
    R, v = s["ring"], s["vec"]; n = len(R)
    out = [R, R + v]
    for i in range(n):
        j = (i + 1) % n
        out.append(np.array([R[i], R[j], R[j] + v, R[i] + v]))
    return out


def inside(px, py, poly):
    res = np.zeros(px.shape, bool); n = len(poly)
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[(i + 1) % n]
        if yi == yj:
            continue
        c = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        res ^= c
    return res


def scan(solids, sp, seed, bottom_frac=0.5, holes=(), margin=20.0, sigma=0.25):
    rng = np.random.default_rng(seed)
    allp = np.vstack([np.vstack(faces(s)) for s in solids])
    lo, hi = allp[:, :2].min(0) - margin, allp[:, :2].max(0) + margin
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], sp), np.arange(lo[1], hi[1], sp))
    x = gx.ravel() + rng.uniform(-0.15, 0.15, gx.size) * sp; y = gy.ravel() + rng.uniform(-0.15, 0.15, gx.size) * sp
    top = np.full(x.size, -np.inf); bot = np.full(x.size, np.inf)
    for s in solids:
        for F in faces(s):
            nr = np.zeros(3)
            for i in range(len(F)):
                a, b = F[i], F[(i + 1) % len(F)]
                nr += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
            if abs(nr[2]) < 1e-9 * max(np.linalg.norm(nr), 1e-12):
                continue
            mn, mx = F[:, :2].min(0), F[:, :2].max(0)
            cand = np.nonzero((x >= mn[0]) & (x <= mx[0]) & (y >= mn[1]) & (y <= mx[1]))[0]
            if cand.size == 0:
                continue
            m = inside(x[cand], y[cand], F[:, :2]); idx = cand[m]
            z = (nr @ F[0] - nr[0] * x[idx] - nr[1] * y[idx]) / nr[2]
            top[idx] = np.maximum(top[idx], z)
            if s.get("see_bottom"):
                bot[idx] = np.minimum(bot[idx], z)
    keep = np.isfinite(top)
    for (hx0, hy0, hx1, hy1) in holes:                       # occlusion shadow: no data at all
        keep &= ~((x >= hx0) & (x <= hx1) & (y >= hy0) & (y <= hy1))
    pts = [np.c_[x[keep], y[keep], top[keep] + rng.normal(0, sigma, keep.sum())]]
    kb = np.isfinite(bot) & (bot < top - 1.0) & keep & (rng.uniform(0, 1, x.size) < bottom_frac)
    if kb.any():
        pts.append(np.c_[x[kb], y[kb], bot[kb] + rng.normal(0, sigma, kb.sum())])
    return np.vstack(pts)


def pose(yaw_deg, tilt_deg, t):
    a, b = np.radians(yaw_deg), np.radians(tilt_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Rz, np.asarray(t, float)


def write_pcd(path, P):
    P = P.astype(np.float32); n = len(P)
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (n, n))
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii")); fh.write(P.tobytes())


def case(name):
    base = lambda x1, y1, t=20.0, **kw: box("base", 0, 0, x1, y1, -t, 0, kind="base", **kw)
    if name == "S1_tee":            # T joint on a base: B abuts A's side face
        S = [base(600, 400), vplate("A", (50, 200), (550, 200), 8, 100), vplate("B", (300, 204), (300, 380), 10, 80)]
        return S, dict(sp=0.6, holes=(), pose=(7, 1.0, (100, -50, -400)), has_base=True)
    if name == "S2_oblique":        # plate at 30 deg with a sloped top + perpendicular plate abutting it
        d = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]); n = np.array([-d[1], d[0]])
        p0 = np.array([150.0, 150.0]); p1 = p0 + d * 500; pm = p0 + d * 250
        S = [base(800, 600), vplate("C", p0, p1, 6, 60, 140), vplate("D", pm + n * 3, pm + n * 203, 8, 90)]
        return S, dict(sp=0.6, holes=(), pose=(-12, 0.8, (-300, 80, -350)), has_base=True)
    if name == "S3_truss_nobase":   # no base: wide chord + lower wide web crossing under it; bottoms visible (sim-like)
        S = [box("chord", 0, 0, 1000, 60, 0, 85, kind="plate", see_bottom=True),
             box("web", 485, -170, 515, 230, 0, 50, kind="plate", see_bottom=True)]
        return S, dict(sp=0.6, holes=(), pose=(3, 0.0, (0, 0, -600)), has_base=False)
    if name == "S4_lean":           # one plate leaning 10 deg + one vertical plate (must stay vertical)
        S = [base(600, 400), lplate("L", (100, 120), (500, 120), 10, 10, 120), vplate("V", (100, 300), (500, 300), 8, 100)]
        return S, dict(sp=0.6, holes=(), pose=(-5, 0.5, (50, 50, -380)), has_base=True)
    if name == "S5_sparse_grid":    # SS30-like: 10 mm base with visible bottom, sparse points, '#' rib grid
        S = [base(800, 600, 10, see_bottom=True),
             vplate("X1", (50, 200), (750, 200), 8, 100), vplate("X2", (50, 400), (750, 400), 8, 100),
             vplate("Y1", (250, 50), (250, 550), 8, 100), vplate("Y2", (550, 50), (550, 550), 8, 100)]
        return S, dict(sp=1.5, holes=(), pose=(0, 0.0, (0, 0, -500)), has_base=True, bottom_frac=0.25)
    if name == "S6_gap":            # E: 100 mm no-data shadow (occlusion, must merge); F: real 100 mm gap (base visible, two plates)
        S = [base(800, 300), vplate("E", (50, 80), (750, 80), 8, 60),
             vplate("F1", (50, 220), (350, 220), 8, 60), vplate("F2", (450, 220), (750, 220), 8, 60)]
        return S, dict(sp=0.6, holes=((350, 40, 450, 120),), pose=(10, 0.6, (200, 0, -420)), has_base=True)
    if name == "S7_block":          # thick block + thin plate
        S = [base(600, 400), box("K", 100, 100, 200, 180, 0, 40), vplate("P", (100, 300), (500, 300), 6, 70)]
        return S, dict(sp=0.6, holes=(), pose=(20, 1.2, (-100, -100, -450)), has_base=True)
    raise SystemExit("unknown case " + name)


CASES = ["S1_tee", "S2_oblique", "S3_truss_nobase", "S4_lean", "S5_sparse_grid", "S6_gap", "S7_block"]

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for nm in (sys.argv[2:] or CASES):
        S, cfg = case(nm)
        P = scan(S, cfg["sp"], seed=CASES.index(nm), bottom_frac=cfg.get("bottom_frac", 0.5), holes=cfg["holes"])
        Rm, tr = pose(*cfg["pose"][:2], cfg["pose"][2])
        Pw = P @ Rm.T + tr
        write_pcd(os.path.join(out, nm + ".pcd"), Pw)
        write_step(os.path.join(out, nm + ".step"), [(s["name"], s["ring"] @ Rm.T + tr, s["vec"] @ Rm.T) for s in S])
        truth = dict(case=nm, points=int(len(Pw)), spacing_mm=cfg["sp"], has_base=cfg["has_base"],
                     solids=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("ring", "vec")} for s in S])
        json.dump(truth, open(os.path.join(out, nm + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(nm, len(Pw))
