"""最小 AP214 STEP 写出器（纯文本，不依赖 OCCT）——C++ 版 step_writer 的对照实现。
每个实体 = 平面多边形 ring(n x 3, 首尾不重复) 沿向量 vec 拉伸成棱柱；输出多个 MANIFOLD_SOLID_BREP。
用法(自测): python -I step_writer.py <out.step> [OCP_deps]   -> 写 4 个样例实体，若给 OCP 路径则读回校验体积/拓扑。
"""
import sys
import numpy as np


def _clean_ring(ring, eps=1e-3):
    """去掉重复点和共线点；返回 None 表示退化。"""
    P = [np.asarray(p, float) for p in ring]
    out = []
    for p in P:
        if not out or np.linalg.norm(p - out[-1]) > eps:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
        out.pop()
    changed = True
    while changed and len(out) >= 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            if np.linalg.norm(np.cross(b - a, c - b)) <= eps * max(np.linalg.norm(c - a), eps):
                out.pop(i); changed = True; break
    return np.array(out) if len(out) >= 3 else None


def _newell(R):
    n = np.zeros(3)
    for i in range(len(R)):
        a, b = R[i], R[(i + 1) % len(R)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    return n


def _f(x):
    s = "%.6f" % float(x)
    return "0." if s in ("0.000000", "-0.000000") else s.rstrip("0")


class _W:
    def __init__(self):
        self.lines = []; self.n = 0

    def add(self, txt):
        self.n += 1; self.lines.append("#%d=%s;" % (self.n, txt)); return self.n

    def pt(self, p):
        return self.add("CARTESIAN_POINT('',(%s,%s,%s))" % tuple(_f(v) for v in p))

    def dir(self, d):
        d = np.asarray(d, float); d = d / np.linalg.norm(d)
        return self.add("DIRECTION('',(%s,%s,%s))" % tuple(_f(v) for v in d))

    def axis(self, p, nrm, ref):
        return self.add("AXIS2_PLACEMENT_3D('',#%d,#%d,#%d)" % (self.pt(p), self.dir(nrm), self.dir(ref)))


def prism_ok(ring, vec):
    R = _clean_ring(ring)
    if R is None:
        return False
    nr = _newell(R); a2 = np.linalg.norm(nr)
    return a2 > 1e-6 and abs(np.asarray(vec, float) @ (nr / a2)) > 1e-3


def write_step(path, solids, product="plate_model"):
    """solids: [(name, ring(n,3), vec(3,))]。返回写出的实体数。"""
    w = _W()
    ctx = w.add("APPLICATION_CONTEXT('automotive_design')")
    w.add("APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#%d)" % ctx)
    pctx = w.add("PRODUCT_CONTEXT('',#%d,'mechanical')" % ctx)
    prod = w.add("PRODUCT('%s','%s','',(#%d))" % (product, product, pctx))
    w.add("PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#%d))" % prod)
    pdf = w.add("PRODUCT_DEFINITION_FORMATION('','',#%d)" % prod)
    pdc = w.add("PRODUCT_DEFINITION_CONTEXT('part definition',#%d,'design')" % ctx)
    pd = w.add("PRODUCT_DEFINITION('design','',#%d,#%d)" % (pdf, pdc))
    pds = w.add("PRODUCT_DEFINITION_SHAPE('','',#%d)" % pd)
    lu = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add("UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-03),#%d,'distance_accuracy_value','confusion accuracy')" % lu)
    gctx = w.add("(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#%d))"
                 "GLOBAL_UNIT_ASSIGNED_CONTEXT((#%d,#%d,#%d))REPRESENTATION_CONTEXT('Context3D','3D Context'))" % (unc, lu, au, su))
    items = [w.axis((0, 0, 0), (0, 0, 1), (1, 0, 0))]
    for name, ring, vec in solids:
        R = _clean_ring(ring); v = np.asarray(vec, float)
        if R is None:
            continue
        nr = _newell(R)
        if np.linalg.norm(nr) < 1e-6 or abs(v @ nr) / np.linalg.norm(nr) < 1e-3:
            continue
        if v @ nr < 0:                      # 统一成: 底面多边形绕 vec 逆时针
            R = R[::-1]; nr = -nr
        nr = nr / np.linalg.norm(nr); n = len(R); T = R + v
        vb = [w.add("VERTEX_POINT('',#%d)" % w.pt(p)) for p in R]
        vt = [w.add("VERTEX_POINT('',#%d)" % w.pt(p)) for p in T]

        def edge(pa, pb, va, vb_):
            d = pb - pa; L = np.linalg.norm(d)
            ln = w.add("LINE('',#%d,#%d)" % (w.pt(pa), w.add("VECTOR('',#%d,%s)" % (w.dir(d), _f(L)))))
            return w.add("EDGE_CURVE('',#%d,#%d,#%d,.T.)" % (va, vb_, ln))

        eb = [edge(R[i], R[(i + 1) % n], vb[i], vb[(i + 1) % n]) for i in range(n)]
        et = [edge(T[i], T[(i + 1) % n], vt[i], vt[(i + 1) % n]) for i in range(n)]
        ev = [edge(R[i], T[i], vb[i], vt[i]) for i in range(n)]

        def face(oriented, p0, nrm, ref):
            oe = [w.add("ORIENTED_EDGE('',*,*,#%d,%s)" % (e, ".T." if s else ".F.")) for e, s in oriented]
            loop = w.add("EDGE_LOOP('',(%s))" % ",".join("#%d" % x for x in oe))
            bnd = w.add("FACE_OUTER_BOUND('',#%d,.T.)" % loop)
            pl = w.add("PLANE('',#%d)" % w.axis(p0, nrm, ref))
            return w.add("ADVANCED_FACE('',(#%d),#%d,.T.)" % (bnd, pl))

        e0 = (R[1] - R[0]) / np.linalg.norm(R[1] - R[0])
        faces = [face([(eb[i], False) for i in range(n - 1, -1, -1)], R[0], -nr, e0),
                 face([(et[i], True) for i in range(n)], T[0], nr, e0)]
        for i in range(n):
            j = (i + 1) % n; e = R[j] - R[i]; out = np.cross(e, v)
            faces.append(face([(eb[i], True), (ev[j], True), (et[i], False), (ev[i], False)], R[i], out, e))
        shell = w.add("CLOSED_SHELL('',(%s))" % ",".join("#%d" % x for x in faces))
        items.append(w.add("MANIFOLD_SOLID_BREP('%s',#%d)" % (str(name).replace("'", ""), shell)))
    rep = w.add("ADVANCED_BREP_SHAPE_REPRESENTATION('',(%s),#%d)" % (",".join("#%d" % x for x in items), gctx))
    w.add("SHAPE_DEFINITION_REPRESENTATION(#%d,#%d)" % (pds, rep))
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('plate reconstruction'),'2;1');",
            "FILE_NAME('model.step','2026-01-01T00:00:00',(''),(''),'plate_recon','plate_recon','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(head + w.lines + ["ENDSEC;", "END-ISO-10303-21;", ""]))
    return len(items) - 1


if __name__ == "__main__":
    out = sys.argv[1]
    box = [(0, 0, 0), (100, 0, 0), (100, 10, 0), (0, 10, 0)]                       # 100x10 底, 拉 50
    ell = [(200, 0, 0), (200, 0, 80), (210, 0, 80), (210, 0, 10), (300, 0, 10), (300, 0, 0)]  # 竖直 L 形(凹), 沿 +y 厚 8
    gus = [(400, 0, 0), (520, 0, 0), (520, 0, 30), (400, 0, 120)]                   # 顶边带坡的筋板, 沿 -y 厚 12（ring 反向）
    S = [("box", box, (0, 0, 50)), ("ell_concave", ell, (0, 8, 0)), ("gusset", gus[::-1], (0, -12, 0)),
         ("tilted", [(0, 100, 0), (60, 100, 0), (60, 100, 40), (0, 100, 40)], (5, 15, 3))]
    print("written", write_step(out, S))
    if len(sys.argv) > 2:
        sys.path.insert(0, sys.argv[2])
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopoDS import TopoDS
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        Sf = lambda c, nm: getattr(c, nm + "_s", None) or getattr(c, nm)
        r = STEPControl_Reader(); st = r.ReadFile(out); r.TransferRoots(); shp = r.OneShape()
        ex = TopExp_Explorer(shp, TopAbs_SOLID); k = 0
        expect = [100 * 10 * 50, (80 * 10 + 90 * 10) * 8, (120 * 30 + 0.5 * 120 * 90) * 12, 60 * 40 * 15]
        while ex.More():
            so = Sf(TopoDS, "Solid")(ex.Current()); gp = GProp_GProps(); Sf(BRepGProp, "VolumeProperties")(so, gp)
            print("solid %d valid=%s volume=%.1f expect=%.1f" % (k, BRepCheck_Analyzer(so).IsValid(), gp.Mass(), expect[k]))
            k += 1; ex.Next()
        print("read status", st, "solids", k)
