"""合成样例基准: tools/synth_gen.py 生成的 7 个样例 -> 运行重建命令 -> check_step -> eval_recon(对真值 STEP) -> 对照期望值。
用法:
  python -I synth_bench.py --cmd "<程序> {pcd} {out}" --synth <synth_gen 输出目录> --ocp <OCP 目录> --out <结果目录>
         [--only S1_tee,S4_lean] [--no-seams] [--time-limit 10]
  --no-seams: 不检查 seams.csv (Python 原型不写焊缝表, 用它自测本脚本时加上)。
输出: <结果目录>/synth_summary.md, synth_summary.json; 退出码 0 = 全部通过, 1 = 有不通过。
"""
import argparse, csv, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True)
ap.add_argument("--synth", required=True)
ap.add_argument("--ocp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--no-seams", action="store_true")
ap.add_argument("--time-limit", type=float, default=10.0)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# 期望的实体: (类别, t, t容差, L, L容差, top, top容差, 倾角) ; 类别 plate = strip_*, lean = lean_*, block, base; None = 不查
P_ = lambda t, tt, L, Lt, top, tpt: ("plate", t, tt, L, Lt, top, tpt, None)
EXPECT = {
    "S1_tee": dict(has_base=True, px=(1.0,), base_t=None, rp=98, seams=(4, 2),
                   solids=[("base", 20, 0.01, None, 0, 0, 0.01, None), P_(8, 0.6, 500, 5, 100, 1), P_(10, 0.6, 176, 3, 80, 1)]),
    "S2_oblique": dict(has_base=True, px=(1.0,), base_t=None, rp=98, seams=(4, 2),
                       solids=[("base", 20, 0.01, None, 0, 0, 0.01, None), P_(6, 0.6, 500, 5, 140, 1), P_(8, 0.6, 200, 3, 90, 1)]),
    "S3_truss_nobase": dict(has_base=False, px=(1.0,), base_t=None, rp=95, seams=(0, 0),
                            solids=[P_(60, 1, 1000, 10, 85, 1.5), P_(30, 1, 400, 5, 50, 1.5)]),
    "S4_lean": dict(has_base=True, px=(1.0,), base_t=None, rp=98, seams=(4, 0),
                    solids=[("base", 20, 0.01, None, 0, 0, 0.01, None), ("lean", 10, 1, 400, 5, 120, 1, (10, 1)), P_(8, 0.6, 400, 5, 100, 1)]),
    "S5_sparse_grid": dict(has_base=True, px=(1.5,), base_t=(10, 0.5), rp=98, seams=(8, 0),
                           solids=[("base", 10, 0.5, None, 0, 0, 0.01, None)] + [P_(7.3, 1.3, 700, 5, 100, 1)] * 2 + [P_(7.3, 1.3, 500, 5, 100, 1)] * 2),
    "S6_gap": dict(has_base=True, px=(1.0,), base_t=None, rp=98, seams=(6, 0),
                   solids=[("base", 20, 0.01, None, 0, 0, 0.01, None), P_(8, 0.6, 700, 5, 60, 1), P_(8, 0.6, 300, 5, 60, 1), P_(8, 0.6, 300, 5, 60, 1)]),
    "S7_block": dict(has_base=True, px=(1.0,), base_t=None, rp=98, seams=(2, 0),
                     solids=[("base", 20, 0.01, None, 0, 0, 0.01, None), ("block", 40, 1, None, 0, 40, 1, None), P_(6, 0.6, 400, 5, 70, 1)]),
}


def split_cmd(t):
    out, cur, q = [], "", False
    for ch in t:
        if ch == '"':
            q = not q
        elif ch.isspace() and not q:
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def family(kind):
    k = str(kind)
    return "plate" if k.startswith("strip") else "lean" if k.startswith("lean") else k


def match(exp, got):
    """每个期望实体按类别 + 尺寸找最接近的输出实体; 返回问题列表。"""
    fails, used = [], set()
    if len(got) != len(exp):
        fails.append("solids %d != %d" % (len(got), len(exp)))
    for fam, t, tt, L, Lt, top, tpt, lean in exp:
        best, bc = None, 1e18
        for i, s in enumerate(got):
            if i in used or family(s.get("kind")) != fam:
                continue
            c = abs((s.get("t") or 0) - t) / tt + (abs((s.get("L") or 0) - L) / Lt if L is not None else 0) + abs((s.get("top") or 0) - top) / tpt
            if c < bc:
                best, bc = i, c
        tag = "%s(t%s L%s top%s)" % (fam, t, L, top)
        if best is None:
            fails.append("missing " + tag); continue
        used.add(best); s = got[best]
        if abs((s.get("t") or 0) - t) > tt:
            fails.append("%s: t %s" % (tag, s.get("t")))
        if L is not None and abs((s.get("L") or 0) - L) > Lt:
            fails.append("%s: L %s" % (tag, s.get("L")))
        if abs((s.get("top") or 0) - top) > tpt:
            fails.append("%s: top %s" % (tag, s.get("top")))
        if lean is not None and abs((s.get("lean_deg") or 0) - lean[0]) > lean[1]:
            fails.append("%s: lean %s" % (tag, s.get("lean_deg")))
    return fails


names = [n for n in EXPECT if not args.only or n in args.only.split(",")]
os.makedirs(args.out, exist_ok=True)
rows, all_ok = [], True
for n in names:
    X = EXPECT[n]; pcd = os.path.join(args.synth, n + ".pcd"); ref = os.path.join(args.synth, n + ".step")
    od = os.path.join(args.out, n); step = os.path.join(od, "model.step"); os.makedirs(od, exist_ok=True)
    row = dict(name=n, fails=[])
    argv = [a.replace("{pcd}", pcd).replace("{out}", od) for a in split_cmd(args.cmd)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    open(os.path.join(od, "run.log"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    row["exit"] = p.returncode; row["wall_s"] = round(dt, 2)
    if p.returncode != 0 or not os.path.exists(step):
        row["fails"].append("run failed (exit %s)" % p.returncode)
    if args.time_limit > 0 and dt > args.time_limit:
        row["fails"].append("time %.1fs > %.0fs" % (dt, args.time_limit))
    if os.path.exists(step):
        cj = os.path.join(od, "check_step.json")
        subprocess.run([PY, "-I", os.path.join(HERE, "check_step.py"), step, "--ocp", args.ocp, "--json", cj], capture_output=True)
        c = json.load(open(cj, encoding="utf-8")) if os.path.exists(cj) else {}
        row["solids"] = "%s/%s" % (c.get("valid"), c.get("solids"))
        if not c or not c.get("read_ok") or c.get("valid") != c.get("solids") or not c.get("solids"):
            row["fails"].append("invalid STEP solids %s" % row["solids"])
        ej = os.path.join(args.out, n + "_eval.json")
        subprocess.run([PY, "-I", os.path.join(HERE, "eval_recon.py"), "--ocp", args.ocp, "--cloud", pcd, "--ref", ref,
                        "--out", step, "--json", ej], capture_output=True)
        e = json.load(open(ej, encoding="utf-8")).get("out_vs_ref", {}) if os.path.exists(ej) else {}
        if "surface_recall" in e:
            R, P = 100 * e["surface_recall"], 100 * e["surface_precision"]
            row.update(R=round(R, 1), P=round(P, 1), found="%d/%d" % (e["ref_units_found"], e["ref_units"]))
            if R < X["rp"] or P < X["rp"]:
                row["fails"].append("R/P %.1f/%.1f < %s" % (R, P, X["rp"]))
            if e["ref_units_found"] != e["ref_units"]:
                row["fails"].append("found %s" % row["found"])
        else:
            row["fails"].append("eval failed")
    pj = os.path.join(od, "parts.json")
    if os.path.exists(pj):
        d = json.load(open(pj, encoding="utf-8"))
        hb = d.get("has_base"); px = (d.get("heightmap") or {}).get("px"); bt = d.get("base_t_used")
        row.update(has_base=hb, px=px, base_t=bt)
        if hb != X["has_base"]:
            row["fails"].append("has_base %s" % hb)
        if px is None or not any(abs(px - a) < 1e-6 for a in X["px"]):
            row["fails"].append("px %s not in %s" % (px, X["px"]))
        if X["base_t"] and (bt is None or abs(bt - X["base_t"][0]) > X["base_t"][1]):
            row["fails"].append("base_t %s" % bt)
        row["fails"] += match(X["solids"], d.get("solids", []))
    else:
        row["fails"].append("no parts.json")
    if not args.no_seams:
        sc = os.path.join(od, "seams.csv")
        if os.path.exists(sc):
            ty = [r.get("type") for r in csv.DictReader(open(sc, encoding="utf-8"))]
            got = (ty.count("base_fillet"), ty.count("tee_fillet")); row["seams"] = got
            if got != X["seams"]:
                row["fails"].append("seams (base_fillet, tee_fillet) %s != %s" % (got, X["seams"]))
        else:
            row["fails"].append("no seams.csv")
    row["ok"] = not row["fails"]; all_ok &= row["ok"]; rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

lines = ["| 样例 | 通过 | R% | P% | found | 实体有效 | 用时s | has_base | px | base_t | 焊缝 | 问题 |", "|" + "---|" * 12]
for r in rows:
    lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["name"], "OK" if r["ok"] else "FAIL", r.get("R"), r.get("P"), r.get("found"), r.get("solids"), r.get("wall_s"),
        r.get("has_base"), r.get("px"), r.get("base_t"), r.get("seams", "-"), "; ".join(r["fails"])))
open(os.path.join(args.out, "synth_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(rows, open(os.path.join(args.out, "synth_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ALL OK" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
