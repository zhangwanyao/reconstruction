"""基准测试: 每份点云 -> 运行重建命令 -> 检查 STEP 实体 -> eval_recon 评分 -> 对照验收线。
用法:
  python -I run_bench.py --cmd "<程序> {pcd} {out}" --data <数据目录> --ocp <OCP 目录> --out <结果目录>
         [--only F168,WP338] [--eval-cloud SS34=<未压缩的 SS34.pcd>] [--repeat F168] [--time-limit 10]
  数据目录里要有 <名字>.pcd 和 <名字>.step, 名字 = F168 F173 F455 WP SS30 SS34 WP169 WP338。
  {pcd} {out} 会被替换成点云路径和该份点云的输出目录。例:
    --cmd "build\\plate_recon.exe {pcd} {out}"
    --cmd "<python.exe> -I tools\\proto_v2.py {pcd} {out}"      (用 Python 原型自测本脚本; 原型较慢, 用 --time-limit 0 关掉时间检查)
  --repeat F168: 对 F168 再跑一次, 比较两次 model.step 是否逐字节相同(确定性)。
输出: <结果目录>/summary.md, summary.json; 退出码 0 = 全部通过, 1 = 有不通过。
"""
import argparse, hashlib, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--ocp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--eval-cloud", action="append", default=[])
ap.add_argument("--repeat", default="")
ap.add_argument("--time-limit", type=float, default=10.0)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

NAMES = ["F168", "F173", "F455", "WP", "SS30", "SS34", "WP169", "WP338"]
# 最终版 v2 (Python 原型 proto_v2.py = proto2c) 的成绩: 召回%, 精度%
V2 = {"F168": (98.9, 99.0), "F173": (98.5, 97.9), "F455": (95.9, 96.5), "WP": (75.5, 77.0),
      "SS30": (86.0, 96.4), "SS34": (98.3, 98.0), "WP169": (96.3, 98.7), "WP338": (97.3, 96.3)}
FOUND_MIN = {"F168": 8, "F173": 8, "F455": 18, "WP": 7, "SS30": 24, "SS34": 12, "WP169": 8, "WP338": 5}
A_F1 = {"F168": 89.9, "F173": 93.8, "F455": 95.5, "SS30": 61.3, "SS34": 88.5}   # A 在同一份点云上的 F1
HAS_BASE = {n: n != "SS34" for n in NAMES}
# 允许的自动像素(mm), 其余点云 1.0。SS30 的占用率 0.701 正好压在门槛 0.7 上, 实测 px=2.0/2.5/3.0 分数都过线, 所以都算对
PX = {"SS30": (2.0, 2.5, 3.0), "SS34": (1.5,)}
LEAN = {"WP169": [4.42, 8.25, 19.6], "WP338": [7.06]}                            # 参考模型里倾斜板的倾角; 其余点云不应出现倾斜板


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


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def run_one(name, pcd, od):
    os.makedirs(od, exist_ok=True)
    argv = [a.replace("{pcd}", pcd).replace("{out}", od) for a in split_cmd(args.cmd)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    open(os.path.join(od, "run.log"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, dt


ec = dict(kv.split("=", 1) for kv in args.eval_cloud)
names = [n for n in NAMES if not args.only or n in args.only.split(",")]
os.makedirs(args.out, exist_ok=True)
rows, all_ok = [], True
for n in names:
    pcd = os.path.join(args.data, n + ".pcd"); ref = os.path.join(args.data, n + ".step")
    od = os.path.join(args.out, n); step = os.path.join(od, "model.step")
    row = dict(name=n, fails=[])
    rc, dt = run_one(n, pcd, od)
    row["exit"] = rc; row["wall_s"] = round(dt, 2)
    if rc != 0 or not os.path.exists(step):
        row["fails"].append("run failed (exit %s), see %s" % (rc, os.path.join(od, "run.log")))
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
        subprocess.run([PY, "-I", os.path.join(HERE, "eval_recon.py"), "--ocp", args.ocp, "--cloud", ec.get(n, pcd),
                        "--ref", ref, "--out", step, "--json", ej], capture_output=True)
        e = json.load(open(ej, encoding="utf-8")).get("out_vs_ref", {}) if os.path.exists(ej) else {}
        if "surface_recall" in e:
            R, P = 100 * e["surface_recall"], 100 * e["surface_precision"]
            F1 = 2 * R * P / max(R + P, 1e-9)
            row.update(R=round(R, 1), P=round(P, 1), F1=round(F1, 1), found="%d/%d" % (e["ref_units_found"], e["ref_units"]),
                       valid="%d/%d" % (e["out_units_valid"], e["out_units"]))
            if R < V2[n][0] - 2:
                row["fails"].append("recall %.1f < %.1f" % (R, V2[n][0] - 2))
            if P < V2[n][1] - 2:
                row["fails"].append("precision %.1f < %.1f" % (P, V2[n][1] - 2))
            if n in A_F1 and F1 < A_F1[n]:
                row["fails"].append("F1 %.1f < A %.1f" % (F1, A_F1[n]))
            if e["ref_units_found"] < FOUND_MIN[n]:
                row["fails"].append("found %d < %d" % (e["ref_units_found"], FOUND_MIN[n]))
        else:
            row["fails"].append("eval failed: %s" % e.get("error", "no json"))
    pj = os.path.join(od, "parts.json")
    if os.path.exists(pj):
        d = json.load(open(pj, encoding="utf-8"))
        hb = d.get("has_base"); px = (d.get("heightmap") or {}).get("px"); bt = d.get("base_t_used")
        leans = sorted(s.get("lean_deg") for s in d.get("solids", []) if str(s.get("kind", "")).startswith("lean"))
        row.update(has_base=hb, px=px, base_t=bt, leans=leans, total_s=d.get("total_s"))
        if hb != HAS_BASE[n]:
            row["fails"].append("has_base %s != %s" % (hb, HAS_BASE[n]))
        if px is None or not any(abs(px - a) < 1e-6 for a in PX.get(n, (1.0,))):
            row["fails"].append("px %s not in %s" % (px, PX.get(n, (1.0,))))
        if n == "SS30" and (bt is None or abs(bt - 10.0) > 0.5):
            row["fails"].append("SS30 base_t %s not 10+-0.5" % bt)
        want = sorted(LEAN.get(n, []))
        if len(leans) != len(want) or any(abs(a - b) > 1.5 for a, b in zip(leans, want)):
            row["fails"].append("lean %s != ref %s (+-1.5 deg)" % (leans, want))
    else:
        row["fails"].append("no parts.json")
    if n == args.repeat and os.path.exists(step):
        od2 = od + "_repeat"
        run_one(n, pcd, od2)
        same = sha(step) == sha(os.path.join(od2, "model.step"))
        row["deterministic"] = same
        if not same:
            row["fails"].append("model.step differs between two runs")
    row["ok"] = not row["fails"]
    all_ok &= row["ok"]
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

hdr = "| 点云 | 通过 | R% | P% | F1 | found | valid | 实体有效 | 用时s | has_base | px | base_t | 倾斜板(度) | 问题 |"
lines = [hdr, "|" + "---|" * 14]
for r in rows:
    lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["name"], "OK" if r["ok"] else "FAIL", r.get("R"), r.get("P"), r.get("F1"), r.get("found"), r.get("valid"),
        r.get("solids"), r.get("wall_s"), r.get("has_base"), r.get("px"), r.get("base_t"), r.get("leans"), "; ".join(r["fails"])))
open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(rows, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ALL OK" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
