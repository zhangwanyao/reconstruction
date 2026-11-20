"""无参考点云批量跑(任务书 5.2): 每份 -> 运行重建命令 -> 退出码/用时 -> check_step -> parts.json 摘要 -> 汇总表。
用法:
  python -I run_noref.py --cmd "<程序> {pcd} {out}" --data <放 .pcd 的目录> --ocp <OCP 目录> --out <结果目录>
         [--include "PC*.pcd,SS*.pcd"] [--exclude "SS30_*,SS34_*"] [--time-limit 10] [--no-check]
  只看 --data 这一层(不进子目录)。每份的 debug_top.png 另外复制到 <结果目录>/_png/<名字>.png, 方便一起翻看。
  通过 = 退出码 0、用时不超(--time-limit 0 不查)、model.step 至少 1 个实体且全部有效。
  --expect <上一次的 noref_summary.json>(例如 Python 原型的): 有底板、px、实体数(差 > 30% 且 > 2)、倾斜板块数
  和它不一样时, 在表格最后一列"和 expect 比"里提示。只提示, 不影响通过与否。
输出: <结果目录>/noref_summary.md, noref_summary.json; 退出码 0 = 全部通过, 1 = 有不通过。
"""
import argparse, fnmatch, json, os, shutil, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--ocp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--include", default="*.pcd")
ap.add_argument("--exclude", default="")
ap.add_argument("--time-limit", type=float, default=10.0)
ap.add_argument("--no-check", action="store_true")
ap.add_argument("--expect", default="")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


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


inc = [p for p in args.include.split(",") if p]
exc = [p for p in args.exclude.split(",") if p]
files = sorted(f for f in os.listdir(args.data)
               if os.path.isfile(os.path.join(args.data, f)) and any(fnmatch.fnmatch(f, p) for p in inc)
               and not any(fnmatch.fnmatch(f, p) for p in exc))
os.makedirs(os.path.join(args.out, "_png"), exist_ok=True)
expect = {r["name"]: r for r in json.load(open(args.expect, encoding="utf-8"))} if args.expect else None


def vs_expect(e, r):
    if e is None:
        return ["expect 里没有这份"]
    d = []
    for k in ("has_base", "px"):
        if e.get(k) != r.get(k):
            d.append("%s %s→%s" % (k, e.get(k), r.get(k)))
    a, b = e.get("solids") or 0, r.get("solids") or 0
    if abs(a - b) > max(2, 0.3 * a):
        d.append("实体 %s→%s" % (a, b))
    a, b = len(e.get("lean") or []), len(r.get("lean") or [])
    if a != b:
        d.append("倾斜板 %d→%d" % (a, b))
    return d


rows, all_ok = [], True
for f in files:
    nm = os.path.splitext(f)[0]; pcd = os.path.join(args.data, f); od = os.path.join(args.out, nm)
    os.makedirs(od, exist_ok=True)
    for old in ("model.step", "parts.json", "debug_top.png", "seams.csv", "check_step.json"):  # 不让上一次的结果冒充这一次
        if os.path.exists(os.path.join(od, old)):
            os.remove(os.path.join(od, old))
    row = dict(name=nm, fails=[])
    argv = [a.replace("{pcd}", pcd).replace("{out}", od) for a in split_cmd(args.cmd)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    open(os.path.join(od, "run.log"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    row.update(exit=p.returncode, wall_s=round(dt, 2), mb=round(os.path.getsize(pcd) / 2 ** 20, 1))
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or [""])[-1][:160]
        row["fails"].append("exit %s %s" % (p.returncode, tail))
    if args.time_limit > 0 and dt > args.time_limit:
        row["fails"].append("time %.1fs > %.0fs" % (dt, args.time_limit))
    pj = os.path.join(od, "parts.json")
    if os.path.exists(pj):
        d = json.load(open(pj, encoding="utf-8"))
        S = d.get("solids", []) or []
        kinds = {}
        for s in S:
            kinds[s.get("kind")] = kinds.get(s.get("kind"), 0) + 1
        row.update(points=d.get("points"), has_base=d.get("has_base"), px=(d.get("heightmap") or {}).get("px"),
                   base_t=d.get("base_t_used"), solids=len(S), kinds=kinds,
                   lean=[s.get("lean_deg") for s in S if str(s.get("kind", "")).startswith("lean")],
                   uncovered=len(d.get("uncovered") or []) if "uncovered" in d else None, total_s=d.get("total_s"))
    else:
        row["fails"].append("no parts.json")
    step = os.path.join(od, "model.step")
    if not os.path.exists(step):
        row["fails"].append("no model.step")
    elif not args.no_check:
        cj = os.path.join(od, "check_step.json")
        subprocess.run([PY, "-I", os.path.join(HERE, "check_step.py"), step, "--ocp", args.ocp, "--json", cj], capture_output=True)
        c = json.load(open(cj, encoding="utf-8")) if os.path.exists(cj) else {}
        row["valid"] = "%s/%s" % (c.get("valid"), c.get("solids"))
        if not c or not c.get("read_ok") or not c.get("solids") or c.get("valid") != c.get("solids"):
            row["fails"].append("invalid STEP solids %s" % row["valid"])
    png = os.path.join(od, "debug_top.png")
    if os.path.exists(png):
        shutil.copyfile(png, os.path.join(args.out, "_png", nm + ".png"))
    row["ok"] = not row["fails"]; all_ok &= row["ok"]
    if expect is not None:
        row["vs_expect"] = vs_expect(expect.get(nm), row)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

cols = ["点云", "通过", "MB", "点数", "用时s", "有底板", "px", "底板厚", "实体", "有效", "kind 统计", "倾斜板°", "问题"]
if expect is not None:
    cols.append("和 expect 比")
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in rows:
    ks = ", ".join("%s×%d" % (k, v) for k, v in sorted((r.get("kinds") or {}).items()))
    cells = [r["name"], "OK" if r["ok"] else "FAIL", r.get("mb"), r.get("points"), r.get("wall_s"), r.get("has_base"), r.get("px"),
             r.get("base_t"), r.get("solids"), r.get("valid", "-"), ks or "-", ", ".join(str(x) for x in r.get("lean") or []) or "-",
             "; ".join(r["fails"])]
    if expect is not None:
        cells.append("; ".join(r["vs_expect"]))
    lines.append("| " + " | ".join(str(x) for x in cells) + " |")
lines.append("")
lines.append("%d 份, 通过 %d 份" % (len(rows), sum(r["ok"] for r in rows)))
if expect is not None:
    lines.append("和 expect 不一样的 %d 份(只是提示)" % sum(1 for r in rows if r["vs_expect"]))
open(os.path.join(args.out, "noref_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(rows, open(os.path.join(args.out, "noref_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ALL OK" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
