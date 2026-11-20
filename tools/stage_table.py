"""把 proto_v2 每份点云的 parts.json 汇总成逐步对照表(markdown)。用法: python -I stage_table.py <proto_out_root> <out.md>"""
import json, os, sys

NAMES = ["F168", "F173", "F455", "WP", "SS30", "SS34", "WP169", "WP338"]
root, out = sys.argv[1], sys.argv[2]
L = []
rows1 = ["| 点云 | 点数 | 上方向偏角° | 网格格数 | 平整格 | 水平层(z: 面积m², 实心度, 凸包比, far, 底面深) | 有底板 | 底板倾角° | 底板rms | 底板厚 |",
         "|" + "---|" * 10]
rows2 = ["| 点云 | px | 占用率 ratio | rho_top | 骨架段 | 块 | 拟合成功 | 合并后 | 倾斜候选(ok/dense/采用) | 去重删掉 | 底板填充率 | 实体 | Python 用时 s |",
         "|" + "---|" * 13]
rows3 = ["| 点云 | 实体 kind 统计 | 板厚范围 mm | 倾斜板(度, 厚) |", "|---|---|---|---|"]
for n in NAMES:
    pj = os.path.join(root, n, "parts.json")
    if not os.path.exists(pj):
        continue
    d = json.load(open(pj, encoding="utf-8"))
    lv = []
    for c in d.get("h_levels", []):
        s = "%.1f: %.3f, %.2f, %.2f, %.3f" % (c["z"], c["area_m2"], c["solidity"], c["hull_ratio"], c["far"])
        if "bottom_depth" in c:
            s += ", %.1f(下方 %.3f)" % (c["bottom_depth"], c.get("far_below_bottom", -1))
        lv.append(s)
    bp = d.get("base_plane", {})
    rows1.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        n, d["points"], d.get("up_tilt_from_sensor_deg"), d.get("grid_cells"), d.get("flat_cells"), "<br>".join(lv) or "-",
        "是" if d.get("has_base") else "否", bp.get("tilt_deg", "-"), bp.get("rms", "-"), d.get("base_t_used")))
    hm = d.get("heightmap", {})
    le = d.get("lean", [])
    n_ok = sum(1 for x in le if x.get("ok")); n_dense = sum(1 for x in le if x.get("ok") and x.get("dense"))
    sol = d.get("solids", [])
    n_lean = sum(1 for s in sol if str(s.get("kind", "")).startswith("lean"))
    rows2.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d 个(%d/%d/%d) | %s | %s | %d | %s |" % (
        n, hm.get("px"), hm.get("occ_ratio"), hm.get("rho_top"), d.get("raw_segments"), d.get("blobs"), d.get("fitted"),
        d.get("merged"), len(le), n_ok, n_dense, n_lean, d.get("lean_dups_dropped"), d.get("base_fill_rect", "-"), len(sol),
        d.get("total_s")))
    kinds = {}
    for s in sol:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    ts = [s["t"] for s in sol if s["kind"] not in ("base", "block")]
    lean = ["%.2f°, t%.2f" % (s["lean_deg"], s["t"]) for s in sol if str(s.get("kind", "")).startswith("lean")]
    rows3.append("| %s | %s | %s | %s |" % (n, ", ".join("%s×%d" % kv for kv in sorted(kinds.items())),
                                            ("%.1f ~ %.1f" % (min(ts), max(ts))) if ts else "-", "; ".join(lean) or "-"))
open(out, "w", encoding="utf-8").write("\n".join(rows1 + [""] + rows2 + [""] + rows3) + "\n")
print("ok")
