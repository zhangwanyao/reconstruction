"""Summarise proto runs on the synthetic cases. Usage: python -I synth_report.py <synth_dir> <run_root> <out_txt>
<run_root>/<case>/parts.json, <run_root>/<case>.eval.json, <run_root>/<case>.check.log (json line) are read."""
import json, os, sys

sd, rr, out = sys.argv[1], sys.argv[2], sys.argv[3]
lines = []
for fn in sorted(os.listdir(sd)):
    if not fn.endswith(".json"):
        continue
    nm = fn[:-5]
    tr = json.load(open(os.path.join(sd, fn), encoding="utf-8"))
    lines.append("== %s  points=%d spacing=%s truth_has_base=%s" % (nm, tr["points"], tr["spacing_mm"], tr["has_base"]))
    for s in tr["solids"]:
        lines.append("   truth %-6s %-6s t=%s top=%s L=%s lean=%s" % (s["name"], s["kind"], s.get("t"), s.get("top"), s.get("L"), s.get("lean_deg")))
    pj = os.path.join(rr, nm, "parts.json")
    if not os.path.exists(pj):
        lines.append("   NO parts.json"); continue
    d = json.load(open(pj, encoding="utf-8"))
    hm = d.get("heightmap", {})
    lines.append("   out has_base=%s base_t_used=%s px=%s occ=%s rho=%s raw=%s fitted=%s merged=%s blobs=%s dups=%s fill=%s total_s=%s" % (
        d.get("has_base"), d.get("base_t_used"), hm.get("px"), hm.get("occ_ratio"), hm.get("rho_top"), d.get("raw_segments"),
        d.get("fitted"), d.get("merged"), d.get("blobs"), d.get("lean_dups_dropped"), d.get("base_fill_rect"), d.get("total_s")))
    lv = ["%s:%s/%s/%s/%s%s" % (c["z"], c["area_m2"], c["solidity"], c["hull_ratio"], c["far"],
                               (" bd=%s fbb=%s" % (c["bottom_depth"], c["far_below_bottom"])) if "bottom_depth" in c else "") for c in d.get("h_levels", [])]
    lines.append("   levels " + "; ".join(lv))
    for s in d.get("solids", []):
        lines.append("   out   %-9s %-11s t=%s L=%s top=%s lean=%s prof=%s" % (s["name"], s["kind"], s.get("t"), s.get("L"), s.get("top"), s.get("lean_deg"), s.get("prof_pts")))
    for li in d.get("lean", []):
        lines.append("   leancand L=%s inl=%s cand=%s zspan=%s rms=%s lean=%s f2c=%s ok=%s rel=%s dense=%s" % (
            li["L"], li["inl"], li["cand"], li["zspan"], li["rms"], li["lean"], li["face_to_cap"], li["ok"], li["rel_dens"], li["dense"]))
    ej = os.path.join(rr, nm + ".eval.json")
    if os.path.exists(ej):
        e = json.load(open(ej, encoding="utf-8")).get("out_vs_ref", {})
        if e:
            lines.append("   eval R=%.1f P=%.1f found=%s/%s valid=%s/%s per_ref=%s per_out=%s" % (
                100 * e["surface_recall"], 100 * e["surface_precision"], e["ref_units_found"], e["ref_units"],
                e["out_units_valid"], e["out_units"], e.get("per_ref_recall"), e.get("per_out_precision")))
    cl = os.path.join(rr, nm + ".check.log")
    if os.path.exists(cl):
        raw = open(cl, "rb").read()
        txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
        js = [l for l in txt.splitlines() if l.startswith("{")]
        if js:
            c = json.loads(js[-1]); lines.append("   check solids=%s valid=%s volumes=%s" % (c.get("solids"), c.get("valid"), c.get("volumes")))
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("ok", len(lines))
