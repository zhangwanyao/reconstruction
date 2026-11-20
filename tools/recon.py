"""recon.py —— v2 流水线的单文件 / 批量入口（proto_v2.py 的包装，不修改原代码）。

单文件（行为和直接跑 proto_v2.py 完全一样，多打一行提示）:
  python -I tools/recon.py data/synth/S1_tee.pcd out/S1_tee [proto_v2 的其余参数]
  recon.exe data/synth/S1_tee.pcd out/S1_tee [proto_v2 的其余参数]

批量（每个 .pcd 跑一遍，出错不中断；每份输出 <out>/<名字>/ 下的 model.step、parts.json、
debug_top.png、run.log，最后写 <out>/batch_summary.json）:
  python -I tools/recon.py --in data/synth --out out/batch [proto_v2 的其余参数]
  python -I tools/recon.py --list files.txt --out out/batch
  recon.exe --in D:\\pcds --out D:\\out --px 1.5 --jobs 4

批量可选参数:
  --list 文件     每行一个 .pcd 路径（# 开头是注释）
  --jobs N        多进程并行（默认 1，即顺序跑）
  --skip-existing 输出目录里已有 parts.json 的云跳过（断点续跑）
"""
import argparse
import json
import multiprocessing
import os
import runpy
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr

# proto_v2.py 经 runpy 执行，打包工具看不到它的 import。
# 这里显式导入，让 PyInstaller 能把依赖（及 scipy/cv2 的钩子）收进来；运行时无副作用。
import numpy  # noqa: F401
import scipy.ndimage  # noqa: F401
import scipy.spatial  # noqa: F401
import cv2  # noqa: F401

if getattr(sys, "frozen", False):
    TOOLS = os.path.join(sys._MEIPASS, "tools")
else:
    TOOLS = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(TOOLS, "proto_v2.py")


def run_one(pcd, out_dir, extras, quiet=False):
    """跑一份点云。返回 dict（cloud/out/time_s/error + parts.json 的关键数）。"""
    os.makedirs(out_dir, exist_ok=True)
    argv_backup, argv0_backup = sys.argv, None
    t0 = time.time()
    err = None
    log = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") if quiet else None
    try:
        sys.argv = [PROTO, pcd, out_dir] + list(extras)
        if quiet:
            with redirect_stdout(log), redirect_stderr(log):
                runpy.run_path(PROTO, run_name="__main__")
        else:
            runpy.run_path(PROTO, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = "SystemExit %r" % (e.code,)
    except BaseException:
        err = traceback.format_exc(limit=8)
    finally:
        sys.argv = argv_backup
        if log:
            log.close()
    rec = {"cloud": os.path.basename(pcd), "out": out_dir, "time_s": round(time.time() - t0, 2), "error": err}
    pj = os.path.join(out_dir, "parts.json")
    if err is None and os.path.exists(pj):
        try:
            d = json.load(open(pj, encoding="utf-8"))
            hm = d.get("heightmap") or {}
            rec.update(points=d.get("points"), has_base=d.get("has_base"),
                       px=hm.get("px"), base_t=d.get("base_t_used"),
                       segs=d.get("raw_segments"), fitted=d.get("fitted"), merged=d.get("merged"),
                       blobs=d.get("blobs"), solids=len(d.get("solids", [])),
                       step=d.get("step_solids"), total_s=d.get("total_s"))
        except BaseException:
            pass
    return rec


def worker(args):
    return run_one(*args)


def batch(paths, out_root, extras, jobs, skip):
    n = len(paths)
    print("batch: %d clouds -> %s  (jobs=%d)" % (n, out_root, jobs))
    tasks = []
    for i, p in enumerate(paths):
        d = os.path.join(out_root, os.path.splitext(os.path.basename(p))[0])
        if skip and os.path.exists(os.path.join(d, "parts.json")):
            print("  [%3d/%d] SKIP %s" % (i + 1, n, os.path.basename(p)))
            continue
        tasks.append((p, d, extras, True))
    results = []
    t0 = time.time()
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(worker, tasks))
    else:
        results = [worker(t) for t in tasks]
    ok = 0
    for i, r in enumerate(results):
        if r["error"]:
            print("  [%3d/%d] FAIL %s  %s" % (i + 1, n, r["cloud"], r["error"].splitlines()[-1]))
        else:
            ok += 1
            print("  [%3d/%d] OK   %-28s t=%5.1fs  has_base=%-5s px=%-4s fitted=%d merged=%d solids=%d" % (
                i + 1, n, r["cloud"], r["time_s"], r.get("has_base"), r.get("px"),
                r.get("fitted", 0), r.get("merged", 0), r.get("solids", 0)))
    summary = dict(results=results, ok=ok, failed=len(results) - ok,
                   total_time_s=round(time.time() - t0, 1))
    with open(os.path.join(out_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("batch done: %d/%d ok, %.1f s total" % (ok, len(results), summary["total_time_s"]))


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if sys.argv[1].lower().endswith(".pcd"):
        # 单文件：直接把命令行转发给 proto_v2.py（行为与直接运行它一致）
        runpy.run_path(PROTO, run_name="__main__")
        return
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--in", dest="indir")
    ap.add_argument("--list", dest="listfile")
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--base-t", type=float)
    ap.add_argument("--px", type=float)
    ap.add_argument("--h-min", type=float)
    ap.add_argument("--w-thin", type=float)
    ap.add_argument("--mode")
    ap.add_argument("--lean-min", type=float)
    a, rest = ap.parse_known_args()
    extras = []
    for flag, name in (("--base-t", "base_t"), ("--px", "px"), ("--h-min", "h_min"),
                       ("--w-thin", "w_thin"), ("--mode", "mode"), ("--lean-min", "lean_min")):
        v = getattr(a, name)
        if v is not None:
            extras += [flag, str(v)]
    paths = []
    if a.indir:
        paths = sorted(p for p in (os.path.join(a.indir, f) for f in os.listdir(a.indir))
                       if p.lower().endswith(".pcd"))
    elif a.listfile:
        paths = [ln.strip() for ln in open(a.listfile, encoding="utf-8")
                 if ln.strip() and not ln.strip().startswith("#")]
    if not paths:
        print("no .pcd found (need --in <dir> or --list <file>); use --help for usage")
        sys.exit(2)
    batch(paths, a.out, extras, a.jobs, a.skip_existing)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # 打包成 exe 后，--jobs 多进程需要它
    main()
