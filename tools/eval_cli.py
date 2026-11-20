"""eval_cli.py —— check_step / eval_recon 的打包入口（供打 exe 用，平时直接用 tools/ 里的脚本）。

用法:
  python -I tools/eval_cli.py check_step out/S1_tee/model.step [OCP_deps 目录]
  python -I tools/eval_cli.py eval_recon --cloud X.pcd --ref r.step --out o.step --json r.json [--ocp 目录]
打包后 OCP 已内置，--ocp / 目录参数可以省略。
"""
import argparse  # noqa: F401
import json  # noqa: F401
import os
import runpy
import sys
import time  # noqa: F401

import numpy  # noqa: F401
import scipy  # noqa: F401
import scipy.spatial  # noqa: F401

if getattr(sys, "frozen", False):
    TOOLS = os.path.join(sys._MEIPASS, "tools")
else:
    TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

# 打包说明：OCP 整体（OCP/ 树 + vtk.libs + cadquery_ocp.libs）在打 exe 时用 --add-data
# 原样收进包内（OCP 的模块是运行时按需加载的，静态分析收集不全会漏掉子目录）。



def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check_step", "eval_recon"):
        print(__doc__)
        sys.exit(2)
    sub = sys.argv[1]
    sys.argv = [os.path.join(TOOLS, sub + ".py")] + sys.argv[2:]
    runpy.run_path(os.path.join(TOOLS, sub + ".py"), run_name="__main__")


if __name__ == "__main__":
    main()
