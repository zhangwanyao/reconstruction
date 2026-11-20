# reconstruction — 板件焊接结构点云逆向建模（Python 参考实现）

从**自上而下的大线扫点云**（`.pcd`）自动重建板件焊接结构的 CAD 模型（`.step`）：

```
点云 (.pcd)  →  判断有没有底板、建工件坐标系  →  俯视高度图分块  →  骨架找板
            →  沿板轴线逐段横切量尺寸  →  共线合并、交线定端点  →  每块板一个棱柱实体  →  STEP + parts.json + 调试图
```

- **输入**：一份 PCD 点云（binary / binary_compressed），单位 mm，传感器 +Z 朝上。
- **输出**：`model.step`（每块板一个独立棱柱实体，不做布尔合并）、`parts.json`（中间结果与每个实体的参数）、`debug_top.png`（俯视调试图）。
- **有底板、无底板两类工件走同一条流程**，程序自己判断有没有底板。
- 在 8 份带参考模型的现场/仿真点云上，与参考模型的偏差 3 mm 以内：召回 75.5% ~ 98.9%，精度 77.0% ~ 99.0%（成绩见 `docs/01_background.md`）。

## 文档地图（按学习顺序读）

| 文档 | 内容 |
|---|---|
| [01 项目背景与目标](docs/01_background.md) | 做什么、数据长什么样、能做到什么程度 |
| [02 核心思路与流程总览](docs/02_overview.md) | 11 步流程地图、"板"的数据结构、坐标系约定 |
| [03 第 1–2 步](docs/03_frame.md) | 读点云；找上方向、判底板、建工件坐标系 |
| [04 第 3 步](docs/04_heightmap.md) | 自动像素、俯视高度图、凸起区 / 细条 / 宽区 |
| [05 第 4 步](docs/05_segments.md) | 骨架 → 直线段；宽区分层 → 斜坡段 / 宽段 / 块 |
| [06 第 5 步](docs/06_slices.md) | 沿轴线每 5 mm 横切一刀：顶高、板厚、中线 |
| [07 第 6–7 步](docs/07_merge_junctions.md) | 共线合并（遮挡桥接）；交线定端点 |
| [08 第 8 步](docs/08_profile_lean.md) | 顶边剖面；倾斜板判定与去重 |
| [09 第 9–10 步](docs/09_solids_output.md) | 底板轮廓；生成棱柱实体；写 STEP |
| [10 输出文件详解](docs/10_outputs.md) | parts.json、debug_top.png 的颜色、辅助工具 |
| [11 参数说明](docs/11_params.md) | 命令行参数与内置常数一览 |
| [12 评测与工具](docs/12_eval_tools.md) | 怎么打分；合成样例；全部工具速查 |
| [13 踩坑记录与设计问答](docs/13_pitfalls_faq.md) | 28 个坑（现象 → 原因 → 原型怎么处理）+ 6 个问答 |
| [14 练习](docs/14_exercises.md) | 7 个动手实验：把关键设计改坏，看分数掉多少 |
| [附录：设计思路原稿](docs/design_rationale.md) | 这套方法是怎么一步步想出来的 |

代码主体是 `tools/proto_v2.py`（一个从上往下顺序执行的脚本，每一步前有 `# ---- N 标题` 分隔注释），复用的函数在 `tools/protolib.py`。读文档时对着代码看效果最好。

## 快速开始

```powershell
# 1. 依赖（Python 3.10+）
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install cadquery-ocp          # 评分/检查工具才需要；主流程不依赖它

# 2. 生成 7 个带标准答案的合成样例
python -I tools/synth_gen.py data/synth

# 3. 跑最简单的 S1（T 形接头：底板 + 两块垂直立板）
python -I tools/proto_v2.py data/synth/S1_tee.pcd out/S1_tee
# 输出：out/S1_tee/model.step、parts.json、debug_top.png

# 4. 看调试图和中间结果
start out/S1_tee/debug_top.png
type out/S1_tee/parts.json

# 5. 评分（对照标准答案）
python -I tools/eval_recon.py --ocp .venv/Lib/site-packages --cloud data/synth/S1_tee.pcd `
       --ref data/synth/S1_tee.step --out out/S1_tee/model.step --json out/S1_tee/eval.json
```

第一次跑建议按 [02 第 9 节](docs/02_overview.md) 和 [14 练习](docs/14_exercises.md) 的顺序动手：先跑一遍 S1，再逐个做 7 个实验。

## 批量跑多份点云（recon.py / recon.exe）

`tools/recon.py` 是流水线的批量入口（不改动 `proto_v2.py`，只是包装）：

```
# 单文件：和直接跑 proto_v2.py 完全一样
python -I tools/recon.py data/synth/S1_tee.pcd out/S1_tee

# 批量：目录里每个 .pcd 跑一遍，出错不中断
python -I tools/recon.py --in data/synth --out out/batch --jobs 4
python -I tools/recon.py --list files.txt --out out/batch --skip-existing
```

每份输出 `<out>/<名字>/` 下的 `model.step`、`parts.json`、`debug_top.png`、`run.log`，最后写 `<out>/batch_summary.json`（每份的关键数字 + 成败统计）。`--jobs` 多进程并行；`--skip-existing` 断点续跑；其余参数（`--px`、`--base-t`、`--mode`…）原样传给流水线。

## 打包成 exe（可选）

主流程只依赖 numpy / scipy / opencv，可以打成独立 exe 批量使用（评分工具依赖 OCP，见下注）：

```powershell
pip install pyinstaller
pyinstaller --noconfirm --clean --onedir --name recon `
  --add-data "tools;tools" `
  --add-binary "<env>/Lib/site-packages/opencv_*.dll;." `
  --exclude-module matplotlib `
  --distpath D:\work\recon_dist --workpath D:\work\recon_build --specpath D:\work\recon_build `
  tools\recon.py
```

产出的 `recon_dist\recon\recon.exe` 用法和 `recon.py` 一样（单文件 / `--in` / `--list` / `--jobs`）。注意事项：

- `tools/recon.py` 顶部显式 `import numpy/scipy/cv2`，是为了让打包工具静态发现依赖（主流程代码是经 `runpy` 执行的，打包器看不到）。
- conda 环境的 opencv 是拆分布局（`cv2.pyd` 依赖 `Library\bin\opencv_*.dll`），必须用 `--add-binary` 收进来；PyPI 的自包含 wheel 则不需要。
- **打包后 `--jobs` 多进程可用**（入口已调用 `multiprocessing.freeze_support()`）；输出与 Python 逐字节一致。
- 评分（`eval_recon.py`）、有效性检查（`check_step.py`）依赖 OCP（约几百 MB），不建议打进主 exe；有参考模型的工件请用 Python 环境跑评分，主 exe 专注"大量出模型"。

## 目录结构

```
reconstruction/
  tools/
    proto_v2.py        主流程（第 1~11 步都在这里）
    protolib.py        坐标系/判底板、高度图、骨架分段、多边形清理
    read_any_pcd.py    PCD 读取（binary / binary_compressed）
    step_writer.py     纯文本 STEP 写入（不依赖 OCCT）
    synth_gen.py       合成样例生成器（7 个带标准答案的小用例）
    synth_bench.py     合成样例批量检查（尺寸级回归测试）
    eval_recon.py      评分：点云 / 参考 STEP / 输出 STEP 三方对比
    check_step.py      用 OCCT 读回 STEP 做有效性检查
    run_bench.py       真实数据全流程（需要 8 份带参考模型的点云）
    run_noref.py       无参考点云冒烟测试
    ...                其余为辅助分析工具，见 docs/12_eval_tools.md
  docs/                学习文档（见上面的地图）＋ images/
  requirements.txt
```

`data/`（生成的样例数据）和 `out/`（运行结果）不提交，`tools/synth_gen.py` 随时可以重新生成。

## 局限（诚实地说）

- 只做"板"：每块板 = 平面多边形 × 厚度。不追孔、倒角、焊缝余高、曲面和扫不到的背面细节。
- 立板侧面、背面、根部大多扫不到，这些面靠"看得到的面 + 板厚"推出来。
- 点稀的云（俯视每 mm² 少于 1 个点）量出的板厚会偏薄约一个点距（见 [14 练习 E1](docs/14_exercises.md)）。
- 参考模型和点云本身对不齐的工件（部分现场云差 2~3 mm），分数天生上不去。

详细的设计权衡见 [附录：设计思路原稿](docs/design_rationale.md)。
