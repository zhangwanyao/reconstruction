# 12 评测、合成样例与工具

> 读完这一章，你应该能回答：程序输出的好坏怎么量化？没有真实数据时怎么自测？工具目录里那一堆脚本分别干什么？
> 评分器 `tools/eval_recon.py`；合成样例 `tools/synth_gen.py`；批量检查 `tools/synth_bench.py`、`tools/run_bench.py`、`tools/run_noref.py`。

## 1. 怎么打分：eval_recon.py

三方对比：**点云**、**参考 STEP**、**输出 STEP**。两个模型都按面积在表面上均匀撒点（间距 `max(1.5, sqrt(面积/250 万))` mm，即面积大时自动放稀，总数上限约 250 万），然后：

| 指标 | 定义 | 说明 |
|---|---|---|
| 召回 R | 参考模型的表面点里，离输出模型 < τ 的比例 | 参考里有的，输出里也有 |
| 精度 P | 输出模型的表面点里，离参考模型 < τ 的比例 | 输出里有的，参考里也有 |
| F1 | 2RP/(R + P) | 两者的调和平均 |
| found | 参考里召回 ≥ 0.5 的实体数 | 找到了几块板 |
| valid | 输出里精度 ≥ 0.5 的实体数 | 输出的板有几块是对的 |
| 距离分布 | ref_to_out、out_to_ref 的 p50/p95 | 偏差到底多大 |

τ 默认 **3 mm**（`--tau`）。分数按**表面积**加权：大板（底板）分量重。看不到的背面也算分——所以"看不到的面用厚度补出来"很重要。

用法：

```
python -I tools/eval_recon.py --cloud X.pcd --ref ref.step --out out/model.step --json r.json
```

只给 `--ref` 不给出 `--out`：只统计参考模型 + 算点云到参考的偏差（看这份数据天生能打多少分）。结果写进 `--json` 指定的文件，也打印到屏幕。

## 2. 合成样例：synth_gen.py

7 个小样例，每个都带**标准答案**（`.step` 真值 + `.json` 参数真值），专为自学和回归测试设计。生成方式是"正交 z-buffer"：从上往下对抖动网格采样，所以它们的遮挡、边缘混合像素、点稀等毛病和真实线扫是同一种。

```
python -I tools/synth_gen.py data/synth        # 生成全部 7 个
python -I tools/synth_gen.py data/synth S1_tee # 只生成一个
```

| 样例 | 内容 | 教学点 |
|---|---|---|
| S1_tee | 底板 + 两块互相垂直的立板（T 形接头） | 全流程入门：合并 + 接头 |
| S2_oblique | 30° 摆放、斜顶的板 + 垂直顶上的板 | 骨架分叉（crossing number）、斜顶剖面 |
| S3_truss_nobase | 无底板：宽弦杆 + 交叉的宽腹杆 | 无底板流程、宽区 |
| S4_lean | 一块 10° 斜板 + 一块竖直板 | 倾斜板判定（竖直板必须保持竖直） |
| S5_sparse_grid | 点距 1.5 mm 的稀疏"井"字筋板 + 可见底面的 10 mm 薄底板 | 自动像素放大、稀疏点偏薄 |
| S6_gap | 一块板上有 100 mm 无数据阴影；另一块板有 100 mm 真缺口 | 遮挡合并 vs 真缺口 |
| S7_block | 厚块 + 一块薄板 | 宽区分层、块 |

## 3. 批量检查

**尺寸级自测（不需要真实数据）**：`tools/synth_bench.py` 生成 7 个样例、逐个跑 `proto_v2.py`、再用 `eval_recon.py` 和 `check_step.py` 检查，最后打印 `ALL OK`（退出码 0）才算过：

```
python -I tools/synth_bench.py --out out/bench --ocp <OCP目录> --tools tools --data data/synth
```

实测（合成样例的尺寸检查全过）：

| 样例 | 通过 | R% | P% | found | 用时 s | has_base | px | base_t |
|---|---|---|---|---|---|---|---|---|
| S1_tee | OK | 100.0 | 100.0 | 3/3 | 2.59 | 是 | 1.0 | 20.0 |
| S2_oblique | OK | 100.0 | 100.0 | 3/3 | 4.55 | 是 | 1.0 | 20.0 |
| S3_truss_nobase | OK | 97.2 | 97.7 | 2/2 | 2.28 | 否 | 1.0 | — |
| S4_lean | OK | 100.0 | 100.0 | 3/3 | 2.65 | 是 | 1.0 | 20.0 |
| S5_sparse_grid | OK | 100.0 | 100.0 | 5/5 | 1.89 | 是 | 1.5 | 10.0 |
| S6_gap | OK | 100.0 | 100.0 | 4/4 | 2.55 | 是 | 1.0 | 20.0 |
| S7_block | OK | 99.8 | 99.9 | 3/3 | 2.43 | 是 | 1.0 | 20.0 |

**真实数据全流程**（8 份带参考模型的点云在 `data/clouds/`）：

```
python -I tools/run_bench.py --cmd "<python.exe> -I tools/proto_v2.py {pcd} {out}" ^
       --data data/clouds --ocp <OCP目录> --out out/bench ^
       --eval-cloud SS34=data/clouds/SS34u.pcd --repeat F168 --time-limit 0
```

跑 8 份 → check_step → eval_recon → 对照分数线 → 写 summary.md；`--repeat F168` 再跑一遍 F168 验证输出逐字节一致。数据清单、每份点云的参考模型内容和成绩见 `data/clouds/README.md`。

- `tools/run_noref.py`：对一批无参考的点云做冒烟测试（不崩、不超时、实体有效、调试图目测合理）。34 份无参考的仿真云不随仓库分发，如有需要可单独提供。

## 4. 其他工具速查

| 工具 | 干什么 |
|---|---|
| `check_step.py` | 用 OCCT 读回 STEP，逐个实体 BRepCheck，体积必须 > 0 |
| `cmp_keys.py` / `cmp_plates.py` | 对比两份 parts.json 的键结构 / 板参数（第 10 章） |
| `dbg_base.py` | 把判底板的过程画出来（层、平整格、凸包、足迹） |
| `dbg_skel.py` | 把细条的骨架画出来（第 14 章练习 E6 用它） |
| `slice_probe.py` | 参考 STEP 里每块板的横截面点云 + 面统计 |
| `lean_probe.py` | 参考 STEP 里每块倾斜板的侧面点云 + 倾角 |
| `probe_planes.py` | 法向分组 + 偏移分层，早期探路工具 |
| `stage_table.py` | 把各点云各阶段的中间结果汇总成表 |
| `summarize_evals.py` / `summarize_refs.py` | 汇总评分结果 / 参考模型统计 |
| `px_stability.py` / `px_probe.py` | 检查自动像素选型稳不稳 |
| `ref_detail.py` / `ref_offset.py` / `ref_lean.py` / `inspect_ref.py` | 分析参考 STEP 的内容、和点云的对齐偏差、倾斜板 |
| `read_any_pcd.py` | 读 PCD（binary / binary_compressed），别的工具都 import 它 |
| `step_writer.py` | 写 STEP（第 9 章） |
| `unit_expect.py` / `test_frame.py` | 对照 C++ 版的单元测试期望值（本仓库用不上） |

下一章：[13 踩坑记录与设计问答](13_pitfalls_faq.md)
