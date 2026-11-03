# Newton 与 Phymeo 的模块对照案例

这组案例服务于刚体之间的接触、刚体与可变形表面的接触、机器人与资产工具链三个独立方向。每个方向先定义可复算的输入与预期，再分别在 Newton 和 Phymeo 上读取实际输出。同一案例可以在两个引擎上有不同的支持范围。机器人的最终递布案例另外验收。

## 本机基线与开发入口

2026-11-03 在 Windows 上使用 Python 3.12.13、Newton 1.7.0、Warp 1.17.0 和 NumPy 2.5.4。CPU 和 RTX 4060 Laptop GPU（驱动 566.07）都实测了球–地面接触和一个 XPBD 重力步。Phymeo 对照使用原有 Python 3.12.14 与 tyx。URDF、MJCF、USD 的小型资产随案例库提交，运行无需下载机器人模型。

[35 个案例逐项实测](../../docs/benchmarks/newton-1.7-reference-baseline.md) · [完整物理读数与失败记录](../../docs/benchmarks/newton-1.7-reference-observations.json) · [按依赖和难度排列的开发计划](../../docs/superpowers/plans/2026-11-03-newton-reference-development.md)

初始 35 行表保留当时的实测。后续在同一数值判据下，固定地形与球的 4 项、URDF/MJCF 的 2 项已从 Phymeo 的入口缺口变为可执行。[本次增量与复现](../../docs/benchmarks/p65-robot-terrain-20261103.md)

双凹网格 4 项及解析曲面 3 项已在 Phymeo HOST 验收，`ROBOT-USD-ZUP` 已在 HOST 与实际 CUDA 验收。Phymeo 的厘米/Y-up 等尺寸物理转换另有试片。原 Newton 告警项继续保留原判据，在 Phymeo 列按不同语义分类。[三线测量与支持域](../../docs/benchmarks/p65-independent-rigid-usd-20261103.md)

## 输入与报告

每个方向有独立的 `cases.json` 与 `baseline.py`。目录名为 `rr`、`rd` 和 `robot`。根对象使用 `schema="newton-reference-v1"`、`lane` 及非空 `cases` 列表。

| 字段 | 约定 |
| --- | --- |
| `id`、`family`、`difficulty` | ID 以大写方向名加连字符开头。难度为 `easy`、`medium` 或 `hard`。 |
| `inputs`、`expected`、`tolerances`、`oracle` | 完整输入或由输入确定的本地资产，预期物理量与单位，具名非负容差，以及独立几何、运动学或动力学参照。 |
| `newton`、`phymeo` | 分别登记 `status`、`devices` 和 `reason`。可运行项用 `run` 并列出设备。公共入口缺失用 `gap`，物理语义无法等价对照用 `not_comparable`。后两者的设备列表为空。 |
| 执行 | 各方向实现 `run_case(case, *, engine, device)`，实际读取结果并核对预期，返回包含实测量的非空字典。断言或运行异常记为 `fail`，命令返回非零值。 |

`cpu` 与 `cuda:0` 是显式设备。资产解析通常发生在主机，在 CUDA 上核对 FK 或接触的案例分别记录这两部分。检测到初始交叠、离散时刻的触点和整步首次相触时间分别设定断言。统计中单列 `gap`、`not_comparable` 和 `device_not_covered`。

## 运行

从仓库根目录执行。Newton 版本固定为 1.7.0。Windows 上本文测试组合的精确依赖见 [独立环境锁文件](../../requirements/newton-reference-win-py312.txt)。先创建独立环境，再安装依赖。Phymeo 仍使用其原有环境。命令示例中的 `python` 为可用的 CPython 3.12。

```powershell
python -m venv .venv-newton-1.7.0
.\.venv-newton-1.7.0\Scripts\python.exe -m pip install -r requirements/newton-reference-win-py312.txt
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/smoke.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/smoke.py --device cuda:0
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/run_matrix.py --list
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/run_matrix.py --engine newton --lane rr --device cpu --out rr-newton-cpu.json
```

切换 `--lane` 可选择 `rd`、`robot` 或 `all`。用项目原有的 Python 并将 `PYTHONPATH` 指向当前工作树的 `src`，即可选择 `--engine phymeo`。输出文件须使用新路径。只复现单个构型时可以传入 `--case` 和完整案例 ID。

[Newton 1.7.0 安装与驱动要求](https://newton-physics.github.io/newton/1.7.0/guide/installation.html) · [碰撞管线](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html) · [求解器支持矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)
