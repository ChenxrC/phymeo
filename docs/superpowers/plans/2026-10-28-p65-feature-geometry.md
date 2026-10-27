# P6.5 H1b 静态表面几何 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。主实施者按测试先行执行，合并到本轮末一次整分支独立审阅；不重复请求用户实施授权。

**Goal:** 提供可与重心物理端口组合的点/三角、边/边最近点。

**Architecture:** 独立主机FP64凸距离算子，枚举内部和边界候选，返回自有最近点/凸权重；不接时间积分或接触响应。

**Tech Stack:** Python3.11/3.12、NumPy；无新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-28-p65-feature-geometry-design.md`。

## Global Constraints

- 当前`codex/p6-5-handoff`；截至2026-10-28 00:00UTC。完整P6.5仍须H2–H5，不将两个几何算子充当碰撞器。
- FP64有限自有数据；退化三角阈值二倍面积/最长边平方<=1e-12；非零近平行边sin<=1e-12明确SolverFailure，精确平行走边界解。
- 不选择零距离法向、不做CCD/内外检测/摩擦，不优化。所有不可表示结果拒绝。

## Review Focus

- 面内部比任一边更近：Task1精确面投影，避免只有边测试也通过。
- 两段交叉而四端点都很远：Task2内点交叉独立反例。
- 共线重叠/零距离具有多解：Task2确定性和距离，不要求不合理的唯一世界点。
- 极端尺度、退化、近平行：Task1/2拒绝路径；不能静默漏碰撞。
- 重心几何正确但载荷分配错误：Task3真实端口逐节点/功率对偶。

## Task1：点与三角面

**Files:** Create `src/phymeo/geometry/surface_features.py`、`tests/test_surface_features.py`。

- [ ] 先写解析内部/边/顶点、KKT、重心重建、刚体变换/尺度、所有权及非法/退化输入测试，运行文件确认缺实现RED。
- [ ] 实现`ClosestFeaturePair`与`closest_point_triangle`，FP64平移缩放、正交面基、边界枚举；验证GREEN及ruff，小提交。

## Task2：两条有限边

**Files:** Modify上述源/测试文件。

- [ ] 先写交叉/偏斜/平行/共线重叠/端点钳制、交换对称、KKT、近平行拒绝和溢出测试；Task1保持通过，Task2缺实现RED。
- [ ] 实现`closest_segment_segment`的内点/四边界候选；所有权及输入域检查；全文件和相关旧几何回归GREEN，ruff，小提交。

## Task3：物理组合和交付

**Files:** Modify `src/phymeo/__init__.py`、`tests/test_surface_features.py`、README/进度/性能台账。

- [ ] 公开导出RED与三角几何→H1a实际端口的独立载荷/功率测试；实现导出并GREEN。
- [ ] 记录静态小负载耗时、全部拒绝限制与后续CCD/接触程序入口；本轮切片全回归、wheel源码外smoke、CI、一次整分支审阅与Git交付。尚未执行的检查不可勾选。

## 自查

Task1返回自有数据类，Task2复用同合同；Task3使用现有WeightedPointPort权重机制，不向SceneStepper加入几何特例。两算子既不读取隐藏状态也不改变同域反作用聚合。后续网格层必须处理本设计明确拒绝的几何，不得捕获后跳过接触对。
