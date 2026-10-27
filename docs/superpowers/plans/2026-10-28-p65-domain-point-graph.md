# P6.5 H1c 多域点载荷图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。主实施者测试先行；与H1a/H1b/H2a共享本轮末一次整分支审阅。

**Goal:** A/B/布/固定支持域的共享响应和一次反作用聚合，保持原始各域状态所有权。

**Architecture:** 每域一个不可变PointPort，世界有符号关联路由；所有候选验证成功才返回整组状态，不推进时间或认定接触成立。

**Tech Stack:** Python3.11/3.12、NumPy，底层可使用固定tyx三模式。

**Spec:** `docs/superpowers/specs/2026-10-28-p65-domain-point-graph-design.md`。

## Global Constraints

- `codex/p6-5-handoff`；截止2026-10-28 00:00UTC。本接口只处理瞬时冲量Ns，不能当位置load、摩擦图或全局时间事务。
- 配置/结果自有，FP64主机路由，底层保持后端；数组差错拒绝、数值失败SolverFailure、无部分候选。
- 域key=port.domain，调用方负责不同域不别名同一物理状态；新图不绕过现有SceneStepper。

## Review Focus

- 两条接触共享布时非对角响应丢失：Task1三质量解析矩阵与手工冲量测试。
- 同域重复/自环产生双次候选或溢出：Task1一次计数与精确零测试。
- 后端/最后域失败时先前候选外泄：Task1故障与状态隔离。
- 相反线冲量不等于任意两点角动量守恒：Task1共点角动量，非共点明确只验证功率/线冲量。
- 名称tyx但某个域实际未执行：Task2逐域完成计数增长。

## Task1：图响应与候选

**Files:** Create `src/phymeo/coupling/domain_graph.py`、`tests/test_domain_point_graph.py`。

- [ ] 三质量解析、共享节点/机器人、功率/动量、每域一次apply、自环/重复/空/固定、非法输入/溢出与最后域失败测试先RED。
- [ ] 实现规格中完整API，复用现有端口，不改P6求解器；全文件与相关端口回归GREEN、ruff后小提交。

## Task2：实际后端与交接

**Files:** Create `tests/test_tyx_domain_point_graph.py`；Modify `src/phymeo/__init__.py`、README/进度/性能记录。

- [ ] 公共导出RED、实际多域参与者测试；分别interpret/C++/CUDA，逐域完成launch与解析状态证据。
- [ ] 固定小负载性能观察、明确局部算子限制；与本轮其他切片统一必要回归/打包/CI/独立审阅及推送。

## 自查

Task2只导出/验证Task1，不新增控制或接触规则；各域状态始终由其端口候选验证。完整双臂案例未实现部分继续留在总目标，不将本图替代SceneStepper的后段子步回滚验收。
