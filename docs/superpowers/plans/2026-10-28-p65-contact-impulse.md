# P6.5 H2b 固定构型联合接触冲量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。继续已授权范围，主实施者测试先行；本轮末一次整分支独立审阅覆盖所有新增切片。

**Goal:** 联合求法向非穿透与各向异性库仑摩擦，并由实际多域响应验收一次总冲量。

**Architecture:** 不可变PointGraph上的稠密完整响应＋固定接触基块GS；局部切向调用H2a，整图独立Signorini/摩擦/能量门禁，原子返回所有域候选。

**Tech Stack:** Python3.11/3.12、NumPy主机求解，底层保持明确tyx三模式。

**Spec:** `docs/superpowers/specs/2026-10-28-p65-contact-impulse-design.md`。

## Global Constraints

- `codex/p6-5-handoff`；截止2026-10-28 00:00UTC。仅共点、固定构型、零恢复瞬时问题；不接受位置乘子，不修改旧P6程序。
- FP64自有输入/输出；法向单位误差1e-12；默认速度容差1e-10m/s、响应容差1e-12、位置1e-9m、200轮。实际tyx预设速度/响应容差2e-5；不随实验倒推。
- 响应对称化明确记录，不截断特征值；全图实际响应最终验收，不用矩阵预测冒充设备结果。
- 没有完整时间步/CCD/控制验收声明，不能以本小例替代P6.5综合案例。

## Review Focus

- 独立接触各自解会漏共享响应：Task1双侧移动关节/节点小例。
- 速度冲量被当迭代增量重复施加：Task1从同一原快照一次总冲量独立算状态。
- 静/动摩擦半径或方向正确但normal互补失败：Task1全图独立残差和低预算拒绝。
- FP32响应的非对称误差被隐藏：Task2实际raw查询/预设容差和最终响应审计。
- 共点假设未成立仍算成功、或最后域候选失败：Task1几何拒绝/原子性反例。

## Task1：主机联合冲量与物理拒绝门禁

**Files:** Create `src/phymeo/constraints/contact_impulse.py`、`tests/test_contact_impulse.py`。

- [x] 先写平面/分离/静滑解析、双侧有限质量移动关节夹持冲量与低μ对照、共享响应、一次总冲量、能量/基座反力、非法/预算/失败原子性测试并确认RED。
- [x] 实现ContactImpulseResult、基/响应组装、块GS和独立实际响应验收；GREEN及旧摩擦/端口回归、ruff，小提交。

## Task2：实际参与者与公共接口

**Files:** Create `tests/test_tyx_contact_impulse.py`；Modify `src/phymeo/__init__.py`、README/进度/性能记录。

- [x] 公共导出RED；两真实机器人＋节点的固定构型夹持冲量与低μ测试，全部模式逐域执行证据、配置不变及能量/速度解析门槛。
- [x] 记录固定小负载构造、响应组装/总求解延迟、迭代/残差与失败；明确主机/设备边界，不优化。
- [x] 与本轮H1a/H1b/H1c/H2a统一必要全回归、打包/源码外smoke、一次独立整分支审阅、准确提交CI与推送；未执行不能勾选。

## 自查

H1c供全图线性响应和原子候选，H2a供固定normal切向圆盘解，H2b只组装同一瞬时问题。新模块没有内部物理时钟、位置修正、接触缓存或控制记忆；后续完整时间积分仍须另冻结，不能把H2b组装直接称连续仿真。

阶段工程收口：修复源码1e5314248fea397ca14038fe953cfc9dc7d115bd完整测试、唯一审阅修复、源码外wheel、串行观测和准确CI/Documentation均通过，见[基础验收](../../benchmarks/p65-foundation-acceptance.md)。这里只勾选本独立基础计划，不表示完整P6.5完成。
