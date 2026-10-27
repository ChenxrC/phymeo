# P6.5 H3a 有限驱动控制 Implementation Plan

> 使用 superpowers:executing-plans，主实施者inline TDD。新范围基线6fb29ff，旧基础唯一审阅已关闭，不重新审阅旧修复；本新模块完成后一次fresh review覆盖新增范围。

**Goal:** 有限effort的隐式自由预测控制律，真实参与者仍负责状态推进。
**Spec:** [合同](../specs/2026-10-28-p65-bounded-control-design.md)。
**Deadline:** 2026-10-28 00:00UTC，不自动延长。

## Review Focus

- 零增益与零限幅轴不能让QP除零或丢失与其他轴的耦合。
- 原始M的非对称度、控制律和动力学方程都要独立复核，不能只验证对称近似方程。
- 输出预测不等于实际状态；实际tyx必须查询M/b、执行动力学并检查launch与状态误差。
- 饱和不是事后裁切无界解；非对角质量有独立活动集oracle。
- 不承诺含接触或完整机器人稳定；没有目标/控制状态机历史的隐式全局变量。

## Task1 数学核心

Files: Create `src/phymeo/control/__init__.py`, `src/phymeo/control/bounded_pd.py`, `tests/test_bounded_pd.py`。

- [x] 先写单轴解析/饱和/零增益/耦合活动集oracle/拒绝/所有权测试，运行确认缺模块RED。
- [x] 实现严格输入合同、盒QP、独立控制律和原M方程验收；运行整个新测试文件GREEN及ruff，提交。

Expected: 配置错误明确抛TypeError/ValueError，数值/预算失败SolverFailure；无裁切物理状态。

## Task2 实际参与者与工程

Files: Create `tests/test_tyx_bounded_pd.py`; Modify公共导出/README/进度；性能观测单独记录。

- [x] 公共导出RED→GREEN；NumPy实际自由预测和tyx三模式实际M/b+predict验证，资源关闭、输入不变、后端launch证据。
- [x] 新控制小负载串行观测和范围记录；旧模块保持回归，不测Warp、不优化。
- [x] 本新范围一次独立审阅、修复、必要全回归、两Python、wheel、准确Git/CI；未执行不勾选。

Expected: 实际tyx预设mass_tolerance=2e-5，状态/控制比较atol=rtol2e-5；不修改已冻结接触容差，不写回本函数预测。

收口：源码c7935e3，CUDA1521、两CPU核心各1249、实际C++2、核心/解释器37、107文件源码外wheel、600次串行主机观测通过；准确提交CI40600274542、Documentation40600274559成功。一次审阅1Important已按正负坐标反例修复，无延期Minor。证据见[工程记录](../../benchmarks/p65-control-engineering.json)与[审阅裁定](../../benchmarks/p65-control-review.md)。完整P6.5目标仍未完成，本计划仅H3a自由预测控制核心。
