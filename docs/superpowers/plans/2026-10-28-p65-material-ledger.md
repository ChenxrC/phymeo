# P6.5 距离材料账本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline with TDD. Same authorized branch, one final fresh review for this independent stage.

**Goal:** 独立认证真实距离材料路径的输入/预测/逐边原式/速度重建并报告物理记账。
**Architecture:** MaterialPathLedger拥有原始快照；Fraction与sqrt包络只验证native候选，逐边严格连续、绝对累计误差门禁；物理能量与角动量报告不替代下一接触outer门禁。
**Tech Stack:** Python3.11+、NumPy、现有固定tyx公开API；原FP64存储和实际FP32设备。
**Spec:** [材料账本设计](../specs/2026-10-28-p65-material-ledger-design.md)。

## Global Constraints

不更改积分、原StepSettings容差、trace18列、FreeParticleLedger/旧接触允许范围。不得host生成候选覆盖设备；不把能量/角动量损失当接触摩擦。两固定compliant边同样必须满足累计标量增量/乘子原式。单次fresh审阅及必要TDD修复。性能只观测，C++长期TYX-003不重复无修复压力。提交push专用分支、不合主分支、不发包。

## Review Focus

- 大坐标相消或假预测能否绕过原绝对速度预算；精确存储量检查。
- 全固定compliant边、零denominator哨兵、正分母coincident必须分别处理，不以w0豁免标量方程。
- 逐边/跨sweep连续性和候选末态严格绑定，额外隐藏位置/速度载荷必须拒绝。
- 参数、diagnostics与expected Fraction原请求的独立所有权，不能用浮点输入差代替请求。
- 不把全步能量/角动量报告误标为已通过物理守恒；下一outer还需要独立预算与细化。

### Task 1: 生产账本与独立物理/事务测试

**Files:** Create `src/phymeo/validation/material_path.py`、`_material_path_math.py`、`tests/test_material_ledger.py`、`tests/test_tyx_material_ledger.py`；Modify `src/phymeo/__init__.py`、`scripts/smoke_installed.py`。
**Interfaces:** MaterialPathLedger构造/require精确按spec；输出scope、residuals/max_residual、material_impulses/support_impulses、conversion_displacement/projection_rounding_displacement、energy与momentum子mapping，全部owned只读。残差名称input/prediction_velocity/prediction_position/alpha/edge_equation/lambda_update/direction/projection/rounding_budget/velocity/constitutive。

- [x] 新测试延用真实run_material数据，公开类缺失RED；三节点路径、None/empty/fixed/coincident通过；单边解析冲量/能量，schema/shape/context/sweeps/布尔/别名配置拒绝。初始force/position后改不能污染ledger。
- [x] 实现owned构造与math helper（q保留Fraction、sqrt与保守区间、显式FP32转换），独立预测/输入/逐边/最终位置/速度/本构门禁；精确比较再输出上界，不按边数降低预算。fixed compliant标量与零哨兵是必需测试。
- [x] 对before/after/direction/dlambda/lambda/alpha/prediction/candidate、隐藏load/重复请求、累计小误差/大坐标错误构造RED；实施最小修复，正常控制保持原状态。
- [x] 输出K/U/重力/分解端点功与线角动量、材料贡献；旋转硬边h=.1/.05/.025符合已知离散损失，软边K增/U降、总能量降；硬边初态不满足长度造成的正能量增量仍明确报告，不把equations通过称能量合格。
- [x] actual interpret/CUDA与有限C++短例，FP32转换/fixed host恢复记录；SceneLoadReplay同初态不同请求/两节点小三角及真实机器人旁域相反输入账本例（非接触求解）；后子步污染trace/close失败回滚history并retry=fresh。公开wheel smoke。提交。

### Task 2: 工程门禁与观测交付

**Files:** Create `scripts/benchmark_material_ledger.py`、阶段acceptance/engineering/review、PERF记录；更新NUM-005/progress/continuation。
**Interfaces:** 使用Task1公共require与真实参与者，不重做native trace或旧接触。

- [x] 唯一fresh最终review，Important/Critical先RED后修复；必要核心/actualinterpret/CUDA全量、两CPU、Linux新核心、boundedCPP、ruff/docs/import/compile/uvpip、build/源码外wheel；记录准确版本与实际计数。
- [x] 所有数值进程退出后NumPy→interpret→CUDA串行同一三节点两边路径与认证，各3×100；记录端到端及审计分项（相同请求/范围），不优化/不宣称Warp差距。
- [ ] 遗留/性能/tyx问题分目录，提交push核对SHA/精确CI、更新既有ACTIVE自动任务。后续先确定材料/接触outer反作用各一次、能量/角动量误差及细化门槛，不直接放行旧自由节点程序。

Builder check缺项已裁定并写回spec：每条边每sweep累计edge residual/L_e与D_e*lambda更新差/L_e，固定点也不能豁免。没有待用户决定项。
