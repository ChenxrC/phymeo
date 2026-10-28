# P6.5 固定区间载荷重放 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 为动态接触提供不累积时间、不泄漏候选状态且只计一次输入总载荷的可验证重放入口。

**Architecture:** 用显式域/输入字段映射将区间积分载荷变为恒定平均输入；从同一StateRegistry快照重复调用既有SceneStepper。保留原参与者材料阶段、history事务和实际后端执行，不另写时间积分器。

**Tech Stack:** Python3.11+、NumPy2.x、既有tyx pinned三模式、uv/pip。

**Spec:** [固定合同](../specs/2026-10-28-p65-load-replay-design.md)。

## Global Constraints

继续codex/p6-5-handoff，不修改旧积分/SceneStepper行为。FP64存储，tyx真实FP32数值；禁止host候选覆盖。单轮独立最终审阅，不逐Task审阅。不把解析预设接触说成自动几何/夹持。用户已授权push专用分支，不合并/发布；本轮无截止。性能只观测。

## Review Focus

- 大背景率吞掉微小积分载荷：拒绝，不能按大背景放宽。
- 自定义已注册额外状态/history：完整快照，后段失败及修改返回值不能污染后续trial。
- 非线性M/J与材料内部乘子：p→0→p等于独立新实例，不依赖负载撤回。
- 资源关闭失败及复入/绑定改变：沿用事务边界，不能有已泄漏的候选。
- 输入字段的单位声明与实际后端：不能隐式猜测位置字段，也不能用CPU试算假装设备结果。

### Task 1: 重放入口和事务合同

**Files:** Create src/phymeo/runtime/load_replay.py；Modify src/phymeo/__init__.py；Test tests/test_load_replay.py。

**Interfaces:** SceneLoadReplay构造、evaluate、snapshot、close；ReplayTrial.result/integrated_loads/effective_inputs/duration，按Spec精确语义。

- [x] 写解析单/多域、p→0→p、所有权、channels错误、输入表示丢失、自定义字段/history、后段failure/close/重试测试。
- [x] 跑pytest新测试，Expected:缺少模块/公开API失败。
- [x] 实现原输入快照、显式字段合成、表示检查、调用既有SceneStepper和owned账本。
- [x] 新测试、旧SceneStepper和participant相关回归，Expected:全部通过；ruff通过。
- [x] 提交Task1，记录RED/GREEN及实际覆盖范围。

### Task 2: 动态解析例、实际tyx与工程验收

**Files:** tests/test_load_replay.py新增解析摩擦/真实材料对照；tests/test_tyx_load_replay.py；scripts/benchmark_load_replay.py；scripts/smoke_installed.py；docs/benchmarks/p65-load-replay-*；性能/遗留/tyx问题按证据分类。

**Interfaces:** Task1 evaluate持有整个dt的积分输入；使用原ArticulatedCouplingProgram/TyxArticulatedCouplingProgram空接触模型验证真实机器人/节点输入，不改变其点接触规律。

- [x] h=.01/.005、T=2s解析平面静摩擦/低μ滑动；布XPBD、非线性机器人基准重放对照；实际tyx三模式载荷与launch。
- [x] 核心全量/全量CUDA、3.11与nativepip3.12、uv build、源码外wheel、公用边界和docs检查，Expected:通过。
- [x] 对本计划BASE后的新代码做一次fresh review，重要问题一次TDD修复及复验，保留证据；不重审旧阶段。
- [ ] 数值测试进程退出后串行计时：小单域/双域重放，明确包括构造/原program/关闭/验算；不优化，不声称Warp差距。
- [ ] 更新续作入口并push、核对准确SHA/CI。只有本切片验收完成，完整P6.5继续动态接触外层和几何，不能标整体complete。

## 实施结果及外部阻塞

产品3f23378，验收与机器记录见docs/benchmarks/p65-load-replay-*。全部功能/安装测试完成，一次独立review及4个RED回归修复完成。正式计时限定双域小负载（单域解析由单元测试覆盖），NumPy/interpret/CUDA各300次完成；C++在长循环加载DLL失败，未完成原300次门禁，故上方长期计量项仍未勾选。TYX-003保存独立资源生命周期复现及集成诊断，不能以C++2项短测试替代。提交/推送条目在实际核对后才完成；完整P6.5继续独立动态/几何工作，本阶段全后端长期验收仍open。
