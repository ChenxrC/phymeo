# P6.5 节点平面动态接触 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 自动求节点-静止平面支撑/摩擦载荷，原求解器重放后同时满足材料、运动学及接触原式，保持完整事务。
**Architecture:** 冻结节点J=I，位置级非穿透和末材料速度库仑摩擦；GS候选载荷每轮经SceneLoadReplay完整重算；以SceneProgram包装复用原SceneStepper。
**Tech Stack:** Python3.11+、NumPy FP64审计、原tyx FP32 program，uv/pip。
**Spec:** [冻结合同](../specs/2026-10-28-p65-plane-replay-design.md)。

## Global Constraints

继续用户专用codex/p6-5-handoff，BASE1ce5c03；不改旧积分器/事务，不扩大到机器人点/动态平面/网格。mu静动相同；无精确TOI或CCD声明。实际后端不得host覆盖；TYX-003 Windows C++长期阻塞不算通过，依赖pin不变。性能只观测。一次fresh最终审阅，不逐任务复审；专用分支push已授权，不merge/publish。

## Review Focus

- 非闭合接近时的离散末负法向速度是否被错误解释为冲量残差或CCD通过。
- 同节点/多平面聚合及载荷相消，反作用和功只计一次。
- 材料完整重算、历史一次推进、base factory生命周期和后段关闭失败恢复。
- 大背景坐标/输入和极小质量导致虚假闭合或载荷表示损失，必须拒绝。
- 内层accepted但运动学不兼容、摩擦方向不对或未收敛时不能提交。

### Task 1: 绑定及原程序重放的动态接触

**Files:** Create src/phymeo/constraints/particle_plane.py、src/phymeo/solvers/particle_plane_contact.py；Modify src/phymeo/__init__.py；Test tests/test_particle_plane_contact.py。
**Interfaces:** ParticlePlaneContacts及ParticlePlaneContactProgram按spec精确签名；SceneStepper输入输出不变，inner_settings.substeps必须1，history schema继承base。

- [ ] 写缺失公开API、参数所有权/合法性、无摩擦接近/支撑/分离、mu静置/滑移、双平面聚合、全帧失败与factory关闭测试；运行新测试Expected缺失实现失败。
- [ ] 实现固定绑定、借用base wrapper、GS与最终独立门禁，接触位置阶段重算，速度阶段不再施冲量。Expected上述测试通过。
- [ ] 跑SceneStepper与load_replay回归、ruff；Expected通过。提交，记录RED/GREEN。

### Task 2: 材料、实际tyx及工程验收

**Files:** 扩展tests/test_particle_plane_contact.py；Create tests/test_tyx_particle_plane_contact.py、scripts/benchmark_particle_plane_contact.py；Modify scripts/smoke_installed.py；新增docs/benchmarks/p65-plane-*、性能记录和续作入口。
**Interfaces:** Task1自动接触载荷；复用ArticulatedCouplingProgram/TyxArticulatedCouplingProgram空旧接触，节点force和原机器人自由状态共同真实推进。

- [ ] 增加真实布距离材料、两个dt解析完整时域、隐藏history、最后子步失败/close重试、大坐标拒绝、实际tyx静置/滑动/material；先RED后实现必要补充，Expected全部门禁成立。
- [ ] 一次fresh最终review审阅本计划BASE后全范围；Critical/Important一次TDD修复，重新必要测试，旧阶段不重复审阅。
- [ ] 全量CUDA、core3.11与nativepip3.12、interpret、短C++（不超过已知生命周期边界且不冒称长期）、uv build/源码外wheel/ruff/public imports/docs，Expected通过；已知C++长期失败持续记录。
- [ ] 全数值进程退出后NumPy/interpret/CUDA各至少2预热和3×100次串行小负载；记录迭代/重放次数、真实launch和全部失败，不优化。
- [ ] 更新进度/工程manifest/续作，提交push专用分支，核对准确远程SHA/CI；只宣布本限定平面阶段结果，完整P6.5继续。
