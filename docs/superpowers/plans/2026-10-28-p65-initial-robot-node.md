# P6.5 步初机器人—自由节点接触 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task with TDD. Keep the existing dedicated branch.

**Goal:** 完整native重放中的步初共点双向冲量，实际单转子整步角动量与机器人/自由节点反作用闭环。
**Architecture:** 冻结J0/M0与步初作用点；节点force和机器人effort同时积分重放。原q/v方程、接触速度原式、配对矩、冻结能量与末非穿透分别审计，原候选不覆盖。
**Tech Stack:** Python3.11+、NumPy、Fraction、原tyx三后端、uv/pip。
**Spec:** [冻结设计](../specs/2026-10-28-p65-initial-robot-node-design.md)。

## Global Constraints

BASE eb4010ac44ef63e020b0bac2dbdc4d1ecc921739；用户已授权持续P6.5与专用分支提交push，不重复审批，不merge/publish。当前新计划在a45ff89之后继续。一次builder check已返回nothing important left；实施者沿用步初求积选择。一次fresh最终review，TDD修Important/Critical。Windows C++仅有限2replay，TYX-003无依赖修复不重复长期失败。所选粒子域全部inverse_mass>0，且必须遵守自由粒子方程；其他域可独立材料演化。初态闭合、末态非穿透，不称CCD/新接触/完整布/夹爪。

## Review Focus

- 双向载荷只施一次，节点重复行/机器人共享祖先交叉响应不丢；不能用末J1代替步初J0。
- 固定基座开放动量分量与闭合单转子Lz区分；配对矩不能代替实际整步守恒测试。
- 全域粒子原式不能被材料/后处理或大数相消绕过；动能门禁独立于接触功。
- 世界平移/投影/旋转平面的几何表示和真实末穿透不能由同源残差隐藏。
- history/后段关闭/借用factory/返回污染需要完整恢复；零接触不能改变自由候选。

### Task 1: 双向重放与自由节点精确账本

**Files:** Create src/phymeo/constraints/robot_node_plane.py、src/phymeo/solvers/initial_robot_node_contact.py、src/phymeo/solvers/_particle_replay_ledger.py；Modify src/phymeo/__init__.py；Create tests/test_initial_robot_node.py。
**Interfaces:** 公开绑定及program签名按spec；复用FrozenRobotLedger/audit_affine_geometry、SceneLoadReplay、_BorrowedProgram/_disk，不改原积分器。内部FreeParticleLedger(model,state,inputs,h,settings)审计实际节点积分/q/v、kinetic。

- [x] 写binding/类型/所有权、m2机器人+m1节点p=2/3、转子p=.5/Lz=-1、两个共享点p=(2/3,1/3)/Lz=-8两顺序，初始开/穿透与曲面末穿透拒绝；运行Expected缺少API RED。
- [x] 实现相反载荷聚合、J0重放、真实自由粒子/机器人原式与能量/配对矩/末gap门禁；Expected上述GREEN。不得添加位置修正。
- [x] 增加材料不兼容、输入/速度后处理、大坐标/单位/inner尺度/返回污染/历史后段失败回归，先RED再必要修复；Expected相关新旧回归通过。提交。

### Task 2: 时域、实际后端与交付

**Files:** 扩展tests/test_initial_robot_node.py；Create tests/test_tyx_initial_robot_node.py、scripts/benchmark_initial_robot_node.py；Modify scripts/smoke_installed.py；Create阶段验收/工程/审阅、PERF-013及进度续作。
**Interfaces:** Task1诊断initial_robot_node_contact；base空旧contacts，ArticulatedCouplingProgram/TyxArticulatedCouplingProgram。所有初始/末/求积速度明确命名。

- [x] T2s h=.01/.005：m2移动机器人外effort+9.81，m1节点g=(9.81,-9.81,0)，初态共平面；mu1.2静置，mu.2滑移a_x=7.848，p_y=9.81h，p_x=-9.81h或-1.962h，机器人始终静置。Expected独立半隐式解通过。
- [x] 实际interpret/CUDA移动/转子/共享接触与独立材料旁域；C++仅移动短例；安装smoke加公开接口。Expected实际launch>0，无host覆盖。
- [x] 一次fresh最终review BASE到当前HEAD；Critical/Important一次RED→GREEN。修复后必要全量CUDA、core311/pip312、interpret/短C++、ruff/docs/边界/compileall/uvpipcheck、build与源码外wheel。Expected通过。
- [x] 数值进程退出后NumPy/interpret/CUDA各3×100串行观测，保留失败与计数；不优化/不测无修复C++压力。Expected如实报告。
- [x] 更新工程/性能/遗留/续作，提交push专用分支、核对SHA/CI、更新自动续作。完整P6.5继续；未实现范围不能删除。

执行证据：产品aef9b257，单次最终review两Important经五RED修复，四正常控制；完整CUDA1778/core311与pip312各1490、interpret4、boundedCPP1、118源码外wheel、900串行观测通过。见[验收](../../benchmarks/p65-initial-pair-acceptance.md)及其工程/审阅链接。最后交付项在push及自动续作更新后核对，不以旧CI代替新head。

远程闭环：准确f259e69的CI40662503851/Documentation40662503807均success，见CI修复记录。
