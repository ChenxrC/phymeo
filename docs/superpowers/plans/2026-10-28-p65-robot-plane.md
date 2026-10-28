# P6.5 机器人固定平面重放 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 原机器人积分器通过重放自动求材料点/固定平面的接触，最终构型载荷、材料速度及广义动力学共同闭环。
**Architecture:** 固定M0/b0/A0，迭代最终J1.T*p作为effort积分；原SceneStepper事务和actual tyx候选不变；有理审计存储系数下的动力学/输入/q运动学。
**Tech Stack:** Python3.11+、NumPy、Fraction、原tyx后端；uv/pip。
**Spec:** [冻结设计](../specs/2026-10-28-p65-robot-plane-design.md)。

## Global Constraints

BASE cfa76bd，沿用用户指定codex/p6-5-handoff；不改旧积分器、容差或tyx pin。一步M0/b0/A0取q0/v0；映射、材料速度取最终q1。reference_angular_speed默认1rad/s，移动关节用outer.reference_speed。世界弦速度不是材料速度。只有固定平面，本轮不冒称机器人-节点/CCD/夹爪。一次最终review；Windows C++长期TYX-003仍开放，不重复失败压力；性能只观察。专用分支push已授权，不merge/publish。

## Review Focus

- M0和J1构型混淆；最后一次映射未重算却报接触闭环。
- 巨大初速/输入相消及高条件数让同源浮点残差错误为零。
- 零法向/切向响应、秩亏切向块、跨接触同域反作用重复；不允许伪逆/相对地板掩盖。
- 弧运动的末材料速度与弦速度混用，功/动能错误标签。
- 显式注册状态/history、后段子步或关闭失败、借用factory、返回对象污染破坏事务。

### Task 1: 动态机器人固定平面与精确账本

**Files:** Create src/phymeo/constraints/robot_plane.py、src/phymeo/solvers/robot_plane_contact.py、src/phymeo/solvers/_robot_replay_ledger.py；Modify src/phymeo/__init__.py；Create tests/test_robot_plane_contact.py。
**Interfaces:** RobotPlaneContacts及RobotPlaneContactProgram精确签名见spec。内部FrozenRobotLedger(model,state,inputs,h,settings,reference_angular_speed)保存M0、精确外力投影和bias；暴露solve、joint_error、trial_audit、mapping_error给workspace，不作为公开API。

- [x] 写公开绑定/所有权/单位/错误族测试；写单移动支撑、分离、初始穿透/预算拒绝；运行新文件Expected缺失API RED。
- [x] 写单转动独立数值例：q0=-.3,v0=1,h=.15、局部端点(0,-1,0)、平面-cos(.2)，Expected q1=-.2、v1=2/3、p=M*(-1/3)/sin(-.2)；检查J1反力矩、材料速度≠弦速度和冻结动能变化。Expected缺失实现 RED。
- [x] 实现冻结正定质量的有理求解/精确输入和动力学/q审计，绑定、GS重放和最终映射/接触门禁。Expected新文件GREEN。
- [x] 增加大数相消、高逆惯量、q增量吞掉、违反原机器人动力学的base、零响应、污染/多子步/history/关闭失败回归；先RED后必要修复。跑机器人/SceneStepper/load_replay/节点平面相关回归和ruff，Expected通过后提交。

### Task 2: 材料旁域、完整时域、actual后端及工程交付

**Files:** 扩展tests/test_robot_plane_contact.py；Create tests/test_tyx_robot_plane_contact.py、scripts/benchmark_robot_plane_contact.py；Modify scripts/smoke_installed.py；新增benchmarks验收/工程/审阅、PERF-012及续作。
**Interfaces:** Task1程序直接包ArticulatedCouplingProgram或TyxArticulatedCouplingProgram空旧接触，公开诊断robot_plane_contact/执行计数；原布域保留原材料求解。

- [x] 独立两移动关节T2s、h=.01/.005、mu=.6/.2静置/滑移、共享祖先双接触顺序和布材料旁域；按独立解析运动/反力矩而非调用同一实现当oracle。Expected新验证通过，发现新正确性问题必须TDD修复。
- [x] actual interpret/CUDA静置与转动，Windows C++仅有限2次重放的支撑短测，明确长期门禁未通过；源码外wheel增加公开API smoke。Expected非零真实launch，native候选不覆盖。
- [x] 一次fresh最终review BASE后全范围，Critical/Important一次TDD修复；修复后必要全量CUDA/core311/nativepip312、interpret/短C++、uv构建/源码外wheel、ruff/公共边界/docs/compileall/dependency check。Expected通过。
- [x] 全数值进程退出后NumPy/interpret/CUDA各3×100串行小负载，记录真实replay/launch和全部失败；不重试旧C++长期，不优化。Expected成功或如实阻塞记录。
- [ ] 更新进度/工程/遗留/续作与唯一审阅结论，提交push用户专用分支，核对远程SHA/准确CI。完整P6.5仍继续，下一步机器人-节点双向碰撞。

Self-check: Task2只消费Task1公开包装器与原base合同，无SceneStepper修改；全局审计、量纲、生命周期、实际后端都有明确测试归属。用户已选持续自主执行，采用inline加一次fresh final review，不重复请求许可。

本地验收：产品a4b1ad4；CUDA1725，core311/nativepip312各1441，actualinterpret3，boundedCPP1，114文件wheel，900次串行观测。新交付head的远程SHA/CI等待推送后核对；不把尚未执行CI标成功。
