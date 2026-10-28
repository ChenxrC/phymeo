# P6.5 初态闭合材料接触 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline with TDD. One fresh final review for this independent implementation stage; the builder design check is separate.

**Goal:** 使同一粒子域的真实距离材料、机器人反作用与接触联合收敛，并独立限制能量和材料角动量误差。
**Architecture:** 新InitialRobotMaterialContactProgram复用原步初几何/载荷重放，MaterialPathLedger逐候选认证；新的精确物理预算取代仅适用于自由节点的K-free比较。旧自由节点程序保持原合同。
**Tech Stack:** Python3.11+、NumPy与现有固定tyx公开后端，FP64存储/FP32计算，Fraction与sqrt包络。
**Spec:** [材料接触设计](../specs/2026-10-28-p65-material-contact-design.md)。

## Global Constraints

不改原积分、tyx pin或StepSettings容差；不覆盖native状态，不删除FreeParticleLedger。预算必须显式给定，不根据观测增大。仅步初闭合、全动态距离材料域；自动几何/CCD/弯曲/真实夹爪仍后续。每次完整重放后同时核对材料、接触、末几何和物理预算。C++只一次有界短例，不重跑TYX-003压力。授权专用分支commit/push，不合主分支/发包。

## Review Focus

- 多行接触/共享节点不得重复计域K/U或重复反作用；原始request而非舍入差形成物理账本。
- 原非接触输入功不得混入接触p，否则会把能量注入藏在外功中；冻结机器人能量范围必须明确命名。
- 材料角动量缺陷按域L1和而非域间相消，直接Fraction核对，不以舍入后的展示报告作门禁。
- 仅速度停止会累积gap；完整T2静态及低mu解析轨迹和步长细化必须保留。
- materials/预算/diagnostics深拷贝与signature，后子步预算拒绝/trace损坏/关闭失败必须恢复完整history/state，未接受trial不得泄漏。
- 旧free程序仍拒绝材料域；任何共用私有hook不能让新物理规则静默污染旧规则。

### Task 1: 真实材料接触程序与独立物理/事务测试

**Files:** Create `src/phymeo/solvers/initial_robot_material_contact.py`、可选`_material_contact_balance.py`、`tests/test_initial_robot_material.py`、`tests/test_tyx_initial_robot_material.py`；Modify `src/phymeo/__init__.py`、必要的旧initial_robot_node_contact私有共用方法与`tests/test_initial_robot_node.py`、`scripts/smoke_installed.py`。
**Interfaces:** MaterialContactBudgets与InitialRobotMaterialContactProgram严格按spec；消费MaterialPathLedger/FrozenRobotLedger/SceneLoadReplay/RobotNodePlaneContacts；产生owned initial_robot_material_contact诊断与正常SceneStepResult。

- [ ] 先写公开类缺失RED：显式budgets/materials构造验证，压缩双节点三h独立p/状态/K+U/original-input反作用，旧K-free误拒负例；少量同域三角/共享点和无材料控制。builder check未裁定的重要问题先写进spec，不猜。
- [ ] 实现owned配置/签名、独立原始快照、每候选robot+material审计、法切GS完整重放；预条件不是材料切线。实现末gap联合收敛，不使用速度-only提前退出。保持旧free回归与执行计数。
- [ ] 对能量/角动量预算与语义写RED：原始外功、单域计数、错误effective功、跨域抵消、display数值舍入边界；先原式通过再物理拒绝的不相容硬边；固定.001Nms角动量限额下旋转硬边h=.1拒绝/.025接受。最小精确helper实现，不改原tol。
- [ ] 冻结T2静置与mu1/mu.1双接触材料场景，h1/16与1/32；独立离散低mu递推及连续cos参照、真实Py/Lz/能量外功/反作用次数；满足spec数值门槛再算通过。交换接触顺序、同节点多平面与共享祖先，不能仅断言solver内部residual。
- [ ] 先RED后实现：missing/mismatched material_path、参数/请求/精度损坏与后子步预算拒绝、trace/close故障，完整state/history rollback及retry=fresh；所有权与base关闭一次。
- [ ] actualinterpret/CUDA覆盖压缩材料、T2支持/低mu与事务；boundedC++只一次小例。安装包smoke引用新公共API。完成测试后commit，不把实验脚本当生产交付。

### Task 2: 独立审阅、工程与串行观测

**Files:** 新阶段acceptance/engineering/review、`scripts/benchmark_material_contact.py`；更新PERF-016、NUM-005、progress/continuation。
**Interfaces:** 仅使用Task1公共程序和真实参与者，不重复旧材料路径/账本阶段验证。

- [ ] 唯一fresh最终review（实际新阶段完整diff），Important/Critical按独立RED最小修复，不派第二轮。完整CUDA、两CPU、actualinterpret与boundedCPP、Ubuntu新核心、ruff/docs/import/compile/uvpip/exports、build/源码外wheel。原始失败与修复后结果分开记录；源码hash必须最终版。
- [ ] 数值/review全部退出后NumPy→interpret→CUDA串行各3×100，压缩边同初态完整outer、native执行、物理门禁与关闭都计时；记录迭代/replay/launch/传输及预算范围。精度不同不称加速比，性能只观测。
- [ ] 更新真实边界和后续几何/CCD路线，commit/push核对远程SHA和准确CI，更新既有heartbeat。完整P6.5仍需后续弯曲/自碰撞/有限单爪/控制/双臂回放，不提前complete。
