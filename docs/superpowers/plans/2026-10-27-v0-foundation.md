# phymeo v0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在北京时间 2026-10-27 08:00 前推进可验证的基础架构和部分数学求解，逐批测试、提交与推送；未完成的任务留给后续，不为赶时间伪造能力。

**Architecture:** core 定义状态/模型/能力与结果，NumPy 提供独立参考，tyx 通过公开适配器执行计算。基础积分与 XPBD 约束组合成候选 step；runtime 决定接受与回退。完整机器人、强耦合、AI 训练、通用碰撞按后续任务扩展。

**Tech Stack:** Python >=3.11、NumPy、pytest、uv、pip/venv、GitHub Actions；tyx 为固定提交的可选集成。

**Spec:** [架构设计](../../architecture/overview.md)；[分工与测试](../../development/team-and-tests.md)；[环境与 CI](../../development/environment-and-ci.md)。

## Global Constraints

- Python >=3.11；CPU 参考路径支持 Windows/Linux；CUDA 支持必须由实际硬件测试证明。
- phymeo -> tyx 单向依赖；不导入 tyx 私有模块；tyx 不依赖 phymeo。
- NumPy 是参考数据与 CPU 基础；PyTorch、tyx 和 GPU 工具链按可选能力分离。
- 首版仅承诺 FP32 tyx 计算；NumPy FP64 可作独立参考，不等于 tyx FP64 支持。
- 不复制 Newton 源码、注释或测试实现；按数学推导与独立验收构建。
- CI 不用跳过结果冒充 GPU 通过；未收敛、缓冲区溢出、能力不支持必须显式报告。
- 不发布软件包、不合并主分支；验证后可提交与推送专用 codex 分支。

## Review Focus

五类重点均分配到以下任务：空模型/零元素设备限制（2/5）；重合约束端点（4）；输入/输出数组别名（2/3）；AI/候选失败后的状态回滚（6）；无硬件或全跳过被误报 GPU 成功（5/7）。

## 执行和截止规则

开始与每个任务边界读取 UTC 时间，截止为 `2026-10-27T00:00:00Z`。临近截止只完成当前可验证小任务，停止扩张范围；08:00 后整理、测试和推送已完成工作，记录未完成项并停用 heartbeat。每项完成均更新 `docs/progress.md`；本文件的未完成项不代表要求越过截止继续实现。

空仓库无既有提交或用户改动，无法从 HEAD 创建 worktree；使用当前专用项目目录和 `codex/architecture-v0` 分支，后续有共享开发再创建隔离 worktree。

## Task 1: 设计与仓库基线

**Files:** README、docs/architecture、docs/development、scripts/check_docs.py、.github/workflows/docs.yml。

**Interfaces:** 产生本计划和图表；不产生物理运行时 API。

- [x] 完成离线 HTML/SVG/PNG 与模板许可，移动端和桌面检查标签/连线。
- [x] 实现标准库文档检查，运行并检查所有本地链接与 SVG 引用。
- [x] 新建专用分支、配置用户指定 remote，提交设计包并推送。
- [x] 记录验证方式；尚未获取远程 Actions 证据，不声称通过。

## Task 2: 包、状态与数据合同

**Files:** pyproject.toml、uv.lock、requirements/dev.lock、src/phymeo/__init__.py、core/model.py、core/state.py、core/result.py、tests/test_core.py。

**Interfaces:** `ParticleModel(inverse_mass, gravity)`；`ParticleState(position, velocity)`；`StepInputs(force)`；`StepReport` 与 `StepResult(state, report)`。数组 `(n,3)`，逆质量 `(n,)`；模型和输入 shape/dtype 约束统一。

- [x] 先写非法质量/形状/NaN、空系统、状态复制无别名等 17 项细分测试；观察缺失实现导致失败。
- [x] 实现验证与复制合同，固定点模型语义按 spec；不引入 solver 依赖。
- [x] 建立公开依赖锁与 pip 导出，uv/pip 分别安装并运行全套现有测试。
- [x] 从源码树外验证 wheel 导入；提交此独立交付。

## Task 3: 半隐式积分与首个可运行例子

**Files:** dynamics/integrate.py、solvers/semi_implicit.py、tests/test_semi_implicit.py、examples/free_fall.py。

**Interfaces:** `SemiImplicitSolver.step(model, state_in, inputs, dt) -> StepResult`，消费 Task 2，返回独立候选。

- [x] 写并运行自由落体离散解析解、外力/质量、固定点、非法 dt、输入不变测试，观察预期失败。
- [x] 实现 NumPy 数学基线；不把显式 Euler 混入速度优先的更新。
- [x] 运行完整测试与无图形 free-fall 示例，核对有限结果；提交。

## Task 4: XPBD 距离与简化平面接触

**Files:** constraints/distance.py、geometry/plane.py、solvers/xpbd.py、tests/test_xpbd.py、examples/hanging_chain.py。

**Interfaces:** `DistanceConstraints(indices, rest_length, compliance)`；`XPBDSolver(constraints, iterations, ...)` 使用同一 step 边界。Plane 明确 height/normal；首批可限定 y-up 并显式拒绝不支持的选项。

- [x] 先写单约束解析 delta_lambda、逆质量分配、固定点、零长度端点、compliance/h²、未收敛报告测试；非零伸长的正确柔顺解必须可被接受。
- [x] 实现每子步清零、子步内累计乘子的 XPBD；约束顺序确定；投影后用 (x_corrected-x_start)/dt 重建速度，以 C+alpha_tilde*lambda 而非 C 评估柔顺收敛。
- [x] 加入平面非穿透基线与固定对象测试；活动平面接触截去负法向速度，保留切向速度，说明无摩擦/CCD 局限。
- [x] 运行链示例和全套测试，确认输入不变；提交。

## Task 5: tyx 公共 API 适配

**Files:** backends/tyx.py、requirements/tyx.txt、tests/test_tyx_backend.py、scripts/check_public_tyx_imports.py。

**Interfaces:** 与 Task 3 相同的积分语义；显式 mode 与 availability；缺依赖时有清晰错误，参考路径仍可用。

- [x] 在安装固定 tyx 后，先写 NumPy 解析对照、FP32 限制、空数组绕过、可选依赖缺失测试并观察失败。
- [x] 用 `tyx.kernel` 等公开 API 实现真实积分；不使用伪后端或私有模块。
- [x] 分开报告 interpreter、CPU 编译和 CUDA；仅实际执行的路径算通过。
- [x] 完整 suite + public-import guard；提交并记录 tyx SHA、工具链和 skip 原因。

## Task 6: 候选验收与数学回退协议

**Files:** core/capabilities.py、runtime/stepper.py、tests/test_stepper.py。

**Interfaces:** `Stepper(primary, fallback=None).step(...) -> StepResult`；solver 的 `problem_signature()` 比较冻结的物理约束、平面/接触定律和语义版本，忽略 backend/迭代次数/容差。接受策略支持有限性/报告状态；首版 fallback 最多一次且从原快照开始。

- [x] 先测试拒绝 NaN/未收敛候选、恢复原始输入、fallback 失败可观察、零次或一次重算；有约束 XPBD 回退到自由积分器必须在调用前失败，相同物理配置更高迭代预算允许回退。
- [x] 实现协议与明确的受控失败；用真实数学 solver 集成，不把 mock 通过等同模型有效。
- [x] 加入能力不匹配拒绝；暂不宣称已有训练网络、全部隐状态或随机回放能力。
- [x] 全套测试通过，更新能力表并提交。

## Task 7: CI 与交付收口

**Files:** .github/workflows/ci.yml、README、docs/progress.md、相关环境/测试文档。

**Interfaces:** CPU 套件、wheel smoke、依赖导出与文档检查是实际命令；GPU 工作流只有硬件和授权配置齐备才启用。

- [x] 增加 Windows/Linux 与 uv/pip 的 CPU 检查，固定 Action SHA。
- [x] 验证导出一致、wheel 隔离安装；GPU 需求用显式 require-device，不能全 skip 过关。
- [x] 做一次独立全分支审阅；修复重要缺陷时先添加会失败的回归测试。
- [x] 运行完整验证，推送并比较远程 SHA；记录不可访问的远程 CI，不猜测状态。

## Task 8: 独立刚体姿态与角动力学基础

**进入条件：** Task 1–7 完成，审阅三个 P2 已修复且复核；01:55 距截止仍有约六小时。该任务来自既有后续候选“真实刚体旋转”，保持单独可验证交付。

**Files:** core/rigid.py、dynamics/rotation.py、solvers/rigid_semi_implicit.py、tests/test_rigid.py、examples/spinning_body.py；StepResult 改为状态泛型，保持已有粒子运行行为。

**Interfaces:** `RigidBodyModel(inverse_mass, inertia_diagonal, gravity)`；`RigidBodyState(position, orientation, velocity, angular_velocity)`；`RigidBodyInputs(force, torque)`；`RigidSemiImplicitSolver.step(...) -> StepResult[RigidBodyState]`。数组都是拥有的 FP32/FP64 缓冲；orientation `(n,4)` 为 `(x,y,z,w)`、body-to-world；线速度、角速度、外力和力矩均在世界坐标系，惯量为质心处体坐标主轴对角量，单位 kg·m²。动态惯量正且满足主惯量三角不等式；固定物体位置/姿态不变、两种速度归零。

**Numerics:** 平移沿用速度优先积分；角加速度在体坐标按 Euler 刚体方程计算，包含陀螺项，再转回世界坐标更新角速度。新世界角速度产生指数映射四元数并左乘旧姿态，归一化输出；不对四元数直接加欧氏位移。数值非有限时拒绝并保留原状态副本。一般非球形自由转动是第一阶离散近似，不宣称能量/角动量精确守恒。

**Boundary:** 先验证独立刚体自由运动，无接触/关节/广义坐标/耦合或 GPU 刚体承诺。capabilities 的 state_kind 为 rigid_bodies；现有粒子 Stepper 在调用前明确拒绝该类，不通过鸭子类型丢弃角状态。后续统一调度需另扩展完整姿态验收/回退合同。

- [x] 先写惯量/四元数合同、球形惯量常角速度解析解、主轴恒力矩离散解、旋转坐标系力矩、陀螺项、固定刚体及非有限结果回归；观察缺失实现失败。
- [x] 实现独立数据与旋转数学、角动力学基线；保持已有粒子 API/测试通过。
- [x] 运行无图形旋转示例，核对单位四元数和解析误差；更新能力说明及 CI 示例。
- [ ] 验证 Python 3.11/3.12 CPU 与包、代码审阅新增切片；提交并推送。

## 后续候选（有余量才按完整验证单独立项）

球体碰撞、布片拓扑/弯曲、耦合端口是下一批。广义坐标机器人、FEM、本构学习、GPU 图捕获和机械臂抓布不属于本夜间任务保证完成的范围。需要时可推进一个已明确合同的小切片，必须先更新计划与测试，不临时堆砌完整子系统。
