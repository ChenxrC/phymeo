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
- [x] 验证 Python 3.11/3.12 CPU 与包、代码审阅新增切片；提交并推送。

## Task 9: 布片拓扑、面积质量与 XPBD 演示

**进入条件：** Task 8 完成并独立审阅无阻断；来自已有“布片拓扑/弯曲”候选中的拓扑小切片。

**Files:** builders/cloth.py、tests/test_cloth.py、examples/cloth_patch.py、相关 README/CI/能力说明。

**Interfaces:** `cloth_grid(rows, columns, spacing=..., density=..., structural_compliance=..., shear_compliance=..., pinned=..., origin=..., dtype=...) -> ClothPatch`。输入尺寸至少 2×2，点按 row*columns+column 编号，在 XZ 平面展开、+Y 法向。每格两三角形，三角形面积质量按三等份集中到顶点；返回只读 triangles/vertex_mass、ParticleModel、初始 ParticleState 和 DistanceConstraints。固定点逆质量为 0，但 vertex_mass 保留物理面积质量以便审计总质量。

**Constraints:** 相邻行/列为 structural，格子两条对角线为 shear；无重复边，顺序确定。静止长度来自可表示的初始几何；量化后退化的网格、非有限或不能表示的正质量/逆质量/柔度明确拒绝。柔度是逐边参数，不冒充由 Young/Poisson 标定的连续材料。

**Boundary:** 首批仅拉伸/剪切网络，没有弯曲能、自碰撞、面积/体积保持、FEM 或刚柔耦合。使用现有 NumPy XPBD 和平面接触；不把它声明为 GPU 布料或完整布料本构。

- [x] 先写三角形方向/数量、边去重/静止长度、面积质量守恒、固定点、dtype/输入范围、退化几何和复制边界测试，观察缺失接口失败。
- [x] 实现独立 builder 与明确的结构/剪切柔度分配，保持 solver 不依赖 builder。
- [x] 运行小布片悬挂/平面接触实例，逐步检查 accepted、固定点与有限性；记录容差、步长及迭代成本。
- [x] 跑全套、多 Python、wheel 与新增切片审阅；更新两人模块分工并提交推送。

## Task 10: tyx 着色 XPBD 距离与平面投影

**进入条件：** Task 9 完成并审阅；把已验证的公开 tyx 计算从自由积分扩展到约束，与用户要求的 Warp 替换目标对应。约束着色属于调度变化，物理签名仍与 NumPy XPBD 相同。

**Files:** constraints/coloring.py、backends/tyx_xpbd.py、backends/_tyx_kernels.py、tests/test_coloring.py、tests/test_tyx_xpbd.py、examples/tyx_cloth.py；相关能力/环境文档。

**Interfaces:** `color_edges(indices) -> EdgeColoring(order, offsets)` 按输入顺序做确定性贪心边着色，同色中没有共享顶点，每边恰好出现一次；输出拥有、只读 int64 索引。`TyxXPBDSolver(constraints=None, *, mode="interpret", iterations=20, tolerance=1e-5, plane=None, reference_length=1.0, cache_root=None)` 与粒子 step/生命周期合同一致，只有 FP32；公开导入仍限制于 backends。每色一批真实 tyx 投影，颜色顺序确定；颜色内并行不使用争抢位置的原子更新。

**Numerics:** 每步清零 FP32 乘子；缩放柔度先在主机 FP64 计算再检查 FP32 可表示性，正值下溢或非有限时显式拒绝，不能悄悄成为硬约束。距离范数用缩放避免平方的无谓溢出。任何投影数值失败保留失败标记并回滚原状态，有限未收敛返回诊断候选；更新位置后重建速度，活动平面法向速度非负。平面高度取不低于真实高度的最近 FP32 值，不能表示则拒绝；主机用原参数与 FP64 几何检查本构残差/穿透。

**Boundary:** 状态在一步内可驻留设备，但首版每次迭代同步到主机检查残差，跨步重新分配/传输；先确保正确性，不声称高性能、图捕获或长期设备常驻。颜色顺序不同于串行输入顺序，不要求非线性多步逐位一致，以小解析解、容差和固定点/接触不变量验收。CPU C++ 和 CUDA 必须实际执行；LLVM 未验证仍标记未验证。无弯曲、自碰撞、刚体约束、梯度或学习网络。

- [x] 先验证颜色覆盖/互斥/确定性/所有权/非法索引；编写单边解析、柔度与 dt、耦合链、重合点、固定点/接触、空集合、dtype、失败回滚和 Stepper 同物理回退测试，观察缺失实现失败。
- [x] 实现颜色数据与公开 tyx 内核、生命周期/空数组/数值检查；解释器测试通过。
- [x] 实测 CPU C++ 和 CUDA 的全部新增后端测试，运行小布片多步对照并记录容差、迭代/传输成本及限制。
- [x] 全套、多 Python/包、公开导入检查和独立新增切片审阅；更新文档、提交、推送并核对远程 SHA。

## Task 11: 粒子与刚体共用的事务步进边界

**进入条件：** Task 10 审阅完成；从架构中的统一候选接受/回退目标继续，先补齐真实刚体的状态合同，不把刚体当作缺失角状态的粒子处理。

**Files:** runtime/state_contract.py、runtime/stepper.py、core/state.py、core/capabilities.py、tests/test_rigid_stepper.py；旋转示例和 wheel smoke 使用共用 Stepper。

**Interfaces:** 保持 `Stepper(primary, fallback=None, ...).step(model, state, inputs, dt)`。按 primary 的 `state_kind` 选择明确的内置状态合同，当前只允许 particles / rigid_bodies；两 solver 必须始终声明相同 kind、物理签名与支持的精度。入口验证 model/state/inputs 类型匹配。Solver protocol 以模型/状态/输入类型参数描述，不能继续把所有实现标注成粒子。

**Ownership:** 粒子快照为 position/velocity/force；刚体快照另包含 orientation/angular_velocity/torque。每次尝试独立复制全部状态和输入；固定刚体同时保持位置/姿态、两种速度归零；候选四元数必须满足已有单位长度合同。失败后从完全相同快照重算一次，接受结果再次复制，杜绝缓存别名。

**Boundary:** 不修改刚体动力学算法；不在此任务添加接触、关节或训练模型。残差仍由数学 solver 报告，未知学习模型的独立物理验证与隐藏状态事务仍未实现。未知 state_kind/历史/梯度组合继续显式拒绝，禁止依赖字段相似来猜测状态类型。

- [x] 先写真实刚体步进/空集合、姿态与角速度非法候选、固定姿态、力矩/全部状态回滚、返回无别名、类型/kind 误配和粒子兼容测试，观察既有粒子专用门禁失败。
- [x] 实现独立状态合同和共用事务机制；旧粒子行为保持通过，刚体协议不丢字段。
- [x] 旋转示例和 wheel smoke 走共用 Stepper；多 Python/原生 pip/全套检查。
- [x] 审阅新增状态边界，更新能力与分工、提交并推送。

## Task 12: 刚体球形几何与无摩擦平面接触

**进入条件：** Task 11 状态事务审阅完成；沿既有“球体接触”候选实现一个清楚限定的几何/响应切片。球与平面单点法向响应经过质心，不需要伪造一般刚体碰撞系统。

**Files:** geometry/sphere.py、solvers/rigid_sphere_plane.py、tests/test_rigid_contact.py、examples/falling_spheres.py；物理签名、导出、CI/包与文档。

**Interfaces:** `SphereShapes(radii)` 拥有只读 FP32/FP64 正半径 `(n,)`；空集合合法。`RigidSpherePlaneSolver(shapes, plane=Plane(), tolerance=1e-6, reference_length=1.0)` 使用刚体 step 合同。球心与质心重合，惯量由 RigidBodyModel 明确给定；不能从几何自动假设质量均匀。半径数量和 dtype 必须与模型/状态匹配。

**Numerics:** 先用已有刚体自由积分预测；动态球若低于 `plane.height + radius`，做位置去穿透。接触法向为 +Y，恢复系数固定 0、无摩擦；活动接触的负法向速度置 0，保留切向速度和自由预测角状态，因为法向冲量经过质心。位置稳定化与冲量速度响应分开，不用去穿透位移/dt 人为注入弹开速度。已经分离的向上速度保持不变。可表示的中心最低高度向上取整并复核真实间隙，不能无声缩小球半径。

**Identity / failure:** 物理签名包含全部半径、平面高度和固定接触定律；不能回退到无接触自由刚体或不同半径问题。固定物体不被移动，若其几何穿透超出阈值返回未接受候选；非有限预测/接触算术返回原状态副本。按长度尺度归一化穿透，StepReport 分别给出归一化残差与米制误差；所有失败由共用 Stepper 保持事务性。

**Boundary:** 本批仅球形边界对静态 y-up 平面，离散步末检测/位置稳定化；没有球球碰撞、任意形状、宽相、摩擦、恢复反弹、CCD、关节或 GPU 刚体。投影可改变势能，不能宣称连续碰撞时间精确或总能量严格守恒。冲量定义参考 Baraff《Rigid Body Simulation II》公开讲义，独立实现此限定算法，不复制求解器代码。

- [x] 先写半径所有权/非法参数、单球塑性碰撞解析速度与旋转不变、不同质量/半径、固定穿透、空系统、非有限回滚、同签名回退和多步静止接触测试，观察缺失实现失败。
- [x] 实现球形几何、严格物理签名与分离的位置/速度接触处理；不改变刚体自由算法。
- [x] 用共用 Stepper 运行落球/旋转示例，检验每步 accepted、最低高度、最终静止法向速度、固定物体与解析姿态；加入 CI/包 smoke。
- [x] 完成多环境/全套/包、独立接触切片审阅并提交推送；明确剩余碰撞边界。

## Task 13: 可选 CUDA 图重放一轮着色 XPBD 投影

**进入条件：** Task 12 完成。Task 10 的真实 GPU 布片每步包含数百轮颜色 kernel 启动；tyx 固定提交的公开 `Graph / cuda_stream` 支持显式 CUDA 图。本任务只减少同一步内的启动开销，不改变方程、颜色顺序、迭代停止规则或状态所有权。

**Files:** backends/tyx_xpbd.py；tests/test_tyx_graph.py；examples/tyx_cloth.py 与独立比较脚本；架构、环境与进度文档。保持 NumPy/core 无 tyx 导入。

**Interfaces:** `TyxXPBDSolver(..., cuda_graph=False)`，严格 bool；True 只允许 `mode="cuda"`，配置错误在创建运行时前报告。默认路径不变，物理签名和能力不变。示例增加 `--cuda-graph`。所有引用只用 tyx 顶层公开 API。

**Execution:** 每步预测并验收有限性，分配约束缓冲后，用独立 stream 和 `Graph(stream=..., mode="cuda", optimize=False)` 记录一轮完整的颜色投影与可选平面投影。捕获只记录不执行，完成后确认 backend 为 cuda；每次迭代 replay 一次，然后仍同步和在 host 验收本构残差。无约束/平面的自由步及空系统不捕获空图。图不跨步保留，乘子/失败标记仍每步清零。

**Ownership / failure:** 图和 stream 均由本次 step 拥有；任意捕获、重放、读回异常先同步已提交工作，再销毁图/stream，最后释放全部设备数组。异常传播，不隐式退化成 launch list、解释器或 NumPy。数学失败仍返回 owned snapshot；Stepper 可进行已配置且同物理签名的一次数学回退。不能把图执行路径当成新物理模型。

**Review focus:** 捕获误执行一轮、读取不同 stream 的未完成结果、异常清理图仍引用的设备内存、空图、跨步误用旧模型/输入/dt。分别用真实 GPU 等价/解析例、资源故障注入和变化输入重复调用测试覆盖。

**Performance boundary:** 比较同场景 graph on/off，逐步断言状态、迭代数和残差一致；计时包含每步分配、传输、捕获和 host 验收，编译/首步单列。记录实际设备与工作量，不承诺通用加速；没有测得收益则保留实验结论而不把默认路径切换过去。尚无跨步驻留、批量残差检查、GPU 刚体或训练模型。

- [x] 为配置、真实图/单轮等价、变化输入重复步、空/自由系统、非有限拒绝与捕获/重放失败资源清理编写测试，先观察缺失接口失败。
- [x] 实现显式公开 API 图路径及生命周期；运行真实 CUDA 和既有全套，保持默认模式行为。
- [x] 增加可复现同场景比较，报告完整步耗时与结果一致性；更新示例/文档。
- [x] 独立审阅图边界，完成多环境/包验证、提交并推送。

## Task 14: 学习候选的独立物理验收与自由粒子回归示例

**进入条件：** Task 13 完成；已有 Stepper 只独立检查结构与固定状态，物理残差仍来自 provider 自报。为架构的学习候选入口增加可执行的验收合同，先覆盖一个有独立离散方程的小问题。

**Files:** core/capabilities.py、core/verification.py、runtime/stepper.py、validation/free_particle.py、导出、tests/test_verification.py、examples/learned_free_particles.py；CI/wheel smoke 与能力文档。

**Interfaces:** `SolverCapabilities.provider_kind` 默认 `"mathematical"`，只允许 `"mathematical" | "learned"`。`CandidateVerifier` 声明 state_kind、problem_signature()，实现 `verify(model, state_in, inputs, dt, candidate) -> VerificationReport`。不可变报告含 accepted、max_residual、max_constraint_error、reason；残差无量纲、几何误差米制。`Stepper(..., verifier=None)` 的任何 learned provider 必须配独立 verifier；所有尝试（含数学 fallback）使用同一 verifier。

**Runtime contract:** 初始化及每步检查 verifier 与冻结的状态族/物理签名一致；先完成既有候选结构与报告检查，再将原始状态、输入和候选的独立副本交给 verifier。验收不得改写要提交的候选或原始快照。验证报告非法、拒绝或超过 Stepper 阈值则不提交，按既有规则从原快照最多回退一次；SolverFailure 属于预期拒绝，其他代码异常传播。成功报告取 provider/verifier 残差与米制误差的较大值，不覆盖 backend/iterations 等来源信息。

**Concrete verifier:** `FreeParticleVerifier(tolerance=1e-5, reference_length=1.0, reference_speed=1.0)`，只匹配无约束粒子 force/gravity 物理签名。用 FP64 独立重算速度优先离散方程 `v'=v+h*(g+w*f)`、`x'=x+h*v'`，固定点保持位置且速度为零；不调用现有 solver。按各粒子的欧氏位置/速度误差分别除以长度/速度尺度，再取最大无量纲误差；非有限与下溢保护沿用已验证数值规则。大坐标或低精度舍入超过指定物理容差时允许拒绝，不能为了放行扩大阈值。

**Learning demonstration:** 示例用 NumPy 最小二乘拟合自由粒子训练样本中的重力/外力响应系数，训练样本由数学基线生成。provider 声明 learned，限定 dt 训练域，持有只读系数且无隐藏状态；预测候选交给独立 verifier。留出初态/力测试正常接受，再注入错误系数和域外 dt 验证数学回退。只是低维数据回归与接入示范，不是完整神经网络、通用 AI 求解器、训练系统或速度优势声明。

**Boundary / review:** verifier 本身是维护者信任的物理逻辑，不是执行不受信代码的沙箱；provider_kind 是显式能力声明，不靠 backend 名称猜测。没有 compliant XPBD 的独立动力学证据、刚体 verifier、学习隐状态/RNG 回滚或梯度支持。重点验收伪造零残差、错误物理签名、验证器篡改副本、错误报告、正常候选和 fallback 同等校验、FP32/64/极端尺度与空场景。

- [x] 写独立合同、伪造候选拒绝/同问题回退、所有权和解析误差测试，观察缺失接口失败。
- [x] 实现 verifier 协议、学习能力门禁和自由粒子验收；保留数学 provider 无 verifier 的兼容行为。
- [x] 编写可复现回归训练/留出验证/故障与域外回退示例，加入公共 CPU CI 与包 smoke。
- [x] 独立审阅学习候选边界，完成多环境/真实 CUDA 回归、文档、提交和推送。

## Task 15: 粒子—刚体同作用点冲量端口

**进入条件：** Task 14 完成；按既有后续候选推进耦合端口。先提供可验证的瞬时交换算子，不把尚未实现的接触/约束求解或多速率调度混进接口。

**Files:** coupling/point_port.py、core/coupling.py、tests/test_point_port.py、examples/point_impulse.py；导出、CI/wheel smoke、架构与分工文档。

**Interfaces:** `ParticleRigidPorts(particle_indices, body_indices)` 拥有等长只读非负 int64 索引；允许重复端口、空集合。`ports.effective_inverse_mass(particle_model, particle_state, rigid_model, rigid_state)` 返回每端口 FP64 `(m,3,3)` 自响应矩阵。`ports.apply_impulses(..., impulses)` 接收每端口世界坐标冲量 `(m,3)`（单位 kg·m/s），冲量施加给粒子，刚体接受反冲量；两状态族必须同 FP32/FP64 dtype，索引和冲量行数匹配。

**Point / response:** 世界作用点取当前粒子位置，刚体力臂 `r=x_particle-x_body_COM`。`J_body=-J_particle`，`L_body=r×J_body`；线速度增量为逆质量乘线冲量，角速度增量为 `R * ((R^T * L_body) / inertia_diagonal)`。姿态与位置均不积分。固定对象的速度/角速度为零、配置不变，其交换载荷仍进入 ledger，代表外部固定约束吸收的反力。

**Effective mass:** `K=(w_particle+w_body)*I - skew(r)*I_world_inverse*skew(r)`，只计算本端口相对点速度对本端口冲量的响应；多个端口共享对象时仍有交叉项，本函数不声称返回全局耦合矩阵。双固定端口 K=0，动态情形用解析值、对称半正定和单冲量实测响应检验。不可表示的数值计算显式报 SolverFailure，不输出 NaN。

**Atomic result:** `ImpulseLedger(particle, rigid_linear, rigid_angular)` 拥有 FP64 只读累计载荷数组；`ImpulseExchangeResult(particle_state, rigid_state, accepted, ledger=None, reason=None)` 拥有两组状态副本。重复索引使用补偿累计避免大冲量抵消吞掉小冲量；全部响应在 FP64 计算，回写原 dtype 后检查有限性。任何响应/转换数值失败同时返回两组原始快照、accepted=False、ledger=None，不能只提交一边。非法输入仍抛 TypeError/ValueError。调用者状态和输入不变。

**Boundary:** 本算子消费已知冲量，不求接触法向、摩擦、约束乘子、位置修正、能量守恒或连续碰撞。没有 dt 参数；力必须由上游按时间积分成冲量，多子步累计与时间区间去重另立合同。尚无整体耦合 Stepper、GPU 耦合 kernel 或学习耦合模型。当前 FP64 中间计算和输出舍入的误差有正常浮点容差，不声称任意尺度精确守恒。

- [ ] 写端口/ledger 所有权和非法输入、同作用点线/角动量、主轴旋转、固定反力、重复/空端口、有效质量解析值、双方数值失败原子回滚测试，观察缺失接口失败。
- [ ] 实现公开端口和 FP64 补偿累计/响应，不改变现有单状态族 Stepper。
- [ ] 提供离心作用点冲量示例和独立动量审计，加入 CI 与安装包 smoke；更新物理边界及 A/B 分工。
- [ ] 独立审阅符号/坐标/惯量/事务性，完成全套、多环境、当前源码 wheel 验证，提交推送。

## 后续候选（有余量才按完整验证单独立项）

球体碰撞、布片拓扑/弯曲、耦合端口是下一批。广义坐标机器人、FEM、本构学习、GPU 图捕获和机械臂抓布不属于本夜间任务保证完成的范围。需要时可推进一个已明确合同的小切片，必须先更新计划与测试，不临时堆砌完整子系统。
