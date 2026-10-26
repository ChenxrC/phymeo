# phymeo 架构设计 v0

日期：2026-10-27（Asia/Shanghai）。状态：已获授权的实施方向；具体能力按进度记录验收。

## 1. 目标、范围与证据

支持机器人/刚体与布料/柔性体，在数学仿真基础上提供学习材料、求解加速和代理求解接口。第一版本优先完成可运行的基础架构与部分数学求解，不承诺完整机械臂抓布、通用碰撞、所有 GPU 后端或端到端接触梯度。

本轮调研读取 tyx 提交 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee` 的 README、公共导出、PyTorch 互操作、LinearOperator 与 Krylov 源码。Newton 参考为 2026-10-26/27 查阅的 main/latest 文档，后续差分实验须另记录实际固定 SHA。未运行的上游测试不能计入本项目验证证据。

tyx 部分旧支持矩阵与较新的源码不一致，例如多设备接口；以固定源码、安装包与目标设备测试为准。旧研究中的 phemto 是另一个名称，本项目不自动继承其实现或仓库政策。

## 2. Newton 的可借鉴边界

| 边界 | Newton 的做法 | phymeo 的采用方式 |
| --- | --- | --- |
| 建模/运行 | ModelBuilder 构建，Model 保存模型 | 保留构建与运行分离，自定 SceneSpec/CompiledScene |
| 数据/算法 | State、Control、Contacts 与 Solver 分开 | 显式模型、状态、输入、接触、工作区 |
| 碰撞/响应 | CollisionPipeline 产生接触，由 solver 消费 | 碰撞几何与材料响应分开，禁止重复施力 |
| 多算法 | 多种 SolverBase 后端，能力不等价 | 物理 Capability 与 tyx compute capability 分层 |
| 耦合 | 已有实验性 proxy/ADMM、对象所有权 | 自定交互端口与质量验收，不宣称多 solver 是原创 |
| 计算 | Warp 的类型、kernel、几何和微分深入各层 | tyx 公共计算接口；物理几何由 phymeo 持有 |

不保留 MuJoCo Warp 作为核心运行时依赖。Newton/MuJoCo 可以作为隔离的比较工具；比较结果不等于真值。

## 3. 方案取舍

1. 模仿 Newton 的结构并换后端：对应容易，但会继承数据与算法边界，AI 接入被动。
2. 一次构建通用多物理方程编译器：表达能力强，首期成本与验证范围过大。
3. **选用：精简物理描述 + 可组合算子 + 专用求解器。** 先验证刚体/柔体共用状态与交互契约，再扩展学习模型。

用户已授权根据分析与大纲开始基础实现。以下细节是实施者的默认决策，变更需在进度记录中注明原因与影响。

## 4. 项目约束

- Python >=3.11；CPU 参考路径支持 Windows/Linux；CUDA 支持必须由实际硬件测试证明。
- phymeo -> tyx 单向依赖；不导入 tyx 私有模块；tyx 不依赖 phymeo。
- NumPy 是参考数据与 CPU 基础；PyTorch、tyx 和 GPU 工具链按可选能力分离。
- 首版仅承诺 FP32 tyx 计算；NumPy FP64 可作独立参考，不等于 tyx FP64 支持。
- 不复制 Newton 源码、注释或测试实现；按数学推导与独立验收构建。
- CI 不用跳过结果冒充 GPU 通过；未收敛、缓冲区溢出、能力不支持必须显式报告。
- 不发布软件包、不合并主分支；验证后可提交与推送专用 codex 分支。

## 5. 模块和依赖

目标目录如下；不存在的目录属于计划。

```text
src/phymeo/
  builders/      # 场景构建；布片拓扑、面积质量与初始条件
  core/          # 模型、状态、输入、能力、报告；不含具体算法
  backends/      # NumPy 参考与 tyx 公共 API 适配
  geometry/      # 接触数据、基本形状、后续宽相/窄相
  dynamics/      # 积分、空间代数、后续机器人动力学
  constraints/   # 距离、接触、连接；物理参数与乘子语义
  solvers/       # 数学 solver；不管理训练数据
  runtime/       # 步进、快照、验收、回退、后续耦合
  learning/      # 可选学习 provider、适用域、模型元数据
examples/        # 可重复的小场景
tests/           # 合同、数学、集成、设备、打包与回归
```

导入方向：`runtime -> solver protocols/core`；具体 solver 使用 `dynamics/constraints/geometry/backends`；各实现依赖 `core`。core 不导入具体 solver、tyx、torch。learning 不绕过 runtime 直接提交状态。训练控制策略通过输入接口进入，不冒充物理求解器。

`builders -> core/constraints`；solver 不依赖 builder。构建返回明确的静态模型与初始状态，后续修改动态状态不改写静态质量、连接或材料参数。

## 6. 数据与接口契约

| 名称 | 语义 | 首版范围 |
| --- | --- | --- |
| SceneSpec / CompiledScene | 可编辑描述 / 验证后拓扑、参数、资源 | 先用简单不可变模型，后续独立编译步骤 |
| SimulationState | 随时间变化的物理量与内部变量 | 粒子位置/速度；独立刚体位置/姿态/线速度/角速度 |
| StepInputs | 外力、控制、边界驱动 | 有量纲的外力，不与冲量混用 |
| PhysicsOperator | 质量、力、能量、约束及导数能力 | 质量逆、距离约束、平面接触 |
| SolverCapabilities | 对象、后端、精度、梯度和历史要求 | 显式拒绝不支持组合 |
| StepReport | 接受状态、残差、迭代数、失败原因 | 有限性、约束残差、时间步合法性 |

稳定的拟议 solver 边界：

```text
step(model, state_in, inputs, dt) -> StepResult(state, report)
```

输入状态不被修改；结果提交由 runtime 决定。首版用独立数组实现候选状态，后续允许预分配工作区，但必须保持相同所有权与别名语义。

单位：SI；右手坐标系；世界重力默认 `(0, -9.81, 0)`；长度 m、质量 kg、时间 s。位置/速度不同维度不混用。旋转阶段采用 `(x,y,z,w)` 单位四元数和显式局部/世界角速度约定，不能对四元数直接作欧氏位置更新。

粒子/刚体用非负逆质量表示，0 为固定对象。NaN/Inf、负质量/柔度、非正时间步、非法索引与不匹配形状应在入口报错。空集合合法；不能把 tyx 不接受零长度设备数组的限制泄露成物理层不支持空场景。

## 7. 数学求解路线

### 7.1 首版基础

先实现半隐式 Euler：`v_next = v + dt*(gravity + inverse_mass*force)`，`x_next = x + dt*v_next`。固定粒子位置保持不变，速度语义明确为 0；输入状态不可变。解析自由落体的离散解用于验证，不用相同实现重算 expected。

XPBD 距离约束：`C = ||x_i-x_j|| - rest_length`，`alpha_tilde = compliance/dt^2`，每子步将乘子初始化为 0，再在子步内累计。使用论文形式的 `delta_lambda`；零长度边需要确定性退化处理，不产生 NaN。柔度单位与约束形式绑定，不能跨不同约束直接复用数值。投影完成必须用 `v_next = (x_corrected - x_start)/dt` 重建动态粒子速度，固定点速度仍为 0；这是首版算法组成部分。

收敛判据使用归一化的本构残差 `C + alpha_tilde*lambda`，并分别报告几何伸长 `C`。受力的柔顺约束允许非零伸长，不因该伸长触发失败。接触间隙残差另行度量；不同量纲先用场景特征尺度归一化。

首版具体合同：`DistanceConstraints` 的索引为 `(m,2)` 整数，静止长度和柔度为各 `(m,)` 浮点数组，与状态精度一致；不隐式广播参数。自连接边拒绝，重合的不同端点取世界 x 轴作确定性退化方向。`XPBDSolver` 的 `reference_length` 默认 1 m，距离残差除以 `max(rest_length, reference_length)`，平面穿透除以 `reference_length`；场景尺度显著不同时应显式设定。内部乘子和柔度缩放以 FP64 累计，位置/速度保持请求的 FP32/FP64。有限但未收敛的候选保留用于诊断，调用者必须检查 `report.accepted` 后才提交；非有限结果则返回原状态副本。

接触首版仅静态平面：明确为简化非穿透、无摩擦、恢复系数 0 的基线。位置投影与上述速度重建后，活动接触的负法向速度截为 0，保留切向速度。非零恢复系数和摩擦留待后续单独验证。该路径不是通用 CCD，也不保证高速/复杂几何无穿透。

### 7.2 刚体与柔性体

独立刚体采用最大坐标；机器人后续采用广义坐标动力学。布料先用有明确局限的距离/弯曲模型，再升级应变能；软体采用材料模型与隐式积分。CG/PCG 仅用于满足正定等条件的系统，不能代替一般接触互补/KKT 求解。

已实现的独立刚体基线为 `RigidBodyModel / RigidBodyState / RigidBodyInputs / RigidSemiImplicitSolver`。惯量是质心处体坐标主轴对角量；body-to-world 姿态为 xyzw 四元数，所有速度与外力/力矩在世界坐标。动态主惯量必须为正且满足三角不等式；0 逆质量同时固定平移和旋转。四元数输入允许 1e-6 的单位长度舍入误差，坐标变换时先在 FP64 归一化；动态姿态更新也归一化，固定姿态保留原值。

先把世界角速度和力矩转入体坐标，用 `omega_dot_body = (tau_body - omega_body × (I_body * omega_body)) / I_body` 得到角加速度，再转回世界坐标更新角速度。新世界角速度构成旋转向量，指数映射增量从左侧复合旧四元数。该实现从刚体方程独立编写，参考 [Baraff 刚体动力学讲义](https://www.cs.cmu.edu/~baraff/sigcourse/notesd1.pdf) 的物理定义，以及 [SciPy 旋转向量定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html)；运行时不依赖 SciPy。它是一阶自由运动基线，非球形惯量不精确保持能量/角动量，须用步长收敛检查控制误差。

`StepResult` 已参数化状态类型；当前 `Stepper` 仍只处理粒子，构造时明确拒绝 `state_kind="rigid_bodies"`，防止遗漏姿态/角速度的验收与回滚。刚体接触、关节、机器人和 GPU 刚体内核需要后续独立验收。

布片起点为 `cloth_grid(rows, columns, ...) -> ClothPatch`：在 XZ 平面生成 +Y 朝向的三角形，顶点编号 `row*columns+column`。每个三角形的 `area*density` 质量各分给三个顶点；三角形划分导致边界/对角顶点质量不同，不按顶点数均分。固定点逆质量为 0，但只读 `vertex_mass` 保留真实面积质量。行/列邻接和每格两条对角线分别形成结构、剪切距离约束，柔度按两组赋值；初始静止长度由实际可表示的位置计算，量化后退化几何与不可表示物理参数显式拒绝。

这是拉伸/剪切网络，逐边柔度尚未标定为连续材料的 Young/Poisson 参数。它没有弯曲能、自碰撞、面积守恒或 FEM，当前使用 NumPy XPBD；不能把悬挂示例解释为完整布料材料或 GPU 布料。

### 7.3 双向耦合

端口传递同一作用点与坐标系中的运动、冲量/力矩及有效质量。对象只有一个状态写入者；同一接触只有一个响应所有者。多子步必须按时间区间累计冲量，不能把瞬时力直接重复使用。先交替迭代+松弛，再按验证需求扩展强耦合算法。

## 8. 学习模型与可微能力

可插拔角色依次为 warm start、材料响应、残差修正、局部代理和完整候选状态。首个 AI 功能优先 warm start，保持数学终止容差不变，验证总耗时而非只看网络推理。

学习 provider 声明模型版本、训练域、单位、归一化、拓扑/步长要求、隐状态、随机种子和梯度能力。PyTorch 管训练与网络，tyx 管计算；共享数组必须约定 stream、ownership、mutation 与生命周期。

候选验收包含有限性、归一化约束残差、穿透、材料有效域；能量检查计入外部做功/耗散。拒绝后从同一原始快照重算，回滚范围包含物理历史、学习隐状态、随机状态及影响重算的求解缓存。数学回退失败后返回失败报告，不能无限重试。

首版 solver 必须提供可比较的 `problem_signature()`，描述其持有的约束索引/静止长度/柔度、接触面/接触定律和语义版本；model 与 inputs 由 Stepper 原样提供给两者。配置在构造时复制并冻结，签名不得依赖调用者之后可变的数组。backend、迭代次数与容差属于求解设置，不是物理问题身份。Stepper 在执行前拒绝不同签名，因此有距离/平面约束的 XPBD 不能回退到忽略约束的自由积分器。首版有效回退是相同约束配置的数学 solver（例如更高迭代预算）；未来学习 provider 从共享物理描述取得签名，并通过独立物理验收器验证候选，签名相同本身不是正确性证明。

梯度分为 unsupported、smooth、custom、fixed-active-set、smoothed-surrogate；训练者必须显式选择。接触切换处不承诺真实导数。自定义 VJP/隐式微分要有独立梯度测试；普通浮点容差通过不代表梯度正确。

### 当前实现的协议边界

`Solver` protocol 声明 `capabilities`、`problem_signature()` 和 `step(...)`。`SolverCapabilities` 包含 state_kind、features、dtypes、backend、gradient 和 history。`Stepper(primary, fallback=None, required_features=..., gradient="unsupported", max_residual=1e-5)` 在调用前检查同一物理签名与精度/能力；signature 后续改变也会被拒绝。

每次尝试独立复制位置/速度和外力；不可变 model 共用。验收独立检查 shape/dtype/有限性/固定点，再检查 solver 自报 accepted 和归一化残差阈值。柔顺几何伸长不与零比较。成功候选再次复制，避免返回 solver 内部存储的别名；全部失败则返回原快照。`report.fallback_used` 与 `report.primary_failure` 记录是否重算及原失败原因。预期域外拒绝可抛 `SolverFailure`；其他程序/配置异常原样传播，便于诊断。

| 当前 solver | features | 精度 | 梯度 / 物理历史 |
| --- | --- | --- | --- |
| SemiImplicitSolver | particles, free_dynamics | FP32/FP64 | unsupported / none |
| XPBDSolver | 基础 + 按配置 distance、plane_contact | FP32/FP64（乘子 FP64） | unsupported / none |
| TyxSemiImplicitSolver | particles, free_dynamics | FP32 | unsupported / none |
| RigidSemiImplicitSolver | rigid_bodies, free_dynamics, rotation | FP32/FP64（旋转算术 FP64） | unsupported / none |

这是学习候选未来接入的协议基础，不是已经完成 AI 物理验收。当前残差仍由 solver 计算，不能据此信任任意神经网络的自报数值；后续学习 provider 必须增加独立物理残差/适用域检验。`history != "none"` 当前拒绝，防止隐藏状态未回滚；模型随机状态、材料历史和学习缓存的完整事务协议尚未实现。

## 9. 首版验收场景和后续边界

首批：空场景、自由落体、固定点、受力粒子、两点柔顺约束、悬挂链/布片、平面碰撞；后续：旋转刚体、球体接触、夹爪压软块、机械臂抓布。

每条能力记录模型假设、后端、dtype、时间步、容差和已执行测试。GPU、完整机器人、自碰撞、四面体 FEM、AI 训练与长时可微 rollout 都必须另过验收，不能从架构图推断实现。

## 10. 一手参考

- [Newton 核心 API](https://newton-physics.github.io/newton/latest/api/newton.html)
- [Newton 求解器能力](https://newton-physics.github.io/newton/latest/solvers/index.html)
- [Newton 碰撞](https://newton-physics.github.io/newton/latest/concepts/collisions.html)
- [Newton 已有耦合机制](https://newton-physics.github.io/newton/latest/concepts/coupling.html)
- [XPBD 原论文](https://mmacklin.com/xpbd.pdf)
- [VBD 作者项目](https://ankachan.github.io/Projects/VertexBlockDescent/index.html)
- [tyx 固定快照](https://github.com/ChenxrC/tyx/tree/99f6df7c0fd8fdeef77f97df83914da27b64b7ee)
- [tyx PyTorch 互操作](https://github.com/ChenxrC/tyx/blob/99f6df7c0fd8fdeef77f97df83914da27b64b7ee/docs/pytorch-interoperability.md)
- [图网络仿真 GNS](https://arxiv.org/abs/2002.09405)

配套：[架构图](phymeo.html) · [两人开发与测试](../development/team-and-tests.md) · [环境与 CI](../development/environment-and-ci.md)
