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

`StepResult` 和 `Solver` protocol 已参数化状态类型；共用 `Stepper` 按显式 `state_kind` 选择粒子或刚体合同，入口检查 model/state/inputs 类型匹配。刚体候选额外检查单位四元数、角速度和固定姿态；快照/回退额外复制 orientation、angular_velocity、torque，避免只恢复平移。未知状态族不自动通过鸭子类型接入。当前刚体接触限于下述球/静态平面；关节、机器人、通用碰撞和 GPU 刚体内核需要后续独立验收。

布片起点为 `cloth_grid(rows, columns, ...) -> ClothPatch`：在 XZ 平面生成 +Y 朝向的三角形，顶点编号 `row*columns+column`。每个三角形的 `area*density` 质量各分给三个顶点；三角形划分导致边界/对角顶点质量不同，不按顶点数均分。固定点逆质量为 0，但只读 `vertex_mass` 保留真实面积质量。行/列邻接和每格两条对角线分别形成结构、剪切距离约束，柔度按两组赋值；初始静止长度由实际可表示的位置计算，量化后退化几何与不可表示物理参数显式拒绝。

这是拉伸/剪切网络，逐边柔度尚未标定为连续材料的 Young/Poisson 参数。它没有弯曲能、自碰撞、面积守恒或 FEM；可使用 NumPy XPBD 或下述 tyx 着色求解。GPU 上执行同一网络并不意味着完整布料本构或性能已达标。

### 7.2.1 tyx 着色约束路径

`TyxXPBDSolver` 采用稳定贪心边着色：每边恰好分入一色，同色边没有共享顶点；颜色按顺序执行，同色通过 `tyx.range` 批量投影。位置没有并发写争抢，不依赖原子加的执行顺序。重复边保留为独立约束并分开颜色；着色不是最少颜色优化。

当前状态和乘子均为 FP32；每步清零乘子。缩放柔度先在主机 FP64 计算，若转换到 FP32 下溢成零或非有限，则显式拒绝，避免将柔顺问题变成硬约束。距离范数通过缩放计算避免平方溢出；非有限中间量保留失败标记，不能被后续平面夹取掩盖。平面高度使用不低于真实高度的最近 FP32 值，主机用真实参数与 FP64 几何验收本构残差和穿透。位置修正后由真实 tyx 内核重建速度。

该 solver 与同配置 `XPBDSolver` 的物理签名一致，可以在粒子 Stepper 中做一次数学回退。非线性约束的着色顺序与 NumPy 串行输入顺序不同，加上 FP32 乘子舍入，不能要求多步轨迹逐位相同。小解析解、归一化本构残差、固定点和接触不变量为验收基础，另报告相对 FP64 参考的轨迹差。

一步内部设备缓冲保持存活，每迭代同步位置/乘子到主机验收；跨步重新分配/传输。可选 `cuda_graph=True` 只支持显式 CUDA 模式，用公开 `tyx.Graph(mode="cuda", optimize=False)` 记录整轮颜色/平面投影，每迭代重放一次，保持原顺序和验收频率。捕获只记录，空系统/自由步不创建空图；不自动降级为 launch list。

图、独立 stream 和数组均只活在本次 step；正常或异常退出先同步、销毁图引用与 stream，再释放设备数组。图开关不改变物理签名；默认关闭。没有跨步设备常驻、设备残差归约或 GPU 刚体。图/直接启动的公平比较与本机测量见进度记录，不能将 GPU 与不同精度/顺序的 NumPy 轨迹耗时直接相除。

### 7.2.2 球形刚体与静态平面

`SphereShapes(radii)` 拥有只读正半径数组，球心等于模型质心；形状本身不假设均匀质量分布，惯量仍由 RigidBodyModel 提供。`RigidSpherePlaneSolver` 先调用已验证的自由刚体积分，再对动态球做平面去穿透，接触时仅去除朝向平面的法向速度。恢复系数固定为 0，切向无摩擦；法向冲量经过质心，角速度与自由预测姿态保持不变。冲量与角冲量定义参考 [Baraff 非穿透/碰撞讲义](https://www.cs.cmu.edu/~baraff/sigcourse/notesd2.pdf) 的物理方程，独立实现此限定响应。

位置稳定化与速度冲量分开：不能将去穿透位移除以 dt 加入法向速度，避免产生人为弹开。已经向上分离的速度保留。中心最低高度按状态精度向上取整，并对 `center_y - radius - plane_height` 做补偿求和来复核间隙，避免小半径或小平面偏移在大尺度相加/相减时消失。固定物体不投影；穿透超出归一化阈值则返回未接受候选。数值设置统一为有限 Python/FP64 标量；接触与 XPBD 共用的归一化会把下溢为零的正商保守表示为最小正 FP64，防止零容差误接受。它保护已经算出的非零残差，不声称消除任意浮点几何误差。

`SpherePlaneSignature` 扩展通用 ProblemSignature，冻结全部半径、平面高度与接触定律；同形状不同容差可比较，无接触自由刚体或不同半径不能作为回退。碰撞按离散步末检测，没有求精确碰撞时刻，也没有球球/任意形状/宽相、摩擦、恢复反弹或 CCD。去穿透可改变势能，不宣称总能量严格守恒。

### 7.3 双向耦合

端口传递同一作用点与坐标系中的运动、冲量/力矩及有效质量。对象只有一个状态写入者；同一接触只有一个响应所有者。多子步必须按时间区间累计冲量，不能把瞬时力直接重复使用。先交替迭代+松弛，再按验证需求扩展强耦合算法。

当前落地的最小算子为 `ParticleRigidPorts(particle_indices, body_indices)`。两组等长只读 int64 索引允许重复；世界作用点取粒子当前位置。`apply_impulses(particle_model, particle_state, rigid_model, rigid_state, impulses)` 向粒子分配 J，向刚体分配 -J 与关于质心的 `r×(-J)`。用世界线冲量和体坐标主惯量计算速度／角速度响应，不积分位置和姿态。

全部累计与响应使用 FP64，重复端口用补偿求和保留抵消中的小项，最终回写共同的 FP32 或 FP64 状态精度。`ImpulseExchangeResult` 同时返回两组拥有独立存储的状态；任何响应或转换数值失败，两边都返回原始副本并给出原因，ledger 为空。成功的只读 `ImpulseLedger` 分别记录粒子线冲量、刚体线／角冲量；固定对象速度保持零，ledger 中的载荷由外部固定约束吸收，不能当作其实际动量变化。

`effective_inverse_mass(...)` 返回每端口的 `(w_particle+w_body)I - skew(r)I_world_inverse*skew(r)`，单位 kg⁻¹，描述本端口冲量造成的相对点速度变化。实现用等价 Gram 形式 `K=(wp+wb)I+A Aᵀ`、`A=skew(r)R/sqrt(I_body)`，避免零／小力臂下显式倒惯量的无谓溢出。共享对象的多个端口存在交叉响应，此数组不是全局块矩阵。双固定端口自响应为零；不可表示的数值通过 `SolverFailure` 拒绝。

该算子不决定冲量、不检测碰撞，也不执行位置投影或约束迭代；任意输入冲量可以做功，不能声称能量守恒。接口没有 dt，力应由上游按物理时间积分后送入；多速率区间累计、完整耦合 Stepper 和 GPU 耦合仍待实现。示例 `point_impulse.py` 独立检查偏离质心冲量的线／角动量及有效质量响应。

## 8. 学习模型与可微能力

可插拔角色依次为 warm start、材料响应、残差修正、局部代理和完整候选状态。面向真实任务的首个 AI 加速功能仍优先 warm start，保持数学终止容差不变，验证总耗时而非只看网络推理。当前小型自由粒子回归例子用于验证候选入口，不承担生产加速角色。

学习 provider 声明模型版本、训练域、单位、归一化、拓扑/步长要求、隐状态、随机种子和梯度能力。PyTorch 管训练与网络，tyx 管计算；共享数组必须约定 stream、ownership、mutation 与生命周期。

候选验收包含有限性、归一化约束残差、穿透、材料有效域；能量检查计入外部做功/耗散。拒绝后从同一原始快照重算，回滚范围包含物理历史、学习隐状态、随机状态及影响重算的求解缓存。数学回退失败后返回失败报告，不能无限重试。

首版 solver 必须提供可比较的 `problem_signature()`，描述其持有的约束索引/静止长度/柔度、接触面/接触定律和语义版本；model 与 inputs 由 Stepper 原样提供给两者。配置在构造时复制并冻结，签名不得依赖调用者之后可变的数组。backend、迭代次数与容差属于求解设置，不是物理问题身份。Stepper 在执行前拒绝不同签名，因此有距离/平面约束的 XPBD 不能回退到忽略约束的自由积分器。首版有效回退是相同约束配置的数学 solver（例如更高迭代预算）；未来学习 provider 从共享物理描述取得签名，并通过独立物理验收器验证候选，签名相同本身不是正确性证明。

梯度分为 unsupported、smooth、custom、fixed-active-set、smoothed-surrogate；训练者必须显式选择。接触切换处不承诺真实导数。自定义 VJP/隐式微分要有独立梯度测试；普通浮点容差通过不代表梯度正确。

### 当前实现的协议边界

`Solver` protocol 声明 `capabilities`、`problem_signature()` 和 `step(...)`。`SolverCapabilities` 包含 state_kind、features、dtypes、backend、gradient、history 和 provider_kind。provider_kind 只允许 mathematical（默认）或 learned，不从 backend 名称推断。`Stepper(primary, fallback=None, required_features=..., gradient="unsupported", max_residual=1e-5, verifier=None)` 在调用前检查同一物理签名与精度/能力；signature 后续改变也会被拒绝。

每次尝试独立复制对应状态族的全部可变字段和输入；不可变 model 共用。粒子复制位置/速度/外力，刚体另含姿态/角速度/力矩。验收独立检查 shape/dtype/有限性/固定配置与零固定速度，刚体还检查单位四元数；再检查 solver 自报 accepted 和归一化残差阈值。柔顺几何伸长不与零比较。成功候选再次复制，避免返回 solver 内部存储的别名；全部失败则返回原快照。`report.fallback_used` 与 `report.primary_failure` 记录是否重算及原失败原因。预期域外拒绝可抛 `SolverFailure`；其他程序/配置异常原样传播，便于诊断。

`runtime/state_contract.py` 持有当前两种状态族的类型/复制/固定字段规则；所有尝试共享 `Stepper` 的报告检查、一次回退预算与物理签名门禁。两 solver 的 state_kind 必须一致且在构造后保持不变。新增状态族应先提供完整合同和回滚测试，再接入；该模块只依赖 core，不依赖具体数学或学习 solver。

入口和候选都按合同字段用内置数据构造器规范化，不信任扩展子类覆盖的 `copy()`；这样普通缓存/浅拷贝实现不会绕过有限性、四元数或所有权验证。子类可作为已有字段的载体，但新增物理字段/历史必须明确扩展状态合同，不能假定会随 Python 子类自动保存。

直接调用数学 solver 时也通过 core 验证器检查明确的数据类型并构造自有基类快照；数值失败回传同样使用内置复制实现。这个要求覆盖绕过 Stepper 的分析/测试入口，防止普通子类的缓存 copy 修改调用者状态，或把另一状态族的角字段/力矩悄悄丢掉。

| 当前 solver | features | 精度 | 梯度 / 物理历史 |
| --- | --- | --- | --- |
| SemiImplicitSolver | particles, free_dynamics | FP32/FP64 | unsupported / none |
| XPBDSolver | 基础 + 按配置 distance、plane_contact | FP32/FP64（乘子 FP64） | unsupported / none |
| TyxSemiImplicitSolver | particles, free_dynamics | FP32 | unsupported / none |
| TyxXPBDSolver | 基础 + 按配置 distance、plane_contact | FP32（乘子 FP32，验收 FP64） | unsupported / none |
| RigidSemiImplicitSolver | rigid_bodies, free_dynamics, rotation | FP32/FP64（旋转算术 FP64） | unsupported / none |
| RigidSpherePlaneSolver | 刚体基础 + sphere_plane_contact | FP32/FP64（间隙验收 FP64） | unsupported / none |

`CandidateVerifier` 声明 state_kind、problem_signature() 和 `verify(model, state_in, inputs, dt, candidate) -> VerificationReport`。任一 learned provider（含 fallback）必须配置同状态族、同物理签名的 verifier，且初始化与每步都检查匹配。原始状态、输入和已规范化候选的副本交给 verifier；验证器修改这些副本不能改变待提交状态或下一次尝试。所有尝试，包括数学 fallback，都要通过相同独立验收。成功报告取自报/独立残差和米制误差的较大值；拒绝或非法验证报告触发已有的一次回退规则。

已实现 `FreeParticleVerifier(tolerance=1e-5, reference_length=1.0, reference_speed=1.0)`，仅支持无约束粒子 force/gravity 签名。用独立 FP64 算术展开速度优先离散方程，并以补偿求和直接计算位置/速度方程残差，避免大基值吞掉小运动贡献；不调用数学 solver。位置/速度欧氏误差分别以米和米/秒尺度归一化，取最大值。无法安全表示的参考乘积/和、乘法触发的不精确下溢（包括非零次正规结果）会保守拒绝；精确次正规结果保留。浮点算术仍有舍入误差，容差须按实际物理尺度设定。这是正确性参考，不是免费或通用物理验收器。

`examples/learned_free_particles.py` 用 NumPy 最小二乘从数学标签拟合重力/外力两个响应系数，报告版本、种子、SI 单位及 dt 适用区间；64 训练场景与 64 留出场景分开。正常候选独立验收，损坏系数和域外 dt 通过同签名数学回退处理。它是无隐藏状态的低维回归示例，不是神经网络、通用 AI 求解器或完整训练域检测系统。

数学 provider 未配置 verifier 时仍沿用自报物理残差；对 compliant XPBD、接触与刚体尚无独立动力学 verifier。verifier 本身是维护者信任的、物理上无历史依赖的检查代码，这个合同不隔离任意恶意 Python。`history != "none"` 当前拒绝；模型随机状态、材料历史和学习缓存的完整事务协议尚未实现。

## 9. 首版验收场景和后续边界

已实现的首批：空场景、自由落体、固定点、受力粒子、两点柔顺约束、悬挂链/布片、粒子平面碰撞、旋转刚体、球形刚体对静态平面；后续：球球/通用接触、夹爪压软块、机械臂抓布。

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
