# 完整固定根 R/P 树驻留与当前姿态宽相：实施建议

2026-11-07，状态 **RESEARCH_PROPOSAL / 等待父裁定 / 无测试或产品授权**。用户取消原 08:00 截止；本文件不以时限缩减必要物理与事务验收。作者在 clean `8439964` 上只新增本文件和[案例册](../../development/p65-resident-tree-casebook.md)，未合并父、未修改 C3、未执行任何项目命令。父共享基线为 `bbfe7f2`；后续父 F2 工作不改变本方案依据的 TYX 接缝。

## 1. 目标与选择

交付目标是一个有明确容量的普通固定根树设备 world：真实接受 q/v 在同一设备 runtime 内形成完整 FK/J/M/bias、求驱动与当前外载响应、一次推进并共同发布历史/FK，然后把多个活动 link 的有限形状 bounds 直接交原 BVH。静态模型/材料可以准备时 HOST 编译，暖态不能靠 HOST FK、M 求解或每轴独立 rotor world 代替。

推荐 **N=1..16 个真实活动 R/P 行、每 world 各自 n/parents**；分支、非主轴关节、任意受支持安装、偏心 COM 和已折叠固定工具为首域。16 覆盖普通 9 行 7臂+2指结构，8 不足；N=9/16 是独立容量验收，不能用“Panda 有 7 个臂关节”略掉两个真实指关节。首片不含 mimic、浮根、闭环、joint stops 或全机器人接触响应。未来 mimic 必须区分物理行数与约束后速度列数，不继承 N=velocity_count 的假设。

不用另建 runtime 或复制 TyxWorldBatch。推荐有类型的树 device-storage 扩展：保留原 scalar 12 列布局及旧纯 batch 的宽度，树缓冲单独登记在原 owner/锁/resources 下，使用同一个 active/ticks/epoch 和发布点。实际 n 循环与静态三维小循环相结合；不按资产名字或某张拓扑编译特例。全树存储与计算成本、N8/9/16 对比见案例册，**没有性能收益结论**。

## 2. 已核源码结论

- `backends/_tyx_robot_kernels.py:9–269` 已实现真实 FK/J/M/解析 bias；`tyx_robot_math.py:47–187` 可以复用其算法。`tyx_math.py:76–130` 的逐次分配/上传/读回和 `tyx_participants.py:124–150` 的 HOST 候选不能原样作为驻留实现。
- `control/bounded_pd.py:61–183` 是全耦合 box-QP 投影坐标迭代，含 d=0 固定行、全 clipping 残差和原质量方程核。应保真实方程；无需改叫 active-set 或每轴裁剪。`driven_robot.py:175–218` 已明确外载即时、三控制列延迟和一次物理推进。
- `builders/_urdf_fixed.py:54–129`、`_urdf_frames.py:20–58` 已有固定工具质量聚合和原材料/sites。复用主轴转换与真实物理组，不另造 DH/资产专用几何。
- `backends/tyx_world_batch.py:240–279,310–379,456–534` 是接受态/借用/restore/close 的唯一共享所有者。`_world_step_layout.py:128–218` 与 `_world_kinematics.py:9–49` 仍是单轴/球类型；新树必须显式扩展或拒绝，不截断。
- `_resident_collision_plan.py:213–347` 当前只允许每 world 一个动态 body；新树需要完整 `(world, tree_row, group, body, shape, source)` 注册。`445–560` 已有 device bounds→BVH 及事务，复用它，不重做空间查询。
- 本机 Newton 1.7 的 `State`/`eval_fk`/Featherstone 源码已经确认设备 q/v、COM/world body twist/wrench、原生全树推进与限幅控制差异；精确路径/行号及独立参考语义在案例册第1、7节。无需新安装或联网 CI。

## 3. 拟新增与父共享边界（待逐阶段批准）

本轮实际允许的文件只有本计划与案例册。以下是未来 allowlist 建议，不是本轮写入授权。

| 未来 NEW 文件 | 单一职责 |
| --- | --- |
| `src/phymeo/backends/_resident_tree_world.py` | `FixedRobotTree(program,state,control_max_iterations=512)`；owned model/topology/完整 command 编译；TreeWorldCheckpoint/TreeKinematics/数值域 |
| `src/phymeo/backends/_resident_tree_storage.py` | 在原 batch 中准备并拥有树 buffers/scratch，stage step/restore/readback、借用 FK view；无独立 runtime、无独立 swap |
| `src/phymeo/backends/_tyx_resident_tree_kernels.py` | 批量 FK/J/M/bias、Cholesky/全耦合 PD、候选 q/v/历史、末态 FK/能量/状态；双 active 方向共享数学路径 |
| `src/phymeo/backends/_resident_tree_binding.py` | `ResidentTreeBinding.from_asset`，原材料到真实 group、source/IDs/revision/body registry 与静态几何编译 |
| `src/phymeo/backends/_tyx_resident_tree_bounds.py` | 当前 accepted FK→sphere/capsule/hull 完整有限 bounds、工程外扩、同 tick 写入原 BVH buffer |
| `tests/scale/test_resident_tree_step.py` | T01–T06 的物理、延迟、异构/接受事务和容量 |
| `tests/scale/test_resident_tree_collision.py` | T03/T07/T08 的原材料、多 body、设备边界、restore/failure/资源 |
| `tests/reference_newton/robot/resident_tree_reference.py` | pinned Newton 1.7 CPU/CUDA 独立树推进和完整原始读数，不 import Phymeo 生成 oracle |
| `docs/benchmarks/p65-resident-tree-collision.md` | 原始失败、逐阶段范围、真实设备/计数/窗口与限制；初始应 UNEXECUTED |

父拥有的旧文件：`src/phymeo/backends/_world_step_layout.py`、`_tyx_world_kernels.py`、`tyx_world_batch.py`、`_world_kinematics.py`、`_world_readback_plan.py`、`_tyx_world_readback_kernels.py`、`_resident_collision_plan.py` 及 `src/phymeo/__init__.py` 的必要公共 exports。原 `_tyx_robot_kernels.py`/`tyx_math.py` 如需提取复用原语也由父改，作者不得顺手改老物理。`tyx_collision.py`/`_tyx_collision_bvh.py` 优先零行为改动，仅消费原有 `_bounds`，若确需更改另报精确接缝。

既有 `tyx_world_contact.py` 是另一受限 cloth/slider 工作流（其文档和 `WorldMaterial` 明确结构），不是完整树的原生接触 solver。首片不自动接管它、不更改 RobotScene/MultiRobotScene 或其材料律。

## 4. 最小内部合同建议

- `compile_tree_world(spec)` 返回 owned 的 TreeWorldSpec：domain、n、parents/kinds、静态 joint frames/axes、mass/COM/principal inertia/gravity、kp/kd/cap、delay、初始 q/v/history/held packet、topology token。HOST 只编译静态与初值；没有当前状态回算入口。
- `tree_packet(inputs, domain, n)` 返回 `[n,10]` 真实完整命令。树 command 存储 `[T,Nmax,10]`，不使旧 sphere/slider 的所有行膨胀到 160 列；`set_controls` 在任何上传前先校验全部选中 world，再统一提交 held 输入。
- `TreeDeviceStorage(batch, specs_by_world)` 只通过 batch `_buffer/_launch/_device` 等受控设施分配/调用，向原资源栈登记；`stage_advance(h, src, dst, accepted, ticks)` 不改 active/epoch，所有中间 kernel 仍在同一 stream/锁内。
- `stage_restore(checkpoints, src, dst, ticks)` 初始 HOST 校验所有输入，设备重建对应 FK；未选树复制原态。只有父统一验证/同步后发布，不能先换 q 再补 FK。
- `borrow_geometry(active)` 给只在父同锁借用期有效的 TreeGeometryView：owner/topology/epoch/active、tree/world slot/counts、静态 anchor、accepted relative-FK/ticks、source registry。碰撞 plan 不关闭借用 runtime，不缓存跨 epoch 候选 pose。
- `compile_tree_binding(shape,binding,owned_topology)` 证明 whole-model/source identity 和实际 group，返回 group-local 原材料点、slot/group 索引、半径/工程预算；原 vertex IDs 不变。直接构造只接受明确 group-local 安装声明，from_asset 才携 source 证明。
- 新树 readback 独立有类型，预留完整 checkpoint/kinematics 的准备式索引读回；旧 6/12 列 plan 对树必须显式路由或拒绝，不能产生“正常但少关节”的输出。公共碰撞结果继续原 immutable schema。

具体 kernel positional tuple 仍需父在测试授权时冻结，不现在猜 30 个数组参数。静态参数/命令/双状态/FK/scratch 的维度与时间来源已由案例册确定；不能为了方便改成 HOST 计算 J/M 或第二套设备 runtime。

### 4.1 选定的 mixed 布局与原子性

本建议选择 **原 batch 内独立 tree banks**，不改所有 state 为大 Nmax row。共享 header 仍 `[W,12]`；tree kind 的 header 是未承载物理量的零占位，旧 pass 只能复制/跳过，不能读 column0/3 当根坐标。旧 command `[W,C_old]` 的 C_old 只按旧 kinds 取4或10，旧 properties `[W,P_old]` 只取9或16；纯 tree batch 的 header 也不借用这些列存 M 或 q。Tree static properties 在独立有类型表里。

`tree_state[active][tree_row,joint,field]` 为 `[T,Nmax,7]`，明确 field0=q、1=v、2=last_drive、3=last_generalized_applied、4/5/6=delay triplet。`tree_held[T,Nmax,10]` 为完整待执行命令，padding 清零；delay0 时历史列为0。`tree_fk[active][T,Nmax,18]` 为 anchor-relative origin3/R9/axis3/COM3。`tree_slots[T]` 与 `world_to_tree[W]` 是双向映射；counts/parents/所有几何 owner 均查实际 tree_row，不按 world key 大小或 body编号索引。后续若新增接受输入 receipt，应保存与 tick 对齐的副本，不能拿可更新的 held 代替。

- **controls**：先校验全部请求，分别形成旧 held 与 tree held 的完整候选。持同锁一次提交两个 HOST 输入快照；下一 step 在发任何 advance 前上传所有 dirty banks。上传部分失败不清 dirty、不 swap；重试重传，不能让一部分新命令配另一部分旧命令进入物理步。
- **step**：所有 passes 共享 src/dst、同一个 W 长度 ticks/accepted。旧 pass 只 advance 原 kinds；tree pass 只写其 world槽，完成全树候选 q/v/历史及 FK 后才设置该槽 accepted/tick。数值拒绝复制整棵树旧值；统一 sync/status 后父只 swap 一次 active、增一次 epoch。不能在任一个 storage helper 内提前发布。
- **snapshot(keys)**：一次锁内按 world kind 分路。旧 key 保旧 WorldCheckpoint；tree key 从对应 active bank 取全部 n×7、当前 held n×10 与同一个 world tick，生成 TreeWorldCheckpoint。读回映射保持 requested key 顺序、数组 owned；任何路径不能调用旧 `_row[12]`/`binding.checkpoint` 去截树。prepared tree reader 的索引/staging 预分配，旧读回 API 对未支持的 tree kind 明确拒绝。
- **restore/reset**：先一次校验全部混合 checkpoints 的 kind/owner/topology/n/队列/控制/域；分别 stage 旧 header 和 tree 数据，未选世界保旧。所有 restore kernels 指向共同 dst；tree 对候选 q 重建 FK。任一失败不 swap、held 和 epoch 不发布；全部成功才一次换 active、增 epoch，并更新对应 held。same-tick restore 同样使旧 bounds result stale。不得先调用旧 restore 发布，再补 tree restore。
- **borrow/query**：旧 AcceptedCollisionState 继续提供 legacy accepted handle，另附同一 active 的 TreeGeometryView。tree FK 必须是初始化或上次 step/restore 已同步的 accepted 缓冲。query 不触候选、不 snapshot；只使用原 runtime 与已准备的 tree-shape rows 写原 device bounds。close 和构造失败同时清理两类资源，每个 callback 都尝试，借用方不 close batch runtime。

旧计数意义保留；新增 tree topology/control/restore 上传、state-readback、tree dynamics/FK/bounds launches 及实际 bytes/allocations 单列，总量显式相加，不隐藏诊断 readback。这样的扩展保原 slots/runtime/时钟，增加的是一种真实 world-kind 存储接口，非另造第二个 batch。

## 5. 物理与事务不可省略的门

**完整 mass 与控制。** 在 q− 上求真实 J、M、解析 b、外载投影，随后求限幅隐式 PD。d=0 固定行对其它行的 cross mobility 必须保留。所有数值验收在设备完成，不能只由 HOST 读回判通过再把 HOST 解写回。逐坐标重算 `expected=clip(kp*(target-q-h*u)+kd*(rate-u)+ff,±cap)`，原始 clipping 残差门为 `5e-5*max(1,abs(tau_i),abs(expected_i))` N/N·m，另保质量方程 `2e-5`；任一中间非有限即拒绝。宽 cap 或大项相消不得放宽实际小驱动，T05 增加原 cap 不变的两个独立留出。迭代上限/非 SPD/非有限/全残差不合格须有可读原因；保原 accepted bool，再增加诊断状态码，如 `M_NOT_SPD`、`M_RESIDUAL`、`PD_NOT_CONVERGED`、`STATE_DOMAIN`、`FK_NOT_FINITE`。没有收敛就整 world 拒绝，不裁 M/改质量/忽略某轴。

**一次推进及正确采样。** held drive 和 current external 只施加一次，candidate v/q 与队列共同计算。从 q+ 重算 candidate FK；入步 M/J/b 是带入步时刻的 scratch，不是接受末态缓存。末态能量由当前 q+/v+ 的逐 body 动能求和，不能用旧 scalar mass 或 q− M 冒当前能量。

**统一发布。** 父旧 pass 对 tree kind 只做明确复制/跳过，不提前 accept；tree pass 成功后的所有 q/v/history/FK/tick 才由父统一 swap。单 world 数值拒绝不污染邻 world；全设备异常不发布。same-tick restore 仍增 epoch、旧 result 失效，局部 reset 保旁观树；所有 query/close/callback 失败沿原事务和 ExitStack 清理。

**宽相语义。** 首片是当前接受姿态的活动 link–显式有限静态体候选。多个物理 link/source/mapping 真校验；同组多 shape 同 tick。自碰撞/活动–活动 pair 首片要求明确排除或拒绝，不悄悄漏检。CCD、TOI、接触反力和完整机器人场景都未包含。工程外扩采用 anchor-relative FK、路径尺度/三角函数预算与外向 FP32，真实 near-touch/no-leak 留出见案例册；不回到无限精度证明路线。

## 6. 建议里程碑、先红后绿与依赖

| 节点 | 交付与必要案例 | 放行后才可声称 |
| --- | --- | --- |
| T0 本次研究 | 两 docs；本机源码/容量/所有权/手推多例；父确认首域和 API | 研究提案完成，不是实现 |
| T1 tests-first | 冻结 T01 的正常公共入口 RED 与完整 T01–T08 表；旧 batch 真实缺能力作为有效 RED；先保原 log/source | 已证明缺口，不是物理失败归因 |
| T2 真全树无反馈推进 | NEW 编译/storage/FK/J/M/bias/外载与半隐式推进，T01/T02/T03/T04/N9/N16；interpret→CUDA；N17/非法/数值拒步 | 通用有界固定根树的真实接受设备 q/v；尚未全控制/宽相 |
| T3 耦合 drive 与控制事务 | T05，delay0/1、当前 COM 外载、未收敛原态、两缓冲/局部 restore/observations；原 one-R/slider 少量受影响兼容 | 有限驱动、历史和设备状态共同接受；不含 stops/mimic |
| T4 全 link 当前宽相 | T07/T08 的有限 shape/source/group、W1/4/32、完整 trace 和严格 warm 无 snapshot/envelope/newalloc；owner/epoch/capacity/close | 受限多 link 接受态直接产生 device bounds 并进入原 BVH |
| T5 独立参照与打包 | Newton 同物理输入的原生 CPU/CUDA树窗口；公开正常入口；源码外 wheel 真实推进+query；相同完整输出成本统计 | 当前有界能力可发布，仍按边界记限域 |

T2 不必等全部控制优化才交第一段真实功能；但在 T4 之前不能把静态 shape 上传测试列作 tree resident broadphase。父负责唯一项目执行、有效 RED、正式 green、静态/格式/源码绑定/wheel；作者不并发 import/probe。失败后保原输入、阈值、状态和退出码，再定位算法/接口/精度，禁止为过门调 mass、caps 或 clearances。

冻结验收库包含八组，详数值在案例册：T01 两 R 的完整交叉 M/b；T02 R+双 P 分支未驱动分支反应；T03 非对角聚合 fixed tool 与原材料；T04 非主轴/current COM/gravity/虚功；T05 部分饱和/非饱和/延迟/未收敛；T06 N9/N16/N17、拒步/restore/观察/资源；T07 正负多周回、局部 pose/近触/有限端外 miss；T08 异构 W1/4/32、40步与相同完整输出。Panda 可作后续真实资产留出，不能替代这些机制测试。

## 7. 风险与后继节点

首要风险是 dense FP32 质量矩阵条件数、耦合 drive 收敛、每 world 串行长循环、全树位置舍入及跨两类 state 的 restore。先用 N1/2/3/9/16 和真实两后端原输入揭示边界；同输入失败必须暴露，不隐藏 HOST fallback。N16 的算力和编译成本需要测量；若实际过重，优先改善 batched kernel/因子复用，不能未经父裁定把验收改回只 N2。

后继一：同 world link–link/多动态体的设备候选与通用 contact mobility/材料适配。依赖 T4 的真实多 body/source 身份、accepted FK/J/M 和原接触材料地址；必须新增双方载荷/实际位移/历史共同发布，难度高。仅宽相多候选不等于设备机器人接触场景。

后继二：mimic/更长 delay/浮根。mimic 应先接完整物理行到独立列的约束投影，不能把 9 行树的两个 finger 删除；浮根要扩 6D 速度/COM/姿态积分与全 M 交叉块，依赖本仓已有 HOST 浮树/机械联动正确合同。各为独立物理任务，难度高，不由“树驻留”名称自动宣告完成。

本轮只会提交上述两份 NEW 文档；没有新增源码、测试、例程或数值证据。所有未来矩阵为 **UNEXECUTED**，等待父冻结后再逐阶段 tests-first。

## 父裁定与T1测试授权（2026-11-07）

研究两docs e07cac9已全文审阅并正常导入299fb73，八组手推输入、N16容量、原batch内tree banks和原子性路线接受为本片合同。父核T01/T02/T03/T05的主要独立解析式；这不是项目数值执行或功能验收。原PD建议已修为按实际tau/完整original clipped expected归一，宽cap及大项相消留出必须保留。

原robot继续同一隔离树、start e07cac9，**TEST_AUTHORING_ONLY**：恰NEW `tests/scale/test_resident_tree_step.py`、`tests/scale/test_resident_tree_collision.py`、`tests/reference_newton/robot/resident_tree_reference.py`；可只追加本casebook中的T1 API/保存schema说明。无product、旧source/tests、公共exports、第二runtime或父merge授权。父独占旧共享源码与全部执行。物理URDF可在这些测试/参考脚本中自包含文本生成；测试oracle禁止import产品几何/动力学函数生成期望，Newton脚本禁止import Phymeo。

正常入口采用`FixedRobotTree(program, state, control_max_iterations=512)`、原`TyxWorldBatch`、`set_controls/step/snapshot/restore/reset`、显式`tree_kinematics(keys)`与原`prepare_collision`。完整树checkpoint、混合单位读回和必要有界诊断的具体字段在T1 checkpoint先声明，父据此接薄hook；不得拿失败候选scratch当最后accepted记录。旧scalar prepared reader选中tree时本片允许明确拒绝，不能静默截断；不为首次核心功能强加新通用读回框架。shape绑定为本册明确group-local构造及`ResidentTreeBinding.from_asset`。

两专项覆盖T01–T08全部固定正负控、三个规模、二缓冲/restore/控制失败。第一有效RED从公共入口真正缺FixedRobotTree开始，先验证既有RobotModel/DrivenRobotProgram能表示T01，不在模块顶层导入尚不存在的新模块使collection报错。不得先运行collection或数值probe；测试、oracle、固定资产和计数规则均文本编写。完整保存每个实际窗口的输入/真实状态/时间/载荷/几何/计数；暖态区间的真读回与oracle明确区分。

Newton参考至少T01–T04，同输入各自推进，不喂Phymeo中间态；PD语义区别单列。N9/N16容量亦须独立质量/几何/推进核，N17分配前拒绝。测试不以一组Panda或一条固定姿态代替全部机制与材料。固定工程门沿案例册，不调输入/质量/cap/clearance求过门。

先交小commit、clean、精确文件列表/源码行数及UNEXECUTED报告，释放authoring；父全文review和唯一串行首RED后再授五新product及报告。产品kernel tuple可在那时依据既定storage布局冻结，本轮不得提前实现。原F2/C3所有已验收同源组不重跑，无新项目执行槽。
