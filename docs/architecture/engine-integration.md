# 外部接口层与多求解器集成

本页是 `feature/wh/phymeo-engine-integration` 的集成合同，仅文中明确标记的接口已实现。
现有公共入口和支持范围见 [场景编译](unified-scene-compilation.md)，原生实验缺口见
[PhySolve 运行时](../development/physolve-runtime.md)。P2PE 的 ABI/schema/handle 不进入本仓库。

## 承接路线

首个公共声明到原生输入的实现切片见 [原生场景编译](native-scene-lowering.md)，
包含绑定版本、坐标、控制/接触律及未接入的运行期边界。

PHYMEO 使用已有物理描述、资产、装配和可组合算子。外部接口层把 authored 场景转成
这些公共表达；专用 solver 适配再转换为其输入。不能创建 Newton ModelBuilder/Model/State
兼容别名，不能为使旧消费代码可运行而维护另一套权威模型或状态。

现有 `SceneDeclaration` 面向独立 world 的单 articulation 和有限静态环境；这不是
任意多参与者场景已经支持的证明。扩展应落到已有装配职责，保持可解释的能力报告。
缺少的组合明确拒绝，不能拆成不相互作用的独立仿真伪装支持。

| 边界 | PHYMEO 的职责 | 保留的限制 |
| --- | --- | --- |
| 公共编译 | 自有表达、初态、控制、来源身份、能力和诊断 | 原生文件入口也须保留完整环境和身份 |
| 专用模型转换 | 物理参数、坐标、控制及结果转换 | 不要求所有 solver 使用 MjModel |
| 字段绑定 | 单位/frame/参考点/时刻、存储借用及转换 | 不把同步 host snapshot 当作设备接口 |
| 执行 | 准备、提交、完成、接受结果、失败状态 | 明确部分推进，不能发布错误帧 |
| 生命周期 | 真实物理实例及可写资源的 owner | close 失败保留资源，外部 stream 只借用 |
| 测量 | 接触、frame、物理传感器真值及质量 | 产品相机、噪声和训练编排不进入此边界 |

## PhySolve 转换

CPU MuJoCo 3.8.0 仅作为本实验 solver profile 的模型编译、初始化和参考依赖，
不成为 PHYMEO 核心必选依赖。运行候选不安装或导入 Newton、Warp、MuJoCo-Warp。
`physolve_tyx` 的算法来源与运行依赖分别记录。现有六 CUBIN 保持字节不变。

碰撞排除由公共身份表达，PHYMEO 适配转换到已有原生输入，codedev 负责专属候选表及
派生数据。先验证现有原生表达是否足够；不先增加平行过滤机制。不配置新增规则时行为
不变。来源映射的 native geom 下标只在本次编译内有效，不能作为公共持久身份。

不同接触律没有名称相同即等价的保证。resident 的材料/响应与 PhySolve 原生约束律
分别声明；不得同时施加两次响应。关节命令力、驱动实际输出、约束广义力与 wrench
分别绑定，不能因为某个数组存在就声明结果为 measured。

## 执行与演进

PhySolveRuntime 提供版本 1 的字段借用和候选执行入口；同步 step/forward/reset 复用
同一事务，snapshot 仍为显式全量 D2H。执行阶段与限制见下节，不以快照包装高频接口。
resident 的单子步共同发布见下方 publication v1；原有入口默认仍逐 world 接受。
外部完整事务还需字段、生命周期及多子步验收，不能由 P2PE 整批 host 快照补偿。

每个缓存/备份/bank 登记 owner、容量、用途、更新和失效条件。编译时冻结索引、缓冲
和绑定；热循环不重复 FK/forward/接触响应，不逐 world 重新编译或跨层调度。
Graph 捕获资源完整保活，同一 stream 包装复用；捕获失败不 fallback direct。

公共集成合同、包版本、源提交、TYX 互操作、原生 ABI 和载荷身份分别管理。
字段借用合同首版为 1；完整跨 solver 合同仍待后续资格，不能推断任意版本组合已兼容。字段语义、采样
时刻和存储有效期变化要显式迁移；升级使用实际旧包和新候选跑同一测试套件。

### 字段借用与候选执行 v1

编译模型新增只读 `model_body_mass`、`model_body_com`、`model_body_inertia`，
分别为 float32 `[1,native_body]`、`[1,native_body,3]`、`[1,native_body,9]`。
首维为共享模板，不能作为 world 数量；两个 bank 借用相同存储。单位依次 kg、m、
kg*m²；COM 在 body-local，完整惯量为 COM 点、body 轴下 row-major 矩阵。
质量/COM 借用原生 model；惯量在冷准备由主惯量与惯量轴转换，PHYMEO 持有
36×native_body 字节的设备派生表及等量上传源，随 runtime 关闭释放。
不逐 world 复制，不增加 step/forward/reset 工作；本轮参数只读，变化需重新编译。
`execution_costs.body_parameter_derived_device_bytes` 单列设备表成本。
公共存储借用的 generation、候选写权限及失败关闭保活规则照常适用。

`storage_binding(contract_version=1)` 返回固定字段描述及两个 bank 的 TYX 标量视图，
描述含 shape/dtype、单位、frame、native index space、采样时刻和写权限。
字段含义以描述为事实来源；这不是 P2PE 的公共字段布局，消费适配仍须转换身份和坐标。
不暴露原生 model/Data/Host；字段借用本身不复制物理状态，不要求 resident 使用 MuJoCo 表达。

顺序为 `begin_execution → enqueue_step/forward/reset → complete_execution → accept_execution`。
begin 每事务复制一次 integration 字段；多个子步推进同一候选 bank。完成只确认和检查，
accept 才发布。外部消费者只在 prepared 阶段写候选，不能写 accepted bank；同一实例
只允许一个 pending 事务，异实例/过期 ticket 拒绝。出错后使用 pending_execution 取回
恢复 ticket，再 discard 或 close；完成/清理失败保留 owner 与资源以便重试。

借用数组与 canonical stream 在冷准备时创建；读写均使用该 stream，消费期间不得关闭
owner。owner 关闭后 binding 失效；原始指针
不能被 Python 撤销，消费者必须在准入处检查 generation，不能长期越权使用已缓存指针。
写入前由调用方验证有限值、四元数及物理坐标约束。设备端输出检查只阻止错误候选发布，
不代替输入预检，也尚不抑制错误之后的原生子步；P2PE execution epoch gate 仍待接入。

每子步将容量及有限值错误累积到固定 int32，完成时一次 4 字节 D2H 和 stream 同步。
masked reset 的未选观测保留与 forward 检查在同一 stream 完成，无第二次主机等待。
检查复杂度为 O(substeps × worlds × (q + DOF + controls))，主机 dispatch 随字段/子步数
增长，不逐 world 调用；候选复制为 O(worlds × integration width)，不复制完整模型。
原生 direct 条件循环仍可能内部同步，enqueue 不承诺完全非阻塞；Graph、外部 stream、
contact 借用、resident 及实际 C ABI 尚未因此获得资格。并发仅有实例锁，TYX 跨线程能力
另行资格化，本批不声称线程安全或吞吐提升。
校验预热当前执行 bank 0；bank 1 的 TYX 指针绑定首次使用成本尚未独立计数，不宣称
两 bank 的所有底层 launch 缓存均已预热。设备数组视图本身由准备阶段固定复用。

## 分批验收

### 接触测量 v1

`contact_observations=True` 冷准备启用，缺省 False 保持原成本。消费者检查
`contact_observation_contract_version=1`，通过同一 RuntimeStorage 借用接触字段。
position 为世界系接触中点，frame 为 row-major 法线/两切向，distance 为有符号距离；
wrench/local 为 `[force,torque]`，单位 N/N*m、参考点是中点，作用在 geom pair 第二侧。
world 为 0-based，geom 为本次编译原生 geom，须经编译身份映射转换为消费者 ID。

只有前 contact_count 行有效；position/frame/distance/wrench 是共享 float32 packed
记录的 strided 借用（每行 25 个 float），world/geom/dim 是每行 4 个 int32 的借用。
未用尾部不保证清零；不允许将行号当持续 contact ID。contact_count_by_world 是 world
索引，contact_count 是共享池总数。contact_sample_time 是每 world 的求解时刻秒数：
Euler/implicitfast 的 step 在最后子步前以一次 `4*N` 字节 D2D 捕获实际原生时间，
forward/reset 使用对应当前时刻；
命令更新和 masked reset 的未选 world 保留原测量及原采样时刻，不冒充步后 body 数据。

适配复用锁定 frontend.contact_force，检查 world/geom/dim、EFC 有效行及容量后提取，
并将有限值错误并入同一 4 字节发布 gate。每次 enqueue_step 只在最后子步提取一次，
没有新增 collision/forward/solve、CPU 物理或逐 world dispatch。纯命令更新 D2D 保留
旧测量；masked reset 用新 selected 与旧 unselected 记录在设备上合并，保持记录值，
不承诺行序。联合记录超过共享 C 容量明确拒绝整个候选，不截断或增大原生接触池。

runtime 拥有两份发布记录、一份 reset scratch、force/ID scratch 与各 world 计数/时间；
存储为 `376*C + 20*N + 12` 字节，C 是原生共享容量，不再乘 N。update/merge/copy
均 O(C+N)，计数由 execution_costs 披露；这是派生测量副本，不是第二份物理状态。
关闭失败保留全部 owner，旧 binding 在成功关闭后 stale。此合同本身不授予 Graph、
SensorView 或 D2D C ABI 资格；这些后续沿同一公共存储消费。

### 标量控制与字段更新

`NativeTemplate.scalar_controls` 绑定公开关节名、原生 actuator、增益和初始目标。
标量 PD profile 的 `target_position` / `target_velocity` 是 runtime 持有的命令；
`ctrl` 是按 `kp*q_target + kd*qd_target` 生成的执行输入，不是目标角度。
此 profile 拒绝 writable ctrl 和同步 step(ctrl=...)，避免双来源控制。
其他原生 profile 保留原有 ctrl 语义，不隐式加入这些字段。

写候选目标后必须调用 `enqueue_update(compose_controls=True)`；写 qpos/qvel 必须
请求 `refresh_state=True`，两个标志可组合。它们描述调用方已准备的写入计划，
不是自动 dirty tracking。更新不推进时间；状态写入执行一次原生 forward，纯命令
更新不做 forward，而保留上次 solve-time 观测及容量计数。discard 后不能因旧失败
候选的容量计数再次拒绝合法命令。目标与原生 ctrl 一起接受，masked reset 一起恢复。

每次候选准备复制 11 个 integration 字段及两个目标 bank；纯命令更新另外 D2D
保留 qacc、actuator/constraint force、xpos/xquat 和三个容量计数。复制量随 world
和相应字段宽度线性增长，主机调用次数不随 world 数增长；这不是零复制动作。
没有整批 host snapshot，完成仍是一次 4 字节检查读回；原生 direct 内部同步另计。
URDF dynamics/mimic 的 scalar_pd 转换在 B3e 承接，见[原生装配](native-assembly.md)；
转换通过仍不能替代机器人运动资格。

### 选择世界更新 v1

`selected_update_contract_version=1` 为 `enqueue_update` 增加可选
`selected_worlds`，必须是 canonical device 上连续的 TYX bool[nworld]。
这是观测保持范围，不替代调用方正确的 masked scatter：调用方仍负责只修改
选中世界并在设备预检通过后写入。mask 内容和存储保持到 complete/accept/discard，
所有写入与更新使用 canonical stream；runtime 保留 mask owner 到事务结束。
qpos/qvel 更新仍执行一次批量 forward，随后保留未选世界旧力/body/接触观测；
纯命令仍不做 forward。没有参数时沿用原行为，不改 Host、CUBIN 或碰撞排序。

### 设备 reset v1

`device_reset_contract_version=1` 允许 `enqueue_reset(ticket, selected_worlds=mask)`，
mask 为 canonical device 上连续 TYX bool[nworld]，与 host `mask` 参数互斥。
runtime 保留借用直到 complete/accept/discard；调用方负责 canonical stream 顺序及
期间不改变 mask 内容。恢复 integration、control 和 warm-start 后执行一次批量
forward，未选世界保留旧派生观测和接触；设备全选/空选不读回 mask 判定。
本接口不增加 H2D 或 host snapshot，完成仍检查紧凑状态；原生 direct 条件同步存在。
host reset 不传 selected_worlds 时保持原路径。该合同不是 Graph 或并发资格声明。

### FREE body update v1

`free_body_update_contract_version=1` 提供显式冷准备
`prepare_free_body_updates()`，返回当前编译的 `(name, native_body, q_address,
dof_address, local_com)` 不可变绑定。仅 world-parent 单 FREE joint 的 body 合格，
受约束 child 不拥有独立绝对位姿。返回的原生地址不充当跨编译公共实体身份。

准备一次后，`enqueue_free_body_update(ticket, values, selected_bodies=mask,
components=...)` 使用既有 begin/complete/accept/discard 事务。输入是 canonical
device 连续 float32[N,F,13]：世界 body-origin XYZ、body-to-world XYZW、世界系
COM 线速度及角速度；mask 是 bool[N,F]。F 和列序来自本次冷绑定。components
支持 position/rotation/linear_velocity/angular_velocity 的非空无重复子集；忽略
未请求分量，未选 body 不写，未选世界的派生观测保持。调用方持有并保持输入不变到
完成或丢弃；runtime 留存借用，拒绝与原生公开存储或彼此重叠的输入。
本入口的观测发布范围由 body mask 的 world 并集决定。增量
`free_body_component_contract_version=1` 允许 components=None，借用 int32[F]
component_flags（position=1、rotation=2、COM linear=4、world angular=8），选中
body 须为 1..15；未选 body 的 flags 不消费。可选 bool[N] selected_worlds 与 body
世界并集合并，供同 ticket 的其他字段写入发布；compose_controls 在同次 forward
组合控制。新增输入均须 canonical/连续、彼此及与原生存储不重叠，借用到事务结束。
只有单独输入转换错误抛 PhySolveInputError，调用者可 discard 后继续；混合其他
设备错误仍是 RuntimeError，不将求解/容量故障伪装成可恢复输入错误。

旋转改变时保留未写的世界 COM twist，原生 qvel 转换为 origin/world 线速度和
body-local 角速度。位置单独修改直接保留原生 qvel，不增加无用的往返误差。
单位四元数的 float64 平方范数误差须 <=64*float32.eps，通过后归一化；不翻转符号。
有限值、范数及转换溢出在 device 上整批预检，所有写入受同一错误位门控。非法值
不进入原生 q/qd，完成时报错；丢弃后可再用。有效输入执行一次原生 forward，
不推进时间、不改变结构 qpos0。原生条件循环同步依旧存在，不宣称异步/Graph 已合格。

该 PhySolve 专用适配使用已有原生坐标及 COM，不建立通用 Newton-like 模型。
冷分配为地址 8F、COM 12F、转换 scratch 52NF、body mask NF 和 world mask N 字节，
共 20F+53NF+N；scratch 只用于验证后 scatter，不是第二份权威物理状态。
每次 O(NF) 三个批量转换/选择/scatter 内核，随后一次现有 forward；无逐 world
Host 调用、热期 TYX 分配或 host mask/状态快照。错误归并复用现有 4-byte completion。
准备失败资源保留到成功 close；关闭后原绑定不能用于新的 runtime。

P2PE 的 Stage 自由刚体/浮根建模、q/DOF 布局及 body Array/TensorView 消费仍需后续
接入，本合同通过不能被记为 P2PE C ABI body-write、安装或 GUI 资格。

### 候选设备观测执行 v1

`observation_execution_contract_version=1` 增加只读 `observation_bank`：idle 返回
已接受 bank，enqueued/completed 返回待接受 bank；prepared 阶段明确拒绝。
消费者在同一 canonical stream 排入观测 kernel，须以公开 RuntimeStorage 的
`execution_error` int32[1] 为门禁，非零时不写消费者输出。它是 runtime 全局错误
位，不是按 world 分配的物理字段；两个 bank 借用同一数组，只读。
这个入口不提前接受物理状态，仍须 complete/accept；discard 等待已提交工作后
清除该 gate，再回到原 bank。执行失败和异常观测仍须通过显式完成边界报告。
借用有效期、stream 顺序和关闭重试沿用存储合同，不新增物理状态副本。

### 步后 body 观测 v1

PhySolveRuntime 的 `body_observations=True` 在冷准备启用端点观测，默认 False
保留原有成本。消费者检查 `body_observation_contract_version=1`，从同一
RuntimeStorage 借用 body_position、body_rotation_wxyz、body_com_twist。
索引为 native_body，位置是 body origin，旋转为 body-to-world WXYZ；twist 为
世界系 `[linear_at_body_COM,angular]`，单位分别 m/s、rad/s。采样为 integration
endpoint，与原 xpos/xquat 的 solver evaluation 区分，不改变旧字段含义。

适配复用锁定 physolve_tyx 的 kinematics/com_pos/com_vel，每次 enqueue_step 的
最后子步后运行一次，不执行接触、动力学或完整 forward。forward/reset 已有正确
运动学输出时只转换到端点输出；命令写入只复制旧输出。masked reset 保持未选
端点输出逐位不变。新输出进入同一设备有限值 gate，完成失败不允许发布。

Runtime 拥有一份共享的派生 workspace 和两组只读端点输出；只借 q/v/mocap，
不再维护积分状态。workspace 包含 body/geom/site frame、subtree COM、cinert、
cdof、cvel 和 cdof_dot，这是复用现有算法的额外计算/存储成本，不是只算 pose。
输出成本为 `2*N*nbody*13*4` 字节；workspace 按实际数组容量另计。execution_costs
披露端点运动学次数、输出生成次数与总设备存储字节。纯命令更新增加三个输出
数组 D2D，读回不需要 host snapshot；原生 direct 条件同步和显式完成仍保留。
准备失败和清理失败必须保留这些 owner；关闭成功后借用与原 storage 一同失效。

B1 先实现公共表达到原生输入的最小转换，并用 resident 小样证明其不依赖 PhySolve
格式；覆盖固定/自由根、偏心 COM、多 shape、初态/控制、来源和 world 映射。
B2 接通原生 step/forward/reset，关闭机器人历史失败，逐层记录分配/传输/同步成本。
随后与 P2PE 验证 host、D2D、固定 Graph、测量及独立安装，不将源码测试等同消费者资格。
resident 本轮限定受支持简单模型的 C ABI direct/host、动作、状态、接触及局部 reset；
其 HOST 窄相/诊断成本单列，完整 Graph/传感器/GUI 不是该 profile 的本轮要求。

同实现沿用原容差；新案例先冻结 SI/frame/初态/精度/步长及物理标准，再比较候选。
自由落体、支撑、COM/惯量、驱动与接触方向要有独立物理参照。Panda/FANUC 历史失败
保持未通过，不能通过放宽门限、改初态、关闭接触或改求解 CUBIN 关闭。

## 本分支 B0 实测

2026-11-11，新独立 CPython 3.13.13 环境从本分支源码构建 wheel 安装，MuJoCo 3.8.0。
禁止三包导入进程运行 `test_mujoco_compiler.py`：53 通过；原生 runtime 小模型选择集
10 通过、14 个机器人相关项未选。本批未重验机器人、不授予其资格。安装包来源和
实际 Host、六 CUBIN 哈希保存在配套实验根 sources/evidence，原始载荷未修改。
全仓文档检查仍失败：479 个缺失本地目标与旧检查点逐行相同，没有新增链接错误。
此次只增文档，未改变运行接口、数值或碰撞行为；完整 CUDA/C ABI/Graph/GUI 资格待后续。

## Stream binding v1

`PhySolveRuntime.stream_binding_contract_version=1` declares cold
`bind_stream(handle=0, owner=None)`. A nonzero handle requires a lifetime owner
and must belong to the selected TYX CUDA context. Zero restores the runtime's
own stream. The caller must finish and accept/discard pending execution first;
capture on the old or new stream is rejected. The caller keeps external stream
handles alive until successful unbind or close, including cleanup retries.

Binding completes the old stream, prepares both native banks on the candidate,
completes preparation, publishes the canonical stream, and releases retired
native sessions. Physical arrays and storage generation do not change. All
borrowers must use `RuntimeStorage.stream` at submission time or update their
cold launch bindings when the operation returns. An unchanged handle reuses
the existing wrapper without preparing native resources again.

Preparation failure restores both old bank bindings without advancing physics.
Failed rollback or retirement retains both stream wrappers and permits close
retry only (`error.stream_bindings_changed=True`). Close never destroys a
consumer stream handle. TYX `CudaStream.validate_context` is required; this
contract does not qualify fixed Graph, cross-context streams, or concurrent
mutation of one runtime. Locks remain per runtime. Native direct conditional
readbacks and explicit completion costs remain visible.

## Fixed step Graph v1

`step_graph_contract_version=1` adds `prepare_step_graph(steps=4)` and
`enqueue_step(ticket, steps=4, execution_mode="cuda_graph")`. Capture is cold,
requires idle execution, and does not advance physics. `step_graph_ready`
reports actual preparation. The runtime owns two TYX native stream Graphs;
each borrows one native bank, shared prepared observations and error checks.
Direct remains explicit and Graph failures never fall back to direct.

Begin/candidate preparation and completion/publication keep their existing
contracts. Every Graph contains four native advances, per-step validation,
endpoint body measurements and last-substep contact sampling. Capture does not
increment physical observation counters; runtime capture/replay counters are
cumulative. Forward, updates and reset remain direct operations.

Successful stream rebinding retires Graphs before native Host resources and
requires preparation again. Failed native rebinding restores the previous
Graph after successful rollback; failure of that restoration permits close
retry only. Close completes the stream, destroys Graphs, then closes Host
handles and releases storage/streams. Failed Graph destruction retains owners.

Borrowed native writable inputs still require caller-validated finite values.
Compact execution errors prevent publication, but arbitrary invalid raw input
or a fatal CUDA context error does not promise in-process recovery. Consumers
must prevalidate input; P2PE tests its finite-value rejection separately.

## Solve-time body measurements v1

`PhySolveRuntime(body_measurements=True)` opts into
`body_measurement_contract_version=1`. The default is disabled with zero
additional measurement storage or evaluations. It requires the paired
`physolve-tyx-frontend` a7 source with `PostConstraintWorkspace.contract_version=1`;
older frontends fail explicitly when this capability is requested.

Readonly storage fields are defined once in `PreparedMeasurements.fields` and
exposed through `RuntimeStorage.fields`: COM position, COM linear specific
acceleration, angular acceleration, COM linear/angular twist, and internal
force/torque at COM. All use float32, world coordinates, SI units and native
body indices. Specific acceleration is physical acceleration minus gravity.
Internal wrench is the parent-on-child resultant, not commanded joint effort
or a scalar constraint force. Native body 0 is the world aggregate and must not
be exposed as a valid joint F/T measurement. Compilation mappings resolve
public bodies; no consumer may assume public IDs are native offsets.

`measurement_sample_time: float32[nworld]` is the retained solver clock. A
multi-substep advance measures its final solver evaluation, before the final
integration, not the endpoint pose. The prepared RNE composition retains that
evaluation's qvel and consumes matching native cvel/qacc/geometry. It evaluates
once per logical advance without another forward, solve, or contact response.
Forward/reset evaluate at their new solver time. Selected refresh/reset retain
every unselected measurement and timestamp; command-only updates copy prior
measurements without RNE. Mounted site transforms and actual SensorView
IMU/F-T projection are not implemented by this body-level contract.

The runtime owns two published measurement banks and one shared RNE workspace;
all operations use its canonical stream. Completion merges non-finite
measurement error bit 1048576 into the existing compact validation readback;
failed execution cannot publish and can be discarded. Graphs retain this
workspace; close completes work, retires Graph/Host resources, then releases it.
Preparation followed by cleanup failure retains the runtime for explicit retry.
Native arrays may not be replaced while borrowed descriptors are live.

Additional device storage is `240*N*B + 4*N*nv + 8*N` bytes, plus `8*N` for
equality counters when present (`B` includes the native world body). The
N=1/8/64 two-body hinge reports 492/3936/31488 bytes. Each logical advance adds
one device velocity snapshot, one time snapshot, one existing RNE composition
and one publication kernel; no additional CPU physics or host snapshot.
The native direct conditional readbacks remain visible. Throughput and peak
memory are not qualified here. Source tests do not qualify an installed wheel.

## Selected body-site projection v1

`BodySite(name, body, position, rotation_xyzw)` declares a named body-local
measurement site independently of native indices. `NativeTemplate.bind_body_sites`
resolves public body names into an immutable compilation-bound `NativeBodySites`.
Unknown bodies, duplicate names, nonfinite poses and non-unit rotations fail.
This does not insert native sensors or recompile the physical model.

`PhySolveRuntime.prepare_site_measurements(binding)` requires body measurements
and the identical compilation object. It returns a runtime-owned prepared plan
with `contract_version=1`; the consumer borrows it. Preparation registers the
owner before uploads. Partial preparation and failed close retain resources for
retry. Close completes stream work before releasing projection buffers and then
native resources. A successful runtime stream rebind is reflected by the shared
storage binding; no second stream wrapper is created.

`sample(selected, count, gate, rows)` enqueues on the canonical stream and returns
without waiting or producing a host snapshot. The caller must prevalidate unique
world IDs and `count[0] == rows` in stream order, exclude alias with the plan's
outputs/errors, retain inputs, and exclude concurrent mutation/close. The method
checks capacity, dtype, shape, stride and device. It guards world indexing and
reports nonfinite output through `errors`; publication requires completed error
validation. This prepared API is not a replacement for a consumer selector gate.

Output is float32 `[selected_row * site_count + site, 12]`, ordered as specific
acceleration (m/s²), angular velocity (rad/s), force (N), torque (N·m). All vectors
use site axes; wrench is parent-on-child at site origin. Let `R` map site axes to
world and `r` point from COM to site in world coordinates:

- acceleration: `R.T * (f_com + alpha × r + omega × (omega × r))`;
- angular velocity: `R.T * omega`;
- force: `R.T * F`;
- torque: `R.T * (tau_com - r × F)`.

Input pose borrows existing retained solve-time `xpos/xquat`; measurements borrow
the matching body measurement bank. There is no new pose mirror, FK, forward,
solver advance, or RNE evaluation. Sample time is `measurement_sample_time`, not
the integrated endpoint clock. Masked reset and command-only updates preserve
the established bank/time semantics. Consumers may explicitly convert wrench
direction to their public schema; PHYMEO does not silently change it.

Projection is O(selected worlds × sites). Each plan owns `48*N*S + 32*S + 4`
device bytes: output, body indices, local poses and an error flag. These bytes
are reported by `site_measurement_device_bytes`; consumer borrowed-byte reports
must not be added again. Body measurement storage remains separately reported.
Plans are prepared cold and reused, with no host snapshot or per-world dispatch.
This contract qualifies the PhySolve path only; resident, ray sensors and mounted
sensor runtime mutation require separate capabilities and tests.

## Prepared selected ray queries v1

`RaySurface` is an immutable named body-local geometry declaration, independent
of MuJoCo and solver state. It admits plane, sphere, capsule, cylinder, ellipsoid,
box and triangular mesh. Dimensions are metres (box half-extents, capsule/cylinder
radius and half-length); local pose uses XYZW. Unsupported geometry is rejected.
Dimensions and mesh triangles must remain valid after FP32 representation.

`NativeTemplate.compile_ray_scene()` converts native geometry once into this
public representation plus compilation-bound body indices (`NativeRayScene`).
All public shape identities must be present. Mesh vertices/faces and geom pose
come from the same compiled model, including native mesh preprocessing. This is
an explicit PhySolve adapter, not a requirement for every PHYMEO solver to use
MjModel. A returned label indexes the scene's named surface tuple, not a consumer
public ID or a globally valid native geom ID.

`PhySolveRuntime.prepare_ray_query(scene, max_rays=...)` returns a runtime-owned
`PreparedRayQuery` and requires endpoint body observations. The plan borrows both
body pose banks, owns static geometry, mesh BVHs and output/error scratch, and
uses the current canonical stream. No state snapshot, extra FK or forward is
performed. Geometry is not rebuilt or re-uploaded while stepping or sampling.

`sample(selected, count, gate, origins, directions, rows=..., near=..., far=...)`
accepts same-device contiguous int32 selection/count/gate and float32
`[capacity,rays_per_world,3]` origins/unit directions, with rows <= capacity <= nworld.
The `prepared_ray_capacity_rows` capability permits persistent input capacity;
only the selected prefix is read and dispatched. Coordinates are physical-world
coordinates without display/env-origin offsets. It emits only selected rays;
there is no render-all-then-gather. Caller prevalidates unique selectors and
count==rows, excludes alias with plan resources, retains inputs and prevents
concurrent close/mutation. Capacity, layout and device are checked on host;
world bounds, finite/unit directions and transform arithmetic are guarded on
device. Nonzero errors must suppress downstream output publication. An existing
external/native error gate suppresses query work.

Output labels/hits are flattened selected-row then ray order. Hits have four
float32 columns: range in metres and a world-space **ray-facing** unit normal.
All surface kinds share that normal policy, including interior exits. Misses
use label -1, range +inf and zero normal. Equal ranges keep the first declared
surface. Clipping is closed [near,far], rounded inward to representable FP32
limits. This prepared contract does not alter the older host query normal policy;
consumer depth/miss conventions require explicit field conversion.

Mesh BVHs are built cold in local coordinates and shared across worlds and
identical geometry; dynamic rigid poses transform each ray instead of refitting
triangles. Every ray still traverses the surface list: cost is
O(selected worlds * rays * (surface count + mesh BVH traversal)). This is **not**
a scene-level BVH or a claimed O(log surface count) query. Allocation and dispatch
do not scale through per-world host calls. Persistent output is 20*N*max_rays
bytes; geometry/indices/errors are separately owned. `costs()` reports explicit
array bytes and shared mesh count; TYX internal Mesh/BVH storage is additional.
Counters count host query submissions (including capture), not external Graph
replays. Full throughput and peak-memory qualification remain pending.

Preparation registers owners before async work, including nested TYX Mesh/BVH
construction. Failed completion/close retains resources for retry. Consumers
must destroy externally captured queries before closing/rebinding the runtime;
captures bind one pose bank, input/output addresses, clip bounds and stream.
After a published bank switch, use the corresponding bank's capture or recapture;
a replay does not re-evaluate the Python bank selector. Fixed query Graph
replay is tested, but dynamic bank selection inside an externally captured query
is not implied. Closing invalidates storage; reusing a stale plan fails.

Camera calibration, mounted-ray generation and P2PE SensorView dispatch are the
next integration step. This prerequisite alone does not enable Camera/LiDAR in
P2PE metadata, qualify resident rays, or qualify an installed SDK/GUI.

## Contact reference v1

`contact_reference_contract_version=1` adds readonly borrowed storage
`contact_body_com_position: float32[nworld,native_body_count,3]` when contact
observations are enabled. Units are metres, frame is the physical world's
coordinates, and the COM positions belong to the same retained solver
evaluation as `contact_sample_time` and the contact wrench. Native body 0 is
included; consumers use the compilation identity mapping, not public body IDs
as raw offsets. No rendering/environment display offset is applied here.

This is not endpoint body pose. After integration it is copied from native
solver `xipos`, which endpoint body kinematics never overwrite. Forward/reset
capture their new evaluation; masked reset retains unselected COM and sample
time together. Command-only updates copy the previous measurement provenance.
Two runtime-owned banks keep the values alive for fixed Graph replay and share
the original storage generation, error admission and close ordering.

Nonfinite selected COM sets the existing contact measurement error bit 131072;
completion rejects publication. Ignored worlds retain the accepted finite
reference. This check does not replace caller input prevalidation.

Added persistent storage is `24*nworld*native_body_count` bytes. One reference
copy is O(N*B) per contact extraction (once after the final substep), and
command-only copying adds `12*N*B` D2D bytes. Combined measurement storage is
O(contact_capacity + N*B); the existing contact byte metric includes this
allocation. No new collision solve, FK, host readback or per-world dispatch is
introduced. The reference is a prerequisite for consumer sensor moments, not
an implementation or qualification of SensorView itself.

## Cold visual scene v1

`compile_native_template(..., capture_visuals=True)` optionally captures a
`geometry.visual.VisualScene` at compilation. `NativeTemplate.compile_visual_scene()`
returns that owned immutable snapshot without reopening sources or re-running FK.
Instances name public physical bodies, not native geom indices. A null body means
static world geometry. Position is metres; rotation is xyzw; dimensions are native
half extents/radii, with local Z as the capsule/cylinder axis. Mesh vertices are
local metres, triangles int32, and optional float32 UVs share the vertex index space.
UV values retain the native compiled convention, not an unverified texture contract.

URDF visuals must be retained before MuJoCo parses the file. A separate display-only
spec preserves named bodies and removes collision geoms; it never enters the solver.
The existing physical compilation is unchanged. Assets without any URDF visual use
explicit `collision_geometry` fallback; empty visual links acquire no placeholder.
Finite native planes export zero-thickness rectangle meshes, including their rotation;
fully infinite planes retain native display spacing. A partly infinite rectangle,
unsupported geom or texture rejects the optional visual query with a diagnostic.
URDF texture declarations are checked before the importer can discard them.
Physical compilation and execution can continue after that optional refusal.

The snapshot owns bytes-backed readonly buffers; consumers cannot re-enable writes.
Native UV corner indices expand once at compilation, without changing physics mesh
topology. P2PE borrows these buffers and copies only requested caller output. Assets
are per template, not per world; complexity and storage are O(visual vertices + faces
+ instances), with one additional cold URDF display compilation per visual asset.
No per-step parsing, model copy, device allocation or synchronization is introduced.
The native adapter's optional CPU MuJoCo dependency does not enter the geometry types.
Textured display, generic resident export and installed/GUI qualification remain pending.

## Resident publication v1

`backends.tyx_world_batch.TyxWorldBatch.publication_contract_version=1` 声明
`step(h, require_all=True)` 与 `step_contacts(plan, h, require_all=True)` 支持单子步
共同发布。默认 False 保留原有逐 world 接受行为；非 bool 在执行前拒绝。

已有 device 完成、状态摘要、tree 与 contact candidate 检查全部结束后，若有 world
拒绝，抛出 `BatchStepRejected`，`rejected_worlds` 是实际拒绝 world 的注册顺序元组。
所有 world 的权威 state bank、tick、延迟控制历史与 accepted contact receipt 均不
发布候选，state_epoch 不变；已显式提交的 held controls 保留。恢复控制后可继续，
无需 full reset。树的工作诊断仍按已有语义失效，不承诺保留失败尝试的诊断输出。

该策略只覆盖一次 substep；连续多次调用不构成共同事务。它没有改变数学、排序、
容差或接触律，也未增加物理 bank、checkpoint 下载、device allocation 或同步。
新增 HOST 判断复用原有每 world int32 状态摘要，O(N)。数值拒绝不会额外触发
设备异常恢复同步；异常计入 failed_steps，并单列 aborted_batch_steps。已有
completed/accepted/rejected_world_steps 仍只统计已发布调用，含义不变。

step_contacts 的原有 HOST 窄相、物理 candidate/诊断读取和成本仍存在；上述“无新增
下载”不是该接触路径零传输承诺。此合同不使用 Newton、Warp、MuJoCo-Warp、
MuJoCo 或 PhySolve 的运行 API，也不意味着两种 solver 的物理响应相同。

正式 `tests/scale/test_resident_publication.py` 与既有 test_resident_step 共 31 项通过，
覆盖 interpret/实际 CUDA、N=1/8/64、球/平面支撑冲量、失败恢复、局部 reset、默认
语义、树接触与延迟控制历史。跨实现微差不在本批范围；证据见配套实验根
`evidence/b7a-summary.json`。当前仍需公共 caller-buffer 字段读取、批量输入/reset
准备、可重试清理及 P2PE 实际 C ABI；不能把本项算作 B7 完成。

## Resident cleanup retry

B7b 收敛上述清理前置：batch、readback、collision/contact、contact bundle 与 convex
query 开始 close 后拒绝继续执行；仅在全部释放成功后标记 closed。释放回调按依赖
逆序执行，成功才移除；失败回调、剩余缓冲及其 runtime 保留给下一次 close。已经
释放成功的缓冲不重复 free，owned bytes 只在成功后扣除。底层 TyxCollisionBatch
自身的公共 query/prepare 入口也拒绝 closing 状态，不只依靠外层 plan。

构造期间发生分配失败且清理也失败，抛出 ResidentCleanupError，其 cleanup_owner
保留未返回的对象及嵌套失败依赖；调用方必须保留异常或该 owner 并重试 close。
已有 batch 的失败子 plan 在首次分配前登记，仍可由 batch.close 统一回收。释放失败
不自动丢弃 owner，不承诺 CUDA context 永久损坏后一定能回收。

正常推进和关闭不新增显式同步。若既有设备异常恢复同步失败，关闭须先重试完成
确认，成功前不得 free；这是故障清理边界，不能用于掩盖正常热路径同步成本。
TYX 原有 runtime/module 关闭内部同步保持不变。无新物理 bank、设备分配、状态
复制或求解计算；关闭复杂度 O(持有资源数)，失败停止后允许独立 owner 稍后重试。

正式测试 test_resident_cleanup 覆盖真实 CUDA free、runtime.close、恢复同步与
嵌套构造失败，N=1/8/64 的推进后清理，以及读回、凸查询、树接触和 bundle。
原有关闭测试改为验证保留依赖和重试，数值断言与容差不变。测试/审查和源码身份
见配套实验根 evidence/b7b-summary.json；这仍不是 resident 实际 C ABI 或安装资格。

## Resident caller-buffer fields v1

`TyxWorldBatch.prepare_field_read(keys=None, fields=("position", "velocity"))` 冷准备
`ResidentFieldReadPlan`。支持球体、标量移动/转动关节和普通固定/浮根树的同构选择。
不同坐标布局混选及仿射树字段明确拒绝；原有 snapshot、kinematics 不改变。
keys 默认注册顺序，显式 keys 固定选择及顺序，重复/未知 world 拒绝。空选择布局
为 `(0, 0)`，不启动计算；非空布局由同构坐标数和 fields 顺序确定。

plan.layout 是不可变冷 metadata：contract_version、world_keys、shape、float32、
coordinate_types，以及各 field 的 offset/width/units/frame/reference/sample/source。
每一行是一个所选 world，每个 field 的完整坐标连续存放；普通树保持模型 joint
顺序。球体 position/velocity 是 world COM 的 m、m/s；关节是 m 或 rad 及对应速度。
`applied_effort` 读取最近接受子步的量：球体为外加 COM 力，关节为 actuator drive
实际输出；不包括额外广义负载或接触，不是 held command，也不是 P2PE joint_force。
初态为零，显式恢复可恢复历史实际值。控制更新本身不重写该测量。

`plan.read_into(values, ticks=None)` 接受精确 layout shape 的 writable C-contiguous
float32 NumPy caller buffer；可选 ticks 为 int32 `(selected_count,)`，计数已接受子步。
返回当前接受状态 epoch；不返回逐 world 包装或完整快照。所有 buffer 先验证，再
启动写入；dtype/shape/stride/只读/输出互相别名及物理/控制缓存别名均拒绝。HOST
缓存检查覆盖 CUDA 模式和 owner 替换。plan 和 batch 必须保持开启，同 owner 使用
原有 RLock；不新建跨实例锁，也不授予异步或外部 stream 资格。

准备阶段建立索引、输出 scratch 并预热两 bank 读取 kernel。一次读取只选择当前
接受 bank，按冻结索引 gather，不再次 FK、mechanics、checkpoint 或解析字段。
interpret 直接写 caller，不建立中间 values 矩阵；CUDA 先确认 gather 完成，然后
一次 values D2H，只有请求 ticks 时再一次 ticks D2H。正常 CUDA 读有一次显式完成
同步，blocking D2H 的完成成本另计，不声称零同步。HOST 本来就可能只读被接受的
world；拒绝推进的 world 保留其旧值/tick，读取不冒充新的成功推进。

CUDA 持久空间为 `4*S*W + 12*S + 8*W` bytes，分别是 values、ticks、world 索引和
字段列索引；S 为所选 world 数，W 为 packed 行宽。CPU 仅留索引/ticks，values
预热临时矩阵在构造返回时释放。索引准备 O(S+W)，执行 O(S*W)，无逐 world Python
读写；解释器 HOST alias registry 只保留既有数组引用，不复制物理状态。
一旦 transfer 失败，caller 输出可能部分写入，不能消费；异常不改变物理接受
状态，恢复后可重读。cleanup 沿前节失败重试合同，无静默 snapshot fallback。

正式 tests/scale/test_resident_field_read.py 验证 N=1/8/64、字段与 world 顺序、
双 bank、局部 reset、caller 边界、别名/失效、传输失败和清理重试。测试的多 world
固定树控制组合存在原路径部分拒绝，单列记录而非修改容差或追加 solver 调查。
这只是公共数据前置；P2PE resident 编译/控制/reset、完整 body/contact 字段、
实际 C ABI 和独立安装资格仍待完成。证据见配套实验根 evidence/b7c-summary.json。

## Resident prepared HOST controls v1

`TyxWorldBatch.prepare_control_write(keys=None, fields=...)` 冷准备同构所选 world 的
控制列及顺序；支持球、标量移动/转动关节、普通固定根/浮根树，空选择、重复/未知 key、
混合布局、仿射树和未知字段明确拒绝。fields 必须非空且无重复。不可变 layout
声明 shape、float32、offset/width、单位、frame、reference、source 和采样含义。

球只开放 world COM `com_force`；关节开放 target_position、target_velocity、
feedforward、external_effort，转动关节/固定树还开放 COM force/torque。字段先按
fields 顺序、各字段再按模型关节顺序打包。标量 slider 的 external_effort 是原控制
包已经合并广义负载与轴向外力后的 total_axial_load，不能直接当作 P2PE joint_force。
target/rate/feedforward 服从原模型延迟；外部负载即时生效，未改驱动或物理算法。

`plan.write(values)` 接收精确 shape 的连续 float32 HOST NumPy buffer，只读输入
允许，非有限/非零 subnormal 拒绝。一次全量预检后只替换所选控制列；未选世界与
字段保持不变。`read_into(values)` 直接填 writable caller buffer，读取 held command，
不读取执行器实际输出。输入/输出均不得与物理或控制 owner、plan scratch 别名。
原接口/reset 更换 held buffer 后使用当前 owner；布局变化或关闭后拒绝继续。

没有逐 world 包装、snapshot 或全批 held 副本；预备索引/校验/回滚缓冲为
`(sizeof(intp)+10)*S*W` bytes，Windows x64 为 `18*S*W`。正常读写 O(S*W)，
另有 O(owned HOST buffer 数) 的别名检查；无新 device 分配或即时上传/同步。
等值输入保留已有 dirty 状态，不制造新上传。变值仍沿原逻辑在下一 step 一次上传
整个所属控制 bank，成本随完整 owner 增长，尚未实现 selected-only H2D 或 D2D。
这些 HOST 控制缓冲本来存在，不增加第二份权威物理状态。

执行 scatter 异常恢复所选旧值；回滚也失败则关闭该 batch 的后续操作准入，仅允许
close/rebuild，owner 保留到清理完成。控制写入不更改接受状态 epoch、延迟历史或
已发布观测。锁仍为每 batch 原有 RLock，没有新增跨实例锁。固定树既有控制提交
改为填充自身可写 held owner，保持原先全量验证和外部 packet 不可变语义。

选测正式 test_resident_control_plan 覆盖 interpret/实际 CUDA、N=1/8/64、乱序选择、
新旧控制入口一致、局部 reset 未选世界保持、延迟历史、字段语义、错误回滚和关闭；
同时复用原固定树 delay/restore CUDA 回归。仅验证受影响集合，不重复跨 solver
微差调查。配套实验根 evidence/b7d-summary.json 保存命令、身份、计数与审查。
prepared reset、完整物理字段及实际 P2PE C ABI/安装仍待完成；本批不授予 B7 资格。

## Resident prepared reset v1

B7e 的 `TyxWorldBatch.prepare_reset()` 冻结构造初态与控制，返回同步 HOST reset
plan。`plan.reset(mask=None)` 的 bool 连续 NumPy mask 与不可变 layout.world_keys
顺序一致；None 为全选，全 false/空 batch 无上传、kernel 或 epoch 发布。支持球、
球/平面、标量移动/转动关节、普通固定根/浮根树；含仿射树的 batch 明确拒绝。
这不是通用状态写入、D2D、异步或 Graph 接口，原 reset/restore 保持可用。

准备仅一次拆解原有初始 checkpoint，并保存 device 初态。固定树使用原几何/能量
kernel 冷计算初始 FK/energy；只借 frame/flags/spatial 工作区，不覆盖接受状态、
当前 FK/energy 或有效 diagnostics。预热两方向 reset 仅写 standby，不发布。
初态缓存与既有 checkpoint 都是恢复来源，不成为第二套当前物理权威状态。

每次执行一次全量 mask/owner 预检，上传一个 int32 初始 tick/选择向量，选择初态或
当前值复制到现有 candidate banks；一次显式完成后恢复所选 held control，再统一
发布 bank/epoch。q/qd、实际 drive 输出、延迟历史、tick、支撑冲量与固定树 FK/energy
一同恢复，未选值逐值保持。无热 snapshot、逐 world packet、完整状态 H2D、额外
FK/energy 或 D2H；控制仍沿既有 dirty 规则在下一步上传，不能据此声称零传输。

kernel/完成失败不发布；控制恢复失败回滚所选旧值，回滚也失败则锁定 batch、保留
owner 供 close/rebuild。构造与 close 失败沿既有可重试清理合同。跨 read/control
plan 的 alias 检查覆盖 reset 初始值及 HOST scratch，不能覆盖后续恢复来源。
两个 metadata bank 在每次执行前从当前树 owner 更新，所选 diagnostics 失效，未选
validity/status 保留；支持多个 reset plan 与 legacy reset/step 交错。

不含浮根的设备持久空间为 `4*(13*N + 25*T*J + T)` bytes：N 为 batch worlds，T/J 为固定树数
和 padded 最大关节数；无树时 T=0。每次非空 CUDA reset 上传 `4*N` bytes，一次
显式完成；正常执行不分配 device buffer、不重新编译。共同 bank 发布仍需复制
O(N + T*J) 状态/派生值；HOST 控制 mask 校验/回滚 O(控制列数)，metadata 更新 O(T)。
HOST NumPy 缓冲单列统计，另有 O(T) 的预备 metadata dict，不把它们算作峰值内存
实测。复制是现有双 bank 发布所需，尚未授予更高效的设备 selector 或局部 bank 资格。

正式 test_resident_reset_plan 覆盖 N=1/8/64 的 interpret/实际 CUDA、非零初态与
控制历史、局部未选观测、双 plan、预检、kernel/完成/控制恢复/二次回滚失败、跨
plan alias、分配/清理重试。证据见实验根 evidence/b7e-summary.json，历史跨 solver
微差不重开调查。P2PE resident 模型/完整字段映射及实际 C ABI 仍待接，B7 未完成。

## Resident endpoint body reads v1

`TyxWorldBatch.prepare_body_read(keys=None, fields=("position", "rotation"))` 冷准备
caller-buffer 观测。普通固定根树按 RobotModel moving link 顺序输出，不把 anchor
参考 frame 当成有质量/COM 的 body；静态 root/资产 frame 须由编译绑定另行表达。
FreeSphere/SpherePlaneWorld 是平移模型，单 body 的原点就是 COM，姿态恒 identity、
角速度恒零，这是该模型的物理定义，不代表完整自由刚体旋转能力。标量专用模型的
body 读取仍拒绝。普通浮根树的 root/child 读取见下文；仿射树与不同 family/body_count 的混选明确
拒绝，空选择不启动 kernel。

可选字段为 position、com_position、rotation、linear_velocity、com_velocity 和
angular_velocity。位置单位 m；原点/COM 线速度 m/s；世界角速度 rad/s；rotation
是 body-to-world 的 row-major 3×3。所有字段对应 accepted endpoint，不是求解
入口时刻。layout 给出 world_keys、body_count、body_order、family、各字段的
offset/width/shape/unit/frame/reference/sample。packed float32 行按 fields 顺序，
每字段内 body-major/component-major；公共 P2PE mat4 转换仍归其适配层。

读取借用当前 accepted FK 和 qdot：直接读取原点/COM/旋转，速度在设备上收缩祖先
轴与 qdot，不借上一步入口 Jacobian、不再运行完整 FK/forward/物理推进。使用既有
read plan 的精确 dtype/shape/stride/writable 与 alias 检查、batch lock、双 bank、
错误传播和可重试清理。read/step 不允许在同一 batch 并发交错；不同 batch 不共享锁。
HOST caller buffer 在传输失败后可能部分写入，物理状态不变；完成及恢复完成均
失败时沿原 runtime 锁定规则拒绝继续读/step，只允许清理/重建，不能强行恢复可用。

每次 CUDA read 一次显式完成、一次 values D2H，可选 ticks 再增加一次 D2H；没有
热 device 分配、全模型复制或逐 world HOST 读回。S 为所选世界、W 为 packed width，
device scratch 为 `12*S + 8*W + 4*S*W` bytes（indices/columns/ticks/output）。
普通固定树速度当前固定循环 J 次，全部 body 读取上界 O(S×B×J)，即链状树 O(S×J²)；
只读 pose 为 O(S×W)。不把没有额外 FK 宣称为线性速度读取或性能已验收。

正式 `tests/scale/test_resident_body_read.py` 用独立 FP64 homogeneous transform 与
方向导数验证多分支 R/P、旋转 frame、偏心 COM 的步后语义；沿用树测试 ATOL/RTOL，
覆盖 interpret/实际 CUDA、N=1/8/64、乱序选择、局部 reset、未选保持及执行/清理
失败。证据见实验根 `evidence/b7g-summary.json`；这仍不是 resident 实际 C ABI、
独立安装或 GUI 资格，P2PE 模型/控制/字段映射继续推进。

## Compiled resident scene IO v1

`CompiledScene.resident_io_contract_version=1` 在既有编译结果上提供
`prepare_field_read`、`prepare_body_read`、`prepare_control_write`、`prepare_reset`。
返回 plan 的 layout/read_into/write/reset/close 沿用上面的原生合同，同时检查
scene 生命周期、partial-publication 状态与资产 target position/velocity 限额。
HOST backend 明确拒绝这些 prepared 接口，原有 snapshot/set_controls/reset 保留。
各 physical family 的限制仍由原生 plan 判断，不能从方法存在推断该族读写/reset 均支持。
控制选择仍须非空；字段读取可为空。不同 scene 独立锁，同 scene 串行调用。

TYX 场景不再保存第二份当前 held control。`TyxWorldBatch(initial_controls=...)`
在构造时安装初始命令并捕获一次完整 reset baseline；因此首次输入、legacy reset
与 prepared reset 都恢复 authored 控制，局部 reset 不以过期 scene 副本覆盖未选
控制。该 snapshot 是冷构造成本，含既有按状态族的完成/D2H 和 HOST checkpoint；
热 prepared 路径不调用 snapshot、不逐 world 构造输入 packet、不重建计划。
目标限额属于不可变编译参数，预检复用 bool scratch，额外 HOST 空间由
scene_limit_workspace_bytes 报告；它不是运行期控制副本。

`scene.step(h, require_all=True)` 在 resident 单子步完成后共同接受或拒绝，拒绝
转成 known-all-false PhysicsStepFailure，保留既有时间/观测并允许修正输入重试。
默认 require_all=False 的部分发布行为不变：若部分世界推进，prepared read/write
与继续 step 被拒绝，局部 reset 不能解除 poison；成功全量 reset 后恢复准入。
这不是多子步事务，也不承诺执行器/设备故障可直接恢复。HOST 不提供该共同发布选项。
reset 成功后仅清除所选时钟与公开 contact receipt；未选世界保留。

构造采用既有 cleanup_owner 合同，close 按 LIFO 成功才移除 owner；子资源释放失败
不继续释放父 batch，完整依赖链可重试。prepared plan 借用 scene/batch，不增加
权威物理 state 或 solver。原生接触仍包含 HOST narrow phase、候选诊断读回及
receipt 组织，这些现有工作没有被 prepared IO 隐藏或删除，不声称完整设备常驻。
`tests/test_scene_resident_io.py` 与原 scene compiler 测试覆盖 N=1/8/64、实际 CUDA、
初态/控制、有限环境接触、未选 reset、限额、共同/部分发布及清理。实验记录为
`evidence/b7h-summary.json`；P2PE 的 actual DLL resident 路由仍须后续单独验收。

## Resident floating reads v1

B7k 为普通 `FloatingRobotTree` 接通上述 prepared field/body 读取，同样可由
`CompiledScene.prepare_field_read/prepare_body_read` 使用。自由根不表示六个标量
关节：field position/velocity/applied_effort 仅包含 child scalar coordinates 与
实际 motor 输出，顺序与 asset.joint_names 一致；纯 root 模型合法返回 `(S,0)`。
它不等于原 snapshot 的完整 q/v，也不包含 root 广义载荷或外加 effort。

body layout 为 `family:floating_tree`、`body_order:root_then_model_links`，第一项
是真实有质量的根部，后续按原生 moving-link 顺序。position/com_position/rotation
直接借用 active bank 的 accepted FK，COM/world angular velocity 借用同 bank 的
accepted twist。原点速度按 `v_origin=v_COM-omega×(COM-origin)` 在设备 gather；
不使用上一步入口 Jacobian，不再次 FK、forward 或产生 HOST 物理 snapshot。
静态 anchor 加回位置，旋转为 body-to-world row-major 3×3；单位沿原 body 合同。

原有冷 prepared 索引、caller dtype/shape/stride/alias、bank lifetime、单 batch
锁、失败传播和可重试释放保持。浮根新增借用的 FK/twist/anchor 纳入 alias 检查；
不同物理族即使 body_count 相同也不能混选。设备读取 O(S×W)，每次一份 values D2H，
仅请求 ticks 时增加一次 D2H；没有热设备分配、逐 world 读回或新的权威状态。
scratch 仍为 `12*S+8*W+4*S*W` 字节；这里不授予异步、D2D 或 Graph 资格。

正式测试 `tests/scale/test_resident_floating_read.py` 覆盖 interpret/实际 CUDA、
N=1/8/64、纯 root 与带滑动关节、独立自由落体与内部驱动反冲、偏心旋转 COM、
乱序选择、零 DOF、现有 public masked reset 的未选保持、CompiledScene 消费、
失败输出与关闭重试。共享 fixed/scalar 读取测试继续执行，容差不变。
本节点当时未接通浮根 prepared control/reset，后续 B7l 见下文；读取节点不等于 P2PE 自由根 C ABI，
也不等于完整 B7、独立安装或 GUI。实验根证据入口 `evidence/b7k-*`。

本批三个读取模块 96 个唯一 case 分段通过：首轮 94 通过、两项旧“浮根必须拒绝”
断言失败；更新为新增读取/混族拒绝并保留 alias 门禁后，与旋转推进追加验证共
四项复测通过。P2PE 既有 resident 实际 DLL 12 项回归通过。独立复审无阻断。
原全仓文档门禁仍有 479 项历史链接错误，与 B7h 日志逐字相同，本批不声称该门禁通过。

## Resident floating controls and reset v1

B7l 将普通浮根接入同一 prepared control/reset 公共入口。child 目标、目标速度、
feedforward 和 external_effort 仍按标量关节顺序；COM force/torque 使用
`root_then_model_links`，包含真实根 body，不伪造六个根关节或根电机。
上述单位、frame、全量预检、selected-only 回滚和关闭合同保持。

控制 plan 借用 `_held_drive`、`_held_effort`、`_held_loads` 三个现有 owner；
冷准备每字段的索引及 selected scratch，没有新增 packed 权威控制或物理状态。
每次访问验证当前 owner 属性，legacy packet 更换等布局数组后不会绑定旧地址。
旧 packet 在所有 family 发布前准备三块独占可写数组，提交只替换引用，避免发布
期间的分配失败造成部分更新；caller packet 仍不可变，prepared 热写不走该路径。写入失败回滚全部所选字段，
回滚失败仍锁定 batch。输入全等不制造 dirty；变化后的下一 step 仍上传三块 owner。

普通浮根控制 workspace 为 `18*S*W + 4*S*Ws` bytes，Ws 为需要连续打包 scratch 的
字段宽度总和；单字段覆盖完整输出时 Ws=0。正常访问 O(S*W)，没有即时 CUDA 上传、
同步或设备分配。多个字段涉及的 HOST packing 是显式成本，不称为零复制。

reset plan 冻结原有初态的根 pose/COM velocity/omega、child q/qd/drive/history、
generalized applied load 以及 FK/twist/energy，三块初始控制同样保留。初始 FK/twist/
energy 只在冷准备计算；运行时从初态或接受 bank 选择复制，保持未选世界的原值。
全局 tick/mask 仍仅上传一次 `4*N` bytes、一次完成、一次共同 bank/epoch 发布；
没有运行期 snapshot、packet、FK/energy 或整状态 H2D。diagnostics 仅所选失效。

每个浮根族的额外 device 初态为
`4*F*(13 + 7*Jp + D + 24*B + 1)` bytes，F 为该族 world 数，Jp 为 child padded
宽度（至少 1），D=最大 child 数+6，B=最大 child 数+1。reset 复杂度包含整族
candidate 复制 O(F*B)，不是仅与选择数量相关；控制恢复的 HOST scratch 由 stats
单列。所有 native 资源仍归 reset plan，辅助绑定只借用；清理失败保留 owner。

正式入口 `tests/scale/test_resident_floating_control_reset.py` 覆盖普通浮根的
N=1/8/64、interpret/真实 CUDA、乱序控制、masked/full reset、未选观测/历史保持、
旧 owner 替换及失败恢复；公共 CompiledScene 验证 authored 初始力与局部恢复。
配套共享 control/reset/scene 回归，证据保存在实验根 `evidence/b7l-*`。
此依赖节点不自动开放 P2PE 浮根 Stage，也不授予 D2D、Graph、安装或 GUI 资格。

## Resident quaternion read v1

prepared body fields 新增 `rotation_xyzw`，每 body 四个 float32，无量纲、body-to-world
旋转，采样时刻与已有 rotation 矩阵相同。translation sphere 返回身份旋转；固定/
浮根树在原读取内核按最大对角分支从当前接受 FK 转换并单位化，无 host 矩阵中转、
额外 FK 或物理推进。不保证四元数的符号跨步连续；q 与 -q 表示同一旋转。
与原有字段一样遵守选择、布局、caller buffer、失败和清理合同，零新增长期状态。
正式 `tests/scale/test_resident_orientation_read.py` 覆盖四个数值分支、纯球/固定/浮根、
推进前后及矩阵等价，配套 body/floating read 回归。证据为实验根 `evidence/b7m-*`。

## Retained contact sample identity v1

`contact_sample_identity_contract_version=1` adds read-only device field
`contact_sample_id: int64[nworld]`. This is an opaque runtime-local sample identity,
not a simulation clock, native row index, or public shape identity. Consumers must
scope their caches to the storage generation and world. The two physical banks
have separate ID arrays. A new solve/forward changes IDs only for refreshed worlds;
masked refresh retains unselected IDs and measurements; command-only copy retains
all IDs. New forward evaluations at the same time still receive a new identity.

A device serial is incremented on the canonical stream for every extraction,
including Graph replay. Discard never rewinds it, preventing candidate ID reuse.
IDs of failed candidates are not published; consumers still apply execution/error
gates and keep accepted and candidate derived caches separate. A storage replacement
invalidates all bindings and caches. Raw arrays remain read-only to consumers.
No contact sorting, physical response, CUBIN or tolerance changes are involved.

This adds `16*N+8` persistent device bytes, two constant-count kernel dispatches
per extraction, and one `8*N` D2D ID copy on the existing command-copy boundary.
There is no new synchronization or host transfer. IDs are valid for the lifetime
of a runtime with fewer than 2^63-1 extractions; they are not serialized checkpoints.
Native tests in `tests/test_physolve_execution.py` cover direct/Graph N=1/8/64,
same-time forward, masked retention, command copy and discard without ID reuse.
