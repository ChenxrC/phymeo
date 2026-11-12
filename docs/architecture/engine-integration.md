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
resident 公共入口的部分 world 发布行为同样必须补齐共同发布/恢复边界后才授予外部
事务契约资格。不能由 P2PE 每步整批 host 快照补偿底层合同缺失。

每个缓存/备份/bank 登记 owner、容量、用途、更新和失效条件。编译时冻结索引、缓冲
和绑定；热循环不重复 FK/forward/接触响应，不逐 world 重新编译或跨层调度。
Graph 捕获资源完整保活，同一 stream 包装复用；捕获失败不 fallback direct。

公共集成合同、包版本、源提交、TYX 互操作、原生 ABI 和载荷身份分别管理。
字段借用合同首版为 1；完整跨 solver 合同仍待后续资格，不能推断任意版本组合已兼容。字段语义、采样
时刻和存储有效期变化要显式迁移；升级使用实际旧包和新候选跑同一测试套件。

### 字段借用与候选执行 v1

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
