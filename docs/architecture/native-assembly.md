# 原生资产装配与完整初态

## B8o：按来源 body 设置滑动摩擦

`NativeAssetInstance.body_friction` 接收不可变的 `(source_body_name, sliding_mu)`
序列；值须为非负有限 FP32 可表示标量，名称唯一且为资产局部名称。装配前绑定
来源 body，对其直接形状修改 `friction[0]`，保留其它摩擦分量、condim、solref 等
接触参数；不传播到子 body 或其他实例。未知 body 或没有形状的目标拒绝。
未传入时没有新增模型修改。该输入是编译合同，不是运行期参数写入或另一套状态。
输入被冻结，后续修改 caller 列表不影响实例；资源及原生运行时 owner 不变。
成本是冷编译的来源 body 查找及指定形状遍历，没有按 world 重复解析或热路径成本。

`builders.native_assembly` 是 **native-only/physolve** 专属适配入口，不是 resident
模型，也不替代 `ArticulationWorld`。P2PE 只提供资产、placement、已解析的标量关节
输入及有限静态环境；MjSpec、资源引用和原生下标转换均留在 PHYMEO。
两个编译入口最终消费同一个 `PhySolveRuntime`，没有第二套运行状态 owner。

## 资产及模板合同 1

- B8d 增量输入 `NativeAssetInstance.urdf_xml` 接受非空 URDF 文本，优先于 path 文件；
  path 只作来源标识和相对资源基准，可不存在。`resource_roots` 为有序文件系统根，
  不替换模型文本；无这两项时沿原生文件/include 路径加载，不改变既有模型合同。
  模型、标量 drive 和 display 使用同一来源文本，`asset_sources` 的主 SHA 为该文本
  UTF-8 字节（文件路径则为原文件字节）；完整 MJB/archive 身份仍单独记录。
  资源先按 strippath/meshdir 及来源目录、显式根解析，再冻结到模板资源；不写临时
  URDF、不在 step/reset 重读源文件。当前 URI scheme 仍拒绝，不能猜 package 所在目录。
  默认丢弃的 visual 资源失败只记录 visual_diagnostics，物理模型仍可用；视觉查询明确
  拒绝，不伪造完整 visual。缺失碰撞资源始终失败；显式 discardvisual=false 仍要求
  原生模型使用的所有资源有效。P2PE 负责将自身快照/根目录配置映射到这些公共参数。
- `NativeAssetInstance` 以实例名区分同一资产的不同参与者；实例 pose 为 XYZ/XYZW。
  所有参与者 attach 到同一模型，保留跨参与者碰撞，不拆成独立仿真。
- `drive_mode=preserve` 保留原生 actuator/equality/contact/resource 引用，不接受
  joints 或 joint_defaults。`scalar_pd` 接 URDF 及下述限定 MJCF 标量关节，已有 actuator 拒绝，
  不重复增加驱动。B3e 补齐标准 dynamics 和仿射 mimic，合同见下节。
- gravity/timestep 来自 assembly；`integrator` 显式接受 `euler` / `implicitfast`，
  缺省保持 Euler。源默认 Euler 可采用 assembly 选择，非默认源积分器必须与选择
  一致，否则拒绝；其他非默认源 option 仍拒绝。此 profile 使用
  Newton 约束算法、dense Jacobian、默认原生 pyramidal 接触，不承诺与源
  自定义全局选项、resident 或旧 Newton 导出行为等价。
- 默认保留 native 自碰撞和 pair/exclude 规则。显式 self_collision=false 在冷装配
  生成实例内 body exclude，保留外部相互作用；有显式原生 pair 时拒绝冲突请求。
  B1c 另接公共 CollisionKey 显式排除计划，限定转换范围见下文。
- 静态形状包含有限球、盒、胶囊、圆柱及无限碰撞平面；有限实体 size 使用原生半尺寸，支持显式
  contact_dim=1/3/4/6，不能传入会被维度忽略的非零摩擦分量。
- 静态 `plane` 使用局部 +Z 法向，position/XYZW 决定平面位置和朝向；size 三维
  仅表达显示半宽、半高及正网格间距，前两维可同时为零表示无限显示，不能一零一正。
  有限显示尺寸不裁剪碰撞；平面归属 world，`NativeRigidShape` 明确拒绝 plane。
  接触摩擦与 condim 沿原生合同，不把恢复系数或刚度字段暗自换算成另一接触律。
  B8c 依赖切片的装配/visual 36 项通过（含 CPU MuJoCo 非零高度与显示范围外接触、
  动态 plane 拒绝）；证据为配套实验根 `evidence/b8c-plane-assembly-final.log`。
  初跑缺 FANUC 来源的一项跳过已用冻结来源配置补跑通过。独立只读复审通过。
  本切片尚未完成 P2PE add_ground 转换、GPU 或消费者 GUI 资格，不改变原载荷。
- 完整 MJB 和 VFS 资源由原编译快照持有。主文件散列只描述该文件；完整编译身份
  使用 MJB/archive 散列，不把主文件散列冒充 mesh/include 依赖闭包。
- public identity 是 `(world, scoped_name)`；body/joint/qpos/DOF/geom 下标仅对本次
  编译有效。固定融合后的来源另由原有 collision_source 提供。
- `NativeTemplate.bind(N)` 不重复导入、复制模型或编译；数据是 `[N,width]`。
  `HomogeneousNativeBinding.create_runtime` 校验合同版本并创建独立原生 owner。

URDF 的 scalar_pd 使用 `kp*(target-q)+kd*(target_velocity-v)`；初始 target 缺省时
才取初始 q。effort_limit=None 保留资产额定值，显式 0 在这个原生 profile 表示不限力。
不改变 resident/ArticulationWorld 中 0 表示零驱动的含义。applied_force 独立写
qfrc_applied，不受 actuator cap 限制。被动 damping、armature、frictionloss 分开。

### ASCII STL 原生资源编码（B8f）

原生资源冻结支持单 solid ASCII STL，在 template 自有 VFS 内编码为 binary STL，
供 MuJoCo 原生解码器消费；不改磁盘原件、不重建网格或改变面序。binary 输入按
长度/面数优先识别，包含 solid 开头的 binary header 仍逐字节保留。ASCII 顶点和
法向必须可表示为有限 float32，语法/溢出拒绝；不做补面或非法数值修复。
visual 与物理资源共用此冷入口，各自保持独立几何角色；成本 O(文件字节数)，
不增加运行期复制、设备同步或 Python 依赖。源移除后 template 仍拥有资源。
资源/视觉/装配/mimic 合同78项通过；真实消费者Stage的物理模型与视觉13实例/
11mesh编译通过。源码冷准备证据见实验根 evidence/b8f-*，不代替安装或GUI资格。

### 原生配置读回（B8g）

PhySolveRuntime.configuration_info() 提供 configuration_readback_contract_version=1：
实际CPU模板与设备option中的timestep/gravity、原生integrator、world/设备身份和
两bank容量。返回值独立，不借出可写model；pending、capture和closed时拒绝。
这是冷同步API：一次显式完成加两块option数组D2H，不能用于高频观测或宣称零同步。
不调用CPU step/forward，也不读取整批状态。execution_costs新增step_graph_substeps，
来自实际准备图的steps，未准备时为0；不改变捕获、推进或资源生命周期。
实际原生测试覆盖返回值独立、pending/capture/closed门禁；消费者配置比较和
candidate发布仍由消费方负责，导入默认policy与逐关节override不得混为同一实际值。

### 标准 dynamics 与 mimic（B3e）

B8e 的 `NativeJointInput.lower_limit/upper_limit` 是原生位置约束的有限覆盖值，
平动用 m、转动用 rad；省略保留源值，已有有界 joint 可以只覆盖一侧。无界 joint
需提供完整有限区间，lower 必须小于 upper。`joint_limits` 以 public name 保存
编译后的界限和声明速度；没有第二套可写 solver 状态。velocity_limit 是非负声明，
不增加速度裁剪，未强制执行的信息进入 import_diagnostics。mimic follower 可重申
与 URDF 相同的 effort rating，明确没有独立 actuator；改变 follower cap、独立目标
或增益仍拒绝。P2PE 的 null 解除限位不能映射成这里的 None/保留，须另行资格化。

NativeJointInput 的可选增益、被动参数和速度以 None 表示未提供；解析优先级为
逐关节显式值 → asset.joint_defaults → 源 URDF 标准 damping/friction → profile
缺省（kp=500、kd=50、armature=.01、damping=0、frictionloss=1e-5）。源中明确
friction=0 保留零；没有 dynamics 的旧模型缺省不变。D、K、mu_coulomb、mu_viscous
等非标准扩展不被 MuJoCo 3.8 解释，逐项进入 import_diagnostics，不冒充已实现。

mimic 采用 PHYMEO 已有严格 XML 提取，再在专用适配中生成 native joint equality。
关系为 `q_follower = multiplier*q_driver + offset`；转换系数包含原生 ref 修正，
支持同类型标量关节、非零有限比例、非零偏移及无环链。循环、未知引用、类型冲突
与 FP32 无法表示的系数拒绝。所有关节保留 q/DOF；只有独立根关节生成 actuator。
约束是求解器的软 equality，不做逐步强制坐标投影，不承诺每步物理 q/qvel 严格相等。
直接外加力仍可作用在任意物理 DOF，follower 不拥有独立 drive/target/effort_limit。
初态从 driver 传播到未提供的 follower；显式冲突 q/v 拒绝，不通过 stepping 修正。
标量初始 q 超出原生限位时记录 initial_position_outside_limit diagnostic，保持原值，
不 clamp 或自动选择另一姿态；消费者需要明确提供任务初态。

`target_contract_version=1` 的 `target_bindings` 给出每个公共关节到独立命令列的
multiplier/position offset；速度不含 offset。`scalar_controls` 与 runtime 目标
bank 只存独立命令，不重复维护 follower 目标。消费方可批量派生完整目标读回，
但不能把派生命令当作 measured 状态。native 原始 q/qvel 和反力仍由 Host 求解。
空间开销为冷绑定 O(J+M)，装配时拓扑解析 O(J+M)；未增加每步 CPU 物理工作。
系数语义依据 [MuJoCo 3.8 joint equality](https://mujoco.readthedocs.io/en/3.8.0/XMLreference.html#equality-joint)，
本机 3.8 URDF 导入的 mimic 补齐由本适配负责；不继承未来版本导入器的行为。

## 初态、reset 和 owner

`backends.physolve_initial.NativeInitialState` 与 MJB 散列绑定，拥有只读积分输入：
time、qpos/qvel、act、qacc_warmstart、ctrl、qfrc_applied、xfrc_applied、eq_active、
mocap_pos/quat。原生 qpos/mocap 使用 WXYZ；shape、有限 FP32 表示及单位四元数
在写入与原生准备前全量检查。它是冷初始化合同，不是 P2PE 公共状态坐标。

Runtime 可直接接受 B1a 的 `CompiledMuJoCoWorld`，不再丢失 q/v/ctrl；原生 compilation
也可配显式 initial_state。qpos0 始终是结构参考。CPU forward 只在准备时执行。
运行期仅 Host 推进；step 后不追加 CPU forward。

Runtime 为完整 reset 保存一份积分备份，准备两个 bank 的扁平借用视图与恢复内核。
mask 缓冲复用；full reset 无 mask 上传，masked reset 仅显式上传 N 个 bool。
Runtime 接管 reset owner 后在创建阶段预热所需内核签名并完成确认，首个 reset 不 JIT。
恢复只在 TYX device 内核中执行，没有按 world 的 Python 调度或 host 状态中转。
恢复后仍由 Host 整批 forward，先完成并验证，再从旧 bank 在 device 上恢复未选
`OBSERVATION_FIELDS` 的派生观测，确认复制完成才发布。旧积分输入和 solver-time
snapshot 保持不变；这不等于全局 packed contact/SensorView 的发布合同已经实现。

成本：冷编译 O(资产/关节/几何规模)，显式关闭自碰撞的 body pair 为 O(B²)；
绑定 N 不编译。两个已有候选 bank 加一份 O(N×积分宽度) reset 备份及两个 N-byte
mask；每步仍复制 11 个积分字段，验证仍有一次显式 completion 和 7 组 host 读回。
局部 reset 增加一次 stream completion；掩码内核扫描 O(N×派生观测宽度)，只写入
未选 world。没有新增 host 状态快照、派生量备份或全设备同步。借用视图及内核在创建时准备。
snapshot 另有 completion 和观测 D2H。Host 条件求解循环自身同步成本另计。
分配拦截另发现既有 codedev 前端 `smooth.transmission` 的 `moment_nnz: int32[N]`
和 `sensor_acc` 的 `weld_geom_count: int32[N,nbody]`；后者还创建 weld_geom_list。
扩展的 empty/full 跟踪也观察到 collision context 与 sensor_pos 工作数组。
本批未改 codedev；这些成本不属于新增 reset 复制，也不能
宣称完整 reset 零分配。资格化高频执行/Graph 前仍需由 solver 工作区 owner 承接。
本入口是同步 direct 验证接口，不是 Graph/异步/D2D 高频 executor 的性能资格。

## 验证与限制

正式入口：`test_native_assembly.py`（装配、资源、身份和拒绝）、
`test_physolve_initial.py`（完整初态、独立力、N=1/8/64 reset/replay）、
`test_native_template_runtime.py`（真实 FANUC，单参数独立进程）。
P2PE 正式测试清单登记 `tests.test_phymeo_stage`，验证真实 SceneModel 物化字段。

尚未完成：Stage 的自由根/动态刚体、程序化 tree、MJCF 控制和实例语义、mimic、
任意形状对排除、完整字段/传感器/render 绑定、实际 C ABI 和 GUI。原 Panda/FANUC
求解数值失败未因此关闭，不能用编译或简单滑块通过替代机器人资格。

2026-11-11 B1b 最终 wheel 安装选择集 116 通过、14 个旧机器人项未选，禁止三包导入。
另跑本页 FANUC N=1/8/64 三个独立进程，全部因推进后 qvel 全零而未通过原 motion
检查；后续 forward/reset 未到达。原始 mesh、geom pose 和初始 contact 集合的冷
保真测试通过；不通过关闭自碰撞、修改初态或放宽门限处理此失败。
全库文档检查仍为 479 个历史缺失目标，与 B0 日志 SHA256 相同。

## B1c 显式排除与观测保持

`template.with_collision_exclusions(plan)` 消费原 `collision_source.plan_exclusions`
返回的计划，重新核对 source identity、规范化内容和 fingerprint。无规则不修改模型。
只允许把声明的形状对无损编码为原生 body exclude：两端必须是保留的非 world body，
双方所有 geom 的笛卡尔积必须全部在声明中。多形状的部分排除、world、同 body、
过期/伪造计划及与显式原生 contact pair 冲突均拒绝；不重写 mask 或修改 codedev。
固定根 weldid=0 可接受，但 bodyid=0 不可；不会连带屏蔽其他 world 障碍物。

FANUC 使用显式 base_link—link_1 来源形状对规则；不按资产路径或名称自动添加。
原始无规则模型的停滞失败仍保留，不把有规则通过表述为原始资产已修好。
新模板持有规则记录和新的编译来源身份；旧模板、初态和调用方输入不被修改。

原 `MjSpec.to_zip()` 不自动拥有磁盘 mesh；装配现在先把单文件 mesh/texture/hfield
资源收入 VFS，按内容生成唯一资源键，并显式绑定 attach 后的引用，避免目录中同名
文件混用；hfield 服从原生 meshdir，路径剥离服从原生 strippath。
六面 cube texture 尚未资格化，提前明确拒绝。
无规则模型与 B1b 安装版本对比的
508 项数组/option/初态中，507 项精确相同；唯一变化是 VFS 路径变长导致的
mesh_pathadr 元数据。物理参数与候选不变。
XML 归档重新解析会舍入原生 authoring 数值，不能用于精确规则重编译。因此模板
额外持有一个私有 MjSpec 深复制（含资源），只复制它来编辑新规则；不借用公开
可变 mj_spec，也不重读源目录。它是冷编译 owner，成本随单模板和资源规模增长，
与 N 无关。已测试删除源文件、修改公开 mj_spec/mj_model 后仍使用原精确内容。
该内存快照不冒充可跨进程序列化的精度保真编译格式，MJB 仍是运行模型事实源。

规则转换预建 body→geom 索引；除原生编译外成本 O(G+展开的形状对数)，非逐 world。
新旧非排除数值数组/option 必须完全相同，排除签名恰好为旧集合加声明集合；
name-table offsets/hash slots 允许因新增 exclude 槽变化，但实体名/数量与索引
必须不变。初态全字段验证后才绑定新 MJB hash；未知编译漂移直接拒绝。

reset 验证先于观测恢复，防止旧 qacc 掩盖未选 world 的 NaN。失败保留旧发布 bank。
内部 packed contact/constraint 工作区仍是本次整批 forward 的结果，不公开为与保留
snapshot 同一时刻的接触观测；后续 contact/SensorView 接入必须补独立发布合同。
正式新增入口为 `test_native_exclusions.py`、`test_native_resources.py`、
`test_native_exclusions_runtime.py`、`test_masked_observations.py`；原无规则 FANUC
motion 用例和容差不删除、不改 xfail。

B1c 最终 wheel 安装回归 132 通过、14 个历史机器人项未选；另行独立进程验证显式
FANUC 配置 N=1/8/64，三项全部通过，包括逐 world motion、forward 不推进 q/v/time、
full/masked reset、未选 snapshot 精确保持及未选世界下一步与不中断轨迹的 1e-6 对照。
无规则 FANUC 的三个原用例仍全部失败于 motion，后续 forward/reset 未到达；这三项
不记为通过。障碍物接触保留有 CPU 接触集合和生产前端完整/压缩候选表证据，尚不
等于完整动态障碍物接触力资格。已有 Panda 求解数值问题仍未关闭。

来源记录为实验根 `sources.json`，安装/过程/负例日志为 `evidence/b1c-*`，最终矩阵
与制品散列为 `b1c-final-matrix-progress.json` 和 `b1c-summary.json`。禁止三包安装和
导入的检查通过；原生 Host 与六 CUBIN 散列保持冻结值。数值测试文件/容差未变。
独立只读复审及主执行者完整 diff 自审完成；初态身份、资源目录/strippath 和所有权
问题修正后才构建最终 wheel。完整 C ABI、Graph、D2D、SensorView 和 GUI 未执行。

masked reset 功能 8 项通过（自由落体/接触 N=1/8/64、错误抑制和失败后重试）；
新增相对成本用例单独复测 1 项通过。N=8 自由根的 forward 共观察到 11 次公共
zeros/full/empty 调用，包含零尺寸数组；partial/full reset 与该记录逐项一致，
新增 restore/preserve 内禁用这些工厂及 clone 后仍通过。它不是所有底层分配的统计。
局部 reset 完成确认 2 次、full reset 1 次；借用绑定身份不变。三份新增计数仪器
中间失败日志保留，错误依次为全局零分配假设、单项分配假设及来源全集假设；修正
仪器不改生产包、原功能断言或容差。既有分配仍阻止稳态性能资格，不以测试通过豁免。

## B1d 原生前端工作区

PhySolve 专用清单与载荷版本门禁锁定 frontend 0.0.0a6。Runtime 创建每个 Data
bank 后、创建 Host handle 前调用其公共 `prepare_workspace`，资源仍由该 bank
拥有；不增加 PHYMEO 状态镜像、全局缓存或跨 runtime 锁。准备失败进入原 candidate
清理；关闭失败仍保留原 bank/Host owner 以便重试。

前端持久化三个 collision 临时数组和一个 transmission 计数数组，空 sensor
调度只在没有 callback/energy/RNE 副作用时跳过。未准备的原生 Data 保留旧路径；
Graph、SAP、非零 sensor、并发和完整 allocator 行为不从本切片自动继承资格。
每 bank 额外持久逻辑容量为 `20*naconmax + 4*nworld` 字节，替代对应逐调用分配；
不包含 allocator 对齐和原 Host workspace。reset 的两次/一次完成边界不变。

正式测试入口新增 `tests/test_physolve_workspace.py`：已安装旧 a5 的冻结短轨迹与
候选 a6 对照，要求 `PHYMEO_WORKSPACE_BASELINE` 显式指向参考目录；无参考时 skip，
不计通过。覆盖自由落体/接触/显式规则 FANUC N=1/8/64，原容差 `1e-6, rtol=0`。
另测工厂禁用下的连续 step/forward/reset、接触数减少后恢复及独立 runtime/bank。
原 `test_masked_observations.py` 仅加强成本断言，功能/数值断言不变。

冻结旧安装版时，FANUC 在新增连续计时窗口复现 CUDA 719，原日志保留于实验根
`evidence/b1d-baseline.log`。短闭环通过不能关闭长轨迹失败，本批不改其物理配置或
载荷来消除错误。Panda 和原无规则 FANUC 的历史失败仍未关闭；完整 B1/B2、C ABI、
Graph、传感器、GUI 资格均未完成。来源与最终实测索引为 `evidence/b1d-summary.json`。

B1d 最终安装专项 23 项通过，扩大回归 132 通过、14 个历史机器人项未选，P2PE
Stage 消费 8 项及10个子测试通过。N=8 自由根的原生 forward 公共数组工厂调用由
B1c 的11次降为0，full/masked reset 保持0；这不是全路径底层 allocator 统计。
原功能容差不变、Host及六 CUBIN 哈希不变，源码安装逐文件一致；pip check 通过。
独立两轮只读复审无剩余阻断，GPU实测由主执行者完成。依赖先提交 codedev
`4d462fa1`，TYX `7272d3d` 未改；原环境保留 a5，新安装仅在 `env-b1d`。

## B1e 长轨迹故障分离

原 FANUC scalar_pd 配置保持 kp=500、kd=50、armature=0.01、Euler、dt=0.002 s、
qfrc_applied=0.03，不修改初态、容差或接触规则。速度在出现新接触前已交替放大；
同一 MJB 的 CPU MuJoCo 也发散，在第10步报告 BADQACC 并自动 reset，不能当作
CPU 稳定通过。局部 CPU mj_step 有限差分 Jacobian 的谱半径约8.769888（两个扰动
尺度一致），冻结质量/驱动的离散分析约8.769903。这是原步长与驱动配置的局部
不稳定证据，不是全局非线性稳定性判定，也不授权自动更改消费者的驱动参数。

CUDA 719 是另一项已定位问题：仅把故障前有限 qpos 交给原生 collision，即可在
没有 Host、积分或 CUBIN 求解的进程复现。Compute Sanitizer 定位为 ccd kernel 的
设备断言；逐标记诊断确认 TYX 漏记 tuple 解包中的局部变量赋值。修复属于 TYX
共享编译器，PHYMEO/codedev 生产代码和六 CUBIN 均不因此改变。

正式新增 `tests/test_physolve_collision.py`，覆盖 N=1/8/64 的冻结姿态、8个指定
接触对及完整接触几何，与官方 Warp 的冻结参考按原有绝对容差1e-6、rtol=0比较。
先前只修断言时，pos/frame 仍分别偏差约0.433 m/1.93；独立检查表明第一 witness
向量丢失。TYX 的条件局部向量检查曾把分量写入包成对返回副本的写入，导致原
`x1` 保持零。第二个修复保留原 lvalue 与索引求值顺序；无需改变 EPA/GJK 算法。
它独立于原轨迹稳定性门禁，不替换或放宽现有机器人测试。
测试使用安装版依赖；源码树仅提供 pytest 入口和资产夹具，不注入生产源码路径。
原无规则 FANUC、Panda 数值与完整 B1/B2 仍未关闭。最终证据及版本身份见实验根
`evidence/b1e-summary.json`；可复现入口与诊断脚本散列一并归档。

最终 wheel 的冻结碰撞及 MJB 身份门禁 N=1/8/64 全部通过；与官方 Warp 对照，
geom/dist 精确一致，pos 最大差2.98e-8 m，frame 最大差3.37e-7。Compute Sanitizer
报告0错误。原 FANUC 短闭环三个规模通过，工作区/reset 23项通过；25步长轨迹三个
规模均执行完成且不再719，但速度发散，明确不计物理通过。没有用缩短窗口、改参数
或改容差关闭原失败。下一步须独立复审驱动/积分合同与稳定性，再继续完整 B1/B2。
TYX 本地修复为 `e70781aa`；本仓生产源码未改变，候选仍使用 a6 前端及冻结 Host。
扩大安装回归132通过、14个历史机器人项未选；后者不计通过。原有479个文档缺失
引用的输出散列与B1d相同，继续记失败。完整diff自审与独立只读复审已完成。

## B1f 显式积分器与驱动合同

原生 Host 已具有 implicitfast 分支及对应工作区，但 assembly 曾固定 Euler、Runtime
也只准入 Euler。旧 P2PE Newton 路径默认 implicitfast；只承接相同 kp/kd 并不代表
承接了相同离散系统。本批增加显式编译选项，不自动改变任何已冻结的 Euler 用例。
`compile_native_template(..., integrator="implicitfast")` 生成新的 MJB 身份，原模板不变；
切换必须重新编译及创建 model、双 bank、Host，不能在运行期只改一个 enum。
源默认 Euler 允许采用装配选择；非默认源选项须匹配选择，冲突明确拒绝。
RK4 和 implicit 仍未资格化并被拒绝。这里不新增 resident 的全局积分器合同。

新正式入口 `tests/test_physolve_integrator.py` 在独立 `env-b1f` 安装中7项通过：
N=1/8/64 的一自由度驱动使用独立离散递推（M=.01、kp=500、kd=50、dt=.002 s），
100步逐步比较 q/v；FANUC 显式 implicitfast 配置保持原驱动、步长、初态与显式
base_link/link_1 规则，250步逐步与同模型 CPU MuJoCo 对照，绝对容差1e-6、rtol=0。
CPU 无 warning，最终时间 .5 s；forward 不改变 q/v/time，masked reset 未选世界
全部 snapshot 字段精确保持，full reset 后250步重放通过。该时间窗口不是长时稳定性证明。

运行时所有权、11字段候选复制和显式完成/验证读回边界不变。implicitfast 使用既有
Host 隐式速度求解工作区和计算，dense 矩阵容量随 N×nv² 增长；本批未测吞吐或峰值
显存，不声称与 Euler 同成本。没有新增 P2PE/PHYMEO 逐步封装、全设备同步或状态镜像。
原 Euler 高刚度失败及无规则 FANUC 失败保留；本批不授予 Graph、接触力或完整 B2
资格。来源、测试命令、失败与制品散列集中于实验根 `evidence/b1f-summary.json`。

Panda 原接触点几何 CPU/CUDA 两项与 N=1 forward/reset 本批复测通过；dual-bank
原生对照仍失败于初始 qacc（最大差1.5258789e-5 > 原1e-6），未改原测试或容差。
N=8/64 的 Panda 推进矩阵未在本批执行，不以单项通过代替完整机器人验收。

最终安装扩大回归141通过、14个历史机器人项未选（其中本批独立复测结果如上，
不重复计为通过）；工作区/reset 24通过，新增 implicitfast Host 准备/清理失败与
关闭重试仍保留 owner。冻结碰撞及模型身份3通过。原 Euler 用例、接触几何参考、
1e-6门限未改。PHYMEO 文档仍有479个历史缺失引用，日志散列与B1e一致，保持失败。
完整diff自审及独立代理两轮只读复审完成，实际GPU测试由主执行者执行。
最终wheel只安装到 `env-b1f`，pip check通过；原环境与原工作目录不变，不推送或合并。

后续 B1g 保留上述原测试，在独立安装中继续 Panda 数值与 N=1/8/64 复核；
输入排查及尚未关闭的停止条件数值阻断见
[PhySolve 运行时诊断](../development/physolve-runtime.md#b1g-panda-初始加速度诊断)。
该节点未更改模型、公共装配合同或生产实现，不继承为机器人资格通过。

## B1h Panda 显式连接处规则

原始 URDF 不携带 `load_panda()` 在 Python 中声明的父子刚体排除。后者还合并
capsule 代理、转换刚体表达，不能把它与完整 native 路线整包视为等价。本批只
审查并显式选择底座—第一节这一个连接处，不自动复制所有相邻过滤政策。

原 URDF 的 link1 柱体/球心与 joint1 的 z 轴同轴，其几何集合不随该关节转角
变化。独立解析计算和 CPU 全范围九点扫描一致：底座两个端球与第一节柱体的
穿透为0.06/0.03 m，与下端球的穿透约0.035147186/0.011833462 m；法向关节
Jacobian 为零（检查1e-12）。这类连接处代理重叠不是关节可运动消除的法向接触。
该依据不等于制造商真实几何认证，也不授权过滤其他邻接或非邻接接触。

使用既有公共入口显式配置，`panda_path` 指向调用方确认的资产：

```python
from itertools import product
from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template

template = compile_native_template(
    (NativeAssetInstance("panda", panda_path),),
    timestep=0.002, gravity=(0, 0, -9.81),
)
source = template.compilation.collision_source
groups = [
    [g for g in source.geoms if g.body_name == name]
    for name in ("panda::panda_link0_sc", "panda::panda_link1_sc")
]
if [len(g) for g in groups] != [3, 3]:
    raise ValueError("review this asset's source identities before applying the rule")
plan = source.plan_exclusions(
    [(a.key, b.key) for a, b in product(*groups)], source_identity=source.identity,
)
configured = template.with_collision_exclusions(plan)
```

保留静态 body 的 native 装配中，形状属于 `_sc` 固定子 body；不能猜它们直接属于
`panda_link0/1`，也不能把 native geom 编号当公共身份。上述9对声明完整覆盖两端
各3个形状，现有 exact lowering 可原样编码为一个 body exclude。没有添加新的
碰撞表机制、特殊机器人分支或热路径解析；codedev、TYX、六CUBIN不需改动。

CPU审计确认只有指定候选对消失，非排除数值数组、option、初态完全相同。link5
与hand、两个finger pad、外部obstacle与link1的接触仍存在；零初态的joint4仍超出
其上限-0.0698 rad，限位行保留。此次未更改初态、驱动、积分器、摩擦或容差。
无规则模板仍没有新增exclude；B1g原始无规则数值失败不因新配置而关闭。

正式入口为 `tests/test_panda_collision_model.py` 和
`tests/test_panda_collision_runtime.py`，由现有pytest清单收集。CUDA矩阵的最终结果
与审查记录见实验根 `evidence/b1h-summary.json`；N=1探针的一致性不代替N=8/64资格。
同载荷单bank对照只验证原生运行组织，没有据此声明CPU数值一致、接触力完整物理
验收、长期稳定、C ABI、Graph、传感器或GUI通过。

复审后正式矩阵为4通过、2失败（模型3项通过，CUDA N=1通过，N=8/64失败），
无跳过。原 `atol=1e-6, rtol=0` 保持；reset核心状态和轨迹replay保持逐位要求。

| N | 初态qacc差 | 六步中最大qacc差 | forward qacc差 | reset及replay |
| --- | --- | --- | --- | --- |
| 1 | 通过 | 通过 | 通过 | 全部通过 |
| 8 | 通过 | 9.1552734e-5 | 7.6293945e-5 | selected qacc失败；精确replay失败 |
| 64 | 6.1035156e-5 | 6.7138672e-4 | 1.7852783e-3 | selected/full qacc失败；精确replay失败 |

三种规模的reset核心 `qpos/qvel/time/qfrc_applied/ctrl`、未选世界全部snapshot字段
以及forward自身不改变 `qpos/qvel/time` 的检查通过。N=8/64另有约束力差异，N=64
第六步qvel差1.4305115e-6。完整失败字段见原始日志；发生早期数值分歧后的阶段
只用于诊断，不能解释为独立合格轨迹。首轮N=8/64失败亦完整保留，不择优取值。
这些结果说明规则转换通过，但未解决完整数值资格，B1g无规则失败仍然成立。

## B3n：动态参与者与原生坐标绑定

`compile_native_template` 接受独立 `rigid_shapes`，元素为 `NativeRigidShape`。
每个 solid primitive 生成一个真实 FREE body；显式正质量优先，否则使用正密度，
惯量由原生几何编译计算。尺寸沿用 NativeStaticShape，box 为半尺寸；静态几何
不接受动态类型。资产、静态和动态参与者共享一次同 world 编译及碰撞来源映射。
空资产列表可表达纯程序化模型、纯环境或空模板；不增加隐式机器人或人工质量。

`NativeAssetInstance(floating=True)` 目前仅接受 scalar_pd URDF：在导入/标量驱动
准备完成后，为唯一且无既有关节的根添加 FREE joint，再施加实例 pose。缺失合法
惯量仍由编译器拒绝；不把无质量根偷偷补成动态物体。默认 floating=False 行为保持。

编译结果增加 `coordinate_contract_version=1` 和与完整 `joint_bindings` 对齐的
`joint_dimensions`，按实际原生 q/DOF 地址记录宽度。公共 `scalar_joint_bindings`
仅取 (1,1)，忽略 (7,6) FREE 根，其他多 DOF 类型明确拒绝。原完整映射保留，
不将 FREE 根伪装为标量，也不改变内部 qpos/qvel 顺序。P2PE 的自由根通过 body
字段公开；PHYMEO 本身不依赖该消费方的 public joint 列表约定。

本批没有新运行期 bank、全模型副本、逐 world 调用或求解算法；新增身份/宽度计算
只在冷编译发生。原生数组生命周期与现有 storage/执行合同不变。实际测试、载荷
散列和独立审查结果见实验根 `evidence/b3n-summary.json`；body 写入消费仍另行接入。

冷编译/排除/mimic 合同 36 项通过，包括两浮根资产混合刚体、非单位旋转 placement
与原生地址偏移；实际 DLL 的四个新增用例覆盖 N=1/8/64 的落体、接触、浮根标量
host/设备访问及空场景。独立只读复审无阻断。全文档检查仍为 479 条历史缺失链接，
输出散列不变；没有重新构建或安装 wheel，不据源码消费结果宣称独立安装资格。

## B8h：资产来源与编译身份

`NativeTemplate.asset_identity_contract_version=1` 对应只读 `asset_identities`。
每个 URDF 实例记录 instance、format、原始 robot model_name，以及 bodies 中的
`(source_link_name, compiled_body_name, native_body_index)`。原始斜杠原样保存；
实例改名不改来源名称。native 下标属于本次编译的 world-local 空间，不是稳定公共 ID。
来源取自已冻结的文件或内存快照，绑定取自本次实际编译结果；查询不重新读资产。
固定 body 当前不折叠，重新编译后必须消费新绑定，不能跨 generation 缓存旧下标。

这只增加 O(资产 link 数) 的冷身份记录，不增加物理状态、逐 world 构建、设备复制或
同步。无 URDF 的程序化模板保持空身份集；非 URDF 资产不伪造 URDF 身份。
装配、visual、显式排除和 mimic 共74项通过；新增覆盖两个重命名实例、前导/内部
斜杠、固定根、删除源文件后绑定64 world。安装及消费者结论由 P2PE 单独记录。

## B8i：原生来源 frame 合同

`compile_native_template(capture_frames=True)` 显式导出 `frame_metadata`；缺省为
None，不扩大已有非URDF路径的要求。已请求且为空的模型返回版本1的空合同。
首版覆盖保留body的URDF fixed/revolute/continuous/prismatic逻辑关节；continuous
按原生revolute描述。每行保留instance/source_name，与原生控制列数量独立。
端点引用scoped body名称；anchor为body原点局部的米与xyzw旋转，axis为关节局部
单位向量。body COM是本次编译参考值，不能冒充后续参数修改或逐帧状态。

固定关节从实际编译的父子body关系导出；标量关节同时验证原生类型和唯一child
绑定。parent anchor由body相对姿态与joint局部位置组合，child anchor为joint局部
位置及单位旋转。不执行额外forward/FK，不复制设备state，不引入P2PE字段定义。
此合同不单独资格化reaction wrench、MJCF来源frame或resident折叠frame。
冷成本O(B+J)，结果为不可变tuple/dataclass，随template生命周期和重编译失效。

装配、visual、排除与mimic共78项通过（2.00s），覆盖旋转轴、非零COM、固定关节、
多个实例和文件删除后的绑定。首轮未设置资产来源有5项skip，补齐显式来源后全过。
真实消费者Stage冷编译得到14个body frame和13个逻辑关节；不将冷编译当作GUI通过。

`native_resources.read_urdf_joint_catalog(asset)` 提供源关节名称、类型和上下限的
不可变 tuple，URDF snapshot 优先于 locator；不加载 mesh、不编译模型或触发
设备工作。continuous 映射为无限范围 revolute，固定关节仍保留来源身份。
非 URDF、空名或重复名拒绝。用于消费者冷 authoring 预检，不能代替编译绑定
或依据来源顺序猜原生关节下标。正式 native assembly 测试覆盖删除来源文件后
使用 snapshot、缺失 mesh 时仍可读取目录。


## RC7 后 MJCF 标量承接

`NativeTemplate.mjcf_scalar_contract_version=1` 声明本增量；未知版本由消费方拒绝。
`scalar_pd` 增量接受 compiler angle=radian 的 MJCF hinge/slide；源 actuator、
keyframe、非标量关节、高阶 damping、springdamper 自动推导及 floating 增量根
明确拒绝。未覆盖的 assembly 全局 option 仍拒绝。缺省 armature/damping/frictionloss
保留来源值；显式覆盖遵循公共 NativeJointInput，target 与 q、命令力与 actuator
输出继续分开。URDF 原默认与 mimic 不变，resident 不消费 MjModel。

NativeAssetIdentity.format 真实标记 mjcf；同一 body 的多个 shape 各有来源名称。
read_native_joint_catalog 在冷阶段从原生编译结果读取标量 SI 限位（URDF 仍沿
无 mesh 的源码目录）。frame 导出允许 world parent、单 moving body 一个标量关节；
q 为绝对坐标，slide anchor 减 R axis ref，hinge parent rotation 乘 Rot(-ref)。
编译期补偿不改初态或求解模型，不增加热循环 FK。固定 body 不生成虚构 joint。

正式 test_native_assembly 覆盖完整初态/控制、被动参数保留、多 shape、来源身份、
非零 ref/anchor/旋转 placement 的独立 FK 和拒绝边界；P2PE 配套实际 DLL 负责
N=1/8/64 direct/Graph 字段与状态/action/reset 验收。完整证据在实验根
`evidence/model-summary.json`。不声称完整 MJCF、resident、安装或 GUI 资格；
原生 Host、六 CUBIN 和已有数值门槛保持。本批改动均在编译/绑定阶段。
