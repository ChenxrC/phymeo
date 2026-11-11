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
