# 完整固定根 R/P 树驻留：研究案例册

状态：**DOCS_RESEARCH_ONLY / PROPOSAL / UNEXECUTED**。2026-11-07，仅研究与手推输入；本册不授权产品实现，不报告新测试通过或性能收益。作者树为 `8439964`，共享源码依据父 `bbfe7f2`。阅读期间父到 `9326aa1` 的差分只涉及 F2 文档、例程和测试，本文引用的共享源码没有变化。实施顺序见[配套计划](../superpowers/plans/2026-11-07-resident-tree-broad-phase.md)。

## 1. 已有实现与真正缺口

| 已核本地源码 | 可复用内容 | 不能据此宣称的能力 |
| --- | --- | --- |
| `src/phymeo/core/robot.py:20–85,103–143` | 每活动组一条 R/P、父先于子、真实质量/COM/正主惯量；`RobotInputs` 是广义力及世界 COM 力/矩，排除重力 | 非对角惯量不能只保特征值；mimic 的物理行和独立坐标不能混同 |
| `src/phymeo/backends/_tyx_robot_kernels.py:9–269` | 真实设备 FK、祖先 J、完整 `M=sum(m Jv.T Jv + Jw.T Iworld Jw)`、解析速度偏置/重力/陀螺矩及载荷投影 | 当前 kernel 入口并非批量驻留树 |
| `src/phymeo/backends/tyx_robot_math.py:14–187`、`tyx_math.py:76–130,182–210`、`tyx_participants.py:124–150` | 已有 TYX 物理算法与 Cholesky；归一化方程残差门 `2e-5` | `call` 逐次上传、分配、读回；参与者最终构造 HOST `RobotState`，不能把它包装成 accepted-device tree |
| `src/phymeo/control/bounded_pd.py:61–183`、`solvers/driven_robot.py:46–97,175–218` | 全质量矩阵隐式末速度 PD，投影坐标迭代，独立电机帽；只延迟 target/rate/feedforward | 当前并非设备 active-set；独立外载/重力/偏置不得延迟或重复施加 |
| `src/phymeo/builders/_urdf_fixed.py:54–129`、`_urdf_frames.py:20–58` | 固定工具聚合质量/COM/完整惯量，主轴旋入关节帧；`sites` 保持原 authored link 到真实组的变换 | collider 原 link 序号不是可靠的折叠组号；原材料点不是主轴组局部点 |
| `src/phymeo/backends/tyx_world_batch.py:240–279,294–379,456–534` | 同锁借用 runtime、双缓冲、tick/epoch、局部 restore/reset、统一发布与 close | 新树不能拥有另一套 runtime/独立发布时钟 |
| `src/phymeo/backends/_world_step_layout.py:16–18,128–218`、`_world_kinematics.py:9–49`、`_world_readback_plan.py:37–67` | 旧 scalar/sphere 12 列状态与 6 列读数、一次索引上传 | 多关节不能塞进单轴 width=1，或静默截成 6 列 |
| `src/phymeo/backends/_resident_collision_plan.py:30–49,213–347,445–560` | accepted borrow、现有 device bounds→BVH、注册/版本/完整输出与失败事务 | 当前每 world 一个动态 body 的限制须按真实 link 身份扩展；不能只删校验 |

本机 Newton 参考环境保持父目录 `.venv-newton-1.7.0`，Newton 1.7.0 / Warp 1.17.0，不引入依赖。已读该安装的 `_src/sim/state.py:119–165`、`_src/sim/articulation.py:539–661`、`_src/sim/control.py:31–59`、`_src/solvers/featherstone/solver_featherstone.py:60–164,833–1040,1077–1194` 及 `featherstone/kernels.py:549–574,603–612`：

- `State.joint_q/joint_qd` 是设备广义坐标；`eval_fk` 从它们写原生 `body_q/body_qd`，公开 twist 为世界 COM 线速度在前、世界角速度在后。
- `body_f` 为 COM 处世界力在前、世界力矩在后；`control.joint_f` 为 N 或 N·m。Featherstone 同步建立 J、空间惯量、广义 H、Cholesky 并半隐式积分，之后重建原生 FK。其私有 `M` 是空间块矩阵，广义质量是 `H`，不可错列比较。
- 原生 R/P drive 在入步 `q/qd` 上取力；源码明确不支持 `joint_effort_limit`。它不等于本仓隐式限幅 PD。主要动力学对照先关原生 PD、armature、阻尼/限位弹簧与接触，显式给同一预声明广义载荷/COM wrench；PD 对照另列数学方程，不冒“同 gain 即同控制律”。

## 2. 推荐支持域与容量

推荐首域 **每 world 一棵普通固定根树，1≤N≤16 个真实活动 R/P 行**；各 world 的 N、parents、轴、惯量和安装可不同。Nmax 取本 batch 实际最大值，16 是显式资源帽，17 在分配前拒绝。父先于子；分支及多个挂在同一固定根上的根活动行合法。一个 world 内的全部行同一步耦合求解，不拆成独立 rotor worlds。

8 不足覆盖普通展开的 7 臂+2 指 Panda，也不足为未来 mimic 留真实物理行预算。选 16 是覆盖常用 9 行树并留通用余量的资源选择，不是性能结论。首片明确拒绝 mimic/floating/闭环/球铰/刚性约束/`program.limits`；已有 HOST/其他入口的这些能力保留，不改成全项目不支持。固定工具通过原 `load_urdf` 折叠后的真实质量、COM、主轴与 sites 进入，不新增假 DOF。

旋转轴可不沿主惯量轴；关节静态朝向、固定根安装和 COM 偏心均保留。`RobotModel` 的惯量已在其局部主轴中，设备使用 `R diag(I) R.T`。原 authored 非对角惯量由已有折叠加载流程处理，设备不再自行特征分解、也不能丢掉主轴朝向。安装直接来自 model 的固定根关节帧，避免再叠一个未经定义的 pose。

每个 R 的实际存储 q 域沿 C3 为 `[-4096,4096] rad`，不折回一周；P 首片也声明有限存储域 `[-4096,4096] m`，这是数值域而非物理行程/限位。创建、restore、候选接受同检。非有限/FTZ 不可表示值沿原 FP32 合同拒绝。目标只是命令，不能当状态写入或静默裁 q。delay 先为 0/1 个物理子步；更长队列是后续扩展。

### 核心持久缓冲预算（方案，不是实测分配）

以下为一棵树的 FP32 数，按 Nmax padded；静态模型、读回缓冲、BVH、几何和运行时不在主表内。

| 数据 | FP32 数 |
| --- | ---: |
| 双 q/v、drive/applied、三历史列 | `14N` |
| 双接受/候选 FK：origin3 + R9 + axis3 + COM3 | `36N` |
| 持有完整命令 | `10N` |
| COM 线/角 J | `6N²` |
| M、Cholesky、mobility、QP 与其可选因子 | 最多 `5N²` |
| 递推、解向量、残差/动能 scratch 预留 | `32N` |
| 合计 | `11N²+92N` |

N=8/9/16 分别为 1440/1719/4288 个 float，5760/6876/17152 bytes；W32 全 N16 为 548864 bytes（536 KiB）。若 restore staging 另留 `17N`，W32 N16 再加 34816 bytes。静态模型约 `25N+3` float 加 parents/kinds 等整数；几何按实际总顶点数另外计。产品以真实分配统计为准，不把此预算写成实测 RSS/显存。

从 8 到 16，密集 M/J 累加与 Cholesky 的主要立方循环约增 8 倍，QP 每 sweep 约增 4 倍。首实现建议一个 world lane 顺拓扑递推、循环使用实际 n，不按某资产展开代码；局部 3D 轴可静态展开，树行不要全展开成 N=16 模板巨核。真实编译时间、launch、占用和 W32 墙钟必须后测；无默认加速承诺。

## 3. 推荐公共接口与状态表示

拟沿已有正常入口，名称待父批准：

```python
spec = FixedRobotTree(program, program.initial_state(robot_state),
                      control_max_iterations=512)
batch = TyxWorldBatch({world_key: spec, ...}, mode="cuda")
batch.set_controls({world_key: program.inputs(target, rate,
                    feedforward=ff, loads=RobotInputs(effort, force, torque))})
accepted = batch.step(h)
plan = batch.prepare_collision(collision_model, bindings)
result = batch.collision(plan)
```

`program` 复用 `DrivenRobotProgram`，`state` 是其完整 `SceneState`；不另造目标/驱动语义。每行 command10 为 `[target, rate, feedforward, generalized, Fx,Fy,Fz,Tx,Ty,Tz]`；后三维矩以该活动组 COM 为参考，外载不含重力。每列单位按 R/P 标注；只有前三列进入延迟。zero gains 允许非零 feedforward，外载不经过 actuator cap。

推荐在原 batch 下建立有类型的 tree storage 扩展，仍借用唯一 runtime、锁、active、ticks、status、epoch、资源栈。旧 scalar 的 12 列状态与纯旧 batch 4/9 或 10/16 command/property 宽度保持。树不拓宽所有旧 world：`tree_slots[T]`、`world_to_tree[W]`、`count[T]`、`parents/kinds[T,Nmax]`；树命令 `[T,Nmax,10]`，双树状态 `[T,Nmax,7]`、双 FK `[T,Nmax,18]`，scratch 按上表。Padding 必须初始化，所有读写受实际 n 限制；独立 world slot 不能用 CollisionKey.body 代替。

新 `TreeWorldCheckpoint` 应保存 owner/topology、q/v、applied/drive、完整延迟队列、held command10、tick、关节 kinds/顺序；数组必须 owned/不可变。恢复源 model/gains/parents/静态 frames 不符即拒绝，不能只比较 N。新 `TreeKinematics` 读数使用 offsets + joint kinds + q/qdot 明确混合单位；旧 `WorldKinematics` 不能静默截树成一行。建议 `tree_kinematics(keys)` 与新的 prepared-readback kind；旧 scalar reader 对选中树明确报不支持，纯旧调用不变。新 summary 的每轴单位明确，动能从当前接受 q/v 的 COM/角速度求和，不能复用上一入步 M 或 props[8] 单惯量。

`AcceptedCollisionState` 增加可选只借用的 tree view：共享 owner/generation/epoch/active，实际 tree-slot 表、当前 FK device handle、当前 ticks/counts 和 owned topology/source token。没有第二个“候选接受”计数器，也不传 HOST 的当前 pose。每种旧 reader/kernel 对 kind5/6（拟 delay0/1）必须显式路由或拒绝，不能走旧 else-slider 或 scalar summary。

## 4. 同一步设备力学与事务

1. 同锁取 accepted q−/v− 与当前完整控制；从相同 q− 建 FK、祖先 J、完整 M 和解析 b。b 包括重力、离心/Coriolis 及 `ω×Iω`；COM forces/torques 通过相同 J 投影。静态模型可初次 HOST 编译，当前 q、FK、M 或轴投影不得 HOST 上传。
2. 令 `e = generalized + Σ(Jv.T F + Jw.T T)`，`b_eff=b−e`，`u_free=v−h M⁻¹ b_eff`，`a=kp(target−q)+kd rate+ff`，`d=h kp+kd`。求 `tau=clip(a−d u, ±cap)`，`u=u_free+h M⁻¹ tau`。d=0 的行先固定为 clip(a)，其余保留完整 cross mobility。等价 SPD box-QP 的矩阵为 `diag(1/d)+h M⁻¹`，RHS 需包含固定行载荷。
3. 复用已有数学方程和投影坐标迭代，不将其改称 active-set。Cholesky、正定性、对称性、原 M 方程、全坐标 clipping 残差均在设备验证；最多 512 sweeps，可显式缩小预算以测试未收敛。不能逐轴解标量 PD，不能 HOST NumPy 接管失败，也不能只检测末改一行。
4. 以求得的 held tau 和同一 e/b/M 做一次真实半隐式推进 `v+=v+h M⁻¹(tau+e−b)`，`q+=q+h v+`。可以复用因子/解，仍核对原方程；不是第二次施力。独立保存 drive、外载、b 及其入步时刻。
5. 从 q+ 在设备重新计算候选 FK，并计算本次需要的末态速度/动能；检 q 域/有限性/数值残差。入步 J/M/b scratch 的时间戳是 q−，不可把它伪作 q+ 几何或末态能量。候选失败时整棵树（所有行、FK、历史、applied、tick）均回到旧缓冲值。
6. 与旧 world passes 同一次 sync/status 后统一 swap。数值拒绝只令该 world false，其余 world 可按既有合同接受；完成批事务的全局 epoch 沿旧规则变化，不能拿 world tick 代 epoch。设备/同步异常无新发布；重试从旧接受态和未消费队列重新算。restore 同 tick 仍使旧 collision result 失效，局部 reset 不动旁观树；候选 FK/历史与 q 一起恢复/发布。

工程残差沿原 TYX 质量方程 `2e-5`，并对每坐标重新计算完整原始律 `expected_i=clip(kp_i*(target_i-q_i-h*u_i)+kd_i*(rate_i-u_i)+ff_i, -cap_i, cap_i)`；要求 `abs(tau_i-expected_i) <= 5e-5*max(1,abs(tau_i),abs(expected_i))`，按该轴 N 或 N·m 计。不能用 inactive cap、相消前 a 或 d*u 放宽小实际驱动的门；原始式和候选配置的舍入后读数都要一致。任何中间非有限拒绝。精度门用于诚实拒绝而非改质量；未收敛保原输入和失败，禁止 HOST rescue、强制对角化或特征值截断。

## 5. 多 link 的材料、设备 FK 与 bounds

拟 `ResidentTreeBinding.from_asset(asset, source_key, *, key, world)`：返回原有限形状、真实 physical group、authored-to-group 固定变换及完整 model/source token。普通资产用 collider owner；折叠资产必须从 `sites[collider.link_name].link_index` 取 owner，并将原 `local_vertices` 经 site 变为真实组局部材料。原顶点 ID、shape 内容/revision、原局部 shape pose 保持；不能读取 `collision_motion(q_current)` 再上传。

同一组的多个 shape 必须使用同一目标 body ID；不同活动组不得别名到同一个 body ID。source key 与全场 key 显式分开；同 body ID 在不同 world 合法。root 的固定 shape 仍为 `ResidentStaticBinding`，不能接成活动行。直接构造器若提供 group-local 材料，只是显式安装声明，不冒称已证明某 URDF 归属；from_asset 才有源资产内容/owner 交叉验证。

首片 shape 复用 C3 的 **sphere、有限 capsule、hull（含 box 的真实角点）**。每个活动 link 可有多个，固定工具的实际 group 也可绑定；sphere 半径和 capsule 两端皮肤都入 bounds，hull 包全顶点。非刚体、无限面、其它 curved shape/动态 mesh 首片具名拒绝，不以 OBB 代理冒原几何。静态对方沿 C1/C2 已支持的有限形状。首片 registered domain 是每个活动 link 对同 world 显式注册静态体；自碰撞/活动 link–link 宽相尚不放行，调用者须明确 exclusion，不能默默滤掉一个已请求的 pair 后声称全场检测。

为避免大共同平移在每层相加，建议在 prepare 取一个固定 root anchor（如第一个根活动关节的原世界原点），静态地从各根位置减去它；设备 FK 的 position/COM 用**世界方向、anchor 相对坐标**保存。R 和 axis 仍为世界方向。J/力臂用相对坐标，材料输出/bounds 最后才加一次静态 anchor。这是数值坐标中心化，不是更改 root pose；多个根活动行分别保留其真实相对 offset。tree view 必须带 anchor 并明示 frame，公共读回再给 world pose。

每 shape 以同一 accepted FK buffer/tick 计算实际全有限表面范围，然后直接写原 TyxCollisionBatch `_bounds`，调用 `query_device_bounds`。共享碰撞核不从 q− scratch、HOST FK 或 endpoint chord 推测 q+。候选 FK 失败就不能接受该物理步；query 的 bounds/BVH 失败只不发布新 result，不倒改已接受物理状态。多个 body/shape 的 tick 需一致，owner/generation/epoch/revision 一并绑定。

沿 C3 的工程精度路线，不新增 Taylor 证明核。正常 sinf/cosf、fast_math=False；原 C3 casebook 已记录本机 TYX lowering 与 NVIDIA observed error 表的非形式保证。多层组合必须扩大误差预算，不能照搬单 rotor 的固定常数。一个可编码的预声明起点为：

`p = 64*eps32*max(1, |anchor|inf, L) + 128*(d+1)*eps32*L + 16*(d+1)*2^-20*L + 32*(d+1)*tiny32`，

其中 d 是该组祖先路径长度，L 是非负路径长度上界：原 joint offset、该路径实际 `abs(q_P)`、原 group-local 材料及半径的 1-norm 上界之和。它不依赖物体名、目标位置或实测误差。静态系数和 p/radius 上传向外取整，最终 lower/upper 向外 FP32；非有限 bound 拒绝。该预算建立在上述 anchor-relative FK 上；若父采用逐层世界绝对坐标，需要另计每层平移舍入，不能沿用此式。工程 padding 允许保守多候选，不声称所有实数认证，也不能按 miss 样本反调 padding。

## 6. 预声明多案例与手推 oracle

以下均为未来 tests-first 输入，本轮未运行。惯量为 COM 主轴惯量，米/千克/秒；h 为真实物理步长。除明确说明外 g=0、初 v=0、KP/KD=0，表中 tau 是即时广义外载，非直接改状态。一般姿态用实际 FP32 stored q 在独立 FP64 公式中求真值。

### T01：两转铰真实交叉质量及速度偏置

两 link 各 m=1、COM=(1/2,0,0)、I=(1/24,1/24,1/12)，parents=(-1,0)，第二 joint offset=(1,0,0)，两轴 z。独立平面运动学为 `c1=R(q1)(1/2,0)`，`c2=R(q1)(1,0)+R(q1+q2)(1/2,0)`。

```
M = [[5/3+cos(q2), 1/3+cos(q2)/2],
     [1/3+cos(q2)/2, 1/3]]
b = [-sin(q2)*(2*v1*v2+v2*v2)/2, sin(q2)*v1*v1/2]
```

q=(0,0)，tau=(1,1/6)，h=1/128：`M=[[8/3,5/6],[5/6,1/3]]`，a=(1,-2)，v+=(h,-2h)，q+=(h²,-2h²)。只保对角或拆两 rotor 明显错误。另一入态 q=(.37,π/2)、v=(1/2,-1/4)、tau=0：b=(3/32,1/8)，a=(3/128,-51/128)。另固定 q=(.37,-.51)，以及转/平移协变与同一 URDF 的元素重排（按名轴/载荷映射保持），核完整 J/M/b，而非只有末端。

真实连续窗口固定 40 步、h=1/1024，从第一静止入态恒 tau=(1,1/6)；独立 oracle 每步用上式重新计算 M/b 并半隐式推进，不能每步灌状态或冻结 M。保存入/末 q/v 与实际 FK。能量用当前姿态，半隐式一般不严格守恒；同 q 上的虚功/冲量方程可精确核，窗口能量与独立离散 oracle 比较，不把末速度乘力的功误叫连续守恒。

### T02：R + 两个 P 分支与逆质量跨支耦合

根活动 Rz，m=2、COM=0、I=(1/8,1/8,1/4)；两个子 P 均 m=1、COM=0、I=(1/16,1/16,1/8)，parents=(-1,0,0)。P1 offset=(1,0,0)、axis=y；P2 offset=(0,1,0)、axis=x。令 q=(r,s,t)：两子原点为 `Rz(r)(1,s,0)` 与 `Rz(r)(t,1,0)`。

```
M = [[5/2+s²+t², 1, -1], [1,1,0], [-1,0,1]]
b = [2*s*vr*vs+2*t*vr*vt, -s*vr², -t*vr²]
```

q/v=0，tau=(0,1,0)，h=1/128，a=(-2,3,-2)，v+=h*a，q+=h²*a。未受直接驱动的另一支 P2 必须响应。非零留出 q=(.37,-1/16,3/16)、v=(1/2,1/4,-1/8)：M00=325/128，b=(-5/128,1/64,-3/64)。交换两个 sibling 的存储次序须保持按名读数/shape 相同；反转某 P 轴时相应 q/v/effort 变号。40 步 h=1/1024、tau=(0,1,0)，同独立公式更新。

### T03：固定工具、非对角聚合惯量与原材料

在 T01 第二活动 link 加一个 fixed tool：tool 原点相对 link2 为 (3/4,0,0)，朝向 Rz(π/2)；tool m=1/2、其 authored COM=(1/4,0,0)、I=(1/64,1/32,5/128)。tool COM 在原 link2 中为 (3/4,1/4,0)。手推聚合：m=3/2、COM=(7/12,1/12,0)，在原 link2 frame 的 COM 惯量为

```
[[3/32, -1/48, 0], [-1/48, 5/64, 0], [0, 0, 21/128]].
```

q=0 时工具对全 M 的独立增量为 `[[205,93],[93,45]]/128`。这由工具 COM 的两列 J 与自身 Izz 求得，不取 importer 输出反写。用同一 source URDF 的三个质量部件逐项和验证折叠质量/动能；原 authored box/球中心经固定变换及真实第二链 FK，与原 source material IDs/修订号同时核对。q=(0,0)、(.37,-.51)、(-.51,.37)，再共同 xyz 循环与平移(.5,-.25,.75)。Newton 必须保 fixed joints，不能用另一套 collapse 输出当独立真值。

### T04：非主轴旋转、当前 COM 力/矩及重力

两行 R/P：R axis=(3/5,0,4/5)，m0=2、c0=(0,1/4,0)、I0=(1/8,1/4,3/8)；P parent=0、offset=(1/2,0,0)、axis=y，m1=1、c1=(0,1/8,0)、I1=(1/16,1/8,3/16)。q=(.37,-1/16)，v=(1/4,-1/8)，g=(0,0,-2)。F0=(0,-1,0)、T0=(1/8,0,0)；F1=(1,2,3)、T1=(1/4,-1/2,3/4)；generalized=(1/8,-1/4)，h=1/1024。

独立 Rodrigues R(a,qR)，`pCOM0=R*c0`，`pCOM1=R*(1/2,qP+1/8,0)`。R 列为 `a×pCOM`，P 列为 R*y；角 J 只有 R 列 a。qdd=0 的 COM 加速度为 `ω×(ω×p)`，P 子另加 `2ω×(R*y*vP)`；矩为 `ω×(R I R.T ω)`。据此独立算 J/M/b 和 `J.T*wrench`，含全六维载荷。重力单独关/开，与仅外载对照；共同 xyz 循环及平移不改变广义响应。任意入步 v 的虚功须满足 `v·e = Σ(F·vCOM+T·ω)+v·generalized`。这是同一 q 的投影，不以初始轴/初始力臂代替后续姿态。

### T05：全耦合限幅、物理步延迟与未收敛

用 T01 零入态，h=1/16，kp=(4,8)，kd=(1,2)，caps=(1/4,1/2)，target/rate=0，ff=(1,-75/448)，无即时外载。独立隐式解：tau=(1/4,0)，u=(3/112,-15/224)，q+=h*u；第一轴饱和而第二轴不饱和，后者若独立标量求解将错误。

同输入 delay=1、初始 queue 为零 packet：第一个无外载步不动，下一步才使用 ff；另给第一步即时 tau_ext=(1,1/6)，必须当步响应，不能随 target 延迟。拒步不消费两轴队列，恢复/重试同控制确定相同输出。max_iterations=1 的另例固定 ff=(1/4,1/4)、caps=(10,10)，其余同上；非零 cross 项使一次从零开始的坐标 sweep 未解全方程，须返回未收敛状态并保整 world，不准 HOST 补解。完整预算的同例是正控。


另冻结两个不削 cap 的原始律留出，均用 T01 模型、q/v=0、h=1/16、caps=(2^20,2^20)：

- **宽 cap、小命令**：kp=kd=0，ff=(1/8,-1/16)，无即时外载；实际 tau 必须为原 ff，不能因大 cap 允许几十 N·m 的误差。
- **大项相消、小实际驱动**：kp=(1,1)、kd=0，target=(2^15,-2^15)、rate=0，ff=(-2^15+1/8,2^15-1/16)，即时 generalized=(-1/8,1/16)。精确平衡为 tau=(1/8,-1/16)、u=0、q+=0；目标是有限命令而非状态，不因目标超 q 数值域改写它。原始 kp 项与 ff 相消，门只按真实 tau/expected；宽帽不变。独立核实际力抵消及状态，不以“finite”替代验收。
### T06：表示、接受/候选与失败

N=1 保原真实 one-R/one-P 物理能力；N=9、N=16 用通用交替 R/P 串链，parents=i−1，offset=(1/8,1/16,0)，轴按 z/x/y 循环，m=1+(i mod 3)/2，c=(1/16,1/32,0)，I=(1/64,1/32,3/64)。R 初 q=(i mod 3−1)/8，P 为 (i mod 3−1)/64，v=0，恒 generalized=(-1)^i/16、g=0、h=1/1024。以独立逐体解析 J/递推和 Newton 为 oracle，不能只检查 finite；N=17 分配前拒绝。

冻结负控：一个 R=4096、向外初速度引起越域；一个行的有限外载导致溢出/超域；不合法 parent/axis/inertia；delay=2、stops、mimic/floating；错误 command 行数/域名、未知 key；未收敛；同步或读 status 注入异常；close 两 plan 中一个 callback 抛错。整树 q/v/FK/history/applied/tick 不部分更新，其它 world 按原数值拒绝合同继续。设备异常全批无 swap/新 epoch；空 batch 和空 query 无伪 device 工作。reset/restore 一棵树、same-tick restore、换 owner/generation/source/revision、读回后改外部数组均有对应反例。当前持有命令与上个接受输入分开记录，不能把 set_controls 后尚未执行的载荷标成已施加。

### T07：各 link 的真正有限几何

在 T01/T02/T03 不同活动组绑定：偏心球 local center=(1/4,-1/8,0)、r=1/16、ID=(101,)；capsule 端点(-1/2,0,0)/(1/2,0,0)、r=1/32、IDs=(201,202)；局部 Rz(π/4) 的 box 半尺寸(1/4,1/8,1/16)、IDs=301..308；非盒五顶点 hull 为矩形底(-/+1/4,-/+1/8,-1/16)加顶(0,0,1/4)、IDs=401..405。revision=7；同组 box/hull 多 shape、不同组不同 body、root 静态 shape 均覆盖。另细长 capsule 长16、r=1/1024。

预声明姿态为零、(.37,-.51)、(-.51,.37)、含正负多周回及 ±4096 的合法 R；按实际树行数补零。共同 xyz 循环和大平移(4096,-2048,1024)不换材料。独立逐点 Rodrigues FK + sphere/capsule 皮肤计算 FP64 完整范围；prepared 静态目标面在该固定姿态 oracle 的 x 极值内 `1/65536 m`（真实重叠）及外 `1/4 m`（清楚 miss），其 y/z 面覆盖真实有限范围。目标在执行前从已声明姿态生成，不能从设备测出的 bounds 倒造。胶囊轴端外额外 miss；不按无限直线包围。允许有界 padding 多候选，以上清楚 miss 需通过，不移动 fixture 去适应结果。

先真实 step，再 collision；一组显著 q−→q+ 变化同时证明不是入步 FK。全部 bounds/候选带原 key、source ID、owner/link、epoch、tick。冒绑另一 link、改变 source radius/IDs/revision、漏一个 shape、不同组别名 body、同 world self pair 未显式 exclusion 必须拒绝，不返回假净空。

### T08：W1/W4/W32、完整输出与成本

W1 为 T01；W4 为 sphere/slider/rotor/T02 异构 worlds；W32 按八项循环排入 sphere、slider、rotor、T01、T02、T03、N9、N16，各项 4 份。key 非连续、注册反序；多个 world 可复用相同 body/shape IDs。树保持各自 n/parents/model/controls，40 步 h=1/1024，不重置。两个 active 方向都真实读取；局部 restore 后再次推进。

功能 trace 保存真实 q/v、applied、当前 FK/COM、完整 bounds/candidates/ticks、接受状态、所有传输/分配计数。另设严格 warm 区间：先保存真实 checkpoint，区间内只 step+collision，禁止 snapshot/kinematics/FK/M/J 的 HOST 动态上传和新 device allocation，完整保留当次 bounds/candidates/ticks/status/stats；区间 q/v 若用独立递推只能标 oracle，不能冒实测。区间后实际读回 q/v 与 oracle核对，不为收 trace 偷加未计 readback。BVH layout/params metadata 上传和原候选/界/status读回可以保留，逐类如实计数；不宣称零传输。计时须用相同完整公共输出；未做配对测量前不报告速度收益。

## 7. 精度与独立对照放行条件

初始工程门建议：每轴 a/v `rtol=5e-5, atol=5e-6`（按 R/P 单位），q 常规 `atol=5e-6`、40 步 `atol=2e-4`；J/M/b 同量纲分项 `rtol=5e-5, atol=5e-6`。大平移 pose/bounds 使用所声明位置预算，不能把世界坐标相对误差当全部物体的尺寸容差。原二进制可精确例保精确oracle；物理驱动残差门见第4节。所有数值门先于首次执行冻结，不根据观察抬 cap、减质量或重定目标。

参考矩阵有三层：手推有理数/平面公式；独立 FP64 Rodrigues、点导数与逐部件惯性和（不 import Phymeo 生成期望）；安装的 Newton 1.7 CPU/真实 CUDA 原生推进。Newton 首窗口至少 T01、T02、T03、T04；同预声明初态、h、外载，各自推进自己的 state，不给它注入 Phymeo 的后续状态。原生目标控制与 Phymeo 隐式 PD 的差异另列；若把已冻结广义 effort 序列给双方，只能称积分/载荷对照，不能称独立控制器对照。

Native `add_urdf(..., collapse_fixed_joints=False)` 保 T03 工具部件；root 固定，armature/damping/PD/limits/contact 关闭且读回元数据。从关节名和 q/qd start 映射，不把 fixed joint 计作活动行。记录所有 `joint_q/qd`、`body_q/qd/body_f`、原始惯量/COM、shape/frames、control、device/version和 source snapshot；COM/world 与 authored frame 的换算注明。可借鉴已有 `tests/reference_newton/robot/floating_fixed_tools_reference.py` 的元数据/force/完整 JSON做法，但新固定树 oracle 独立，不复跑旧 F1 验收。

只有真实 interpret/CUDA 轨迹、全矩阵物理、事务、材料归属和 warm device-bounds 路径共同通过后，才可称“受限固定根 R/P 树的接受设备状态直接供宽相”。不能据此声称完整机器人场景驻留、接触力反馈、CCD/self collision、浮根/mimic、任意 N、严格实数包围、自动可微或性能更快。上述均为后续独立节点；本册所有新案例仍 **UNEXECUTED**。

## T1 API 与保存 schema（父授权测试作者追加，仍 UNEXECUTED）

正常入口为 `resident.FixedRobotTree(program, initial_scene_state, control_max_iterations=512)`，其中 resident 为 `phymeo.backends.tyx_world_batch`，沿原 TyxWorldBatch/set_controls/step/snapshot/restore/reset/prepare_collision。新增接口在 test function 内取用，首有效 RED 先构造并断言既有 T01 RobotModel/DrivenRobotProgram 有效，不能在 collection 导入缺模块。

- `snapshot()[world]` 为 frozen TreeWorldCheckpoint：沿旧 `world,topology,tick,position,velocity,applied_effort,history,control`，增 `generalized_applied`。position/velocity/applied_effort/generalized_applied 为长度 n 的 tuple；applied_effort 只为实际 actuator drive；generalized_applied=drive+即时广义力/COM wrench 投影，尚未减 bias。control 是 n×10 tuple，delay1 的 history 是 n×3 tuple（每行 target/rate/ff），delay0 为空 tuple。topology 区分种类、n、完整模型/静态帧/增益/延迟；不以只有相同 n 允许跨机构 restore。
- `tree_kinematics(keys)` 返回按请求顺序 world→owned 不可变条目的 mapping。每项 `tick,joint_types,position,velocity,link_position_m,link_rotation,com_position_m`；position/velocity 每行单位依 joint_types 为 rad/rad/s 或 m/m/s，link/COM 是当前接受末态世界坐标。position/velocity shape[n]，link/COM[n,3]，rotation[n,3,3]。空选择返回空 mapping；旧 prepared scalar reader 选中 tree 允许具名拒绝。
- `tree_diagnostics(keys)` 为有界的最近接受 attempt 的**入步力学**读回，返回同序 world→record 或 None。record 字段固定 `entry_tick,exit_tick,dt_s,entry_position,entry_velocity,linear_jacobian,angular_jacobian,mass,bias,external,drive,acceleration,iterations,control_residual,equation_residual`；J 为[n,3,n]，mass[n,n]，其余向量[n]；external 不含 drive 或重力，bias 包重力和速度偏置。residual 两项分别是原始律最大绝对 effort 差和归一化质量方程差，iterations 为实际 sweep 数。零 gain 的直接 drive 路径可为0。
- 初始化、成功 restore/reset 的对应树、该树最新数值拒步时 diagnostic=None；全设备异常的本次尝试也令涉及树的 diagnostic 无效/None，但不改物理 state/epoch。HOST 输入校验失败且未启动设备 attempt 不使既有诊断失效。已返回 record 是 owned 不可变快照，后续失败或成功重试不能改它。不能用被失败 candidate 覆盖的 scratch 继续提供“最后成功诊断”。该小读回不是永久历史库；当前接受 FK 与入步诊断要分别标时刻。
- 诊断真实 readback 累加 `tree_diagnostic_reads`、`tree_diagnostic_downloaded_bytes`（interpret 使用对应 `tree_diagnostic_host_read_bytes`），不能假称免费；warm 碰撞区间不调用它。新 tree 资源/launch 统计并入原 total 与独立 tree 项，旧计数意义不改。
- `ResidentTreeBinding(key, world, group_index, local_points_m, *, topology)` 为显式 group-local 安装声明；topology 取本 tree checkpoint.topology。`from_asset(asset,source_key,*,key,world)` 携真实 owned model/source/group证明，父在 prepare 与当前物理 tree 全模型交叉核。直接声明不能被报告成 URDF source 证明。

测试 JSON 用相同执行读数写入 tmp_path：input（模型、q/v/history、完整 command、dt、source/shape 定义）、entry/actual checkpoint、attempt acceptance、入步 diagnostic、末态 kinematics、collision result 完整 key/revision/registered_pairs/candidates/bounds/ticks/epoch、batch/plan stats、逐项独立 oracle。每步先保存已取得的原读数再断言；发生失败保留已到达的数据。严格 warm trace 的 q/v 若为独立递推，字段必须明确 `oracle_*`，真实 checkpoint只存区间前后，不增加隐蔽 snapshot。

### T1 完整作者 checkpoint：观察与执行入口（未执行）

父已在固定首 checkpoint 的正常公共入口取得有效 RED：T01 既有模型/状态/输入与输入 JSON 先成功，再因 `FixedRobotTree` 尚不存在失败。该 RED 不能称物理数值失败；作者没有运行 collection、import、数值或静态工具。本节与三份测试/参考脚本为 **TEST_AUTHORING_ONLY / UNEXECUTED**，不是产品交付。

补充既有公开观察名称：`batch.summary()` 的 tree 项仅有 `kinetic_energy_j`，使用真实 q+/v+ 的全树末态动能；不为混合 R/P 输出虚假的单一 q_m/q_rad。其他 world 的原 summary 字段保持。诊断仍是带 entry/exit tick 的入步 M/b/J/载荷，不能混成末态缓存。`tree_kinematics([])`、`tree_diagnostics([])` 返回空映射；空 collision model 沿旧合同没有 registered world/tick 行，即使所属 batch 内有树也不凭空登记几何。

专项分工与固定记录：

- `tests/scale/test_resident_tree_step.py`：T01 解析 2R/非零速度及共同变换，T02 R+双 P 的未驱动分支、重排/负轴，T03 三部件原 URDF 与聚合非对角惯量，T04 当前姿态的重力/COM 外载/虚功；六种机构各真实40步。T05 包括原部分饱和 packet 的0/1步延迟、即时外载、宽 cap/大项相消、原1次 sweep未收敛及完整预算正控。T06 N1/N9/N16/N17、合法物理输入边界、整树拒步/邻域继续/重试、完整 held/history/drive/generalized checkpoint、不可变诊断和同尺寸异模型拒绝。
- `tests/scale/test_resident_tree_collision.py`：T07 保球/有限轴皮肤/局部旋转盒/非盒凸体的原材料 IDs，含正负多周回、±4096、斜安装、大公共平移、近触/清楚 miss、长16 m细胶囊、真实 q−→q+ 的显著转动、普通/固定工具 source 和两个 P 分支、固定根静态材料。T08 W1/W4/W32 沿原 batch 的40步实际全输出；独立 warm 组先真实用过两个 buffer 方向，再40步只 step/query，禁 snapshot、kinematics、diagnostic、HOST envelope路径与新 device allocation。
- warm 组的中间 q/v 只有独立递推，明确保存为 `oracle_only_q_v_not_device_readback`；中间几何只作同量纲轨迹一致性检查，位置预算由原40步 `2e-4` 坐标门乘 `P祖先数 + R祖先数×有限链reach` 推得，未增加新的状态验收门。严格 FP64 材料包围在完整读回组的**每个实际 q**及 warm 末态再验。该区间不是靠 HOST FK 上传生成界，也不声称独立递推等于每步实测状态。
- trace 保存所有当次输入、实际 state/FK/入步诊断、完整 bounds/candidates/注册 pair/ticks/epoch、真实分配/上传/读回计数。数值窗口用 finally 保留已取得的 trace；无效对象/资源前置拒绝另列。`tree_bounds_kernel_launches` 单计新 pass；旧 launch/bytes 计数不改，metadata 与结果实际读回不隐藏。当前无性能测量结论。

唯一参考文件 `tests/reference_newton/robot/resident_tree_reference.py` 同时容纳自包含原始模型常量和独立 FP64 oracle，专项延迟加载其纯 NumPy 函数；该文件从不 import Phymeo。它以齐次变换的一阶坐标导数构造 J，以沿 qdot 的二阶时间导数构造真实 COM/角偏置，加逐物理部件惯量与 Iω项；不是调用产品 FK/M/b，也没有数值差分。T01/T02 的闭式有理数/三角式另外检验该 oracle。T03 保三个 authored 物理部件；与产品折叠后的两真实活动组相比较，不从产品主轴输出反写真值。

Newton CLI 仅 main 导入原安装 Newton1.7.0/Warp1.17.0，CPU/CUDA 显式选且不回退。原 URDF、每个 body/joint/shape 的模型数组、标签/帧/惯量、每步全部 q/qd/body_q/body_qd/body_f、实际 control、solver 原 H/qdd 原量、原始 source 字节快照及 SHA/版本/device 均保存。Native body_qd 为世界 COM 线速度在前/角速度在后，body_f 为世界 COM 力/矩；固定工具不 collapse。先三个静态姿态读原生 FK/速度，再从独立初态连续40步 Featherstone；只有初始/独立静态样本设置 state，不灌入另一引擎的后续状态。原生 PD、armature、damping、stop力显式归零并读回；Featherstone 1.7 声明不支持的 joint friction/effort/velocity limit/target_mode 只作原始元数据，不冒称这些原生控制与 Phymeo隐式PD同义。当前对照是完整耦合惯性、COM外载和时间推进，尚未执行。

父串行入口（必须按父 fresh guard/证据路径运行；以下只是命令合同）：

```text
python -m pytest tests/scale/test_resident_tree_step.py tests/scale/test_resident_tree_collision.py --tyx-mode interpret --require-tyx
python -m pytest tests/scale/test_resident_tree_step.py tests/scale/test_resident_tree_collision.py --tyx-mode cuda --require-tyx
<Newton-1.7-python> tests/reference_newton/robot/resident_tree_reference.py --device cpu --output-dir <fresh-evidence>/tree-newton-cpu
<Newton-1.7-python> tests/reference_newton/robot/resident_tree_reference.py --device cuda:0 --output-dir <fresh-evidence>/tree-newton-cuda
```

CLI 默认 T01/T02/T03/T04/N9/N16，各40步；`--case` 可重复做明确子集，`--cycled` 是预声明共同轴循环+平移留出。每例保存真实首错并继续独立下一例，summary 只在所有实际 gates 和40步完整时 exit0。脚本未执行，不将安装、源阅读、手推、有效公共 RED 或可写脚本本身列作 Newton 通过。
