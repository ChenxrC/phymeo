# Resident 接触反馈首片草案

状态：**NOT_FROZEN / DOCS_ONLY / UNEXECUTED**。本文件是父对既有后继节点的实施裁定材料，不是另一个已启动项目，也不授权产品、测试或性能工作。只新增本文；全部项目执行仍由父唯一串行安排。

正文依据父工作树 `fd5bec57751d9f54d75e619efbfb067aa1ce2782` 当次实际源文本，不以作者较旧隔离树推断 C4/C5 接缝。主入口仍为 [C4 计划 §7 后继一](../superpowers/plans/2026-11-07-resident-tree-broad-phase.md) 和 [C5 后继边界](../superpowers/plans/2026-11-07-resident-affine-tree.md)。后续正式阶段由父补在原计划，本文不替代它们。

## 1. 下一最小可交付能力

每 world 一棵普通 fixed-root R/P tree，对显式登记的有限静态 sphere/capsule/unrounded full convex hull，执行一次有真实接触载荷的半隐式设备步。采用 **accepted-device 状态 → HOST 即时窄相 → 原 `RobotContactMaterial` 法向律（严格 μ=0）→ TYX 完整树响应 → candidate 几何/收据审核 → 一次发布**。

首片是混合执行，状态和树力学在原 runtime，窄相与法向材料采样在 HOST；必须公布实际姿态/J 读回及载荷上传。它不是全设备窄相、连续碰撞检测、完整 GPU 场景，也不借用 sphere-island 的独立球冲量模型。静态支撑获得真实等反力/关于指定固定参考点的矩，但不积分静态体；此反力不能叫树固定根的全部关节支撑反力。

- 复用原树 N∈[1,16]、q∈[-4096,4096]、delay0/1、默认512且上限4096 sweeps、FP32可表示域；不扩大 physical joint stops、非正定、闭环或特殊关节域。
- 材料必须显式为原 `RobotContactMaterial` 且 `friction == 0`。非零摩擦、`NormalLayer`、隐式 restitution 等不转换、不静默丢项；声明时拒绝并说明后继。
- 几何保持当前有限形状及其 source/group/local vertices/IDs/revision。沿已有 patch 的唯一法向/浅接触域；圆轴深包含、多法向不确定、rounded hull 等原 unsupported 原样暴露。多个同组 shape 可登记，dynamic/dynamic 对仍须明确排除；不能借本片打开原未实现配对。
- 每步只从入步 q-/v- 采样一次物理法向力，candidate 的再次几何读取仅作末态观测/审计，不能用末态力回填本步，也不额外执行一次完整 step 来猜碰撞。
- 当前姿态离散查询可能漏过整步内发生又消失的接触；公开模式必须明示 `discrete_entry`。请求 CCD/TOI 时拒绝。路径保障另用已有且确实支持的连续查询；不得用两端弦、仅端点无重叠或减小步长声明一般 CCD。

## 2. 源码依据与确切缺口

以下为父源当次行号，正式实施前父核当前 diff，不反复重验旧数学。

| 当前入口 | 可复用内容 / 缺口 |
| --- | --- |
| [tyx_world_batch.py:283](../../src/phymeo/backends/tyx_world_batch.py) `_borrow_collision_state` | 同锁 owner/generation/epoch/active/ticks/FK 只读借用已存在；没有带入步 J/M/COM/twist 的接触机制视图。 |
| [_resident_collision_plan.py:417](../../src/phymeo/backends/_resident_collision_plan.py) `_prepare_topology` | 真实 current-pose dynamic/static pair 域；tree moving/moving 必须排除。名字不是旧树的 `_prepare_host`。 |
| 同文件 `query():653–690` | 读取并验证后立即写 `layouts/_last`。不能在物理候选接受前用这个公开方法提前发布“本步已接受接触”；需要纯 stage 与最后 publish 的薄拆分，原 query 行为保持。 |
| [_resident_tree_storage.py:278,300,306,327](../../src/phymeo/backends/_resident_tree_storage.py) | 已有入步 `stage_mechanics`、candidate FK/energy、finalize；`stage_advance` 目前把 mechanics/solve/FK/finalize串起来，没有法向载荷插入点。可拆出原 solve 调用，不重写算法。 |
| [_tyx_resident_tree_kernels.py:249–267,339–345](../../src/phymeo/backends/_tyx_resident_tree_kernels.py) | 原 command 中 full joint effort + COM F/T 经 J 投影为 external，再计算 free velocity。接触应在此之后、同一次 solve 之前加入；不能改 held command 把暂态接触带到下一步或延迟队列。 |
| 同文件 `tree_solve:452–490`、`tree_energy:587–589`、`tree_finalize:592` | 原门是 clipped-drive残差、质量方程残差、数值域/FK/动能有限性。**不存在接触能量守恒或接触功闭合门。**不得借“finalize已过”宣称新接触能量已通过。 |
| [contact_patch_adapter.py:283](../../src/phymeo/geometry/contact_patch_adapter.py)、[contact_patch_loads.py:10](../../src/phymeo/solvers/contact_patch_loads.py) | 已有真实双侧 witness、共同作用点、patch 权重一次和原法向律；不再写平行几何/材料求解器。 |
| [_convex_owner_contacts.py:39–69](../../src/phymeo/runtime/_convex_owner_contacts.py) | 即时 patch/原兼容分支、两表面和共同点、COM 力矩、入步功率字段可复用。纯适配不创建 RobotScene/MultiRobotScene，也不让其推进第二份状态。 |
| [tyx_world_batch.py:262,370–456](../../src/phymeo/backends/tyx_world_batch.py) | 原恢复同步与 unavailable 语义、统一 swap/epoch 可保留；新路径须在 swap 前完成所有可能失败的结果构造。 |

特别保留当前诊断语义：数值/设备尝试失败可令 `tree_diagnostics()` 为 None，失败计数可增加；这不等于物理状态或 held/history 被发布。旧已返回 owned 读数不可被后续失败改写。

## 3. 拟接口和状态所有权（待父冻结）

建议只增加一个显式入口，不更改 `batch.step(h)` 默认行为：

```python
contact_plan = batch.prepare_contacts(
    collision_model, bindings, materials_by_pair,
    candidate_capacity=..., site_capacity=...,
    sampling="discrete_entry",
)
accepted = batch.step_contacts(contact_plan, h)
receipt = contact_plan.last_accepted  # owned immutable；入步/末态时刻分别标出
```

容量沿现有模型/工作区上限，由调用方显式声明；本文不另冻结一组未经审核的 station/page/world 帽。配置在任何分配前检查同 world、实际树/group/source、材料支持域和原排除集。`prepare_contacts` 只编译静态声明/准备 scratch，不执行接触力、不改变现态。空登记合法；有候选但无正载必须返回明确零力，而 unsupported/overflow 是独立结果或拒绝，不能吞为分离。

内部最小合同建议（不是要求复制这些类名）：

- `EntryContactView`：只在原 batch 锁和 `_device` 恢复边界内有效，带 owner、topology generation、state epoch、active、world→tree row、tick、当前 q-/v-、FK/COM、full J/M/b 和真实当前 external/已选延迟控制。借用 device handles 不外泄；需要 HOST 几何的数据单次形成 owned 包。M/J 必须来自现 mechanics kernel，不拿 HOST 重算值上传冒充设备力学。
- `StagedContactLoads`：绑定上述完整 stamp、canonical pair/source IDs，保各 site 的两侧 material 权重/local anchor、世界表面点、共同作用点、gap、normal、weight、两端入步速度及原 law；另保真实 FP32 待施加的每 link COM wrench 和其 full generalized 投影。它不是 accepted 历史，重排 row 不能改变物理 owner。
- `ContactStepAttempt.finish(candidate_view)`：消费同一 candidate bank 的实际 q+/v+/FK，构造末态材料点/位移、状态对应的力/冲量/功诊断及 owned receipt。不重新调用 prepare/integrate，不给调用方裸可变 candidate。若后续赋值外的校验/分配仍可能失败，必须在 swap 前完成。
- 最后发布只由 batch 执行：一个全局 active/epoch 变化，各 world 成功则自身 tick+1；失败 world 的所有 q/v/FK/延迟/last applied/tick保留，成功邻 world按旧规则继续。accepted receipt只关联成功 world与本次确切 entry→exit；失败原因另存，不把 rejected attempt 当 last accepted。same-tick restore 也使旧 attempt/result失效。
- plan只拥有自己的静态绑定/scratch/owned收据，借用 batch runtime/资源计数。close/构造中途失败清理本plan而不关闭batch；batch关闭使所有plan失效。不得第二runtime、第二state store或把W次HOST求解称设备合批。

时刻先用 entry_tick、exit_tick 和本步 h 表达；变步长时不能用 tick×当前h 伪造累计秒数。若要绝对物理秒，需父明确可恢复时钟合同。本片不通过新事件观察器推进另一套时钟。

公共 checkpoint 的 held control 保持原命令，不能混入暂态接触。新的 receipt 至少区分 `input_external`、`contact_wrench`、`contact_generalized`、`total_generalized_applied` 和 actuator effort；原诊断若输出总 external，必须在该新入口文档明确，并保可独立还原原输入，不能含糊复用字段藏一次加力。

## 4. 单次步进时序与计量

1. 验证 plan归属、模型/修订、h、所有当前输入及容量；锁住 `(src, dst, epoch)`。沿原规则上传 dirty controls，暂不推进物理时钟。
2. 在 src 上运行一次原 `stage_mechanics(src)`；取同源 FK/COM/J、v-。device accepted bounds→原BVH得到入步候选，用纯staging路径保留 pending结果；不是公开query先发布。HOST窄相使用这一时刻的真实有限 world vertices，不以 bounds角点代形状，也不以 HOST q 重算替换实际device FK。
3. 对每 site 计算 `v_point = J_linear*v- + (J_angular*v-) × (p-COM)`；静态侧速度严格0。该入步速度进入原法向律 `N=max(0,k*depth-c*normal_rate)`，depth≤0时N=0。patch权重只乘一次，两侧共同点施±F；动侧COM矩为 `(p-COM-)×F`，静态侧记录对应反矩。不能用 free/candidate速度替代v-而悄悄改成隐式阻尼。
4. HOST记录原law FP64结果，按既有FP32可表示规则形成实际上传wrench；保存转换后值与量化差。原device J以原投影语义把接触wrench加到**本步临时载荷**；input external、gravity/bias与contact各计一次。不得overwrite held/commands或经actuator cap裁接触。原 full mass及coupled drive solve只执行一次；有PD时contact必须影响本次free velocity与驱动解，不能在已解PD后补一个速度kick。
5. 得到candidate q+/v+/队列后重建真实FK/energy。实际末态材料点由入口local anchors通过candidate真实frame搬运，不重采样最近点来冒材料运动；末态几何/normal可单列，但本步反力仍带 entry time。接触试算的有限性/绑定/容量失败在该 world 接受前处理，必要时通过原retain机制拷回其完整旧 bank。
6. 保原残差/有限域，构造完整 immutable receipt，统一sync/status并执行原单次发布。若发生设备或同步异常，沿原 `_device` drain/rebuild要求保accepted bank，不HOST rescue。结果分配失败也不得先换state再报错。全局失败时已有last accepted接触读数不被候选覆盖。

入步力乘h是本离散步接触冲量；`F·v_point`/`tau·v-`是入步功率。另记录 `F·(x_anchor+−x_anchor-)`、实际ΔK、重力/外载和弹性压缩变化，**不强行令它们相等**。旋转中入口冻结wrench与随点变化的连续力矩并不完全相同，实际anchor位移也不等于 `h*J(q-)*v+`。首片复用原离散算法并用独立同算法递推验算；不以任意新增能量预算盖过差值，不宣称连续功或严格能量守恒。若要增加接受门，父必须先从原案例/误差合同裁定。

数值接受继续使用原质量方程相对残差 2e-5 和原 clipped-drive 5e-5×max(1,|tau|,|expected|)，其中 expected 仍由原完整驱动式计算；不用cap或相消前大项放宽。接触几何/容量拒绝需要父补独立原因与原check_advance认可的状态映射，不能伪装成M不正定或写入未知code后破坏原retain协议。所有结果构造完成前，candidate只可写dst。

实际HOST输入包含设备FP32 FK的扩宽值；不得为了通过材料frame检查静默正交化/吸附。纯patch可处理实际world points；如果下游现接口要求比该FP32读数更严格的刚性frame，先交实际失败与精确validator，再裁定适配，不能改原几何/阈值。

计数分别报告：mechanics/FK/solve/bounds实际launch，入步pose/J读回、contact上传、status/末态审计读回字节，同步及HOST narrow/law/构造墙钟。公开完整读数和最小必要读回分开计量，不把HOST工作报为kernel时间。本草案不设速度目标；功能接受后只比较相同输出的新旧执行任务。

## 5. 建议固定案例输入与独立预期

以下是可立即据此写tests-first的**候选输入**，尚未父冻结、未执行。米/千克/秒；除写明外重力、初速、drive/外载、kp/kd均0；`h=1/1024`，同物理窗口的 `h/2`留出。target/rate/feedforward显式0、delay0、effort_limit沿原C4专项每轴8、control_max_iterations=512；原slip_speed保默认（μ0不改变结果）。原C4步数/精度与残差门作为复用起点，不另增加能量门。所有world用真实建模声明，力从几何/law产生，不能直接把期望N注入。

### R01：有限面压载与入步阻尼

单P轴+z，m=2、COM=关节原点、I=(1/8,1/8,1/8)，安装原点 `(0,0,3/32)`，q=0。动态盒相对COM半宽均1/8；静态盒 `x,y∈[-1/2,1/2], z∈[-1/4,0]`。k=64、c=0、μ=0。真实重叠1/32，归一patch总N=2，不论四站点分布总力不能变8；支撑总反力−2z，动侧a=1、v+=h、q+=h²。

独立阻尼留出：保持几何，v-=+1/8、c=4，N=3/2、a=3/4；反向v-=−1/8给N=5/2。同一v-=+1/8、c4再加原instantaneous COM外载+2z N，N仍为3/2、总a=7/4；确认law使用原v-，不是加外载后的free velocity。将安装x改为1，使两个有限盒间真分离，N必须0；不以无限平面补力。

R01驱动留出仅将kp/kd改为原C4已有的4/1，q-/v-=0、target/rate/ff=0、c0，仍由几何产生N2。未饱和解析解为 `v+=2h/(2+h+4h²)`、`tau=-(1+4h)*v+`、`q+=h*v+`。这是接触必须进入同一次隐式PD解的正控；不能先解零载PD再补接触kick。

### R02：R+双P分支的接触力矩抵消及跨支响应

完整质量/拓扑沿 [C4 T02](p65-resident-tree-casebook.md)：根Rz m2、COM0、I=(1/8,1/8,1/4)；两个子P各m1、COM0、I=(1/16,1/16,1/8)，parents=(-1,0,0)，offset分别(1,0,0)/(0,1,0)，轴分别y/x，q/v=0。

只在P1登记球：local center `(-1,7/64,0)`、r=1/8；静态盒 `x,z∈[-1/2,1/2], y∈[-1/4,0]`。k=64产生真实+1y N，作用点x=0。P1 COM在x=1，故COM矩为−1z N·m，原J散回得到e=(0,1,0)，不是(1,1,0)。独立M为 `[[5/2,1,-1],[1,1,0],[-1,0,1]]`，因此a=(-2,3,-2)，未直接接触P2也响应。共同作用点两侧的净力/净矩为0；固定树整体基座反力不可冒称为这一个静态盒接触矩。

交换两个sibling声明次序并按材料source重映射；再整体轴循环 `(x,y,z)→(y,z,x)` 和平移 `(1/2,-1/4,3/4)`，广义结果按原名字保持，不以body编号查world或机械列。

### R03：偏心转动工具的真实入步速度与实际末态材料点

单Rz，m1、COM0、I=(1/8,1/8,1/4)。球local center `(1/2,3/32,0)`、r=1/8；静态盒 `x∈[-1,1], y∈[-1/4,0], z∈[-1/2,1/2]`，k64、μ0。q/v=0、c0：N=2y，COM矩+1z，a=4 rad/s²。留出v-=1/2 rad/s、c4：入步法向速度1/4 m/s、N=1、矩1/2、a=2。忽略ω×r或把载荷施COM会直接失败。

实际candidate球心用独立 `Rz(q+) local_center`重建，不能用入口切线位移替代圆弧。40步短窗独立令 `(cx,cy)=Rz(q)*(1/2,3/32)`、`delta=max(0,1/8-cy)`、`normal_rate=cx*v`，按原law得N，`tau=cx*N`、`a=4*tau`再半隐式递推（有限面内/唯一法向前提需逐步核）；保存加载/未加载时刻，所有深包含或非唯一法向按实际原因停，不调原位置逃避失败。旋转力矩与有限步功差按§4单列。

### R04：无复位的卸载→再接触

沿R01，初v=+1/2 m/s、c0。前1/8s外载0，随后固定COM外载−4z N（不是电机effort），最多继续3/4s，首个真实重新出现正入步N的接受步后停止；不重定位、不改stiffness、不按预期时间换接触标签。独立1D递推为 `delta=max(0,1/32-q)`, `N=64*delta`, `v+=v+h*(N+external)/2`, `q+=q+h*v+`。

记录最后正载、首个分离entry、再次正载entry及其两侧材料地址，分离阶段N/接触冲量必须0；末态几何出现接触与下个入步采力分开标时刻，不能称TOI。若物理轨迹未在预算内重入，保存真实结果，先查实现/离散域，不能延窗或调控制后倒算通过。h/2使用同物理切换时刻和原输入。

### R05：W1/W4隔离、同局部ID与事务

W1单跑R01–R03及空几何非零驱动树，再合为W4，keys候选(17,103,7,409)分别对应R01/R02/R03/空几何P；空几何P沿R01模型但ff=1 N、其余载荷0，独立a=1/2。局部body/shape ID重复，登记顺序反转。按world/name逐步比较原输入和完整接受态、site力/矩、审计与ticks；不从混合批读取结果反造单world oracle。

对其中一个已知正载world预设site_capacity=0：其完整树/队列/tick保持，其他world继续；增容后从未消费的同态同输入重试。另注入：非finite site、错source/epoch、第二次buffer分配失败、载荷上传/求解/同步/末态读回/receipt构造失败。设备全局失败不得发布任何world；可恢复时同输入重放一致，无法drain则沿旧unavailable要求显式从存档重建。close两次、关闭后读、跨batch plan均拒绝或按原幂等规则处理。

零接触和空model必须保真实原运动；每次成功前后证明两个state bank均被实际使用。返回records为owned，无外部数组别名；观察不推进clock。原最新tree诊断失效策略与失败计数不误写成“所有观察字节完全不变”。

## 6. 父需裁定的文件/阶段与停止线

首片建议NEW `_resident_contact_plan.py` 管配置/作用域/收据，NEW `_resident_contact_bridge.py` 做纯HOST patch→owner载荷适配，NEW专项测试及一个公开声明例程；无独立求解器或平行Scene对象。父拥有旧 `tyx_world_batch.py`、`_resident_collision_plan.py`、`_resident_tree_storage.py` 和原树kernel的最小分拆/载荷注入；需要device散回时才批准小kernel，不能提前写一个完整新contact backend。

1. **接口/有效RED**：已成功建普通树、真实入步2N旧法向正控后，调用缺失的公开contact入口；先冻结receipt时刻/载荷口径及R01–R05。不得拿import失败代功能RED。
2. **最小真实反馈**：R01普通P与R03偏心R单步，原输入/完整state/静态反力/独立oracle；然后R02全跨支和W4。先HOST独立参照，再interpret/真实CUDA，两种模式各自保实际launch/传输。
3. **连续功能与故障**：R01/R02/R03短窗、R04卸载重入、W4失败恢复及h/2；真实第一失败保留。只有受影响旧零接触/控制/恢复节点需要相邻兼容，不重跑旧整套。
4. **发布**：完整source/参数/输出绑定、源码外wheel调用、同输入同输出成本观察；没有证据不宣告性能收益或默认切换。

后继分开裁定：moving/moving需要完整同worldowner/pair域与双方状态共同接受；affine需要C5完整物理行与P投影接触/原行反力，不删从动实体；persistent tangent复用现station/history/全cross W与真实motion/work门，不能把法向episode账本叫静摩擦；多树/自由刚体要扩当前“一world一law/tree”的owner布局；floating/full rigid6D还需真实root/COM/惯性frame和root–joint全交叉设备状态。以上不随首片验收自动通过。

停止无效绕圈：当前接口无法保同一入步J/M/速度或单发布时，先交精确缺口，不以两个完整step拼接；同一独立输入重复暴露不支持几何/FTZ/残差时，保首失败后只修明确实现原因，不调物性/采样/门；没有明确收益不做第二轮同义优化。本文所有接口、案例和新能力仍 **NOT_FROZEN / UNEXECUTED**，待父单独授权tests-first。
