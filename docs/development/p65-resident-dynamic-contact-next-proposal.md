# 常驻 dynamic/dynamic 接触：下一片作者提案

状态：**AUTHOR_PROPOSAL / NOT_FROZEN / UNEXECUTED**。2026-11-08，仅源码调研与本文件；不是实施授权或新验收。
源码取父 `8fe797c99d46c1b33b37777aed2de822aa32ada0`，C7完整矩阵仍由父串行验收；本提案不重审或重跑该矩阵。
`P = C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`。
`N = P/.venv-newton-1.7.0/Lib/site-packages/newton`；本机distribution metadata为Newton1.7.0、Warp1.17.0，未导入运行。

## 1. 建议先做什么

推荐首片：**同一world、同一真实完整ordinary/affine树（可含多个固定根分支）内，两个不同非相邻动态group的有限接触**；可与原dynamic/static对在同一步共存。
沿原sphere、finite capsule、unrounded convex hull、mu=0、discrete_entry和当前shape/material/source能力，不增加第二solver/runtime，不做跨world配对。
不是将多个世界改同key或把树拆成独立one-R；一个world仍只有一个完整RobotModel及其全M/b/J，affine保同一原始P/c/C图。
固定森林已是实际模型域：`P/src/phymeo/backends/_resident_tree_world.py:263–288`逐个真实root减共同anchor，保parents、安装、COM、惯量；不需要dummy关节或虚拟合并质量。

| 方向 | 可复用与实际缺口 | 建议顺序 |
| --- | --- | --- |
| 同world同完整tree/forest，含affine | 全N mechanics、K solve、设备FK和双bank均已有；宽相与HOST桥目前主动拒绝dynamic/dynamic。现逐侧wrench和端点运输已足够表达双方 | 首片，边界小且直接增加树自接触/支链互碰能力 |
| 同world多独立tree + 自由刚体 | 需要world→participant集合、每participant坐标/控制/owner映射、同world整组接受位；目前world key只能绑定单spec，FreeSphere仅平移 | 后续架构节点；不能通过删duplicate-world检查或复用别的world slot交付 |
| 浮根驻留 | HOST有完整浮树动力学与场景，设备没有7+n姿态/6+n速度、root COM/惯量frame和姿态漂移bank | 独立设备能力节点；先有真实浮根接受态与FK，再接同一接触桥 |

## 2. 当前源码依据

- `P/src/phymeo/backends/_resident_contact_bridge.py:242–267`保key/world/group及同group一致stored frame；`:348–355`唯一dynamic/static域门要求恰一侧静态。
- 同文件`:383–413`已经用双方COM线角速率求共同作用点速度，各侧按真实group累加`[F,(p-COM)×F]`；`:424–490`逐侧运输入口材料点和作用点，均不重采law、不积分。
- `P/src/phymeo/backends/_resident_collision_plan.py:375–412`保world/body↔真实group一一对应及static不冒用moving body；`:431–447`当前所有tree dynamic/dynamic（包括同body多shape）在空间过滤前均须显式excluded，正常域只接受动静异或。
- `P/src/phymeo/geometry/_collision_spatial.py:15–21`已有真实world、同rigid-body、group/mask、excluded过滤；没有按父子关节或名称判邻接。后述父子限制是建议新增的首片边界，不能描述为现有自动邻接过滤。
- `P/src/phymeo/backends/_resident_contact_plan.py:425–458`已在一次entry mechanics之后加入接触，再选ordinary solve或affine原投影/K solve，随后candidate FK/finalize；无需修改本构或改成末端kick。
- `P/src/phymeo/backends/_resident_affine_storage.py:238–325`复用full mechanics、P投影、原K代数solve、full展开/原图反力和统一retain；`P/src/phymeo/backends/tyx_world_batch.py:478–503`在所有检查/receipt之后唯一swap。
- `P/src/phymeo/backends/tyx_world_batch.py:100–109`明确每world一个spec，ordinary/affine存储只是不同world的布局，不是同world多个参与者接口。
- `P/src/phymeo/dynamics/floating_robot.py:15–69`已有HOST root inertia frame、7+n/6+n状态；`P/src/phymeo/solvers/floating_tree.py:142–186`已有完整M/b、COM外载投影和真实姿态漂移。此事实不能替代驻留实现。

## 3. 最小公开接口与所有权建议（待父裁定）

沿用 `batch.prepare_contacts(model, bindings, materials_by_pair, ..., tree_dynamic_pairs=False)` / `batch.step_contacts(plan,h)`，建议只加一个显式opt-in布尔门，默认保旧行为。
`ResidentTreeBinding` / `ResidentAffineTreeBinding`、CollisionKey、CollisionModel.excluded、材料map、checkpoint、receipt和stats继续复用；不新增一套pair名单或接触law。
启用时，dynamic/dynamic必须同world、同一真实spec、不同group；ordinary与affine两个storage不能用同world相互拼接。两侧binding各自保完整源身份，不能仅靠shape.kind推owner。
同body多shape沿原空间层same-body flag不互撞；放宽现resident总拒绝门时须保留该过滤，不能将同body绑定成两个动态group。直接父子首片建议仍要求显式excluded，缺失时报能力边界，不暗中吞掉接触。
邻接仅由真实full模型的`parents[a]==b`或`parents[b]==a`判断；共同`parent=-1`不使两个root相邻，affine的P/C联动也不改物理body/邻接身份，folded工具按实际owner group。非相邻分支须有显式law；group/mask和excluded仍权威，材料map不得含已排除pair或跨worldpair。
同group的多个不同表面可同时与另一group接触，每个实际site计权一次；group/body映射冲突、重复pair/site来源、static伪装dynamic或graph不匹配先拒绝。
建议宽相一般入口也接受同名opt-in，并共用父的pair域判断；若父希望首片只暴露于contact plan，基类仅增加内部域能力参数，不能在query时临时篡改exclusions。

父OLD所有权：
1. `_resident_collision_plan.py`：验证同spec/不同group/邻接策略，放行明确动态pair到原BVH；current-pose非CCD，设备bounds核无需重写。
2. `_resident_contact_bridge.py`：替换恰一侧static门，显式拒绝同group/static-static；复用现逐侧wrench/finish循环。
3. `_resident_contact_plan.py`、`tyx_world_batch.py`：薄opt-in传递及原receipt/source声明；沿原一次stage与swap，不修改M/PD/affine数值核。
若必须改原物理核，先给可复现依赖再冻结新范围；不能为了省事补第二次推进。

未来作者NEW候选（本轮均不创建）：
- `tests/scale/test_resident_tree_dynamic_contacts.py`：独立几何/物理/多world窗口。
- `tests/scale/test_resident_tree_dynamic_contact_transactions.py`：整world容量、晚故障、restore与owner负控。
- `tests/reference_newton/robot/resident_dynamic_contacts_reference.py`：少量明确匹配的原生双动态读数。
- `tests/wheel_consumers/resident_dynamic_contacts.py` 与 `docs/benchmarks/p65-resident-tree-dynamic-contacts.md`：源码外混合窗口与诚实结果。

## 4. 物理与事务合同

每site的canonical normal n指first→second，共同作用点p，法向幅值N≥0、patch面积权重w只乘一次：Fa=−wNn，Fb=+wNn。
两侧真实COM分别ca/cb；tau_a=(p−ca)×Fa、tau_b=(p−cb)×Fb。每side先入完整N×6 COM wrench，再由同次全J散回：
`g_contact = Jv_a^T Fa + Jw_a^T tau_a + Jv_b^T Fb + Jw_b^T tau_b`。
因此等价于`(Jpoint_b−Jpoint_a)^T(wNn)`；共同祖先项不能先删，M的交叉块也不能拆开。两侧速度来自同一entry q/v的完整J，不能拿candidate速度或只一侧速度算阻尼。
ordinary按原`M a = drive + input_external + g_contact − b`；affine用`P^T M P`及`P^T(input_external+g_contact−b)`，在原K bounded PD前加入，保持原图`M(Pa)+b−E drive−external−g_contact=C^T lambda`读数。
接触只写临时wrench，held/full loads与drive delay不变；同一力不能同时写held和临时wrench。entry力、实际有限步功、KE变化分列，不加严格能量守恒门。
一对动态接触的合力与关于同一世界参考点的合矩为零；固定基座可以施加约束反力，所以不能无条件要求整棵固定树线动量守恒。下面D01仅对其自由平移方向检查净动量。
`static_reactions`仍只属于真正static shapes；动态两侧由原sites/full wrench表达，不伪装固定支撑反力。
site帽对同world所有动动+动静patch一次合计；溢出整world所有N/K/FK/energy/held/history/tick/旧receipt保留，ordinary码7与affine码9不混原affine残差7/8；别的world可推进。
candidate帽、未知几何、driver/上传/候选/receipt晚异常仍全batch无swap。全部receipt、双方实际material/application位移和状态来源检查成功后才统一发布；不增加跨world接触或跨world反作用。

## 5. 建议预注册独立案例（数值为手推，尚未执行）

公共默认：h=1/1024，gravity=0，kp=kd=0，drive=0、delay0，effort cap8、迭代512；normal k64 N/m、d0、mu0。除明确写的初速外q/v为0。
接触几何是指定刚体的有限碰撞表面，mass/COM/I为单独物理声明，不从碰撞球半径反推均匀球惯量。全I须正定且有可核的主轴。
首步沿原ATOL5e-6、RTOL5e-5，40步ATOL2e-4；stored frame材料/作用点运输1e-10 m仅记录构造一致性，旋转域仍5e-5，M与PD原门不变。

| ID | 固定输入与独立预期 | 能击穿什么 |
| --- | --- | --- |
| D01 两P森林 | parents=(-1,-1)，两轴+x，mass=(2,1)，COM=0；r=(1/8,1/8)，中心(0,0,0)/(7/32,0,0)。gap=−1/32、力=(-2,+2)x、M=diag(2,1)，a=(-1,2)，v+=ha，q+=h²a，2v1+v2=0。40步用距离7/32+q2−q1独立递推N与半隐式q/v | 只施一侧、加两次、world/row错配、去掉真实惯量 |
| D02 真共享祖先R-P-P | parents=(-1,0,0)，轴(z,x,y)，mass=(1,2,1)，COM皆0；root Izz=1/4，孩子各Izz=1/8；孩子安装(0,1/4,0)/(7/32,1/4,0)，两个孩子球r1/8接触。M见下，g=(0,−2,0)，a=(−8/9,−11/9,7/36)。整体Rz(.37)及xyz循环+平移(.5,−.25,.75)保广义结果；注册反序不改来源 | 漏双方祖先投影、只用M对角、把fixed base反力当动量错误 |
| D03 偏心双R森林 | 两Rz pivot(0,0,0)/(7/32,0,0)，local球心均(0,1/4,0)，r1/8，COM=pivot，mass=(2,1)，Izz=(1/4,1/2)。首力±2x、COM矩(+1/2,−1/2)z、a=(2,−1)。40步用两旋转球心/距离和各pivot力臂独立递推，保存实际弧位移与功 | 零力臂、错COM、以hJv冒有限弧位移 |
| D04 非单位负gain+offset | D01质量；q=(s,−s/2+1/8)，第二安装x=3/32，初s=0，P=(1,−1/2)。Mr=9/4，Pᵀg=−3，a_s=−4/3，full a=(−4/3,2/3)，C=(1/2,1)，lambda=−4/3；40步距离7/32−3s/2独立递推 | 只给follower或driver单侧力、把gain取绝对值、错误约束反力 |
| D04Z 同速约束零K载荷 | D01质量与同一7/32初始中心距离；q=(s,s+1/8)，第二安装x=3/32，P=(1,1)，Mr=3，full g=(−2,+2)，Pᵀg=0，a_s=full a=0。C=(−1,1)，full reaction=(+2,−2)，lambda=−2。再给独立速度1/8：两侧等速移动、距离和2N载荷不变、约束功0 | 按K净载荷0丢完整contact/原图反力；运动不变不等于无接触 |
| D05 原链/等价flatten | 三P根mass=(2,1,1)，P=(1,2,−1)，c=(0,1/8,1/8)，仅group0/2为D01位置接触。g=(−2,0,2)，Mr7，a_s=−4/7；原链行(q1−2q0−1/8,q2+q1/2−3/16)的lambda=(−3/7,−10/7)，flatten行(q1−2q0−1/8,q2+q0−1/8)的lambda=(−8/7,−10/7)。两者状态相同，wrong-graph restore拒绝 | 丢原图身份、用flatten反力伪装原行反力 |
| D06 有限形状与权重 | D01改两halfsize1/8盒面，gap−1/32、全部site权重和1，总力仍2N。sphere–capsule两r1/8、cap轴z端点±1/4：sphere在(7/32,0,0)测轴部；(0,0,1/4+7/32)测有限端；(0,0,1/4+9/32)为gap+1/32零力。两侧沿本例normal轴活动，质量2/1 | 无限轴错误、面site数翻倍、端点/skin/材料IDs混用 |
| D06H 既有线/非盒面代表 | 复用A10原有限CC（各端点x=±.5、r=.05、第二中心z=.09）与非盒HH（下述原五边环挤出z=±.1、第二z=.19），均原gap=−.01。两侧Pz、mass=(2,1)，k64，line/area全部site权重和1，首总力16/25 N，a=(−8/25,+16/25)。保原IDs、完整双侧wrench/材料点，不能用OBB替代H | 动态线/面多站点重复施力、丢一侧COM矩、用包围盒代真实H |
| D07 相对速度/过滤/多形状 | D01共同+x初速1/8、d4仍2N；相向(+1/8,−1/8)为3N；分离初速(−1/8,+1/8)为1N。同body多shape、父子必须excluded负控、law/excluded冲突、同ID异world共址零跨world力。两对相对z=±1/8的显式球pair各2N，交叉pair显式excluded，总4N且力矩相消 | 只读一侧速度、重复接触、暗中跨world或body别名 |
| D08 事务/混合 | W1/W4非连续keys/逆序，D01/D04+原静态接触world+未绑定scalar；40步，两个active bank分别做合法restore/replay、容量1不足盒patch、candidate帽、第二真实scratch分配/上传、candidate/receipt晚故障。输入/数值门不变，完整所有bank与旧owned receipt比对 | 一个group拒绝而另一侧推进、半步回执、row=slot、资源/epoch错配 |

D02独立矩阵为 `[[753/1024,−1/2,7/32], [−1/2,2,0], [7/32,0,1]]`；消去两P后root有效惯量9/16，第一行残量为`(9/16)a0+1/2=0`，故上表a成立。
D05 full反力为`(6/7,−8/7,−10/7)`，两套lambda都须各自满足原Cᵀlambda与Pᵀreaction=0；不能由产品诊断反写此预期。
D03双方实际work按同一入口力乘各自真实stored末态运输位移；同时单列entry power×h和KE，不要求三者逐项相等。
D04Z须同时检查正载sites、两侧full N wrench、非零原图lambda及零K接触投影；零q加速度本身不是验收，负gain D04仍独立保留。
D06H形状直接来自`P/tests/collision/convex_family_cases.py:17–28`：H环为`((−.5,−.1),(.5,−.1),(.6,0),(.3,.1),(−.5,.1))`，各点挤出z=±.1，真实10顶点。A10十组合的既有HOST几何接受只复用；本提案仅新增代表性圆/有限线/面/非盒H的双动态载荷与推进，不重跑十组合同源几何组。
D08不重复C7完整矩阵；只保新增双动态负控及少量受到pair域修改影响的原动静控制。首有效RED应先证HOST两动态patch/±2N和原full模型，再触新opt-in缺口。

## 6. Newton1.7参照与不能同义的部分

源码依据均为本机pin，不把未执行脚本当结果：
- `N/_src/sim/state.py:123–145`：public body_qd线速度在COM、后3为世界角速度；body_f为COM世界力/矩。
- `N/_src/sim/collide.py:148–154`：shape_body分别得到两真实owner，contact表面点转成各自body-local；不能用shape ID当body ID。
- `N/_src/solvers/semi_implicit/kernels_contact.py:423–478,485–558`：原生接触双侧owner/COM力臂、相对点速度和等反身体载荷；`:452–459`为材料override，`:521`阻尼只取接近分支。
- `N/_src/solvers/featherstone/solver_featherstone.py:591–604,717–746,767–775`：复制本步body_f外载，用body_qd_fk及COM通道加contact，再进入原articulation joint torque；不是独立body外力后伪FK。
- `N/_src/geometry/broad_phase_common.py:209–215`用真实body索引过滤同一非static body；`:256–282`拒不同普通world。Newton另有world=-1的global特例，本Phy提案不引入该特例。
- `N/_src/sim/builder.py:2184–2198,5513–5528,5681–5688`：显式shape过滤、父子默认过滤、joint parent/child必须同world。两实现默认过滤不应假定相同，参考脚本须保存实际表。

先只为D01/D03作独立native CPU/CUDA小窗：真实两动body/joints/shapes、明确关闭无关pair、c=0/mu=0、相同k和初态，保存原shape/body/world/normal/point/margin与完整两侧力/步后状态。
原生Newton多点manifold不保证与Phy面积求积相同；盒面总力和行数须分开观测，不把原生每行k叠加当同律通过。接近/分离阻尼也不全同义，D07分离预期不能套Newton原生。
可另用相同entry见证与同律外载施给Newton自身状态，关闭对应native contact防双加；该通道明确标matched-law integration，不冒native detection/law等价。不为追齐改原尺寸/阈值/力帽。

## 7. 后续架构节点与放行条件

第一片通过独立首步、40步、双bank/owner拒绝、两backend、保存审计及新wheel后，才可称“单驻留完整树内双动态接触反馈”；不能称任意多机器人或自由体交互。
多独立participant同world的先决条件：独立participant键到storage row/shape owner的注册、world→多个row的统一状态/controls/checkpoint、任一参与者拒绝则整world保留的接受mask；同一world同时取样、各物理模型仅推进一次，仍复用当前batch runtime与最终swap。
仅固定普通森林可以显式组成一个真实RobotModel，不等于已解决不同program/domain/affine图与自由体的自动组合；跨world交互永远不作为绕行方案。
浮根设备节点先迁移现HOST root COM/世界ω/惯量frame、完整6+n M/b/J及四元数漂移到真实bank；补q与v宽度不同、restore/延迟/接受FK和几何owner。不得用6个虚拟关节或HOST FK上传冒驻留。
C6/C7桥已可消费每body真实frame/COM速度，因此上述后续节点可复用几何与双侧载荷收据；必须先解决设备状态与world参与者事务，不能仅放开pair门。
本轮只有本提案；所有新增功能、源码修改、测试数量与执行顺序均待父裁定。未运行import、数值、测试、静态、build或native程序，未写OLD文件。
