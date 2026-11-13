# 2026-11-13 功能验收矩阵：场景、机器人组合、规模与保留的支撑失败

本页是**当前事实截点**，按父任务实际源码结果、已结束命令及安装消费者的小摘要更新。
最新已验安装产品为 `e1401fbf22396e802af18c3b8d7b6d1187d6ffbe`（G2b-A）；536成员同源，W32/B32包外interpret12步与CUDA256步均STRICT_PASS。BALL-S d7105a9d、原M-B2 6b4a639c及23757ae6/08d7a1b6/ec0e2997资格按原结果复用。本文仅文本/file IO/Git，
没有执行项目、读取大轨迹、重做保存审计或产品字节核对。以下结果引用原父收据，
后续新应用不得自动继承。整个 P6.5 / Newton 功能对齐仍未完成。

状态分为**公共功能、指定源码资格、实际后端、安装包、完整应用、性能**。
`FUNCTIONAL_PASS` 与保留的 `STRICT_FAIL` 可以同时成立；不能合成无条件 PASS，
也不能按测试数、world 数或清单数给整体完成百分比。

## 一、当前公共能力与实际资格

| 公共功能 / 路径 | 源码资格 | interpret / CUDA | 安装包资格 | 完整应用与限制 | 性能 |
| --- | --- | --- | --- | --- | --- |
| **G2b-A shared CUDA payload与W32/B32** | 新38＋受影响26＝64项通过；六solver-bank实际分配、精确/少1byte、UNKNOWN、晚桶失败/retry、clone/reset/restore峰值 | HOST/interpret40、CUDA24；四组32真实物理构型短窗严格通过 | e1401fbf新包536成员同源；interpret384、CUDA8192worldsteps均STRICT_PASS；正常关闭 | 257帧实际CUDA视频；固定/FREE R/P/B且无接触。32cap，超过旧8/16需显式预算；非全进程资源、任意接触组合或跨world碰撞 | 同source分桶，未融合kernel；暂无Newton公平比较 |
| **BALL-S 球树标量限位与DOF干摩擦**：fixed/FREE BALL/R/P、NONE/FILTEREXACT、joint-only或原condim1/3共用完整T4 | 22 HOST＋55 interpret＋55 CUDA＝132唯一节点；公共checkpoint修复另验旧8项；原run15失败保留 | 两后端十二连续窗口、十八原XML与完整事务通过 | 新包534成员同字节；24/25各224batch steps/448worldsteps PASSED，正常关闭、strict installed origins | scalar上下侧/margin/释放、dry滑/粘/零、三类行共同求解、反转卸载；无BALL角限位/G1长舞/BALL刚柔/扩容 | 未比较Newton |
| **D5-A 公共刚柔声明**：`ParticipantSceneDeclaration(contact_response="deformable_normal")`＋`ParticipantDeformable` → `compile_participant_scene` → 唯一 D3 owner → step/query/reset/restore | 新源码26节点/后端，旧受影响5/后端；坐标修复后资产短窗另各2通过 | 两后端源码及两批安装均通过 | 首包81/74，第二包132/133均 PASS；每次38 calls、57 worldsteps、6显式当前query | 此首片验证资产球/程序胶囊短窗、每world单体；多独立单体后继见M-B1，仍无任意关节机器人或持续稳态资格 | 有HOST载荷/诊断往返；未比较Newton |
| **R3-N FREE＋BALL法向接触**：原FREE根＋BALL/R/P、NONE JOINT motor、primitive condim1，完整M/J/积分与力运动共同发布 | 完整55节点＋7个原兼容节点通过 | 两后端包含在原固定源码矩阵；首包75/76均 PASS | 首包每后端56 calls /112 worldsteps；未冒称第二包重跑同一N消费者 | feet/siblings各24步非零速度、正反控制与卸载，另生命周期后继；非人形长时支撑/舞蹈 | 未做同模型同质量性能比较 |
| **R3-P FREE＋BALL pyramidal接触**：复用原约束求解与CSR，primitive condim1/3 | run88完整69通过：31 HOST、19 interpret、19 CUDA；run89原路由兼容4通过 | 新源码及第二包130/131均 PASS | 第二包两后端各56 calls /112 worldsteps，完整native同入步＋独立连续参考 | 正控为condim3；mixed1/3仅编译路由资格。本原R3-P批不含activation；R3-A另验。limits/dry、condim4/6、elliptic3、更多shape仍未放行 | 无Newton对比；不是新求解算法 |
| **R3-A FREE＋BALL FILTEREXACT**：混合NONE/FILTEREXACT JOINT、no-contact/normal/pyramidal与同次act/force/motion | run160完整117通过，run153受影响兼容6通过 | 两后端完整新源码矩阵通过 | 第三包08d7a1b6/532成员同字节；164/165各176 calls/352 worldsteps PASS，cleanup空 | 六连续族＋四原精确XML＋恢复/reset/晚失败；仍无limits/dry、多FREE根或任意机器人组合 | 未比较Newton |
| **G2a 原容量内异构功能规模**：完整来源分桶，同一次接受所有桶的力、运动及生命周期 | run37六个W1节点＋run51其余20节点全部通过；包含W8/B3、W16/B3、W16/B8、规模晚失败/retry及W17/model9前置拒绝 | 两后端完整源码格通过；包外仅指定W16格追加资格 | 首包77/78为W16/B3、79/80为W16/B8，四次均 PASS，各27 calls /432 worldsteps；**W1/W8只保源码资格，未新增包外运行** | cap仍8模型/16world，同h独立world；不是单异构kernel、跨桶碰撞或W32/128/256资格 | 分段walltime仅观察；live/peak设备内存UNKNOWN，未比较Newton |
| **D6-A 可选轴向材料速度耗散**：elastic reconstruct后一次有序隐式速度pass，候选验证后共同发布；D1/D3/D5可声明 | 核：102解释33（11 HOST＋22 TYX）、103 CUDA22；D1共35唯一节点，解释公共＋HOST34分组通过，111 CUDA29含UNKNOWN边界；D3 116/117各4；D5 121/122各2 | 新耗散/算术/能量/事务与连续工程门均通过；**104原动量strict、112原elastic速度strict保留FAIL** | 第二包128/129均 **FUNCTIONAL_PASS**，各22 calls /44 worldsteps；同时 `strict_elastic_precision=FAIL_RECORDED`，每后端20项旧短门记录，最大速度差约1.253865e-4 m/s | pair/chain及sphere/斜capsule短窗，真实恢复/reset后继；轴向一阶分裂，有平膜法向零模，不是全网络BE、全局drag或D4稳态合格 | 无性能/全布模型结论 |
| **D4 原无阻尼N1持续法向载荷**：完整入层→承载→卸载→离层→再接触 | 原strict run46第301步速度门FAIL；runner、独立算术审计及诊断路径可用 | run50解释、run69 CUDA均完成原3072步/3秒，但均 **FAIL / DIAGNOSTIC_ONLY** | 没有已通过的完整持续支撑安装资格；不把D5/D6包外短窗代替它 | 两预定稳态窗的支撑力与速度RMS不合格，**不是仅精度问题**；其余原三格未执行 | 原始两窗有计时和保存图；无公平性能比较 |
| **D6-B 新c=4096持续载荷应用** | 短合同140解释14及141 CUDA单返回故障1通过 | 143解释完整3072步/3秒后 **FAIL / DIAGNOSTIC_ONLY**；其余格按协议停止 | 第二包完整N1解释失败，力/运动及连续参考逐步门完成，但稳态均值/RMS不合格 | 两窗21.4613/19.3665N，刚体RMS .3044/.2136m/s；不扫系数，不改原D4结果，详见[实际结果](p65-damped-normal-load-result-20261113.md) | 无Newton比较 |
| **M-B1 多独立刚体—变形共同响应** | 186解释16＋187CUDA12＝28唯一节点；199/200容量各2；旧兼容171/172各8 | 两后端均通过，S8实测4world/8participants、M3混合1/2body world | 第四包191/192各44worldsteps/88participantsteps；201/202容量各24/44，全部PASS及关闭成功 | 每participant仍ordinary floating单body，sphere/capsule法向；不含关节树、RR新增响应、摩擦/CCD或稳态 | 未比较Newton |
| **M-B2 非根关节树—变形共同响应** | 新13解释15＋14CUDA10＝25；受影响旧路径06/07/15/16共39通过 | R/P串联末端胶囊、分支两连杆受载，两world/非零入速/控制变化/连续八步/完整事务通过 | 新包534成员同字节；20/21包外各40worldsteps/40participantsteps、三个owner均关闭成功 | ordinary FloatingRobotTree；每world一个关节participant、完整B×6与J/M；可混旧M-B1 world。无多机器人同world、BALL刚柔、摩擦/CCD或稳态资格 | 有HOST往返；未比较Newton |
| **G2-R0 restore候选资源归属** | 178预期RED后185完整12GREEN | 两后端真实晚失败、保原发布、恢复重试与连续后继 | 第四包193源码外12通过；原异常＋cleanup聚合、保留owner并真实释放 | 不注入/重试UNKNOWN；不是完整资源预算或扩容 | 不改变数值核 |

D5减少公共装配成本，M-B1扩多个独立刚体，D4检查持续功能，D6新增明确材料模型，不能互相替代。各R3历史行“不含limits/dry”是原首片范围；当前后继BALL-S已覆盖上述限定组合，不再把它列为整体未实现。
R3/G2a/G2b-A属于MuJoCo-TYX路径；其支持表不自动与resident刚柔路径作笛卡尔积。G2a行中的8/16与UNKNOWN是该历史首片范围；当前solver-bank已登记payload及32格资格见G2b-A，总进程资源仍UNKNOWN。
D6的D5带耗散声明已有源码资格；本轮D6安装消费者直接消费D1/D3，D5安装消费者
验证原场景路径，**不额外声称专用带耗散D5包外矩阵已执行**。

## 二、分批安装版本与实际消费范围

| 包 / 来源 | 原父字节资格 | 本页引用的已结束消费者 |
| --- | --- | --- |
| 首包 `113629b3` | 530个产品成员 source/wheel/installed 同字节；原父资格保留 | D5 81 interpret /74 CUDA；R3-N 75/76；G2a W16/B3 77/78、W16/B8 79/80 |
| 第二包 `ec0e2997` | run127，532个产品成员完整比较，`BYTE_IDENTICAL`、issues空；wheel SHA256 `48cc7e3938a7ba511ebce7e2efe04dfbc5b9b4d8163ab71edb6f91ba1932fb14` | D6 128/129 FUNCTIONAL_PASS＋strict FAIL_RECORDED；R3-P 130/131 PASS；D5 132/133 PASS。六次RAW_EXIT均0，结果标签仍分别保留 |

第二包D5两后端摘要均为38 returned/checked calls、57 accepted worldsteps、6 queries、
57 checked sites、cpu_physics_calls=0。D6各有4次restore、4次reset、2次stale拒绝，
其44 worldsteps来自真实后继；20项旧弹性精度记录没有从日志删除或变成PASS。
R3-P每后端feet48＋siblings48＋生命周期16 worldsteps。G2a每次W16格完整24步
及3次边界后继，没有抽样world或截短；源码六格共1161 worldsteps/后端不冒充安装总数。

所有消费以源码外脚本/cwd、实际安装根和完整模块来源为前提，保原输入、返回、
state/receipt、dtype/shape/字节、负控及cleanup结果。原日志顺序是attempt→实际
返回先存→观察/检查；post-return观察失败不能称物理未提交或回滚。
新包资格是这些指定消费者的有限闭环，不是全部历史功能重新测试。

## 三、保留失败与不能混淆的数值账目

D4原门仍见[完整协议](p65-deformable-normal-load-protocol.md)：
两个稳态窗[0.75,1)、[2.75,3)，支撑力均值需12.8–19.2 N，刚体与中心各自
法向速度RMS≤0.1 m/s。实际：

| 原N1结果 | 两窗支撑均值 N | 刚体速度RMS m/s | 中心速度RMS m/s | 结论 |
| --- | --- | --- | --- | --- |
| run50 interpret | 24.8393461 /5.41425136 | .584440 /.477767 | .340356 /.384283 | 稳态功能FAIL；另10项原短速度失败，最大2.69912155e-5 m/s |
| run69 CUDA | 24.8393962 /5.41416865 | .58443762 /.47776549 | .34035643 /.38428665 | 稳态功能FAIL；另9项原短速度失败，最大2.85153656e-5 m/s |

两条完整原轨迹中的入层/释放/重入、力/几何、stored arithmetic、长参考、原侧、
防贯穿与峰值门完成，不抵消上述失败。解释run50在生成载荷subnormal修复前，
CUDA run69在该窄修复后，不能把两者伪称相同产品快照。原另三格保持未验。

三类数值记录必须分别保留：

- 原elastic predictor/ordered-position/reconstruction舍入余项有独立先验界；
  它们解释实际离散算术，不改原材料请求冲量、状态或物理力。
- 设备生成接触载荷仅FP32 subnormal→同号零；raw/delivered/loss/bound完整，
  正常值逐字不变，外部输入guard不放宽。这不是添加材料耗散或内部能量补贴。
- D6 requested damping impulse与实际速度冲量分列；固定反力保原外/接触载荷、
  elastic与damping冲量负和；实际动能、理想离散功、有符号舍入差不相互抹掉。
  原run104/112及安装strict记录保留，`FUNCTIONAL_PASS`不宣称精确守恒或旧门通过。

D6-B的c=4096 N·s/m是另行预登记的物理参数；NormalLayer接触damping单位仍s/m。
原N1不加此参数、不改统计窗/质量/h来追绿，新应用必须用独立case与结果标签。

## 四、有限范围、故障资格与后继

| 范围 | 已验边界与未完成部分 |
| --- | --- |
| D5/D3/M-B1/M-B2刚柔 | 同world多个独立ordinary单body，或单个ordinary R/P关节树＋一个节点模型，sphere/capsule–triangle、入步facet NormalLayer。M-B2实测两world、3/4body树；旧M-B1容量资格保留。HOST往返存在；多关节机器人同world、BALL刚柔、面积压力、摩擦/换面历史、CCD仍缺 |
| R3机器人 | FREE保nv≤35/nq≤45、fixed nv8/nq10、body128、actor32、geom128/pair16/rawslot32及原solver预算；不以cap常量证明全部极限组合。FILTEREXACT由R3-A、scalar limits/DOF dry由BALL-S另验；BALL角限位、更多transmission/shape仍是后继 |
| G2a异构 | 原完整来源分桶与末桶status/download/aggregate失败、retry及W17/model9前置拒绝已获源码资格；包外只公共正格和边界操作，不冒充重新注入私有故障 |
| D6材料 | 参数/边序/模型身份、None/全零原路径、fixed反力、late candidate拒绝、真实恢复与UNKNOWN生命周期有对应源码资格；不能把轴向零模或未验应用藏到“已阻尼”中 |
| 安装与资源 | 主错和cleanup错分开，UNKNOWN不宣称可盲重试；G2b-A已登记六solver-bank数组的共享预算及current/peak/free。该已知payload不是RSS或总显存：全进程HOST/native/context/JIT/外部owner仍未完整计量；公平Newton速度结论保持NOT_COMPARED |

后续按真实缺口登记独立功能合同，不重跑同源已绿组：先完成新应用的既定资格，
再扩多关节机器人刚柔组合、摩擦/历史/CCD，或BALL角限位等尚缺的通用组合；
G2b-A已按六solver-bank数组生命周期预算获得32模型/32world限定资格；超过32的异构规模（包括128/256）须另补资源边界和完整案例矩阵，公平性能比较另立阶段。
旧G1长舞、J/R学习、B-point表示及全布/体软体问题不会随本表关闭。

依据：[D5报告](p65-deformable-scene-report.md)、[R3-N](p65-free-ball-normal-report.md)、
[R3-P](p65-free-ball-pyramidal-report.md)、[G2a](p65-heterogeneous-scale-protocol.md)、
[D6源码报告](p65-distance-dissipation-report.md)、[D6用法](p65-distance-dissipation-usage.md)、
[D6-B协议](p65-damped-normal-load-application-20261113.md)。本轮原命令/快照/失败入口：
`artifacts/p65-feature-until-0730-20261113-01`；本页不替代其完整证据。

第三包R3-A正式范围详见[报告](p65-free-ball-activation-report.md#parent-acceptance-2026-11-13-0447)。M-B1已按[冻结合同](p65-multibody-deformable-implementation-20261113.md)完成限定源码及第四包安装资格，以下段落记录版本边界。

### 第三、第四包的后继范围

第三包08d7a1b6/532成员、R3-A164/165各352worldsteps已接受。第四包23757ae6/534成员，190 parity BYTE_IDENTICAL/issues0；新M-B1及容量191/192/201/202、R0故障193已接受。第四包没有重新跑完整历史功能；旧资格按对应版本保留。详见[M-B1报告](p65-multibody-deformable-report-20261113.md)、[用法](p65-multibody-deformable-usage-20261113.md)、[恢复清理报告](p65-restore-cleanup-report-20261113.md)。完整canonical结果在本轮evidence的installed-closed-archive-04/wheel-04。

### M-B2 后继包

产品6b4a639c，534成员完整 source/wheel/installed parity、issues0；wheel SHA256 `c8cdf8d182e120a8c687e9ded18408852135ff475ab57ce0f05b5d47b4e6d642`。新源码25和受影响兼容39全部通过，安装20/21各40worldsteps。完整结果见[M-B2报告](p65-articulated-deformable-report-20261113.md)，新canonical证据为 `artifacts/p65-articulated-deformable-20261113-01`。原D4/D6-B和历史strict失败保持；本包不冒充全部旧功能重新验收。

### BALL-S 后继包

产品d7105a9d，534成员同字节，wheel SHA256 `567a8e98bf93be01d38687b8e7ea3fbfc485bfc36c73181ef18582563c806c37`。新增132唯一节点＋受影响恢复8项，包外两端各448worldsteps/1181事件/332模块strict origins。run15暴露并修复receipt与accepted mirror输出不一致可恢复的问题，未改物理核或门限。完整[验收报告](p65-ball-scalar-constraints-report-20261113.md)及25份已闭命令在 `artifacts/p65-ball-scalar-constraints-20261113-01`。在BALL-S包的版本边界内尚无G2b资源预算，不由本包获得扩容或峰值显存资格。后继G2b-A已独立验收32模型/32world及已登记数组current/peak预算，仍不代表全进程总内存/总显存资格。
