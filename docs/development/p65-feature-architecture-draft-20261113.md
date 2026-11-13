# 当前功能连接架构：2026-11-13

状态：**当前连接架构，已加入M-B2、BALL-S与G2b-A共享设备预算/32构型后继**。最新安装产品e1401fbf，
资格依据为父已报告的源码结果和同次安装summary顶层字段。此更新只读取文本、
JSON与Git，没有执行项目、数值、保存审计或安装消费者。安装结果只授予对应
有限范围；D4持续稳态支撑失败、D6原弹性严格精度失败与待执行项分别保留。

现有[统一编译架构](../architecture/unified-scene-compilation.md)、
[续作入口](p65-continuation.md)与[全景](phymeo-newton-panorama-2026-11-12.md)
由父任务统一更新，本文件不覆盖它们。[本轮计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)
的07:30窗口也不表示整个P6.5完成。下图的三条runtime路径各有自己的来源、
准入和发布事务，不能把它们的支持表相乘成任意耦合。

## 1. 三条实际路径，不把同名step画成任意联合求解

~~~mermaid
flowchart TD
  Asset["URDF / 程序资产<br/>静态质量、惯量、collider来源"]
  OldDecl["旧 SceneDeclaration"]
  PDecl["ParticipantSceneDeclaration<br/>world / participant / asset / control / collider身份"]
  Asset --> OldDecl
  Asset --> PDecl
  OldDecl --> Old["compile_scene<br/>原HOST或resident受限场景"]

  PDecl --> Factory["compile_participant_scene"]
  Factory --> RScene["host_patch / none<br/>原多参与者ResidentSceneBatch"]
  RScene --> M1["真实accepted设备FK<br/>gather → bounds/BVH → 有限边界查询"]
  RScene --> RResponse["既有刚体材料接触路径<br/>完整J/M → 共同发布"]

  NodeDecl["ParticipantDeformable<br/>D1节点/距离材料/可选D6耗散/稳定point与facet IDs"]
  NodeDecl --> Factory
  Factory --> D5["deformable_normal<br/>静态lowering和原资产来源核验"]
  D5 --> Joint["唯一 TyxDeformableContactBatch<br/>同runtime节点与独立浮体或单R/P树owner"]
  Joint --> Dual["同入步accepted双来源<br/>节点bank + 刚体FK"]
  Dual --> D2["gather → bounds/BVH → sphere/capsule–triangle有限距离"]
  D2 --> Law["仅合法RESOLVED近接触<br/>原离散facet NormalLayer"]
  Law --> NodeCandidate["节点外力+接触力<br/>原integrate / ordered材料 / reconstruct候选"]
  Law --> RigidCandidate["反向COM wrench<br/>按真实group交完整B×6载荷<br/>整树J/M与完整root–joint候选"]
  NodeCandidate --> D6Choice{"显式非零 DistanceDissipation？"}
  D6Choice -->|是| D6["保存reconstructed v → 一次ordered轴向速度耗散<br/>设备status + 完整phase观察/验证"]
  D6Choice -->|None或全零，保持原路径| Commit["所有候选与完整receipt先验证<br/>一次共同发布两个accepted bank"]
  D6 --> Commit
  RigidCandidate --> Commit
  Commit --> Public["状态 / 力与矩 / 弹性与耗散冲量 / 固定反力<br/>checkpoint / subset reset / restore"]

  Native["原MJCF/URDF及assets"]
  Native --> Compile["CPU MuJoCo 3.3.7编译<br/>原模型/MJB/来源与坐标记录"]
  Compile --> Admission["MuJoCo-TYX显式模型准入"]
  Admission --> NativeTYX["原TYX动力学 / primitive接触 / 约束<br/>R3-N/P接触与R3-A FILTEREXACT明确分域"]
  NativeTYX --> NativeRuntime["MuJoCoRuntime<br/>controls / step / snapshot / checkpoint"]
  NativeRuntime --> Train["VecEnv / task-action设备接口<br/>随机reset / control windows / RSL PPO"]
  Compile --> Budget["共享solver-bank CUDA payload预算<br/>manifest / 预留 / 实际分配 / UNKNOWN保留"]
  Budget --> Buckets["MuJoCoTyxHeterogeneousBatch<br/>32 models/worlds；完整来源分桶"]
  Buckets --> Multi["每桶候选 / 全world验证<br/>一个异构聚合发布"]
~~~

图中没有 D3→MuJoCo-TYX 或 ParticipantScene→任意MjSpec 的边：当前没有
这些通用转换/联合求解。异构桶之间是独立world，不因同一次发布产生跨桶碰撞。G2b-A已验32无接触构型，solver-bank payload有current/peak，context/JIT/HOST/native等总峰值仍UNKNOWN，不能把预算节点当全进程资源管理。
M1有限距离也不会自动变成旧刚体HOST patch或MuJoCo-TYX primitive响应。

## 2. 声明和编译：新增的是可复用入口

旧compile_scene仍消费SceneDeclaration的既有共同域；它不是本轮多参与者
声明的同义词。多参与者入口是ParticipantSceneDeclaration和
compile_participant_scene，保留旧host_patch/none模式。

D5-A新增显式deformable_normal模式：

1. ParticipantArticulation保存资产、真实初态、控制、完整body/collider安装。
2. ParticipantDeformable保存D1模型、材料、初态、稳定point/facet与surface安装；
   各DeformableNodeWorld可显式声明DistanceDissipation，D5保留该声明与来源身份。
3. 静态lowering复用原资产/控制编译和完整model-token/source-shape检查；
   from_asset binding不会被去掉来源后伪装成direct binding。
4. 完整允许pair必须有NormalLayer或显式None；原mask/exclusion仍有效，
   不以实际当前接触输出反推材料或候选集合。
5. 直接构造一个D3共同owner，不先创建独立刚体runtime再顺序接节点step。

scene.identity绑定声明；source_map保原资产key到installed key以及原面/点身份；
revision是发布事务身份，不等于物理tick。初始held controls提交后保存公开
共同checkpoint，reset用该checkpoint恢复所选world两侧状态和初始控制。

主要源码：
[participant_scene_compiler.py](../../src/phymeo/builders/participant_scene_compiler.py)、
[deformable_scene_lowering.py](../../src/phymeo/builders/deformable_scene_lowering.py)。

## 3. 持久状态、查询与响应：明确哪些量在设备，哪些仍经过HOST

D1持久owner保存accepted/candidate节点位置、速度和时钟，原ordered距离
材料在设备推进；不是每步从HOST重建节点。固定节点、真实边长改变、完整材料
冲量和固定点反力均保留。查询通过同步的有期限loan借用真实accepted bank，
没有用HOST当前坐标回填来伪造设备来源。

D2/M2-Q从当次节点bank gather原面，在同runtime生成bounds/BVH和完整候选，
复用有限sphere/capsule–triangle距离核。当前open surface有稳定材料身份；
距离不是闭合实体的内外判定，也不是CCD。CONDITIONING/无合法法向不能施力。

D3同一次入步查询还借用运动浮体accepted FK。原NormalLayer按facet声明为
离散弹簧，产生节点barycentric散射与反向刚体COM wrench；完整刚体J/M和节点
材料候选分别按原积分算法形成。这里是entry-sampled显式载荷，不是迭代隐式
刚柔KKT，不要求精确相触，也不具有任意网格细分不变的面积压力语义。

实际法则/散射有TYX计算，但载荷校验读回及送入原节点/刚体通道仍存在HOST往返。
完整原始query、材料trace、力矩、功和状态观察也有HOST读回；不能标成整条闭环
零HOST、纯设备训练或已完成性能优化。

当前公开距离行的selected witnesses已经包含pair-local frame。响应只应再加
packet.origin_m；再加pair_origins会重复平移。本轮D4实际失败揭示并促成这条
共享坐标修复，不能由修复本身推导全部连续轨迹通过。新坐标回归及旧弹性重构
的严格速度误差失败保留，不因后续功能资格改写。

### D6插在节点候选内部，不在发布后修状态

DistanceDissipation按原材料边顺序绑定稳定point IDs，每边系数c单位N·s/m。
D1直接声明与D5场景透传使用同一实现。非零配置在原integrate、ordered弹性
projection、reconstruct全部完成后，先设备复制reconstructed velocity，再执行
**一次**ordered轴向耗散；各边只使用一次h，不随弹性迭代次数重复。它只改变
未发布候选速度，不修改候选位置、接触法则、质量或初态。None/全零配置仍走
原数值路径，不增加耗散核/phase；显式零参数仍属于模型来源身份。

每条边以当次位置方向和注册后实际FP32 inverse mass计算隐式轴向速度冲量，
分母保留实际存储方向的g²。整段是**一阶分裂、按边有序**的材料耗散，不是
整个网络的同时隐式解，也不是全局速度缩放。刚体瞬时平移/转动零模不应被阻尼；
平膜的法向一阶运动可能仍是轴向零模，不能据此承诺弯曲耗散或所有振型衰减。

耗散status、逐边32列trace、完整速度链、有限值和phase记录都在共同发布前
检查。任何世界候选失败不发布两侧新状态。HOST observer只读实际设备输出，
不生成替换速度；checkpoint绑定完整phase及accepted速度，restore不再执行
耗散。下一真实物理步才消费已经接受的新速度，D3本步的entry-sampled力不会
因耗散完成而重算成隐式接触。

| 记录 | 含义及用途 | 不可混称 |
| --- | --- | --- |
| phase.node_impulse / D3 damping_impulse | 对trace中实际FP32 requested j·n按边正负散射，再以FP64累计；包含固定节点收到的请求冲量 | 不等于最终速度变化反推的冲量 |
| phase.node_velocity_impulse | 对自由节点用实际stored速度差除以注册FP32 inverse mass；固定节点为零 | 不使用未舍入的声明质量，不强制与requested逐字相等 |
| fixed_reaction_impulse | 固定约束抵消外力/接触冲量、全部ordered弹性冲量以及requested耗散冲量；是N·s，除以h才是平均反力 | 固定v=0不代表反力为零；不能把自由节点的舍入差全归给支座 |
| kinetic_change / ideal_work / rounding_work | 每边实际stored动能变化、由实际n/g²/j/weights推导的理想功、两者有符号差 | 不能用跨边负能抵消遮盖单边误差，不能用D6通过覆盖旧弹性精度失败 |

D3的material_impulse仍专指原弹性材料项，damping_impulse另列。固定反力
按完整原弹性trace累计后，再减对应requested耗散冲量，避免漏项或重复记账。
声明参数和校准边界见[用法指南](p65-distance-dissipation-usage.md)，算法/记录
见[源码报告](p65-distance-dissipation-report.md)。

主要源码：
[tyx_deformable_batch.py](../../src/phymeo/backends/tyx_deformable_batch.py)、
[_deformable_boundary_source.py](../../src/phymeo/backends/_deformable_boundary_source.py)、
[_deformable_response_source.py](../../src/phymeo/backends/_deformable_response_source.py)、
[tyx_deformable_contact.py](../../src/phymeo/backends/tyx_deformable_contact.py)、
[deformable_dissipation.py](../../src/phymeo/backends/deformable_dissipation.py)、
[_tyx_distance_dissipation.py](../../src/phymeo/backends/_tyx_distance_dissipation.py)。

## 4. 共同发布和生命周期为何不是顺序step的包装

D3持两侧owner锁，原节点_stage_step只形成pending候选；
ResidentBatchTransaction完成刚体候选、全部world状态、节点pending、tick和
完整publication验证。最后只发布预构建字段与bank，不在一侧swap后调用可失败
的额外回调。因此晚准备失败不产生一侧新、一侧旧的accepted运动。

公开query(expected_revision)读取当前共同accepted源，不推进物理；
require_query_current同时核共同revision和原双来源。单纯改变控制也会使旧
显式query失效。last_step.query始终是该步的入步证据，不能当作步后当前查询。

checkpoint/restore/subset reset使用同一共同替换事务。未选world状态、时钟、
held controls和历史保留；restore不回退单调revision。失败后不是重新执行旧步
补日志，也不先提交物理后query再声称能回滚。

资源borrow依赖实际runtime lease；入口/出口同步、UNKNOWN释放和清理未决
状态分别处理。异常保留owner便于诊断，不授权盲目重试未知的底层释放。
原D1数组reservation与D3 query/response native预算仍分开，不冒充RSS峰值。

共享源码：
[_resident_batch_transaction.py](../../src/phymeo/backends/_resident_batch_transaction.py)、
[_deformable_runtime_lease.py](../../src/phymeo/backends/_deformable_runtime_lease.py)。

## 5. FREE+BALL法向/摩擦接触属于另一套原生语义路径

CPU MuJoCo编译只负责解析/defaults/原生模型与MJB，不在TYX step里代跑物理。
原native坐标、完整M/J/bias、JOINT CSR、电机力与功、物理M和Euler M+hD以及
FREE/child BALL四元数积分均由原TYX实现消费。

路由先完整检查FREE+BALL及R/P结构和原native地址，再按完整potential pairs
选择wrapper。R3-N的condim1保留原normal路径；R3-P增加NONE JOINT电机下的
primitive pyramidal condim3，允许明确的condim1/3混合。仅当存在condim3时要求
pyramidal cone；原纯condim1仍保原准入。完整四边EFC、原物理M及共同约束求解
复用已有核，不是把法向力独立扩成摩擦力后再积分。

两域都保nv35/nq45、body128、actor32、geom128/pair16/rawslot32边界。
无接触模型继续原floating路径；失败不会fallback。R3-P不扩展activation或
FILTEREXACT、joint limits、dry friction、condim4/6、任意transmission、多FREE
或新shape。固定BALL已有其它独立组合资格，不能把支持表作笛卡尔积。

这条路径没有把BALL模型转换成D5 ordinary单body资产，也没有让原生机器人
与D1节点共用D3事务。扩展这些连接需要新的模型/关节/材料来源和共同载荷合同。

主要源码：
[_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py)、
[_floating_model.py](../../src/phymeo/backends/mujoco_tyx/_floating_model.py)、
[_mujoco_tyx_runtime.py](../../src/phymeo/backends/_mujoco_tyx_runtime.py)。
实际组合边界见[R3-N报告](p65-free-ball-normal-report.md)及[R3-P报告](p65-free-ball-pyramidal-report.md)。

## 6. 异构GPU分桶连接的是多个独立模型world

MuJoCoTyxHeterogeneousBatch先按完整XML/archive/MJB、native版本、mode和容量
身份分桶，而不是仅按nq/nv合并。各桶复用原同构runtime；所有桶都准备并验证后，
一次发布聚合状态。迟到的末桶失败保留全部accepted world，reset/restore沿原
叶checkpoint替换事务进行。

公开上限仍为8模型/16world、相同原生h。G2a源矩阵已完整执行，首包中的
W16B3和W16B8均已在interpret/CUDA通过；这是16个独立world、3或8个模型桶，
没有把上限扩成256world，也不是256台G1。既有同构S256结果属于不同合同。
没有单异构kernel、跨桶碰撞或不同h联合积分的承诺。

异构能力消费各原生runtime准入的模型，但R3-N/R3-P新增可准入域不自动证明
它们与全部异构生命周期/规模组合。D5/D6节点共同事务也不经此原生模型分桶。
本轮没有同模型、同质量门、同完整输出的Newton实测，不能宣布更快。
累计上传/分配量不是live/peak内存，完整HOST观察成本仍需计入性能结论。

源码：[mujoco_heterogeneous.py](../../src/phymeo/backends/mujoco_heterogeneous.py)、
[mujoco_heterogeneous_checkpoint.py](../../src/phymeo/backends/mujoco_heterogeneous_checkpoint.py)。

## 7. 此草稿的验收状态，不用图代替结果

| 功能 | 当前可依据事实 | 仍须保留的边界 |
| --- | --- | --- |
| D1/D2/D3原有限范围 | Batch A/B/C各有源码与历史安装资格，D5继续复用同一共同事务 | 不等于任意刚柔、摩擦、CCD或整个阶段验收 |
| D5-A公共声明 | 源码26及受影响5每后端通过；首包双端通过；第二包132/133的interpret/CUDA summary均为PASS | 只覆盖原声明/短窗/生命周期；每world单ordinary浮体，多刚体/关节刚柔不在首片 |
| R3-N / R3-P | normal资格保留；R3-P源码69及兼容4完成，包外130/131双端PASS | FREE+BALL+R/P、NONE JOINT、primitive condim1/3；activation、limits/dry和condim4/6未扩展 |
| G2a异构分桶 | 源矩阵完整，首包77–80的W16B3/W16B8双端PASS | 上限仍8模型16world；不是新任意模型组合或公平Newton性能比较 |
| D6-A轴向材料耗散 | 源码功能检查完成；包外128/129双端FUNCTIONAL_PASS，各22 batch steps/44 worldsteps | 两包外summary同时保留strict_elastic_precision=FAIL_RECORDED；不授予D4稳态支撑 |
| D4-N持续载荷 | run50诊断解释完成3072步/3秒，入层、卸载、离层、再接触均发生 | 严格短速度门及原稳态支撑/速度门失败；不是只有精度问题 |
| D6-B持续载荷应用 | run143粗N1 interpret完成3072步/3秒，逐步law/phase/算术/长参考通过 | 两稳态窗支撑/速度RMS均FAIL，DIAGNOSTIC_ONLY；停止该系数的CUDA和其余格，不扫描参数 |
| 新坐标回归 | 已固定非零浮根anchor/变形pair frame的真实共同步 | 严格速度失败保留，不改输入或放宽门 |

D4诊断只允许记录原短position/velocity/multiplier门失败并继续相同轨迹；
原force/geometry、stored arithmetic、长窗误差、原侧、防贯穿、峰值、
支撑/卸载/再接触要求继续有效。没有修改h、质量、阻尼、初态、统计窗或accepted
state来制造功能结果。父run50已完成整条诊断轨迹：几何、原始算术、长窗运动、
原侧、防贯穿及峰值硬门通过，但两个预定稳态窗的平均支撑力与刚体/中心速度
未达原门。事件发生和完整走完3秒，不等于持续承载功能合格；D4严格FAIL保留。
此处仅引用父同次结果，不在作者树另算、改参数或重跑。

D6-A的功能资格与精度资格是两个字段：新的耗散phase、真实FP32原弹性路径、
连续运动工程门及事务检查有通过证据；原理想总动量/弹性重构严格门的run104、
run112失败保留。包外128/129记录的旧弹性重构速度最大差约1.253865e-4 m/s，
因此summary明确为FUNCTIONAL_PASS且strict_elastic_precision=FAIL_RECORDED，
不是把原严格门改为通过。耗散自身有效也不能消去D4两个稳态窗的功能失败。

本次文档核阅只读取以下已闭summary的顶层状态：首包74/81（D5）、77–80（G2a）；
第二包128/129（D6）、130/131（R3-P）、132/133（D5）。133在任务发出时尚待
执行，本次读取其summary已为PASS。父源记录位于
artifacts/p65-feature-until-0730-20261113-01，安装summary位于对应outside结果目录。
这不是另一次保存审计，也不据这些有限消费者宣称统一包的全部功能已验收。
D6-B现按闭合run143更新为功能FAIL；[独立结果](p65-damped-normal-load-result-20261113.md)保留原门、完整轨迹和7项旧弹性短精度记录。saved-only run148将原D4 CSV与新D6-B轨迹绘成对比图，没有重新运行物理或改写任何资格。

## 8. 后继连接仍应独立登记

- D5-A之后的M-B1多独立单体和M-B2非根R/P关节刚柔已接受；下一多机器人同world/BALL刚柔等组合仍需独立来源、完整力学和多案例，不能只放开构造guard。
- D4/D6之后：先单独判断固定持续载荷应用，再扩切向摩擦、换面测度/历史和CCD；耗散phase通过不替代持续支撑。
- 原节点模型之后：弯曲、密集布和体软体等材料/拓扑，不能用五条距离行代表全部软体。
- R3-A activation和BALL-S scalar limits/DOF dry已分别接受；下一BALL角限位、condim4/6等具体组合，逐项登记真实原生参考；不重复已通过组合。
- G2之后：完整生命周期资源/吞吐，再进行同模型同输出的Newton性能比较。
- 训练/人形：现有真实CUDA PPO和N5/N6/N7连接保留；J/R学习门、G1完整动作、
  传感与可微物理等问题不会由场景装配或法向接触通过自动关闭。

本页记录已存在的有限结果；文档更新不新增产品能力、项目执行或验收资格。

## 9. M-B1 / R3-A / G2-R0当前后继覆盖

前文D5-A/R3-N/P表格的限制属于各首片历史范围。当前M-B1提供同world多个独立单body与原节点模型的共同响应，总participants≤8/worlds≤8；新layout保存world body范围，每个site绑定真实participant/body槽，所有body包括零载荷者一次推进。新receipt明确逐participant wrench/delivery/外功，不再用含糊单world wrench。源及安装的S8实测4world/8participants，M3验证单/多body world混合；不授予任意关节树或新RR响应。

R3-A在原FREE+BALL/R/P与direct JOINT组合中接入NONE/FILTEREXACT、no-contact/normal/pyramidal；完整117新源码节点和第三包各352worldsteps通过。它没有接入D1，也不使所有异构族自动合格。

G2-R0修复restore提交前候选清理双失败的owner归属与异常聚合，保原accepted发布并真实恢复/释放；源码及第四包各12项通过，UNKNOWN原语义保留。完整资源manifest/生命周期预算仍是后继。

当前图见[独立HTML](p65-functional-architecture-20261113.html)，实际细目见[功能矩阵](p65-feature-acceptance-matrix-20261113.md)及[M-B1用法](p65-multibody-deformable-usage-20261113.md)。第四包534成员同源；M-B1及容量各后端68worldsteps/132participantsteps，与原D4/D6-B稳态FAIL分别记录。

## 10. M-B2：完整树载荷接入原共同事务

layout从每participant单body扩为完整canonical `(participant, group, body_id)` 地址；没有collider的root/child也进入source和预算。实际accepted FK/current group提供点速、COM lever与有限步功；设备transpose后按participant切出完整B×6零/非零载荷，原Jᵀ投影、全耦合M、偏置和控制只推进一次。没有新增第二套机器人积分或独立发布。

每world允许原多个独立单体，或一个ordinary R/P树，两个域可跨world混合。本批新源码25、受影响兼容39及安装每后端40worldsteps通过；新包534成员同源。[M-B2报告](p65-articulated-deformable-report-20261113.md)保留完整边界。图中的ordinary树不等于MuJoCo FREE＋BALL；多个关节机器人同world、摩擦/CCD/稳态和带D6的新树组合仍待各自资格。

## 11. BALL-S：把关节行接到既有完整约束路径

fixed/FREE BALL/R/P 的完整结构与 native 地址检查先完成，再按 scalar limit、DOF dry、contact 需要选择原T4；joint-only不会被no-pair早退跳过。已有NONE/FILTEREXACT JOINT与原condim1/3复用同一full-J混合上下界QP、质量方程与一次act/force/motion发布，没有新建一套接触求解器。能力报告区分native行来源、静态上界、请求硬容量和每步实际行。

恢复入口在替换设备前核对已接受mirror与receipt的qacc/qfrc_actuator/qfrc_constraint完整bytes。实际负控曾暴露缺口，修复和首失败均保留。132新唯一节点、8受影响兼容、两后端包外各448worldsteps已验，见[BALL-S报告](p65-ball-scalar-constraints-report-20261113.md)。这条MuJoCo-TYX路径仍不连接resident刚柔，BALL角限位/G1长舞/扩容也未取得资格；前文历史R3范围不代表当前BALL-S仍未实现。
