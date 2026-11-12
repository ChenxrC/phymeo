# 当前功能连接架构草稿：2026-11-13

状态：**架构草稿，供父任务最终收尾更新**。文本依据为父源码80204636及
本轮可读共享WIP、当前计划和父已报告的源码结果；没有执行项目、数值、
保存审计或安装消费者。本文件不替换最终结果报告，不能用于宣称D4严格
通过、本轮统一wheel通过或任意求解器耦合。

现有[统一编译架构](../architecture/unified-scene-compilation.md)保留历史阶段；
[续作入口](p65-continuation.md)与[全景](phymeo-newton-panorama-2026-11-12.md)
目前主要记到Batch C。下述图明确加入本轮源码接线，并逐项区分历史安装、
当前源码验证与尚未通过的轨迹。[本轮计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)
仍以07:30为窗口上限，不代表整个P6.5完成。

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

  NodeDecl["ParticipantDeformable<br/>D1节点/距离材料/初态/稳定point与facet IDs"]
  NodeDecl --> Factory
  Factory --> D5["deformable_normal<br/>静态lowering和原资产来源核验"]
  D5 --> Joint["唯一 TyxDeformableContactBatch<br/>同runtime节点与单浮体owner"]
  Joint --> Dual["同入步accepted双来源<br/>节点bank + 刚体FK"]
  Dual --> D2["gather → bounds/BVH → sphere/capsule–triangle有限距离"]
  D2 --> Law["仅合法RESOLVED近接触<br/>原离散facet NormalLayer"]
  Law --> NodeCandidate["节点外力+接触力<br/>原integrate / ordered材料 / reconstruct候选"]
  Law --> RigidCandidate["反向COM wrench<br/>原6DOF J/M与刚体候选"]
  NodeCandidate --> Commit["所有候选与完整receipt先验证<br/>一次共同发布两个accepted bank"]
  RigidCandidate --> Commit
  Commit --> Public["状态 / 力与矩 / 材料反力<br/>checkpoint / subset reset / restore"]

  Native["原MJCF/URDF及assets"]
  Native --> Compile["CPU MuJoCo 3.3.7编译<br/>原模型/MJB/来源与坐标记录"]
  Compile --> Admission["MuJoCo-TYX显式模型准入"]
  Admission --> NativeTYX["原TYX动力学 / primitive接触 / 约束"]
  NativeTYX --> NativeRuntime["MuJoCoRuntime<br/>controls / step / snapshot / checkpoint"]
  NativeRuntime --> Train["VecEnv / task-action设备接口<br/>随机reset / control windows / RSL PPO"]
  Compile --> Buckets["MuJoCoTyxHeterogeneousBatch<br/>完整来源分桶，同模型原runtime"]
  Buckets --> Multi["每桶候选 / 全world验证<br/>一个异构聚合发布"]
~~~

图中没有 D3→MuJoCo-TYX 或 ParticipantScene→任意MjSpec 的边：当前没有
这些通用转换/联合求解。异构桶之间是独立world，不因同一次发布产生跨桶碰撞。
M1有限距离也不会自动变成旧刚体HOST patch或MuJoCo-TYX primitive响应。

## 2. 声明和编译：新增的是可复用入口

旧compile_scene仍消费SceneDeclaration的既有共同域；它不是本轮多参与者
声明的同义词。多参与者入口是ParticipantSceneDeclaration和
compile_participant_scene，保留旧host_patch/none模式。

D5-A新增显式deformable_normal模式：

1. ParticipantArticulation保存资产、真实初态、控制、完整body/collider安装。
2. ParticipantDeformable保存D1模型、材料、初态、稳定point/facet与surface安装。
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
共享坐标修复，不能由修复本身推导全部连续轨迹通过。新坐标回归仍有严格速度
误差失败，原证据和门保留。

主要源码：
[tyx_deformable_batch.py](../../src/phymeo/backends/tyx_deformable_batch.py)、
[_deformable_boundary_source.py](../../src/phymeo/backends/_deformable_boundary_source.py)、
[_deformable_response_source.py](../../src/phymeo/backends/_deformable_response_source.py)、
[tyx_deformable_contact.py](../../src/phymeo/backends/tyx_deformable_contact.py)。

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

## 5. FREE+BALL正常接触属于另一套原生语义路径

CPU MuJoCo编译只负责解析/defaults/原生模型与MJB，不在TYX step里代跑物理。
原native坐标、完整M/J/bias、JOINT CSR、电机力与功、物理M和Euler M+hD以及
FREE/child BALL四元数积分均由原TYX实现消费。

本轮R3-N路由先完整检查FREE+BALL结构，再按完整potential pairs选择原normal
wrapper。只接受原primitive与混合condim1，保nv35/nq45、body128、actor32、
geom128/pair16/rawslot32边界。无接触模型继续原floating路径；失败不会fallback。

FREE+BALL/NONE JOINT电机的法向接触不扩大为pyramidal friction、FILTEREXACT、
joint limits、dry friction、任意transmission、多FREE、任意shape。固定BALL已有
其它独立组合资格，不能把两个支持表作笛卡尔积。

这条路径没有把BALL模型转换成D5 ordinary单body资产，也没有让原生机器人
与D1节点共用D3事务。扩展这些连接需要新的模型/关节/材料来源和共同载荷合同。

主要源码：
[_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py)、
[_floating_model.py](../../src/phymeo/backends/mujoco_tyx/_floating_model.py)、
[_mujoco_tyx_runtime.py](../../src/phymeo/backends/_mujoco_tyx_runtime.py)。
实际组合边界见[R3报告](p65-free-ball-normal-report.md)。

## 6. 异构GPU分桶连接的是多个独立模型world

MuJoCoTyxHeterogeneousBatch先按完整XML/archive/MJB、native版本、mode和容量
身份分桶，而不是仅按nq/nv合并。各桶复用原同构runtime；所有桶都准备并验证后，
一次发布聚合状态。迟到的末桶失败保留全部accepted world，reset/restore沿原
叶checkpoint替换事务进行。

现有公开上限8模型/16world、相同原生h；已接受的Batch C实际正控3模型6world。
G2a原容量内更大矩阵仍需自己的当前结果，不由旧同构S256或上限常量推导。
没有单异构kernel、跨桶碰撞或不同h联合积分的承诺。

异构能力可以消费各原生runtime准入的模型，但R3-N加入一个新的可准入域，
并不自动证明R3-N与所有异构生命周期/规模组合。本轮无同模型、同质量门、
同完整输出的Newton实测，不能宣布更快。累计上传/分配量也不是live/peak内存。

源码：[mujoco_heterogeneous.py](../../src/phymeo/backends/mujoco_heterogeneous.py)、
[mujoco_heterogeneous_checkpoint.py](../../src/phymeo/backends/mujoco_heterogeneous_checkpoint.py)。

## 7. 此草稿的验收状态，不用图代替结果

| 功能 | 当前可依据事实 | 仍须保留的边界 |
| --- | --- | --- |
| D1/D2/D3原有限范围 | Batch A/B/C各有源码与历史安装资格 | 不等于本轮新源码新包全量验收 |
| D5-A公共声明 | 父已报告新源码26及受影响5每后端通过 | 本轮安装仍待父闭环；多刚体/关节刚柔不在首片 |
| R3-N FREE+BALL normal | 父报告run22完整55源码节点及run31七兼容节点通过 | 本轮安装未在此草稿宣布通过；friction/activation/limits/dry仍拒绝 |
| D4-N持续载荷 | 原固定轨迹在诊断模式继续记录功能行为 | 严格短速度门失败保持FAIL；DIAGNOSTIC_COMPLETED不是PASS |
| 新坐标回归 | 已固定非零浮根anchor/变形pair frame的真实共同步 | 严格速度门失败保留，不改输入或放宽门 |
| 异构分桶 | Batch C完整源码/安装及原8模型16world容量合同 | G2a新规模结果、完整峰值预算和公平Newton性能比较另验 |

D4诊断只允许记录原短position/velocity/multiplier门失败并继续相同轨迹；
原force/geometry、stored arithmetic、长窗误差、原侧、防贯穿、峰值、
支撑/卸载/再接触要求继续有效。没有修改h、质量、阻尼、初态、统计窗或accepted
state来制造功能结果。最终是否完成诊断与哪些门实际满足由父同次证据决定。

## 8. 后继连接仍应独立登记

- D5-A之后：同world多刚体/关节资产与变形节点的共同载荷，不能只放开构造guard。
- D4之后：切向摩擦、换面测度/历史和连续碰撞；法向短窗不替代这些模块。
- 原节点模型之后：弯曲、密集布和体软体等材料/拓扑，不能用五条距离行代表全部软体。
- R3之后：pyramidal、activation、limits/dry等具体组合，逐项复用真实原生参考。
- G2之后：完整生命周期资源/吞吐，再进行同模型同输出的Newton性能比较。
- 训练/人形：现有真实CUDA PPO和N5/N6/N7连接保留；J/R学习门、G1完整动作、
  传感与可微物理等问题不会由场景装配或法向接触通过自动关闭。

本草稿只增加架构解释，没有产品改动、旧文档覆盖、项目执行或新的通过声明。
