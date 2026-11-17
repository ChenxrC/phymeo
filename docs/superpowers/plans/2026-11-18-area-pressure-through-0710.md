# 面积压力接触：2026-11-18 至 07:10 执行计划

起点：codex/exp d385760d；北京时间约 00:25 开始，07:10 结束。dev 不变、不 push；原 root 为唯一项目执行 owner，原两 agent 可并行只读审阅和隔离测试编写。全部数值、import、pytest、静态检查和构建串行，执行前 fresh guard。三个既有未跟踪文件保持。

## 当前依据

当前状态/FK → HOST 有限接触面 → 共同 TYX 动力学 → 统一观察已经打通。I3e 混合路径由设备提出裁剪拓扑，HOST 精确复核后发布。固定 5 world / 3 bucket 的 12 步 CUDA 比较中，后 11 步 HOST 0.9153 s、mixed 0.9268 s；192 对有 156 次 HOST 回退。尚无提速依据，不能把设备 kernel 运行称为全 GPU 碰撞或性能胜出。

## 顺序与分工

| 节点 | 难度 | 交付 | 验收 / 决策 |
| --- | --- | --- | --- |
| I3f，候选筛选与批量提交 | 中高 | 可证明空候选、跨 world/pair 请求批次、固定 workspace 分块，single/hetero 接线 | 几何/压力/状态与原 HOST 同输出；贴边不漏；末 query/chunk 故障不发布部分结果；容量和 lifetime 明确 |
| I3g，减少不变资产重复准备 | 高 | 根据 I3f 分项成本选择，建立 owned 资产的准备与失效机制 | 不信调用者 private caches；当前位置仍重算；reset/restore/clone 与材料身份保真；有实际成本收益 |
| I3h，固定多规模成本比较 | 中 | 同模型、同物理输出、同观察范围的小/中规模矩阵 | 包含 FK/几何/传输/求解/读回；报告初始化与稳态；无收益保留 HOST 默认，不调案例求绿 |
| 后续功能节点，按余量选择 | 中高 | area body-pair 观察的应用读取，或当前表示的通用场景入口 | 必须补真实缺口；不同时引入 FFT 材料、摩擦和另一响应体系；完成独立验收后提交 |
| 收尾 | 低 | 更新报告、续作入口、当前架构/差距概览 | 06:35 起优先关闭在途批次；06:50 后不新启预计无法在截止前结束的大组，07:10 汇总 |

root 负责共享源码和串行执行。multitask_scene_review 只读检查数学/事务/资源；multitask_training_review 在原隔离树编写下一批新增测试，父审阅并导入后执行。不得重启已验收旧组或创建新的周期任务。

## I3f 固定合同

新增 immutable request 和 intersect_many(tuple)。每 query 保留原 workspace capacity 的总 cell×face 上限；整个请求批次总 pairs 不超过现 MAX_CANDIDATE_PAIRS=2048。全部请求先检查身份、原输入网格与容量，再进行设备工作；非空候选按固定 workspace capacity 分块。严格 AABB 分离或 tetra 的某个 exact barycentric plane 对 rigid triangle 三顶点均小于零，才可证明 empty；贴边/未证空结果沿原 exact_host/reject。不存在静默截断、概率丢候选或改变物理误差门。

全部 query 成功后同时发布 last_results / last_batch_routes。旧 last_result / last_routes 保留最后一个 query 的接口语义，诊断 counters 不作为物理状态。single 与 heterogeneous scene 先从当前 owner 搜集请求，再整批求交；物理仍使用原全批事务。不得缓存旧状态或跨 world 混用材料。

## 避免循环

1. 每一批先固定语义、测试和比较范围，再执行；已通过同源码不重复运行。
2. 功能正确但整步无提速：定位最大的实际成本，只做一个有依据的改进再比较；结果仍无收益就记录边界，转向功能缺口。
3. 不为通过修改质量、几何、步长、预算或精度；不以理论证明代替实际工程功能，也不为不可观察的精度无限扩展证明。
4. 测试失败先保留原始证据、定位并修复对应原因；不能通过新增大批近似重复测试掩盖失败。
5. 到期只汇报实际完成；未完成节点保留明确起点，不标全部 Phymeo/P6.5 完成。

相关依据：[I3e 报告](../../development/area-pressure-device-topology-report-20261118.md)、[当前续作](../../development/p65-continuation.md)。

## 01:00 执行前调整与下一验收配置

- I3f 已完成：f8d902ea；12 新＋56 受影响兼容节点全部通过。156/192 原候选得到严格分离证据，4 个 world 查询合并后每步一次设备提交；默认仍 HOST。
- I3g 已进入验证：内部 raw 声明在本步统一构造一次完整 mesh，HOST/mixed 同时去重，公开 mesh 输入仍全量重验，不跨步缓存。
- I3h 在运行前读到公共 MAX_WORLDS=32，因此固定为 **5、20、32** world（80 不受支持，不执行、不标 skip、不分批冒充）。不扩容量或改 model/h/material。按原五世界顺序循环复制，32 使用前32条，保存完整 world→model 清单；每档只比较自身 HOST/mixed 相同输入，不以不同模板占比推断线性扩展。
- I3h 固定 CUDA、capacity=64、25 连续步（首步＋3段各8暖步），每步交替路径先后，比较完整状态/几何/统一观察字节。模型声明的共用准备耗时、各批次/scene/workspace setup、整步含读回时间分别记录；不设速度通过门。若仍无稳定收益保留 HOST 默认，下一功能节点为显式 area body-pair 载荷 reader，同帧消费和checkpoint身份；不把 reader 接通说成 Env 或训练已经接通。

## I3i 执行前合同：显式面积载荷观察适配

新增 `AreaPressureRoleBinding(identity, object_side, role)`，通过 `ManipulationSuccessReader(..., area_roles=...)` 按模型显式绑定全部声明 pair。role 只允许 finger0/finger1/support/other，object_side 为 A/B；pair 不含任务对象时仅 None/other。沿用真实独立 FREE object、XML/timestep、命名 geom/body 准入，不靠数字 body ID 猜任务角色。一个 pair 只负责一个角色，不隐式分面。

读取共同 accepted-input 帧的法向力 N 和 A 方世界力；对象为 B 时反号。dA 只验证真实面积，不重复乘已求解力。全部 identity/material、pair/sample/source 顺序、历史 stamp、面积声明 mask 和数值数组参与同 tick/恢复绑定。native 又出现声明 area 的同一 body pair 时拒绝重复载荷。整次读取或恢复成功后才提交 cursor；native-only 原 binding/fingerprint 字节路径保持。

验收固定新明示协议夹具：D=0、unlimited、真实3P+3motor，独立FREE球(.08kg,R=.05m,z=.049m)，h=.0005，gravity=-9.81，最多3步。A/B/split/native 四world；每pair真实面积1e-4m²、g=2e6Pa/m、delta=.001m，split仅四分同面积与压力。独立receipt全样本oracle、A/B方向、细分载荷、缺声明/身份与目录故障、同tick变化、controls-only/reset/成对restore。此节点只接HOST评估reader，不改Env、actor特征或声称任务训练成功。

I3h 首20-world在构造时发现夹具漏传已有expanded-batch资源预算；未进入20-world仿真。修正为每条比较路径独立64MiB registered CUDA payload预算（不含context/JIT/allocator/host等），不改公共32上限，不调整几何/物理参数。保留5-world已绿结果，只续20/32，报告预算实用峰值与排除项。

## I3j 执行前合同：自动当前几何接入任务 Env

仅新增 `area_scene=None, area_roles=None` 可选参数；面积声明要求两者齐备、Scene.borrowed batch恰为Env batch。Scene公开只读asset_sha256。构造前置核所有asset identity与role完整相符；Env静态binding加入asset SHA和reader角色binding，native-only原结构不变。动作通过原_submit接受之后，用新的batch revision调用area_scene.step一次推进全部native/area worlds，然后原export→reader→evaluate_goal。调用者不逐步供应surface。

自动/显式reset沿scene.reset；checkpoint保持现codec，保存scene capsule的原batch_checkpoint、静态asset身份绑定在Env元数据。恢复先验证binding，再用既有SceneCheckpoint.make资产封套调用scene.restore，随后配对新reader。不会重新生成历史数值，也不保存诊断routes作为资产。动作已接受后geometry失败必须needs_recovery；物理已接受后观察失败仍须成对恢复，不能返回旧transition冒成功。

新固定真实CUDA夹具：复用I3i的D=0/unlimited/真实3P+motors，object改为质量.08kg、半宽(.05,.05,.05)m的独立FREE box，center z=.049，h=.0005，g=-9.81；support为halfwidth(.01,.01,.005) box，中心z=-.005，顶面z=0。对象压力资产为与box相同边界的8corner+center/12tet完整立方体，corner压力0，center压力100000Pa；第二area model仅center125000Pa、独立材料identity。plate资产为真实support顶面两三角。初始1mm压入时中央小支撑落在bottom两tet域内，侧tet无正面积交集；不将零有效梯度面强行接入。3world非连续顺序(area-soft/native/area-stiff)，native使用同明示Scene但无area声明。动作常量，最长4步；不调物理参数追成功。

测试四组：与独立batch+scene+reader顺序逐字段对照（只运行revision/config tokens允许不同）；短horizon自动/手动选择reset；fresh Env paired restore重放及变资产/角色预拒；末world几何与物理后观察故障、无新transition、poison及恢复。保留actor设备数组中的area字段；本批不声称策略已经消费新字段、不做PPO或抓取成功率。

## 02:00 进度

I3j 已完成9b928c2d：4新CUDA/48.22s、6native兼容/82.34s，当前状态自动接触面接通任务Env，成对恢复和晚故障通过。下一先只读定位实际策略编码入口，不把设备数组存在当成网络已经消费。

## I3k 执行前合同：通用同入步接触力/矩读取

代码确认异构语义encoder/actor/PPO仍未实现；旧同构PPO只取q/v，不作为本批现成接入口。先补真实接触应用缺口，不另建训练体系。

公共 AreaPressureEntryReader(workspace, identities_by_model) 统一owned材料/有序pair与同入步面积数组、目录、native元数据/时钟/owner、padding和重复bodypair验证；返回immutable样本与原digest。AreaSuccessAdapter复用它，checkpoint digest次序保持，最多一次workspace.readback。

新增 BodyContactWrenchReader(workspace, area_identities=None).measure(frame, world_key, body_name, reference_point_world, group_body_names=())，按公共metadata命名body、同一当前frame读取native+area。native保真实intrinsic couple，area A正/B反；force已为N，不二乘dA。area用已发布world-origin moment平移到显式参考点，不把它自动称COM。相关sample moment缺省时完整wrench明确拒绝；无关body/world缺省不影响该body结果。保留native/area来源和分项合力矩、全部载荷及body分组，不伪造area geomID。总量只含接触，不含重力/驱动/外载。

旧native-only wrench/slip入口若选中行声明area，必须明确拒绝，不能忽略area返回少算载荷。无area旧路径保持。

固定HOST手向量：native p=(1,2,3),F=(1,2,4),couple=(.5,-.25,.75)；area p=(-1,0,2),n=ey,Fn总3N,dA总.02m²，以3与12目录样本细分。总F=(1,5,4),M0=(-3.5,-1.25,-2.25)；参考r=(.2,-.3,.4)后的M=(-.3,-.85,-3.55)。不同bodypair避免双算；测试双方反力、组别、真实面积、坏身份/phase/mask/矩表示与缺省。实际CUDA只复用I3j固定cube/3world自动几何，最长4stepcalls，不改物理门；已受影响I3i/legacy readers需相应兼容核。完成后优先同一真实连续轨迹与可视化，不宣称共享训练完成。

## I3l 执行前合同：4 world连续自动接触与可视化

固定CUDA默认HOST几何，h=.001s、T=.45s、450整批step/1800 world-step，3个model。world顺序41(area-soft)、7(area-offset)、305(共享soft model独立初态)、11(native)。全部竖直无限位slider、无dry friction；导轨承担横向/转矩，不称FREE平衡或任务Env/PPO。

soft cube尺寸40×40×20mm，m=.30kg、D=70Ns/m、center压力100000Pa、plate20×20×5mm居中，g=1e7Pa/m、K=4000N/m。offset cube同尺寸，m=.45kg、D=90、center压力150000Pa、plate20×15×5mm且中心x=4mm，g=1.5e7、K=4500。全8corner+center/12tet、边界压力零；刚面取真实plate顶面。world41初始gap2mm/v0；world7 gap3mm/v0；world305 gap1mm/v=-.03m/s。native world11 m=.30,D70,R10mm球/居中20×20×5mm plate、gap2mm/v0，condim1/零摩擦；其几何与律均不同，只作混合路径正常工作对照，不称算法精度公平比较。

全部world公开qfrc外载为[0,.15):0N，[.15,.30):-1.5N，[.30,.45):+7N，gravity=-9.81。仅提交力，不重置/指定运动。area原生geom碰撞关闭，native模型无area声明。读取每步真实状态→自动面→共同solve→统一BodyWrench，保存完整入步/出步/实际载荷、面与样本/身份、公共数组、wrench分项、耗时；失败保已接受前缀。

area逐步独立标量oracle：Meff=m+hD，vfree=(mv+h(fext-mg))/Meff；入步delta>=0时F=max(0,K(delta-hvfree)/(1+h²K/Meff))，否则F=0，vnext=vfree+hF/Meff。delta从同入步body-origin和已声明底面/plate-local高度独立计算，不调用裁剪/装配器。另从固定初态推进全FP64参考轨迹，不用产品状态回填，完整偏差保存而不新增精度优化目标。

固定门：全部有限且450步接受；area入/出步压入<2.5mm，正交集真实面积soft4e-4、offset3e-4m²。逐步力误差<=.001N+.001relative，v误差<=3e-5m/s。平台[.12,.15)、[.27,.30)均值力/深度<=3%，max|v|<=.002；soft目标2.943/4.443N，offset4.4145/5.9145。最后.04s area力0且gap>1mm，三area均空→有载→离层；native需真实出生/最终分离但不套面积刚度门。offset入步bodyorigin矩My=-.004Fz，误差<=2e-6Nm+2e-4relative，A/B反号。area累计正能量残差(含外载功和真实D耗散)<=初始正能量2%，仅工程门。没有第二h/调参或已绿旧窗口重复。

中文四格画真实入步姿态与同相位力，颜色为求解牵引Fn/dA(Pa)，与入步弹性压力分开；显示gap/Fz/My/阶段及曲线。从同次保存轨迹取帧，不能插值造接触，明确CUDA动力学+HOST有限面生成/导轨案例。新可运行experiment、独立测试、离线plot/viewer，之后再按余量补实际使用连接。

## I3m / I3n 后续安排（依据 I3k/I3l 实际接口）

I3l 连续轨迹完成后，优先 I3m 同帧多 body wrench 查询。当前 reader 每查一个 body 就完整读回一次；四 world 双方测量会重复八次。新增有界 immutable request 与 measure_many：全批请求先验证，工作区只读回一次、面积 ABI/目录只核一次，所有结果按请求次序完整返回，末请求失败不部分发布。单 measure 通过相同实现保留原结果。没有跨 frame 缓存、数值重算、控制或求解语义变化。固定 HOST 手向量验证与单查询逐字节一致、一次读取、坏末请求零读取、末样本失败无部分结果；一个原短 CUDA 场景验证完整 A/B 同帧及 stale/reset 边界，不重跑 I3l 长窗。

若 I3m 顺利且时间允许，I3n 补当前 body-local 压力资产的可携带声明：现 scene/checkpoint 只有资产身份 hash，调用者仍需 Python 重建资产。提供有版本/单位/模型及有序 pair 的有界数据封套，完整保存原节点、cells/faces、压力、材料身份，解码按现构造器重新验真实几何。只支持当前 full zero-boundary pressure 对 finite rigid mesh、无摩擦法向律；不把 XML/URDF 自动变压力资产，不声称 FFT 或新材料标定。先做数据往返/篡改/容量/owned 生命周期及小 CUDA 加载后与原资产同输出，再给独立运行说明。具体 schema 在实现前单独冻结。

06:35 起关闭在途验证与可视化，补当前“声明→当前状态→候选/有限面→共同解→设备观察→应用”的架构及差距；07:10 汇总实际交付和后续优先级。未完成的节点明确保留，不用扩大测试矩阵填时间。
### I3m 具体接口冻结

`BodyContactWrenchRequest(world_key, body_name, reference_point_world, group_body_names=())` 为 frozen 请求，拥有三元FP64参考点tuple和原有序group tuple。`measure_many(frame, requests)`接受builtin tuple、1..1024个exact request；重复查询合法，输出次序保持。构造请求时验证类型/有限值；整个批次先按reader实际metadata验证world/body/group，末请求错误不得触发readback。随后一次workspace.readback，完整area validation一次，所有query共享这份owned输入。所有结果算完才返回tuple；缺历史的query保持None，相关矩缺省/溢出整体raise。无跨调用缓存、无部分结果状态。原measure构造单request并走同实现。1024为诊断请求上限，不修改32world、body或物理容量。

冻结验证：已有I3k手向量的多body/不同reference/group、反序/重复、owned mutable input、invalid末请求零读取、末相关矩缺省和数值错误原子拒绝、空/invalid history、freshness委托及无跨帧缓存；短CUDA仅用I3j原场景两个原step再paired restore一步，按需selectedreset，many结果与同frame singles完全一致，公开readback次数1比N。该节点直接修改reader，因此其原HOST兼容组需要重验；不重跑I3l连续窗口。
### I3n 具体接口冻结：可携带 body-local 资产

只新增 backends/area_pressure_assets.py，不改checkpoint/runtime/Scene。`AreaPressureAssetBundle.from_assets(model_specs, assets)`、`from_bytes(raw)`、`to_bytes()`、`assets_for(model_specs)`；assets_for返回Scene既有model→ordered tuple，全部area model必须完整匹配，允许额外native model。bundle只拥有bytes，不存调用者mesh别名或可信私有几何cache。每次导出为新owned资产，任一导出的非法修改不影响bundle与其他导出。

固定schema phymeo-body-area-assets-v1；coordinates=mujoco-body-local（不是COM/URDF link），units length=m/pressure=Pa，pressure_domain=full-zero-boundary，law=frozen-surface-elastic-backward-euler-v1，friction=none。每model保存source primary/archive/model SHA256和mujoco_version、有序pair的四identity字段及压力/刚面原node IDs、body-local vertices、cells/faces、nodal_pressure、cell/face IDs；保存既有异构asset_sha256。source SHA由compilation.source实际三组bytes重新计算并与元数据核符；origin路径、world ID、tick/revision不入绑定。首版精确模型/版本绑定，不自动缩放/移动/重标定，也不从URDF猜压力。

封套MAGIC+payload SHA256+JSON；SHA仅损坏检测非认证。16MiB总bytes、JSON深度16/节点262144、32model/每model16pair、既有mesh128node/32tet/64rigidface，先整包字段/数量/预期dtype/shape/base64长度预验再做exact几何。仅固定位置数组<f8/int64<i8，可复用既有数组codec但绝不对整包通用_decode，不接受$record或未知fields。JSON重复key/非有限值/额外字段拒绝，原始数组byte包含signed zero保留。

全部原TetraPressureMesh、RigidPressureMesh、BodyAreaPressureAsset构造器重新验证后才发布。合法tet重排不禁；资产通过不保证所有姿态的g>0、共享面梯度或查询容量通过，后续query仍独立拒绝。

新HOST验收精确roundtrip/hash/顺序/byte/owned，重算checksum后的schema单位law/source/pair/未知record/dtype/shape/bool/容量篡改，非零边界/重叠tet/退化面和caller导出修改；短CUDA复用原I3j三world原资产/加载资产各2步、状态与几何/压力/观察一致（只排除实际run的owner/revision派生token），相同资产SHA可与原scene checkpoint配对。不重跑450步。给基于同一真实案例的独立bundle保存/加载说明，不做新训练结论。
## 03:20 后续批次：I3o 三体耦合与 I3p 摩擦模型选择

I3l/I3m/I3n均已完成。I2b已覆盖真实hinge＋可动slider的法向转矩与面积启闭，I2c已有双区共用单个运动DOF，因此不把重复这两项当新进展。当前未验的是接触另一侧也分别具有独立质量与速度；面积路径切向力仍为空缺。先完成下列两个有界交付，root串行执行、原两agent隔离纯编码。

### I3o：三个可动物体、两个接触区

复用I2c完整压力资产和左右有限三角面，面积(.000225,.00045)m²、刚度(4500,9000)N/m。压力体与左右支撑各有独立无限竖直slider，质量(.5,1.25,.75)kg，gravity/damping/外载均零；初始两区压入1mm、全部速度零，h=.001s、40连续步、capacity64、真实CUDA、默认HOST当前几何。只由预载弹性能驱动运动，不指定轨迹；导轨承担横向/转矩，不称FREE三体平衡。

独立参考J=[[1,-1,0],[1,0,-1]]，B=diag(1/K)+h² J M^-1 J^T，r=delta-hJv。枚举四active sets，求f>=0、Bf-r>=0、f·(Bf-r)=0，再核v+=v+hM^-1J^Tf。已在入步分离的pair遵守现运行时无接触行语义；禁止用参考器偷偷增加预测接触。完整跨pair B12=h²/m0保留，不按pair独立求力后相加。另从固定原初态推进FP64参考，仅报告全轨迹差异。

执行前固定门：每步力误差<=.001N+.001relative、速度<=3e-5m/s、q积分<=2e-7m；总竖直动量误差<=2e-5kg m/s；累计正能量残差<=初始预载弹性能2%。完整sample/source/dA/Jacobian、双方共同原点反力矩和无重复native载荷；真实事件须两区有载→单区→全部离层。若事件或准入失败保留证据，不调质量/步长/面积求绿。首步另辨别错误独立pair响应；反序pair只运行一个新首步，不重跑40步。最多两个HOST与一个CUDA完整节点；保存完整输入/输出/原始receipt/几何/公共arrays和一次measure_many三体wrench，失败保已接受前缀。

### I3p：摩擦候选的固定独立对照

不改type7、现friction=none资产law或法向ABI。独立解耦2DOF参考，不称TYX或一般多接触求解。固定m=.5kg、K=4000N/m、mu=.5、vs=.001m/s；四输入各h=.001/.0005，两档共8格、三方法共24行，无连续轨迹或参数扫描：

| 输入 | phi0 m | vn0 m/s | vt0 m/s | Fext,n N | Fext,t N |
| --- | ---: | ---: | ---: | ---: | ---: |
| 支撑滑动 | -.001 | 0 | .02 | -4 | 0 |
| 支撑低载启动 | -.001 | 0 | 0 | -4 | 1 |
| 层内冲击 | -.0001 | -.5 | .02 | 0 | 0 |
| 卸载分离 | -.0001 | .3 | .02 | 0 | 0 |

同一法向闭式Fn=max(0,-K(phi0+h vnfree)/(1+h²K/m))。切向分别为最终Fn的精确1D Coulomb闭式、最终Fn的平滑律、入步F0=max(0,-Kphi0)的Lagged平滑律。平滑方程v+h mu N/m·v/sqrt(v²+vs²)=vtfree，固定[0,vtfree]有向区间，最多80次二分；残差<=1e-13m/s、全部finite、摩擦功率<=1e-14W，未达门不拿最后iterate冒成功。

预定判别：支撑Fn=4N；低载启动exact vt+=0、smooth非零；冲击Fn=50/21或700/501N；卸载Fn_final=0，exact Ft=0，Lagged |Ft|>.19N。最后一条作为模型差异必须公开，不能用能量耗散通过掩盖最终Fn的Coulomb上界不满足。保存输入/free/new速度、F0/Fn/Ft/impulse/残差/迭代、动能/弹性能/BE外功/法切功率。只评估既定三律，不为Lagged改名宣称即时Coulomb。

依据：[Lagged原论文 §V-B、VII-C](https://arxiv.org/html/2312.03908v3)、[Drake近似类型](https://drake.mit.edu/pydrake/pydrake.multibody.plant.html#pydrake.multibody.plant.DiscreteContactApproximation)、[MuJoCo摩擦锥](https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones)。后续是否实现新显式摩擦law，取决于同一步载荷/运动、最终法向capacity要求和一般多接触求解方案；不由本解耦参考直接推出产品已支持摩擦。

### 03:30 I3p实际结果与I3q执行前合同

I3p源码596ad132，10新HOST/0.19s通过，固定24行结果完整保存。卸载两h的最终Fn=0，Lagged仍分别有约.199740/.199745N切向力；支撑低载平滑律存在.427537/.346014mm/s微滑，精确1D Coulomb为零。由于当前目标要求与最终法向承载一致，后续先研究 **final-normal smooth**，不把Lagged当现法向路径的无差别替换，也不把平滑律称精确静摩擦。

I3q限定独立HOST共享自由度参考，新增experiments/area_pressure/coupled_friction_reference.py及独立tests，不修改生产type7/观察ABI/资产版本。通用输入完整M、非负对角关节阻尼D、v0、外广义力f、h，以及每点p/n/JA/JB/k/phi/mu/vs；1..6 DOF、0..8 samples。M须有限对称正定，全部输入owned不可写，phi<=0、k>0、mu>=0、vs>0。法向先核近单位(1e-12)再显式单位化，保存实际使用n；投影用双叉积/(n·n)，不将纯法向误作滑移。

A=M+h diag(D)，vfree=A^-1(Mv0+hf)。G=JA-JB，vn=n·Gv，vt=(I-nnT/(n·n))Gv，Fn=max(0,-k(phi+h vn))，Ft=-mu Fn vt/sqrt(vt·vt+vs²)。求同一v的R=A(v-vfree)-hΣGT(Fn n+Ft)=0。保留Fn对v的交叉项，Jacobian一般非对称，禁止对称化/Cholesky解Newton矩阵或先独立求法向再补摩擦。物理M的Cholesky只用于准入及残差度量。

固定上限24 Newton updates、每次最多12实际回溯(power0..11)，初值vfree；用完整实际残差的质量度量回溯，不对候选状态施修复。最终重验 ||A^-1/2 R||2 <=1e-10*(1+||vfree||A)。奇异/非有限/预算失败明确拒绝，保存诊断，不回退Lagged或返回最后iterate冒通过；不保证任意输入收敛/唯一。允许更小预算作故障验证。

固定六输入，各h=.001：

- support/release：两个平移DOF(x,z)，M=.5I、D=0，JA把两个DOF映到ex/ez，JB=0、point=0、n=ez、mu=.5、vs=.001、k4000。support v0=(.02,0)、f=(0,-4)、phi=-.001；release v0=(.02,.3)、f=0、phi=-.0001。独立解析normal和标量单调方程参考。
- shared-one-dof：M=.5、D=0、v0=.02、f=0，JA=(1,0,1)^T、JB=0、n=ez、point0、k4000、phi=-.001、mu=.5、vs=.001。法切响应共用一个DOF，独立标量根区间[.02,.028]。
- two-point-manufactured：3DOF(vx,vz,omega_y)，M=diag(.5,.5,.0004)、D=(1,2,.001)、v0=(.02,-.01,.2)，points=(±.02,0,-.01)，JA rows=(1,0,-.01),(0,0,0),(0,1,-x)，JB0，n=ez，K=(2000,3000)、phi=(-.001,-.0008)、mu=(.4,.6)、vs=.001。预定v*=(.015,-.002,.3)，独立手算vt=.012、vn=(.004,-.008)、Fn=(1.992,2.424)，由这些力倒算固定f使其成为根。只作有已知解的耦合数值验证，不能冒真实外载场景。
- two-point-frictionless：同两point/M/J/K/phi，D=0、v0=(0,-.1,.2)、f0、mu0。独立四active-set法向参考，保留非对角质量响应。
- empty：support的M/D/v0/f，但0samples，精确回到自由速度。

另验pair反序、空间正交旋转/共同平移后的force/moment与虚功、近单位法向纯法向无伪摩擦、owned输入和shape/容量/非正定/非法参数、预算失败原子性。保存同次六结果完整输入/exit速度、样本法切force/双侧wrench/广义力、全部尝试/残差，以及真实M动能、线性代理gap弹性能、BE外功/D耗散/摩擦功；最后能量只针对冻结J/phi线性代理，不冒实际非线性轨迹。

该节点查清最终Fn律在共享质量中的可执行接缝；通过后再决定小型TYX设备参考或生产集成计划。已有450步正常支撑/三体窗口不作为摩擦通过证据，也不为新接口重复跑旧几何。

### I3r：I3q通过后的有界TYX设备参考（并行纯编码，执行串行）

这是独立实验workspace，不接生产type7/scene/bundle，不更新q，也不声称完整机器人摩擦已实现。I3q HOST验收通过是执行前置；可先在隔离树纯编码。首版CUDA、1..32world、每world1..6DOF/0..8samples，solve_many(tuple(problem))按异构尺寸一次设备批次，固定容量persistent buffers。公共问题定义复用I3q，完整M/D/v0/f/h与所有JA/JB/n/p/k/phi/mu/vs进入同一解。

所有原输入先完整owned准入再FP32转换；正参数下溢/非有限/转换后非SPD拒绝。保存原FP64与真正上传FP32输入。设备计算A和free velocity、最终Fn平滑摩擦、全非对称Newton Jacobian；24更新×12回溯，不变。仅SPD A用Cholesky；Newton采用部分选主元行交换消元，记录实际swap计数。每次完整覆盖有效区/清padding，无跨调用warmstart。真实CUDA失败禁止继续用该实例；无数值HOST polish、对称化或Lagged fallback。

全部world通过设备status、实际输出finite及独立HOST对原始输出的残差/载荷核验后，才整批发布immutable HOST-owned结果；坏末请求零launch，末world失败/晚下载/结果封装失败不发布前缀，原last_results整体保留。数值预算拒绝可用全新输入重试，不能让调用者把上批last_results误认为失败本批成功。按既有DeviceResources保释放次序/未知释放，不新增native生命周期策略。

**FP32资格与HOST参考分开**：设备目标质量残差<=1e-5*(1+||vfree||A)，独立FP64代回实际上传输入/实际输出，质量残差<=2e-5*(1+||vfree||A)。原I3q 1e-10是否满足另报告，不冒相同精度。与原FP64参考解比较的工程门：速度每分量5e-5*(1+abs(reference))，力每分量.002N+.002*abs(reference)，world-origin矩1e-5Nm+.002relative；最终Fn摩擦上界容许FP32核验余量1e-5N。物理量、原始设备残差及独立核验值分别保存，无覆盖。

固定验证：I3q六输入各一次W1；同六输入反序一次batch，按身份与W1比；32world/nv6/ns8容量例只将原两点µ0案例扩成6DOF并各四分同位置k、额外3DOF零contact映射/正质量，保原前三DOF物理响应，不冒新场景规模性能。另一个明确触发行交换的非对角M案例：M=[[.01,.1],[.1,2]]，D0/v0=0/f0，h=.001，JA映(x,z)，JB0、n=ez、point0、k4000、phi=-.001、mu0、vs=.001；独立Fn=4/1.004，v=(-.01Fn,.001Fn)，Newton首列.1>.01需交换。实际swap必须>0。

故障覆盖末输入非法零launch、预算0的晚world求解失败、晚下载失败、可重试数值拒绝、close/UNKNOWN_RELEASE；原生资源故障只mock既有ledger回调，不制造真实CUDA设备崩溃。记录setup/上传/kernel+sync/下载/核验/整批时间，无速度通过门，不把少量case GPU运行叫优于Newton。最多一个首次完整focused组，必要诊断修复按具体失败，不重跑I3l长窗/旧静态几何。06:35后优先收尾，06:50后不新启无法结束的组。

## 04:20 后续调整：I3s 真实 FREE 六自由度有限面窗口

I3q/r已分别通过10HOST与6个独立CUDA/故障节点。生产摩擦还需新law/ABI、完整积分点容量和真实动力学接缝；见[接缝图](../../development/area-pressure-friction-integration-map-20261118.md)。不在剩余时间仓促改变旧friction=none合同。下一批补生产法向路径尚缺的无导轨连续自由刚体证据；不是重复slider/hinge窗口。

新增实验 free_body_support.py 与独立 test_area_pressure_free_body_support.py；root唯一串行执行，原scene agent纯author实验，training agent纯author测试。无需改生产src、旧例或门槛。固定一个model、world顺序(41,7,305)，h=.001s、150整批step/450world-step，CUDA dynamics＋默认HOST当前面；m=.30kg，40×40×20mm cube，COM=bodyorigin，均匀惯量diag(m/6000,m/6000,m/3750)，gravity=(0,0,-9.81)，D=0、无控制/外载/摩擦，native geom collision关闭。一个FREE joint，不锁平移/旋转，不每步重设q/v。

压力资产沿I3l完整8corner+center/12tet，boundary p=0、center100000Pa；有限plate20×20×5mm、worldorigin(0,0,-.0025)、top z=0，真实顶面两三角。contact cap16、constraint cap64，模型solver128iterations/tol1e-10沿既有声明。三个世界仅初态不同：
- 41 center-drop：xyz=(0,0,.011)，R=I，v=0；
- 7 tilted-support：xyz=(0,0,(.01-.00073575)/cos(.02))，R=Rz(.35)Ry(.02)，v=0；
- 305 offset-support：xyz=(.001,0,.00926425)，R=I，v=0。
初始quaternion WXYZ；运行free角速度为body-local，平移速度为world。偏心是压力体真实x偏移，不移动样本伪造转矩。所有资产原数组和初态全保存。

独立oracle每个实际入步只用另一个MjData的mj_forward/mj_fullM/mj_jac得到M、bias/passive/applied、COM/bodypose和完整双侧point J；不mj_step、不用CPU写产品状态。native无接触/约束。每个实际sample反变换到cube-local，独立帐篷场p=100000*max(0,min(1-|x|/.02,1-|y|/.02,1-|z|/.01))，本有限域底面项须独立最小且留余量；gradient=1e7*R[:,2]，g=n·gradient，k=dA*g，phi=-p/g。独立Jn=n·(JA-JB)，Fn_hat=max(0,-k(phi+h*Jn*vout))。全点保留，不删面/重归一/回退另一律。

逐步固定门：
- M平移块atol2e-6、转动块atol2e-9，其余atol2e-8，rtol3e-4；完整J rtol2e-5/atol2e-6，bias/passive/applied rtol3e-4/atol2e-5；
- Fn与实际行：atol.001N+rtol.001；用独立Fn_hat、J、M重建vout，线速度atol3e-5m/s、角速度atol3e-4rad/s，均另rtol5e-4；六DOF广义力与矩不能只核合力；
- 独立mj_integratePos(qin副本,实际vout,stored h)：位置atol2e-7m、四元数sign-invariant atol3e-7，实际quatnorm²误差<=2e-6；
- 公开双方F/M对同参考点互反；world0/bodyorigin/COM矩核atol1e-5Nm+rtol.002、合力atol.002N+rtol.002；公开omega_world=R*vrot，不能把局部角速度当world；
- 首步center为空；tilt与offset若独立闭式所有Fn严格正，使用[M+h²JᵀKJ]完整6×6解再核首步，未正不能按全部active冒通过。偏心首My>0、omega_y>0，倾斜首矩沿Rz(.35)ey为负，姿态须发生真实变化。

窗口工程门：150步完整接受；center空→有载，中心姿态漂移<=2e-4rad；所有world无横向力，x/y漂移<=5e-6m。body z始终(.003,.03)m，线速度norm<.5m/s、角速度norm<20rad/s，相对初态旋转<.15rad；全部cube底角最大压入<3mm。tilt姿态最大变化>1e-4rad、offset>1e-3rad。不要求终态完全静止/水平，不改初态追绿。

保存每步完整入/出q/v、receipt动力学、全部surface/quad/目录、真实raw设备观察、一次measure_many双方多参考点wrench、原生oracle和差异汇总、实际K/重力势及每步冻结J/phi代理BE余额。旋转/换区下不把ΣdA*p²/(2g)称全轨迹真实势能或用其守恒作门。失败保接受前缀，若几何进入未支持g/sharedface域则明确记录，不能过滤绕过。

预定最多两HOST（fixture/独立参考）＋一完整CUDA窗口。通过后只从同次轨迹生成三格姿态/接触力/角度图，不重跑仿真。若发生实质问题按一次明确诊断修复对应实现；不扩大证明或调参数。06:35进入关闭/报告，07:10交总览。

## 04:45 I3t：源码外 wheel 的新面积能力边界

I3s 的150步真实FREE窗口已完成；第一次pytest为2HOST通过/1个保存审计断言失败，原因是固定plate原点的FP32表示与原FP64局部顶面相加不严格等于零。保留原失败，只修独立oracle的同入步平面绑定，原450world-step保存审计通过，没有重跑物理窗口。媒体只读该轨迹。

I3t不重跑长窗，验证本轮新增生产模块确实进入可安装wheel并能从源码外使用。root唯一执行owner；原training agent仅author NEW scripts/verify_area_pressure_wheel.py。冻结当前tracked src、pyproject/README/LICENSE等构建输入到独立stage，保留working bytes。用已有Python3.11 build/setuptools无隔离无网络构建；记录完整源清单及wheel所有Python/资产集合与字节同一性。CUDA313环境以--no-deps/--no-index/--no-compile安装本地wheel到全新target，不升级或复制依赖。消费者在checkout外运行，-I、清PYTHONPATH；拒预载phymeo与目标外phymeo候选，显式加入target，finally逐个核loaded phymeo module origin。现有依赖来自原venv及其明确.pth路径，记录而不冒全新依赖环境。

固定2model/3world，顺序(41,7,305)，h=.001。area模型/资产沿I3s，41 qz=.00926425、v0，305 qz=.0095、vz=-.02，其他自由坐标零/identity。7复用I3l native-only垂直sphere：m=.3、r=.01、geom center .012、q=-.003、v0、D70、gravity=-9.81、condim1/friction0。所有XML/资产内联消费者，禁止import experiments/tests。area bundle从原数组构造→bytes→from_bytes→assets_for实际用于scene。mixed intersector(mode=cuda,capacity=64,fallback=exact_host)，两area各24 raw pairs合48不越容量；承认HOST精确复核，不能声称全设备geometry。

仅四次真实整批step：step1→checkpoint→step2→restorecheckpoint→重复step2→selected reset(305)→step3。step2/replay比较actual q/v/time/tick及完整raw numeric arrays逐字节；owner/revision等token逐项独立绑定，不任意normalize。reset后只选定世界清历史，其他世界保留；fresh后继标签独立核。每次一次公开measure_many，area双方多参考点合力/矩与raw积分独立核，native世界面积计数零且存在原生接触载荷。全部world状态实际改变、Fn非零，raw dA正/身份与对应frame绑定；sum(dA)仍真实面，不再归一。checkpoint仅相同配置恢复；固定失败路径/可恢复错误沿现公共API，不另改生产实现。

保完整资产bytes、每步原输入/输出、raw观察、wrench、checkpoint、reset观察、intersector统计、依赖/模块origin和文件hash。CLI接收--site、--output绝对路径且output新建；原本脚本在repo存放，实际执行复制到外部consumer目录。若新wheel暴露导出/包数据或导入问题，只修该问题，不跑旧全套或调物理参数。06:35优先报告/收口，07:10停止；不push/devmerge/发包。

## 05:05 I3u：可复用128点设备摩擦求解模块

I3r的8点参考不足以容纳完整有限面。I3u只将已验证算术提升为可安装backend模块并扩完整样本容量，不接scene/type7/v1观察，不推进q，不称生产摩擦已接通。新增且仅新增 src/phymeo/solvers/area_friction.py（owned问题/样本）、src/phymeo/backends/tyx_area_friction.py、src/phymeo/backends/_tyx_area_friction_kernels.py 和 tests/test_tyx_area_friction.py。原scene agent只author三source，原training agent只author测试；root唯一执行与旧接缝owner。原experiment I3q/r保持不变，不重跑已绿旧组。

公开 AreaFrictionSample/AreaFrictionProblem 字段沿I3q；TyxAreaFrictionWorkspace(capacity=32,mode="cuda").solve_many(tuple(problem),max_iterations=24,max_backtracks=12)，返回TyxAreaFrictionResult，owned原/FP32问题、全部Fn/Ft/双方world0矩/广义力/trace/独立代回与原严格门诊断沿I3r。LAW="final_normal_smooth_v1"明确只此律。最大32world、6DOF、128完整samples；0sample合法，129/7DOF/33world先拒，无截断/自动降级。必须修改schema、pack、设备初始化及样本loop全部相关边界，不能误改8个vector scratch slot。不共享experiment高级实现；生产文件不能import experiments。沿原DeviceResources/原子last_results/失败poison/UNKNOWN_RELEASE语义。

手计14buffers每world 23076+252*S字节，S128/W32=1770624B；每solve上传761216B、下载1009408B，trace两数组705280B不增。24updates×12backtracks仍最多289次样本评价，满32×128最多1183744次samplelaw与42614784次6×6 Jacobian项更新。保留非对称Jacobian、部分选主元、原FP32门（设备1e-5质量scale、HOST代回2e-5），原HOST1e-10只如实报告。没有速度通过门。

固定新128点制造例是两个可动平面刚体的6DOF，origins zA=.01/zB=-.01，128个不同x_i=(2i-127)*.000125、point=(x_i,0,0)。JA x行=(1,0,-.01,0,0,0)、z行=(0,1,-x_i,0,0,0)；JB x行=(0,0,0,1,0,.01)、z行=(0,0,0,0,1,-x_i)，y行0，n=ez。v*=(.00575,-.002,.4,.002,.001,-.1)，vt=.00075 ex、vn_i=-.003-.5*x_i，vs=.001，所以Ft=-.6 mu Fn ex精确有理。k_i=50+10*(i%4)、phi_i=-(.0004+.00001*(i%5))、mu_i=.1*(1+i%3)，全点不同杠杆/载荷且压紧。Mdiag=(.5,.5,.0002,1.25,1.25,.0003)、D=(1,2,.001,2,3,.002)、h=.001、v0=0；独立Fraction计算Fn与Q，再设f=(M+hD)v*/h-Q。禁止调用新solver制造oracle或重复一点拆k。

另固定12点holdout：将上述前12点及所有JA/JB/point/n统一旋转Q，其columns为((.6,.8,0),(-.64,.48,-.6),(-.48,.36,.8))，得到斜法向和两个非零世界切向分量；所有标量/M/D/v*及独立制造f按对应12点重新算。不是新物理窗口；用于防仅x切向/忽略JB错误。128案例另以完整样本逆序核总载荷和速度工程等价，保存原顺序，不要求异序求和bitwise。

验证新schema owned/非法dtype与非SPD/正参数FP32下溢；单world128与单world12；32world交替128/12/0，原v0/f固定、最后world128，按身份与W1工程比较；128点原序/逆序；坏最后problem零launch、数值预算0末world失败保last_results、随后可重试、关闭释放。沿原速度5e-5*(1+abs(ref))、力.002+.002rel、矩1e-5+.002rel，另外直接代回实际上传问题/实际设备v核全部方程及最终Fn容量；不只比较制造目标。全部buffer/trace/字节和真实时间留存，不把新容量叫真实32机器人场景。一个首次focused组，失败保前缀并只诊断对应问题，不放宽门。

完成后新wheel须包含这三文件并验证安装来源，原I3t混合场景结果仍绑定其原wheel，不倒灌新版本。若06:35仍未通过则保明确未完成状态并收尾，不赶时间接新scene或改旧合同。07:10交本轮总览。

## 05:30 I3v：真实入步数据的摩擦装配接口

I3u首次4节点通过。下一小批只做NEW src/phymeo/backends/mujoco_area_friction.py与tests/test_mujoco_area_friction.py，原scene agent author source、training agent完成outside consumer后author tests，root独占执行。禁止改旧Scene/type7/v1，不把新摩擦解再次施加到已正常推进的状态。

API：AreaFrictionPairLaw(identity,mu,slip_speed_m_s)；MuJoCoAreaFrictionAdapter(model_spec,pair_laws)；prepare_from_receipt(receipt,layout=...,quadratures=...)返回owned PreparedAreaFrictionEntry(problem,quadratures,pair_laws,sample_catalog,binding,scope)。scope固定historical-same-entry-counterfactual-no-integration。模型静态信息从source-bound MJB重建仅读取，无FK/step。核三source SHA、layout尺寸/modelkey、完整声明pairs和材料身份、历史world/tick/time/configuration、receipt/dynamics q/v/time/phase，完整quad含空；不把历史configuration与restore后当前owner比较。任何剩余native可能pair/limit/dry friction/activation/BALL先拒。1..6DOF、完整样本<=128；三点每三角的quad最多126点，129明确拒，不能截尾。

精度明确：M/J/COM/bias/passive/applied/v0为receipt实际设备数据的FP64解码；几何为source_quadrature原FP64 point/normal/dA/p/g；完整JA/JB在该原point重建。h=RN32(receipt.h)，D=RN32(source dof_damping)。f=applied+passive-bias+D*v0，禁止加入旧constraint力。新backend最后统一RN32 pack。与旧type7 packed-point算术只作工程比较，不保证逐字节同一。adapter没有伪造live能力；未来生产F2必须在积分前消费同阶段device数据，不能下载post-step receipt后再施力。

固定新验证：一批两个模型各一实际step，不重跑I3s150步。模型A沿I3s完整cube/finiteplate资产，mass=.3/inertia同原，FREE初态沿tilted7，新增COM=(.001,-.002,.001)m与v0=(.02,-.01,-.005,.1,.2,-.1)，D0；原source pose与材料保持，目的是非对角M与完整角向pointJ。模型B水平slide，mass=.5、D40、h=.001、gravity0、v0=.02，无外载/控制，压力体初态高于同有限plate，完整空面。正确自由速度为m*v0/(m+hD)，漏+Dv0会重复阻尼，需明确区分。两模型native geom collision关闭，无limit/activation，剩余原生约束拒。

独立CPU只mj_forward/mj_fullM/mj_jac，在actual entry q/v/controls计算M、所有双侧pointJ和bias/passive/applied，不mj_step。门沿I3s：M rtol3e-4并按平移/转动块2e-6/2e-9、其他2e-8；J rtol2e-5/a2e-6，力项rtol3e-4/a2e-5；f=D反消为0的浮点误差单独保存。生产实际normal step只用于获得真实入步字段，不把它称摩擦轨迹。

由A同入步分别构造mu=.5与mu=0、vs=.001，B构造mu=.5空面；一个新CUDA solve_many共同求三条counterfactual问题。保存完整来源和结果；新律用I3u固定FP32残差/力矩门，µ0用独立全active线性解（先核所有Fn>0）或有限active-set oracle，不能调用旧solver当唯一oracle；D空解独立解析。全部新输出不回写场景；调用前后snapshot与revision保持。负控含错source/layout/phase/qv/time/identity/遗漏pair/非area rows/129样本/immutable alias，不能通过删字段或放宽误差门。

I3v完成后构建包含新增backend/adapter的fresh wheel，旧I3t混合场景证据仍绑定旧wheel；新源码外consumer仅一真实128点CUDA解及完整原始输出比对。06:35起收尾，07:10停止。
## 05:55 I3w：设备驻留布局直接装配的下一原子批

I3v首次3节点通过，新wheel源码外128点消费通过。剩余时间增加一个独立设备接缝，不改旧advance/积分/Scene/type7/v1，不把历史输入重放称为live零读回生产。新增且仅新增 backends/_tyx_area_friction_entry.py、tests/test_tyx_area_friction_entry_pack.py、固定fixtures/area_friction_entry/entry.json.gz与README；原scene agent author模块，原training agent author测试，root唯一执行。父只复制I3v完整原证据72010B/SHA50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6作为可移植fixture，不重新运行scene。

固定源layout：M[W,P,P]、Jcom[W,B,6,P]（前三world linear、后三world angular）、motion[W,B,7,3]（slot5=COM）、v/bias/passive/applied[W,P]、vectors[9,max(8,nq)]（row1=D）、FP32 h；P=max(8,nv)。area_input[W,C,8]=(point3,n3,k,phi)、pair_index[W,C]、sizes[W]、body_ids[pair,2]、新pair_law[pair,2]=(mu,vs)。几何是原实际打包FP32点，不能与I3v原FP64点假称逐字节等价。

一个metadata-only wrapper核实际数组shape/dtype/backend、整数extent和源/输出别名；不读取设备值。一个pack kernel清所有输出与padding，验证上游status、完整count<=128、全部pair/body索引及有限参数/物理域后，写I3u七输入：sizes、M、(D,v0,f)、h、完整10字段sample、JA、JB。f=applied+passive-bias+Dv0；Jpoint=Jcom_linear+Jcom_angular_column×(point-COM)。不能改D、重归一normal或截尾。只支持1..6DOF、1..32world，原128完整点边界。

失败独立pack_status保留，上游status原buffer不改；失败lane的sizes nv=0/ns=0使已有solve_batch先置meta1并跳过数值分支。不要让solver清meta掩盖pack错误，不额外增加重复gate kernel。caller需在最终下载后同时核pack_status和solve meta，全批成功才发布。本批没有生产场景publish，不能宣称生产事务已接通。

验证同一个runtime pack→现solve_batch，之间不得download或HOST重新pack，随后保存全部source、七实际pack输出、solverraw/trace和错误码。两个固定pipeline：nv6/W4前三为actualA mu=.5、actualA mu0、actualA空面；第四明确synthetic-layout-only，复制A但plate COM=(.01,-.02,.005)、plate J=FP32(.25*pressure J)，防JB硬编码0，不称额外真实物理。另nv1/W1为实际空面D40滑块。独立HOST在实际packed point上核完整J/f与全部方程、原FP32误差门；不调用新pack制造oracle。实际输入fixture保持原字节。

错误覆盖：最后world上游失败、非法count/pair/body索引、NaN/zero normal/非法mu/vs/k/phi/h、算术溢出；不能越界/静默降级/回旧solver成功。metadata非法extent/dtype/alias先拒零launch；设备失败完整保原始字，不以成功前缀冒整批发布。允许原pipeline失败后明确新输入重试，所有状态从本次完整清理得到。

目标06:20纯author交接、06:35前首次focused验证；若出现实现问题只修明确根因、不改fixture/门槛。没有时间则保UNEXECUTED/失败原证据并报告，不把独立pack包装成F3。07:10停止。
