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