# P6.5 人工续作入口

## 当前安排（2026-11-09）

**H1本机首个受载窗口通过（11月10日）**：执行器反射惯量/隐式阻尼已接入HOST、共享场景与普通浮根设备；HOST44项+MuJoCo8例、实际G1三个姿态动力学通过。32关节有界容量的interpret/CUDA各7专项+2相邻，以及真实29关节G1各两步通过。有限双脚静态/偏载、528步离地重入和拒步通过。G1使用原始q/v与官方反馈策略，在Phymeo HOST以h=.0005完成2秒/4000步，稳定段平均支撑99.9624%体重，未跌倒；默认姿态PD的两次站立失败仍保留。详[本批结果与精度边界](../benchmarks/p65-h1-actuator-capacity-support.md)。下一步为完整20秒反馈舞蹈与设备受载窗口；固定材料站点的接触片转换、非零affine执行器和W32训练吞吐仍开放。不重复本批已绿组。

**训练首批入口保持**：ResidentVectorEnv的interpret/CUDA各5项、独立MuJoCo完整G1参考20秒已完成；不是大规模训练或Phymeo完整舞蹈验收。详[首批历史](../benchmarks/p65-training-humanoid-first.md)。


用户最新要求：立即收束 K0，提高大规模训练优先级，并使用真实人形机器人和公开驱动策略完成舞蹈。**K0 默认距离/见证已完成本机限域验收**：最终 interpret/CUDA 各142通过，284完整gzip留存，新wheel源码外各4接受，440产品成员同源；见[正式报告](../benchmarks/p65-device-convex-query-acceptance.md)。新的最高优先主线是[训练与G1舞蹈计划](../superpowers/plans/2026-11-09-training-humanoid-priority.md)。不重建长期goal或周期任务。所有项目执行仍父唯一owner、严格串行；已有可选K1作者内容继续保持未验收。原11月8日缺明细记录见K0计划§12，本轮完整留档见§13。

**C5固定根affine-mimic设备树已正式限域接受**：HOST13、解释89、CUDA89，独立Newton CPU/CUDA共400步，301固定保存审计及新wheel源码外五混合世界305 world-steps完成。详[验收报告](../benchmarks/p65-resident-affine-tree.md)。C4普通完整树、F3浮根惯量与F01–F06摩擦接受继续复用；禁止重复同源已绿组、旧wheel或完整证据审计。C5不是设备接触反馈、浮根设备树或CCD通过。

**抓取应用当前边界**：原material模式四窗280–283均已执行，Panda完成3006步/1.87875s，原八门通过；G01/G02/G03在594/618/580接受步因原功/运动门拒绝。297精确复现三个原请求，312/319新作用点模型和相邻兼容通过。新显式motion_reference=application使J/W/力矩/弹簧功使用同一对入口作用点；默认material保旧门，模式绑定episode，不热切换。320只对297已保存候选做独立功坐标算术对照：三组application门成立，0物理步、无历史重新解释；这不等于新模型可达轨迹或持稳通过。321–323从原初态的application完整窗口已结束：G01实现原获取/20mm抬升/100ms持稳，1420步后释放阶段总功门拒绝；G02零摩擦1681步完整滑落、释放并恢复台面支撑，通过原八门；G03超载597步后总功门拒绝，不能叫完整负控通过。324仅精确复现两拒绝请求，七字段保留，原总功残差为+1.002949414e-8/-1.165306988e-8 J，逐站与运动门通过；没有重复释放证据。该工程精度边界保留，独立C6继续，不再重复绿窗或长前缀。详[原四窗计划T3](../superpowers/plans/2026-11-07-persistent-grasp-holdouts.md)及[专项报告](../benchmarks/p65-tangent-application-reference.md)。

**C6常驻有限静态接触反馈已正式限域接受**：产品48eeb89，HOST桥50、解释74分组、实际CUDA74、相邻15通过；112完整payload审计与新wheel两backend各41 batch/164 world-steps完成。包含有限球/胶囊/凸体、真实偏心转动/跨分支惯量、732/1463步卸载重入、W1/W4及容量/晚故障共同发布。详[正式接受](../benchmarks/p65-resident-tree-contacts.md#2026-11-08-父执行与正式限域接受)。本片不是affine/floating接触、动态双侧、持续摩擦、CCD或全GPU窄相，不重复已绿组/旧审计。

**C7固定根affine常驻接触反馈已正式限域接受**：产品876ae848，每backend50专项联合覆盖、90主保存记录与两摘要补件、新wheel均通过。详[正式报告](../benchmarks/p65-resident-affine-contacts.md#2026-11-08-父执行与正式限域接受)。旧generic摘要差异与原失败保持，不重跑已绿组。
**C8完整固定根tree/forest内双动态反馈已正式限域接受**：产品13c3fad，各backend74项联合覆盖、112完整保存payload、Newton原生160步及新wheel双backend各251接受+1拒绝world-step完成。CPU原力观察错误由同一80步独立解码补件闭环，原RAW1保留；无物理重跑或matched-law回退。完整源/安装/恢复/材料记录审计均通过，详[正式报告](../benchmarks/p65-resident-tree-dynamic-contacts.md#2026-11-08-父执行与正式限域接受)。C8原执行槽已释放。

**C9普通浮根完整设备树已正式限域接受**：产品4927ef9，HOST15、解释分组86、最终源码完整CUDA86、相邻双backend各8、独立Newton七例每端280步，以及新wheel两backend各258接受/2拒绝world-step完成。最终495只读保存审计核20原始任务、172完整gzip、7284绑定文件，RAW0；原FP32下溢、精确pi伪陀螺及观察器首失败与不同源码通过范围全部保留，详[正式报告](../benchmarks/p65-resident-floating-tree.md)。C9项目执行已全部结束，不重跑同源绿组或同次保存审计。当前能力含真实root COM/世界角速度、根–关节完整M/b/J、0–16关节、PD、可逆local检查点、当前姿态设备宽相与原事务；尚非浮根接触或affine。

**C10普通浮根接触已正式限域接受，保留URDF来源限定**：[正式报告](../benchmarks/p65-resident-floating-contacts.md)。interpret分组与CUDA各76唯一节点、相邻各3、独立Newton原生每端80步、新wheel源码外每端302接受/2拒绝world-step完成。541保存补件RAW0核14原任务/133完整gzip，复用原9项完整审计、只补5项；原URDF漏快照以明确retrospective qualification保留，不能称同次raw来源齐全。三项观察器首失败与536均留证。431产品构建成员全同。C10项目执行已结束，不重复绿组或已完审计；能力仍限ordinary浮根、HOST有限mu0离散窄相及TYX完整响应。

**C11浮根affine/mimic已正式限域接受**：[正式报告](../benchmarks/p65-resident-floating-affine.md)。每backend核心52唯一节点/46完整保存身份、相邻各6、W32各4步；Newton原生每端9步、新wheel源码外每端266接受/2拒绝world-step全部完成。587同次保存审计RAW0核20原任务/7751文件，102完整输出及3原失败产物分开记录，435产品成员与wheel逐字节一致。完整6+n载荷→6+K归约、原关系行反力/root余额和共同发布已验；553保存token、580观察器零反力集合首失败均保留。所有C11执行结束，不重复已绿组/同次审计。K0默认设备查询现已限域验收，见[报告](../benchmarks/p65-device-convex-query-acceptance.md)。K1可选patch仍未验收，K2真实消费者尚未冻结。

[功能差异全景](phymeo-newton-panorama-2026-11-07.md) · [可视化](phymeo-newton-panorama-2026-11-07.html)。完整设备窄相/CCD、跨独立participant的设备共同求解、抓取释放工程精度边界和软布两初态放台等仍开放；历史数学证明或局部几何不代表整个P6.5完成。

## 最近已验收基线（2026-11-06）

[第七批：浮根共同场景、逐面设备候选与局部复用](../benchmarks/p65-top-three-batch7.md)已本地集成，产品8128440，新wheel已在源码外通过三线组合。浮根专项43+共享回归73与40步承重/撤离通过；B逐面HOST44+故障1、interpret4/CUDA4通过；C复用HOST29、interpret2/CUDA2通过。三原作者交叉审阅完成。

C的W1F8/W32F8三对AB/BA/AB共3960 world-substeps完整输出与计数一致；W32 p50比值1.008/0.957/1.006，无稳定规模收益，默认False并收束此候选。下一项先调研/冻结非球机器人互碰多案例、转动mesh–round逐面候选、已接受设备状态直接供宽相的公共合同。浮mimic/fixed折叠/pads、全设备窄相、目标软布两初态放台及原mimic受载失败仍开放。所有本批执行槽释放，无自动任务。[已执行计划](../superpowers/plans/2026-11-06-top-three-batch7.md) · [大模块差异](phymeo-newton-functional-overview-2026-11-06.md)

### 11月6日第六批（历史）

[第六批：真实机器人互碰、双转轴与分页成本](../benchmarks/p65-top-three-batch6.md)已本地集成，产品 `3e6dff4`，新wheel在源码外通过。多固定根普通/mimic树可直接球–球/有限胶囊/OBB双侧受力；闭网格与球/有限胶囊支持双方独立旋转和移动枢轴。A专项25+共享21、B HOST18+interpret1+CUDA1+受影响兼容4通过，公开40步保持/撤离通过。

C工具27协议+解释7+CUDA7通过，五档正式CUDA与独立未插桩完整输出一致。稠密W32×8面额外分页仅占0.155%，未触发15%优化门，默认保持；后续优先分析HOST几何/材料装配。原作者交叉审阅和父证据读取结束，所有执行槽释放，无自动任务。下一项是A3浮根共同事务、B3逐面设备候选及C热点拆解，先处理B/C接口依赖。非球机器人互碰、全设备窄相、原软布两初态放台及mimic原失败仍开放。[本批计划](../superpowers/plans/2026-11-06-top-three-batch6.md) · [当前大模块差异](phymeo-newton-functional-overview-2026-11-06.md)

### 11月6日第五批（历史）

[第五批：多机器人、胶囊转动与选择观测](../benchmarks/p65-top-three-batch5.md)已本地集成，产品 `dddb612`，源码外 wheel 通过。新增同一 HOST world 内多个普通/mimic 固定根机器人共同作用与原子发布；胶囊自身转弧对无转动闭网格的有限特征连续检测；显式 TYX 选择读回计划。三原作者隔离编码、父串行验证与共享接缝集成。

18组同输出配对在W32/W128选8个世界的完整snapshot显示4.44–4.81倍p50局部收益；W1及kinematics更慢，全部旧默认保持。关闭异常负控发现并修复了剩余计划未清理问题。多机器人直接互碰/浮根、双方独立转轴和更广设备驻留仍未覆盖；软布两初态放台与原mimic受载失败不变。下一批按优先顺序推进A2双机器人直接接触、B2双方独立转轴、C2材料/力读回合并，先冻结各自新案例与不交叉文件范围。[已实施的前三线计划](../superpowers/plans/2026-11-06-priority-top-three-parallel.md) · [较大功能差异](phymeo-newton-functional-overview-2026-11-06.md)。本批所有项目执行已结束，无新执行槽或自动任务。

### 11月6日第四批

[第四批：真实场景耦合切向与固定凸包复用](../benchmarks/p65-scene-tangent-batch4.md)已完成本地集成、119 HOST + interpret 1 + CUDA 1、静态检查及源码外 wheel 使用验证。产品 8c4a7db：真实法向、全跨站点响应、材料/施力点实际位移和功审核进入同一步共同发布；两质量压载、URDF 双站点、卸载重入与回滚均有实测。新模式仍限显式材料站点的 HOST 刚体/普通移动关节，不覆盖粒子/柔体、mimic 或 revolute 机器人；原 mimic 100g 失败与两种软布未放台保持。

固定 hull 边界/chart 在完整绑定后复用，四档24帧输出一致；W32 CUDA 的两个暖态观察 step+geometry p50 为207.183→89.495ms，只是本引擎局部对照。所有本批执行结束，原 agents 已释放。[已完成计划](../superpowers/plans/2026-11-06-scene-coupled-tangent-batch4.md) · [较大功能差异与下一优先级](phymeo-newton-functional-overview-2026-11-06.md)。

用户已重新指定顺序：场景与机器人 → 碰撞检测 → GPU与多场景 → 可微仿真与训练工具 → 传感与交互 → 跨域求解编排 → 布与体软体 → 杆缆、MPM/流体，覆盖旧优先级。前三线按 [子 agent 分工计划](../superpowers/plans/2026-11-06-priority-top-three-parallel.md) 的首片已完成实施与本地验收；后续依旧采用独立编码并行、父验证串行。持续摩擦随相关场景的实际阻塞推进，不另置于整个路线之前。相机、在线Python控制和mimic图已存在，不能按旧清单重复开发。

### 11月6日第三批

[第三批：耦合切向状态与受载诊断](../benchmarks/p65-coupled-tangent-loaded-batch3.md)已完成本地集成、63项专项、193步真实组合窗口和源码外wheel使用验证。新增HOST完整交叉W切向核、显式材料站点历史与纯事务；默认scene摩擦未自动改变，原100g mimic失败仍保留。CUDA同进程4档256帧完整一致，W32受载/出生有局部过滤收益，W1无一致收益；发现重复HullShape边界/chart构造，插桩时间不当无插桩占比。下一批从共享场景适配继续：真实法向采样、完整自由运动与反作用、实际材料位移/能量审核、共同发布；用有限双面压载、双站点URDF滑轨、偏心转动留出分别验收。性能可并行做不变hull边界/chart复用，但先保完整输出与全部检查。详见[本批已执行计划](../superpowers/plans/2026-11-05-coupled-tangent-loaded-batch3.md)。

### 11月5日第二批

[获取控制、统一事件与过滤复用第二批](../benchmarks/p65-force-events-filtered-batch2.md)已合入并通过源码外wheel使用验证。新增接受步力反馈控制、完整多component的两scene统一contact_events，以及保全局输出的过滤行复用；80项专项分组通过，九档576帧与旧版完整相同。Panda无预压获取/抬持/释放通过，mimic100g箱在1.048s滑移超3mm，原RAW1保持；零摩擦/超载负控符合预期。性能只在free W128观察到1.56x，受载/出生本次更慢，不作整体加速结论。下一批优先通用持续切向摩擦/材料身份和受载窄相成本，而不是调原夹力或重复旧窗口。[本批计划与收尾](../superpowers/plans/2026-11-05-force-events-filtered-batch2.md)。

### 同日第一批

08:00之后的[场景、接触与规模第一批](../benchmarks/p65-scene-contact-scale-batch1.md)已完成集成和源码外wheel验收。公共构建器现可登记一个URDF、多个自由球/箱及有限台，普通场景增加不可变接触生命周期；九档CUDA规模矩阵576步通过。Panda预载保持/释放288步通过，mimic192步能运行但连续持稳门仍失败并保留。下一批从报告的A2通用获取控制、B2材料/事件统一、C2完整过滤记录复用继续；未将预载释放等同完整抓取。

本轮新增功能、最终源码外 wheel 验收、限域性能与尚未达标的数值/软布案例见[08:00 总结](../benchmarks/p65-newton-20261105-0800.md)和[可点击概览](../benchmarks/p65-newton-20261105-overview.html)。下一轮从[场景编排、通用持续接触与 GPU 规模计划](../superpowers/plans/2026-11-05-after-night-newton-parity.md)选择新的工作节点。

北京时间 11 月 5 日凌晨的工作以[08:00 批次计划](../superpowers/plans/2026-11-05-newton-overnight-0800.md)和[分模块实测](../benchmarks/p65-newton-20261105-0800.md)为入口。按刚体与关节、柔性接触、运行时观测和 GPU 多世界分别验收，保留本机 Newton 1.7.0 的参照以及尚未通过的案例。

### 同日较早结果

自由球已能向非指垫机器人连杆施加反力，50 mm 球撞臂的真实响应已在 Phymeo 和 Newton 独立推进。小折角材料片可带压换面，实际离散角动量偏差及同窗细分已记录。W4×8 的重复材料记录重建已消除，合批物理窗较上一版缩短约 13.3%，与同几何的普通逐世界调用耗时接近。[三线结果与下一批判据](../superpowers/plans/2026-11-05-structural-impact-fold-and-host-cost.md) · [首碰与峰值受力回放](../../artifacts/p65-fold-impact-and-batch-20261105-01/panda-link-impact.html)

### 同日上一批

多面接触现可跨世界共用 TYX 设备提交，W4×8 核数从 48 降至 12，墙钟尚未加速。独立布节点已在共面应变下完成受压换面。Panda 在两引擎中都搬运了 40 mm 实心球，50 mm 球的失持和首次未求解结构接触按真实时刻记录。[集成结果与下一批判据](../superpowers/plans/2026-11-05-ragged-contact-and-payloads.md) · [真实球形回放](../../artifacts/p65-ragged-seam-payload-20261105-01/panda-grasp-replay.html)

## 2026-11-04 续作记录

已验证共面平移材料片的带压跨边、旋转偏心长方体抓放，以及 2/8 个真实受载面在 W1/W4 上的成本。当前多面法向层可在 TYX 运行，跨世界接触批入口仍限定每世界单面。下一批优先验证自由节点换面与多面变长站点合批。[验收、数据和下一批判据](../superpowers/plans/2026-11-04-shared-edge-offset-and-density.md) · [偏心抓取与滑落的交互回放](../../artifacts/p65-crossing-rotated-density-20261104-01/panda-grasp-replay.html)

上一批已合入。Newton Featherstone CPU 采用同一材料接触律后，Panda 强夹完成抓放，弱夹仍留台；可动布面支持带预载共转和转滑；32 个混合世界的三个独立同场 CUDA 测量分别加速 4.64×、4.97×、4.45×，逐帧输出一致。下一批从受压跨面、旋转偏心抓取和更多受载面下的剖析继续。[本批证据与下批入口](../superpowers/plans/2026-11-04-contact-model-and-geometry.md)

[受载接触第一批分工与结果](../superpowers/plans/2026-11-04-contact-load-parallel.md)已集成平面–凸包支撑、有限布面静持/转滑、长方体强弱夹持与同姿态几何复核。Phymeo 的 120 g 强夹已放台，Newton Featherstone 同输入原生接触尚未复现整个受载过程。该批 W32 优化前的计量中单姿态最近特征查询占插桩墙钟约 86%。

[与 Newton 1.7.0 的逐项功能盘点](phymeo-newton-1.7-functional-inventory.md)覆盖 19 个功能族，并区分物理实测、同引擎加速与跨引擎对照。测试工具开源已列入后续待优化工作。

**经典 Panda 和自由方块（11月4日，08:00 后的集成批次）。** 打开[抓放例程与实测](../benchmarks/p65-panda-rigid-grasp.md)：从官方经典 Panda URDF 和真实关节惯量出发，100 g 方块在指面摩擦 0.65 时离台 59.9 mm、横移 77.4 mm、放下后桌面承重 0.981 N；零指面摩擦时留在桌面。相同资产的 17 个真实入步姿态已由 Newton CPU/CUDA 核实两侧指面位置。三条独立物理工作线已补[凸面多点冲量](../benchmarks/p65-rigid-polygon-impulses.md)、[旋转网格与有限杆身](../benchmarks/p65-rotating-mesh-finite-capsule.md)、[双铰链连续球路径](../benchmarks/p65-two-hinge-ellipse.md)和[有限布面离边动摩擦](../benchmarks/p65-dd-material-sliding.md)。[同输出 W32 混合受载合批](../benchmarks/p65-mixed-physical-w32.md)已达三轮 1.33–1.48 倍的本引擎逐世界吞吐。

下一批优先复用同一 Panda 资产与两条保存轨迹，扩展抓取起始姿态、质量和夹持力，在 Newton 做相同初态与控制定义的受载对照。通用碰撞线从胶囊自身独立旋转、跨步抓持面、双铰链非球工具和柔性材料的静止持载继续；GPU 线按 W32 的读回、复杂窄相与 HOST 审计分别定位时间。软布递交保留为另一条集成验收。

**08:00 阶段概览。** 刚体单转轴闭网格–球 CCD、柔体顶点–面与边–边完整冲击、浮动树持续承载和真实双端线支撑，以及 W1/W32 与 Newton、W4/W16 真实混合世界的吞吐对照已有实测。源码外 wheel 完成参考布九阶段递交，目标软布两种倾角仍未完成放台。下一批从旋转闭网格–胶囊、跨步材料摩擦、多活动关节链和带载 GPU 的分项计时继续。[可放大概览图与力曲线](../benchmarks/p65-newton-20261104-overview.html) · [数据、开放项与下一批验收](../benchmarks/p65-newton-20261104-0800.md)


- 机器人已从移动参考系观测推进到浮动基座与子树的动力学、固定基座的双边点闭环及带实际加速度的站点 IMU。闭环反力、两端功率和离散投影分别记录。IMU 的瞬时样本和 Newton 步平均加速度按各自时刻对照。[浮动树](../benchmarks/p65-floating-tree-dynamics.md) · [四杆反力](../benchmarks/p65-robot-point-closures.md) · [站点 IMU](../benchmarks/p65-robot-site-imu.md)
- 刚体几何已接通闭网格与球/有限胶囊的实体查询、受限单相对轴的闭网格旋转首触，以及有界曲面之间和曲面与有限布面的材料见证。首触时刻、末态边界净距和整体退让深度为不同观测。[闭网格与圆体](../benchmarks/p65-mesh-round-contact.md) · [转动闭体](../benchmarks/p65-rotating-mesh-solid.md) · [曲面与布](../benchmarks/p65-convex-support-pairs.md)
- GPU 已有独立的世界内配对、常驻自由球/滑轨推进和多世界布–连杆双向受力。相同局部编号的 RR、RD、DD 和空世界也经过同批查询。GPU 启动、必要读回与 HOST 几何复核分别计数。[世界内配对](../benchmarks/p65-world-partition.md) · [G2 常驻推进](../benchmarks/p65-g2-resident-world-step.md) · [G3 双向受力](../benchmarks/p65-g3-world-contact-loads.md) · [混合世界回归](../../tests/scale/test_mixed_world_queries.py)
- 本轮继续检验多目标 IK、凹闭网格与完整柔性三角面、稠密单世界规模及曲面几何产生的真实物理载荷。阶段签收使用多个独立材料/运动反例。两种软布初态递布沿用同一条真实轨迹的接管、桌面支撑和撤爪条件。[应用基线](../benchmarks/p65-soft-handoff-acquisition.md) · [规模计划](../superpowers/plans/2026-11-03-tyx-gpu-multiworld.md)

## 2026-11-03 进展和入口

- 当前可并行的三项是移动参考系传感、非零转动刚体首触试片和 TYX 世界内配对。固定基座止挡、延迟驱动及设备地形候选已经组合验证。一台滑轨因限位在半步时触地，另一台共址滑轨保持净空。G1 已测到 64 个世界，单世界扰动没有串扰。在该档位，完整配对表保留了 32,256 个跨世界过滤行，暖态合批查询慢于逐世界调用。[组合验收](../benchmarks/p65-driven-terrain-worlds.md) · [G1 成本](../benchmarks/p65-g1-world-isolation.md) · [下一批子任务分派与验收](../superpowers/plans/2026-11-03-p65-next-parallel-dispatch.md)

- 前一批接通了固定基座 URDF、MJCF 与可选 USD、双闭合凹网格的平移首触及三类静态曲面，并验证了 USD 工具球与圆柱的组合。[资产与刚体实测](../benchmarks/p65-independent-rigid-usd-20261103.md) · [地形与资产](../benchmarks/p65-robot-terrain-20261103.md)

- Newton 1.7.0 的独立 CPU/CUDA 参考环境和刚–刚 13、刚–柔 11、机器人与资产 11 个案例已实测。首轮记录作为差异基线，新增入口的通过结果单列在上方本批实测中。[逐项物理对照](../benchmarks/newton-1.7-reference-baseline.md) · [三方向后续开发计划](../superpowers/plans/2026-11-03-newton-reference-development.md) · [复现入口](../../tests/reference_newton/README.md)

- 公共碰撞检测已覆盖 15 个案例族、74 个登记构型。RR、RD、DD 各有独立受力运动窗口，三端共 636 个物理步。实际 TYX 与 HOST 的职责和各形状支持范围分别记录。[碰撞验收表](../benchmarks/p65-collision-public-catalog.md) · [三个动力学案例](../benchmarks/p65-collision-m4-loads.md)
- 单爪软布已真实承载、离台并弯曲。原双臂装配的两种初态都未放台。材料点跟踪使 2° 布在 B 单独持有时横移最多 9.13 cm；竖直布的 A/B 交接重叠仍不足。[单爪结果](../benchmarks/p65-single-gripper-bend.md) · [双臂对照](../benchmarks/p65-soft-handoff-acquisition.md)
- 已补通用胶囊–布面载荷、铰接关节反力及同源 Three.js 记录。原装配在初态有布–腕重叠，新掌部与两根移动指杆保留全部结构形状，两种软布初态各 240 对配对均分离。两种原协同轨迹已记录。竖直旧轨迹还检出了同一张布的远处材料面横穿；连通折叠试片已在 HOST、tyx interpret 和 CUDA 实现双向受力，并接入真实弹性布子步。完整放台、最终集成审阅和源码外 wheel 分开验收。[结构接触证据](../benchmarks/p65-cloth-link-response.md) · [当前实施计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
- 两种原始软布初态均已完成实测，尚未共同承载和放台。常用 `cooperative` 保留了竖直布 B 的双侧夹持及同轨迹 1,639 个子步的布自接触力。可达地址策略单列为 `aperture-guided` 实验模式，它在 2° 接取窗较早丢布。下一项是用已定位的 B 指杆–布面和 B 腕–布面两个入口先验证提前接近的路径。[双臂实测与后续验收](../benchmarks/p65-soft-handoff-acquisition.md#对照结果与下一项验收) · [竖直与 2° 回放](../benchmarks/p65-soft-handoff-acquisition.md#倾斜软布与两种布边路径)


## 2026-11-02 实施记录

- **碰撞组合持续任务已完成** RR04、RR05、RD03、RD05 已提交。全部 15 族、74 个登记构型已经在一个 164 形状模型中分别运行 HOST、tyx interpret 和实际 CUDA。82 条目标配对记录的完整内容逐字节一致。[验收总表与复现](../benchmarks/p65-collision-public-catalog.md)
- **框架目标** 建立基于 tyx 的通用碰撞检测，分别验收刚体–刚体（RR）、刚体–柔体（RD）、柔体–柔体（DD，含不同物体与自碰撞）。递布是必过的最终系统回归。可记录的小穿透和数值误差沿工程预算处理。
- **M0 首批已运行** RR04 转动扫掠、RD01 刚球撞布面、DD02 两边首触已完成独立 HOST 回归及现有接口对照，12 条逐构型观测。现有接口均未返回这三对物体的首次接触区间。[15 个案例的差异表](../benchmarks/p65-general-collision-baseline.md) · [可运行回归](../../tests/collision/test_case_catalog.py)
- **M0 第二批已运行** RR03 已知轴的裁剪可复用，RD02 沿静态胶囊轴找到穿面见证，DD03 确认旧共享顶点过滤会排除异处相交的面。5 个构型/族，共 15 条观测。[新增回归](../../tests/collision/test_contact_regions.py) · [证据与范围](../benchmarks/p65-general-collision-baseline.md)
- **M0 第三批已运行** RR05、RD04、DD05 有 16 条几何观测。表面正距离同时可能出现在凹槽空区和实体内部；已有拓扑函数可消去两个四面体的共享面，材料所有者需另行绑定。初始包含、退化输入和变形体外部首次接触分别记录。[第三批回归](../../tests/collision/test_solid_boundaries.py) · [实测差异](../benchmarks/p65-general-collision-baseline.md)
- **M0 已完成** 收尾批 RR01、RR02、RD03、RD05、DD01、DD04 有 31 条几何观测。四批共覆盖 15 个基础案例、74 个构型。DD04 保留了先触外面、后触斜面的原始失败和修正记录。[逐案例结果](../benchmarks/p65-general-collision-baseline.md) · [收尾批回归](../../tests/collision/test_detection_matrix.py)
- **M1 公共入口已验收** `phymeo.detect` 可在一个混合模型里筛选 RR/RD/DD。11 个形状的 55 对在 HOST、tyx interpret、实际 CUDA 上逐对一致。静态球、胶囊及球–零厚三角面已接入。尚无扫掠窄相的候选公开标为 `unsupported`。[用法和实测](../benchmarks/p65-collision-m1.md)
- **M2-RR 首批已验收** RR01 球–平面/球–球与 RR02 胶囊轴部/端帽已接到公开入口。运动时独立保存首触与末态几何。两端分离的胶囊穿越、擦边与一刻度近失均有回归。[支持范围与实测](../benchmarks/p65-collision-m2-round.md)
- **M2-RD 的 RD01 已验收** 平移球与双侧零厚三角面可返回整步首触和材料权重。支持三个顶点各自仿射运动，五个登记构型和变形留出通过三端核对。[用法和实测](../benchmarks/p65-collision-m2-sphere-face.md)
- **M2-DD 的 DD01/DD02 已验收** 异体零厚三角面对的点–面、边–边首触可返回两侧材料权重和取样时的交集。九个登记构型、双端分离的穿越和孤立擦碰已有三端实测。[范围、结果与复现](../benchmarks/p65-collision-m2-triangle-pairs.md)
- **DD03 同体邻接已验收** 同一布体按材料地址区别合法重合与异处相交。原五个构型和三条运动路径已通过 HOST、tyx interpret 及实际 CUDA 对照。[用法、范围与实测](../benchmarks/p65-collision-m2-cloth-adjacency.md)
- **RD04/DD04 外表面已集成** 原四面体单元共用同一份定向外面。平移球与零厚布可扫过变形体；厚布在当前恒定有理单位法线范围内返回完整层截面、首次及随后各面事件。十个构型、形状对调与独立负控已通过三端验收。[用法与结果](../benchmarks/p65-collision-m2-tetrahedral-surfaces.md)
- **DD05 闭体占据已验收** 不同物体的正向四面体网格按单元及整物体分别返回包含、交体积、内部材料点与外边界首触。跨接缝和纯边–边首触等独立对照通过三端验收。[用法与实测](../benchmarks/p65-collision-m2-closed-volumes.md)
- **RD02 已验收** 平移胶囊的端帽、完整轴身与非退化仿射零厚布面可共用公开查询。五个原构型、轴身–边首触、平行接触段与采样间孤立擦碰已三端核对。[用法与实测](../benchmarks/p65-collision-m2-capsule-face.md)
- **RR03 已验收** 从无序顶点集构造满维凸刚体，静止或平移查询可返回接触区间、并列支撑区域及双方材料点。滑动箱的面–线–分离、纯边–边首触、孤立擦碰和设备分页通过三端验证。[用法与实测](../benchmarks/p65-collision-m2-convex-hulls.md)
- **RR04 已验收** 旋转胶囊与球在原案例、留出、孤立相切和独立球位移上通过三端验证，两端分离时仍能返回途中首触。[结果与范围](../benchmarks/p65-collision-m2-rotating-round.md)
- **RR05 已验收** 凸体与闭合凹网格可区分空槽、内壁首触、交叠以及谁包含谁。左右内壁与两种三角化已通过三端回归。[结果与支持范围](../benchmarks/p65-collision-m2-closed-mesh.md)
- **RD03 已验收** 有限圆边板与布点、整张仿射三角面能返回正面、背面、侧边和角部的见证；整面覆盖时可返回面积，边接触可保留线段。[结果与支持范围](../benchmarks/p65-collision-m2-rounded-plates.md)
- **RD05 已验收** 变形布面进入满维凸刚体时，查询给出首触、进入实体的布面区域、真实边界面与有效法向。箱顶正例、留出及无理数首触均已通过三端回归。[结果与支持范围](../benchmarks/p65-collision-m2-hull-surfaces.md)
- **单爪承载与弯曲已验收** 唯一夹爪在 0.52 s 将原 48 g 软布抬离桌面 46.1 mm，零抓取摩擦时布落回桌面。两种软布初态的完整双臂轨迹另行验收。[实测和回放入口](../benchmarks/p65-single-gripper-bend.md)
- **夹爪已接入有限指垫反馈** 2° 软布的 0.36–0.37 s、竖直软布的 1.39–1.40 s 均在原输入和 10 mm 门槛下通过。两段共有 256 个真实关节及布料子步，仍需单爪及完整双臂验收。[同次力与运动](../benchmarks/p65-collision-m4-pads.md#铰接机器人与软布的实际反馈)
- **M4 有限指垫** 正背面、边角与滑动反力已联测，原软布失败步的 5 个材料观测均在指垫背后。公开接触力已能作用在同一个布面空间点上。[原失败位置与正面载荷对照](../benchmarks/p65-collision-m4-pads.md)
- **M4 三族动力学窗口已通过** 箱堆静载与卸载、球撞布后的反力和反弹、两张布在 2 秒内两次相遇与分离已用 HOST、tyx interpret 和实际 CUDA 验证，九条轨迹共 636 步。后续应用仍需单爪受载折叠及两种软布递布。[实测与边界](../benchmarks/p65-collision-m4-loads.md)
- **M3 已按声明范围验收** SAP、形状 BVH 与网格面级 BVH 保留全部 74 个构型的公开结果；另验收真实转弧与变形面的首触、持续、材料特征切换及拒绝后重试。下一阶段是 M4 的三族独立受力与运动回归。[筛选及成本](../benchmarks/p65-collision-m3-spatial.md) · [连续几何](../benchmarks/p65-collision-m3-curved-faces.md)
- **可复用基础** 球–平面、连杆胶囊、有限棱柱、线性特征扫掠和限定的壳几何可作起点。已有布内力、机器人树、双向点载荷、统一子步及回滚负责消费检测输出。
- **已完成的参考布案例** 带关节限位与六个连杆碰撞体的双臂交接。独立碰撞/限位/回滚检查已通过。3.62 s 的同次轨迹完成递布，B 独立横移 10.36 cm，末帧桌面支撑 0.47104 N；源码目录外的 wheel 逐位复现 B 接管步。[结果](../benchmarks/p65-robot-contact-results.json)及[本节点计划](../superpowers/plans/2026-11-02-robot-limits-contact.md)。
- **布料软硬实测（2026-11-02）** 同一 21 节点网格、48 g、重力和支承下，铰链柔量 20、200、2000 在 80 ms 的下垂分别为 0.72、8.22、18.49 mm。竖直软布已由 A 递给 B，在 1.39 s 的搬运步被 B 右指垫拒绝。2° 的试验在 A 左指垫产生另一条可重复的边缘反例。[参数与证据](../benchmarks/p65-soft-fabric-results.json)
- **应用回归** 上述 0.36 s 和 1.39 s 失败步保留在软布回归集中。完整软布递布需用原控制、两种初态、同次状态与力的 Three.js 回放验收。通用模块的放行先看各自的检测案例。
- **判停** 同一分支连续两轮没有新增能力、最小失败或可复现对照，先定位候选、特征、时间路径或响应中的具体差异，再决定改哪个模块。
- **工作纪律**：父分支独占共享接口；执行串行且每次 fresh guard；不重跑同源已绿组，不恢复周期任务，不修改依赖 pin，不碰已归档证据及 Temp junction。

[公共接口与运行示例](../../README.md#布料接触与搬运案例) · [Newton 对标与交付计划](../superpowers/plans/2026-10-31-p65-delivery-replan.md#2026-11-02-与-newton-的差距和执行顺序) · [双臂案例要求](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)

## 历史研究快照（保留追溯，不作为当前任务队列）

状态（2026-11-01）　用户已选定可标定接触层研究路线，已授权的独立算术验证已完成。07:00 限时续作已删除。N1 完成，当前 N2，普通质量联合物理尚未冻结。

- **交付物**　A 取布、B 接稳、A 松开、B 放台并撤离。同次仿真驱动可暂停、逐帧查看接触力的 Three.js 回放。[原始验收](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)
- **可复用基线**　有限夹爪、独立 B2/R7 材料、限域的自接触响应与几何已集成。严格二进端点的倾斜相触限制及旧 R7 范围反例按原版本保留。[框架范围](../superpowers/plans/2026-10-30-p65-material-self-gripper-integration.md) · [端点反例](p65-b2-sc2-first-touch.md)
- **新实证**　独立固定十向量在解释器及 CUDA 各 10/10，实际中间字和返回字一致，CUDA 启动一次。次正规、假零、上溢各有原始记录。没有联合回放或完整材料行的通过结论。[逐字收据](../benchmarks/p65-fp32-reduction-verification.json)
- **当前研究**　保留原几何半厚度，以单独 δ、面积刚度 k 和阻尼 d 描述有限接触层。零间隙有有限初储能。SC-L1 设计试片五个留出点和六个一维对照均通过。支撑载荷全窗至少为相对外载的 93.71%，冲击硬核余量大于 0.5760 mm。[候选定律与标定](p65-contact-layer-calibration.md) · [材料结果](../benchmarks/p65-contact-layer-screen-results.json)
- **步内实证**　SC-L1 的两种工况、三质量、三步长共 336 步满足峰值、冲量、时间矩、硬核间隙和功的数值判据。0.5 kg 冲击的卸载尾段有 8 个原时间步，发布的速度与间隙无法同时由非负压力产生，其中 7 步冲量恰为零。这排除了给当前端点单换插值的修复。[完整数据和反例](../benchmarks/p65-contact-layer-intrastep-results.json) · [逐步观测](../benchmarks/p65-contact-layer-intrastep-steps.csv)
- **共同发布**　固定的 8 个卸载反例各自从原入口重新推进，再由新进程接续一整步。16 段、822 个内部小段的非负力均同时给出读回速度和间隙，核对使用 FP64 主字加有理余量，运动包最大 217 字节。去掉余量时，同一压力在全部 16 段失去这种一致性。[逐例读回与负控](../benchmarks/p65-contact-layer-publication-results.json)
- **入层与峰值**　0.5 kg、1/64 s 的线性压力在峰值幅值上过关，但材料冲量与功超限。预登记的七阶对照通过前两个原时间步及跨进程读回，7.4375 ms 入层，11.15957–11.16007 ms 达到 17.13159482–17.13159489 N。续步为 504/512 个系数计数，算术开销单列。[两种更新的完整结果](../benchmarks/p65-contact-layer-entry-results.json)
- **持续支撑**　0.5 kg、1/64 s、0.125 N 外载的原支撑窗口已连续验证。八步七次跨进程续接，全窗最低支撑力约 0.124939464 N，末间隙 0.108072 mm。初始储能 0.001 J、外载有符号功 −13.509 µJ 和耗散独立核对。[逐步与能量结果](../benchmarks/p65-contact-layer-support-results.json)
- **连续冲击**　0.5 kg、1/64 s 的冲击从同一层外初态贯通 0.125 s，八步七次真实读回。7.4375 ms 入层，11.160 ms 左右达到 17.131595 N，25.1281 ms 转为开离。末压力约 2.89×10⁻¹⁵ N，仍有 0.596312 mm 层压缩。卸载曲线段界有微小上跳，最大 4.20×10⁻¹⁵ N。[全窗相位、能量和状态链](../benchmarks/p65-contact-layer-impact-window-results.json)
- **四节点实测**　原弯曲布片和 2 kg 平台完成一个 HOST 给定载荷步。布与平台的竖直净冲量误差为 4.34×10⁻¹⁹ N·s。载荷的时间矩在现有公开重放入口丢失，平台位移误差为 8.064 µm，限额为 0.010 µm。厚接触构造器拒绝弯曲材料，平台空间角冲量尚无实测通道。两面通用扫掠均未决，已对本例返回的仿射路径另证硬核安全。[诊断及下一次判别](p65-contact-layer-calibration.md#四节点布片和有限质量平台的一次载荷读回) · [原始量与独立核对](../benchmarks/p65-b2-joint-readback-results.json)
- **同冲量对照**　同一原始压力和倒序压力给平台相同末速度，末位移分别为 −14.0251624 和 −8.0637115 µm。移点使已交付 y 角冲量相差 0.000235614806 N·m·s，导轨独立约束反力仍不可读。三组候选的六个扫掠全部证实穿透，三组均拒绝续步。[三组审计](../benchmarks/p65-paired-load-history-results.json)
- **重力与六点联立**　三组原重力时间矩对照的质心均通过。倒序的质心间隙为 +0.869 µm，局部仍穿透 2.289 µm。已转入六点联立求力并送回真实参与者。最小末端硬间隙为 +11.229 µm，实际仿射弦段有独立安全证明。通用扫掠仍未决。至少一处定律冲量不一致，末间隙相对参考最多差 0.472205 µm。四个候选均未续步。[逐点结果](../benchmarks/p65-gravity-six-owner-results.json)
- **共同路径**　独立 HOST 研究程序从静止入口及真实磁盘非零速度分别接受一个 1/64 s 步，累计至 31.25 ms。在实际曲线上证明硬核安全，每点的冲量与时间矩、逐节点材料作用、原始能量和角动量均通过。首步末间隙与独立参考相差至多 4.09×10⁻¹² m。三个实际事务反例回滚，同一积分核心还通过了另一种双质量系统的解析对照。[通用框架](p65-common-motion-framework.md) · [两步证书](../benchmarks/p65-common-motion-results.json)
- **四节点 SC1 全窗**　从已验检查点完成余下六步，八步连续至 125 ms，七处衔接逐字节一致。全窗法向合力至少 0.12493478 N，最终六点硬间隙为 0.099112–0.112960 mm。非抵消正能量缺口 ≤6.82×10⁻¹⁴ J，成对角动量误差上界和 ≤1.91×10⁻⁹ N·m·s。[八步结果与五个负控](../benchmarks/p65-common-motion-window-results.json)
- **下一动作**　在 N2 处理原四节点 SC2 的层外入射，隔离最早入层的材料地址及整段硬核，绑定零载、出生、峰值与卸载。生产 B2/R7、原生与一般机器人仍开放。[框架与后续判据](p65-common-motion-framework.md)
- **执行边界**　独立算术及本次一维筛选均已结束，不重复同源已绿组。N3/N4、远程 CI 均未启动。
