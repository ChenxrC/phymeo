# C11 浮根 affine/mimic 设备树：开发与多案例验收

状态：**NEW_AUTHORING_FROZEN / PARENT_INTEGRATION_PENDING_C10_ACCEPTANCE**，2026-11-08，持续无截止。用户持续授权功能补齐。以[实际源码提案](../../development/p65-resident-floating-affine-next-proposal.md)和[固定案例清单](../../development/p65-resident-floating-affine-case-manifest.md)为输入；两文档已全文审阅并导入，数字是预声明，尚无C11执行结果。

本裁定细化提案的排期：C10验证期间允许隔离NEW纯编码及测试作者；**真正OLD接入、C11项目执行和正式验收仍须C10正式接受**。这是父在现有持续授权内安排依赖，不改变C10输入/源码快照，不要求等候才写独立NEW。C9已正式接受；父唯一共享接缝与执行owner，原robot/RD分别写产品/测试，RR额度错误不重启、不建替代。所有import、collection、数值、pytest、lint、formatter、compile/build/install/native和离线audit严格单队列。

## 1. 功能边界与复用

新增公开 FloatingAffineRobotTree，让现有 FloatingMimicRobotAsset 的完整浮动底座+常量同类型无环mimic图接入原 TyxWorldBatch。复用原loader、fixed工具折叠、图编译、完整物理浮根mechanics、当前设备FK/bounds、原PD/指数旋转、C10有限接触和原事务；不另造物理模型或runtime。

首域 1<=K<n<=16；B=n+1、Dfull=6+n、Dreduced=6+K、Qfull=7+n、Qreduced=7+K。根6维是真实COM平移/世界角速度，没有根电机。fixed/ordinary-floating原域保持；root-only仍走C9。所有真实body包括16参与力/惯量/材料，不按K或padding省略。

接触仍有限sphere/capsule/未圆角full-dimensional hull、mu=0、discrete_entry、有限static或同完整树非相邻双dynamic。HOST窄相+TYX响应；不宣称全GPU窄相、CCD、持续摩擦、跨独立participant、非线性/闭环图或训练梯度。不得按某个案例/关节名/坐标/时间选择施力算法。

## 2. 公开数据与精确维数

冻结名称及主要语义：

- FloatingAffineRobotTree(asset,state,*,kp,kd,effort_limit,delay_steps=0,control_max_iterations=512)。state为原独立FloatingRobotState，q长7+K、v长6+K；asset必须是原FloatingMimicRobotAsset。K gains/caps、原独立motor额定上限、联合位置/速度域和C9有限FP32域均预检。只有drive延迟0/1步。
- .inputs(drive,external=None) 返回不可变 FloatingAffineTreeInputs(graph_token,drive,external)。drive为K×3(target/rate/feedforward)；external为原 FloatingRobotInputs，full joint effort n必须全0，B×3 COM force/torque是即时输入。沿C9，不借C5的full effort能力扩大本域。根COM torque不是PD前馈。
- FloatingAffineTreeCheckpoint 沿C5字段含world/topology/tick/position/velocity/applied_effort/history/control/generalized_applied/full_position/full_velocity/full_applied_effort/full_generalized_applied，再保C9 local_origin_m。position/velocity是Qreduced/Dreduced；full_position/full_velocity是Qfull/Dfull；applied_effort为K，full_applied_effort为n且仅真实driver行注入电机，generalized_applied分别Dreduced/Dfull。控制包保完整graph与即时外载。full/reduced/root local不一致直接拒绝，不能静默投回。
- FloatingAffineTreeKinematics 沿 FloatingTreeKinematics，真实position/velocity/FK/COM/twist及joint/body/velocity/configuration数量均为full；另 independent_position/independent_velocity（含同一root）、joint_names（独立）/full_joint_names。完整数组owned不可写。
- FloatingAffineTreeDiagnostic 沿C5 full/reduced语义但维数含根：base entry_position/entry_velocity为Qreduced/Dreduced，原完整J/M/b/external按full维数，full_entry_position/full_entry_velocity/full_acceleration另存完整值；reduced_mass/bias/external/J；K实际drive、Dreduced acceleration；constraint_effort_full为Dfull，row_multipliers/row_powers保持原关系顺序、constraint_power_w及root_constraint_residual（六维原余额）。控制/方程门保守聚合，不伪造一个tau。
- ResidentFloatingAffineTreeBinding 与普通floating binding具有相同key/world/group_index/local_points/topology，以及原graph_token；from_asset(asset,source_key,*,key,world)委托普通floating来源及fixed工具变换一次，再绑定原mimic图。真实parents决定碰撞邻接，mimic关系不变成几何邻接。

私有 FloatingAffineTreeWorldSpec / compile_floating_affine_tree_world 冻结为入口名：physical为真实旧FloatingTreeWorldSpec；n/K分别存n/k，b=n+1、d=6+n、reduced_d=6+K、q_count=7+n、reduced_q_count=7+K，domain='floating_affine_tree'，kind=11+delay。存储类 FloatingAffineTreeDeviceStorage。它不是缩成K个body的旧spec。

原FP64 model/原关节limits/关系行、P/c全绑定graph identity，不只比较舍入后的P。命令/检查点/geometry/receipt的tick及active bank明确。大anchor保持local_origin_m，不从world Q反减恢复。

## 3. 唯一共享求解与父OLD接口

父与原robot已完成当前源码文本依赖审阅；最小可复用路线为**同一个 floating_solve**，不复制360行积分器，不先无约束step再覆写followers：

1. 物理 FloatingTreeDeviceStorage 的meta/FK/J/M/b/外载始终使用完整n/B/D，入口mechanics只一次。C10完整COM wrench先经full J转置加到full vector[...,1]一次。
2. NEW以S=diag(I6,P)投影Mr=SᵀMS、br=Sᵀb、er=Sᵀefull、Jr=Jfull S；完整root–joint交叉块保留。reduced solve_meta第二列是K，slot/delay/budget沿原约定。
3. 父把旧 floating_solve 的 properties 参数拆成 motor_gains[T,Kpad,3] 和 physical_root_com[T,3]，其余参数顺序保持：count,h,meta,motor_gains,physical_root_com,commands,root,state,fk,applied,tick,next_root,candidate,next_applied,mass,factor,mobility,vector,info,diag。ordinary传K=n，NEW传K；root/fk仍借真实full根，不建立假reduced body。原求解顺序、浮点helper、控制/方程检查、根指数漂移及失败保留不变。physical_root_com是原根body-local COM，非全树合成COM或principal frame。
4. solve只产生一次reduced candidate/root drift。NEW展开joint q=P*u+c、v=P*udot、full acceleration=S*ar，full motor=E*tau（仅独立driver行），不是P*tau。full_applied=efull+E*tau，reduced_applied=er+(0_6,tau)，约束力另列；full follower不能独立先漂移。
5. r=Mfull*afull+bfull-efull-E*tau，按原关系行恢复lambda，核r=[0_6,Cᵀlambda]、Sᵀr及逐行/总功。root六行必须纳入full残差。反例：P=(1,1)、full g=(0_6,-2,+2)有reduced g=0但lambda非0；不能以K载荷0省略原行反力。
6. full/reduced原位置/速度限位、offset、图关系/加速度全验；例如qfull=2*u+1、u=2048不能只看K有限域。新投影错误用独立标志保留，不能被solve初始化info.code清掉。原full finalize按n保留root13、全n states、Dfull applied、B FK/twist/energy/tick；NEW reduced retain采用同一accepted决定。最后仍只有父batch一次swap。

NEW storage组合physical并提供原C5风格 stage_mechanics / stage_solve_from_mechanics / stage_candidate_geometry / stage_finalize_and_retain / stage_advance、check/finish/restore/snapshot/kinematics/diagnostics/summary/borrow_geometry/close；具体私有小字段可在原NEW内协调，任何父OLD签名变更先以精确文本报父。full输入/控制上传与K drive/延迟队列分别owned。

父 contact group 用两个正交事实 floating、affine 及明确kind。不可沿C10当前“非ordinary且非floating即fixed-affine”推测。floating-affine同次packet保存独立checkpoint和真实full geometry；temporary wrench B×6、full projected Dfull，再Sᵀ一次。receipt input_external/contact_generalized均full Dfull，selected_drive K×3、actual_drive K；diagnostics coordinate_kind='floating_affine'，同时标明full与reduced维数、完整同次浮根affine诊断。没有第二次force application或publication。

## 4. 文件所有权与作者顺序

原robot只NEW五路径：
1. src/phymeo/backends/_resident_floating_affine_tree.py
2. src/phymeo/backends/_resident_floating_affine_storage.py
3. src/phymeo/backends/_tyx_resident_floating_affine.py
4. src/phymeo/backends/_resident_floating_affine_binding.py
5. docs/benchmarks/p65-resident-floating-affine-implementation.md

原RD只NEW五路径：
1. tests/scale/test_resident_floating_affine_step.py
2. tests/scale/test_resident_floating_affine_contacts.py
3. tests/scale/test_resident_floating_affine_transactions.py
4. tests/reference_newton/robot/resident_floating_affine_reference.py
5. docs/benchmarks/p65-resident-floating-affine-tests.md

先由RD提交有效A01测试及必要原输入/独立前置的首checkpoint，再通知robot开始NEW产品。后续两树并行纯作者。所有NEW先确认父及子树不存在；不merge父、不改旧source/tests/exports/fixtures/reports，不执行、不加助手。依赖从父只读。每个逻辑块小commit，最终精确范围/clean/公开API与限制/测试身份/UNEXECUTED交接。原proposal保留历史状态，以本计划为当前作者授权；不重复派提案。

父OLD允许有限共享接缝：_world_step_layout.py、tyx_world_batch.py、_resident_floating_storage.py、_tyx_resident_floating_kernels.py、_resident_collision_plan.py、_resident_contact_plan.py、_resident_floating_contact.py、_resident_contact_bridge.py（仅实际维数/packet缺口）、src/phymeo/__init__.py及backends/__init__.py。它们在C10正式接受前不改。旧C9普通solve签名提取须有具体受影响ordinary节点，不能整套C5–C10重复。原mimic graph/loader/HOST模型、材料law、形状算法、数值helper/pin不改。

native、wheel consumer与stdlib saved auditor沿案例清单后置，使用原作者另发明确NEW范围；本授权不含它们。

## 5. 固定案例与工程验收

采用案例清单§1–6全部原A01–A05输入、显式参数ID、h/steps/保存名；核心预声明每backend49节点/43gzip，W32是后置1节点/1gzip、4步。测试必须显式ids，先独立HOST literal验证再在函数内查新API，不因module missing-import做假RED。独立full time-jet/KKT和原declared parts构造M/b/J；不读新产品输出造expected。

same-entry atol5e-6/rtol5e-5，窗口2e-4，原PD5e-5/方程2e-5。新增full约束余额和原行关系沿C5归一scale与2e-5，原功门abs(power)<=2e-5*power_scale；root余额进入full_error，不能分配额外放宽预算。材料1e-10仍是stored表示重建门，不硬当FP32真值误差。所有入步速度/外载保真，不以设置v0或补坐标修复轨迹。

延迟首history target=FP32独立initialq，rate/ff0；A01初u0才恰是零target。实际motor仍受当步即时COM外载的Schur耦合，不能把零target当零motor。同P的fan/chain保原不同C/c/identity和逐行lambda。

19短正例先执行；A04-p1与contact W1/W4才做40+19→21。core W1/W4仅4步；大n16/K1或15仅1步。A02-root-lifecycle保持原独立声明与40步，要求实际正载→零载→再正载，失败保原请求，不调g/m/k/h/窗长。contact输入/完整body反力/图历史/全部状态与原功同次保存，不按测试名分支物理。

事务只新增full/reduced/graph相关状态边界，复用原真实upload/readback/receipt后故障机制；两bank容量/重试、root local/full/reduced restore、未绑定world正常前进、stale/close/资源完整。原状态码沿现行public字符串，不能给旧数字新含义。负控不伪造轨迹，缺保存瞬间明确由同源pytest断言支撑。

## 6. 执行与止损

- [ ] NEW纯作者：A01 tests-first → 产品/案例并行 → 源码文本交叉审阅/精确NEW导入。
- [ ] 前置：C10专项、原生/保存/源码外wheel正式限域接受，释放队列。
- [ ] 父首A01有效RED → OLD窄接缝 → 最小A01 interpret/CUDA → A03/A04短档 → A02/A05真实接触。
- [ ] 其余短档及新增事务，原工程门通过后再代表窗；只对修复影响范围复验，同源绿节点不重复。
- [ ] Newton CPU/CUDA原生3短例/端；记录其原法则/限制，不matched外力，不将BODY_PARENT_F当mimic lambda。
- [ ] 新wheel源码外公共API、完整安装/loaded origins、contact W4 40+21及A04短档、两bank拒步恢复；随后同次保存审计，无重算几何/动力学。
- [ ] 更新功能全景与正式范围，再择下一高优先级缺口。

首短档实际成本用于排程；预计单61步窗>5min或W1/W4代表>15min，可先交短档阶段结果、把原完整节点明确NOT_RUN排后，不改steps冒通过。W32仅核心通过且4步预计<=3min时一次后置；容量/内存不合适可明确留未跑。没有性能收益也可接受新功能，禁止为凑倍率重复测量。

连续两次修改无新增入口/缩小复现/排除明确假设，或60min只在扩写条件而没有最小能力进展，暂停该节点扩域并回到可审薄接口。full/reduced漂移、漏原图反力、漏root余额、错误材料或半发布是功能问题；数值精度按原工程域明确报告，不追逐无限精确接触。全部P6.5与Newton功能对齐仍持续，C11局部通过不等于全引擎完成。

## 7. 作者接缝补充（2026-11-08）

已确认的私有交接结构如下；它细化第2–3节，不改变物理、输入或验收范围：

- checkpoint.control=(graph_token,drive[K,3],effort[n]全0,force[B,3],torque[B,3])，每块独立owned并绑定原图。
- spec.packet(packet)返回(owned_drive,owned_effort,owned_loads[B,6])。
- spec.restore_row(checkpoint,world)返回(root[13],reduced[K,7],full[n,7],reduced_applied[6+K],full_applied[6+n],packet,tick)。
- physical spec可以共用C11 topology，但调用ordinary验证helpers时必须构造明确full临时数据/control，不能把K packet交给floating_packet(n)。公开checkpoint保持floating-affine身份。
- 诊断逐行功的精确公开字段名为row_powers_w，总功为constraint_power_w；第2节row_powers仅语义简写，不创建别名。root_constraint_residual保完整6维实际余额。
- 新floating-affine storage沿C5状态名：7=AFFINE_CONSTRAINT_RESIDUAL、8=AFFINE_FULL_EQUATION_RESIDUAL、9=SITE_CAPACITY_EXCEEDED；0..6沿C9。ordinary floating原7=SITE_CAPACITY_EXCEEDED不变。父contact reject按明确affine事实选9，禁止跨storage误解码。

RD已先交A01有效测试checkpoint 91099315bf6ee1c015327eaa7c7c68d592b73e03，首节点test_a01_public_affine_floating_root_reaction；原robot随后开始五NEW产品作者。该记录仅证明tests-first作者顺序，无C11执行或通过结论。

直接binding签名固定为ResidentFloatingAffineTreeBinding(key,world,group_index,local_points_m,*,topology,graph_token)，graph_token为必填tuple；from_asset自动证明原图/形状。编译返回既有FloatingBindingCompiled，复用原full bounds核。receipt diagnostics固定coordinate_kind='floating_affine'、floating_affine_diagnostic；full_dof_count=6+n、reduced_dof_count=6+K、independent_dof_count=K，joint_count=n/body_count=n+1/velocity_count=6+n/position_count=7+n仍指真实full。projection_matrix保存完整S=diag(I6,P)，offset_vector为n关节c，constraint_rows为原(n-K)×n的C（根六列隐含零），graph_token保原图。storage提供_read_diagnostics(selected)和纯数组diagnostics_from_arrays(worlds,full_arrays,reduced_arrays)，父使用同次预swap数据，不为receipt再求mechanics。
## 8. C11 原生短参照作者授权（2026-11-08）

父已只读核本机 pinned Newton 1.7.0 的 `_src/sim/builder.py::set_joint_mimic`、`_src/sim/joint_mimic.py`、`_src/solvers/featherstone/solver_featherstone.py` 与 `kernels.py::{reduce_mimic_inertia,reduce_mimic_forces,expand_mimic_accelerations}`。源码支持同一 articulation 内 FREE 根加 P/R mimic，FREE 自身不能为 mimic；不支持 follower-chain。当前三例均为直接 leader/follower，不需要 flatten。源码可用性不是执行通过，未知行为必须由实际短参照保留。

原 robot 完成 RD final8c5107b 的只读交叉审阅、无阻断后，允许在原隔离树 final6eb43b8 上继续 **NATIVE_AUTHORING_ONLY**，恰两 NEW：
- `tests/reference_newton/robot/resident_floating_affine_native.py`
- `docs/benchmarks/p65-resident-floating-affine-native.md`

不改已交五产品、旧参考/C5/C9/C10、依赖/pin或父共享接缝；不merge父，不import/数值/collection/pytest/static/format/native/build/install，不持执行槽。可从RD final8c5107b只读原独立参考和数据；未来父同目录集成后才用显式路径载入该 NumPy/stdlib reference，绝不 import Phymeo/TYX 或测试 fixture。若需要额外路径先以具体缺口报父，不扩域。

CLI 固定 `--device cpu|cuda:0 --output <fresh absolute directory>`；三例顺序 A01-motor 1步、A03-fan 4步、A05-native-contact 4步，h全部1/1024，分别一个完整gzip和统一summary。输入严格采用[案例清单](../../development/p65-resident-floating-affine-case-manifest.md)及 RD reference，原质量/惯量/COM/轴/offset/图/非零入速/即时外载不变；禁止为了原生通过改变场景。A01真实独立joint_f=1 N，A03同完整COM力和独立3/8 N；A05只原生collision及原生penalty contact，不喂reference/Phy力，不matched-law回退。模型使用显式body/link/关节与 `set_joint_mimic`，所有真实关节含FREE归于一个articulation；保原尺寸/关节/shape ID来源。

独立保存并核模型全部body质量/COM/惯量、FREE与P真实类型、原mimic系数/leader、articulation归属、完整q/qd起止/shape过滤/材料；armature、被动damping、target gains、limit springs为原0，root无伪motor。根公共COM世界速度与Featherstone内部anchor速度分别存原数组和明确基变换，不能把内部qdd直接叫公共加速度。三例初始角运动0且共线对称力维持角运动0，但仍核实际全部角分量，不能丢根六维响应。

每实际step保存完整entry/exit状态、原始control/body_f/contact rows、full M/J及内部力/加速度scratch、原生mimic reduction后的字段、full/independent图关系、完整动量/能量；A05以同一次真实contact原行独立解码材料/接触点/两侧wrench及full generalized load，只作观察，不回写推进。对照独立未约化KKT的full acceleration与原行lambda，lambda来自完整余额重建，明确不是Newton原生反力传感器或BODY_PARENT_F。记录根六余额、原约束残差与功、full反力，并保存模型和driver/reference/native来源bytes与SHA。

沿原C5/C9 native对照：same-entry M/J/FK atol5e-6/rtol5e-5，短窗q/v atol2e-4；实际Δv/h推导acceleration/row reaction采用原C5 atol3e-4、rtol5e-5并标明差分FP32误差。该原生观察门不改变C11产品2e-5门。每个比较保存actual/expected/atol/rtol与实际结果；任一失败保完整前缀/最后返回字/原异常，RAW1，不补状态、不跳case报全过。只有实际device精确匹配允许通过，不静默CPUfallback。完整原输出沿64MiB压缩/256MiB解码上限；不删除旧证据或裁剪字段。

本片不比较不等价的非零PD/effort/velocity-limit、一般转动积分或性能；实际native短窗结束后才有功能结论。父C10正式接受→C11首RED/最小TYX通过后才另发串行执行。交付小commit、clean、精确两NEW、公开CLI/保存schema/限制、UNEXECUTED及AUTHORING_RELEASED。

core W1/W4的旧root-only身份同步明确为原C9 `reference.case('P05', n=0)`，delay1，保其真实COM/主轴/非零初速/外载；另FreeSphere仍2 kg、r1/8、z2、g−8。此为先前父消息确认的身份补足，不是新调参或额外案例。
