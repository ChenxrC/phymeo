# C11 常驻浮根 affine：候选测试与交付清单

状态：**PROPOSAL_NOT_FROZEN / AUTHORING_ONLY / UNEXECUTED**，2026-11-08。本页只为父下一次冻结准备可执行清单，不授权产品、测试或项目执行。原[后继提案](p65-resident-floating-affine-next-proposal.md)保持；读取基线为作者 `f0469a5`、父 C9/C10 当前公开源码及已交付固定案例。C9 已由父正式接受；C10 尚在验证，**C10 正式接受和父另授文件范围之前不得实施本清单**。这里的数量是手工预声明，不是 collection、运行或通过数量。

## 1. 共用合同及独立输入

沿拟公开 `ph.FloatingAffineRobotTree(asset,state,*,kp,kd,effort_limit,delay_steps=0,control_max_iterations=512)` 与 `.inputs(drive,external=None)`，最终拼写由父冻结。根无电机：完整关节 n、独立电机 K、body B=n+1、Dfull=6+n、Dreduced=6+K、Qfull=7+n、Qreduced=7+K 分别保存。S=diag(I6,P)；full body-COM wrench 经完整 Jᵀ、Sᵀ各一次。原 full/J/body 材料身份、root local_origin_m、完整反力及 held/history 不因降维省略。

复用 `load_floating_mimic_urdf`、两原 URDF、`compile_mimic_coordinates`；独立 reference 复用 C9 的 time-jet 低级数学，按 authored 部件装配 full M/b/J 和未约化 KKT，不读取产品 M/J/P 生成唯一 expected。实际产品输出只用于同入口对照和原方程余额。以下五锚不得按后续结果改质量、力、步长或工程门。

| 锚 | 原输入、独立首步预期 | 来源与区别 |
|---|---|---|
| A01 | jaws：root/两子质量2/1/2 kg、+X轴、q₂=−q₁+.004，零速/重力、motor +1 N；Mr 的X块为 `[[5,-1],[-1,3]]`，根/独立/从动 a=`(1/14,5/14,-5/14)`，原行 λ=−4/7 N，根约束反力零。 | `tests/test_floating_mimic.py:76–115`；负比例、offset、内部驱动与真实自由根耦合。h=1/1024。 |
| A02 | 同 jaws 的 right_pad 有限+X面；对手沿原首载球位置，改为明确静态球；k2048/d0/μ0，穿深1/1024，总N=2。full root-X/right-joint载荷各−2，reduced=`(-2,+2)`；根/独立a=`(-2/7,4/7)`，λ=2/7，静态反力+2 N。 | `tests/test_floating_mimic_scene.py:214–242`及casebook§T3；sphere中心x=`1/4+.004+1/32+1/16−1/1024`，r1/16。仅登记原right_pad与该球，不把原会动的普通机器人冒作此静态对手。h=1/1024。 |
| A03 | jaws加原0.5 kg gauge；fan/chain：P=`(1,-1,1/2)`，c=`(0,.004,.002)`/`(0,.004,0)`。完整X块M₀₀=5.5、M₀j=`(1,2,.5)`、joint块diag(1,2,.5)。各body外力=`(1/8,1/4,-3/8,1/2)` N，故full广义外力=`(1/2,1/4,-3/8,1/2)`，motor仅driver +3/8 N，独立入速.02。 | 原test `:180–228`；C的两原行分别`(0,1,1,0)`与fan`(0,-.5,0,1)`/chain`(0,0,.5,1)`。直接literal full KKT核两λ和逐行功；不能用相同P合并图身份。h=1/1024。 |
| A04 | 原 elbow 四 authored 部件及非平行Z/Y轴、ratio1.5/offset−1/8、偏心fixed工具；保原三根姿态/非零root与joint速度、g/COM力矩、motor13/32 N·m。 | [原casebook](p65-floating-mimic-casebook.md)44–69行全量数值原样引用；三姿态ID为p0/p1/p2。使用独立二阶time jets替代旧FD路径，不换原物理输入。h=1/4096。 |
| A05 | 原C10 F03 root2/left2/right1 kg、两Px、球r1/8、offset±7/64、k64/d0/μ0、q/v=0；仅新增qR=−qL。首两侧±2 N，Mr=`[[5,1],[1,3]]`、reduced载荷`(0,-4)`，根/独立/从动a=`(2/7,-10/7,10/7)`，λ=−2/7，总动量增量零。 | `resident_floating_contact_reference.py:100–106`及原提案§5；同一完整树双动态，不能沿用无mimic的旧C10加速度。h=1/1024。 |

same-entry沿C9/C10 `atol=5e-6, rtol=5e-5`（各字段原单位），M/方程绝对门2e-5、原控制残差5e-5、40步状态atol2e-4；材料记录构造1e-10仅是原stored pose/点重建一致性。HOST literal仍用原严格门，不将HOST的1e-12硬加到FP32动量。原约束行full余额、root六维余额、Sᵀr、每行/总功须分别记录；不把有限步功或动量误差改名为精确能量守恒。新逐行设备门在父冻结时沿已有C5/C7单位合同，不能在跑后按结果自设门。

## 2. 拟NEW路径和互不交叉owner

- 产品作者：`src/phymeo/backends/_resident_floating_affine_tree.py`、`_resident_floating_affine_storage.py`、`_tyx_resident_floating_affine.py`、`_resident_floating_affine_binding.py`。只做图/维数适配、full/reduced组合、投影/展开/原行账和来源绑定；不复制完整integrator或contact plan。
- 独立案例作者：下表简称 **S**=`tests/scale/test_resident_floating_affine_step.py`，**C**=`tests/scale/test_resident_floating_affine_contacts.py`，**T**=`tests/scale/test_resident_floating_affine_transactions.py`；另NEW `tests/reference_newton/robot/resident_floating_affine_reference.py`。资产生成仅使用明确原始声明与既有XML图规则，不变更原URDF。
- 后置消费者作者：NEW `tests/reference_newton/robot/resident_floating_affine_native.py`、`tests/scale/resident_floating_affine_wheel_consumer.py`、`tests/scale/resident_floating_affine_saved_audit.py`。与产品/专项不同文件，需各自额外grant，不因本页列出而提前编写。
- 父OLD：`_world_step_layout.py`、`tyx_world_batch.py`、`_resident_floating_storage.py`、`_tyx_resident_floating_kernels.py`的唯一共享代数solve/root drift提取、`_resident_collision_plan.py`、C10最终`_resident_contact_plan.py`、顶层`__init__.py`。模型/loader/HOST图机制直接复用；NEW作者不改这些接缝。父同时维护总计划、执行顺序与正式验收报告。

## 3. 短档先行：19个设备实例、19个完整gzip / backend

每行明确参数ID，`tyx_mode`沿仓库fixture分别跑interpret/CUDA，不额外乘进pytest参数ID。`h`均秒；`steps`是实际推进次数而非solver迭代。每个实例恰一个表列gzip，包含失败前缀；首RED之后同源已绿节点只从后续组明确deselect。

| 文件 / testfunction | 固定variant IDs | h；steps / 实例 | 保存名 | 实例数 |
|---|---|---|---|---:|
| S `test_a01_public_affine_floating_root_reaction` | `base`（不参数化） | 1/1024；1 | `a01-public.json.gz` | 1 |
| S `test_a01_cycle_translation_preserves_full_response` | `cycle` | 1/1024；4 | `a01-cycle.json.gz` | 1 |
| C `test_a02_follower_surface_load_projects_once` | `forward`,`reverse` | 1/1024；2 | `a02-{variant}.json.gz` | 2 |
| S `test_a03_original_graph_rows_and_offset` | `fan`,`chain` | 1/1024；4 | `a03-{variant}.json.gz` | 2 |
| S `test_a04_raw_parts_rotating_tool_response` | `p0`,`p1`,`p2`,`p1-cycle` | 1/4096；1 | `a04-{variant}.json.gz` | 4 |
| C `test_a05_same_tree_bilateral_response` | `forward`,`reverse`,`cycle` | 1/1024；2 | `a05-{variant}.json.gz` | 3 |
| S `test_a01_pd_delay_and_current_com_load` | `d0-c1p5`,`d1-c1p5`,`d0-c0p25`,`d1-c0p25` | 1/128；4 | `pd-{variant}.json.gz` | 4 |
| S `test_full_body_and_reduced_motor_counts` | `n16-k1`,`n16-k15` | 1/1024；1 | `counts-{variant}.json.gz` | 2 |

首节点顺序：在函数内先用现有公开HOST资产确认A01 literal、full6+n/独立6+K及root反力，再查询拟公开设备spec并调用batch。有效RED必须停在新能力缺口，不能在module import/错误dtype/旧夹具前提失败。不得为新增计数再跑旧HOST整套。

`cycle`统一使用原轴循环R(x,y,z)=(z,x,y)及平移(1/2,−1/4,1/8)，同时变换根姿态、COM速度、重力、力矩、静态几何，joint坐标/原关系不变。`reverse`只反向登记同一输入，不改body/material IDs；原bindings顺序单独保留。

PD四项严格复用原jaws：kp100、kd2、target .01、显式ff0、根COM +X外力1 N，其余外载零；spec cap为1.5/.25，delay0/1。独立Meff=14/5、beff=−1/5；首个实际selected target为t时，η=h(100h+2)/Meff，tau=clip((100t+η*beff)/(1+η),±cap)。delay1首队列沿原C9初始化：target首列是FP32独立初始q，不是普遍全零；这里A01的u=0才使首triplet恰为零。kp/kd仍会响应当前外载，**不能把零target误当零actual motor**。其余案例必须从各自实际initial history取selected，不能把full follower的offset当独立target；保留父511发现的旧N9非零initialq证据，不改产品初始化。后续用独立full KKT+同一PD方程，不搬HOST的None自动前馈。

数量边界使用明确的正质量直线树：root沿A01 m2/I，16个root-child +X轴、offset_i=((i+1)/32,0,0)，各m1/2、COM0、I=(1/64,3/128,1/32)，根原点零/单位姿态、独立joint q及全部v/g/PD为零，full joint q按关系展开；所有轴位置界±1、速度1，cap8。k1取q_i=(−1)^i*u；k15取q_15=−q_0+1/256，其余独立，正确展开初态；body16 COM +X力1/8，其余零。完整X质量块M00=10、M0j=1/2、joint块diag(1/2)独立给出，不能用产品M造预期。核真实body16 FK/current bound及B17/Dfull22，不另铺大N长窗。

## 4. 代表窗口与W：核心3节点、11个gzip / backend

| 文件 / testfunction | 固定variant；真实输入 | h；推进/重放 | 固定保存相对名 |
|---|---|---|---|
| S `test_a04_rotating_full_reduced_checkpoint_replay` | `p1`，原A04第二姿态，无新增控制 | 1/4096；40，再恢复tick19继续21 | `a04-p1-window.json.gz` |
| T `test_w1_w4_affine_core_same_input` | `core`；keys(503,91,1207,6)=A01、A03-chain、旧C9 root-only、旧FreeSphere | 1/1024；各W1×4及W4仅4步，不额外restore | `core/w1-{503,91,1207,6}/trace.json.gz`及`core/w4/trace.json.gz`（5份） |
| T `test_w1_w4_affine_contact_replay` | `contact`；keys(503,91,1207,6)=下述A02-root-lifecycle、A05、C7原ordinary-P、C7原fixed-affine A03 | 1/1024；各W1×4及W4 40+19→21 | `contact/w1-{503,91,1207,6}/trace.json.gz`及`contact/w4/trace.json.gz`（5份） |

core档的A01/A03使用delay1但保原drive/COM外载；旧root-only/FreeSphere直接引用原C9混合声明，不重复其独立证明。contact档新/旧四world共用原h，geometry/law及真实full/K响应分别保存；旧ordinary/fixed-affine不成为新增能力统计。W1与W4同world完整输入/状态/FK/诊断/receipt逐字段核等，不删world/owner。批分配/epoch/调度计数另列，不能要求它们和单world相同。

A02-root-lifecycle是显式派生输入，不改A02首载2 N：保A01完整jaws质量/惯量/affine图，u/v=0，零motor/PD及COM外力；根原点z=1/8−1/4096、根COM vz=1/8、g=(0,0,−8)，仅登记根中心r1/8球与C10 F09原有限静态箱，k64/d0/μ0，h1/1024。首N=1/64 N；其余动力学用独立完整树参考，不能套旧m2自由球加速度。40步中从真实记录要求正载→零载→再正载，状态不reset、不按时间切force；若该新系统未出现事件则如实保留并停止，不拉长窗口/调参数硬过。该派生窗同时检查结束/重入没有残余接触载荷污染held或约束反力。

只有A04旋转和contact混合承担61步恢复。A01、fan/chain和参数协变只跑短档；不为所有排列复制40+21。A05完整双侧窗和lifecycle已在contact的W1/W4原记录中，不另重跑单独长窗。

后置 `T::test_w32_bounded_affine_contact_batch[contact]`：只4真实步、1份`contact/w32/trace.json.gz`，world keys为前四加`10001+97*i`（i=0..27），按上述四输入循环且独立ID；保存32world全部原始字段。不重跑对应W1/W4：同源同输入时只比较已存前4帧；若source/默认改变则不得拼成同源结论，重新排测必须父明确授权。W32是容量/隔离留出，不是速度验收，不放n16/大K或整矩阵。

## 5. 只测新增状态边界：27节点、13个gzip / backend

C9/C10已验证的generic owner、全局swap、故障机制不复制整套；这里只检查C11新增full/reduced/graph/drive侧状态是否进入原事务。复用既有实际call-through故障注入位置，不建新模拟runtime。下列全部在T文件，正常h1/1024。

| testfunction | 固定variant IDs | 新增检查与调用上界 | 保存 | 数量 |
|---|---|---|---|---:|
| `test_x01_affine_spec_and_source_preflight` | `fixed_asset`,`full_width`,`joint_position`,`joint_velocity`,`nonfinite_gain`,`negative_cap`,`model_token`,`graph_token` | 从合法原资产触新spec/compile/restore边界，操作前原状态不变且不增device分配；0步 | 无gzip，保原pytest/source/拒绝理由；不伪造轨迹 | 8 |
| `test_x02_local_origin_full_reduced_restore` | `large_anchor`,`full_q_mismatch`,`full_v_mismatch` | 真1步→保存/合法restore→同请求1步；错误restore先拒绝且全态不变，最多2接受步 | `restore-{variant}.json.gz` | 3 |
| `test_x03_contact_capacity_preserves_both_banks` | `bank0`,`bank1` | contact W4；分别限503/91容量0，其余world正常，原记录保留后足额重试；warm0/1+拒绝+重试≤3 batch调用 | `capacity-{variant}.json.gz` | 2 |
| `test_x04_late_failure_preserves_graph_state_and_history` | `upload-b0`,`upload-b1`,`candidate-b0`,`candidate-b1`,`receipt-b0`,`receipt-b1` | W2：A01(delay1,+1 N)未绑定与A05(delay1)绑定；真实原操作后抛，全部full/reduced/history/FK/receipt/epoch不发布，合法retry等fresh同输入；≤3 attempted调用 | `late-{variant}.json.gz` | 6 |
| `test_x05_stale_restore_and_closed_rebuild` | `stale`,`close_rebuild` | 实际候选后restore使token失效；或原accepted checkpoint在新owner重建并重放；≤3 attempted调用，资源归零 | `owner-{variant}.json.gz` | 2 |
| `test_x06_body_motor_and_domain_preflight` | `body17-n16`,`body16-n15`,`drive_full_n`,`external_joint_effort`,`wrong_reduced_width`,`cross_independent_tree` | 真B/D/K与原full parent/filter；不以K当body上限；0步 | 无gzip，同上 | 6 |

X01不重新运行旧loader的cycle/unknown图拒绝：fixed_asset取已有固定根mimic，full_width取合法asset.full_state返回，joint位置/速度取A01联合上界各加1/1024，gain为NaN、cap为−1/4；model_token用同图但root质量3 kg的合法来源，graph_token用A03 fan/chain同P不同原图来源。已有batch先记录状态/分配基线，再测试外来checkpoint或binding拒绝；不能要求建batch本身零分配。large_anchor沿C9原反例anchor=2**60、local_origin=1/1024；只作step/checkpoint，不登记丢失几何offset的碰撞。local/world不一致仍拒绝，不偷偷正交化/投回full图。capacity状态码必须由父核当前枚举冻结，不复用一个已有不同含义的数字。晚异常采用旧事务本身；只有新sidecar保留遗漏才是C11新增finding。

## 6. membership、完整保存和后置消费者

核心合计每backend **49实例=19短档+3代表节点+27事务；43 gzip=19+11+13**，其中14个preflight节点无轨迹、35个节点会保存（两个W节点各5份）。两backend分开标识；不增“独立HOST通过”计数，首节点的HOST前提属于该节点。W32后置额外1实例/1gzip；没有运行不能把计划49或50写成通过。父首次collection若与手工声明不同，先核源码/ID，不改数据凑数。

参数须显式`ids=`采用本表字符串；`nodeid=文件::testfunction[variant]`，未参数化首节点无方括号。每份路径是`<fresh-basetemp>/<pytest node.name清洗后前30字符><同prefix顺序ordinal>/<表列相对名>`；ordinal按原selected顺序并包含无保存的tmp_path节点。原runner cwd、参数、head、source前后SHA、fixtures/reference均绑定。失败有完整输入/已接受前缀/最后retained/原异常；保存器异常单列，不当物理回滚。

完整输入保存原FP64模型/图/每行关系、实际FP32参数、controls/全部COM外载/形状来源/μ0 law；每步保存entry/candidate/accepted full与reduced q/v、local_origin、applied/held/history、B×6 law/uploaded wrench、full J/M/b与reduced矩阵、实际drive、原λ/每行功/root残差、geometry及材料/application点与work、tick/status/bank/epoch。accepted与candidate时态不同不能互相替代。非接触记录不补空receipt冒接触。40+21按全部物理字段重放，epoch/资源计数单独核单调。

沿已冻结的64 MiB压缩/256 MiB解压单payload、2 GiB总解码上限，直接无损gzip；不删既有证据、不按结果挑帧、不用hash替代完整输出。只读saved auditor后置、stdlib-only，先按本页/最终源码枚举membership再读输出，不重新调用reference/几何/求解器。附加首步几何证据可保留，但不能改变case identity或原输入。

Native后置建议每device仅A01-motor 1步、A03-fan 4步、A05真实native-contact 4步，各一个完整gzip（3/device）；CPU/CUDA分列。只比较原生确有的FREE+同articulation affine与实际contact/force，原约束乘子仍由独立KKT给出；Newton不支持的effort/velocity limit不删本项目门。非零PD、联合限位、一般旋转积分不硬说同模型，不给跨框架速度倍率。

Wheel后置每backend一个完整gzip：仅public API，contact W4原40+21、A04-p1额外4步、两真实bank容量拒步/足额重试（若需对齐最多2个真实接受步），保存实际步数；不复制全部49节点。消费者带原URDF/完整声明，不能import仓库tests/reference或私有核推进。核实际wheel SHA、完整installed成员与loaded origins、源码外cwd、完整owned records、关闭资源。失败仍保存prefix与实际状态，不预填通过。

## 7. 执行次序、成功条件与成本止损

1. **前置**：C9/C10正式接受；父冻结B/D/K、公开字段/状态码、共享solve/root drift提取。若只能靠复制完整integrator或先无约束step再覆写follower，本片暂停产品；退回唯一共享阶段提取，不重跑旧矩阵。
2. **首档**：A01有效RED后最小公开step两backend；再A03链、A04三个姿态与短A02/A05。full root反作用、原行反力和实际双侧力未过，不扩长窗/规模；失败只重验受改动节点，保原首次失败。
3. **核心限域门**：49实例按实际阶段完成、43完整文件可关联；五锚/图身份/full与reduced一次发布、PD/历史、材料来源/全载荷、真实生命周期、规定代表恢复及两bank失败均按原门通过。旧generic邻域只选父实际共享diff影响的最小ordinary/fixed-affine节点。未跑native/wheel另列，不能先称正式接受。
4. **成本门只调调度，不调物理**：用首批实际短节点的完整记录成本估算后继；若单61步窗预计>5 min、W1/W4代表组预计>15 min，父可先交短档有限能力并把完整原节点标NOT_RUN/排后，不能把40改4后沿用同节点名称宣称通过。不重复测旧分页/读回或为凑p50再跑物理。
5. W32仅在核心已过、父确认真实内存/磁盘余量且4步预计≤3 min后排一次；不合适即跳过这一个可选节点，保原43文件与能力边界。资源越限在原首次处保存失败/前缀并停该配置，不清旧数据或反复尝试。
6. 任何full/reduced漂移不一致、原图行缺失、模型来源不符、晚异常半发布、未定义法向/无材料来源或真正超原数值门，都保存原请求并收束。未知域不降格成分离，不把HOST narrow称device narrow，不把W次HOST调用称设备批；A04/生命周期若不成立，不靠调物性、放门或增加证明循环救绿。

当前交付只有本页；所有拟NEW产品/测试/consumer未获本页实施授权，未运行任何import、数值、collection、测试、静态检查、构建或原生对照。**AUTHORING_RELEASED / PROPOSAL_NOT_FROZEN**，由父审后另冻实施范围。
