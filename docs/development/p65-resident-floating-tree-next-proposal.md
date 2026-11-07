# 设备浮根完整树：下一片实施提案

2026-11-08。**PROPOSAL / NOT_FROZEN / UNEXECUTED**。本次只读源码并写本文，不授权产品、测试或执行；C8 原生参照/消费者不改。
调研基线：父 `6df08e79ce0c776ecb0047d035967f7c97ae9941` 的对应源码；本机 Newton 1.7.0 / Warp 1.17.0 安装源码。父当前 C8 验收结果另行记录，不由本文推定。

## 1. 推荐范围与现存能力

推荐先交 **普通浮根完整 R/P 树的设备动力学、接受态 FK 与 current-pose 宽相**，复用原 TyxWorldBatch/runtime/锁/epoch；随后分别扩 affine、接触反馈。每 world 一个物理浮根，0..16 个活动标量关节，允许分支、任意安装轴、偏心 COM、根惯量主轴 frame、已有合法 fixed-tool 聚合。不把根拆成六个虚拟关节或六个独立刚体。

HOST 已有这些物理语义：`dynamics/floating_robot.py:15,58,99,145`，`solvers/floating_tree.py:78,114,142,164`；浮根 affine 已在 `dynamics/floating_mimic.py:60,139,183,242` 真正投影/求反力；`builders/floating_urdf.py:69,124` 保 authored frame 与 fixed-tool 物理组。旧 F1 报告里的“非对角根/mimic尚缺”不是当前结论，后续 F3/F2 已扩展，见 `docs/benchmarks/p65-floating-root-inertia-frame.md:29,49,55`。

设备缺口是新状态维度及全根动力学，并非再做一个 parser。`backends/tyx_robot_math.py:13,110,149` 只接受 RobotModel，按调用上传 q、返回数组；它的数学可借鉴，不能当作浮根驻留。现 `TreeDeviceStorage` 的 n 同时是物理 body、速度及标量坐标数（`_resident_tree_storage.py:90,151`）；浮根必须分开 **B=n+1、D=n+6、Q=n+7**。原 fixed/affine 接受范围不变。

| 路线 | 新增实质 | 裁定建议 |
|---|---|---|
| Ordinary 先行，再 affine/contact | 先验证根偏置、完整 M/b/J、无根电机的驱动消元、姿态漂移及 bank；后两项复用已验 full 物理核 | 推荐；每节点都有公开入口与独立真值 |
| 一次 ordinary+affine+全接触 | 同时改 D/B 布局、P 投影、原图反力、body-frame桥、wrench上传和receipt发布 | 不推荐；首错无法区分新根偏置/约束/接触，且不是旧核薄加一个 kind |

首片不声称设备接触反馈、CCD、闭环约束、浮根 affine、同 world 多独立机器人或自由体联合接触。混合 batch 仅并行不同 world；不能用跨 world pair 补缺。首片也不改旧摩擦/抓持门。

## 2. 根约定、完整动力学与驱动

- 公共 `FloatingRobotState.q=[origin_world_xyz, body_to_world_xyzw, qj]`，`v=[root_body_COM_v_world, omega_world, vj]`。这是根体 COM，不是整机器人 COM。body 0 为真实根，child i 为 body i+1；根原点速度为 `vCOM - omega × (R croot)`。
- 根惯量 `Iworld=R Rprincipal diag(I) Rprincipal.T R.T`，保 `root_inertia_frame_xyzw`，不旋转 authored 几何/控制轴来迎合主轴；质量聚合沿现 loader，不再次把 fixed-tool 质量加一遍。
- 为全部 B 个 COM 建立 D 列 Jv/Jw；根平移列是 I，根旋转列绕根 COM，孩子列沿真实祖先链。`M=sum(m Jv.T Jv + Jw.T Iworld Jw)` 保全部 root-child/branch cross-block；`b` 沿 HOST 解析零广义加速度递推、重力及 `omega×Iomega`，不有限差分、不套 fixed M 的左上补丁。
- `M a + b = E tau + g_external`，E 只选最后 n 行。世界 COM 力/矩逐 body 经本入步 J 投影；重力属于 b，g_external 不再加一次重力；禁止 root 六行伪电机或隐藏平衡力。内部电机反作用由完整 M 自动出现，不再另向根手加反力。
- PD 先 Schur 消去根六行：`S=Mjj-Mjr Mrr^-1 Mrj`，`beff=(b-g)j-Mjr Mrr^-1(b-g)r`。用原有有限预算投影坐标求解 S 上的 n 电机，再由全 D 方程求根与关节实际加速度。HOST `_floating_robot_participant.py:192` 已实现此物理消元；新设备实现复用原代数方法/门，不复制第二种接触 solver。
- 保 C4 原每轴 `tau-clip(kp*(target-q-h*vplus)+kd*(rate-vplus)+ff,±cap)` 的 `5e-5*max(1,abs(tau),abs(expected))` 力门和质量方程 `2e-5` 门；再核全 D 方程，不能只看 Schur 残差。宽帽小驱动/大项相消例保留。根无电机帽，原限位碰撞力不顺带实现。
- 一次 velocity-first drift：COM 按最终 vCOM 走 h；世界 omega 用原 HOST 左乘 exp-map 更新 XYZW；`originplus=origin+h*vCOMplus-(Rplus-R)*croot`，孩子用最终 vj 漂移。有限归一化是明确姿态积分步骤，不事后投影错误速度。根惯性frame仅影响 I，不改变这个式子。

建议继续 n≤16：Panda 9 个物理 scalar rows 可覆盖；最大 D=22、B=17、Q=23。每个 D×D 临时矩阵 484 个 float，两个 COM J 共 `2*17*3*22=2244` 个 float；相对固定 N16 的 256/1536 有明确存储和循环增量。这仅是维数计数，不是 GPU 性能结论；实际 scratch/编译/占用须后续计量。

## 3. 最小公共入口和储存接缝（建议，待父冻结）

`FloatingRobotTree(model:FloatingRobotModel,state:FloatingRobotState,*,kp,kd,effort_limit,delay_steps=0,control_max_iterations=512)` 为 world spec，不新建 runtime。n=0 用空 gain/cap；n>0 保 owned n 元参数、每 world 异构 parents 与完整 FP64 model token。

`spec.inputs(drive,external=None)`：drive 为 n×3 `[target,rate,feedforward]`；external 沿 `FloatingRobotInputs` 保存原 n 个非电机广义载荷和 B×3 force/torque。电机 drive 与即时非电机 effort 分块，不能把同一数重复填入。此 packet 不强复用 HOST FloatingRobotCommand（它明确要求 external.effort=0）；该区别需公开文档说明。只 delay 0/1 步的 drive triplets，所有 root/child 外载即时使用；cap/gains 在 spec，原请求和实际 drive 都可读。

沿 `TyxWorldBatch({world:spec,...})`、`set_controls`、`step`、`snapshot/restore`、`tree_kinematics/tree_diagnostics`、`summary`、`prepare_collision/collision`；旧 prepared scalar reader 显式拒绝浮根。新增 `ResidentFloatingTreeBinding.from_asset(asset,source_key,*,key,world)`；显式 direct binding 只声明安装，不冒充资产 owner 证明。

独立浮根 sidecar，同旧 batch 双 active index，不扩写固定树 state[n,7]：建议 root bank `[T,13]` 存相对 world anchor 的 origin3、quat4、COM v3、omega3，joint bank `[T,Nmax,7]` 保存原 q/v/drive/history等语义；FK `[T,Bmax,18]`、energy、状态码同双 bank。力学 scratch `[T,Dmax,Dmax]`、J `[T,Bmax,3,Dmax]`、COM loads `[T,Bmax,6]` 独立计数；parents/kinds/body-world map 真 int32，数值计算 FP32，模型身份不以 FP32 舍入表代替。

checkpoint 明确 Q/D、n 个实际 motor effort、D 个 generalized_applied、n×3 held/history及全部原外载、tick/model token（含根惯性frame/COM/重力）；不能沿旧 position/velocity 各 n 元直接解包。kinematics 返回根 origin/COM、全 B poses/twists 与混合单位 joint q/v；world position/COM 保设备 FP32相对坐标+固定anchor的 FP64展开，R仍实际设备值。summary 全 B 动能；diagnostic 是最新已接受 attempt 的入态 M/b/J/external、选中 drive/最终 a/残差，不是永久历史。initialize/restore/数值拒步后无新成功诊断。

初始/restore/候选统一有限值、正惯量及单位四元数门；孩子沿 C4 ±4096 坐标/速率工程域。根绝对世界平移不进入惯性计算，anchor局部位移与每步转角的有限工程域/归一化门须在 tests-first 时明确，禁止执行后挑域追绿。原有限形状外扩误差策略随新增根旋转深度作显式工程扩展，不称所有实数形式保证。

## 4. 同一次发布、绑定及后续接触

现 `tyx_world_batch.py:417,644` 已先验全部 controls/restore、所有 stage 后同步/status，再一次 active/epoch 发布；新 storage 接同一列表及两向 bank流程。任何上传/readback/晚检查异常不 swap，不清遗漏 dirty；恢复不了真实设备错误则 poisoned，需要重建。逐 world 数值拒步必须同时 retain 根13、所有 joint/history/drive、FK、energy、tick；其他 world 可正常接受，同一次全batch swap。same-tick restore 仍递增 epoch；碰撞借用态按 owner/topology/epoch失效，不能只看 tick。

输入和返回值为 owned 只读副本；不暴露借用设备句柄或内部可写 HOST cache。每 storage/plan 用原 ExitStack 资源回收，关闭一个 plan 不关借用 runtime；constructor局部失败和 batch多plan close异常仍逐个清理。warm FK/bounds 不调用 HOST FK、不 snapshot、不上传动态 envelope、不新 device alloc；允许的状态/参数读回上传如实记数。

`from_asset` 必须核完整原模型/形状/IDs/revision 与 source-local→physical-group-local 固定变换。root collider owner 0，child i→i+1，folded tools 用 sites/material frame；不能因 root 从旧 -1 变0丢静态偏置。用接受态 FK 旋转全部有限球心/胶囊两端/盒或 hull 原材料顶点再生成 AABB；同体多shape、双buffer与同ID异world独立映射。current-pose bounds 不代表连续路径/响应。

后续 contact 仍用 C6/C8 原 entry sampling、有限几何材料、共同作用点和真实两侧速度；root/child均是动态 body，不是固定支撑。力上传 `[T,Bmax,6]` 后一次 `gcontact=sum(Jv.T F+Jw.T torqueCOM)` 加入本次 full external，再 bounded PD/solve；不能重做 mechanics 清掉载荷。每对只采一次，两侧等反力各有自己的 COM 力矩，full root-joint M负责响应。

当前 `_resident_contact_plan.py:95,253,393` 只接受 fixed/affine，假定 B=D=n，并在HOST读实际设备FK/几何做窄相；新浮根要扩 typed packet/group_count、速度核和 D 投影，不仅放宽 isinstance。保材料/作用点两类实际位移、原receipt/gates以及capacity逐world拒绝；候选/receipt/late restore都在唯一swap前完成。未扩前 prepare_contacts 必须明确拒浮根；绝不把接触放下一步或复制HOST浮根状态进设备。

后续 affine 用 `Pfull=diag(I6,P)`，只展开孩子 `qj=P*u+c`；约束行 `[0_(rows,6),C]`，保原每行 multiplier/功以及完整 D 受力平衡，根不受假的 affine约束力。C5 的 full/K组合法可复用生命周期，不能直接套其 N物理body=full速度假设。多独立树/自由刚体共享同world还需 world内参与者/自由体联合 registry与step ownership，不是本浮根world类型顺带获得。

## 5. 建议独立矩阵与最早验收

以下是先行物理输入建议，T1前固定完整惯性/初态/输入/门；本文没有运行任何一个新节点。首步统一建议 h=1/1024 s，首步q/v按各字段的m或rad、m/s或rad/s单位沿原5e-5相对/5e-6绝对工程门，40步沿原2e-4；M/b/J分量门在T1按量纲单列，质量方程/PD用上文原残差门；结构性单位/owner/事务要求准确。参考原同类HOST FP64只作为额外交叉检查，不由新device输出生成oracle。

| 族 | 固定独立真值与击穿目标 |
|---|---|
| P01 自由落体/共同boost | root2kg、child1kg P、无驱动/外载、g=(0,0,-2)，初omega/vj=0、任意固定qj：aCOMroot=g、alpha=0、aj=0；k步v=v0+k h g，系统COM位移=k h V0+h² k(k+1)g/2。根/子/全系统COM分别核，不把aggregate COM当root状态。轴循环/初速度留出。 |
| P02 偏心力与非对角根 | root2kg，I=diag(1/4,3/8,1/2)、c=(1/8,0,0)，F=(0,2,0)作用于body原点，显式tauCOM=(0,0,-1/4)：首aCOMy=1、alphaz=-1/2，origin按新R补偿。另沿F3完整tensor[[3/8,-1/8,0],[-1/8,3/8,0],[0,0,1/2]]，COM纯矩(1,0,0)→alpha=(3,1,0)；不把主值直接当body轴。 |
| P03 根-滑轨反冲 | root2kg、child1kg、两COM初始同点、P轴x、各惯量(.5,.5,.5)/(.25,.25,.25)、g=0、内力+3N：Mxj=[[3,1],[1,1]]，arootx=-1.5、aj=4.5、child世界ax=3、总动量增量0。翻向/共同boost/安装旋转；另child COM外力+3N给aroot=0、aj=3，区分内部电机与外力。 |
| P04 根-转铰反冲 | 上述同心质量/惯量，R轴z，内矩1Nm：根alphaz=-2、相对alphaj=6、child世界alpha=4，总角动量首增量0。再将子COM偏置及根惯性frame旋转，按原 F3 独立7×7公式核 cross M/b/虚功；不要求有限旋转离散能量严格守恒。 |
| P05 通用树/工具/规模 | 非同轴R-P分支、N9和N16、不等质量/非零初q/v；原F01 fixed工具 M=3、COMx=1/12、I=(3/10,53/120,13/24)，工具点F=(0,3,0)→a_y=1、alpha_z=12/13。逐原部件独立求总质量/世界惯量/材料位置，与 folded owner映射交叉核。W1/W4/W32各异构n/parents，含旧scalar/fixed/affine旁观world。 |
| P06 控制/事务/真实几何 | delay0/1、饱和/非饱和、宽cap相消、预算不足、全body原外载、双active控制上传/restore晚失败/设备不可恢复；有限球、capsule端点、旋转box/非盒hull逐ID bounds，静态近触/近失与同key异world；40步不重置，tick19 snapshot重放21步。旧receipt不变/unknown token拒绝，warm成本只记录不许假零。 |

首RED在合法既有 FloatingRobotModel/State/Inputs + HOST解析模型构造后触及缺失公开 world type，不用 collection 缺模块冒有效行为红。P01–P04分别有平移、偏心/陀螺、线性及角反冲真值，不能以单个Panda姿态为充分标准。

接触里程碑再做：有限底座承载/抬离40步；同完整浮树两非相邻分支接触（独立J双方反力、净外力/力矩零而内部运动不零）；affine 同速两P相反full力使 reduced合力0但原图反力非零；另一world完全隔离。只复用已接受几何组合，不重跑同源10组合矩阵；这批若尚无新contact bridge，相关项不得写成通过。

## 6. Newton原生参照与明确限制

本机基路径 `P/.venv-newton-1.7.0/Lib/site-packages/newton/`（P为父工作树）。`_src/solvers/featherstone/solver_featherstone.py:74,90,95` 要求显式 FREE joint，支持R/P/FIXED/FREE；其 effort/velocity limit不等同Phy boundedPD，不能据相同target说相同控制律。用显式力/矩与独立轨迹比较先行，原生接触另列，matched-law推进不冒原生contact通过。

`_src/sim/articulation.py:539,1264,1695` 提供eval_fk/eval_jacobian/eval_mass_matrix；body_qd是world COM linear-first，FREE joint_qd却在parent-anchor axes（`:971,1037`）。选world aligned free frame可先减少转换歧义，斜安装必须保 frame变换；不同基的M/b不可直接逐元素比，先按同一速度/功共轭基变换。`featherstone/kernels.py:1042,1133` 明确COM↔anchor速度转换。

原生 `body_f_ext` 是临时工作buffer，`:1498–1503` 的eval_rigid_tau在step内原位写成负solve-origin wrench。若保存后置值，必须同时保存同次 `body_q_com/body_solve_origin` 并解码 `F=-rawF`、`tauCOM=-rawTau-(xCOM-solveOrigin)×F`；仅反号错误。公开body_parent_f对FREE root是递推残差，不是“固定基座反力”（solver_featherstone.py:108–123）。原始字段与解码字段都保留，禁止把缓存参考点差异当solver bug。

CPU/CUDA各一条P01/P02/P03/P04自身真实推进和40步窗口足以首对照；允许Newton姿态归一Euler与Phy exp-map在原工程门内有差异，不要求逐bit或用Phy轨迹回灌。完整保存原model/frames/q/qd/body状态/原载荷/版本和source；未跑前一律PENDING。

## 7. 文件所有权与分期交付建议

第一里程碑 NEW 全在 `src/phymeo/backends/`：`_resident_floating_tree.py`（spec/compiler/typed snapshots），`_resident_floating_storage.py`（原runtime双bank），`_tyx_resident_floating_kernels.py`（全根FK/J/M/b、Schur驱动、漂移），`_resident_floating_binding.py`、`_tyx_resident_floating_bounds.py`。可借原 algebra求解，但若提取共享函数由父改旧 `_tyx_resident_tree_kernels.py`，不悄改普通/affine公式或引入HOST rescue。

父 OLD（以下均 `src/phymeo/backends/`，根exports例外）：`_world_step_layout.py`/`_tyx_world_kernels.py`显式kind跳过旧scalar；`tyx_world_batch.py`编译/控制/读回/restore/step/close；`_resident_collision_plan.py`/accepted collision view新增浮根geometry；`_world_kinematics.py`/`_world_readback_plan.py`显式类型边界；根exports。碰撞与接触plan的world key/group/token分离，不拿 global slot充storage row。

T1 NEW建议 `tests/scale/test_resident_floating_step.py`、`test_resident_floating_collision.py`，独立Newton `tests/reference_newton/robot/resident_floating_reference.py`，后续公开wheel consumer；不复制已有HOST tests当设备验收。父只在首有效RED后授权产品，完整interpret/CUDA和源码外wheel各按固定案例一次执行，失败保原记录。

第二里程碑在第一验收后分开冻结：先 ordinary 浮根的有限静态/同树动态摩擦零接触（typed B/D桥与一次full装配），再 affine 浮根 P/K+原图反力及相同接触。若优先工作流确需affine，可交换两者顺序，但不得同时扩大首验收。三个节点均沿同runtime，不新造scene/第二solver；持续摩擦、多个独立participant同world、loop、浮根CCD另待明确架构任务。

当前最关键先决条件是把 body_count、velocity_count、configuration_count 与 motor_count 分开，并保持统一发布；没有必须新增第三方依赖。本文没有设备运行结果、性能收益或新的接受声明。
