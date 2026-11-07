# C9 普通浮根完整树驻留：开发与测试

状态：**FROZEN / AUTHORING_ONLY**，2026-11-08，持续无截止。父基线de2a66c（C8正式接受）；调研来源为[浮根提案](../../development/p65-resident-floating-tree-next-proposal.md)、现有 floating_tree / FloatingRobotParticipant / TYX固定普通与affine storage，以及本机Newton1.7源码。此计划裁定提案未定API、工程域和文件所有权。不是设备通过结论。

## 可交付能力

一个world是一台真实floating root + ordinary R/P子树/forest，0≤n≤16；B=n+1，D=n+6，Q=n+7，motor_count=n分别处理。与既有fixed/affine/scalar worlds共用原TyxWorldBatch runtime、锁、global active/epoch、提交/拒步/restore，不创建第二运行时。不拆成6个伪关节，不用HOST推进后上传冒device integration。

首片交完整动态/有界PD、root与全child设备FK、finite sphere/capsule/hull接受态AABB/BVH、公开snapshot/restore/diagnostic/summary及来源。保F3 root principal frame与fixed-tool聚合。prepare_contacts对含floating几何的注册明确拒绝；仅未参与旧contact计划的浮根world可在同batch正常独立推进。不是浮根接触、affine、CCD、持续摩擦、同world多个独立participants或全GPU窄相。后续依次扩ordinary floating contact、floating affine+contact，不重做首片物理核。

## 冻结公共API与状态

- 从phymeo.backends导出 FloatingRobotTree、FloatingTreeInputs、FloatingTreeCheckpoint、FloatingTreeKinematics、FloatingTreeDiagnostic、ResidentFloatingTreeBinding。FloatingRobotModel/State和FloatingRobotInputs沿原公开模块，不再复制模型。
- FloatingRobotTree(model,state,*,kp,kd,effort_limit,delay_steps=0,control_max_iterations=512)。gain/cap长n、非负；n=0显式空数组。delay仅0/1，iterations沿原1..4096整数且非bool。model/state/gains由编译入口深拷贝为真正owned immutable数据，完整FP64模型token保根惯性frame/COM/重力和原children图。
- spec.inputs(drive, external=None)返回FloatingTreeInputs：drive[n,3]为target/rate/feedforward；external沿FloatingRobotInputs的effort[n]、force[B,3]、torque[B,3]，None为原零外载。输入不绑定某个mutable spec，不兼容SceneInputs猜测。batch.set_controls({key:packet})先验证所有world再上传/提交，delay仅drive triplets；external即时使用。电机与显式非电机effort只各加一次。初始held drive target=原qj，rate/ff与外载0，初始delay队列为同一triplet；n0为空队列。
- q=[root body-origin world xyz, body-to-world XYZW, child qj]，v=[root-body-COM world velocity, world omega, vj]。body0是真root；child i→body i+1。root COM不是系统COM。
- FloatingTreeCheckpoint保持字段world/topology/tick/position(Q)/velocity(D)/applied_effort(n)/generalized_applied(D)/history/control。control完整tuple=(drive[n,3], external_effort[n], force[B,3], torque[B,3])；history delay1为[n,3]、delay0空。原position为anchor+设备local原点FP64展开，quat/v为实际FP32值。restore须先校验整个batch，不能把Q/D套固定n维。
- FloatingTreeKinematics：tick、joint_types、position(Q)、velocity(D)、link_position_m[B,3]、link_rotation[B,3,3]、com_position_m[B,3]、linear_velocity_m_s[B,3]（body COM）、angular_velocity_rad_s[B,3]，及显式joint_count/body_count/velocity_count/configuration_count。所有数组owned readonly；root字段由第0体取得，无歧义重复状态。
- FloatingTreeDiagnostic：entry_tick/exit_tick/dt_s、entry_position(Q)/entry_velocity(D)、linear_jacobian/ angular_jacobian[B,3,D]、mass[D,D]、bias/external/acceleration[D]、drive[n]、iterations/control_residual/equation_residual；可增加不重复求解的motor mobility等可核字段。诊断只属于最近成功attempt，初始化/restore/数值拒步后不能返回旧成功诊断冒新值。
- tree_kinematics/tree_diagnostics可含旧树与新typed结果；旧scalar kinematics/prepared scalar reader仍拒绝所有tree。summary累计全B动能，非只根。stats单列floating_world_count及资源/上传/读回/提交；padding/空n不能冒参与物理。

## 完整方程与数值域

全B COM的D列J保根平移I、绕root COM的旋转列、真实祖先joint列。M=Σ(m JvᵀJv+JwᵀIworld Jw)。root Iworld=R Rprincipal diag(I) RprincipalᵀRᵀ；孩子保原惯量语义。bias为公开速度导数为0的完整解析COM/angular加速度、重力一次与omega×Iomega；参考HOST源码仅算法依据，设备不调用HOST当前FK/M/solve。

M a+b=E tau+gexternal，E只含最后n列。先full SPD Cholesky；驱动可以显式Schur消root6，或使用full inverse的joint-joint块（严格等于Schur inverse）进行原有限预算PD投影坐标更新；两者都是完整耦合。根不得加fake motor/cap或手工抵消反力。full D方程残差2e-5，原每轴PD残差5e-5*max(1,abs(tau),abs(expected))不变，无对角补偿/clip eigen/fallback。所有非有限/非SPD/不收敛明确拒步且原状态完整保留。

同一final velocity只drift一次：root COM+=h*vCOMplus，world omega的左乘exp-map更新quat，originplus=origin+h*vCOMplus-(Rplus-R)*croot，child q+=h*vjplus。显式quat归一化是姿态积分的一部分，不能另投影错误速度或用Euler角代替。
- 原初始FloatingRobotState的FP64单位quat门保留；转FP32后及restore/candidate以abs(sum(q²)-1)≤1e-5核单位性，零/NaN拒绝。每接受步exp-map再归一化、保持连续符号，不强制w正造成轨迹跳号。
- anchor固定为初始root origin原FP64，状态只存相对anchor的FP32原点。绝对anchor及全部参数须能转换有限normal FP32/zero；公开位置使用原anchor+实际local FP64展开。device bounds使用anchor FP32且其舍入误差进入下述已声明padding，不能声称无限精度位置。
- root local xyz与child q各绝对值≤4096；root/child速度及loads沿原finite normal FP32/zero输入门，候选必须finite。每步最终rotation vector的欧氏范数≤pi，超域STATE_DOMAIN拒绝。初始/恢复/候选一致，不能测试后改门。
- h沿原正normal FP32可表性。最大n16；无隐式减小h/重试。
- finite bounds沿C4每非负项外向计算padding预算，增加一层root旋转和root local位移reach，覆盖anchor舍入、FP32组合与trig误差；这是明确工程包络，不称formal实数证明。测试从原材料+公开实际q独立计算顶点全包含，巨大共同平移同时检查局部运动/身份，不以巨大world相对容差掩盖局部力学。

## 设备存储与原事务接缝

新storage为原batch sidecar：root[T,13]（origin3/quat4/COMv3/omega3）和joint[T,max(1,nmax),7]各双bank，FK[T,Bmax,18]、全B twist、energy及global ticks配对；n0 padding不得参与。机械临时fullD matrix、J[B,3,D]、full external、motor选择均有明确维数和计数。kind9/10为delay0/1，旧scalar kernel必须跳过，父负责。

FloatingTreeDeviceStorage沿现storage公共protocol：validate_controls/commit_controls/upload_controls/controls_uploaded、stage_advance/check_advance/finish_advance、validate_restore/stage_restore/check_restore/commit_restore、snapshot/kinematics/diagnostics/summary/borrow_geometry/close。stage先完成所有新bank与状态码，再由父一次active/epoch发布；任何上传/读取/late check异常不swap。numeric reject保root、joint、FK、energy、drive/history/tick；邻world可接受。same-tick restore仍使旧query失效。构造/close按原ExitStack，不遗漏晚失败资源。

ResidentFloatingTreeBinding(key,world,group_index,local_points_m,*,topology)：group范围[0,B)，不是motor编号。from_asset(asset,source_key,*,key,world)核真实普通浮根/固定折叠资产模型/全部材料/shape IDs/radius/revision；原source root -1→0、child i→i+1，folded工具site固定变换只应用一次。有限S/C/H局部点+实际接受FK直接生成device bounds，不能warm snapshot/HOST FK/动态包络上传，必须记录实际计数。不同world/body/同体多shape分别保身份。

## 独立多案例库

h=1/1024；same-entry state atol5e-6/rtol5e-5，40步累积atol2e-4；M/b/J/动力学参考沿C4/浮根HOST原物理量门（M2e-5、方程2e-5、PD5e-5）。结构/身份/全数据恢复必须精确。输入以下预声明，扩变换不从输出调参。

- P01 root2kg+child1kg P、g=(0,0,-2)、omega/vj0、无驱动：root a=g、alpha=aj0；40步全系统COM ballistic。固定qj=1/8、共同boost/轴循环。
- P02 root-only2kg、c=(1/8,0,0)、I=(1/4,3/8,1/2)，原点F=(0,2,0)、tauCOM=(0,0,-1/4)：aCOMy1、alphaz-1/2。F3非对角tensor[[3/8,-1/8,0],[-1/8,3/8,0],[0,0,1/2]]的纯COM矩(1,0,0)→alpha(3,1,0)。40步原作用点正确运输与root-origin/COM分别核。
- P03 root2+child1同心P_x，Iroot(.5,.5,.5)/Ichild(.25,.25,.25)，内力+3：Mxj[[3,1],[1,1]]，arootx-1.5、aj4.5、child世界3；同初态child外力+3则root0、aj3。40步反冲/净动量；翻向、非轴安装和共同boost留出。最早public首节点用此，不用孤立球冒浮根树。
- P04同心Rz内矩1：alpharootz-2、alphaj6、child4；再子COM偏置(1/8,0,0)、root惯性frame旋转、非零入步omega/vj，独立全7×7 M/b/J和虚功。不要把离散转动要求为严格能量守恒。
- P05真实非轴R/P分支、异构n=0/1/9/16，W1/W4/W32，完整40步+19→21恢复。fixed-tool聚合root M3、COMx1/12、I(.3,53/120,13/24)，工具F(0,3,0)给ay1、alphaz12/13。全部原工具材料与惯量从真实loader核，不重复加质量。与旧ordinary/affine/free至少一混合W4验同batch发布与拒步隔离。
- P06 delay0/1、低cap与宽cap小残差、有限iterations失败、root/child越域、badquat、非法Q/D与异图restore、错误world/owner/关闭/readonly；两bank上传/真实candidate后latefault/retry、globalcandidatecap与sametickrestore。保失败前完整状态、原输入、最后accepted、资源。
- P07 S/C/H的root/child/工具绑定，实际q/quat变动下所有material vertices含于当前devicebounds；两端旋转不是CCD。warm无alloc/HOST FK/snapshot/包络上传，必要sync/readback如实记数；filtered pair与owner拒绝不生成物理力。floating prepare_contacts明确unsupported，既有C8不改变。

独立参考NEW脚本不得导入Phy product/TYX求M/b/J：用原输入的齐次变换一阶/二阶time jets与惯性求和/全D线性解；产品的递归bias不能当唯一oracle。不使用有限差分自证。解析P01–P04为独立首步锚，HOST浮根可额外交叉核，不作为唯一期望。Newton独立FREE根CPU/CUDA至少freefall/eccentric/recoil两类40步，保root solve-origin、joint_qd的parent-anchor basis与body_qd的COM/world区别；使用pin源码解释原生读数，不能套C8 fixed-root origin0。

所有实际case保存完整input/model/gain/held/queue/M/b/J/entry/accepted/FK/energy/实际计数；测试预配保存身份，不按输出挑重建。功与KE分列观测，未承诺跨引擎性能。

## 分工、次序与验收

原robot作者NEW仅：
src/phymeo/backends/_resident_floating_tree.py；
_resident_floating_storage.py；_tyx_resident_floating_kernels.py；
_resident_floating_binding.py；_tyx_resident_floating_bounds.py；
docs/benchmarks/p65-resident-floating-tree-implementation.md。
可只读原HOST/fixed/affine与本机Newton；不得改OLD/export/batch/plan、merge父、执行任何项目命令或新增助手。先交schema/接口小checkpoint，后kernels/storage/binding与明确UNEXECUTED报告。若数值/架构矛盾及时报告具体最小例。

原RD作者NEW仅：
tests/scale/test_resident_floating_step.py；
tests/scale/test_resident_floating_collision.py；
tests/reference_newton/robot/resident_floating_reference.py；
docs/benchmarks/p65-resident-floating-tree-tests.md。
先提交独立P03 public首节点：旧HOST模型/7×7真值成立后在函数内查询新公开API，不能module import缺失导致假RED。随后完整矩阵与自包含独立参考，不使用作者设备中间值作oracle。Newton driver/源码外consumer在产品交接后沿原agents再另行NEW小范围，不同时抢产品owner。

父唯一OLD：_world_step_layout.py、_tyx_world_kernels.py、tyx_world_batch.py、_resident_collision_plan.py、_world_kinematics.py、_world_readback_plan.py、_resident_contact_plan.py（明确拒域）、backends/__init__.py；实际若其他薄接缝需要改先文本记录。父读接口/首test后唯一freshguard有效RED→薄接缝实施→首interpret/CUDA；完整作者后新矩阵/少量受影响旧回归→Newton原参照→同次保存审计→新wheel源码外消费与完整源绑定。所有执行严格串行，绿同源不重复，首失败保留。其他独立能力按计划继续，禁止用长证明循环代替公开链路测试。

结束本片仅C9普通浮根设备树限域接受；通过后才交浮根接触薄接缝。持续目标ACTIVE，不按测试数/清单数称总体完成。
