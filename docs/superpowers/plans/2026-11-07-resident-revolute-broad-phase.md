# C3 单转铰驻留：真实动力学到设备当前姿态宽相

状态：TESTS_FIRST_AUTHORING。无08:00截止。承接[已接受C2](../../benchmarks/p65-resident-slider-collision.md)，与[F2浮根mimic](2026-11-07-floating-mimic.md)隔离编写；项目执行仍仅父串行。

## 调研与路线选择

原[TYX机器人数学](../../../src/phymeo/backends/tyx_robot_math.py)确有FK/J/M/bias，[机器人参与者](../../../src/phymeo/backends/tyx_participants.py)亦能推进；但[TyxMath.call](../../../src/phymeo/backends/tyx_math.py)每次分配/上传/读回/释放，候选仍为HOST RobotState，不能冒accepted-device resident tree。原[world layout](../../../src/phymeo/backends/_world_step_layout.py)仅球与单滑轨；四列slider命令已在HOST把力投影到不变轴，无法正确代表旋转COM力臂。

选先扩真实单revolute，再扩完整固定根R/P树。后者还缺持久多world q/v与关节偏移、FK/J/M/bias scratch、SPD/active-set、控制历史与完整checkpoint，不只缺一段FK。首片复用原batch/锁/双buffer/epoch/选择复位/统计，不新增第二runtime；更广树明确留后续。

Newton1.7参照为本机Featherstone.step、device eval_fk、CollisionPipeline直接消费state.body_q。DrivePD用当前状态显式算力，与本仓末步implicit bounded PD不同；首参照用无PD的相同恒矩/外载，不能按同一kp数字声称控制律相同。

## 物理与状态合同

拟公共 RevoluteRotor(program,state)，沿DrivenRobotProgram.inputs。首片普通固定根单R、delay0/1、无stops；所有真实质量/COM/主轴惯量和安装变换均从原model编译。轴向惯量 I_axis=aᵀI_COM a+m|a×c|²，在零位joint frame求取且必须正有限。每步在同一accepted q设备计算当前COM力臂，广义外矩为世界轴·(r(q)×(m g+F_COM)+T_COM)+输入generalized effort；不能使用入场时固定转矩代替。随后原标量implicit bounded PD/独立电机帽和一次半隐式推进。外力即时，仅target/rate/feedforward进入原delay历史。

kind3/4必须显式分流，不落入旧kernel的“所有非球都当slider”分支。父拟为每batch按真实kind取command/property最大宽度，旧纯球/滑轨保持原4/9；rotor命令候选10列[target,rate,feedforward,generalized,Fx,Fy,Fz,Tx,Ty,Tz]。这与完整sidecar的成本由案例方案裁定；当前尚未冻结内核签名。snapshot/restore必须完整保存held controls和延迟，失败不得部分消费历史。kinematics的角度单位为rad、速度rad/s，observe动能用I_axis，不能误报q_m或沿用平动质量。

拟ResidentRevoluteBinding(key,world,zero_motion,*,origin_m,orientation_xyzw,axis_local)及from_asset。零位包含原有限形状、安装与局部shape变换，原IDs/revision不变；独立核source shape和真实物理owner。形状可以是偏心sphere、finite capsule、box/hull，同体多shape。同锁读取accepted q→设备旋转全部有限形状→保守AABB→原BVH；不能旋转一次AABB角点就冒精确形状，也不能用只包中心/端点弦的范围。当前姿态、CCD及接触反馈三者明确分开。

三角函数不能直接借用C2乘加膨胀。必须先从TYX实际lowering/官方CUDA primary math确定有限q域与sin/cos误差预算，静态系数及中间运算也向外计入。未知/非有限/超域明确拒绝，不按输入标签走夹具特例；工程宽相允许有界保守多候选，不能漏掉真实有限表面。用户不要求理想无限精度相触，保持合理工程精度和误差报告。

## 首批六组独立案例

1. m2、COM(.25,0,0)、I_COM diag(.125)、z轴，I_axis=.25；g0、τ=.25、h1/64、初速0，40步v_n=n/64、q_n=n(n+1)/8192，不复位。
2. 同例g=(0,−2,0)、COM F=(0,2,0)、drive .25，q0时alpha−1、qπ/2时alpha+1；另做非主轴和共同刚转，检出错误恒力臂/轴解释。
3. 正负角、偏心sphere/finite capsule/局部旋转box/hull多shape同体，独立原角点/轴端点范围与same-tick；相触应候选，真正有限域外应分离。
4. delay0/1、饱和/非饱和、即时外载；d1零预史恒矩独立v_n=(n−1)/64、q_n=n(n−1)/8192。
5. W1/W4/W32及球/滑轨/rotor混合，两个active buffer、同tick restore失效、局部reset恢复history；坏owner/frame/revision/source在分配前拒绝。
6. 单world溢出/超域/最大tick不发布该候选和历史、邻world按合同继续；query/device failure不改last_result；close/empty/capacity。暖态真实统计不得隐藏snapshot/envelope上传/新增allocation。

完整实际质量、shape数据、q域、误差门与边界反例在casebook/tests-first写定，父审阅后冻结。不能凭有限值或一次静态query就宣称驻留成功；至少真实动力学先推进，随后使用接受态，保存每步q/v/载荷/tick/bounds/candidates和必要传输计数。

## 作者与批次

- T1当前：原robot tree c92d730起，恰3NEW：docs/development/p65-resident-revolute-casebook.md；tests/scale/test_resident_revolute_step.py；tests/scale/test_resident_revolute_collision.py。只文本/手推/fileIO/Git，UNEXECUTED，不merge父、不动C2/F1。
- T2拟NEW product：_resident_revolute_world.py、_resident_revolute_binding.py、_tyx_resident_revolute_kernels.py、_tyx_resident_revolute_bounds.py；类型/编译/packet/checkpoint/step/pose-reader按薄接口分离。须先冻结具体签名才授编码。
- 父独占旧_world_step_layout、tyx_world_batch、_tyx_world_kernels、_resident_collision_plan及必要kinematics/readback文档或units。既有球/滑轨行为与纯旧batch字节/计数应保留；必要旧测试因接缝变更才执行。
- T3父串行HOST协议/interpret/CUDA/混合真实窗口，原agent交叉审阅及新源码外wheel；不预collection/probe/重复已绿旧组。性能只有同输出实测后可陈述；本片不默认声称提速、全GPU窄相、通用机器人或CCD。

## 工程三角函数裁定（2026-11-07）

TYX实际CUDA lowering为sinf/cosf且fast_math关闭；NVIDIA误差表的observed maximum不是形式保证。采用工程精度验证，不为该差别新增Taylor区间内核或返回无限精度证明路线。首输入/restore/候选统一有限stored角域拟|q|≤4096 rad，覆盖正负多周回及边界；父审最终casebook后冻结。域外明确拒绝，后续可加入range reduction。

设备AABB按三角函数计算误差、原有限shape尺度、中心/安装与中间FP32计算给显式padding；以固定多姿态、近触、细长shape、大公共平移和真实interpret/CUDA对独立FP64几何oracle验证测试范围内无漏候选，允许少量多候选，不声称全部实数形式认证。动力学仍真实sinf/cosf，不因宽相padding更改积分；不得靠人为紧miss距离倒调padding或夹具。作者保留官方来源和限制，本片不实施Taylor备选。
