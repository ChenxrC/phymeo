# P6.5 机器人材料点与固定平面的动态重放设计

用户已授权连续完成P6.5。本设计是实施者在既定范围内的数值选择，先完成必要的单关节转动反力矩检查；不是把固定平面替代后续机器人-布双向接触。既有节点平面、厚度端口和完整重放不重写。独立builder check后实施，一次最终代码审阅。

## 路线与边界

选择原机器人半隐式积分：M0=M(q0)、b0=b(q0,v0)及原外力广义投影A0固定在步初；接触广义积分G=sum J(q1).T*p在最终接受构型重算。原program每次由完整初始快照重放恒定effort增量G/h，q1/v1始终来自原NumPy或actual tyx候选。相比只在末速度kick，此方案会同步更新q；相比用J0而只检查末几何，它能使报告的最终作用点、材料速度和反力矩一致。它是明确的一阶离散方案，不是隐式更新M/b、变分积分或精确碰撞时刻。

本切片限一个已登记robot域、固定基座树、指定link材料点、静止无限平面；允许多个接触、共享祖先与多关节。机器人域必须符合上述原自由动力学合同；旧接触或任意隐藏机器人后处理如果改变动力学会被最终原式拒绝。其他域按base原程序一起重算，不能宣称它们已与机器人新接触。下一阶段仍必须实现机器人与节点双向碰撞、几何/CCD、完整夹爪。

不选择在此重写机器人积分器，也不将作用点末速度偷换为世界位置弦速度。J(q1)*v1是材料点末速度；(x(q1)-x(q0))/h另行记录，它们对转动一般不同。q1=q0+h*v1按广义坐标独立审计。

## 接口与所有权

新增RobotPlaneContacts(domain, indices, local_points, normals, offsets, friction)。indices为非负整数(k,)；local_points/normals为FP64(k,3)，offsets/friction为FP64(k,)；规范化、有限性及只读副本与ParticlePlaneContacts相同。domain必须是本program的robot_domain，每个link索引有效。法向定义n.x-offset>=0；mu静动相同且非负。实际反作用作用点为最终机器人材料点，接触激活时应在平面容差内。

新增RobotPlaneContactProgram(scene, base_program, contacts, *, robot_domain='robot', registry=None, inner_settings=None, relaxation=1.0, reference_angular_speed=1.0)。registry缺省创建标准注册表并注册robot_family；显式表copy且不得覆盖用户机器人族。机器人模型须RobotModel，状态q/v及输入effort/force/torque已注册。base为mathematical、FP64 storage，后端numpy或当前tyx三模式；未知计算语义拒绝。显式inner的substeps=1、容差/reference_length/reference_speed与outer相同。reference_angular_speed为正rad/s，转动广义速度误差按此归一，移动关节按outer.reference_speed(m/s)归一，避免混用米与弧度。

签名包括所有绑定、base/capabilities、设置/角速度基准。history继承base，所有状态/输入由SceneStepper及SceneLoadReplay完整拥有。每次原workspace关闭正常；Replay借用factory，外层close幂等关闭base一次。构造/末子步/关闭失败按原事务恢复，不改SceneStepper合同。

## 离散方程与接触功

冻结M0、b0及A0；原始输入外力/力矩在q0投影。求M0*(v1-v0)=h*(A0-b0)+G，q1=q0+h*v1。G=sum J(q1).T*p。M0不是M(q1)，最终J1不是J0。所有G一次聚合后由effort输入施加，不通过point_port再施一次冲量。

每接触gap=n.x(q1)-offset。pn>=0、gap>=0、pn*gap=0；ut=T.T*J(q1)*v1。静止时||pt||<=mu*pn，滑动时pt=-mu*pn*ut/||ut||。初态gap<-min(reference_length,h*reference_speed)*tol拒绝，不加未记账稳定化。

末端点功W=sum p.J1*v1，单位J，静止平面必须W<=tol*reference_speed*sum||p||。该功为末材料速度上的离散冲量功，不能叫沿整条曲线的精确功。冻结度量下，相对自由速度vf=v0+h*M0^-1*(A0-b0)，K_M0(v1)-K_M0(vf)=G.v1-.5*(v1-vf).T*M0*(v1-vf)；因此在原动力学/映射满足时，W门禁排除接触给该冻结离散系统增能。报告实际M(q1)能量与冻结能量分别命名，不把此恒等式冒充完整多关节实际能量/重力势能守恒。完整P6.5仍须全场景能量和步长门禁。

环境反作用为-p；关于世界原点的环境矩为-x(q1)×p。报告每点位置、J、材料速度、弦速度、世界冲量/平均力、环境矩及合并广义积分。未达末几何/映射门槛不得提交。

## 外层与精确审计

从p=0的原自由重放开始；每轮GS按绑定顺序更新。当前候选处K=J*M0^-1*J.T，normal mobility kn=n.K.n；pn'=max(0,pn-gap/(h*kn))。切向预条件率kt=lambda_max(T.T*K*T)，pt'=disk(pt-ut/kt,mu*pn')；整体按relaxation松弛。K只作预条件，最终残差必须原式成立。零法向响应且需要纠正时拒绝；零切向响应时ut必须在原容差内并选pt=0。不得伪逆/加背景地板隐藏不可响应。数值溢出、下溢、无收敛均整帧拒绝。

每次更新之后从当前q重新映射全部p并原初态重放；即使p不变，J映射未闭环也须刷新。最终重新计算J1与G1，比较实际存储effort差*h与每点J1.T*p的精确和。精确有理数审计存储的M0、b0、A0、q/v/h、J1/p；差值转为广义速度误差，再按各关节的速度基准比较原tol。不得先舍入账本再比较。可用冻结有理M0分解/求解避免大条件数吞掉审计量，不覆盖native候选。精确算术只证明存储系数下原式，不认证FK/质量系数本身无限精度；实际tyx必须在该原门槛内。

外层metrics包含原内层残差、法向/摩擦原式和自然映射、正冲量时闭合、末材料功、精确广义动力学/载荷映射/q运动学。世界残差用reference_speed；位置用min(reference_length,h*reference_speed)。显式内部tol不能放宽。prepare_velocity只复核，不重建，不再次施加G。

## 首版独立验收

1. 单移动关节质量m2对平面支撑：独立pn=m*g*h，v/q不漂移；自由分离、预算不足、穿透/不可响应拒绝。
2. 单转动杆，零重力，材料端点x=(sin q,-cos q,0)，q0=-.3、v0=1、h=.15、平面y=-cos(.2)。接触应q1=-.2、v1=2/3、pn=M*(v1-v0)/sin(-.2)，检查最终J产生的反力矩和冻结动能下降；J0会得到错误答案，测试必须区分。世界材料速度不等于弦速度。大步只是离散端点例，不声称路径CCD。
3. 两正交移动关节带一个末点，X向20N压力、Y向g9.81，mu=.6静置/mu=.2滑动；T2s、h=.01/.005独立半隐式解，确实施加机器人effort载荷。原布材料域可同时完整演化，但不能称双向接触。
4. 两点共享关节/祖先，多个载荷一次聚合，交换顺序满足原式；自由历史与后段失败恢复，返回污染重试、factory只close一次。
5. 巨大初速/输入相消、q步增量丢失、不同量纲速度基准、零响应、故意不符合冻结动力学的base程序都不能错误接受。
6. actual interpret/CUDA覆盖静置/转动；Windows C++只做明确有限次数功能，不重跑TYX-003长期失败。uv/pip/源码外wheel/全量/一次最终review、性能观察与准确远程CI保持必须。性能只记录，host审计不得冒称tyx性能缺陷。

## Builder check后的冻结裁定

1. 外层规范系数明确使用host FP64：M_raw=mass_matrix(q0)，M0=(M_raw+M_raw.T)/2（逐项对称平均，存储严格对称），b0取host bias；A0从存储原输入及q0的host COM Jacobian精确投影；最终x1/J1也来自host FK/Jacobian。actual tyx仍自己计算FP32 M/b/动力学并返回q/v，外层只审计，超原门槛拒绝，绝不覆盖候选。对称化只定义外层物理审计系数，不更改原积分器。诊断标记outer canonical_host_float64、native实际backend。存储矩阵仍需正定。
2. W及能量门禁均用存储p/J1/v1/M0的精确有理乘积计算，避免相消先舍入。动力学有限容差意味着仅凭W不能证明能量不增加，因此另外直接检查冻结动能差。基准使用本子步第一次p=0的实际native自由候选v_free_native；它已通过原动力学/运动学门禁，避免把tyx自由预测误差算作接触增能。DeltaK=K_M0(v1)-K_M0(v_free_native)。冻结功与DeltaK分别不超过tol*reference_speed*sum(abs(p各世界分量))；L1预算在有理数域计算，不依赖背景速度。p=0重放必须与该自由基准一致，不能给零接触附加能量误差地板。此冻结度量门禁不承诺实际M(q1)/势能的完整能量守恒。

上述选择由实施者根据现有数值合同裁定；代价是某些物理上有效但后端精度不足的候选会被保守拒绝，必须记录和修复表示范围，不能事后放宽门槛。

## 单次最终审阅后的仿射支持域

存储旋转/轴系数保持原host值，使用有理数独立重建树平移、COM和接触点。实际点L1偏差<=tol*min(reference_length,h*reference_speed)，所有旋转祖先臂长L1偏差<=min(tol,1e-10)*该精确臂长L1，实际材料速度L1偏差<=tol*reference_speed。冻结系数前及每次候选检查；空接触仍检查COM。门禁不以世界原点大小放宽，也不替代native候选或认证无限精度三角函数/M/b全算法。范围外SolverFailure并全帧恢复。
