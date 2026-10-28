# P6.5 节点与固定平面的动态接触重放合同

用户已授权持续实现P6.5，按续作入口先做真实预定义平面支撑/摩擦。此设计是实施者在该授权内的数值选择；不是新的双臂范围或完整碰撞声明。沿用既有完整重放、原求解器和全局事务；Windows C++长期TYX-003仍外部阻塞，NumPy/实际tyx CUDA与interpret可以独立推进。

## 路线与有限范围

选择位置级非穿透＋末材料速度的库仑摩擦，通过完整重放求解接触积分输入。相比仅在预测后修改末速度，本方案同时改变位移；相比新写单体积分器，材料与后端仍由原program执行。首版是已登记particles域的指定节点与静止无限平面，固定J=I、节点质量常量，避免提前假定机器人非线性J/M方案。多节点、同节点多平面及底层布距离材料可组合；其他场景域由原program一起推进但不对其添加接触。机器人接触、面厚度、CCD、自碰撞、弯曲及双臂仍后续必需。

本切片使用单个mu>=0，同时作为静、动摩擦系数；静止落在盘内，滑动在盘边。不声称静摩擦系数高于动摩擦的模型、恢复系数或完整接触时刻定位。

## 接口与所有权

`ParticlePlaneContacts(domain, indices, normals, offsets, friction)`为不可变绑定：indices形状(k,)整数非负；normals(k,3)、offsets(k,)、friction(k,)均有限FP64；法向长度需在1±1e-12，再连同offset按其长度归一。接触可行半空间为n·x-offset>=0，法向指向允许一侧。绑定点不能是逆质量0的固定节点。输入数组须copy+readonly，不可借用调用者buffer。

`ParticlePlaneContactProgram(scene, base_program, contacts, *, registry=None, inner_settings=None, relaxation=1.0)`公开导出。contacts为绑定tuple，签名包含base物理签名/capabilities/history、绑定及inner设置/松弛；SceneStepper继续负责绑定变化拒绝和整帧原子性。要求base数学provider、FP64存储；真实后端继承base能力，不冒称全GPU接触外层。选择节点必须有注册position/velocity及force字段、ParticleModel与particles族；其他注册字段/history由原快照保留。inner_settings若给出必须substeps=1，否则从outer设置替换substeps=1/record_substeps=False；外层每个子步对应一个完整原program重放区间。

底层program由外层program拥有，外层close调用一次base.close。每个临时Replay借用base，不在每个子步结束时关闭共享factory；每次原工作区仍必须关闭。关闭异常沿SceneStepper原事务处理，构造失败不能留下资源。

## 离散物理方程

每个外层子步初态(x0,v0,history0)，长度h。接触未知量为世界冲量p=pn*n+T*pt（N·s）；全域按节点聚合，作为原force的p/h加量。每次重放必须从完整初态出发，不从上一候选推进，不再加速度kick。

令候选gap g=n·x1-offset，u=T.T*v1。求pn>=0、g>=0、pn*g=0；摩擦为u=0时||pt||<=mu*pn，否则pt=-mu*pn*u/||u||。g0>0但本步到达平面时，允许末vn=-g0/h：这是位置级一阶非穿透时间离散，下一步继续求接触，不声称精确碰撞时刻或CCD。初态g0低于冻结位置门槛则拒绝；不做未记账的修复位移。

完整原program的节点结果必须满足v1≈(x1-x0)/h（半隐式积分＋材料位置重建），不兼容的速度后处理必须拒绝，不能仅因内层accepted就通过。该关系同时用于检查坐标量化吞掉位移。

接触端点功为sum(p·((x1-x0)/h))，单位J；静止平面下不得为正超过原速度门槛×sum||p||。报告对应平面冲量-p及聚合节点输入，不把数值位置稳定化画成力。本切片没有额外位置稳定化。端点功是离散物理账本，不等于完整材料总能量验收；解析例独立验证动能/外力功及耗散关系，完整P6.5总能量/网格/步长门槛仍需后续场景验证。

## 求解与最终审计

每个outer position_sweep按声明顺序做一轮GS；初始p=0并执行一次自由重放。对于当前接触、节点逆质量d>0：pn'=max(0,pn-g/(h*d))，pt'=projection_disk(pt-u/d,mu*pn')；按0<relaxation<=1松弛更新整个p。候选输入有变化就聚合所有接触载荷并从初态重放。d只是固定点预条件器；材料刚度/跨点影响由真实重放承担，不把d当作完整材料Jacobian。没有全局收敛保证：预算内达不到原式门槛就拒绝并全帧回滚。

position_metrics在最终候选上重新计算，不能复用更新前量。检查法向非负/穿透/有冲量时闭合、摩擦锥及滑动方向/静止速度、自然映射残差、节点运动学和接触端点功。速度残差除reference_speed，gap/h除reference_speed；位置门槛=min(reference_length,h*reference_speed)*outer.tolerance；与outer.tolerance一起使用，不能因为载荷或背景坐标大就加相对误差地板。outer.tolerance必须正。

计算gap采用原始存储FP64系数/坐标的精确有理点积后转换为float，避免大坐标点积的取消误差伪造闭合；这只是静态线性平面表达式的审计，不是完整CCD证书。数值溢出/下溢及不可表示载荷由现有重放和浮点陷阱拒绝。载荷聚合按节点逐分量fsum，检查实际输入账本与聚合值；不要漏同节点多接触。

prepare_velocity不再重建或施冲量，确认接触已收敛后保留内层原v；velocity阶段只复核。诊断包含接触world impulse/平均force、环境反作用、最终gap/速度/功、迭代/重放次数、最后内层report和累计真实执行计数。返回数据独立拥有；失败候选不得泄漏到持久history。

## 独立验收（首版）

1. 无摩擦自由接近、恰好支撑、分离、初始穿透拒绝；g0=.01m、vn=-1m/s、h=.02s、m=1时到达x1=0、vn=-.5、pn=.5，下一步停住；按上面离散方程验收。
2. T=2s、h=.01/.005，m1、法向压力20N、切向g9.81：mu=.6静置不漂移；mu=.2按a=5.81的半隐式解析滑动；无需调用者给接触冲量。无pin/weld。
3. 同节点两正交平面与多个节点，交换顺序仍满足最终原式；合力和环境反作用只计一次。
4. 2节点距离材料与平面同时满足各自残差；独立静态/一维离散解对照，不能先解材料后破坏它。原base program的非接触域/history仍推进一次。
5. 最后子步内层数值/关闭失败、外层迭代预算不足、超尺度表示损失、绑定变化、返回对象污染后重试与新实例一致。
6. 实际tyx interpret/CUDA动态支撑/滑移和材料例有launch；C++仅短功能对照，长期依赖阻塞保持开放，不重复已知压力失败。核心3.11与pip3.12、源码外wheel、新API、CI通过。
7. 一次fresh阶段最终审阅及必要TDD修复；串行小负载计时保留全部失败，记入性能/遗留/tyx各自目录，不优化。

## Builder check后的冻结补充

1. 包装器另外逐节点核验`inverse_mass*norm(delivered_integral-fsum(contact_impulses))/reference_speed <= outer.tolerance`，原Replay相对1e-10检查仍保留；不能以大载荷允许大绝对误差。实际复现F=-2273179069.627076N、h=.056872685527390325s时，候选p=129281798.3743464N·s可产生零末位移/速度，而输入差积分为129281798.37434642，差1.49e-8；outer速度容差1e-10必须拒绝。
2. 法向/切向自然映射冲量残差乘d/reference_speed；gap以min(reference_length,h*reference_speed)归一。初态只有`g0 < -epsilon_g`拒绝。首版为统一门禁，显式inner_settings的tolerance/reference_length/reference_speed必须与outer相同，允许迭代预算不同；不允许隐含放宽材料门槛。outer metrics还纳入最后inner的残差。
3. 每次原工作区关闭失败触发整帧恢复；帧已返回后的factory.close失败只按原SceneStepper传播，不追溯撤回已交付结果。借用factory.close为空操作，外层factory.close幂等调用base.close一次。
4. 累计launch/upload/download诊断只统计成功返回的重放report。失败内层诊断会按原事务清空，失败尝试的实际launch总数不可由该计数推断；性能记录另外保留失败事实。不得把未记录失败执行写成零执行。

公开诊断schema：外层每子步`diagnostics["plane_contact"]`包含world_impulses/world_forces/environment_impulses、input_integrals、gaps、relative_velocities、contact_work_j、replays及inner_report；`diagnostics["execution"]`累加成功重放的原执行计数。输入顺序为contacts绑定顺序及其indices顺序。

## 最终审阅后的数值范围修复

一次独立最终审阅复现两项Important。第一项：以FP64先算完的integrated_loads不能再拿来证明精确输入积分。现在从original/effective实际存储force与h逐分量用Fraction计算差值积分，减去同节点各接触的实际世界冲量之精确和，乘逆质量；以L1速度误差/reference_speed在有理数域直接比较原容差。L1保守覆盖原Euclidean检查，不能被输入减法、乘h或fsum末位舍入掩盖。Replay原账本仍是FP64显示量，没有因此变成精确数值对象。

第二项：v1≈Δx/h可能让两个共同舍入错误相互掩盖。此有限包装器增加canonical自由速度预测器的保守输入尺度门禁：每个有接触的particles域中所有动态节点，令S=sum_axis(|v0|+|h*g|+|h*d*effective_force|)，gamma16=16u/(1-16u)，要求gamma16*S <= tolerance*reference_speed。u=2^-53用于NumPy，u=2^-24用于当前实际tyx FP32路径；全部以存储数的有理值计算和比较。该预算涵盖自由预测运算及输入转换，是支持范围限制，会拒绝一些恰好能精确相消的大输入；不是测量误差，也不是材料迭代全过程的误差证明。现有材料、实际运动学和最终接触原式检查仍不可删除。

首版仅接受已登记计算精度的numpy、tyx:interpret、tyx:cpu_cpp、tyx:cuda；不根据FP64存储推断未知后端的计算精度。未来后端需要明确登记并独立验证。每子步metrics包含prediction_roundoff预算；这不改原积分器、实际tyx执行或物理容差。

独立反例：d=1e20、F=-10、h=.1、p=1的精确积分速度差5551.115m/s；d=1、F=-1e16时为.05551m/s。h=.125、v0=-1e16、d=3时FP64候选p=3333333333333333.5隐含末速度.5m/s；d=1e20、p=.0001时约.4792m/s。原版本都能错误接受零状态，修复后拒绝并恢复原帧；actual tyx大速度范围反例也拒绝。正常T=2s支撑/滑移及布材料例仍按原标准验收。
