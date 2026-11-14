# T1a：异构操作可控性诊断与八画面视频

2026-11-14；基于B3产品135752e4和计划第14节。用户授权继续下一批并展示视频。
本批是诊断控制器、真实CUDA仿真和视频连接；没有预训练语言模型、共享策略或optimizer，视频不得称学习成功。

## 调研结论与实现边界

原8模型全部使用gear=1、NONE JOINT motor、ctrl/force范围±20；原摩擦、solver和成功判据保持。
SCARA RRPR可先用竖直P和双指验证夹持；空间6R用真实TCP位置IK后验证搬运；PPPR用于首次PushT诊断。
RR/RRR的圆周工作域不支持直接追到部分目标中心，推杆必须靠真实边界接触作用；IK不收敛只标本次规划未收敛，不证明全局不可达。
原Panda HOST载荷路径仅作分阶段控制参考，不复用为TYX成功证据。

官方依据：[MuJoCo 3.3.7 Jacobian](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-jac)规定kinematics→comPos前置；
[RNE](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-rne)的flg_acc=0提供bias；
[执行器模型](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#actuation-model)区分ctrl、传动和广义力。
本批CPU仅编译、解析Jacobian、IK和bias诊断；真实物理全部TYX CUDA，并禁止CPU mj_step/step1/step2。

## 先登记再执行

1. 新增通用scalar native JointPD与有界点IK；按真实name解析q/v/actuator地址，拒绝不支持的传动，不通过模型重排硬凑地址。
2. 首登记增益：hinge Kp25/Kd3，arm slide250/12，finger150/2；加真实当前bias，motor限制交集内再限±19.9。
   机器人motor请求经atanh(ctrl/20)送原Env scale20/tanh；不对目标FREE施力或改其q/v。
3. 关节参考使用三次平滑轨迹，IK最大100次、阻尼1e-3、step上限.15、位置误差5e-4m；不收敛的整个路线保持原机器人参考并记录原因。
4. 抓取窗口2.2s：settle .2、descend至.7、close至1.1、lift至1.7、hold至2.2。闭指参考.044m。
   放置另有transport至2.5、lower至3.1、release至3.5、settle至4.0。PushT .2/.8/1.8/2.2阶段，首次只做保守边界接近与直推，不预承诺yaw/IoU成功。
5. 先HOST单元测试和8模型固定IK预检，再一次8-world CUDA 2.2s诊断；h=.001，不改原场景。
   任一原Goal结束则保存final状态并停止本批采集，不把其他world记成功。物理失败保留真实prefix和错误，不生成剩余帧。
6. 保存每个接受步的动作、PD诊断、同入步成功帧/指标、完整公共接触身份与载荷及q/v；每10步加初态/末态保存回放状态。
   12类/数十套已绿组不重跑。root唯一串行执行，作者只纯编码；渲染在物理进程结束后执行。

## 验收和停止

- 通用控制接口：独立native bias/IK重投影/不可达目标/错误地址与传动负控通过；不能只返回有限数组。
- 物理诊断：逐world记录IK路线、真实双指载荷、抬升/搬运history、原Goal结果和失败；连续执行成功不等于任务成功。
- 若出现同类控制失败，最多一次基于新轨迹证据的针对修正；不无界调整增益、目标或原物理门。
- 视频：4×2八个真实构型、所有候选site目标与T形轮廓；只重放保存CUDA端点，不插值/移动对象。
  标明“Diagnostic controller - NOT a trained policy”、任务是否满足、实际时间和回放速度；失败prefix也可展示但明确标注。
- 新执行证据位于artifacts/mujoco-tyx-manipulation-control-20261114-01；完成后再更新报告和续作入口。
- 下一次完整操作验收仍须T1b/T1c原抓放PushT窗口及8例/holdout；L1/L2/T2未自动解锁。