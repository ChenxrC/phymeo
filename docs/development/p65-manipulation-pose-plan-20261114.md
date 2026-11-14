# T1a-2/3 与 T1b-1：姿态筛查、滑移诊断

2026-11-14，起点1e4b370c；按用户“继续下一批”实施。
父/root唯一串行执行，原scene作者只两个新pose模块/测试文件纯编码；training作者只源码审查。

1. 固定消费上一批world0的1100/1300/1500/1700/1900/2000/2100七个步。
   用trace[k-1]端点q/v与trace[k]实际native channels重建第k步入步；独立native mj_forward不推进时间。
   按geom有序对、point、normal配接触，比较实际世界力、切向速度、原regularization/KKT与pad下缘位置。
   保留全部原模型/参数，不以2μN>mg断言无滑移，不将端点差分加速度冒physical-M KKT。
2. 新通用audit_native_contact_poses：显式1..3完整seed、原每seed IK100步，检查旋转BOX pad全部角点相对PLANE的间隙，sphere/capsule/box在pad轴的投影。
   只筛查，不声称全碰撞安全/抓持；完整绑定/immutable结果/未选坐标不变，独立几何oracle与负控测试。
3. 父固定验证：world1后侧TCP(.375,0,.045)、world3原TCP(.4,0,.1)；world2/3/4原合法TCP候选，以zero和腕joint3/4=(+.35,-.35)、(-.35,+.35)三个seed检查分支。
   不修改物体或目标，不扩大IK次数；有碰撞/间隙问题则不进入实际操作。候选全部失败不证明全局无解。
4. 滑移分析后再明确一次反馈对照及验收窗口；不扫描gain、friction或solver参数。只有确认实际问题后才进入新的CUDA控制轨迹。
   原抓取必须离台/双指受载/无台面支撑/低速且保持.2s；完整失败轨迹保留，成功前不接共享策略。

[MuJoCo contactForce](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce)输出contact-frame力/力矩。
[官方滑移说明](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip)区分容量、几何与soft-contact滑移；不直接据此认定本例原因。
本批不重跑此前已绿组、不安装/构建/push/CI、不清理证据或重启周期任务。
## 执行前补充：固定候选与一次反馈试验

- 低位目标精确登记：world2=(.4,0,.045)、world3=(.4,0,.052)、world4=(.4,0,.045)。后侧world1与高位world3各1 seed；三个低位各zero和joint索引3/4的(+.35,-.35)、(-.35,+.35)两seed，共11候选。原模型qpos0的FREE/未选坐标全部保留。每结果独立native FK/contact查询，包含失败、穿透与非目标接触，不进入物理。
- 七个既有轨迹快照实际native比较最大世界接触力差6.789310209587285e-7 N，且球持续相对指垫滑移直至越过下缘。该证据支持优先验证持握反馈，不更改接触定律。
- 只登记world0一次CUDA反馈对照：同原模型、PD、轨迹、h=.001及2200步窗口，原Goal不变；无增益/摩擦扫描。仅两指参考位置使用BoundedSlipClosure，额外闭合最多.016m、速率.02m/s、滑移死区.015m/s、gain2、双侧normal>=.05N才增加，接触丢失保持、禁用清零且遵守原关节范围。阶段close/lift/hold/transport/lower启用，其余禁用。
- 反馈来自上一次已接受公共entry的同相位位置/线角速度/接触点/normal；测量时间为entry_clock[0]-entry_clock[1]，不是当前endpoint。以接触点刚体速度之差求切向速率，逐pad累加normal/max slip。相差一物理步的因果反馈不冒即刻载荷；保存input/output tick和receipt。只加位置参考偏置，原参考速度/PD和原physics参数保持。
- 原0/8结果保留。若该单次对照仍失败，记录明确缺口，不立即改参数重跑。成功也只覆盖world0本窗口，不称八任务或共享网络已完成。