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