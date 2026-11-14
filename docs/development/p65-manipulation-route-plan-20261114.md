# T1a-4：方向受限姿态与全时序采样（2026-11-14）

起点20d170cf；本批按用户“继续下一批次”执行。原两个作者隔离纯编码，root是唯一项目执行owner。
原88个HOST及world0短窗成功按既有证据复用，不重跑；全部模型、Goal、PD、摩擦、solver保持。

## 两个独立通用模块

1. solve_native_pose_ik：目标TCP位置及body世界SO3方向；解析mj_jac位置/角速度Jacobian、世界Log(Rtarget Rcurrent.T)、0.1m方向权重、DLS与最多8次真实残差回溯。最多100迭代/.15rad或m合成步长；位置5e-4m及角5e-3rad同时合格才收敛。不用180度轴cross假零，不宣称全局收敛。
2. audit_native_joint_path：原scalar joints的完整q knots及明确时间，使用与collector相同的cubic smoothstep。每段ceil(dt/.001)覆盖所有采样时刻，最多4097样本/4s；native FK+collision只查询不推进。所有接触原值保存，显式允许pair/depth与禁止碰撞分开。名称sampled_clear仅表示采样未超界，不是CCD或连续数学保证。

模型/输入完整绑定，FREE和未选q字节不动；相同模型查询前后不变。负控包含端点安全但中途碰撞、TCP可达但方向错误、180度、越限、未选节点/模型不一致与有意接触超深。

[官方Jacobian接口](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-jac)提供位置和旋转Jacobian。
[官方四元数转换](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mju-quat2vel)用于旋转差，不调用CPU物理step。

## 固定候选，执行前登记

- world1 RRRR：复用上一批固定rear候选configuration（原fixed-poses.json world1 rear，joint1=.13196221564344993、joint2=.02025608165829705，其余arm0）。这只定义机器人目标姿态，不向物理状态写值。以原native FK得TCP与完整方向，poseIK从原qpos0求解；抬升目标为原qpos0的TCP/方向（约(.4,0,.1)）。其间joint路径全采样。
- world3 6R：接近/闭合为原qpos0的高位TCP(.4,0,.1)、方向I。抬升候选只将机器人joint1=-.8、joint2=.8，其余arm0；原FK定义完整目标，poseIK从原qpos0求解。如此TCP自然向内收回同时抬升，不强行要求伸直臂在最远x原地上升。物体FREE保持原初态。
- 固定位置/方向负控：world3上一批low(.4,0,.052)要求方向I，zero与上一批两镜像腕seed各一次、原100预算，保留未解；不追加随机seed或调目标。
- world1和world3路线时刻同为0 initial、.2 settle、.7 approach、1.1 close、1.7 lift、2.4 hold。close及以后finger参考.044，之前0；上批相同bounded-slip默认参数仅在真实双指载荷下额外闭合。
- 机器人路径的名义finger位置和最大附加.016m两条固定包络轨迹都采样；只允许各pad与该物体各geom的预期接触：world1参考重叠上限.026m、world3上限.023m，其余接触tol1e-4m。这些参考深度来自指垫中心.07/半厚.009/最大闭合.06与物体半宽.025或.022，分别有意重叠.024/.021m加2mm查询余量；不是实际物理允许穿透或Goal阈值。通用API的可声明深度上限为.05m。完整接触列表保留。这是参考构型检查，实际动态物体运动/参考跟踪偏差不由此保证。
- 每world的IK或任一路径不合格则不执行该world物理；不得把保留物体初态的路径查询说成真实搬运。全部固定候选最多一次；输入/实现错误可定因修复，不扫描控制/物理参数。
- 通过者在同一次CUDA公共batch做2400步上限，不因其中一world成功提前截断另一world；success终止world保持真实状态、不强制再施加动作。若public Env生命周期不能直接支持该隔离，就逐world在唯一串行槽各一次，不改Env以赶本批。

## 结果与停止条件

world1使用原grasp Goal。world3原Goal是place，本批只做接近/持握/抬升短诊断，**不把抬起或夹持称place成功**；完整搬运/释放留下一段连续任务方案。
保存所有入步载荷、端点状态、控制/feedback source及原Goal history，报告新运行实际成功和失败。
无新physics成功也保留通用模块及候选排除证据，不改场景/目标来凑通过。不下载/安装/构建/push/CI，不建周期任务。


实现说明：复用原collector，通过显式plan_builder回调提供诊断参考，旧默认build_plan不变；新case脚本只写motor参考，不写FREE物体状态。为保留每个原Goal完整终止语义，本批两个world按唯一串行槽各最多一次独立CUDA窗口，先各自完成IK与路径采样，拒绝者不运行物理。

上界参考具体采用每个原knot的两finger参考统一增加.016m并裁到原.06m，包括初始/接近节点；这是第二条固定参考路径，不伪称覆盖两个轨迹之间所有闭合偏置。实际feedback仍从零开始并按原启用/双侧载荷规则增长。
