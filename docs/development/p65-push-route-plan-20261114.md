# 第十一批：PushT 完整接近与真实短推

起点638c2292；root唯一执行，原两作者只纯编码/文本。沿原3场景world5/6/7、原质量/solver/摩擦/PD/关节界及Goal；不动原已验几何或轨迹。

## 固定输入和有限选择
消费上批push-original-qualification.json，SHA256 8d9a660cd966c4449d0cbde6cd09b57765af1d40e11186e4f54f1b5d8c3fe3c5。完整72候选及pre/contact IK不重跑。每model完整MJB/hash、原q0与候选绑定另核。
通用rank新模块保留全部attempt，先usable，再名义force与目标XY方向dot降序；其后按目标yaw符号选择名义Mz降序（yaw0取|Mz|升序），最后原index。仅几何启发，不等于真实力或最优轨迹。目标方向取原Target位置减原body origin；目标yaw取原Target yaw减原初始yaw0。

按rank遍历有两个收敛端点的候选。从旧contact参考只求一次新push endpoint：contact center沿内法向20mm，原pointIK100步/.15/5e-4，无新seed。按完整关节C1参考检查：0 initial / .2 settle / .8 precontact / 1.1 contact / 1.8 push / 2.0 hold。
为免整段allowance掩盖接近穿物体，分三段调用原path checker：
- approach 0/.2/.8，q0/q0/pre，无声明额外pair，原1e-4m查询误差。
- entry 0/.3，pre/contact，仅完整pusher/object各pair允许参考深度.0005m，来自现IK位置门，不是新物理容差。
- push 0/.7/.9，contact/push/push，仅这些pair允许参考深度.021m（20mm命令加端点余量）；其他pair保持原1e-4。
所有原模型contact filters保留，完整采样.001s，无失败后删样；所有尝试的状态/路径/拒绝原因保存gzip。每world只选rank中首个三段通过者；若全失败不得投入物理。q0中的对象仍静态，仅作为参考检查，不能称动态对象路径或CCD。

## 执行与观察
几何/排序新HOST和实际preflight通过后，world5/6/7依次W1最多一次2000步CUDA，只有通过者执行；CPU禁止step/step1/step2。控制只原motor，原NativeJointPD，不更改物体状态或施加额外力。旧运行脚本新增默认关闭的record_entry保存开关，记录每步完整公共entry字段（同entry q/v/body rotation、meta、contacts与world/body wrench），无新增采样物理。
保存审计用同entry的body origin+body rotation×原body_ipos计算COM；独立查公共force/couple/bodymoment，所有pusher与support/其他接触保留；核原输入到输出ticks、q/v连续性、真实actuation/PD及原Goal。
短推报告接触区间、真实冲量/COM矩时间积分、对象位移/yaw、原Goal；名义Mz不能直接决定yaw，因为台面摩擦/其他载荷也在。短推未到Goal不算故障或整体通过。没有新观测前不加第二窗/扫参数；必要下一步明确据结果修路径或设计同轨迹长目标推进。

整个批次是通用候选排序和路径/真实传力连接，不是共享语言网络训练。无新deadline、CI、安装、push或清理。旧通过集复用。

协议审查补足：检查网格只声明最大采样间隔.001，真实sample times保留，不冒称恰覆盖每个控制tick。保存执行实际qref/vref供后验绑定。collector的COMPLETE只在保存checkpoint和退出上下文成功后设置，异常统一FAILED_PREFIX，并分别保留物理accepted steps及已emit记录数，禁止失败后重跑同窗。最后2000步入步时刻约1.999s，不把.2s计划hold等同通过原dwell。完整记录所有接触（含非pusher对象/机器人接触），report保真实max penetration/force及阶段时间，不隐藏软接触误差。
