# T1a 姿态审计与 T1b 滑移反馈（2026-11-14）

本批新增三项可复用控制能力：有限 seed 的指垫/物体姿态审计、公共入步接触点滑速测量、受限双指闭合反馈。
**world0 的一次 CUDA 抓球对照在原 Goal 下成功；其余七个任务本批没有新增操作成功。**
物理引擎、原模型、摩擦、质量、solver、PD 增益和成功阈值未改；没有学习网络、语义模型或 optimizer。

## 产品与验证

- 作者 pose 9e93bbaf → 父5de77540；fixture/排版修正5d4bf49a。
- 作者 grip 2bdcaa6e → 父7be81b5e；作者 contact 2de78f26 → 父d3f182f0。
- 父集成产品 **02f44795**：公共 entry 测量接入原 CUDA collector 的可选 --slip-guard；默认仍无反馈。
- 新HOST：pose 7、grip 34、contact 47，共88项通过。各自独立几何/手算、负控、不可变输入及事务边界均保留。
- 新固定查询：七份原保存状态的独立native mj_forward接触对照；11个姿态候选的IK/FK/碰撞查询。均不推进原生物理。
- 唯一新物理 run24：2038个CUDA步，h=.001，原登记上限2200步，原成功终止规则触发。wall39.7105s含HOST控制/观察/保存，非solver吞吐比较。
- run26同次保存审计通过：2038个完整样本的前一entry绑定、接触点速度、normal求和、max slip、阈值/速率/闭合量和实际PD目标逐步核对。原XML与基线逐字节相同，trace/state hash完整，未重跑物理。

## 既有滑落的定位

固定消费原world0的1100、1300、1500、1700、1900、2000、2100七步。
使用trace[k-1]端点q/v与trace[k]实际ctrl/qfrc/xfrc还原入步；不将同一条记录中的新端点当作接触求解输入。
独立CPU MuJoCo3.3.7 mj_forward的接触世界力与TYX最大差 **6.789310209587285e-7 N**，native约束KKT剩余为舍入量级。

两者均出现切向滑移。球心相对指垫下缘高度从19.67mm降到0.507mm，再低于下缘；最后失去指接触。
这些快照未显示TYX接触力求解错误，不外推为整个CPU/CUDA轨迹逐位相等。
MuJoCo官方区分软接触滑移、几何与摩擦容量：[防滑说明](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip)。
本批没有启用noslip或改接触参数；力坐标转换按[contactForce文档](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce)。

## 固定姿态结果

原物体与目标完全保留，当前只检查开指姿态。全部11候选都保存，未按输出追加seed或调目标。

| 候选 | 点IK结果 | 几何结果 / 资格 |
| --- | --- | --- |
| world1 RRRR 后侧TCP(.375,0,.045)，zero | 误差.376mm，收敛 | 双pad离台3.732mm；独立native端点无接触；局部x/z投影覆盖物体，开指y方向尚有36mm间隔。只是可继续设计闭合/搬运路径的候选。 |
| world3 6R 原高度TCP(.4,0,.1)，zero | 误差约2e-13m，收敛 | 双pad离台60mm；端点无接触，开指y间隔39mm；可继续验证高位侧夹，不是实际夹持。 |
| world2 5R 低位，zero/两镜像 | 全未收敛，13.54/15.14/15.14mm | 有限集未找到；不证明全局不可达。 |
| world3 6R 低位，zero | 未收敛，12.47mm | wrist与物体穿透9.399mm，不能用于操作。 |
| world3 6R 低位，两镜像 | 均收敛，误差约.000466mm | 一个pad穿台34.586mm，拒绝进入物理。 |
| world4 7R 低位，zero | 未收敛，15.86mm | 有限单候选失败。 |
| world4 7R 低位，两镜像 | 均收敛，误差.493mm | 一个pad穿台36.128mm，拒绝进入物理。 |

因此镜像seed确能打破部分点IK分支，但点位置到达不能代替夹爪方向、完整碰撞或路径可行性。
产品audit返回每个BOX pad的8角点/有向PLANE距离、sphere/capsule/box在pad轴上的解析投影。
三轴投影不是完整SAT；产品不挑候选、不写q，不声称全局无碰撞。

## 同一原模型下的反馈对照

只通过真实finger motors更改参考位置，保留原轨迹和PD。每个pad的法向载荷求和、接触点切向速率取最大值。
公共body-origin线角速度还原接触点速度；geom正反顺序均测，entry_body_mask拒绝padding。
测量只用上一已接受entry，time=clock[0]-clock[1]，当前动作有一物理步因果延迟。
调用方核input/output tick；helper不伪造workspace或模型身份验证。

双侧normal>=.05N且close/lift/hold等阶段开启时，闭合增量率=min(.02,2*max(0,slip-.015))m/s。
额外闭合至多.016m，最终参考在原0..06m关节范围内；失联保持、禁用清零。只加位置偏置，原参考速度保留。
完整登记见[本批计划](p65-manipulation-pose-plan-20261114.md)。

| 同一world0 | 旧无反馈基线 | 新一次反馈 |
| --- | --- | --- |
| 窗口 | 2200步/2.2s，失败 | 第2038步原Goal终止成功 |
| 抬起后行为 | 持续滑落并丢失接触 | 原连续.2s低速保持通过 |
| 峰值球底离台 | 54.513mm | 56.803mm |
| 成功entry | 无 | t=2.037000097s，tick2037 |
| 成功时球底离台 | 无 | 48.383mm |
| 成功时双指法向力 | 无 | 各2.374806N，台面支撑0 |
| 成功时速度 | 无 | 17.438mm/s，角速度约7.54e-7rad/s |
| 额外闭合 | 0 | 最大各8.949mm，低于16mm限额 |

第一次非零修正发生在第1160步。结果说明公共接触观察→控制参考→真实motor→CUDA状态→原Goal这条闭环已在一个案例贯通。
仍存在缓慢下滑；**只通过原.2s保持窗口，不是长期无滑移或全部异构操作通过**。
最终Env checkpoint仅保存Env/物理，不包含此诊断控制器的可恢复状态；完整控制过程在trace，不能据此声明任意断点继续控制已实现。
这些控制器目前是HOST诊断，仍有设备读取；不能称GPU策略训练或性能胜出。

## 首失败与证据

- 03 pose格式检查1，04格式修复；07 pose首HOST为6pass/1fail。
- 07原因：fixture直接修改编译后geom_pos，单geom body的编译优化复用惯性frame，FK没有按该不一致编辑移动几何。按XML重新编译两份独立fixture后12完整7pass。产品数学/原控制场景不变。
- 15 collector import排序在新module尚未导入时失败；18导入后精确检查通过。16排版失败→17格式修复。
- 19 contact格式检查1→20格式修复→21/22绿。其余新HOST均首次通过。
- 25保存审计在Windows默认GBK读取UTF8 control-plan时失败，尚未进入逐步审计；保留v1，v2仅指定UTF8并补末态输出，26通过。不是物理失败，没有重新执行CUDA。

根目录：artifacts/mujoco-tyx-manipulation-pose-20261114-01。
- 01-fixed-native-forward与fixed-native-forward.json：七个既有快照。
- 12-pose-host-final、13-grip-host、23-contact-host：88个新HOST。
- 14-fixed-poses与fixed-poses.json：全部11候选。
- 24-cuda-slip运行记录与同根cuda-slip-01：新原始轨迹、模型、控制计划、205个保存帧与final.checkpoint。
- 26-saved-feedback-audit-v2及cuda-slip-01/saved-feedback-audit.json：完整因果反馈审计。
- 新trace SHA 0d6d0ff2b4fbcdc2eb01377cb08eab3a02f85b9a037c21cf1c5c1ddd48262802。
- 新states SHA b9038400a87c5fd2012758452138d3a4e1ae69f45989dd365d2045b3653a6ec3。

原八世界0/8基线及视频保留，没有重跑、覆写或把新单世界结果倒灌到旧八世界。
本批不安装/build/push/CI、不更改依赖、不清理任何证据。唯一串行owner为root；两作者只纯编码。

## 下一批

1. 通用接触姿态与路径：以完整夹爪方向、pad/物体/台面几何和原关节范围约束有界IK候选；保留本批“TCP到位但pad穿台”作为负控。优先验证world1后侧和world3高位的接近/闭合/抬升，再搬运/释放，先登记全路径与窗口。
2. 独立PushT路线：由真实T边界与pusher半径选择可达接触点，保存接触力矩、yaw及原位姿/IoU；不把物体中心当接触点或改原目标。
3. 首抓/放/推通过后再扩八构型与holdout，随后接真实中英文语义编码和同一网络优化。单次world0成功不关闭T1c/L1/L2。
