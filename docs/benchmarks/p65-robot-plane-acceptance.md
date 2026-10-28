# P6.5 机器人材料点与固定平面动态接触验收

新增RobotPlaneContacts与RobotPlaneContactProgram：原机器人积分器通过完整重放自动求法向/摩擦载荷，最终构型Jacobian、作用点材料速度和广义输入共同验收。**本阶段是机器人对固定无限平面的动态接触；机器人—布节点双向接触、有限夹爪和完整递布尚未实现。**

外层规范M0/b0与几何为host FP64，精确账本使用有理数；实际tyx原程序仍产生候选，超原容差拒绝，不用host状态替换。固定M0/b0在q0/v0取值；载荷G=sum J(q1).T*p一次合并到effort输入，末材料速度J(q1)*v1不等于世界弦速度。完整定义见[冻结合同](../superpowers/specs/2026-10-28-p65-robot-plane-design.md)。

## 物理与事务证据

- 单移动关节m=2kg，h=.01，自动支撑冲量.1962N·s，末q/v零。分离、初始穿透、不可响应、迭代不足分别验证。
- 单转动杆独立解析：I=1/3，q0=-.3、v0=1、h=.15，局部点(0,-1,0)，平面y=-cos(.2)。解q1=-.2、v1=2/3，pn=(1/3)*(2/3-1)/sin(-.2)，广义积分-1/9，环境z反力矩积分+1/9。验证最终J而非J0，材料速度而非弦速度，以及冻结动能下降。
- NumPy完整T=2s：两正交移动关节，20N法向压力、切向g9.81；h=.01/.005，mu=.6保持静止、mu=.2按a=-5.81的半隐式解滑移。另有单转动杆h=.02/.01完整停靠窗，逐步核对q、v和冲量。保留原门槛及离散步长差异，不声称精确TOI。
- 两旋转杆、两接触共享祖先；独立重力力矩和反力冲量，两个接触顺序验证合并反作用只计一次。自由布距离材料旁域仍由原XPBD推进；该旁域验证不能称机器人与布已经双向接触。
- 完整history一次推进、后段预测/关闭失败恢复、重试=fresh、返回对象污染隔离、factory只关闭一次；大背景速度能量注入、广义运动丢失、输入映射相消、角/线速度不同单位及inner容差匹配均覆盖。
- actual interpret/CUDA覆盖支撑、转动与布材料旁域；actual Windows C++只覆盖2次重放的支撑短例。检查真实completed_launches；实际后端例不是2秒多步轨迹或C++长期验收。

实际转动例起初solver tol2e-5虽通过世界残差，但角速度误差3.27e-5超独立2e-5验收。原因是正常几何条件放大1/abs(sin(.2))。将inner/outer求解容差同时收紧至2e-6后通过，独立oracle阈值完全不变；不是tyx缺陷。

## 审阅与数值边界

[单次fresh审阅](p65-robot-plane-review.md)发现1 Important：大世界平移吞掉局部接触/COM臂长却错误接受。四个独立RED及普通平移/后候选越界控制已修复；从存储旋转系数精确重建仿射点和祖先臂长，冻结系数前及每个候选都检查。没有第二次审阅。

精确账本还直接检查冻结动能相对p=0实际native自由候选的增量；不能只用末材料功≤0代替能量门禁。冻结能量不等于完整多关节M(q1)与势能守恒。仿射范围检查不是无限精度三角函数/M/b全部误差证明。极端尺度限制进入[NUM-004](../legacy-issues/NUM-004-candidate-representability.md)，不是新tyx实现问题。

## 工程状态

最终产品源码`a4b1ad4da6c7b60744f88fa9182e088fbb6b0a9c`，114文件源码hash `230f2470e71b72a2098afb7a83e10db2170bbc4b05067ce757fa79434e751b14`。详见[工程机器记录](p65-robot-plane-engineering.json)。

| 验证 | 实际结果 |
| --- | --- |
| 完整CUDA | 1725 passed，481.37s |
| 核心Python3.11 | 1441 passed / 284 deselected，345.04s |
| 核心native-pip Python3.12 | 1441 passed / 284 deselected，348.79s |
| 修复后focused＋actual interpret | 55 passed，21.46s，另4控制单独通过后均进入完整suite |
| final actual interpret | 3 passed，1.76s |
| final actual C++有限功能 | 1 passed / 2 deselected，15.69s，2次重放 |
| uv构建＋源码外全新native-pip安装 | 114个Python文件字节一致，安装/求解smoke/pip check通过 |
| ruff / 公共tyx边界 / compileall / uv与pip依赖检查 | 通过 |
| NumPy/interpret/CUDA串行观测 | 各300，共900次全部接受 |

[串行性能报告](../performance-bottlenecks/measurements/p65-robot-plane.md)与PERF-012记录调用结构和未测项，不优化。全量测试曾并行，测试耗时不是性能基线。所有全量测试收集同一最终源码/测试集合，没有以补测伪造一次全量数量。新交付准确head的远程CI仍需推送后核对，旧cfa76bd两项success不能替代新head。

[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)长期门禁仍开放；未换pin、私调卸载、无声回退或重启绕过。无依赖修复不重复压力失败，修复后必须按原规模复测。

## 后续

下一步先冻结机器人—节点双向冲量作用点的时间积分合同，再做同域多点、构型/特征变化和事务。[独立角动量反例](../development/p65-robot-node-candidate.md)说明直接把本阶段最终J1反冲量配给节点会产生O(h)角动量漂移，尽管末点共点、做功和动能都耗散。它否定候选扩展，并非当前固定环境的产品失败。

之后仍须几何/保守CCD、弯曲/自碰撞、有限驱动单爪承重与低mu滑落、目标/IK/history、真实A取布→B接布→平台、同源Three.js暂停/逐帧/力回放，以及完整步长/网格/扰动/重复/事务门槛。P6.5继续，不宣布完整验收。
