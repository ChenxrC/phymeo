# P6.5 自动法向接触：生产实现与验收进度

状态：生产实现已提交，本阶段唯一独立审阅0Critical/0Important/0Minor，最终串行工程验收已启动，正在执行，尚未交付整个阶段，更未完成H1/P6.5。产品提交4ff19ea7975145fe9d44450164b906f01fe0527d，131个源码文件hash `ac777319d0b8fbc03adab92286656a5572df05bca403e913912500908e991060`。

## 可复用支持域

`DynamicRobotNormalContactProgram` 自动为所选全动态距离材料节点与配置的无限平面生成接触行，支持全prismatic机器人、固定朝向、显式碰撞侧和非负半径，摩擦/恢复均为0。案例只配置并调用公共API，不按对象名、坐标或时间写物理特判。

held阶段从原状态/输入完整原生重放，节点p与机器人反作用各输入一次；末态闭合集合通过真实参与者施加独立terminal冲量eta。两个阶段分别审计，p+eta仅为总冲量诊断。MaterialPathLedger/FrozenRobotLedger只验末冲量前状态，TerminalImpulseLedger审计真实速度跳跃，不覆盖原生状态。生命周期为显式enter/persist/exit/inactive，分离后零冲量，base history每子步只提交一次。

当前及初始作用点分别记录，材料角动量预算区分x0乘held与x1乘eta。能量使用原非接触输入端点功、最终K/U/重力和冻结M0；不以有效接触输入混掉额外能量来源。所有机器人关节（含旁支）必须是prismatic，features仅声明`dynamic_robot_normal_split`。

## 独立物理与事务证据

- 自由探头和两节点材料链两个非递布场景由独立分数解析解验收；链另一端的运动检验同域材料传递，不仅检查接触节点。
- T=2秒有限驱动支撑在两个步长运行；自由碰撞T=1秒三个步长；材料链T=.5秒三个步长按独立隐式Euler递推及连续弹簧解比较。粗步厘米级偏差界仅刻画该离散法，不是最终递布精度验收。
- 多节点/共享祖先、三角材料、同节点多平面、顺序与适用变换、半径和非法/退化输入有独立控制。vx=1末矩1/12 Nms负控制能区分错误x0和正确x1。
- 后段子步history/trace/末冲量/close失败整帧回滚，retry等于fresh。已有best candidate之后的真实执行失败仍拒绝，半构造失败清理已创建参与者。
- 实际interpret/CUDA检查真实模式、FP32计算、原生launch和末阶段执行。开放零eta路径的FP64 nextafter篡改也拒绝。

精确测试数量与版本边界见[工程记录](p65-dynamic-normal-engineering.json)。新CUDA7通过1055.23秒是验收耗时，不能用作正式性能结果。基准脚本已分别完成NumPy/actual interpret/CUDA最小smoke（每模式2计时调用、6warmup），900次正式观测尚未执行。

## 未完成范围与下一步

[唯一审阅](p65-dynamic-normal-review.md)已结束且探针全部退出；随后进行完整两个CPU/CUDA套件、Ubuntu、有界CPP、源码外wheel、串行正式性能与准确新提交CI。旧2f50c1f生命周期兼容修复的CI已成功，不能借此声称当前4ff19ea已完成远程验收。

本能力不是摩擦、有限三角/边特征切换、旋转FK保守CCD或真实夹持。之后仍须这些框架能力、弯曲/自碰撞、单爪承重与低摩擦滑落、目标/IK/history、双臂交接及同源Three.js。NUM-005积分器角动量损失未被修复；TYX-003长期CPP和TYX-004严格解释器cos仍开放。保持[冻结合同](../superpowers/specs/2026-10-29-p65-dynamic-normal-design.md)与[实施计划](../superpowers/plans/2026-10-29-p65-dynamic-normal.md)的门槛，不扩P7/P8。
