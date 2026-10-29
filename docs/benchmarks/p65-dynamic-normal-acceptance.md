# P6.5 自动法向接触：生产实现与验收进度

状态：生产实现、本阶段唯一独立审阅0Critical/0Important/0Minor、最终本地工程验收与正式串行观测均完成；已推送29165ca，准确CI因Windows3.11uv触及15分钟作业上限而取消；数值步骤成功、示例被中断，见[CI时限修复](p65-dynamic-normal-ci-timeout.md)。时限修复aa8ec4282e4baefd23ff8a396ae7d92bac011ca0已推送且远程SHA一致，准确CI40802849502与Documentation40802849436均completed success，本阶段交付闭环。完整H1/P6.5仍未完成。产品提交4ff19ea7975145fe9d44450164b906f01fe0527d，131个源码文件hash `ac777319d0b8fbc03adab92286656a5572df05bca403e913912500908e991060`。

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

最终同一冻结版本：两套CPU完整各1865通过/351未选（416.07/399.69s）；完整actual CUDA2216通过（9148.75s）；Ubuntu新核心112通过（103.42s）；有界CPP解析1通过（28.97s）。CUDA低摩擦长窗最终通过，具体耗时原因未定位，未据此认定依赖缺陷。Ubuntu沿用已验证依赖，不是干净安装；CPP短例不替代TYX-003长期门禁。

最终131文件源码外wheel与新native-pip安装smoke通过，ruff/docs/公开tyx边界/compile/uvpip检查及未改变的锁导出核对通过。源码、测试、脚本摘要与runner哈希在全流程后复核一致；历史pre-cap的局部结果另列，不冒称新版全量。

随后NumPy→interpret→CUDA严格串行，各300计时/306含warmup，共900计时/918调用成功；每模式free/chain各150。精确版本、执行计数、原始记录见[工程记录](p65-dynamic-normal-engineering.json)及[PERF-019](../performance-bottlenecks/PERF-019-dynamic-normal-contact.md)。容差/迭代不同，不构成后端加速比；完整套件耗时不能当正式性能。

## 未完成范围与下一步

[唯一审阅](p65-dynamic-normal-review.md)、所有本地测试和正式计量已结束，不再重复旧pipeline、全量或审阅。本次专用分支远程SHA与准确aa8ec42的两项CI已核对成功；下一步继续摩擦框架的原生候选和生产合同，不重复本阶段验收。

本能力不是摩擦、有限三角/边特征切换、旋转FK保守CCD或真实夹持。之后仍须这些框架能力、弯曲/自碰撞、单爪承重与低摩擦滑落、目标/IK/history、双臂交接及同源Three.js。NUM-005积分器角动量损失未被修复；TYX-003长期CPP和TYX-004严格解释器cos仍开放。保持[冻结合同](../superpowers/specs/2026-10-29-p65-dynamic-normal-design.md)与[实施计划](../superpowers/plans/2026-10-29-p65-dynamic-normal.md)的门槛，不扩P7/P8。
