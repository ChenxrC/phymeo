# P6.5 原生材料投影路径验收

范围：opt-in `record_material_path=True` 为真实NumPy/tyx XPBD距离材料提供逐边观察记录。新增第12项有限基础；**不是生产材料/接触共同账本，不是机器人布料接触或双臂递布完成**。旧FreeParticleLedger的材料域拒绝保持不变。

## 产品与合同

首个产品提交`0bf0619`，审阅修复后源码`dc245f74e2a2877c1137f46dabb4638367eb5c2a`，120文件hash `09a3bd9819cb75fdf5bf463adea9397c452cf07365039707e11a5e057c91d624`。入口为ClothProgram、ClothParticipant、TyxClothParticipant和make_cloth_runtime的同名布尔关键字；默认False。只接受bool/np.bool_，记录开关参与绑定签名。

`diagnostics["material_path"]`含schema、columns、frames、predicted_position/velocity、alpha和h。每次sweep一个(edge_count,18)帧：两端操作前位置、操作后位置、前后乘子、实际方向、delta_lambda。所有对外数组独立且只读；记录仅属于临时workspace，无新增持久history。None/空边仍保留零行帧，零denominator用方向零哨兵明确表示没有施加投影，非零denominator的coincident仍取原X方向；哨兵不是计算得到的单位方向。

NumPy在原投影操作旁记录；tyx使用新增traced kernel直接写实际FP32操作，下载后升为FP64存储，没有host重算伪装。关闭开关走原kernel与原outputs；启用不改变状态、乘子、迭代次数或launch次数。直接trace目标须为可写、C连续FP64数组，且不与物理输入共享存储；无效目标在状态修改/设备launch之前拒绝。

## 物理证据及边界

- 解析单边逐项核对18列；未固定三节点/共享双边逐次验证方向、增量乘子、配置修正和连续性，累计配置载荷/h与真实速度重建一致。
- 同一共享双边例，最终lambda乘最终梯度与真实路径的等效冲量差仍大于.05，不能用终值替代路径。
- 无接触旋转硬边三步长h=.1/.05/.025仍符合`Lz1=2/sqrt(1+h*h)`，小于初始2。这是旧离散算法损失，记录没有修复它。
- 软材料例同时核对动能与弹性能：动能增加但应变能释放，总机械能下降；不能把它称为凭空增能。
- 两次diagnostics、跨sweep、跨trial独立；第二子步速度阶段/关闭阶段故障恢复全部状态和显式history，重试与fresh逐位一致。

上述均为指定普通尺度单步例和事务检验，不是所有构型/材料的守恒证明。设备输入FP64→FP32、fixed节点host恢复、额外position load/velocity impulse尚须由后续共同账本独立核对；本阶段不把这些差项隐藏进接触力或容差。

## 工程与后续

修复后CUDA全1827、核心Py3.11/原生pip3.12各1522（305未选）、核心+actual interpret47、C++有界4（13未选）、Ubuntu新核心30、120文件源码外wheel通过；1800次串行观察全成功。准确测试计数、环境、源码与包hash、串行测量写入[工程记录](p65-material-path-engineering.json)；独立审阅见[审阅记录](p65-material-path-review.md)。性能只观测，不优化；未测Warp，不声明tyx与Warp差距。Windows C++仅有界单边短测，既有[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)长期门禁仍开放。

[NUM-005](../legacy-issues/NUM-005-xpbd-material-path.md)只完成“缺少实际路径证据”这一部分，角动量误差及材料接触验收仍未解决。下一切片先认证完整input→prediction→逐边材料→速度重建账本与接触相反输入各一次，再做小布片/单关节、共享多点、线角动量/材料能量外功、低mu和失败事务。之后几何建立/分离/特征切换与保守CCD、弯曲自碰撞、有限驱动单爪、目标IK/history，最后真实A→B→平台和同源Three.js回放；旧完整步长/网格/扰动/重复门槛不删减。
