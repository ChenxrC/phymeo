# P6.5 机器人—距离材料接触：验收进行中

本阶段实现`InitialRobotMaterialContactProgram`和显式`MaterialContactBudgets`。生产初稿9f504bb、唯一review修复0657c98；124源码hash b56bafcac102b696ebf1a180bb2946bd7ab2a8d103b9a3a4b6bffc5a85645dce。未完成所有设备/计量/交付门禁，不算完整P6.5或新增已交付能力。

## 行为与边界

实际距离材料、节点与机器人经完整native replay联合求解；初态J0/M0/b0冻结，p和-J0.Tp各经原始输入一次。MaterialPathLedger审计真实路径，FrozenRobotLedger审核机器人；几何、法向、摩擦、接触功与反作用矩同时受原tol约束。能量用节点K/U/重力与冻结M0机器人K，扣除原非接触输入端点功；材料角动量缺陷按各域L1累计。两个显式预算每子步固定，不自动放宽，不以舍入展示值作为门禁。

支持仅初态闭合、全动态距离材料与机器人局部无限平面。free响应仅预条件，完整replay保留共享域反应。既有free程序仍拒绝材料，旧能量规则不被替换。不是自动新接触、CCD、一般机器人世界能量守恒或真实夹持。

## 本轮证据

- 压缩边三h独立解析p/速度/位置/Py与K+U；XY旋转硬边固定角动量预算下粗步拒绝/细步通过；不相容硬边增能拒绝。
- NumPy持续T2支撑、双点mu1静止与mu.1滑动，两步长、独立离散递推与连续cos参考；单转子整步Lz/允许正末间隙、非对称三角、同节点双平面、共享域顺序与原始功。
- 真实后子步物理拒绝、trace损坏、close失败完整state/history恢复和retry=fresh；owned配置/报告与精确舍入边界。
- 最终两CPU各1612通过/338设备未选，Ubuntu47通过62.27s，124文件精确匹配的源码外wheel与新native-pip安装smoke通过。
- 唯一fresh review0Critical1Important0Minor，inner/outer迭代合同3RED后修复，相关86通过22.28s。详见[审阅](p65-material-contact-review.md)。

版本区分：初稿actualinterpret8通过649.23s和有界C++1通过/7未选27.19s；review修复新增inner配置已实际interpret通过。旧CUDA定向运行被最终版全量替代而中止，不报其通过数。修复前interpret曾2pass3fail（cos underflow）与最初共享接触31pass2fail均保留在[工程记录](p65-material-contact-engineering.json)，不删除失败历史。

## 剩余验收

完整最终CUDA及通过后NumPy→interpret→CUDA串行各3×100正在pipeline执行。性能只观察，不能用并发测试耗时当加速比；此前1sample只是脚本smoke。须完成结果核对、性能记录、提交推送、精确SHA/CI闭环。TYX-003长期C++门禁不以短测替代；TYX-004解释器cos异常合同仍待依赖维护者确认。之后才推进几何/CCD与真实夹持，双臂与Three.js仍未完成。
