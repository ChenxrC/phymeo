# P6.5 材料原式账本验收

本阶段新增第13项有限基础 `MaterialPathLedger`：独立认证已有 NumPy/tyx 距离材料候选的离散方程、一次输入与完整实际路径；不调用求解器生成替代候选，也不修改原积分器、接触许可或 StepSettings 容差。

## 能认证什么

- 原始 model/constraints/state/input 独立快照；请求支持精确整数与 Fraction，不能先经 float 丢失低位。
- 原输入→预测→逐边增量/方向/乘子→最终位置/速度重建与本构。逐边严格连续，累计绝对预算不按边数或迭代数摊薄；固定 compliant 边同样核对标量方程，零分母保持声明的零方向哨兵。
- 实际设备 FP32 提升、host 输入转换与固定点位置恢复分别认证；隐藏位置/速度载荷、重复输入、跨路径篡改拒绝。所有报告为独立只读快照，包括空域 `(0,3)` 向量。
- 真实 SceneLoadReplay 同初态不同请求；后段路径破坏或关闭失败时，state/history 全恢复，重试等于 fresh。小三角和真实单关节机器人测试给定相反输入各一次，用于输入账本，尚不是接触冲量求解。

## 物理范围

返回 `scope=distance-material-equations-v1`。材料路径冲量、支持反作用、K/U/重力、端点位移功、线角动量及误差分项均由实际存储量独立核算。报告不等于物理守恒或 SceneStepReport 接受。

孤立旋转硬边三步长仍满足原离散式 `Lz1=2/sqrt(1+h²)<2`。共享软边可以 K 增而 U 降；硬边初态不相容也可能产生正能量变化，账本明确报告，不能将方程通过当成能量可接受。原 [NUM-005](../legacy-issues/NUM-005-xpbd-material-path.md) 仍开放。

## 工程证据与下一步

唯一[独立审阅](p65-material-ledger-review.md)的1 Important与1 Minor经12个RED修复；修复产品为 `07b23d64450723f2742962962ce268968e30810c`，122文件源码hash `80831f04b59587c1805b141c26a73a659f4b5b0a399f0f5f954f4c7371de683d`。准确平台结果、失败环境尝试、包与串行观测见[工程记录](p65-material-ledger-engineering.json)。Windows默认pytest临时目录曾触发setup权限错误，改用每套件独立且首次不存在的basetemp重跑；不改产品或放宽测试，失败不计作通过。

下一阶段实现新的同域材料/机器人接触外层。先冻结 M0/b0/J0、双方原输入各一次、末几何与材料实际速度；为能量外功、材料离散角动量及步长细化预定义门槛。当前 `FreeParticleLedger` 门禁保留，不能把材料黑盒放行，也不能复用被否定的最终J1节点反冲量路线。

随后才是几何建立/分离/特征切换与保守CCD、弯曲自碰撞、有限驱动单爪承重和低μ滑落、目标/IK/history、真实A取布→B接布→平台及同源Three.js。C++长期 [TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md) 仍待依赖修复，当前有界测试不代替它。没有新tyx实现缺陷证据，未测Warp。
