# 当前场景的面积摩擦：B1/B2 与连续验收

更新：2026-11-18，codex/exp。承接[三批计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。本轮没有改 dev、push、安装或重建 wheel。

## 新增能力

公开 MuJoCoTyxHeterogeneousAreaPressureScene 现在可以显式选择 final_normal_smooth_v1，在同一步中完成当前几何、全部积分点的法向和摩擦共同求解、原生运动积分、状态与力共同提交。没有 CPU mj_step、历史轨迹回放施力或先法向后摩擦 kick。

```text
ModelSpec：有序 body/material + mu/vs + law/精度版本
 → 当前 accepted q/v → body FK → 全部当前有限面与真实 dA
 → 同一设备 owner 的 M/J/bias/passive/applied/D
 → resident pack → 完整 Fn/Ft 非线性共同解 v*
 → 同一原生 q/WXYZ/local-omega/Kahan-clock 积分尾段
 → actual v、物理 M 加速度、A=M+hD 方程、全力与全矩核验
 → 原子提交 q/v/clock + 新摩擦 receipt/设备记录
 → v2 混合观察、检查点、selected reset、恢复与重演
```

法向仍是 Fn=max(0,-k(phi+h*vn))；切向 Ft=-mu*Fn*vt/sqrt(vt²+vs²)。k=dA*g、phi=-p/g，完整积分点保留；微滑属于正则化模型，不声称精确静摩擦或全球接触势能守恒。

## 为什么改了积分接缝

原 I3x 四个历史入口支持“存储 Q 交回原生 solve”，但连续 live 轨迹暴露其边界：质量范数可掩盖小惯量方向的速度误差，重新从近似 Q 解出的 FP32 vnext 与共同求解器 v* 不完全一致，进而影响最终摩擦和力矩。

第一处诊断增加 live 内部逐分量 A^-1 residual 停止条件；双区窗口随后在第 5 步仍被实际力矩门拒绝。按原计划的备选实现，共同速度现在直接进入原生坐标尾段，旧路径仍走原模式。没有放宽共同残差 2e-5、速度 5e-5、力 0.002 或矩 1e-5 的原工程门。

物理 qacc 仍由 M^-1(base+Q) 得到；integration_acc=(v*-v0)/h。两者及 A*integration_acc=base+Q 仍独立检查，阻尼只计一次。原 q、WXYZ、local omega、时钟代码共用且只执行一次，失败候选不翻 accepted bank。

## B1/B2 结果

唯一最终组合 run23：29 passed / 25.80 s / RAW_EXIT=0。包括：

- HOST law 声明与版本：9；实际 CUDA 场景：2。
- CUDA 双侧和生命周期：4；HOST 读回协议：8。
- 准入 HOST：5；真实 CUDA 多法则分桶与末候选恢复故障：1。

实际场景覆盖 mu=.5 / mu=0 / 旧 normal-only / 空接触 D40；三步推进、上离卸载及有效空面。保留完整物理 M、双方点 J、入步外力/阻尼、原生坐标对照和实际发布载荷。

双侧用当前 hinge 与 moving plate 的完整 JA/JB，核双方广义反作用、虚功与世界力矩。native-only 桶保原路径。selected reset 只变所选 world，控制相位、检查点重演的数值字节保真；新 owner 配置身份单独核验。

故障覆盖真实 pack/common-solve 结束后的错误状态传播、最后桶下载、整批提交前、提交后的观察，以及所有 replacement 已完成后最后一个 restore 报错。前提交失败保原物理/力/receipt/geometry；后提交观察失败仅重试观察；restore 失败保原 owner，随后同 capsule 可重试。伪造 Fn/Ft/v/solver status、错 mu/vs/材料身份均拒绝。

新 AreaFrictionEntryReader 在真实混合设备 frame 上解码，并与 accepted receipt 全样本匹配。v1 reader 明确拒绝 v2；旧法向行的新增字段是零填充，不捏造相对滑速。

## B3 当前状态

run16 的双接触区 40 步已通过（6.20 s），使用共同速度的原生尾段。其后仅格式调整已核 AST 等价，未重跑该窗口。

最终其余窗口及 1/4/32-world 整步成本尚在等待串行执行槽；不把旧积分模式下的通过结果倒灌到新源码。查看器已编写、尚未生成和验证。B3 完成后在本节记录同次结果。

## 使用与边界

在原 model 声明上显式设置 area_friction_laws，按 area_pressure_pairs 顺序传入 AreaFrictionPairLaw(identity, mu, slip_speed_m_s)，再用原 Scene 入口。body-local 几何资产仍可复用；旧 asset bundle v1 的 friction=none 不可直接恢复为新物理模型。

范围：CUDA，每桶 1..32 world、1..6 DOF、完整点≤128（三点三角积分实际≤126）、pair≤16、body/geom≤128；冻结入步几何/J，逐片 g>0。新 law 的同世界仍是 area-only，拒绝 BALL/activation、limits/dry、equality/tendon 和剩余 native pairs。旧 native-only 与 normal-only 可以在别的桶同场景存在。

当前几何仍由 HOST 或设备提出拓扑加 HOST 精确复核；不称全设备碰撞、面积 CCD、29 关节摩擦机器人、任意有限边缘或同世界一般混合约束。g=0 且 p>0 仍需显式拒绝，不能加 epsilon 或删面。FFT/Kamino 与通用自动算法选择仍是独立路线。

## 原始证据与失败记录

本轮根目录：`artifacts/area-friction-b1-live-20261118-01`。保存每次实际命令、fresh guard、stdout/stderr、RAW_EXIT、源清单与完整 gzip 输入/输出。run15/22/24 在预检发现其他执行而未启动项目；不是测试失败。

首轮语法/导入检查及格式检查失败保留；修正仅导入/排版。run06 NumPy 标量转换、run07 oracle 对 FMA 的零值等式、run09 测试控制集合问题均保原日志。run11/12 实际速度不一致、run14 实际矩不一致是结构修复的原始证据，不改写为通过。run21 对最终 36 个 Python 文件确认 34 个 raw AST 同、2 个仅顶层导入顺序同绑定，其余改动之前的测试结果按各次源码说明。
