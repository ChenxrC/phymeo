# P6.5 节点与固定平面的动态接触验收

本切片新增ParticlePlaneContacts与ParticlePlaneContactProgram：从节点位置、末材料速度和质量自动求平面法向及库仑摩擦冲量，每个候选通过SceneLoadReplay完整重算原机器人/布求解程序。调用者只提供平面和外力，不预先提供正确接触载荷。**这是动态接触的一项有限基础，不是机器人夹爪、网格碰撞或完整递布验收。**

固定边界：已登记粒子节点、静止无限平面、常质量/J=I、同一静/动摩擦系数、位置级一阶非穿透。非接触机器人域随原程序推进，当前没有对机器人施加新平面接触。原后端产生候选；外层载荷、精确账本和残差为host FP64/有理审计，不声称全GPU求解。

## 物理和事务证据

- **自动静置/低摩擦滑动**：m=1、法向压力20N、切向g=9.81、T=2s，h=.01/.005。mu=.6时保持静止；mu=.2时切向a=-5.81，v=a*n*h，q=a*h²*n(n+1)/2。完整时间窗按独立半隐式Euler解检验，保留一阶步长差异。节点没有pin/weld。
- **真实布材料与支撑联立**：两个自由节点、距离柔顺c=.001，下节点受平面支撑，原布XPBD重算内部材料。一步独立解核对乘子/位置/总动量；[2秒连续记录](p65-plane-material-window.json)按独立隐式Euler振子递推验证200/400步。位置误差分别6.50e-13/1.99e-13m，速度误差1.91e-11/8.94e-12m/s，支撑冲量误差2.08e-11/9.16e-12N·s以内，仍满足原1e-10归一化门槛。复现：`python scripts/validate_particle_plane_material.py --output <new-json>`。
- **多个接触及账本**：同节点两平面、多节点、接触顺序、聚合输入与环境反作用只计一次；端点接触功不允许无来源增加。此功不等于完整材料总能量证明。
- **接近和分离**：g0=.01、vn=-1、h=.02时末gap=0、vn=-.5、pn=.5，下一步停住。这是冻结位置级离散方程，不冒称精确TOI或恢复系数模型。初始穿透超容差拒绝。
- **全局事务**：隐藏history、后段子步预测/关闭失败、外层预算不足、返回数据污染、工厂只关闭一次、重试等例覆盖；失败恢复原帧，没有上一候选累积。
- **实际tyx**：interpret/CUDA/C++短例推进静置和滑移8步=.08s、一次材料联合解及超尺度拒绝。报告非零completed_launches，候选未由主机替换；这组短例不是tyx完整2s材料窗口或长期C++验收。

## 一次独立审阅与精度限制

[审阅记录](p65-plane-review.md)：0Critical/2Important/0Minor，两种隐藏舍入的错误接受路径均经RED→GREEN修复。原始/实际存储输入的精确积分减每个实际世界接触冲量的精确和，用L1绝对速度误差审计；另按NumPy FP64或当前tyx FP32的gamma16自由预测预算限制尺度。未知计算精度后端拒绝。

预算是保守支持范围，可能拒绝一些恰好精确相消的输入，不代表材料全过程的数值误差已被证明。材料、实际位移/速度和最终接触原式门禁继续保留。详见[NUM-004](../legacy-issues/NUM-004-candidate-representability.md)及[冻结合同](../superpowers/specs/2026-10-28-p65-plane-replay-design.md)；错误接受已修复，极端尺度扩域留给后续研究，不按tyx缺陷登记。

## 工程状态

修复后产品`c6390f463c4efbc3883b8c64050b17d870c1d6ca`，111文件源码hash`2df1ec407cae54a8b24d6fccac92aaf5a6725526c1b5dacb159f41fda8a10e1c`。详见[工程机器记录](p65-plane-engineering.json)。修复前结果不计入本轮最终验收。

| 验证 | 实际结果 |
| --- | --- |
| 全量CUDA | 1664 passed，615.59s |
| 核心Python3.11 | 1383 passed / 281 deselected，486.95s |
| native-pip Python3.12核心 | 1383 passed / 281 deselected，494.83s |
| 新核心＋actual interpret | 59 passed，9.82s |
| 相关新旧回归 | 184 passed，11.30s |
| actual C++短功能 | 4 passed，117.06s |
| 全量收集后补多节点/顺序回归 | 三个环境各2 passed；无产品代码变化 |
| uv构建＋全新源码外native-pip wheel | 111文件字节一致，安装/求解smoke/pip check通过 |
| ruff / public tyx边界 / compileall / uv与pip check | 通过 |
| NumPy/interpret/CUDA串行观测 | 各300，共900次全部接受 |

[性能报告](../performance-bottlenecks/measurements/p65-plane.md)保留原始样本、计数和未测范围，热点进入PERF-011，本轮不优化。全量测试可能并行，测试耗时不作为性能基线。补验的2项测试在原全量收集后加入，不能伪造为一次1666项全量结果；远程CI将收集完整新测试集。

交付推送后的准确远程SHA/CI另行核对；旧1ce5c03已success不能替代本次head。

Windows C++长期运行的[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)仍未修复。此轮只做有限短测，不无理由重跑已知失败压力；未改pin、私调卸载、静默回退或用重启掩盖。后续依赖修复后，原长期规模必须重测。

## 后续推进

下一切片应补机器人单关节动态接触，再同域多点/交叉响应与构型变化；显式冻结M/J/作用点速度和广义输入映射，不能把本节点J=I直接套用旋转机器人。其后接几何特征/厚度包络和保守CCD、弯曲/自碰撞、有限驱动单爪承重及低mu滑落、控制目标/IK/history，最后真实双臂递布、完整步长/网格/扰动/重复/失败门槛与同源Three.js暂停/逐帧/力回放。P6.5继续，不能以本页结果宣布完成。
