# 三维面积压力端口：本批结果与下一步

2026-11-16，codex/exp。测试源码 **777b7f12d1b614fd5facc4c364e10f68359ccfbc**。状态 **HOST_PORT_ACCEPTED**：新增两个完整测试文件首次 **59 passed in 1.75s**，RAW_EXIT=0。没有数值失败或模型/阈值修正。

## 现在有了什么

新增[三维表面与求积](../../src/phymeo/geometry/area_pressure.py)、[行系数与双侧力映射](../../src/phymeo/solvers/area_pressure_rows.py)。入口接收已裁好的三角压力面，保存双方 body/material、compliant cell/rigid face，以及 world/step/time/configuration/sample 身份。

- 三个二次精确重心点保存真实 dA，面积不归一化。不同面允许不同法向。
- 线性压力的合力、压力中心对应的一阶力矩同时保留；总力为零的纯力偶也能表示。
- 同一完整自由度顺序中的双方 Jacobian 全部参与 Jrow=n^T(JA−JB) 与广义力映射，兼容双自由刚体、共享平移、关节型坐标及固定支撑。
- 局部系数 k=dA*g、phi=-p/g、f0=dA*p 单位明确；phi 是冻结几何代理间隙，不是碰撞距离。
- g=grad(pA)·n 必须严格正；不支持的梯度、相位不匹配、重复身份、非法数值、容量超限整体拒绝。数组为 bytes-backed 不可写副本。

函数本身没有状态更新或冲量发布。传入的 normal_forces 是调用者提供的力，不能据此称实际共同求解完成。

## 独立验证覆盖

测试作者只读[冻结合同](../superpowers/plans/2026-11-16-area-pressure-3d-port.md)，未读新实现；源码作者未读新测试。两个文件为[表面测试](../../tests/test_area_pressure_surface.py)、[行与力映射测试](../../tests/test_area_pressure_rows.py)。

| 固定对照 | 首次结果 |
| --- | --- |
| 面积1m2三角，p=(1,2,4)Pa | Fz=7/3N，Mx=11/12Nm，My=-3/2Nm，与独立 simplex 积分相同 |
| 三角细分、循环节点、刚体旋转/平移 | 合力与原点力矩协变，未使用投影面积 |
| 几何尺寸x2、压力不变 | 面积/力x4、原点力矩x8 |
| 相反法向的两个面 | 合力0，Mz=-6Nm；未退化为单个合力点 |
| 12DOF 双体 / 4DOF 共享与关节型坐标 | 全列 Jacobian、实际速度叉积与虚功一致；双方 COM/原点力矩一致 |
| k/phi、零压力、分离扰动 | 正压力减少方向正确，未用梯度模长替代法向梯度 |
| 空接触、不可变副本、相位/ID/容量/非法值 | 全部按合同通过，无静默广播/排序/截断 |

普通尺度代数验证使用预定 rtol=atol=1e-12；这不是 CUDA 或完整物理步精度结论。未重跑 V00–V06、旧产品组或外部 Newton 比较。

## 执行和保存

只使用原 .venv-training-cuda313：Python3.13.3、NumPy2.4.6、pytest9.1.1；两个 product module 的 __file__ 均绑定父 src。root 每条命令 fresh guard、单线程、noautoload/nousersite/nobytecode，唯一串行执行。

9条实际命令：环境0 → 首Ruff1（三个I001）→ 首format1（四文件）→ import修正0 → format0 → qualifiedAST0 → 最终Ruff0 → 最终format0 → 首完整HOST0。所有原始错误日志保留。三文件 raw AST 完全同；area_pressure_rows.py 仅 import 次序改变，四文件非 import AST 与 import 集合均同。不称所有 raw AST 相同。

[完整证据目录](../../artifacts/area-pressure-3d-port-20261116-01)包含两作者交接、原始四文件、每条 command/guard/raw exit/log/source before-after 与最终审计。
[acceptance-audit.json](../../artifacts/area-pressure-3d-port-20261116-01/acceptance-audit.json)核9条退出、13个最终 source/config 绑定及各阶段源变更范围，审计时13项全部匹配；此后仅报告、计划与续作状态文档更新，四个source/test字节保持。父三份无关 untracked SHA 保留。审计SHA256：7392d3a008b69cf2c3d02ec16f345f1bbeb23f8ebb3b3e5db186670ae0d09f60。结束 fresh matching project execution=0，父与两作者均释放。无安装、native/GPU执行、视频或CI。

## 与整个目标的关系

R4 采用了 **一软一硬压力场路线的有限准入**；本批完成 I0 的面积数据/法向弹性系数/HOST力映射。完整场景声明、压力场资产、真实碰撞前端、共同动力学和设备生命周期尚未接上。V03外部双软球面积/收敛 FAIL 原样保留，FFT/Fourier 没有新增运行结果。旧 dev 未改变，T-smoke 共享训练主线仍独立。

下一批 **I1：真实三维 tetra–triangle 裁剪**。固定解析横截面/斜截面/边界空集/相邻 cell 去重与连续压力/非共面刚面，生成当前端口；不能再依赖手工给好的接触面。之后 I2 接同一步共同求解。

[接线裁定](area-pressure-3d-integration-decision-20261116.md)已明确：现有 MuJoCo-TYX 求解使用 physical M，而实际速度更新还用 M+h*Cjoint。新压力律必须与真实更新速度对齐，不能把 dA*g 直接填入 solref/solimp，也不能同时保留 native pair 力。首个零耗散法向行候选已列公式及条件，尚未运行或作为真实物理结果接受。
