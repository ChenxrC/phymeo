# P4 布料与最小体积软体验收

冻结实现 `664c989f47b0a4eafe58d777368e9986d11038b5`，包源码 SHA256
`bcbdcf67f493966cd41e6f893ba56ca89c0ac49d02693f7581d333e30b282384`，2026-10-27。
P4 本地验收完成；P5 跨域耦合、最终设备回归/独立审阅与工程收口仍继续。

## 已实现与验证

- TetrahedralModel 的参考方向/退化/拓扑、集总质量、FP64 所有权，以及完整 StVK 内力和独立材料域。
- 局部 tetrahedra 状态族复用 ParticleState/StepInputs；候选必须通过形变与固定点检查。固定点保持合法入口位置，不强制把平移后的世界位置改回无应变参考坐标。
- TetrahedralProgram 在子步初态计算内力，再半隐式推进；固定点完整支撑载荷保留。报告积分方程误差，未宣称静态弹性平衡收敛。
- ClothProgram 直接消费已有预测/距离投影算子；新程序不调用旧 XPBDSolver.step。相同力、步长和容差下与旧布料结果相符（FP64 绝对误差 1e-12）。速度使用预测值加配置修正/h，避免无修正时因大坐标舍入丢失速度。
- TetrahedralPointPort 具有重复节点汇总、固定点零运动响应、同点相反冲量和候选验证；沿用通用 SceneStepper 的全帧提交/回滚，未按模型种类修改调度器。

材料 55 项测试包含伸缩/剪切解析力、能量中心差分、刚体不变性、共享节点装配、合力/力矩与有限值溢出。
另有 37 项阶段/端口、4 项实验入口测试先失败再实现；含真实第二子步材料越域、最后子步/close 失败回滚后复用、输入/诊断别名与绑定变化。
与旧布料合计 137 项针对测试通过。Python 3.11 和原生 pip 3.12 核心全套均 **905 passed / 124 deselected**；
这 124 项是显式排除的 tyx 测试，不能写成当前 P4 实际 GPU 验收，下一次完整实际 CUDA 回归在 P5 集成阶段。

## 冻结软块加载

[机器可读证据](p4-softbody.json)：18 节点、24 个正向四面体、9 底层固定点，尺寸 0.1 m，
ρ=200 kg/m³、E=500 Pa、ν=0.3、g=0。顶层均分总力 -0.05 N，h=1/4000 s，共 400 步、0.1 s。

| 检查 | 实测 |
| --- | --- |
| 接受/失败 | 400 / 0 |
| 最后平均压缩 | 0.0016118377 m |
| 全程最小 det(F) | 0.98578198 |
| 全程奇异值范围 | [0.97961329, 1.00903148] |
| 最大积分方程误差 | 6.94e-18 |
| 固定点位置/速度 | 逐步不变 / 始终 0 |
| 第二次完整重复 | 10 个数值数组逐元素相同 |

轨迹 SHA256 `2d4fdb0cc6c1080229e4485d7b59f22c79d77a21e61ea69244285530796cd624`。
本地原始文件位于 `artifacts/p4-softblock-664c989/` 与 `artifacts/p4-softblock-repeat-664c989/`，
包含位置、速度、能量、形变、支撑力、载荷和拓扑；仓库保留小型摘要。复现命令：

```powershell
.venv/Scripts/python.exe examples/softbody_block.py --output artifacts/p4-soft-block-new
```

这个成功样本不证明任意刚度、步长或大形变下稳定。[MAT-001](../legacy-issues/MAT-001-bounded-stvk.md)记录剩余适用域和排查方向。

## 性能与工程

结束核心测试后串行运行两次完整加载，再测 [1/24/192 单元性能](../performance-bottlenecks/measurements/p4-softbody.md)，
保持完整材料/事务门禁；900 个计时 step 全部接受。未做优化，重复 SVD/快照的结构候选记入
[PERF-005](../performance-bottlenecks/PERF-005-solid-validation.md)，没有归因给 tyx。
公开依赖边界检查覆盖 79 个 Python 文件；ruff/docs/compileall 通过。CI 新增短软块 smoke 和 wheel 解析软体 smoke，
远程结果仍须按最终 SHA 核验；P3 的旧 wheel 记录不替代最终整分支打包。
