# 面积压力接触：当前已连接架构（2026-11-18）

本页以 codex/exp 本轮源码及保存结果为准。它描述面积压力分支；全引擎其他模块沿既有全景与各自验收，不用本页推导 Newton 全功能对齐。dev 未改，未 push。

[可离线打开的架构图](assets/area-pressure-architecture-20261118.html) · [本轮执行计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md) · [摩擦生产接缝图](area-pressure-friction-integration-map-20261118.md)

## 已接通的主链

```text
模型源 + 明确的 body-local 压力资产 / 有限刚面
  → 每个世界的已接受 q/v 与当前机体 FK
  → 完整当前几何准备一次 + 有界批量候选
  → 有限接触面，真实 dA / p / gradient / 双方材料身份
  → k=dA·gradient、phi=-p/gradient、双方完整点 Jacobian
  → MuJoCo-TYX 共享质量和广义自由度共同法向求解
  → 所有候选和载荷验证后统一提交物理银行
  → 持久设备记录 + 独立原子观察发布
  → 任务 Env、面积角色判断、统一多 body 力/力矩读取
```

当前面积律是显式选择的、无摩擦、冻结入步面与 Jacobian 的弹性法向后向 Euler 模型。它与 MuJoCo 原生 solref 行不同。原生约束仅在已声明准入组合内共解；有阻尼与剩余原生约束的混合仍有限制。

几何默认 HOST。可选混合路径让设备提出裁剪拓扑，再由 HOST 完整精确复核；没有宣称全 GPU 精确裁剪或 CCD。资产来源与单位显式绑定，不从 URDF 自动猜压力材料。

## 本轮真正新增的连接

| 部分 | 已验证行为 | 证据 |
| --- | --- | --- |
| 候选和准备 | 全批请求、严格证空、一次当前网格完整准备；晚失败不部分发布 | [I3f](area-pressure-batched-candidates-report-20261118.md)、[I3g](area-pressure-single-preparation-report-20261118.md) |
| 固定 GPU 规模 | 5/20/32 世界的 HOST/mixed 完整同输出；固定案例后 24 步整步成本减少约 9.7–11.3% | [I3h](area-pressure-batch-scale-report-20261118.md) |
| 应用入口 | 自动当前几何接入现任务 Env，保持 reset/restore/错误恢复与资产身份；没有新 PPO 成功结论 | [I3i](area-pressure-success-reader-report-20261118.md)、[I3j](area-pressure-manipulation-env-report-20261118.md) |
| 载荷观察 | native + area 双方合力/力偶，参考点变换；同帧多个 body 一次读回 | [I3k](area-pressure-body-wrench-report-20261118.md)、[I3m](area-pressure-wrench-batch-report-20261118.md) |
| 连续动力学 | 4 世界、3 模型、450 批次步完成入层、承重、加压与卸载；另有三可动物体共享质量的双区释放 | [I3l 与媒体](area-pressure-heterogeneous-support-report-20261118.md)、[I3o](area-pressure-three-body-release-report-20261118.md) |
| 资产文件 | 明确版本/单位/模型源的完整压力资产保存重载，重新验几何，加载前后相同输出 | [I3n](area-pressure-assets-report-20261118.md) |
| 摩擦选型 | 固定 24 行三律比较；最终法向平滑摩擦的共享 DOF HOST 与 CUDA 独立参考 | [I3p](area-pressure-friction-model-report-20261118.md)、[I3q](area-pressure-coupled-friction-report-20261118.md)、[I3r](area-pressure-tyx-coupled-friction-report-20261118.md) |

规模数字是固定有限模型的证据，不等于支持任意 32 台复杂机器人，也不是 Newton 同机同场景基准。新增[三种 FREE 六自由度初态的150步窗口](area-pressure-free-body-support-report-20261118.md)，已验证真实转动及独立原生 M/J/积分；无摩擦支撑仍不等于自由抓持。

## 摩擦现在处于哪里

独立参考已经用同一个最终速度求 Fn 与 Ft，保留完整非对称 Newton 导数、真实回溯和双方反力矩。CUDA 验证覆盖 32 世界、6 DOF、8 点边界及失败生命周期。其 FP32 误差门与 HOST 1e-10 严格门分开报告。

它还没有接入上面的生产主链。必须补齐：真实场景全部积分点容量、同阶段完整 M/J/载荷、显式材料 law、现 native 坐标积分一致性、新面积观察版本、资产/检查点绑定和原生约束组合。现 v1 明确 friction=none，不能直接覆写其总力字段。

当前生产每个接触三角形有三个积分点；复杂面可能超过独立参考的八点上限。不能删点、缩面或把归一权重当 dA 绕过限制。现 passive 已含关节阻尼，也不能向新求解器再次重复扣阻尼。

## 后续次序及停止条件

1. FREE当前有限面、偏心矩与原生积分已由I3s补齐；下一步验证新wheel的源码外混合场景与生命周期。
2. 冻结首版摩擦生产范围与容量。复用真实动力学/银行，明确暂不支持的同世界原生约束；不另造每步 HOST q 引擎。
3. 在单路径完成自动面 → 同步 Fn/Ft → 实际积分 → 新载荷观察的多案例闭环，覆盖滑动、卸载、偏心转动、双方动体及晚失败。
4. 再扩大 DOF/积分点与混合约束，按相同输入、输出及精度比较完整步成本；有收益才扩大设备几何。
5. 最后连接所需策略特征与训练，随后考虑变形体压力散射。FFT 非局部材料与通用自动选型仍为独立研究。

同一固定失败先诊断一次对应层，再作明确修复；若暴露模型不适用就缩小范围或改变公开合同，不循环调质量/步长/门槛。几何已过而动力学失败时，不重做相同几何组。没有功能或整步成本收益，不扩大近似重复案例。
