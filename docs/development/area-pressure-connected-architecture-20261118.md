# 面积压力接触：当前已连接架构（2026-11-18）

本页描述codex/exp面积压力分支，以本轮源码和保存结果为准。全引擎其他模块沿既有全景与各自验收；不从本页推导Newton全功能对齐。dev未改，本轮未push。

[离线架构图](assets/area-pressure-architecture-20261118.html) · [后续B1–B3计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md) · [具体live接缝与准入](area-friction-live-integration-handoff-20261118.md)

## 已接通的法向主链

```text
模型源 + 显式body-local压力资产 / 有限刚面
  → 每世界已接受q/v → 当前body FK
  → 完整当前网格准备一次 + 批量候选
  → 有限接触面、真实dA/p/g、双方材料身份
  → k=dA*g、phi=-p/g、双方完整点Jacobian
  → MuJoCo-TYX共享质量和自由度共同法向解
  → 原生运动更新 + 整批状态/载荷共同提交
  → 持久设备记录 + 独立原子观察
  → 任务Env / 面积角色判断 / 多body合力与力矩
```

当前生产面积律为无摩擦、冻结入步几何和Jacobian的弹性法向后向Euler。新面积行不冒充MuJoCo原生solref语义。原生约束仅在已准入组合共解；阻尼与剩余原生约束混合仍有限制。

几何默认HOST；可选设备提出裁剪拓扑，再由HOST完整精确复核。还不是全GPU精确裁剪或面积CCD。压力参数和单位显式声明，不能从URDF猜材料；压力参数也不自动等于真实杨氏模量。

## 本轮实际增量

| 连接 | 已验证的功能 | 证据 |
| --- | --- | --- |
| 当前几何批处理 | 严格空候选、跨world/pair请求、每步完整网格准备一次、晚失败不部分发布 | [I3f](area-pressure-batched-candidates-report-20261118.md)、[I3g](area-pressure-single-preparation-report-20261118.md) |
| 完整步成本 | 固定5/20/32world同输出，24暖步成本减少约9.7–11.3% | [I3h](area-pressure-batch-scale-report-20261118.md) |
| 应用与观察 | 自动当前面接Env；native+area多body合力/矩一次读回 | [I3j](area-pressure-manipulation-env-report-20261118.md)、[I3k](area-pressure-body-wrench-report-20261118.md)、[I3m](area-pressure-wrench-batch-report-20261118.md) |
| 资产与生命周期 | 完整资产保存重载、身份与角色绑定、reset/paired restore | [I3n](area-pressure-assets-report-20261118.md)、[I3i](area-pressure-success-reader-report-20261118.md) |
| 连续运动 | 4world/3model的450批次步；三可动物体双区释放；3种FREE初态各150步 | [I3l](area-pressure-heterogeneous-support-report-20261118.md)、[I3o](area-pressure-three-body-release-report-20261118.md)、[I3s视频](area-pressure-free-body-support-report-20261118.md) |
| 源码外运行 | 新wheel中的自动几何混合场景、检查点、重演与reset | [I3t](area-pressure-wheel-report-20261118.md) |
| 共同摩擦backend | 最终Fn与Ft同解、完整非对称导数、6DOF/128完整点/32world边界 | [选型](area-pressure-friction-model-report-20261118.md)、[I3u](area-pressure-backend-friction-report-20261118.md) |
| 实际入步与设备装配 | 历史真实M/J桥；resident布局→共同kernel，中间无HOST数值读回；13类错误与retry | [I3v](area-pressure-entry-friction-report-20261118.md)、[I3w](area-pressure-resident-friction-pack-report-20261118.md) |
| 摩擦安装包验证 | 源码外128点13数组逐字节同；新resident完整组直接在安装包通过 | [wheel](area-pressure-friction-wheel-report-20261118.md)、[resident报告](area-pressure-resident-friction-pack-report-20261118.md) |

I3s初次轨迹保存后的oracle平面绑定失败已保留，修正只审原轨迹，没有重跑450world-step或把原失败pytest改称全绿。规模结果只针对固定有限模型，不等于任意32台复杂机器人，也不是Newton同机性能比较。

## 摩擦接上了哪些、还差哪些

```text
已验证独立组件：
真实历史入步receipt → owned M/J/载荷问题
实际resident布局 → 设备pack → 128点共同Fn/Ft solver

仍须生产连接：
当前runtime的同入步owner/phase
  → 上述pack/solver
  → actual vnext上的力律重核与一次native积分
  → 新law-aware载荷记录 + 状态共同提交
  → 新reader/asset binding/checkpoint/restore
```

I3v是历史counterfactual，I3w上传原设备字节检验真实布局；两者没有对已推进的Scene再次施力。I3u/w没有改旧type7或v1。现v1假定F=n*Fn，不能放入切向力后继续让旧reader读取。

首版新摩擦准入明确为6DOF、128点、32world的area-only桶，现三点三角积分最多126。保留全样本，允许空面；没有解决同世界一般native约束混合或29关节机器人接入。FP32工程门与HOST严格门分开报告，不用HOST修正替代设备结果。

## 模型边界是下一步决策的一部分

普通cube有限边缘可能有正面积、正压力而g=grad(p)·n=0，当前k=dA*g、phi=-p/g表示不能准入。完整几何应在step前拒绝，不漏掉那片面积。已知g>0域的偏心旋转可先走通；任意滑出另做小规模模型比较。不能把这一限制掩盖成舍入误差或用epsilon修好。

FFT的价值仍是非局部半空间响应与标定参考；Kamino是独立刚体动力学参考。双峰实验显示独立峰相加和force-only表的局限。当前并没有“自动选择任意接触算法”，也没有把FFT格点变成互不相关的小弹簧。

## 后续次序

1. **B1真实摩擦单路径**：一起补显式law/身份、真实owner runtime接线、一次native积分、新Fn/Ft载荷记录。目标是公开Scene能走实际步，不再新增相似参考solver。
2. **B2观察与恢复、多侧与事务**：真JB、多接触共享响应、old/new桶并存、晚失败不部分发布、selected reset/restore。
3. **B3连续任务与成本**：滑动、卸载、偏心转动、两区接触；由同轨迹生成视频，再比较完整步成本。
4. 再扩DOF/同世界一般约束、设备几何与规模、策略真实消费面积特征；最后另做变形体压力散射。

具体文件、原案例复用、并行分工和停止条件已写入[后续计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。当前无摩擦FREE支撑不等于真实自由抓持；薄布大折叠、一般软软、面积CCD和通用自动选型均未完成。每批按新增可观察能力收口，不以测试数量或几何证明数量替代功能。
