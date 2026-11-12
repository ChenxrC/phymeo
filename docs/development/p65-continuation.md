# Phymeo 当前续作入口

更新：2026-11-13，**三主线 Batch C 已完成源码与安装资格**。D3真实设备节点/运动刚体双向法向响应、G1异构模型分桶已交付；产品 `3534f742` 的528成员同源，四个源码外解释/CUDA消费者通过。整个P6.5 / Newton功能对齐仍未完成，无新截止。

## 当前入口

1. [Batch C实际结果](p65-three-track-batch-c-20261112.md)：公共能力、实际验证、首失败、资源与边界。
2. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：当前缺口及优先级。
3. [三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)、[Batch C合同](../superpowers/plans/2026-11-12-three-track-batch-c.md)。
4. [D3报告](p65-deformable-contact-report.md)、[异构报告](p65-heterogeneous-batch-report.md)、[统一编译架构](../architecture/unified-scene-compilation.md)。
5. [Batch B](p65-three-track-batch-b-20261112.md)、[Batch A](p65-three-track-batch-a-20261112.md)、[更早记录](p65-continuation-history-through-20261112-0035.md)。按需定位，不恢复旧heartbeat或重审旧证据。

## 最新已验范围

- **S1–S3公共场景查询**：多参与者声明/编译、真实设备FK和统一身份已连接，两类不同源码外场景通过。仍是同runtime有限边界查询，不自动产生新载荷或CCD。
- **D1/D2**：持久真实节点及其当次accepted顶点→bounds/BVH→有限球/胶囊查询已通过；CONDITIONING保持显式拒绝施力。
- **D3首片**：公共TyxDeformableContactBatch连接双来源、原法向层、完整ordered材料反力与真实浮体动力学，共同候选/发布/reset/restore。解释27、CUDA28通过；安装每后端8共同steps/13worldsteps/8queries。限定W≤8、每world一个节点模型及一个ordinary单body运动球/胶囊、零摩擦、入步采样；有HOST载荷往返，非隐式耦合。短窗入层/离层未替代D4持续支撑。
- **机器人组合**：fixed direct BALL FILTEREXACT＋原接触、单FREE＋BALL/NONE电机基础格已验；新浮根球树的contact/activation/limits/dry仍拒绝。
- **G1异构首片**：MuJoCoTyxHeterogeneousBatch按完整模型/资产/cap分桶，全部world力与运动/边界事务共同接受。新源码42通过；安装每后端27calls/162worldsteps。原cap为8模型/16world，实际首片3模型6world；不是最大规模、单异构kernel、性能或G1人形舞蹈资格。
- **统一包**：528产品source/wheel/installed逐字节相同。四消费者全部通过，18文件完整压缩归档，原失败保留；不称新包全量回归。

## 下一批

1. **D4-N连续法向载荷**：先用真正有固定边界且有自由节点的弹性片检验持续支撑、卸载离层和再接触，同一真实轨迹保非零入步速度。当前D3已有无记忆normal law及事务，源码调研未发现必须先新建solver；先补通用连续指标/多案例与独立参考，产品缺陷按实证窄修。原全free压片不能拿来判定固定环境支撑。
2. **G2a原容量内异构规模**：原三已验模型W1各一格、W8/W16及8桶16world，保原h、24步控制序列、每world完整参考及输出。先核功能/资源，live/peak设备内存未有完整observer则明确UNKNOWN；不以累计上传或旧同构S256代替。
3. **后续**：D4摩擦/换面历史、D5统一场景组合；G2b须完整生命周期资源预算后再扩32/128/256，G3同模型/质量/输出Newton比较后置。

下一批具体输入、门、文件owner需在首次执行前登记；不得从失败输出调fixture/质量/时间窗或放宽门。允许事先声明的有限穿入/漂移，不恢复精确相触和无限证明前置。同假设两次诊断不能缩小问题则保最小反例并调整方向。

## 保留失败与操作边界

原G1人形46.29s参考XY门/131.48s完整动作、J/R学习门、R03两项净加速度门、B-point touch-SE仍未通过。N8原768转移/0新更新及297条审计保持，不自动追加训练或调clip/lr。通用CCD、摩擦历史、实体内外、更多机器人组合、批量传感、可微物理/跨域、密集布与体软体、杆缆/MPM/流体仍待补。

父目录 C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo；分支codex/p65-parallel-modules，本地未push。本批证据 `artifacts/p65-three-track-c-20261112-01`，installed-close.json绑定528产品、四消费者及18文件。原三作者沿用隔离worktree；父独占共享协议和集成、全部项目执行串行。下一批先查实时agents/Git/WIP，不重派已验工作。

所有import/collection/数值/pytest/离线probe/lint/formatter/build/install/profile必须单owner及fresh entrypoint-aware guard。无周期任务、新截止、remote CI、push/mainmerge/发包。保留三份未跟踪草稿、用户Torch、当前venv/pins、首失败及旧证据Junction；不跟随Junction清理或全解压补副本。
