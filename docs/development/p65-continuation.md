# Phymeo 当前续作入口

更新：2026-11-12，**三主线 Batch B 已完成源码与安装资格**。S3 公共场景接线、D2/M2-Q 真实变形查询、R2 FREE＋BALL 基础组合已交付。统一产品快照 `0321c5a4`，522 成员同源，六个源码外解释/CUDA消费者通过。整个 P6.5 / Newton 功能对齐仍未完成，无新截止。

## 当前入口

1. [Batch B 实际结果](p65-three-track-batch-b-20261112.md)：当前能力、全部实际执行、首失败与下一步。
2. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：已实现的范围、保留缺口和优先级。
3. [三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)与[Batch B](../superpowers/plans/2026-11-12-three-track-batch-b.md)：D3/G1为下一批入口。
4. [S3报告](p65-participant-scene-boundary-report.md)、[D2报告](p65-deformable-boundary-report.md)、[FREE＋BALL报告](p65-free-ball-composition-report.md)及[设计](p65-free-ball-composition-design.md)。
5. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo 编译、CPU/TYX/显式 Warp 与 resident 路径。
6. [Batch A 已验基础](p65-three-track-batch-a-20261112.md)、[更早续作记录](p65-continuation-history-through-20261112-0035.md)。按需定位；不恢复旧 heartbeat 或重审旧证据。

## 最新能力与范围

- **S3**：编译后的公共场景能准备设备边界查询，统一声明、真实步进、查询和生命周期。新增带固定网格工具的 mimic 结构＋自由物体安装场景，与 Batch A 场景不同。源码解释/HOST27、CUDA20；安装每后端8逻辑/16参与者步、5queries/60rows、3stale拒绝。查询不是新接触力、CCD或跨runtime组合。
- **D2/M2-Q**：D1当次accepted顶点→gather→真实变化的bounds/BVH→原球/胶囊有限查询。解释31、CUDA31；安装每后端32worldsteps/19queries。102行RESOLVED、12行CONDITIONING；斜向平行未解析行无力法向。静态round与节点同runtime；双向响应和连续载荷仍待补。
- **R2 FREE＋BALL**：单FREE根＋BALL/R/P后代、NONE JOINT电机、完整M/J/bias与地址/CSR、同次力与运动、checkpoint/真实后继。64新源码节点及16受影响兼容通过；安装每后端112worldsteps。nv35/nq45/nu32，不含该新组合的contact/activation/limits/dry；容量正控不等于规模或性能。
- **统一包**：产品0321c5a4的522成员source/wheel/installed逐字节相同；六消费者完成，147安装证据文件已归档，首失败保留。后续修正只触及消费者/文档，不改变产品字节。

## 下一批按此推进

1. **D3 接触力与共同发布**：先核既有材料点/接触层和事务接缝，再实现真实节点与运动刚体在同一候选中的双向响应。压片、偏心载荷、运动胶囊推动片为不同正控；核双方力/力矩、力与运动、固定点反力及晚失败。不可将accepted后的独立query冒充步内响应；CONDITIONING不得直接生成法向。
2. **G1 异构机器人功能**：可与D3隔离编码，只组合已验关节/执行器/约束域。先小批、不同DOF与真实非零控制、分桶和世界身份、恢复后继/末world失败；再G2容量梯度。FREE＋BALL的接触与activation扩展另列，保持准入拒绝。
3. **后继D4/D5与G2/G3**：持续支撑/摩擦/换面/卸载、统一场景安装；同模型/输出/质量的Newton性能比较后置。功能优先，性能超越非强制。

S3当前公共查询目标已经闭环，后续维护和响应接线复用此入口，不另起私有bank/场景路径。允许预先声明的有限穿透、漂移和数值误差；不恢复精确相触或无限证明前置。两次同假设诊断不能缩小问题时保存最小反例并调整方向，不改fixture/预算追绿。

## 保留的失败与全景缺口

G1原46.29秒参考XY门及完整131.48秒动作、J/R原学习门、R03两项小净加速度门仍未通过。B-point touch-SE候选仍隔离；新查询能力不自动修复曲面接触表示。N8原768转移/0新Adam更新与297条审计保持，不自动追加训练或调clip/lr。

D3/D4/D5、通用CCD/摩擦历史/闭实体内外、更广机器人组合和异构规模、批量传感、可微物理/通用跨域、密集布/体软体、杆缆与MPM/流体均未因本批完成。旧局部结果按原版本复用，不当新包全量回归或总体完成百分比。

## 所有权与操作边界

父目录 C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo，分支 codex/p65-parallel-modules，本地未push。Batch B证据根 artifacts/p65-three-track-b-20261112-01；installed-close.json绑定522产品、六消费者及147归档文件。三个作者已释放，无子agent项目执行；父完成本批收尾后释放唯一串行槽。

继续沿用原agent隔离编码；父独占共享协议和集成。所有项目import/collection/数值/pytest/离线probe/lint/formatter/build/install/profile必须fresh entrypoint-aware guard后串行。无自动周期任务、新截止、remote CI、push/mainmerge或发包。

保留原K1三个未跟踪草稿、用户Torch、当前venv/pins、首失败、旧证据Junction；不跟随Junction清理，不全量解压补副本。下一批先查实时agents/Git/WIP，再按这里的当前依赖前进。
