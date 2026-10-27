# P1/P2 逐项验收与下一阶段交接

[设计](../superpowers/specs/2026-10-27-p1-p2-runtime-design.md)及[实施计划](../superpowers/plans/2026-10-27-p1-p2-runtime.md)规定的 P1/P2 已完成。这里的 **P1 100%、P2 100%** 仅指这两阶段冻结的验收项，不能折算为整个物理引擎完成度。机器人/三维软体尚未实现，P3/P4 算法为 0%。

| 验收项 | 实际证据 | 结果 |
| --- | --- | --- |
| 完整、可扩展状态与参数合同 | [scene](../../src/phymeo/core/scene.py)、[registry](../../src/phymeo/runtime/registry.py)；粒子/刚体、非 position 的 q/v 与内部量族；缺/多域、精度、形状、非有限、额外字段均检查 | 通过 |
| 显式历史与所有权 | [合同测试](../../tests/test_scene_contracts.py)、[运行器测试](../../tests/test_scene_stepper.py)：规范化复制不依赖虚拟 copy；声明历史、候选隔离、后段失败恢复 | 通过 |
| 世界点端口与交叉响应 | [端口测试](../../tests/test_scene_ports.py)：解析冲量、偏心力矩、重复索引、共享刚体、作用点一致性、位置不前进、双方原子结果 | 通过 |
| 通用阶段和事务 | SceneStepper 拥有子步和迭代预算；真实自由运动、q/v 程序、端口消费者与联合程序共用调度器；预测/未收敛/坏结果/历史/close/设备异常有失败注入 | 通过 |
| 求解器替换与缓存身份 | NumPy、着色 NumPy、tyx 使用相同阶段合同；物理签名与有序绑定分开；新增状态字段拒绝静默丢弃，拓扑重排在工作区前拒绝 | 通过 |
| 数学与学习候选边界 | 缺独立 verifier 的 learned provider 拒绝；独立物理验证失败即拒绝；没有真实 AI 模型、自动跨后端或数学回退声明 | 合同通过，实际 AI 未实施 |
| NumPy/tyx 真正迁移 | 新工作区不调用旧整帧 step；短程同序逐数组比较、实际设备内核及资源故障测试；完整 CPU/CUDA 各 240 帧/960 审计 | 通过 |
| 完整物理证据 | 两后端分别对冻结旧来源的全部 22 数值数组完全相同，质量门槛不变，见[实验报告](p1-p2-runtime.md) | 通过 |
| 性能与遗留台账 | 修复版合同及两后端串行 3×100 次 step、阶段和公开计数；[测量报告](../performance-bottlenecks/measurements/p2-runtime.md)明确未测项，无优化 | 完成规定的小负载基线 |
| tyx 实现问题交接 | [tyx_flaw](../../tyx_flaw/README.md)含模板、证据规则、FP64 独立复现和原始 JSON；未测 Warp 不宣称差距 | 完成 |
| 回归、环境与包 | 实际 CUDA 831 项；Python 3.11 和原生 pip 3.12 各 707 核心项；uv/pip 锁与依赖一致；62 文件 wheel 及源码外安装 smoke | 通过 |
| 独立整分支审阅 | [审阅记录](p1-p2-review.md)：一次新上下文、两个 Important、13 项先失败后通过；完整套件通过，无未处理 Minor/Declined | 通过 |
| 远程 CI | 修复实现 `b946c1924389efcf52f0a57ad6e85e34317cff21` 的 [Documentation](https://github.com/ChenxrC/phymeo/actions/runs/40497001050)、[CI](https://github.com/ChenxrC/phymeo/actions/runs/40497001041) 均成功；[状态摘要](p1-p2-ci.json) | 通过；最终交付文档提交另核验 |

包源码哈希为 `dd1dcd12f14ee1ccc11839dbc3cc5a54516e38f2d4d858109257ce46a2177669`。完整轨迹和性能全部来自修复后的干净 `b946c1924389efcf52f0a57ad6e85e34317cff21`；后续提交只整理证据及交接文档。该分支推送到 `codex/p1-p2-runtime`，保留 checkout，不合并主分支或发布软件包。

## 后续推进

1. P3：冻结机器人首版设计，固定基座树、单自由度旋转/移动关节、q/v、运动学/雅可比、质量/偏置和驱动。先建立单摆/两连杆解析基线，再接入 `J M^-1 Jᵀ` 世界点响应；不把单连杆质量当成关节系统响应。
2. P4：布料沿用已验证 XPBD；为一种四面体材料确定能量/力或约束、积分与倒置门槛，先做刚体运动零应变及能量/力一致测试，再接端口和候选历史。
3. 后续 P5/P6 汇合机器人—布料、机器人—软体和三类共存场景，逐项保持一次反作用、全局回滚和实际 tyx 路径；两人分工继续按[集成大纲](../superpowers/plans/2026-10-27-integrated-framework-outline.md)。
4. 时间步敏感性、精度、几何边界和性能热点继续使用独立台账。tyx agent 可直接从 `TYX-001` 开始；是否将 FP64 纳入正式类型合同由依赖路线决定，不擅自更新固定版本。

后续实施须另有用户授权；本轮完成后结束续跑，不自动开始机器人、软体或优化。
