# P1/T1：状态与历史快照合同测量

2026-10-27；这仅测量合同规范化、模型身份检查和副本创建，不是物理 step、GPU 吞吐或优化效果。

49 个粒子、49 个显式历史数值，FP64；2×100 预热，3 组各 100 次测量。首次记录的中位数约 0.044–0.051 ms，p95 约 0.054–0.057 ms。样本很小，未扣除计时器开销；峰值内存和各函数占比未测，不从总时间推断瓶颈。

[机器可读记录](p1-contracts.json)保留实际源码指纹与环境；这是提交前工作区的测量，`git_dirty=true` 明确标记，不冒充旧提交的结果。数值只用于后续定位开销，不能替代同一物理质量下的 P2 完整 step 测量。

复现：`python scripts/measure_scene_contracts.py --output artifacts/p1-contracts-new.json`；uv 环境加 `uv run --no-sync` 前缀，输出文件必须不存在。

入口：[StateRegistry.snapshot](../../../src/phymeo/runtime/registry.py)、[SceneModel.assert_unchanged](../../../src/phymeo/core/scene.py)。后续若发现模型指纹或复制占主要成本，再以分项证据提出优化；本轮保持所有权与完整验证。
