# PERF-005：体积软体的材料域检查与事务快照

- 状态：阶段时间已测，重复 SVD 调用结构已确认；具体占比未分解。
- 源码：`664c989`，P4 2026-10-27；[完整测量条件与 JSON](measurements/p4-softbody.md)。

## 现象和位置

192 单元完整 step 中位数 3.724–3.871 ms，预测中位数 0.428 ms、setup 0.533 ms、候选校验 0.551 ms、诊断 0.799 ms。
这个参考路径保留数值域和事务门禁，不能把全部开销称为材料力计算。

- [tetrahedral_family](../../src/phymeo/runtime/tetrahedral_family.py)：入口与候选检查形变梯度/奇异值；复制和绑定验证分别由 registry/SceneStepper 执行。
- [StVK](../../src/phymeo/materials/stvk.py)：能量/内力各自验证物理域，deformation_gradient 调用 determinant/SVD。
- [TetrahedralWorkspace](../../src/phymeo/solvers/tetrahedral.py)：position_metrics 与 diagnostics 也读取当前形变，保障没有坏候选被当成成功。

## 后续方向

先在冻结 q/x、相同材料/单元/接触预算下计数并 profiling，拆开快照、指纹、F/det/SVD、力和诊断；
候选变化后再比较是否能共享一个只读的材料评估结果。不能只移除 SVD/固定点检查来制造速度改善。
缓存依赖每次接触改变的位置，且不能留下跨帧隐藏物理状态或污染回滚。
在 P5 的多域接触中按实际 sweep 次数重测，再决定优化优先级。暂无纯 tyx 材料实现，不能归因到 tyx 或宣称 Warp 差距。
