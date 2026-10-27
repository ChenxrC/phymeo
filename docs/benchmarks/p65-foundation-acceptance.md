# P6.5 基础切片验收（完整综合案例尚未完成）

分支 `codex/p6-5-handoff`，P6基线`6dd5372dd8784c481c0c5eed90db93c7efcb86c2`。本记录只覆盖H1a重心/成对端口、H1b静态特征、H1c多域瞬时图、H2a局部摩擦和H2b固定构型联合冲量。

**没有持续抓布、双臂交接、落台、布弯曲/自碰撞、完整网格CCD或Three.js回放的验收声明。** P6已完整完成；P6.5基础通过不等于P6.5完成。

## 审阅和正确性

唯一整分支审阅的2项Important和1项Minor在同一轮修复，完整证据、排除范围和代价见[审阅裁定](p65-foundation-review.md)。最近点增加独立最优性认证；候选新增逐域真实速度/总动能审计，不能用响应预测冒充实际状态。正常FP32舍入与依赖FP64能力缺口分开记录。

产品源码哈希`ceeb38c10cfce15a725da9d7347281daf363a66193cbf08408ff9f6907e85ea2`，105个Python文件，修复提交`1e5314248fea397ca14038fe953cfc9dc7d115bd`已推送并核对远程相同。完整参数、包哈希和结果见[工程JSON](p65-foundation-engineering.json)。

| 验证 | 实际结果 |
| --- | --- |
| 实际CUDA全套 | 1484通过，494.70s |
| Python3.11核心 | 1214通过、270设备测试明确排除，334.86s |
| 原生pip Python3.12核心 | 1214通过、270设备测试明确排除，345.10s |
| 受影响真实C++参与者/表面/多域/接触 | 22通过，306.04s |
| 候选/几何/接触＋解释器针对性组合 | 91通过，3.10s |
| NumPy基础/旧端口针对性回归 | 200通过，1.10s |
| 源码外原生pip wheel | 105文件逐字节一致，干净环境安装、依赖与求解器smoke通过 |
| 质量/依赖 | ruff、公开tyx导入边界105文件、uv和native-pip依赖检查通过 |

构建第一次因沙盒默认uv缓存无写权限失败，改工作区缓存后离线隔离构建又缺setuptools缓存；最终明确使用已安装构建工具的`uv build --no-build-isolation --offline`成功。源码外安装验证另在新原生pip临时环境通过，没有把失败尝试计为通过或归为产品缺陷。

[修复后串行观测](../performance-bottlenecks/measurements/p65-reviewed-foundations.md)120次联合求解及端口/几何小负载均验收通过。只记录开销，没有优化，也没有Warp等价实测。

修复提交的[Documentation40595805153](https://github.com/ChenxrC/phymeo/actions/runs/40595805153)与[CI40595805121](https://github.com/ChenxrC/phymeo/actions/runs/40595805121)均已完成且success，head SHA准确为1e5314248fea397ca14038fe953cfc9dc7d115bd。本次五个基础切片的数值/工程/审阅/性能门禁收齐；后续文档或新功能提交的CI单独核对，不沿用旧状态。

## 历史检查点

635ba74已推送且远程SHA相同，准确[CI](https://github.com/ChenxrC/phymeo/actions/runs/40592982641)和[Documentation](https://github.com/ChenxrC/phymeo/actions/runs/40592982671)成功。该源码全CUDA1471、Py3.11与pip3.12各1204、104文件wheel通过，但上述审阅发生在其后，不能拿旧通过证明修复版已验收。

## 后续方向

先完成这些基础的当前源码工程收口，再冻结非线性时间步/载荷账本、有限驱动控制和几何路径验收。优先实现“单爪真实夹布承重＋低摩擦滑落”最小动态场景，随后两臂共同承重、交接、放置及同源Three.js回放。[完整里程碑](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)的扰动、减半步长、细化网格、碰撞、事务和工程门槛保持不变。
