# P6.5 厚度端口验收

产品代码`10f8cdd72ee4c4d009386267f8adab35ed427255`，源码108文件，SHA256 `eac7e6ccc0c2052fcf54c59adeed74237c3b5b1a5fe8fa4962301bed4557edc6`。承接原六个基础模块，本次新增第七项：固定构型下布表面厚度材料点的速度、冲量响应和反力矩。**完整P6.5双臂递布与Three.js仍未完成。**

## 实现与物理证据

入口`TriangleOffsetPointPort(base, triangles, weights, offsets)`，实现[p=B*x+s*n的完整块Jacobian合同](../superpowers/specs/2026-10-28-p65-offset-port-design.md)。不只是移动渲染位置：材料速度为J*v，冲量用J.T传给底层，响应为J*R*J.T；共享节点/关节先合并载荷，整域候选动能只审计一次。

35项新核心测试包括独立单位三角形解析载荷/力矩/能量、24组中心差分、平移旋转虚功、共享节点交叉响应、机器人直接附件对照、复制所有权、退化/精度损失拒绝。3项实际tyx测试验证正负偏置与双侧纯角冲量；interpret/C++/CUDA实际完成launch，不以主机结果覆盖设备结果。映射及几何是host FP64，底层响应为实际tyx FP32路径；不称整体GPU原生。

## 修复后工程验证

详细值与构建包哈希见[机器记录](p65-offset-engineering.json)。

| 验证 | 实际结果 |
| --- | --- |
| 全量CUDA suite | 1559 passed，511.32s |
| Python3.11核心 | 1284 passed / 275 tyx deselected，374.63s |
| 原生pip Python3.12核心 | 1284 passed / 275 tyx deselected，378.04s |
| 核心＋actual interpret | 38 passed，.55s |
| actual C++ | 3 passed，18.57s |
| uv构建 / 源码外全新pip wheel | 108文件完全一致；公开API及求解器smoke通过 |
| ruff / 公共依赖边界 / compileall / uv、pip check | 通过 |

一次独立[审阅](p65-offset-review.md)发现2项Important，均先补失败测试再修复：微小非零厚度被大坐标吞掉、einsum静默下溢。原未修复CUDA长测主动中断，不计为通过；上表均为修复后源码。没有对旧P6轨迹和前六个基础重复做独立审阅。

[7200次串行观测](../performance-bottlenecks/measurements/p65-offset.md)零失败。包括host几何映射、实际底层执行、读取候选和独立断言；不是完整仿真step性能。未做优化、Warp对比，未定位新的tyx实现缺陷；主机数值边界写入[NUM-004](../legacy-issues/NUM-004-candidate-representability.md)。

远程CI必须在本文件提交并push后以准确head核对，本文件生成时不冒称已通过；最终run链接及远程SHA写入交付记录。CI矩阵延用Windows/Linux、3.11/3.12、uv/pip、wheel与隔离MuJoCo oracle，设备证据仍来自本机实际CUDA。

## 后续顺序

参见[续作入口](../development/p65-continuation.md)及[动态账本反例](../development/p65-dynamic-ledger-candidate.md)。先冻结和实现动态位置/速度/接触冲量的统一账本，验证重力静置、静摩擦与滑移、单关节、特征切换及最后子步回滚；再接保守CCD、弯曲/自碰撞、有限驱动单爪、双臂递布及同源Three.js回放。边/顶点厚度包络和材料点更新仍是必要范围，不能把本端口当作完整碰撞系统。
