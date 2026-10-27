# NUM-002：FP32 设备与参考算术的差异

- 状态：已量化、原场景门槛内；属于已知数值边界，不是已确认缺陷。
- 核对日期：2026-10-27；实验源码 `027e7cd`，定位版本 `67371f9`。

## 描述与原因

同着色 NumPy 参考采用 FP32 状态存储及 FP64 几何/乘子；tyx 采用 FP32 算术。其差异同时包含存储舍入和算术误差。与原顺序 FP64 比较时，还包含约束次序和请求容差差异。

当前球/布最大位置差：相对同顺序参考为 0.000110627 / 0.0000195024 m；相对原顺序 FP64 为 0.000720128 / 0.000628003 m。CPU 和 CUDA 各自重复的 22 个数组逐元素相同，但未验证跨机器逐位一致。

依据：[设备摘要](../benchmarks/rigid-cloth-device.json)、[解释与复现方法](../benchmarks/rigid-cloth.md)。不能用跨后端逐位比较替代合理误差验收，也不能把不同顺序/容差的耗时换算为加速比。

## 代码入口

- [colored_coupled.py](../../src/phymeo/solvers/colored_coupled.py)：同着色 CPU 联合参考。
- [tyx_coupled.py](../../src/phymeo/backends/tyx_coupled.py)：`TyxCoupledXPBDSolver`（48 行）、主机残差门禁。
- [_tyx_coupled_kernels.py](../../src/phymeo/backends/_tyx_coupled_kernels.py)：设备位置/速度接触投影。
- [rigid_cloth.py](../../src/phymeo/experiments/rigid_cloth.py)：配置和结果精度标记。

## 后续方向与升级条件

先匹配顺序、物理签名、时间步、终止条件，再分解误差来源。新机器人和软体分别建立单位与尺度相符的误差预算；不直接沿用原球场景的 2 mm 门槛。大坐标、小质量等输入按已有数值回归验证；超出声明范围应可观察地拒绝。

后续可评估混合精度、坐标局部化和量化误差界，本轮不为追求更小数字改写已验证内核。若新场景越过冻结的误差门槛或设备返回错误成功，转为阻塞项。关闭需有明确适用域和对照证据，不以某一次轨迹更接近作为结论。
