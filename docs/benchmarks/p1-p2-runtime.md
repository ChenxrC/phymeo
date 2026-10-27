# P1/P2 统一运行器完整迁移验收

修复后的干净提交 `b946c1924389efcf52f0a57ad6e85e34317cff21`，包源码 SHA256 `dd1dcd12f14ee1ccc11839dbc3cc5a54516e38f2d4d858109257ce46a2177669`。本轮只交付 P1/P2 合同与统一运行框架；机器人关节树和体积软体算法仍是后续工作。详细接入方法见[运行器合同](../development/runtime-contracts.md)，两项独立审阅问题及修复见[审阅记录](p1-p2-review.md)。

## 完整物理轨迹

场景和物理门槛沿用[刚柔参考实验](rigid-cloth.md)：49 个布节点、一个动态球，240 帧，每帧 1/120 s、4 子步，位置/速度各最多 200 次迭代。NumPy 为 FP64、容差 1e-7；实际 tyx CUDA 为 FP32、容差 1e-6。两次运行串行完成。每个子步独立检查有限性、固定点、姿态、穿透、接触载荷和球体运动平衡。

| 模式 | 接受帧 / 审计子步 | 有接触子步 | 最大报告残差 | 最大独立穿透 m | 总时间 s，含审计 |
| --- | --- | --- | --- | --- | --- |
| numpy | 240 / 960 | 17 | 9.99975279e-08 | 2.77555756e-17 | 392.151 |
| cuda | 240 / 960 | 16 | 9.99998656e-07 | 2.55737456e-08 | 293.077 |

两种后端分别与冻结的旧同后端、同算术次序轨迹比较：22 个数组全部形状、dtype、逐元素完全一致，最大差均为 0。数值指纹：

- numpy: `1416f05763fa3ba1c8bb2aaa810a197ccd5b28254b557e9b4b8d090419216ef3`。
- cuda: `29adf9a60bd3c00c3cca4b4c38564c3136bd20ffb58f09e845d840a4cbd80db7`。

[CPU 原始比较](p1-p2-numpy-migration.json)和[CUDA 原始比较](p1-p2-cuda-migration.json)保存旧/新源码、完整参数、质量指标和每个数组的比较结果。旧来源为干净 `027e7cd`，源码哈希 `74d38ca64a0e70bd1601010e93f1ae527f9729ca9870df60bee884a186d4d75d`。新轨迹位于本地 `artifacts/p1p2-runtime-{numpy,cuda}-reviewed`；仓库保存可再生成的小型证据，未上传大型 NPZ。

## 阶段观测

| 阶段 | NumPy 秒 | CUDA 秒 |
| --- | --- | --- |
| setup | 0.390210 | 0.406656 |
| predict | 0.329884 | 4.306291 |
| position | 386.175266 | 282.347176 |
| prepare_velocity | 0.124603 | 0.847277 |
| velocity | 0.625049 | 0.705404 |
| validation | 0.216379 | 0.230989 |
| verification | 0.000670 | 0.000639 |
| diagnostics | 2.434831 | 1.740087 |
| close | 0.000651 | 0.381524 |

这些阶段由主机墙钟计时，位置阶段同时包含投影、同步/回读和主机残差判断，不能拆解为纯 kernel 或纯传输成本。通用 verifier 阶段未安装额外学习验收器，其接近零的调用开销不代表没有物理验证；本场景的独立审计在实验层执行。阶段总和不等于全运行时间，差额包括外层所有权、身份检查、适配、记录和独立审计。

位置迭代是这个完整场景中最大的已测阶段；阶段内部的进一步归因仍待 profiler。旧记录有并发负载，不拿总时长计算新旧加速比。FP64/FP32 算术不同，也不把两个后端的时间比当作 GPU 加速结论。[串行小负载测量](../performance-bottlenecks/measurements/p2-runtime.md)另外记录 3×100 次热态 step、冷启动边界和公开传输/launch 计数。

## 复现

安装环境按[uv/pip 说明](../development/environment-and-ci.md)。以下用已安装固定 tyx 的 Python，uv 用户可在命令前加 `uv run --no-sync`；输出目录必须未存在。

```sh
python examples/coupled_runtime.py --mode numpy --output artifacts/p1p2-runtime-numpy-reviewed
python examples/coupled_runtime.py --mode cuda --output artifacts/p1p2-runtime-cuda-reviewed
python scripts/verify_runtime_migration.py --reference artifacts/rigid-cloth-reference-reviewed/two_way --candidate artifacts/p1p2-runtime-numpy-reviewed --output artifacts/p1p2-numpy-migration-reviewed.json
python scripts/verify_runtime_migration.py --reference artifacts/rigid-cloth-device-reviewed/device --candidate artifacts/p1p2-runtime-cuda-reviewed --output artifacts/p1p2-cuda-migration-reviewed.json
```

比较脚本校验来源、配置、完整时域和数值哈希后才比较数组。需要旧基准文件时，按旧实验报告在对应固定提交与同后端生成；不要把任意新轨迹重新命名为冻结参考。

## 限制与交接

本轮保持原离散无摩擦顶点/球模型，没有改善已知约 0.145 m 的半步长敏感性；该数字来自上一轮完整实验，本轮没有重复或重新声明步长收敛阶。见 [NUM-001](../legacy-issues/NUM-001-timestep-sensitivity.md)、[NUM-002](../legacy-issues/NUM-002-fp32-reference-semantics.md)和[几何边界](../legacy-issues/GEOM-001-discrete-vertex-contact.md)。tyx FP64 计算缺口有独立复现，已交接到 [TYX-001](../../tyx_flaw/TYX-001-fp64-compute.md)；Warp 未安装、未比较，不宣称性能差距。
