# 当前帧稀疏间接光实验

这是独立源码副本、DLL、SPIR-V 和 Python 传输层。不会修改原产品默认，也不将空间重建称作全图 Monte Carlo 均值。当前作者交付为待编译和设备验证的实验代码；真实状态以每次运行的 JSON 为准。

## 固定定义

- 主光栅全分辨率计算原直接光、环境镜面及可选近场镜面。漫反射 IBL 被零 incoming 取代；保留每像素自身的漫反射系数 `rho`，按 RGB8 编码进 RGBA32F 的 alpha 数值整数，误差每分量不超过 `0.5/255`。这项量化是明确的实验近似。
- 同会话第二次光栅输出当前着色法线（oct 10+10 bit）和 incoming；只有每个 2×2 或 4×4 晶格的一个锚点发出 2、4 或 16 个真实单跳提案，沿用实际 TLAS、终端直接照明及 HDR MIS。tile=1 是同估计器全率对照。
- 只聚合相同真实 ID、有覆盖、相对深度差不超过 `.003`、法线点积至少 `.98` 的锚点 incoming，最后乘**当前像素自己的 rho**。4 候选使用距离倒数权重；25 候选使用 5×5 晶格窗口、sigma=1.5 个晶格宽的 Gaussian。25 候选也过滤锚点，tile=1 例外。
- 无支持邻居使用本像素的无遮挡环境 SH（无环境即零）。这不等价于真实遮挡 GI，薄缝和小对象可能产生可见偏差；通过实际 `fallback_pixels` 和薄片控制暴露。
- 每个 RGB capture 都是独立当前帧曝光，既不累计历史也不声称 full-image spp。`read_reconstructed_linear()` 读的是线性重建结果。只支持 MSAA1、硬件 ray_query、一次漫反射；面积灯格点固定为构建参数 2/4/8。
- `primary_proposals` = **最终可见**锚点数 × 配置提案数，不是硬件查询计数，也不含被遮挡 overdraw/helper invocations；不拿该字段冒充所有 GPU 工作量。

## 构建及小场景（由唯一执行 owner 运行）

设置 `PYTHONPATH`，让 `preview_sparse` 所在的 `rendering/examples` 和保存实际场景工厂的父 `rendering/examples` 可导入。程序首先启用生成副本，再导入 renderer；不要预先 import 原包。`VULKAN_SDK` 指向本机已有 SDK，不下载或安装依赖。

```powershell
python -B -m unittest preview_sparse.test_contract -v
python -B -m preview_sparse.build_source --rendering-root ORIGINAL_RENDERING --output NEW_SOURCE --tile 4 --paths 16 --grid 2 --neighbors 25 --reconstruction-revision 2
cmake -S NEW_SOURCE/source/_native -B NEW_SOURCE/build -A x64
cmake --build NEW_SOURCE/build --config Release
python -B -m preview_sparse.run_smoke --prepared NEW_SOURCE --baseline-library ORIGINAL_DLL --baseline-shaders ORIGINAL_SPV_DIRECTORY --device-index 1 --validation --output NEW_RESULT
```

默认 smoke 包含封闭室、97×95 奇数尺寸、饱和色接收面、没有 tile4 锚点的细片、零光源，共 12 次真实捕获（含仅传感 resize 与同 seed 重复）；`--panda` 另加真实 Panda 第 60 帧的一对 720p 捕获。对照为原 DLL 单跳 16 提案、无滤波；保存 RGB、深度/ID/VALID、线性结果、稀疏 incoming/normal guide、逐 owner 收据及关闭状态。

传感必须逐字相等，零光必须不自发光，resize 后同状态同 seed 必须逐字复现。记录图像误差而不把它自动判定为画质通过。所有输出目录必须不存在。失败前缀保留；几何准备尚未结束时 `report.json` 保持 `started`，GPU 或发布失败记录 `partial_failure`。

## revision 与公平比较

默认 revision 1 保留首包：候选先 clamp 再去重，每个有效邻居一次全局 atomic。显式 revision 2 仅遍历有效 tile 索引，并把邻居计数在本像素累加后一次 atomic；其余实际计数不删。候选顺序、浮点权重和归一不变；奇数边界顺序有独立 CPU 测试。必须用新目录生成第二包，不覆盖首 DLL。

每个进程只启用一个 `--prepared`。比较两个 revision 时分两个进程保存相同小场景或同 Panda 入态，在 CPU 逐字核 RGB、传感、RGBA32F 重建、guide 以及实际计数，然后报告相同模式的时间。改变 tile/paths/邻居数是算法对照，不能要求其 RGB 相同。冷启动、环境准备、调试读回和 capture 时间分别记录；guide 调试读回从不反馈给 GPU 算法。

新生成 runtime 以 `_preview_sparse_requested` 私有标记分派：只有 `SparseBackend` 设置它并检查独立 ABI。原 DLL 的 `none`、`guided` 和 `guided_settling` 均直接选择各自旧 setter，不访问 sparse 符号。早期首包曾在 settling 分派前先查询 sparse 符号，失败证据保留；这个兼容修复必须重新生成目录，不就地改已封存 runtime。实验主/guide/compute 均有专属具名 marker 和配置 profile，拒绝错误混用。

## 可选 visibility-ratio 估计器

`--estimator visibility-ratio` 是独立实验档；默认 `radiance` 不变。沿相同真实方向和原 MIS 权，分别求无遮挡 sky 总和 D、实际 miss 的 sky 总和 V、真实 hit 的终端 radiance 总和 H。输出每通道为 `SH(N) * V/D + H/Nsamples`。Ng 裁剪的方向仍进入 D，V/H 为零；D 某通道为零时使用同权重的 miss 可见率，不设任意能量阈值。没有环境时 SH=0，退化为原 hit 平均。

这会引入有限样本 ratio 偏差，并继承原 SH 近似；不宣称无偏或物理真值。全开会精确回到既有 SH/constant IBL，全挡去掉 sky，彩色真实 hit 保留。不增加射线或重复加 IBL，也不钳制 radiance 伪造收敛。相对于普通估计器，实际 hit 方向还需一次环境查询来计算无遮挡分母，成本需实测。

运行 `python -B -m unittest preview_sparse.test_contract preview_sparse.test_ratio -v` 可核 CPU 边界。新 ratio 包的 smoke 增加开放/封闭/彩色终端三例：共 18 次捕获，附 `--panda` 共 20 次。guide alpha 的 bits 21/22/23 分别保存最终可见锚点是否出现真实 miss/hit/Ng 裁剪；只作独立调试读回，计算不依赖这些统计位。原普通估计器 guide 格式范围不变。
