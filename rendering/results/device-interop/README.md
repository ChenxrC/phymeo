# 真实物理位姿经共享显存到 Vulkan

2026-11-24，在 Windows 的 NVIDIA GeForce RTX 4060 Laptop GPU 上，两个独立的 MuJoCo-TYX 世界各带一个滑块。施加方向相反的控制，每个世界推进 24 步。物理观察器在第 0、12、24 步将位姿发布给同卡的 Vulkan 计算着色器。三个时刻的 144 个 32 位字全部通过往返核对。对照用的 CPU MuJoCo 单独推进，两侧各自生成状态。[真实位姿原始记录](live-poses.json)

| 物理时刻 | 共享槽 | 世界 41 滑块高度 | 世界 7 滑块高度 | 相对 CPU MuJoCo 的最大位置误差 |
| --- | --- | ---: | ---: | ---: |
| 0 ms | 0，第 1 次使用 | 0 mm | 0 mm | 0 m |
| 12 ms | 1，第 1 次使用 | +1.154 mm | −1.154 mm | 1.01 × 10⁻¹⁰ m 以内 |
| 24 ms | 0，第 2 次使用 | +4.430 mm | −4.430 mm | 4.91 × 10⁻¹⁰ m 以内 |

## 三项验证

| 工作线 | 已验证 | 下一道验收 |
| --- | --- | --- |
| A · 已提交的物理状态 | 每次保存两个世界各自的帧键和刚体范围。每个世界有一个世界坐标体和一个滑块，共四个 FP32 3×4 矩阵。位姿发布专项 4 项通过，相关刚体观察回归共 14 项通过。 | 合成刚体到可视网格的局部变换，覆盖偏置、转动和一个刚体挂多个网格。 |
| B · 共享显存与同步 | Vulkan 分配两个 SSBO，CUDA 按设备 UUID 匹配并导入，每个槽各用两个外部信号量。在每槽 513 个字的对照中完成 6 次交换，双方各核对 3,078 个字；故意传错标记时检出 513 个差异。错误设备、句柄类型、部分初始化失败和过早复用均得到预期结果。两个实机报告的 Vulkan 验证错误和退出后跟踪的原生资源数均为 0。[双槽原始记录](exchange-integrated.json) | 将共享位姿交给常驻光栅的实例矩阵与所需的加速结构，验证画面中的运动和阴影同步。 |
| C · 图像消费者 | 在非默认 CUDA stream 上读取 4 张已有的 R1 工位 RGB，分箱 A/B/B/A；打乱输入后分类也按相应顺序改变。4 项 CUDA 消费者测试通过，覆盖图像行跨度和分类边界。[消费者测试](../../examples/galaxea_r1/test_sorting_device_reader.py) | 接入实时 Vulkan 相机附件，检查张量的帧键、释放时机和归档图像对照。 |

## 一帧数据实际去了哪里

| 操作 | 每个双世界采样时刻 |
| --- | --- |
| 打包 4 个刚体位姿，再送入共享 SSBO | GPU 写 192 B 快照，设备内复制 192 B；位姿载荷经主机中转 0 B。初次建立布局另上传 12 B。 |
| 物理观察器报告发布状态 | 设备到主机 40 B。原求解器有独立的传输计数，见每段的 `physics_solver`。 |
| 核对 CUDA 与 Vulkan 的输入、输出 | 测试专用的源和结果各读回 192 B，另读取 16 B 着色器统计。第 24 步复用第 0 步的槽，仍能识别新状态。 |
| R1 分箱消费 | 本次用例先将归档 RGB 上传 GPU，每张仅回读两个 64 位区域计数，共 16 B。实际相机到分类器的整条链路及耗时留待后续测量。 |

## 复现

在仓库根目录执行。需要 Windows x64、同一张支持 Win32 外部内存和信号量的 NVIDIA GPU、Vulkan SDK（含 glslang 和验证层），以及可运行本项目 CUDA 后端的 Python 环境。真实物理用例还使用 `mujoco`、`torch`、项目固定版本的 `tyx` 和 `cuda.bindings`。报告路径须尚未创建。

```powershell
$build = 'artifacts/device-interop-repro'
cmake -S rendering/native/win32_interop_p0 -B $build -A x64
cmake --build $build --config Release --target phymeo_vk_p0
$env:PYTHONPATH = 'src;rendering/src;rendering/examples'
python rendering/examples/verify_cuda_vulkan_exchange.py --library "$build/phymeo_vk_p0.dll" --shader "$build/p0_verify.comp.spv" --device 0 --words 513 --validation --report "$build/exchange.json"
python rendering/examples/verify_live_pose_exchange.py --library "$build/phymeo_vk_p0.dll" --shader "$build/p0_verify.comp.spv" --device 0 --validation --report "$build/live-poses.json"
$env:PHYMEO_R1_READER_ARCHIVE = (Resolve-Path 'rendering/results/galaxea-r1/sorting').Path
python -m unittest -v galaxea_r1.test_sorting_device_reader
```

下一批的画面对照先用单相机，再用两个沿相反方向运动的世界。比较同一物理帧上的像素位置、深度、实例 ID 与有效掩码，同时记录位姿传输字节。之后接实时设备图像，再使用八世界十六相机比较端到端 p50/p95。本批没有测量 FPS。[完整计划与验收指标](../../plans/2026-11-24-physics-vulkan-device-interop.md)
