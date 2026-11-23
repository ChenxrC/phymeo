# Panda 近场漫反射运动预览

这条可选的 Vulkan 路径适合交互观看。[实拍影片、原图和性能记录](../../results/lighting-preview/2026-11-24/README.md) 使用 RTX 4060 Laptop、1280×720 和真实 Panda 运行记录。公有捕获接口仍返回已完成的、CPU 自有的图像。

## 构建及拍摄

以下 PowerShell 在仓库根目录运行。使用项目现有 Python 3.11 环境（NumPy、Pillow）和 Visual Studio 2022 x64、CMake、Vulkan SDK。编码需要 PATH 上的 ffmpeg 与 ffprobe。输出位置须为新目录。示例中的设备序号为这台双显卡机器的 RTX 4060；其他机器选中支持 Vulkan ray query 的实际设备。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
$runDir = Join-Path $PWD 'artifacts/rendering/local-motion-demo'
python -B rendering/examples/preview_nearfield/prepare.py --output "$runDir/packages"
```

脚本按顺序建立当前帧稀疏着色、不透明几何查询、紧凑光栅和近场光照四个包。编译器或 SDK 采用便携目录时，可向脚本显式传入 `--compiler`、`--vulkan-include`、`--vulkan-library`。生成的 `packages/local/source-manifest.json` 绑定实际 DLL、着色器和米制半径。原项目的 Vulkan 光栅 DLL 与它的完整 SPIR-V 目录用于摄影棚工具的基线身份检查；把下面 `$baseline` 指向该已构建的目录。

```powershell
$baseline = Resolve-Path 'artifacts/rendering/composite-diagnosis/build'
python -B rendering/examples/preview_nearfield/validate.py --prepared "$runDir/packages/local" --device-index 1 --output "$runDir/rooms"
python -B rendering/examples/preview_sparse_motion/run.py capture --candidate-backend compact --runtime rendering/examples/preview_compact/runtime.py --prepared "$runDir/packages/local" --baseline-library "$baseline/phymeo_native.dll" --baseline-shaders "$baseline" --device-index 1 --candidate-only --channels rgb --rounds 1 --camera sequence --studio-look soft-studio --film sparse --output "$runDir/film"
python -B rendering/examples/preview_sparse_motion/run.py encode "$runDir/film" --media-output "$runDir/media"
```

正式配对性能检验使用相同 `capture` 入口，选录制灯光与固定细节机位（默认），移除 `--candidate-only`、`--camera sequence` 与 `--studio-look soft-studio`，使用 `--channels rgb all --rounds 2`。每轮按真实时间顺序采集 300 个姿态，两轮反转配置的测试顺序。10 帧预热、完整开关及逐帧时序保存在目录内。运行既有资产回放需要工作区具备登记的 Panda 轨迹及资产，来源见[回放工具](../preview_sparse_motion/README.md)。

## 图像如何生成

主表面直接光、环境镜面和近场镜面保留 1280×720。对间接漫反射，320×180 的光栅在每个可见锚点使用 4 个方向提案；在 K25 邻域内以当前表面的身份、覆盖、深度和着色法线决定哪些样本可共享。接收点使用自己的材质响应。主和次表面在这个速度档均使用 2×2 面光源积分，近场镜面和几何传感不受漫反射半径控制。

对实际命中的方向，距离为 `d` 米，局部反射权重为 `1-smoothstep(0,R,d)`，其余权重交给环境估计。丢失目标或到达 `R=0.12 m` 时全由远场环境负责。因而近处红色墙面会照红相邻表面，远处墙面遮光或反射会产生系统偏差。该配置使用各帧独立的空间重建。任何薄遮挡显露区的误差仍须与真实高样本对照检查。

`validate.py` 在四间实测房间读取低分辨率线性辐亮度和真实几何通道，检验开口/远黑遮挡、近黑遮挡、近红墙的作用。第 60 帧比较图由 `compare_pose.py` 从真实 native PNG 制作；脚本核对时间、状态、相机和 PNG/解码像素哈希，局部按原像素展示。
