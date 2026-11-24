# 星海图 R1 · 三舵轮与双臂回放

[观看 8 秒影片](../../results/galaxea-r1/galaxea-r1.mp4)。开场用地面半米刻度交代底盘行驶，约 3.6 秒切至双臂和夹爪，约 6.3 秒切至后侧全景。两个行驶段合计 1.14 米，车身累计转向约 20°。每个舵轮的偏角和滚动量分别由该轮的位置、滚动半径和底盘曲率求出。

场景使用原版 R1 的官方 URDF、34 个视觉链接及声明的颜色。影片沿预设关节轨迹做正向运动学回放，展示范围是底盘、躯干、双臂与夹爪的协调运动。灯光来自车间 HDR 环境和两盏矩形灯。公开模型缺少实测光学参数，画面采用统一的非金属表面和 0.65 粗糙度。[模型来源与几何转换](SOURCES.md) · [环境贴图署名](../materials/NOTICE.md)

影片由 Phymeo Vulkan 在 RTX 4060 Laptop GPU 上以 2560×1440 着色，线性合成为 1280×720。H.264 文件经解码计数为 240 帧，30 FPS，8.00 秒。Vulkan 验证错误和会话关闭后的存活资源均为零。[本机采集记录](../../../artifacts/rendering/film-aa-r1/r1-film/render.json) · [逐帧关节、相机与连杆位姿](../../../artifacts/rendering/film-aa-r1/r1-film/trajectory.json)

## 复现

先按上面的模型来源说明检出固定版本，并准备已启用轮廓采样的[稳定 Vulkan 渲染包](../preview_stable_lighting/README.md)。环境需要 NumPy、Pillow、FFmpeg 和 ffprobe，新建网格缓存还需 fast-simplification 0.2.0。以下命令从仓库根目录运行，按本机 Vulkan 枚举选择设备序号。模型缓存可重复使用，每次拍摄指定新的输出目录。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
python -m pip install fast-simplification==0.2.0
python -B -m galaxea_r1.film --asset-dir artifacts/rendering/galaxea-r1-assets --cache-dir artifacts/rendering/galaxea-model-cache --output artifacts/rendering/my-r1-movie --samples grid2 --device-index 1 --validation
```

加入 `--still 143` 可单独出双臂近景。使用 `--kinematics-only` 可在无 GPU 时导出完整轨迹和相机。运行 `python -m unittest galaxea_r1.test_model galaxea_r1.test_film` 可独立检查模型身份、关节变换、轮胎接地点的无滑移运动学，以及停车期间的双臂和夹爪行程。
