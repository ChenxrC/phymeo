# 星海图 R1 · 三舵轮与双臂回放

[观看海蓝烤漆与缎面金属版](../../results/galaxea-r1/galaxea-r1-studio.mp4) · [与原配色同帧对照](../../results/galaxea-r1/finish-comparison.png) · [原版 URDF 配色影片](../../results/galaxea-r1/galaxea-r1.mp4)

影片长 8 秒。开场用地面半米刻度交代底盘行驶，约 3.6 秒切至双臂和夹爪，约 6.3 秒切至后侧全景。两个行驶段合计 1.14 米，车身累计转向约 20°。每个舵轮的偏角和滚动量分别由该轮的位置、滚动半径和底盘曲率求出。

场景使用原版 R1 的官方 URDF 和 34 个视觉链接。影片沿预设关节轨迹做正向运动学回放，展示底盘、躯干、双臂与夹爪的协调运动。灯光来自车间 HDR 环境和两盏矩形灯。原配色版本保留 URDF 声明的颜色，所有链接使用非金属表面和 0.65 粗糙度。[模型来源与几何转换](SOURCES.md) · [环境贴图署名](../materials/NOTICE.md)

### 材质设计

摄影棚版使用为展示设计的 PBR 参数。胸壳和底盘上色为海蓝烤漆，双臂用象牙白漆，腰部和腕部搭配钛色、石墨色与小面积香槟金属。涂层的宽高光、金属的环境反射和夹爪聚合物的哑光都由同一组灯光产生。公开模型没有提供实机表面测量值。

三只车轮只在原网格外半径 70 mm 的连续胎圈使用深炭色橡胶，轮毂使用较浅的涂层。头部贴图按 CAD 中心距 120 mm 的两处镜孔定位，分别控制镜窗、边圈和壳体的底色与粗糙度。贴图坐标绑定在零件上，保留原顶点位置、法线、三角形与实例 ID。两版使用相同的相机、关节轨迹、HDR、曝光和四子像素抗锯齿。[两版比对记录](../../results/galaxea-r1/finish-comparison.json)

影片由 Phymeo Vulkan 在 RTX 4060 Laptop GPU 上以 2560×1440 着色，线性合成为 1280×720。H.264 文件经解码计数为 240 帧，30 FPS，8.00 秒。Vulkan 验证错误和会话关闭后的存活资源均为零。[本机采集记录](../../../artifacts/rendering/film-aa-r1/r1-film/render.json) · [逐帧关节、相机与连杆位姿](../../../artifacts/rendering/film-aa-r1/r1-film/trajectory.json)

## 复现

先按上面的模型来源说明检出固定版本，并准备已启用轮廓采样的[稳定 Vulkan 渲染包](../preview_stable_lighting/README.md)。环境需要 NumPy、Pillow、FFmpeg 和 ffprobe，新建网格缓存还需 fast-simplification 0.2.0。以下命令从仓库根目录运行，按本机 Vulkan 枚举选择设备序号。模型缓存可重复使用，每次拍摄指定新的输出目录。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
python -m pip install fast-simplification==0.2.0
python -B -m galaxea_r1.film --asset-dir artifacts/rendering/galaxea-r1-assets --cache-dir artifacts/rendering/galaxea-model-cache --output artifacts/rendering/my-r1-movie --samples grid2 --device-index 1 --validation
```

将拍摄命令加上 `--finish studio` 并指定新的输出目录，即可生成材质版。默认值 `--finish urdf` 用于复现原版。两个完整输出目录可以交给 `python -B -m galaxea_r1.compare_finishes --before <原输出目录> --after <材质版输出目录> --output <新对照目录>`，生成同帧对照图并核对全部 240 帧的输入。

加入 `--still 143` 可单独出双臂近景。使用 `--kinematics-only` 可在无 GPU 时导出完整轨迹和相机。运行 `python -m unittest galaxea_r1.test_model galaxea_r1.test_film` 可独立检查模型身份、关节变换、轮胎接地点的无滑移运动学，以及停车期间的双臂和夹爪行程。
