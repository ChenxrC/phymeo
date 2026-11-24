# 星海图 R1 · 三舵轮与双臂回放

## 装配车间的 A → B 取放

[观看 10 秒操作影片](../../results/galaxea-r1/r1-workshop-transfer.mp4) · [车间全景](../../results/galaxea-r1/workshop-overview.png) · [夹取、搬运、松开三帧对照](../../results/galaxea-r1/workshop-operation.png)

在 04 号工位，R1 从 A 治具夹起橙色三孔工件 P4，抬高 145 mm，平移约 311 mm 后落到 B 治具。约在 3.7 秒夹紧、5.6 秒经过两治具之间、8.4 秒松开。夹爪开始接近前先拍车间全景，此后三个动作共用同一个固定近景。台面、工具、料架、工位铭牌和混凝土地面均为场景内的几何或材质。

这段操作采用 R1 官方 URDF 的关节限位与逆运动学。闭合间距按夹指网格和 56 mm 宽工件的倒角标定。夹持期间由夹爪位姿带动工件，松开后工件停在承托面。摩擦力、夹持载荷和滑移仍需动力学验证。

两盏矩形面积灯与车间 HDR 环境共同照明。第 167 帧另做了 720p 固定机位对照，只移走悬空工件，再放回同一姿态。相同实例、相同中心深度的台面与治具区域内，7,360 个像素的主光项及总线性亮度均因遮挡下降至少 1%。工件复位后，显示图与线性图分别逐字节还原。影片用自研 Vulkan 在 RTX 4060 Laptop GPU 上渲染，四个子像素线性合成 720p，H.264 解码为 300 帧、30 FPS。[阴影对照记录](../../results/galaxea-r1/workshop-shadows.json) · [影片与运动检查记录](../../results/galaxea-r1/workshop-render.json)

使用下方的模型来源与环境配置，从仓库根目录运行：

```powershell
python -B -m galaxea_r1.workshop --asset-dir artifacts/rendering/galaxea-r1-assets --cache-dir artifacts/rendering/galaxea-model-cache --output artifacts/rendering/my-workshop-video --device-index 1 --validation --verify-shadows
```

## 行驶与双臂展示

[观看手臂与头罩亮面版](../../results/galaxea-r1/galaxea-r1-polished.mp4) · [亮面与上一版局部对照](../../results/galaxea-r1/polished-comparison.png) · [海蓝烤漆与缎面金属版](../../results/galaxea-r1/galaxea-r1-studio.mp4) · [原版 URDF 配色影片](../../results/galaxea-r1/galaxea-r1.mp4)

影片长 8 秒。开场用地面半米刻度交代底盘行驶，约 3.6 秒切至双臂和夹爪，约 6.3 秒切至后侧全景。两个行驶段合计 1.14 米，车身累计转向约 20°。每个舵轮的偏角和滚动量分别由该轮的位置、滚动半径和底盘曲率求出。

场景使用原版 R1 的官方 URDF 和 34 个视觉链接。影片沿预设关节轨迹做正向运动学回放，展示底盘、躯干、双臂与夹爪的协调运动。灯光来自车间 HDR 环境和两盏矩形灯。原配色版本保留 URDF 声明的颜色，所有链接使用非金属表面和 0.65 粗糙度。[模型来源与几何转换](SOURCES.md) · [环境贴图署名](../materials/NOTICE.md)

### 材质设计

摄影棚版使用为展示设计的 PBR 参数。胸壳和底盘上色为海蓝烤漆，双臂用象牙白漆，腰部和腕部搭配钛色、石墨色与小面积香槟金属。涂层的宽高光、金属的环境反射和夹爪聚合物的哑光都由同一组灯光产生。公开模型没有提供实机表面测量值。

亮面版沿用摄影棚版的配色。双臂与夹爪背部的白色漆面粗糙度从 0.41 降到 0.24，头罩从 0.40 降到 0.23。镜窗和镜圈仍使用各自的粗糙度。为了让转动时的高光顺着壳体曲面移动，九个外壳部件在连续、同向的相邻三角面之间按角度加权计算着色法线。24° 以上的折边、镜窗纹理接缝和独立零件各自保留边界。第 143 帧的头罩上沿、前臂圆弧最便于对照。[同帧对照记录](../../results/galaxea-r1/polished-comparison.json)

在近景的 81 帧中，分别跟踪头罩和前臂上的九个可见表面点。以每组点的平均显示亮度计算，80 对相邻帧的亮度变化 P95 分别为 0.33 和 1.17 个 8 位色阶。两个取样区域均未出现达到 8 个色阶、下一帧又恢复的孤立尖峰。[运动检查数据](../../../artifacts/rendering/r1-polished/motion-check.json)

三只车轮只在原网格外半径 70 mm 的连续胎圈使用深炭色橡胶，轮毂使用较浅的涂层。头部贴图按 CAD 中心距 120 mm 的两处镜孔定位，分别控制镜窗、边圈和壳体的底色与粗糙度。贴图坐标绑定在零件上，保留原顶点位置、法线、三角形与实例 ID。两版使用相同的相机、关节轨迹、HDR、曝光和四子像素抗锯齿。[两版比对记录](../../results/galaxea-r1/finish-comparison.json)

影片由 Phymeo Vulkan 在 RTX 4060 Laptop GPU 上以 2560×1440 着色，线性合成为 1280×720。H.264 文件经解码计数为 240 帧，30 FPS，8.00 秒。Vulkan 验证错误和会话关闭后的存活资源均为零。[本机采集记录](../../../artifacts/rendering/film-aa-r1/r1-film/render.json) · [逐帧关节、相机与连杆位姿](../../../artifacts/rendering/film-aa-r1/r1-film/trajectory.json)

## 复现

先按上面的模型来源说明检出固定版本，并准备已启用轮廓采样的[稳定 Vulkan 渲染包](../preview_stable_lighting/README.md)。环境需要 NumPy、Pillow、FFmpeg 和 ffprobe，新建网格缓存还需 fast-simplification 0.2.0。以下命令从仓库根目录运行，按本机 Vulkan 枚举选择设备序号。模型缓存可重复使用，每次拍摄指定新的输出目录。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
python -m pip install fast-simplification==0.2.0
python -B -m galaxea_r1.film --asset-dir artifacts/rendering/galaxea-r1-assets --cache-dir artifacts/rendering/galaxea-model-cache --output artifacts/rendering/my-r1-movie --samples grid2 --device-index 1 --validation
```

为拍摄命令加上 `--finish polished` 可生成亮面版，`--finish studio` 使用缎面版，默认值 `--finish urdf` 复现原版。每次拍摄指定新的输出目录。使用 `python -B -m galaxea_r1.compare_finishes --before <之前的完整输出目录> --after <新版完整输出目录> --output <新对照目录>` 生成同帧对照图并核对全部 240 帧。支持原配色与缎面、缎面与亮面两种比较。

加入 `--still 143` 可单独出双臂近景。使用 `--kinematics-only` 可在无 GPU 时导出完整轨迹和相机。运行 `python -m unittest galaxea_r1.test_model galaxea_r1.test_film` 可独立检查模型身份、关节变换、轮胎接地点的无滑移运动学，以及停车期间的双臂和夹爪行程。
