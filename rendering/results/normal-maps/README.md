# 材质法线贴图

在[同帧查看器](index.html)中切换“关闭”和“开启”，观察粗钢球的高光与橡胶球的表面纹理。两个视图裁自同一帧，暂停切换时物体和相机保持原位。镜面钢球和木板保留原材质，可作画面中的对照。[并排影片](normal-map-split.mp4) · [钢球无损局部](steel-pair.png) · [橡胶球无损局部](rubber-pair.png)

## 在第 2 秒看什么

从左到右依次是镜面钢、粗钢、黑橡胶和木板。暂停在 **2.00 秒**，先切换粗钢球。原先连成一片的亮带被细条起伏分开；再看同一帧的橡胶球，移动的是较暗的细颗粒高光。镜面钢和木板是两个原地对照。两侧相机、网格、光源、曝光、金属度、粗糙度均相同。这两张法线图分别由条纹与颗粒高度函数求导生成，适合看清着色作用。真实拉丝的各向异性反射和橡胶参数标定属于另外的材质工作。[纹理来源与参数](textures/provenance.json) · [完整采集协议](protocol.json)

三个预定检查帧是首帧、第 60 帧和第 119 帧（从零计）。在这三帧，深度、实例 ID 和有效掩码全幅逐字节相同；镜面钢和木板的可见像素颜色也完全相同。中间帧的粗钢球有 **92.55%** 的可见像素在至少一个通道改变了 2 个 RGB8 码值以上（含 2），橡胶球为 **74.82%**。相应的每像素最大通道差的第 95 百分位分别是 **68** 和 **8**，可用无损局部观察橡胶的较弱变化。这些差值取自编码前的原始画面。[三帧统计](comparison.json)

## 方向、误差与成本

| 检查 | RTX 4060 Laptop 实测 |
| --- | --- |
| 三块方向标定板 | 平面、倾斜 25° 和镜像 UV，四种材质状态、两个光向。24 个预定 24×24 区域对独立光照计算的最大差为 **0 RGB8**。平坦法线图与关闭之间的最大差为 **1 RGB8**。 |
| 镜像手性与绿色通道 | 更换光向，斜向法线图在三块板的有符号亮度差依次为 **+73、+114、−40**。镜像板换成纯 +Y 探针时实测 **+26**，误翻绿色的模型预测 **−27**。 |
| 四秒材质样片 | 每档 5 帧预热，随后 120 帧同步 RGB 读回，另采三个独立几何检查帧。原图 1280×720、1× MSAA、每灯 2×2 采样、硬件阴影查询。单路影片含标题边为 1280×768，分屏为 2562×768。 |
| 同步采集 p50 / p95 | 关闭 **17.09 / 18.95 ms**，开启 **16.33 / 18.42 ms**。两档顺序各测一次，不能从这个差值推断加速。计时包含渲染到 CPU 拥有读回字节，PNG 写盘和视频编码单列。 |
| 当前上下文明示分配的设备内存峰值 | 关闭 **31,225,392 B**，开启 **40,848,480 B**，增加 **9,623,088 B**。两档关闭后设备和主机可见内存计数均归零。 |

[标定图](calibration.png) · [标定数据](calibration.json) · [原始标定包，64 个文件](calibration.zip) · [关闭采集](off/capture.json) · [开启采集](on/capture.json) · [视频帧序与解码](encoding.json) · [查看器同帧切换实测](viewer-check.json)

动态样片和标定使用同一份原生构建。采集包装脚本最后写出的一个收据哈希引用了上一次 MSAA 构建，执行前的构建检查和影片协议均绑定了本次 DLL 与 22 个着色器，原记录与更正一起保留。当前交付在源码分支验证；此前发布的 wheel 对应较早构建。[构建收据](native-build.json) · [原始运行记录](capture-run.json) · [哈希更正](capture-run-correction.json) · [采集源码与集成版本对应关系](film-source-binding.json)

## 在自己的场景中使用

已有带 UV 的网格可以直接绑定法线图，再通过 `NativeBackend(..., resident=True)` 打开场景。颜色贴图、粗糙度图与法线图可以单独声明。下面的材质沿用原钢面的金属度和粗糙度，法线纹理来自本例的解析高度函数。

```python
from hashlib import sha256
from pathlib import Path
from phymeo_rendering.assets import TexturedMaterial, TextureMap

path = Path("textures/steel-normal.png").resolve()  # 以本报告目录为工作目录
steel = TexturedMaterial(
    id="brushed-steel",
    base_color=(0.56, 0.58, 0.60),
    metallic=1.0,
    roughness=0.45,
    normal_map=TextureMap(path.as_uri(), sha256(path.read_bytes()).hexdigest()),
)
```

把 `steel` 放入 `VisualScene.materials`，将目标实例的 `material_id` 设为 `steel.id`，再打开该场景。此输入将 RGB 视为线性法线数据，绿色沿 UV 的 +Y 方向，强度固定为 1。着色器从三角形的世界位置和第一套 UV 重建切线方向。绑定贴图的网格需要有效的 UV 三角形。已烘焙的显式 glTF `TANGENT`、MikkTSpace 对齐及 mip 过滤留待后续阶段。SAPIEN 参考适配器会明确指出已绑定法线图的材料。[材质约定与标定条件](../../plans/2026-11-23-native-normal-maps.md)
