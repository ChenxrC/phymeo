# 近场镜面反射

打开[同帧查看器](index.html)，先暂停在 **0.50 秒**，在钢球和红块之间确认位置关系，再跳到 **2.50 秒**。此时红块已经移出相机右边界。对照两档画面，看哪一颗钢球仍能映出它。[并排影片](near-reflection-split.mp4) · [画内时刻的钢球](in-frame-pair.png) · [画外时刻的钢球](offscreen-pair.png)

## 这条光路来自哪里

相机看见钢球后，着色器沿镜面方向在当前帧的三维场景中查找最近的三角形。命中时使用那一处的材质、纹理及光照计算返回的颜色；射线穿过空处时读取原来的环境。红块可以出现在倒影中，也可以遮住更远的物体。对相机直接看见的钢球像素，深度和实例身份始终属于钢球。[Vulkan 射线查询](https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html) · [本批合同](../../plans/2026-11-23-native-near-reflections.md)

### 在现有场景中启用

```python
from phymeo_rendering.backends.native import NativeBackend

backend = NativeBackend(
    library_path=library_path,
    shader_directory=shader_directory,
    device_index=device_index,
    resident=True,
    shadow_mode="ray_query",
    reflection_mode="single_bounce",
)
session = backend.open_scene(scene)
try:
    exposure = session.capture(frame, request)
finally:
    session.close()
```

上例的 DLL 与编译后的着色器来自同一构建。默认模式为 `environment`。首期按材质上传的 FP32 系数选择发射镜向射线的表面，金属度至少 0.99、粗糙度至多 0.10。粗糙度图继续参与局部材质响应。粗糙物体和哑光物体都能被射线击中；整场都是哑光时，也能使用同一配置，接收倒影的实例计数为零。此模式每个符合条件的可见表面最多计算一次镜面反弹，命中处的照明包括直接光与已启用的环境光。模糊倒影、多次镜面反弹、漫反射间接光仍待开发。

## 平面镜怎样检验位置和遮挡

[平面镜对照图](calibration.png)的左列只读取环境，右列启用近场反射。从上至下依次是红片初始位置、横移后的红片、近处绿片遮住红片，以及所有色片移开。镜面本身确实会挡住照向色片的灯，所以有些颜色较暗。这里测的是最近命中的身份和边界位置。

在 RTX 4060 Laptop 上，19 次公开采集、3 个会话通过验证。红片在画外的标定帧中，直接相机记录到 **0 个红片 ID 像素**，镜中红区有 **546 个像素**严格位于解析轮廓内部。遮挡帧识别到绿 154、红 490、蓝 0 像素，符合前后次序。所有镜面内部像素按独立虚像投影检查，在预定的 1 像素边界带外无身份错误。射线未命中的内部区域与环境档逐字节相同。

两个世界使用不同姿态，改变 A 的应用回放代次后，A 恢复基准图，B 继续显示自己的移动姿态，二者分别与各自参考四通道逐字节相同。三个会话都在验证层开启下关闭，设备与主机可见的当前分配计数归零。[完整标定数据](calibration.json) · [标定原始文件](calibration.zip) · [验证记录](native-build.json)

## 移动红块的实测

两档共用 3 秒、30 Hz 的同一套输入。每档 5 次预热，随后 90 次公开同步采集，每次同时取得 RGB、前向深度、实例 ID 和有效掩码。球体上的测量区域在曝光前，用解析球面反射线与缩小的方块相交来选定。红度定义为红通道减去绿蓝两者较大值，这能把中性白高光和红色倒影区分开来。

| 时刻 | 直接看见的红块 ID 像素 | 镜内预定区域 | 开启近场反射后的平均红度增量 |
| --- | ---: | ---: | ---: |
| 0.50 秒，画内 | 79,170 | 741 像素 | +171.33 RGB8 |
| 2.50 秒，画外 | 0 | 198 像素 | +186.03 RGB8 |

两处区域的每个像素均超过预定的 10 码值红度增量。全程 90 对画面的背景和保守未命中区域颜色完全一致，深度、ID 与有效掩码逐字节相同。此用例直接指定红块位姿，测量对象是镜面光路。

| RTX 4060 Laptop，同步采集 1280×720 四通道 | 只反射环境 | 开启一次近场反射 |
| --- | ---: | ---: |
| p50 | 27.29 ms | 27.94 ms |
| p95 | 30.73 ms | 31.19 ms |
| 上下文明示分配的设备内存峰值 | 27,246,688 B | 27,246,688 B |

以上计时包括同步到主机拥有的四通道字节，图像保存与编码另计。同设备、同场景各运行一次。两档都复用硬件阴影路径的加速结构，每档关闭后当前分配归零。三个影片各 90 帧；全幅几何传感来自未压缩的原始曝光。[逐帧测量](comparison.json) · [采集协议](protocol.json) · [环境档采集记录](environment/capture.json) · [一次反射采集记录](single_bounce/capture.json) · [视频帧与解码](encoding.json)

交付中保存了两档各两个测量时刻的全幅 PNG 和几何数组，以及全部帧的哈希。把画面暂停在画外时刻，可以对照[环境档原帧](environment/frames/0075.png)、[反射档原帧](single_bounce/frames/0075.png)与[预先计算的球面区域](reference/0075.npz)。查看器在 Edge 上实测了两个时刻的跳转、暂停换档和连续播放。[查看器验证](viewer-check.json)

## 构建范围

本批在 Windows 上从源码构建了 DLL 与 28 个 SPIR-V，其中 6 个新增反射变体供硬件查询使用，原来 22 个着色器的文件字节全部一致。设备负控曾在掠射角发现平滑顶点法线会把反射线送进三角面，目前着色仍使用平滑法线，发射时以真实三角面约束朝向和偏移。新路径通过方向灯、纹理 UV、金属度与粗糙度门槛、1×／4× MSAA 和旧库拒绝等针对性检查。此次验证的可用入口是本地源码构建，此前的独立 wheel 保留其原版本。[原生构建与测试记录](native-build.json) · [默认着色器对照](default-shaders.json) · [两份实拍脚本的集成绑定](script-bindings.json)

## 如何复现

在仓库根目录设置 `PYTHONPATH` 指向 `rendering/src`，准备 NumPy、Pillow 及本次原生构建，分别运行下面两个入口。为每次采集指定尚不存在的输出目录。样片用公开的四通道同步读回，编码需要 PATH 中可用的 FFmpeg 与 ffprobe。设备序号应选本机支持 Vulkan 射线查询的 GPU。

```powershell
python rendering/examples/near_reflection_chart.py --library $dll --shaders $shaders --device-index $gpu --validation --output $chartOutput
python rendering/examples/near_reflection_film.py --library $dll --shader-directory $shaders --device-index $gpu --output $filmOutput
```
