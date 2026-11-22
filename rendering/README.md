# Phymeo 独立渲染引擎

[打开引擎架构图](architecture.html) · [画质目标与 R0–R4 路线图](../docs/rendering/roadmap.html)

[开发计划与 agent 分工](plans/README.md)列出七个模块的任务单、合并次序及整机验收方法。接手单个模块从分工表进入，负责集成或签收可直接阅读[上下游合并计划](plans/08-integration.md)和[最终验收规程](plans/09-acceptance.md)。

这个目录是一个独立 Python 项目，发行名为 `phymeo-rendering`，导入名为 `phymeo_rendering`。当前交付包含场景和相机的数据类型、后端协议、原生管线的阶段名称。原生设备层、着色器与 SAPIEN 适配器均待实现。基础包仅依赖 Python 3.11+ 标准库。

## 七个板块

| 板块 | 开发入口 | 谁负责什么 |
| --- | --- | --- |
| 外观资产 | [assets.py](src/phymeo_rendering/assets.py) | `Mesh` 引用以米为单位的网格，`Material` 描述不透明表面的线性底色、金属度、粗糙度。后续在此定义纹理、UV、色域与来源元数据。 |
| 场景与帧 | [scene.py](src/phymeo_rendering/scene.py) | `VisualScene` 保存静态资产、实例、节点绑定和方向光。`SceneFrame` 只发布这一帧的节点世界位姿、来源标识和时间。多个实例可共享同一网格。 |
| 标定相机 | [cameras.py](src/phymeo_rendering/cameras.py) | `Camera` 表示一次曝光，包含内参、已解析的世界位姿、分辨率、裁剪面与帧标识。固定机位与腕部机位共用这一类型。 |
| 渲染路径与阶段 | [pipeline.py](src/phymeo_rendering/pipeline.py) | `CaptureRequest` 选择相机、光栅或路径追踪、所需通道。`PassPlanner` 是原生后端的扩展点，将请求翻译成有序阶段，例如可见性、阴影、光照或光线积分，最后按通道导出。 |
| 后端与图形设备 | [公开后端合同](src/phymeo_rendering/backends/__init__.py) · [原生](src/phymeo_rendering/backends/native.py) · [SAPIEN](src/phymeo_rendering/backends/sapien.py) | 两种后端通过同一合同打开场景和拍摄。自研路线在此拥有网格上传、缓冲区、纹理、着色器、命令提交、屏障及读回；拟首先实现 Vulkan。SAPIEN 路线把整次拍摄交给它自己的设备与管线，用于首图和对照。 |
| 图像输出 | [outputs.py](src/phymeo_rendering/outputs.py) | `Capture` 打包标定、来源帧和各个 `ImagePlane`。当前约定主机拥有的连续字节，分别表示 sRGB、米制前向深度、整数实例 ID 或有效掩码。查看器、文件编码器和训练消费端从这里取结果。 |
| 外部数据适配 | [adapters/](src/phymeo_rendering/adapters/__init__.py) | `SceneSource` 提供静态清单及动态帧序列。程序创建的桌面、保存的运动轨迹、Phymeo 已发布的物理状态都是候选来源；轴系、单位、连杆与视觉资产映射由对应适配器处理。 |

## 贯穿各板块的一次拍摄

1. 调用方从数据源取得 `VisualScene`，用所选后端的 `open_scene(scene)` 建立会话。资产身份、实例绑定或灯光改变时更新静态清单及其 `key`。
2. 数据源提供一个 `SceneFrame`。帧键记录静态场景键、世界、重置代次、发布修订、tick 和相位。将节点世界位姿与实例局部位姿组合，可得到网格世界位姿。对于腕部摄像机，用同帧手腕位姿与安装外参得到曝光位姿。
3. 调用方把此次曝光和所需通道组成 `CaptureRequest`。会话用 `supports(request)` 检查请求的完整组合；`capture(frame, request)` 应核对场景键和曝光帧键，成功时返回同源的 `Capture`。未实现的能力必须显式拒绝。消费完毕后调用 `close()` 释放该会话持有的资源。

自研后端内部可为光栅组织「可见性 → 阴影 → 材质光照」，为路径追踪组织「加速结构准备 → 射线积分 → 累积」。显示变换只作用于颜色。深度按相机朝前方向计量，缺失值为正无穷；物体 ID 为正整数，背景为零。具体的调度、时域历史、设备帧所有权与同步都属于后续实现。

## 骨架使用示例

以下代码可在独立安装后运行，用来描述程序创建的一次曝光。打开场景和图像生成需要后续的具体后端。

```python
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame

key = FrameKey("table-v1", "studio", 0, 0, 0, "authored")
frame = SceneFrame(key, time_s=0.0, nodes=())
camera = Camera(
    id="front", frame_key=frame.key, world_pose=Pose(),
    width=640, height=480, fx_px=500.0, fy_px=500.0,
    cx_px=320.0, cy_px=240.0, near_m=0.1, far_m=5.0,
)
request = CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB, Channel.FORWARD_DEPTH))
```

从仓库根目录可用 `python -m pip install ./rendering` 单独安装本包。首个实现节点沿用 R0 的标定桌。先在单个世界把静态资产、一次曝光、真实图像闭合，再在同一输入上比较原生后端与参考后端。
