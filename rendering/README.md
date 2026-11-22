# Phymeo 独立渲染引擎

[打开引擎架构图](architecture.html) · [画质目标与 R0–R4 路线图](../docs/rendering/roadmap.html)

[开发计划与 agent 分工](plans/README.md)列出七个模块的任务单、合并次序及整机验收方法。接手单个模块从分工表进入，负责集成或签收可直接阅读[上下游合并计划](plans/08-integration.md)和[最终验收规程](plans/09-acceptance.md)。

这个目录是一个独立 Python 项目，发行名为 `phymeo-rendering`，导入名为 `phymeo_rendering`。基础包仅依赖 Python 3.11+ 标准库，可读取程序场景、规划一次曝光，并校验后端返回的帧包。G3 已接通三条真实离屏路径。可并排查看[标定桌原图、测量结果与复现步骤](results/g3/README.md)。

[Panda 抓取回放与同机位布光对照](results/r1/README.md)展示 R1 的两个归档姿态、固定及腕部相机，以及自研 Vulkan 的局部采集。

[观看 Panda 抓取视频](results/panda-movies/README.md)，包括摄影棚全景、同步布光对比、木纹工作台全景与近景、同一方块的红漆与铜色金属对比。

[打开本地影片页](results/index.html)。用同一段十秒真实抓放切换车间日光、双面积灯和摄影棚材质；点“搬运方块”可直接看夹指、金属反射和桌面投影。另有八世界同步回放与推 T 近景。[Panda 画质及延迟](results/r2/README.md) · [训练相机](results/r3/README.md) · [八世界资源和独跑对照](results/r4/README.md)

## 七个板块

G1 输入合同已接通。仓库根目录下设置 `$env:PYTHONPATH=(Resolve-Path rendering/src).Path`，运行 `python -m unittest discover -s rendering/tests -v` 可验证资产声明、场景引用、位姿组合和相机曝光。两帧使用同一张[自制标定桌](tests/fixtures/g1/README.md)，在无图形设备环境下即可检查腕部转动前后的解析投影。同一份夹具也用于 R0 真机采集。

G2 把同一张标定桌的两帧交给拍摄会话。请求可以只取深度或实例 ID，也可以连同 RGB 和有效掩码一起取。`ValidatedRenderSession` 检查实际返回帧的相机、时间、通道顺序和场景实例身份。[集成测试](tests/integration/test_backend_contract.py)用两个人工像素检查读回和发布流程；R0 再用同一公共接口检查完整设备帧。

| 板块 | 开发入口 | 谁负责什么 |
| --- | --- | --- |
| 外观资产 | [assets.py](src/phymeo_rendering/assets.py) | `Mesh` 引用以米为单位的网格，`Material` 描述不透明表面的线性底色、金属度、粗糙度。`TexturedMaterial` 按 TEXCOORD_0 读取带内容哈希的本地 sRGB 贴图。 |
| 场景与帧 | [scene.py](src/phymeo_rendering/scene.py) | `VisualScene` 保存静态资产、实例、节点绑定和方向光。`SceneFrame` 只发布这一帧的节点世界位姿、来源标识和时间。多个实例可共享同一网格。 |
| 标定相机 | [cameras.py](src/phymeo_rendering/cameras.py) | `Camera` 表示一次曝光，包含内参、已解析的世界位姿、分辨率、裁剪面与帧标识。固定机位与腕部机位共用这一类型。 |
| 渲染路径与阶段 | [pipeline.py](src/phymeo_rendering/pipeline.py) | `CaptureRequest` 选择相机、光栅或路径追踪、所需通道。`PassPlanner` 是原生后端的扩展点，将请求翻译成有序阶段，例如可见性、阴影、光照或光线积分，最后按通道导出。 |
| 后端与图形设备 | [公开后端合同](src/phymeo_rendering/backends/__init__.py) · [原生](src/phymeo_rendering/backends/native.py) · [SAPIEN](src/phymeo_rendering/backends/sapien.py) | 两种后端通过同一合同打开场景和拍摄。自研路线在此拥有网格上传、缓冲区、纹理、着色器、命令提交、屏障及读回；当前实现为 Vulkan 离屏光栅。SAPIEN 路线把整次拍摄交给它自己的设备与管线，用于首图和对照。 |
| 图像输出 | [outputs.py](src/phymeo_rendering/outputs.py) | `Capture` 打包标定、来源帧和各个 `ImagePlane`。当前约定主机拥有的连续字节，分别表示 sRGB、米制前向深度、整数实例 ID 或有效掩码。查看器、文件编码器和训练消费端从这里取结果。 |
| 外部数据适配 | [adapters/](src/phymeo_rendering/adapters/__init__.py) | `SceneSource` 提供静态清单及动态帧序列。程序创建的桌面、保存的运动轨迹、Phymeo 已发布的物理状态都是候选来源；轴系、单位、连杆与视觉资产映射由对应适配器处理。 |

## 贯穿各板块的一次拍摄

1. 调用方从数据源取得 `VisualScene`，用所选后端的 `open_scene(scene)` 建立会话。资产身份、实例绑定或灯光改变时更新静态清单及其 `key`。
2. 数据源提供一个 `SceneFrame`。帧键记录静态场景键、世界、重置代次、发布修订、tick 和相位。将节点世界位姿与实例局部位姿组合，可得到网格世界位姿。对于腕部摄像机，用同帧手腕位姿与安装外参得到曝光位姿。
3. 调用方把此次曝光和所需通道组成 `CaptureRequest`。会话用 `supports(request)` 检查请求的完整组合；`capture(frame, request)` 应核对场景键和曝光帧键，成功时返回同源的 `Capture`。未实现的能力必须显式拒绝。消费完毕后调用 `close()` 释放该会话持有的资源。

自研后端内部可为光栅组织「可见性 → 阴影 → 材质光照」，为路径追踪组织「加速结构准备 → 射线积分 → 累积」。显示变换只作用于颜色。深度按相机朝前方向计量，缺失值为正无穷；物体 ID 为正整数，背景为零。当前两种光栅都以主机拥有的字节发布传感通道；SAPIEN 路径追踪发布 RGB。设备张量互通和跨帧累积留到后续验证。

## 从程序场景规划一次曝光

以下示例可以直接在基础包上运行。它读取一个程序创建的帧，打印一份原生光栅的阶段表。

```python
from phymeo_rendering.adapters import AuthoredSceneSource
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, DefaultPassPlanner, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame, VisualScene

key = FrameKey("table-v1", "studio", 0, 0, 0, "authored")
frame = SceneFrame(key, time_s=0.0, nodes=())
source = AuthoredSceneSource(VisualScene("table-v1", (), (), ()), (frame,))
exposure_frame = next(source.frames())
camera = Camera(
    id="front", frame_key=exposure_frame.key, world_pose=Pose(),
    width=640, height=480, fx_px=500.0, fy_px=500.0,
    cx_px=320.0, cy_px=240.0, near_m=0.1, far_m=5.0,
)
request = CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB, Channel.FORWARD_DEPTH))
print([stage.value for stage in DefaultPassPlanner().plan(request)])
```

从仓库根目录可用 `python -m pip install ./rendering` 单独安装本包。安装可选参考端和编译自研 Vulkan 后端的命令见[R0 复现说明](results/g3/README.md)。下一批 G4 接入真实 Panda 回放和可追溯的视觉资产。
