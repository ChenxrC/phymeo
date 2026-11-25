# Phymeo 独立渲染引擎

[渲染快速入门（中文）](docs/quickstart-zh.md)提供启动命令、后台仿真接入、相机查看和模型下载后的行走示例。

[打开引擎架构图](architecture.html) · [画质目标与 R0–R4 路线图](../docs/rendering/roadmap.html)

[开发计划与 agent 分工](plans/README.md)列出七个模块的任务单、合并次序及整机验收方法。接手单个模块从分工表进入，负责集成或签收可直接阅读[上下游合并计划](plans/08-integration.md)和[最终验收规程](plans/09-acceptance.md)。

[独立安装包与验收结果](results/g5/README.md)提供 Windows 上实测的 wheel，以及从安装包生成的三种后端标定桌画面。

这个目录是一个独立 Python 项目，发行名为 `phymeo-rendering`，导入名为 `phymeo_rendering`。基础包仅依赖 Python 3.11+ 标准库，可读取程序场景、规划一次曝光，并校验后端返回的帧包。G3 已接通三条真实离屏路径。可并排查看[标定桌原图、测量结果与复现步骤](results/g3/README.md)。

## 摄影渲染 API

`StudioRenderer` 默认拍摄 1440 × 810 图像。每盏矩形灯使用 16 个采样点（4 × 4），颜色使用 4× MSAA，阴影使用硬件光线查询。工作室 HDR 环境和两块软箱提供照明。默认曝光为 7.0，合成后的线性 RGB 经 `pbr-neutral` 变换输出。运行需要支持 Vulkan ray query 的 GPU、配有对应 SPIR-V 的原生库，以及 `phymeo-rendering[ibl]`。带纹理的场景还需 `textures` 依赖。

```python
from contextlib import closing
from phymeo_rendering import StudioRenderer
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath

renderer = StudioRenderer(library_path=library_path)
with closing(renderer.open_scene(scene)) as session:
    camera = renderer.camera("main", frame.key, camera_pose)
    image = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)))
```

[家庭餐厅资产展示示例](examples/film_home_showcase.py)从本地家居资产包按编号选前十类，各取编号最小的 GLB，在同一间餐厅内运镜 30 秒。影片以 2880×1620 渲染，每灯采样 64 次，开启 4× MSAA，再在线性显示光空间按面积缩到 1440×810。摄影窗景与扫描材质用作布景；焦点随镜头从餐厅全景移到餐桌和玄关，利用渲染深度计算景深。

### 训练与测试相机查看器

**旁观已有仿真。** 在训练或测试进程中创建 `LiveSceneViewer`，在已有物理循环中发布被接受的 `SceneFrame`。用浏览器打开该进程输出的本机地址，即可跟随最新画面。打开第二个标签页、在一个标签页冻结画面、切换相机或关闭网页，均由物理循环继续决定仿真进度。场景名、机位和可用通道由数据源提供。

```python
from phymeo_rendering import LiveSceneViewer, StudioRenderer
from phymeo_rendering.outputs import Channel

# scene 是已有的 VisualScene。两个回调分别根据当前 SceneFrame 返回标定 Camera。
with LiveSceneViewer(
    scene,
    cameras={"wide": wide_camera, "tool": tool_camera},
    camera_labels={"wide": "全景", "tool": "工具相机"},
    channels=(Channel.RGB, Channel.FORWARD_DEPTH, Channel.INSTANCE_ID),
    renderer_factory=lambda: StudioRenderer(library_path=library, device_index=gpu),
    title="操作策略验证",
) as monitor:
    print(monitor.url, flush=True)
    for frame in accepted_scene_frames():
        monitor.publish(frame)
```

这里的 `accepted_scene_frames()` 代表调用方已有的仿真循环。在物理线程采样位姿并构造不可变帧，镜头回调从传入的该帧计算外参。渲染线程只在浏览器请求图像时捕获所选机位和通道。较慢的浏览器可以跳过中间发布，适合在线检查；完整训练记录仍由采集端保存。每个发布进程有自己的 URL，在其他机器上查看时可通过 SSH 将服务的本机端口转发到相同的本地端口。

[启动后台仿真监看示例](examples/open_simulation_monitor.cmd) 演示如何接入两个不同的 MuJoCo 物理场景。它们复用同一网页和原生渲染路径。[完整发布循环](examples/view_background_simulation.py) · [公共 API](src/phymeo_rendering/live_viewer.py)

**回放与手动单步。**
Windows 下双击 [启动查看器](examples/open_training_viewer.cmd) 可在本机浏览器查看八工位回放。这个入口使用本项目已有的 Python 3.13 环境、Vulkan 库和回放资产。在终端运行同一脚本并附加 `--scene panda` 可查看双 Panda 的总览与两个腕部镜头，`--scene eight-live` 可查看现场推进的八个物理环境。用 Ctrl+C 关闭启动窗口中的服务。其他设备可用环境变量 `PHYVIEW_DEVICE`、`PHYVIEW_LIBRARY` 选择显卡和原生库，也可直接运行 [Python 入口](examples/view_training.py)。

默认暂停在总览。数字 0 返回总览，1–8 切换已注册的近景，空格播放或暂停，左右方向键单步。回放可拖动时间轴或输入帧号，现场仿真只支持当前帧和向前单步。RGB、深度、实例 ID 都从选中镜头按需采集。深度使用镜头固定的米制范围映射到灰度，实例使用稳定的颜色编码。训练代码通过原生 `Capture` 读取数值平面。页面上的采集时间包含位姿更新与原生读回，首次切入工位还包含打开该渲染场景的时间。

其他训练任务可将数据源工厂传给 [ViewerServer](src/phymeo_rendering/viewer.py)。工厂创建、`describe()`、`capture(camera_id, frame_index, channel_id)` 和 `close()` 在同一个线程执行。`describe()` 提供机位、通道、帧范围和采样频率；`capture()` 返回显示 PNG、实际帧索引、仿真秒数和采集毫秒数。[三个源的适配示例](examples/_training_view_sources.py) 演示如何校验相机和仿真帧。

### 按设备容量准备场景

在安装有原生 resident Vulkan 后端的环境，`prepare_scene` 查询所选 GPU 和当前着色器的几何、颜色／粗糙度数组、法线数组限额。它在第一次上传前选定一套显示资产，整段回放和所有相机共享。已经满足限额的场景直接使用原对象。原有 `open_scene(scene)` 仍执行严格上传。

```python
# 每个元素是一对绑定同一物理帧的 (SceneFrame, Camera)。
plan = renderer.prepare_scene(
    scene, usage="film", cameras=shot_list,
    preserve_detail=(gripper_node_id,), cache_dir=cache_directory,
)
with closing(renderer.open_scene(plan)) as session:
    for source_frame, source_camera in replay:
        frame, camera = plan.bind_view(source_frame, source_camera)
        rgb = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,)))
```

| 用途 | 资源选择和输出 |
| --- | --- |
| `sensor`（默认） | 要求原始网格、贴图和帧身份；超出预算时报告失败。适用于训练标签和需要与源几何逐像素对照的观测。 |
| `film`、`preview` | 只有超预算时才允许生成派生资源。两者采用相同的容量规则，使用传入机位的真实内参、分辨率和节点位姿分配网格细节。深度和实例轮廓对应本次选定的几何。 |

`preserve_detail` 接受节点、网格、材质或整数实例 ID。选中实例的网格和材质会被保护，复用同一资源的其他实例也保留对应细节。若关注的是夹指轮廓或椅背细杆，可用 `preserve_meshes=(mesh_id,)` 单独锁定网格，木纹贴图仍参与容量分配。刻字、测量标记等需要原始纹理的对象可用 `preserve_materials=(material_id,)`。相机权重使用所给机位中投影包围盒的最大跨度，仅用于分配三角形。遮挡、纹理清晰度和最终轮廓仍需在成片上检查。

`plan.report` 可直接写入 JSON。它列出设备、适配前后的唯一与实例化几何成本、各网格的内容哈希与双向表面距离采样，以及两个独立纹理数组的带边界字节数、每张派生图的原始和选定尺寸。网格与贴图写入派生缓存，源文件保留。影像适配依赖可用 `pip install "phymeo-rendering[prepare]"` 安装。指定的预算、保护对象或图层数无法同时满足时，准备调用会说明哪一项超限。

在 RTX 4060 Laptop 上的三组验收可用[严格模式对照脚本](examples/verify_scene_preparation.py)和[三十件巡游](examples/film_home_collection.py)复现。

| 场景 | 实测结果 |
| --- | --- |
| 十件餐厅，`--case ten` | 含布景 332,777 个三角形。三个机位的 RGB、正向深度、实例 ID、有效掩码与直接打开原场景逐字节相同。 |
| 三十件房间，`--preview` | 含布景从 1,018,749 降到 524,259 个三角形，保留全部 210 个实例。40 层颜色／粗糙度数组从 250.8 降到 127.8 MiB，29 层法线数组原样保留。13 个实际机位覆盖全部 30 件物体。椅背在三个高度都能区分五根竖条和中间四道缝。 |
| 双 Panda，`--case robots` | 120 个实例共享 58 份网格。两个回放状态各采集一次总览和两个腕部视图，四个输出平面与原场景逐字节相同。运动臂更新时，静止臂腕部视图中的自身 ID 与深度保持一致。 |

在装有 NumPy、Pillow 和 OpenCV（`cv2`）的 Python 环境中运行，并将带 libx264 的 FFmpeg 和 ffprobe 放入 PATH。传入 `--assets`、`--cache`、`--output`、`--library` 和 `--device-index` 可复现。加 `--preview` 可输出同机位静帧、轮廓误差和景深参考；加 `--motion-check` 可先录一秒餐桌特写，检查运镜中的稳定性。三个模式可分别指定输出目录。

[三十件家居巡游](examples/film_home_collection.py)使用相同的命令行路径参数，另需 VTK 生成展示网格。一个常驻场景同时加载前三十个类别，镜头在料理台、早餐架、餐桌、玄关和书桌之间移动。它使用深焦与空间超采样保留椅背竖条和小物件边缘。`--preview` 输出章节静帧与三十件物品的带时间索引图；完整模式录制 30 秒，并记录每件模型在抽检镜头中实际覆盖的像素。

高采样影片可用 `StudioRenderer(library_path=library_path, panel_samples=64)`。`size=(宽, 高)` 设置相机默认尺寸，`studio=` 接受自定义灯光。已有标定相机可以直接传给 `CaptureRequest`，保留其尺寸、焦距和主点。`area_light_grid=8` 同样选择每盏灯 64 个样点。[公共接口](src/phymeo_rendering/photography.py)

[Panda 抓取回放与同机位布光对照](results/r1/README.md)展示 R1 的两个归档姿态、固定及腕部相机，以及自研 Vulkan 的局部采集。

[Panda 默认运动预览与实测影片](examples/preview_stable_lighting/README.md)使用逐像素 HDR 漫反射、面积灯软阴影、短程环境遮蔽和轮廓抗锯齿。

[星海图 R1 轮式双臂机器人材质版](results/galaxea-r1/galaxea-r1-studio.mp4)展示三舵轮弧线行驶、停车展臂与夹爪开合。可同帧比较原厂模型配色、海蓝烤漆、缎面连杆与橡胶胎面。[材质前后对照](results/galaxea-r1/finish-comparison.png) · [原配色影片及复现](examples/galaxea_r1/README.md) · [Panda 抗锯齿局部对照](results/film-aa/finger-comparison.png)

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
