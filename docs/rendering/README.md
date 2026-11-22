# Phymeo 渲染模块开发路线

独立开发入口现位于仓库根目录 [rendering/](../../rendering/README.md)。[引擎架构图](../../rendering/architecture.html)对应当前代码骨架。下文保留首轮技术调研；其中建议的包路径以本入口为准，SAPIEN 定位为可选参考后端。

2026-11-22 · 分支 `codex/rendering` · 基线 `codex/exp@761eb304`。打开[可视路线图](roadmap.html)查看 Phymeo 已有画面、桌面操作的美术目标和分阶段验收。图中的机械臂效果意向由 ImageGen 生成。G3／R0 的[三张真实 GPU 画面与测量结果](../../rendering/results/g3/README.md)已发布。下文为首轮调研时的环境和计划记录。

## 选型

首选 **SAPIEN 3.0.3 的纯渲染场景**。Phymeo 负责物理状态；可独立安装的 Vulkan 后端消费一次已接受状态。先把单场景离屏光栅和光追接通，再让训练相机从同一个场景获取深度、实例 ID 和 RGB。高质量回放与训练各自选择采样和读回策略。这样既可以从同一状态比较画质，也能在画质昂贵时保住训练吞吐。

| 调研对象 | 已核实的实现 | 对 Phymeo 的决定 |
| --- | --- | --- |
| LIBERO | 官方依赖固定 robosuite 1.4.0。离屏环境使用 MuJoCo，相机默认包括外部与腕部两个 128×128 视图。 | 借用命名相机、安装位姿和同步观测接口。训练分辨率与演示分辨率独立配置。[依赖](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) · [环境包装](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py) · [实际任务画面](https://arxiv.org/html/2306.03310v2) |
| robosuite 1.5 | 官方给出同一机器人场景的 MuJoCo、Isaac 光追和路径追踪三张对照图，并提供已保存轨迹的再渲染入口。 | 同机位、同姿态和同资产做画质对照；保留原始动作记录。用原图判断算法能改善哪类效果。[三张对照及回放说明](https://robosuite.ai/docs/modules/renderers.html) |
| SAPIEN / ManiSkill | Vulkan 提供多遍延迟光栅、光追与可配置采样；ManiSkill 为策略观测、视频、界面配置不同摄像机。其并行性能报告使用光栅。 | 首选同一后端的两种模式。软阴影、间接光及真实反射用高采样参考图校核；八世界吞吐专项以普通光栅起步。[着色管线](https://sapien-sim.github.io/docs/user_guide/rendering/customize_shaders.html) · [SAPIEN 自身材质对照](https://sapien-sim.github.io/docs/user_guide/rendering/raytracing_renderer.html#reflection-and-refraction) · [相机输出](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html) · [基准范围](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/additional_resources/performance_benchmarking.md) |
| Isaac Sim RTX | 当前文档区分快速呈现、Real-Time 2.0 与高质量路径追踪。实时路径可结合重建技术；完整采样更适合做参考帧。 | 作为高画质参照。在通过基础几何与动态测试前，优先投资可见资产、光照和相机标定。神经重建、插帧、3D Gaussian Splatting 留待有相应数据与收益时评估。[RTX 模式](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/rendering_modes.html) |

选用稳定版 3.0.3，官方提供 Windows x64 CPython 3.11、3.12、3.13 等 wheel。本机查到 RTX 4060 Laptop、8,188 MiB、驱动 566.07。安装、Vulkan 设备选择、单相机光追、设备批量互通均待实测。固定版的批量源码使用 FD 型外部信号量，动态相机 CUDA 位姿缓冲限定于光栅器。这些限制针对特定接口；单相机光追可以先走主机读图。若 Windows 批互通受阻，保留显式复制和串行视图以提供功能，单列复制费用；可另验 Linux 原生 Vulkan/CUDA。若单机光追后端也无法工作，先在 Linux 同版复核，已保存帧可另用离线路径追踪器做参考，产品后端选型随证据更新。[稳定版发行物](https://pypi.org/project/sapien/3.0.3/#files) · [渲染场景源码](https://github.com/haosulab/SAPIEN/blob/3.0.3/python/py_package/wrapper/scene.py) · [批量实现](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/sapien_renderer/batched_render_system.cpp) · [相机实现](https://github.com/haosulab/SAPIEN/blob/3.0.3/src/sapien_renderer/camera_component.cpp)
## 从现有代码接入

- [八世界回放](../../scripts/render_manipulation_gallery.py)已从保存的 CUDA 状态读取 q，并用 MuJoCo 做 FK 和离屏成像。脚本向空场景加入几何类型、尺寸、位姿与 RGBA。图册中已有这种途径的真实样帧。后续第一组实验从封存轨迹读帧，物理步数为零。
- [Panda 资产说明](../../src/phymeo/builders/panda_assets/SOURCES.md)明确记录移除了视觉网格和纹理。[URDF 物理导入器](../../src/phymeo/builders/urdf.py)忽略 visual。新增独立视觉清单，绑定模型、link/body、visual 局部变换、网格、比例、材质、许可和哈希。可从与已用 `franka_ros` 相同提交解析原外观引用，逐项核许可、尺度、关节轴和贴图完整性。R0 用自有程序几何独立验光照。R1 缺件时在展示模式列出缺件，诊断模式才显式显示碰撞代理。
- [异构工作区](../../src/phymeo/backends/mujoco_heterogeneous_io.py)区分已接受的积分终点位姿与该步输入接触；[深度相机](../../src/phymeo/sensors/depth_camera.py)已有针孔、像素中心、轴向与射线深度及稳定碰撞键。沿用这些作为帧契约和几何测量基准；RGB、可见深度与分割由视觉网格计算，碰撞网格的射线结果单列。仅在两者使用相同几何时直接比深度。
- 渲染框架现位于仓库根目录的独立项目 [rendering/](../../rendering/README.md)。物理包与渲染包可分别安装；渲染包的输入适配、原生后端与参考后端均在该项目内开发。

## 每一帧如何生成

1. 静态端建一次表。视觉网格优先 glTF/GLB，保留原始外观变换、UV/切线、法线、纹理色域、透明模式、材质槽及许可。首版只验不透明金属／粗糙度工作流。颜色、发光纹理按 sRGB 解码；金属度、粗糙度、法线按数据读取。共享网格可实例化，每个世界维持稳定实例身份。视觉材质与接触摩擦、刚度使用各自的标识。[Khronos PBR 规范与示例](https://www.khronos.org/gltf/pbr)
2. 动态端按 `world / epoch / revision / tick / phase` 绑定一个已发布帧。刚体变换按世界、link 与 visual 三层合成。显式处理米制尺度和四元数顺序；相机记录标定矩阵、安装位姿、near/far、像素原点。相机位置纹理的图形坐标和 SAPIEN 实体相机轴分别转换。最终输出遵循 Phymeo 的右、上、−Z 前约定。腕部机位采用当帧连杆变换。[SAPIEN 坐标与图像通道](https://sapien-sim.github.io/docs/user_guide/rendering/camera.html)
3. 展示路径用物理材质、经过许可的 HDR 环境光、可控主光与阴影、线性 HDR 计算和固定曝光/色调映射。若后端辅助输出已完成显示变换，适配器记录编码，只转换一次。光追档记录 samples/pixel、反弹、种子、滤噪与显存。锁定机位、材质及灯光看原始采样和降噪版；薄夹指、反射移动、遮挡边缘和 reset 后拖影是必看区域。
4. 传感路径输出同标定的 RGB、米制前向深度、整型实例 ID、有效掩码与内外参。可选射线深度、世界法线。前向深度为相机坐标 −z，射线深度为距离模长；明确近远裁剪与既有射线裁剪在画角边缘的区别。背景转换为声明的无效值。整数 ID 在每个世界内稳定，跨世界用复合键，禁用插值、去噪和调色。RGB 的曝光、随机化与种子存入元数据。
5. 同一帧可交互观看、离屏写图或按已验证能力交给训练。回放使用源时间戳而非视频帧号作为物理时钟。拒绝跨世界、陈旧修订和来源缺失的力叠加；若显示入步力，叠在相应入步姿态并标时相。reset、相机跳切、材质/灯光修订要使路径追踪历史失效。
## 开发顺序与可见验收

以下数值为拟定的第一版验收目标。尚无本机 SAPIEN 性能成绩。性能采集必须写明硬件、驱动、分辨率、材质、每像素采样、通道和读回方式。

| 节点 | 难度／前置 | 可查看的交付 | 通过条件 |
| --- | --- | --- | --- |
| R0 后端可行性 | 中，第一批 | 同一张标定桌的光栅 RGB、光追 RGB、前向深度和 ID；铝、哑光塑料、橡胶三种材料；遮挡物与已知间距。 | 在独立环境固定版本，用仅含渲染系统的场景离屏取图；每档记录实际后端与设备，装机到输出能复现。移动主光后高光与投影响应；无显示器也能输出。每次采集前后物理状态相同，零步进。 |
| R1 外观资产与真实帧 | 高，R0。资产许可清查可与 R0 并行 | Panda＋方块的外观图、同机位碰撞线框开关、两个不同姿态。 | 真实外观网格可追溯；在两姿态验证基座、腕、手指三类锚点，含固定链接、visual 局部偏移与单位。每帧发布的物体与相机对齐；丢资产、错世界、错版本有明确诊断。 |
| R2 实时与参考画质 | 中高，R1 | 同一帧快速 PBR／高采样光追配对图及固定机位 10 秒回放。可看金属反射、塑料高光、阴影半影和间接光。 | 共用网格、相机、灯光与曝光。提供未降噪高采样参考及采样阶梯；固定 ROI 检查噪声随采样增加总体下降；夹指移动无持续残影，改变粗糙度可见高光展宽。首个交互目标 1280×720 单相机 p95 ≤33.3 ms，记录失败配置。若要认定与实物匹配，另拍标定实物对，测曝光、白平衡和尺度。 |
| R3 相机观测 | 高，R1；可与 R2 并行 | 外部＋腕部 RGB、前向深度、实例掩码并排；输出带标定及时间的帧包。 | 640×480 不透明标定场景，0.25–2 m 处带离轴标记；特征投影误差 ≤1 px。相同平面几何、距轮廓 ≥2 px 的有效像素深度最大误差 ≤2 mm；有效区实例 ID 全部一致。翻转、转腕、遮挡、重置和背景均有样例，前景边缘单列。 |
| R4 八世界、回放与发布 | 高，R2＋R3 | 八个异构世界的训练相机张量或显式主机输出，同一轨迹的高质量单世界视频，配置和资源报告。 | 1→2→8 世界各 2 台 320×240 相机，与单世界相同输入逐一比标签和深度；一次 reset 只改变指定世界。分别报告预热后 300 帧 p50/p95、相机帧/秒、显存峰值、同步与复制、编码成本；同步消费完成才计时。未获证的互通写明实际复制路径。光追成片按单场景离线计时。 |

**最小闭环**是 R0 到 R3 交出一段真实 Panda 状态轨迹、两套画质和可信腕部/外部观测。首版模块的完成点是 R4 同一套公共入口通过上述复现。透明折射、多光谱、带噪立体深度、全动态软布顶点流可在首版之后单独扩展。初版的真值深度对应理想针孔可见表面。

**分开处理三种失败。** 标定错误先修单位、坐标和帧绑定；光照不对先查材质、灯光、纹理色域与曝光；吞吐不够先分开统计同步、拷贝、光栅和读回。每一种都有独立的标定台案例，物理求解器无需为了画面调整。R0 若只有一种后端模式成功，保留该模式的样图和失败日志，优先解决另一条路径的首个可复现问题。

## 路线图图像出处

实际画面取自原保留的 `artifacts/mujoco-tyx-manipulation-control-20261114-01/video-02/preview.png`，其八世界诊断任务、原始帧及相机由旧实验确定。页面嵌入其缩略图以便离线查看。桌面机械臂是本次生成的独立美术意向，构图与模型均不同；后续画质比较统一使用 R0/R2 的同场景实渲图。
