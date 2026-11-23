# 从红黑颗粒到连续的夹爪暗面

打开[同一机位的实测对照](index.html)，先看两根指尖的背光面。左图保留第一批 16 样本，右图可选择 64、256 或 1024 样本。每批都从当前真实几何重新取样，显示时使用同样的曝光、PBR Neutral 和 sRGB 编码。切换「移动后的旧影」可以把运动后的一帧与相同新状态的独立重开逐帧对照。

## 原图为什么呈现红黑颗粒

上一批把直接光固定在夹爪另一侧，关闭环境填充，以便看清一块红板能传回多少间接光。原图两指选区的 4,120 个像素在关闭间接光时全部为黑。开启后，每个像素按余弦分布选 16 个方向，沿当前三角形求交。射线命中受光的红板会带回明显的红色，打到暗处或漏向黑色环境时贡献很小。相邻像素命中亮板的次数不同，由此产生高对比度的颗粒。种子、机位和输入都固定时，重复曝光仍用同一批方向。扩大图像会让这些像素更显眼。[上批实验原图与测量](../diffuse-bounce/README.md) · [旧单帧采样实现](../../src/phymeo_rendering/backends/_shaders/resident_indirect.glsl) · [PBRT 对间接光采样方差的说明](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)

新的累积模式在 GPU 浮点附件中保存未曝光的线性平均光能。每次静止曝光使用下一批方向，先更新平均值，再应用显示变换。固定场景下，增加独立样本会降低抽样噪声。增加路径深度则可以接收到多经过一处表面的光，两种设置分别测试。MSAA 处理的几何覆盖画在另一条采样路径上。

## 调用入口

```python
from phymeo_rendering.backends.native import NativeBackend

renderer = NativeBackend(
    library_path=dll, shader_directory=shaders, device_index=gpu,
    resident=True, shadow_mode="ray_query", msaa_samples=1,
    indirect_mode="diffuse_bounce", indirect_bounces=2,
    indirect_seed=0, progressive=True,
)
session = renderer.open_scene(scene)
try:
    for _ in range(16):
        image = session.capture(frame, request)  # request 包含 RGB
    mean = session.read_linear_mean()            # RGBA32F，显示变换之前
finally:
    session.close()
```

每批 16 条主路径，第二跳为每条可继续的路径另采一个方向。最高 4096 批，达到上限可调用 `session.reset_accumulation()` 显式重置。只读浮点均值或请求独立几何通道不会增加样本。移动相机、改变节点位姿或调用 `session.set_directional_lights(lights)` 改变有效方向灯，下一次 RGB 曝光从首批开始。会话按自己的世界、epoch 和相机状态管理历史。

## 同一只夹爪

下表使用相同的 4,120 个指面像素和四个独立种子。噪声是场景线性 RGB 的种子间 RMS。黑像素为两指选区内显示 RGB 全零的数量，图中使用种子 0。[原始读数](panda/noise.json) · [原尺寸左右对照](panda-convergence.png)

| 每像素累计样本 | 线性噪声 | 相对首批 | 种子 0 的黑像素 |
| ---: | ---: | ---: | ---: |
| 16 | 0.011993 | 100% | 452 |
| 64 | 0.006085 | 50.74% | 0 |
| 256 | 0.003055 | 25.48% | 0 |
| 1024 | 0.001520 | 12.67% | 0 |

256 至 1024 样本的选区平均 RGB 相对变化为 0.0811%。在 RTX 4060 Laptop 上，种子 0 的 64 批全幅 1280 × 960 采集单批中位数 67.32 ms，p95 为 74.67 ms，总采集约 4.18 s，文件落盘与额外线性读回另计。该会话显式设备分配峰值 65,822,672 字节，主机可见分配峰值 35,638,904 字节。每个视图在静止时逐步改善，持续运动时每次重置都会回到首批质量。[采集与计时](panda/seed-0/receipt.json) · [采集方案与会话](panda/summary.json)

## 封闭室

六面墙封闭，室内有两盏矩形面光源、一块遮板和红色表面，环境光为零。保持相机、光源、几何和白墙选区一致。自研端使用两个种子各 16 批、每批每像素 16 条主路径。独立 SAPIEN SDK 参考为每阶四个种子各 512 样本。表中为白墙合并选区的平均线性亮度。96 × 96 是标定采集分辨率。[原生读数](room/native.json) · [独立参考读数](room/reference.json) · [标定图](room/comparison.png)

| 漫反射反弹数 | 自研 Vulkan | SAPIEN 参考 | 自研相对同阶参考 |
| ---: | ---: | ---: | ---: |
| 1 | 0.01634 | 0.01693 | −3.49% |
| 2 | 0.01883 | 0.01974 | −4.61% |
| 6 | — | 0.02264 | — |

第二次反弹把经过另一面墙才到达相机的光加回来。参考自身从两次增加到六次仍有增量，两次结果比六次低 12.82%。固定深度后观察收敛，两组独立种子在第 16 批的逐像素 RMS 差分别为首批的 25.66%（一次）和 26.06%（两次）。选深度控制光路截断，选样本数控制估计噪声。[完整标定数据 ZIP](room/calibration.zip) · [归档校验](room/calibration-zip-verification.json)

## 运动旧影与首帧

在同一会话先累积四批，再分别横移相机 20 mm、旋转方向灯、插入遮光板、横移红板 60 mm。对每种变化，在相同的新状态和采样种子独立重开会话。原会话的新状态第一批与第四批，RGB、深度、实例 ID、掩码和显示前 RGBA32F 均与重开会话逐字相同。移动前后的画面有实际差异。当前行为是全帧清除旧历史，几何重投影和光照重用属于后续开发。[四种运动结果](panda/summary.json) · [同帧查看器](index.html)

在首批 16 样本图像上另行测试了 5 × 5 空间滤波。邻域限定到同一前景对象，检查深度相对差和当前视图的直接光差。用另一个种子的 1024 样本原图作比较，两指线性 RMS 误差分别降低 65.4% 和 67.5%，19,587 个轮廓及直接光边界像素的误差降低 40.3%。滤波本身在 CPU 上用时 1.404 s，还需直接光指引。该结果验证这一静态摆台首帧的质量。下一步可在 GPU 的着色阶段提供几何和直接光指引，对移动边缘、显露区、不同材质和整帧耗时重新验收，随后再评估时域重投影。[首帧原图、滤波与独立高采样参考](preview/first-preview-reference.png) · [误差和计时](preview/metrics.json) · [时空滤波研究](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/)

## 复现与数据范围

夹爪实验使用原生后端采集 305 帧，另作 36 次显示前浮点读回。此目录保留全部夹爪显示图、场景几何、计时和校验清单，以及重现首帧实验的两份全幅原始线性数组。其余全幅线性数组位于原采集工作树的 `artifacts/rendering/progressive-panda-study/capture-01`，可按清单核验。[原始采集清单](panda/manifest.json) · [夹爪实验入口](../../examples/progressive_panda_study.py) · [封闭室入口](../../examples/closed_room_bounce_calibration.py) · [原生编译与测试](native-build.json) · [浏览器切换检查](viewer-check.json)

从仓库根目录运行下面的命令，可用这次归档的原图和独立参考复现空间预览。输出选择新目录。实测使用 Windows、RTX 4060 Laptop 和本地构建的 Vulkan 库，新功能尚待独立安装包验证。

```powershell
python rendering/examples/progressive_first_frame_preview.py --first rendering/results/progressive-room/panda/seed-0/batch-0001.npz --reference rendering/results/progressive-room/panda/seed-1/batch-0064.npz --original-study rendering/results/diffuse-bounce --output artifacts/rendering/first-frame-repro
```
