# 一套声明驱动多箱和凸工具–可动材料片

`ContactSceneBuilder` 接受按世界编号、局部物体编号登记的有限箱、一般凸工具和三角材料片。箱的半尺寸以米计，给出质量后可直接采用均匀箱主惯量。一般凸工具需给出主惯量和相对质心的本地顶点。每个片节点有独立质量和速度，共用节点只登记一次，每条材料边至多建一个 XPBD 距离约束。两类场景均用 `build()`、`sample()` 和 `step(dt, external)`，从同一返回对象读取入口材料力、接受后的节点/刚体状态和真实扫掠诊断。[公共接口](../../src/phymeo/runtime/contact_scene.py) · [两套物体声明与共同推进循环](../../examples/declared_contact_scene.py)

## 多箱承载与卸载

两个共址世界 7 和 42 使用相同的局部物体编号 10、20、30。每组底箱固定，上两箱各 1 kg，重力为 −1 m/s²。箱间有限层宽 1/16 m、刚度 128 N/m，初空间隙 3/64 m 和 7/128 m，解析得到上下接触各 2 N、1 N。每层实际命中四角，初态合力矩为零。前两个 1/64 s 子步保持平衡，后十二步只给世界 42 的顶箱 4 N 向上外力。

| 14 子步之后 | HOST FP64 | TYX CUDA FP32 |
|---|---:|---:|
| 世界 7 顶箱高度 | 0.601562500 m | 0.601562500 m |
| 世界 42 顶箱高度 | 0.666047174 m | 0.666047096 m |
| 世界 42 末子步入口底接触 | 1.016111642 N | 1.016116381 N |

世界 42 上层已卸载，世界 7 仍保持 2 N、1 N。全部刚体均实际注册，模型中的粒子数为零，返回的粒子求解报告为空。[HOST 全轨迹](../../artifacts/p65-contact-scene-builder-20261105-01/stack-host-result.json) · [CUDA 全轨迹](../../artifacts/p65-contact-scene-builder-20261105-01/stack-cuda-result.json)

## 偏心凸工具压两片共边三角

一般四面凸工具质量 2 kg、主惯量各 1/16 kg·m²。先到达材料层的有限尖端在质心局部坐标 (1/16, −1/8, −1/8) m，初间隙 1/32 m。两张三角片有四个独立结点，质量依次 1、2、3、4 kg，共五条材料边。入口压缩 1/32 m，解析与实测均为 4 N；在尖端施于工具的力矩为 (−0.5, −0.25, 0) N·m，布节点法向力依次 −1.75、−0.75、−1.50、0 N。将工具绕自身 z 轴转 180° 后，负载转移到另一张真实面，力矩的 x/y 分量换号。

四个 1/512 s 子步后，HOST 和 CUDA 的工具高度均为 0.156326234 m，角速度约 (−0.0624232, −0.0312108, −0.00000078) rad/s。HOST 四节点的实际竖向位移约为 −66.705、−14.295、−19.058、0 μm；CUDA 各自一致至所列小数。逐步力与路径及材料权重均随公开输出保留，布和工具使用各自真实积分器。[HOST 全轨迹](../../artifacts/p65-contact-scene-builder-20261105-01/tool-sheet-host-result.json) · [CUDA 全轨迹](../../artifacts/p65-contact-scene-builder-20261105-01/tool-sheet-cuda-result.json) · [独立力矩、节点权重及线动量断言](../../tests/test_contact_scene_builder.py)

## 设备与失败边界

两个 CUDA 例程的几何包围与配对分类分别完成 58、18 次设备启动；材料核分别完成 15、5 次启动，累计处理 208、5 个材料站点。两案均运行真实 TYX 刚体积分，含布的例程同时运行 TYX XPBD。材料站点映射、窄相和可接受路径检查包含 HOST 工作，观测会读回数据。此处给出物理结果和实发计数，未测吞吐。模型沿用单工作区 256 个形状、8,192 个登记顶点槽的容量及既有曲线运动判据；本次世界隔离通过原碰撞键实现。[CUDA 刚体/材料收据](../../artifacts/p65-contact-scene-builder-20261105-01/stack-cuda/final.json) · [CUDA 刚柔收据](../../artifacts/p65-contact-scene-builder-20261105-01/tool-sheet-cuda/final.json)

每个同世界形状对必须显式分配材料或排除，共边等实际拓扑信息保留在碰撞模型中。本例将片自身两面的互作用显式排除。漏配、未支持的布–布摩擦法则，以及在 FP32 中退化的正逆质量或三角面均会报错。候选容量不足、两个端点分离但步内穿过实体的外力输入都会保留之前的已接受态与物理帧，同一实例随后可继续。每次 `build()` 都拥有独立状态和资源，原构建器可以继续添加物体；关闭后可从同一声明另建新实例。[HOST 专项](../../artifacts/p65-contact-scene-builder-20261105-01/host/final.json) · [精度与扫掠负控](../../artifacts/p65-contact-scene-builder-20261105-01/boundaries/final.json) · [设备专项](../../artifacts/p65-contact-scene-builder-20261105-01/cuda/final.json)

```powershell
python examples/declared_contact_scene.py --case stack --backend host --output stack.json
python examples/declared_contact_scene.py --case tool-sheet --backend cuda --output tool-sheet.json
```

## 接受状态的光学快照

`scene.geometry()` 返回 `time_s`、`model` 和按完整 `CollisionKey` 索引的静态 `poses`。刚体本地材料顶点随当前质心和实际旋转变换，三角面从共享粒子行读取各自三个当前节点。姿态数组由独立只读存储支撑；后续物理步、修改外部状态副本或关闭运行时之后，先前快照仍可用于同一次曝光。示例中的 `--depth` 在初态和每一接受物理时刻使用真实 HOST 相机。[公开例程](../../examples/declared_contact_scene.py) · [快照与光学专项](../../tests/test_contact_geometry_snapshot.py)

```python
reading, exposure = scene.observe_scene()
image = camera.capture(exposure.model, exposure.poses, world=7)
```

`observe_scene()` 在同一状态锁内取当前材料力和传感器几何。对于 2 kg 受载上箱，一步 1/32 s 的入口力为 4 N，末间隙升至 17/512 m，重新采到 3.75 N。`last_frame.loads` 继续保留已施加的 4 N。时钟累加实际积分使用的步长；容量失败和小到不能推进当前时刻的步长都保留前一帧与时间。[载荷与时间回滚](../../artifacts/p65-contact-geometry-snapshot-20261105-01/host-v2/final.json)

在另一套光学留出中，7 号世界的倾置长箱沿负 z 平移并绕世界 z 轴旋转，四个有限片节点取不等竖直初速。0.1875 s 时，从原相机中心读到箱面深度 1.546875 m；移到片面内的材料采样位置读到 2.98828125 m。42 号世界同局部编号的静止箱始终读到 4.875 m。真实 CUDA 组把设备动力学返回的状态冻结为 HOST 材料顶点，再以 `TyxRaycaster` 运行设备射线；另用独立 HOST 射线核对有限片距离。以 0.1 s 输入时，设备组的曝光时间采用 FP32 实际步长 0.10000000149011612 s。[interpret 实录](../../artifacts/p65-contact-geometry-snapshot-20261105-01/interpret/final.json) · [CUDA 实录](../../artifacts/p65-contact-geometry-snapshot-20261105-01/cuda/final.json)

原凸工具压片示例在世界 3 的五次曝光均读到工具键 (3, 7, 0)，0 到 0.0078125 s 的中心光线距离由 0.718750000 降至 0.718673758 m。[带时刻的原始受载相机输出](../../artifacts/p65-contact-geometry-snapshot-20261105-01/tool-sheet-depth.json)
