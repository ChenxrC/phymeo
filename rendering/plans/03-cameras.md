# M3 标定相机开发计划

G1 已实现相机参数校验和六字段曝光绑定，见[模块测试](../tests/test_cameras.py)。设备成像与 R3 像素级验收待执行。

**目标：** 让固定相机和腕部相机用同一种显式曝光类型，正确绑定标定、米制坐标、曝光位姿与来源帧。

**结构：** `cameras.py` 只负责 `Camera` 的标定和曝光绑定。`CaptureRequest`、渲染路径和通道选择属于 M4 `pipeline.py`；`Channel` 与图像编码属于 M6 `outputs.py`；设备相机转换和实际成像属于 M5。

**依据：** [当前 cameras.py](../src/phymeo_rendering/cameras.py)、[框架说明](../README.md)、[R3 观测指标](../../docs/rendering/README.md)。配合 [M2 场景计划](02-scene.md)、[集成计划](08-integration.md) 与 [统一验收](09-acceptance.md)。

## 文件与测试归属

| 归属 | 路径 | 责任 |
| --- | --- | --- |
| 产品独占 | `rendering/src/phymeo_rendering/cameras.py` | `Camera` 局部合法性和计划中的曝光帧校验函数。 |
| 测试独占 | `rendering/tests/test_cameras.py` | 标准库 `unittest`；标定正负例、曝光来源绑定和解析期望值。 |
| 文档独占 | `rendering/plans/03-cameras.md` | 本模块步骤、指标和交接。 |
| 他方所有 | M2/M4/M5/M6/M7 文件 | 位姿组合／请求／后端转换／输出／来源适配由各负责人完成。 |

不在本任务增加纹理、显示窗口、设备张量、相机噪声或物理传感器步进，也不复制一个 `Channel` 枚举。

## 输入、输出与上下游

当前 `Camera` 的准确字段：

- `id: str`，`frame_key: FrameKey`，`world_pose: Pose`。
- `width`、`height`：整数像素尺寸。
- `fx_px`、`fy_px`、`cx_px`、`cy_px`：以像素为单位的针孔标定。
- `near_m`、`far_m`：沿相机前向测量的米制裁剪面。

相机局部坐标为 **+X 右、+Y 上、−Z 前**；图像坐标从左上角起，首像素中心为 `(0.5, 0.5)`。`world_pose` 使用 M2 的米制 XYZW 四元数。

上游 M7 解析固定外参，或从当帧腕部节点与安装外参组合曝光位姿；M2 提供完整 `FrameKey` 和 `Pose`。下游 M4 将此相机放入 `CaptureRequest`；M5 消费同帧的 `SceneFrame` 与请求；M6 输出保留这次曝光的标定与帧键。

G1 提供 `validate_exposure(frame: SceneFrame, camera: Camera) -> None`。它逐字段核对 `camera.frame_key == frame.key`，键不匹配时抛出带字段名的 `ValueError`。`scene_key` 与已打开场景的关系由 M2/M5 检查。

## 顺序开发步骤

### 1. 固定标定与曝光类型

- [ ] 先写 `test_calibrated_exposure_preserves_parameters`、`test_camera_rejects_invalid_dimensions_or_intrinsics`、`test_camera_rejects_invalid_clip_range`，取得缺少校验的明确失败。
- [ ] 检查 `id` 是非空白字符串；`frame_key` 和 `world_pose` 是实际的 M2 类型，不接受未绑定来源的普通字典或整数 tick 替代。
- [ ] `width`、`height` 为正整数且不能是布尔；`fx_px`、`fy_px` 是有限正数；`cx_px`、`cy_px` 只要求有限实数，不强制图像中心，也不强制落在图像内。
- [ ] 裁剪必须满足有限的 `0 < near_m < far_m`；拒绝 NaN、无穷、相等或颠倒的裁剪面，不自动交换。
- [ ] 保留原值和不可变语义，不隐式从 FOV 推测焦距，不翻轴，不把相机位姿从上一帧补齐。位姿的单位长度规则只复用 M2，不另设容差。
- [ ] 用 `TypeError` 区分错误类型、`ValueError` 区分非法值；运行本模块单测并保留结果。

### 2. 逐字段绑定曝光来源

- [ ] 写 `test_exposure_accepts_equal_complete_frame_key`；使用值完全相同但不同对象实例的 `FrameKey`，应接受，不以 Python 对象身份替代来源检查。
- [ ] 写参数化的 `test_exposure_rejects_each_frame_key_mismatch`：分别仅修改 `scene_key`、`world_id`、`epoch`、`revision`、`tick`、`phase`，每次都应拒绝。
- [ ] 实现 `validate_exposure`；相同 tick、相同相机位姿不能抵消其他五个来源字段的差异。
- [ ] 通知 M5 在真正成像前调用此入口。拒绝路径不得生成可消费结果；是否执行 GPU 命令及失败原子性由 M5 的会话测试证明。
- [ ] 明确 `Camera` 不拥有另一份时间戳：结果时间来自被捕获的 `SceneFrame.time_s`，M6 不从视频帧号或相机索引计算时间。

### 3. 交接独立标定参照与腕部曝光

- [ ] 在 `test_cameras.py` 保存确定性标定夹具：640×480，`fx=fy=500`，`cx=320`、`cy=240`，near=0.1 m、far=5 m，恒等曝光位姿。
- [ ] 同时保存独立手算期望：对相机点 `(x,y,z)`，`d=-z`；`u=fx*x/d+cx`、`v=cy-fy*y/d`。它是测试参考，不要求本模块新增渲染／投影执行接口。
- [ ] M7 使用同帧腕位姿与安装外参得到一次 `Camera.world_pose`。增加两个手腕姿态的合同夹具，第二次曝光必须带第二帧键；不得从已缓存旧相机位姿得到新帧“成功”。
- [ ] M5 对接 SAPIEN 时在后端内转换实体相机轴；不要修改公共 Camera 约定。原生后端与参考后端必须使用同一组公共标定和曝光。
- [ ] 将数值参考交给集成负责人用于 R3 的实际标记、深度、ID 对照；本模块的类型校验不冒充已完成投影或图像对齐。

## 正负例与可计算验收

| 层次 | 用例 | 计划判据 |
| --- | --- | --- |
| 单元正例 | 上述640×480标定及完整帧键 | 所有标定、位姿及来源字段逐值相等，输入改变数为 0；函数无设备调用。 |
| 单元正例 | 主点改为 `(-20, 530)`，其他参数合法 | 接受并原样保留，支持离轴／裁剪相机，不“纠正”为中心。 |
| 单元负例 | 0、负数、布尔或非整数尺寸；焦距≤0；非有限内参；near≥far | 每个独立负例拒绝且定位字段；无自动修正。 |
| 单元负例 | 六个 FrameKey 字段分别错配 | 6/6 拒绝；完整相等的独立对象接受。特别保留同 tick 跨 world 和同 tick 跨 epoch 两例。 |
| 解析参考 | 恒等相机，点 `(0,0,-1)` 与 `(0.1,0.1,-1)` | 期望像素分别为 `(320,240)`、`(370,190)`，前向深度均为 1 m；参考算术最大误差 ≤ `1e-12`，不计作后端成像验收。 |
| 解析参考 | 首像素中心对应点 `((0.5-320)/500, (240-0.5)/500, -1)` | 投影为 `(0.5,0.5)`，区分像素中心与像素角点，不允许额外半像素偏移。 |
| R3 实渲 | 640×480、0.25–2 m 同几何标定场景，含离轴标记 | 特征投影最大误差 ≤1 px；相同平面、离轮廓≥2 px的有效区深度最大误差≤2 mm；有效区实例 ID 全部一致。按原路线图记录边缘而非剔除失败。 |
| R4 集成 | 1→2→8 世界，每世界两台320×240相机 | 曝光／结果键错配数为 0；外部和腕部相机各与单世界同输入比较。GPU 互通与吞吐另由 M5/M6 实测。 |

前向深度与欧氏射线距离不能混用：例如 `(0.1,0.1,-1)` 的前向深度为 1 m，而射线距离为 `sqrt(1.02) m`。背景正无穷、ID零及有效掩码的字节合同由 M6 负责，M3 只交接几何语义。

## 执行与合并依赖

独立包安装后，从仓库根目录计划执行：`python -m unittest discover -s rendering/tests -p test_cameras.py -v`。

- 先合 M2 的完整键、Pose 校验及组合顺序，再合 M3；本测试使用真实 M2 类型，不复制替身。
- M4 使用 `Camera`，但请求和通道字段始终留在 `CaptureRequest`。M6 保留真实曝光及结果来源。
- M5 负责调用 `validate_exposure`、检查打开场景、执行坐标转换和实际捕获；函数存在不代表会话已正确接线。
- M7 拥有安装外参和已接受帧来源；腕部机位必须由本次帧计算，而不是重跑物理或借未来轨迹姿态。
- 对应 R0 首图标定、R1 两姿态相机对齐、R2 同机位画质比较、R3 数值观测及 R4 多世界身份；只有集成证据达到原指标时才标相应节点完成。
