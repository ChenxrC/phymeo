# M6 图像输出开发计划

G2 已接入主机帧包构造校验。[模块测试](../tests/test_outputs.py)用手写像素验证编码、所有权和背景的一致性；设备读回和真实相机观测在 G3–G5 验收。

**目标：** 让每次成功拍摄交付可独立消费、拥有自身主机字节、标定和来源一致的 RGB／前向深度／实例 ID／有效掩码。

**架构：** 保留现有 Channel、ImagePlane、Capture 数据结构。M6 负责输出的字节结构、通道语义和包内一致性；后端负责真实成像、读回同步及与请求、源帧、静态实例清单的交叉核对。查看器和文件编码器消费 Capture，不反向访问设备或物理状态。

**技术栈：** Python 3.11+ 标准库；单测可使用 pytest，基础发行包不增加 NumPy、图形设备或图片编码库依赖。

**依据：** [独立项目说明](../README.md)、[原 R0–R4 路线](../../docs/rendering/README.md)、[当前输出合同](../src/phymeo_rendering/outputs.py)。项目调度见 [计划索引](README.md)，跨模块联调及最终证据分别归 [08-integration](08-integration.md)、[09-acceptance](09-acceptance.md)。

## 范围与文件所有权

| 所属 | 文件／职责 |
| --- | --- |
| M6 独占产品 | rendering/src/phymeo_rendering/outputs.py：四个既有 Channel、ImagePlane、Capture；必要的内部校验 |
| M6 独占未来单测 | rendering/tests/test_outputs.py：标准库构造的小图、字节与元数据反例 |
| M6 计划 | rendering/plans/06-outputs.md |
| 上游协作，不直接修改 | assets.py、scene.py、cameras.py、pipeline.py、backends/：静态 ID、帧键、相机、请求与设备读回 |
| 下游 | 查看器、编码器、训练消费端；最终真实渲染案例由 08/09 组织 |

当前 `ImagePlane` 和 `Capture` 在构造时校验不可变字节与通道合同。GPU 张量、HDR 和额外传感通道属于后续扩展；PNG、MP4 编码由消费端负责。

## 已有接口与上下游边界

当前接口保持为：

~~~python
ImagePlane(channel: Channel, data: bytes)
Capture(frame_key: FrameKey, time_s: float, camera: Camera,
        planes: tuple[ImagePlane, ...])
~~~

上游 RenderSession.capture(frame, request) 返回 Capture。CaptureRequest 提供相机、RenderPath 和请求通道；SceneFrame 提供 time_s 与完整 FrameKey。Capture.camera.frame_key 必须等于 Capture.frame_key；返回帧键和时间还必须等于本次 frame，返回相机必须对应本次 request，通道集合必须恰好是请求集合且无重复。这些涉及外部输入的核对在后端发布门完成，不能仅靠 Capture 自己推断。

后端在其读回／同步完成后构造不可变 bytes，才返回成功结果。Capture 不借用会话下一帧会复写的 staging 区。再次 capture、close、世界 reset 后，先前 Capture 的内容必须保持不变。M6 不承诺设备零拷贝；实际复制与同步成本由 08/09 单列。

## 字节、像素与数值合同

所有通道尺寸取 camera.width × camera.height，紧密排列，无行尾 padding；左上为首像素，先行后列，多分量在像素内交错。多字节样本使用小端；不依赖宿主机原生端序。

| Channel | 每像素字节／解释 | 可接受数值及禁止处理 |
| --- | --- | --- |
| RGB = srgb8_rgb | 3；R、G、B 顺序的 uint8 | 已经完成显示变换的 sRGB；不得再按线性颜色重复 gamma，不接受 BGR 或隐含 alpha |
| FORWARD_DEPTH = forward_depth_m_f32 | 4；IEEE 754 小端 float32 | 相机 −Z 方向的正深度，单位米；miss 为 +inf；拒绝 NaN、−inf、负值及 0 |
| INSTANCE_ID = instance_id_u32 | 4；小端 uint32 | 0 为背景；正值是当前场景 Instance.id，范围 1…2^32−1；不得经浮点编码、插值、调色或降噪 |
| VALID = valid_u8 | 1；uint8 | 仅 0／1；0 表示 miss，1 表示可见 hit，不用 255 代替 1 |

当同时给出相关通道时，逐像素检查 VALID=0 ⇔ depth=+inf、VALID=0 ⇔ ID=0、depth=+inf ⇔ ID=0；没有请求某个通道时不得凭空制造它。RGB 背景允许有颜色，不能用 RGB 是否为黑色判定有效性。当前合同未声明 MSAA 混合标签，边缘标签也只能代表一个一致的可见样本，不能以平滑灰值冒充 ID。

M6 可检查 ID 的字节与背景关系，但正 ID 是否属于静态清单必须由持有 VisualScene 的后端验证。M6 不从 uint8 RGB 猜测输入曾经是否线性或 BGR，也不从一张深度图推断它原本是不是射线距离；这两类错误必须通过已知色块和离轴几何的后端联调发现。

near/far 沿前向度量。M6 不擅自裁剪深度或增加门限；设备 float32 的边界舍入与实际裁剪行为由相机／后端合同统一验收。相同标定平面的 R3 误差门保持原路线数值。

## 可以本地独立完成的步骤

### 1. 锁定格式与不可变所有权

- [x] 在 test_outputs.py 写四种 Channel.value、每通道尺寸和小端字节的正控；先运行确认尚缺少的校验产生预期 RED。
- [x] 写 bytearray／memoryview、非 Channel、非 ImagePlane 成员及可变 planes 列表的负控；Capture.planes 必须为 tuple。首版要求 data 确为 bytes；后端要显式复制，不在数据容器中隐式转换任意对象。
- [x] 在 outputs.py 的构造边界补最小检查：ImagePlane 检查局部类型；Capture 利用相机分辨率检查长度和唯一通道。非法类型报 TypeError，非法值／结构报 ValueError；错误指出通道与所需／实际长度。
- [ ] 用下列 2×2 哨兵值验证行列和分量顺序；不能只测试全零图。

~~~python
from struct import pack, unpack
from phymeo_rendering.outputs import Channel, ImagePlane

def test_depth_little_endian_and_background():
    data = pack("<4f", 0.25, 1.0, float("inf"), 2.0)
    plane = ImagePlane(Channel.FORWARD_DEPTH, data)
    assert plane.data[:4] == b"\x00\x00\x80\x3e"
    assert unpack("<4f", plane.data) == (0.25, 1.0, float("inf"), 2.0)
~~~

RGB 的 12 字节使用互异四像素，例如红、绿、蓝、白；ID 使用 1、256、0、2^32−1，掩码使用 1、1、0、1。验证不得交换行、列、RGB 分量或端序。这里是字节合同，不冒充真实渲染验证。

### 2. 验证包内绑定及传感通道一致性

- [x] 写合法 Camera／FrameKey 的显式测试夹具，使用非方形 3×2 图抓宽高互换；测试 camera.frame_key 与 Capture.frame_key 不等时失败。
- [x] 参数化测试截短／多余字节、重复通道、空通道包、非有限或负 time_s、非法分辨率；在所属模块已有相同校验时复用错误边界，避免复制 Camera 的全部实现。
- [x] 写 NaN、−inf、0、负深度和掩码 2／255 的负控；补 depth／ID／VALID 背景互相矛盾的逐像素反例，明确指出出错像素。
- [x] 实现上述校验；合法通道子集保持可用，包括仅 RGB、仅深度和仅 ID。空请求／空结果应由 pipeline 与 M6 共同明确拒绝，不能默认为成功曝光。
- [x] 运行聚焦单测并保留结果；不得通过把异常像素改成 0 或 +inf 来让负控通过。

建议节点：test_capture_rejects_frame_mismatch、test_capture_rejects_wrong_plane_length、test_capture_rejects_duplicate_channels、test_sensor_planes_reject_inconsistent_background、test_capture_accepts_requested_channel_subsets。

### 3. 与后端交接真实输出

- [x] 给后端作者提供上述固定小图字节与完整 Capture 夹具，先用测试双验证 Capture 在下一次 capture／close 后仍可读且不变。
- [x] 在 08 联调中验证请求与返回的完整相机／帧／时间／通道一致；错世界、错 epoch、错 revision、错 phase 或未知实例 ID 均拒绝发布，不返回部分 Capture。
- [ ] 在 R0／R3 标定台验证 sRGB 灰阶只转换一次。以无光照歧义的确定性显示变换夹具检查线性 0、0.5、1 对应 sRGB8 的 0、188、255；若后端采用其他 tone map，先记录该函数并给对应独立期望，不能把本例强加于未知 tone map。
- [ ] 验证离轴平面深度为 −z 而不是点到相机的距离，背景统一 +inf／0／0；有效像素 ID 与清单精确一致，传感通道不进入颜色后处理。
- [ ] 后端输出的下载、同步、文件编码费用分别记录；M6 单测本身不宣称吞吐或真实图像质量。

未来聚焦执行命令（仓库根目录，当前不执行）：

~~~text
python -m pip install ./rendering
python -m pytest rendering/tests/test_outputs.py -q
~~~

## 验收、合并时机与 R0–R4 对应

| 层次 | 必须成立的门槛 | 合并条件 |
| --- | --- | --- |
| 模块结构 | 四个枚举值保持兼容；bytes 长度、端序、通道集合、所有权、同帧绑定及反例全通过 | 可独立合并 M6；不依赖图形设备，不声称 R0 完成 |
| R0 真实首图 | 一次真实曝光的 RGB／深度／ID 输出、标定及设备信息可消费；采集前后物理状态相同、零步进 | 与后端的发布门一起联调通过，再计为 R0 输出闭环 |
| R3 相机观测 | 640×480；0.25–2 m 离轴标记；投影误差 ≤1 px；距轮廓 ≥2 px 的相同平面有效像素深度最大误差 ≤2 mm；有效区 ID 全部一致 | 外部与腕部相机、遮挡、背景、翻转、reset 均有真实证据；边缘单列 |
| R4 隔离与成本 | 1→2→8 世界，每世界 2 台 320×240；逐世界深度／ID 对单世界相同输入；指定世界 reset 不污染别的结果 | 预热后采集 300 帧，统计同步／复制／编码成本，资源报告交 09；不把 CPU 小图通过当作 R4 |

M6 不承担 R1 外观资产补全或 R2 画质算法，但其编码与元数据必须能忠实保留这些阶段结果。R2 的 1280×720 单相机 p95 ≤33.3 ms 是整体交互目标，不是本模块 Python 打包耗时门槛。修改 Channel、所有权或错误语义前，先同步相机、pipeline、两后端及下游消费者；没有这次契约对齐时不合并破坏兼容的接口变更。
