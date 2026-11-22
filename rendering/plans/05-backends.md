# M5：真实图形后端与设备会话实施计划

G2 提供 `ValidatedRenderSession(scene, delegate)`，负责能力、帧绑定、输出与关闭检查。[会话测试](../tests/test_backends.py)使用记录器。原生和 SAPIEN 入口在 G3 实现设备接入前显式报告不可用。

**目标：** 经同一个 `RenderBackend`／`RenderSession` 公共入口产生真实图像；首版由 native raster 承担快速及传感路径，由可选 SAPIEN 参考 PT 提供真实路径追踪对照，分别记录设备、失败与成本。

**架构：** 原生后端消费 M4 阶段表，自己拥有 Vulkan 设备资源、着色器、屏障、提交、完成等待及读回。SAPIEN 后端把完整曝光交给其纯渲染系统，独立管理其资源，不实现假的原生 RHI。两者共享资产、帧、相机和输出语义，不共享物理引擎所有权。

**技术栈：** 公共 Python 3.11+ 标准库合同；native 使用 Vulkan 及私有编译桥接；参考实现采用调研固定版 SAPIEN 3.0.3。pytest、shader 编译器、C++ 编译／构建工具为开发依赖，具体设备可用性在执行时实测。

**依据及公共入口：** [计划总入口](README.md)、[M4 管线](04-pipeline.md)、[集成计划](08-integration.md)、[验收计划](09-acceptance.md)、[包合同](../README.md)、[R0–R4 路线图](../../docs/rendering/README.md)。README／08／09 由集成负责人维护。

## 1. 所有权、依赖与并行边界

| 类型 | 范围 |
| --- | --- |
| M5 产品独占 | `rendering/src/phymeo_rendering/backends/__init__.py`、`native.py`、`sapien.py` |
| 可新增私有实现 | 同目录 `_native_device.*`、`_native_session.py`、`_sapien_session.py`、`_shaders/*` 及真实需要的私有构建输入；不得扩成第二个物理 runtime |
| M5 测试独占 | `rendering/tests/test_backends.py`；可增加专属 `test_native_backend_device.py`、`test_sapien_backend_device.py` |
| 依赖 M1 | 有界、确定的 URI 解析／网格解码、法线及视觉材质合同；不能由两个 backend 分别猜单位或读取不同文件 |
| 依赖 M2/M3/M6 | 场景／节点唯一身份、合法变换、相机标定、CPU-owned 图像输出及校验 |
| 依赖 M4 | `validate_capture_request`、`DefaultPassPlanner`；参考后端仅用请求校验，不必模拟 native pass |
| 依赖 M7 | 已发布源帧／封存轨迹的单向适配，设备 backend 不导入 Phymeo 或求解 FK |
| 公共修改协调 | pyproject extras、原生扩展打包、CI、包级导出及公共文档由 08 owner 合并，不并行改同一文件 |

可先并行进行 native 设备探测、SAPIEN 纯渲染探测、M1 资产解码和 M3 标定。两条 backend 分支各自拥有私有文件；`backends/__init__.py` 只由一名集成作者修改。R0 双路径参考采集及 G3 原生门必须使用同一公共场景与相机 fixture。

## 2. 公开协议与新增实现入口

保留现有协议签名，不把 Vulkan 或 SAPIEN 句柄放入公共 Scene／Capture：

```python
class RenderSession(Protocol):
    def supports(self, request: CaptureRequest) -> bool: ...
    def capture(self, frame: SceneFrame, request: CaptureRequest) -> Capture: ...
    def close(self) -> None: ...

class RenderBackend(Protocol):
    def open_scene(self, scene: VisualScene) -> RenderSession: ...
```

计划新增 `NativeBackend(config=None)` 和 `SapienBackend(config=None)`，实现 `open_scene`。配置类分别放在 `native.py` 和 `sapien.py`，只描述本后端设备选择、PT 采样／反弹／种子／滤噪、固定显示参数；具体字段在 M5.1 的接口测试中与 08 一次冻结，不向现有 `CaptureRequest` 临时塞私有字段。

统一定义 `BackendUnavailableError(RuntimeError)`、`UnsupportedRequestError(RuntimeError)`、`SessionClosedError(RuntimeError)`。合法结构但不受当前设备／后端支持的请求，`supports` 返回 False，`capture` 抛 `UnsupportedRequestError`；类型／值错误分别用 TypeError／ValueError，不伪装成设备不支持。缺失可选依赖、设备不存在、驱动或所需 feature 不可用，用 `BackendUnavailableError` 保留原异常链。

- `supports` 判断完整路径、通道、相机尺寸／裁剪等组合以及已上传场景的能力，不只判断 RenderPath 枚举。不渲染、不读回、不重新初始化设备。
- `capture` 必须再次检查，不能以“调用方已经 supports”为由跳过。调用 M3 的 `validate_exposure(frame, request.camera)` 核对完整 FrameKey，再检查场景 key 等于打开会话的 key；错 world／epoch／revision／tick／phase 任一项均拒绝。
- SceneFrame 节点必须满足 M2 唯一绑定和完整性。资产／灯光／实例内容变化必须用新场景 key 并重新 open；会话拥有已验证内容，不能让外部改 URI 对应文件而悄悄换模型。
- 公开帧合同支持封存回放；不能仅因 tick 小于上次拍摄就判陈旧。真正错误是把不同源帧的相机、姿态、叠加量混用。历史跳转、epoch／场景／相机／光照改变必须重置 PT 累积，不能泄漏上一帧颜色。
- `capture` 同步完成并返回 CPU-owned `bytes`。下一次拍摄及 close 后旧 Capture 仍有效；设备 scratch 不能冒充已拥有的图像。失败不得返回半个 Capture，重试不能读到旧曝光内容。
- `close` 幂等，等候本会话已提交工作完成后释放自己拥有的资源；不关闭别人的设备或物理 runtime。关闭后 supports／capture 都抛 `SessionClosedError`。

实际诊断以测试／运行 sidecar JSON 保存，不强行塞进当前 Capture。每次运行必须含实际 backend、requested/effective path、设备 UUID／名称、driver/API/library 版本、shader／资产散列、尺寸、通道、配置、读回路径及失败异常。若需要统一公开 diagnostics 方法，先由 08 冻结签名；不得让已有三方法协议在不同 backend 悄悄分叉。

## 3. 三条路线的准确完成边界

| 路线 | 必须做到 | 不能替代什么 |
| --- | --- | --- |
| Native raster | 原生 Vulkan 实际上传、可见性、阴影、线性金属粗糙度光照、显示转换及传感导出；经公共 capture 返回真实像素 | 不能用 SAPIEN／MuJoCo／CPU 软件图隐藏 native 缺失；不能只录命令或输出常量图 |
| SAPIEN reference | 同一输入纯渲染场景；真实 PT（以及可用时独立验证的 raster）；显式转换相机轴、深度、ID、显示编码 | 不推进 PhysX／任何物理；不能把参考 PT 标成 native PT，也不能凭 import 成功标设备可用 |
| Native path trace | 独立实现加速结构、射线积分、材质采样、累积／失效；全部完成后才宣告支持 | 枚举、TRACE 阶段或 Vulkan 库存在都不是光追证据；缺少该能力可明确拒绝 |

**R0 双真实路径参考可行性允许先采 SAPIEN raster＋SAPIEN PT。** 两档都必须由公共入口实际成像，保留真实设备和路径信息。**G3 合并放行及自研首版另硬要求 native raster 在同一桌面实测通过，并有真实参考 PT 可用。** 参考端两档成功不抵消原生光栅未实现或失败。native PT 单列，不作为明确限定于 native raster＋参考 PT 的首版强制门；若申请支持，必须按同一真实 PT 门另行取证。

自动 fallback 禁止。允许操作者显式选择另一 backend、操作系统或复制模式后重新采集，但报告保留原失败、实际选择和成本；不能修改请求后仍返回成功而不披露。

## 4. 实施任务与正负测试

### M5.1 会话拒绝、生命周期与可选依赖

- [ ] 先在 `test_backends.py` 冻结新增配置和异常 API。设备配置必须能明确指定设备，默认选择必须报告实际设备；未知参数拒绝。native 首版对 PATH_TRACE 明确 False／异常，不默许使用参考 backend。
- [x] 用测试替身验证两条实现的通用会话逻辑：正常 open→supports→capture→close，重复 close，关闭后使用，完整 FrameKey 单字段篡改、缺失节点、无效尺寸／请求、请求不支持。
- [ ] 测试仅基础包安装时导入 `phymeo_rendering`、`backends.native`、`backends.sapien` 均不加载 GPU runtime；显式 open 才加载依赖。屏蔽 SAPIEN 安装时 open 精确抛 BackendUnavailableError。
- [x] 先运行该文件取得 RED，再实现最小拒绝和生命周期逻辑；测试替身只属于测试文件，禁止以 fake production backend 通过设备验收。
- [x] 验证错误链包含原始原因，失败不产生合法 Capture；首帧 `Capture.planes` 中各 `ImagePlane.data` 在下一次 capture／close 后逐字不变。

测试中的异常与合同形式：

```python
from phymeo_rendering.backends import UnsupportedRequestError

# session、frame 和 request 由同文件专属 fixture 构造；
# fixture 的 device 实现为测试替身，不计入 R0 设备结果。
def assert_unsupported(session, frame, request):
    import pytest
    assert session.supports(request) is False
    with pytest.raises(UnsupportedRequestError):
        session.capture(frame, request)
```

### M5.2 Vulkan 设备与最小真实光栅

**前置：** M1 网格读取和 M2/M3 变换合同；M4 阶段表。设备初始化可先于完整 PBR 开发。

- [ ] 建立 native 专属测试：真实物理设备枚举，指定设备匹配，离屏 target 创建，已知三角形可见性、遮挡和 ID；缺少设备时显式标 unavailable，不能在正式设备验收中记 PASS。
- [ ] 采用私有 C++ 桥接和 Vulkan SDK，构建 shader 为 SPIR-V；桥接／shader 与安装包绑定版本及 SHA。SDK、编译器、Python ABI、桥接依赖的实测版本交 08 锁定，不把系统 PATH 上任意工具当可重复构建。
- [ ] 先实现有界资源上传、命令提交、completion fence 与独立 HOST readback；再实现 M4 的可见性／阴影／线性 PBR／显示阶段。暂不增加跨进程、CUDA 互通或通用 RHI 插件体系。
- [ ] 输出按 M6 `ImagePlane` 打包为紧密连续、top-left、little-endian；RGB 为 sRGB8，前向深度为米制 f32，ID 为原 Instance.id 的 u32，VALID 为 0/1 u8。
- [ ] 添加真实负控：不可用设备、超出设备限制的分辨率、缺失／坏网格、非法 shader、上传或读回失败。释放已创建资源，保留诊断，不改用软件图。
- [ ] 启用 Vulkan validation 采集窗口检查资源生命周期、布局、barrier 和同步；发现 error 即失败，不因“图看起来对”放行。

### M5.3 SAPIEN 真实参考路径

**前置：** M1/M2/M3/M6 已冻结；可与 M5.2 并行。

- [ ] 在独立参考环境固定 SAPIEN 3.0.3。核实本机设备、离屏初始化和纯 render system；不调用物理 step，不构造依赖物理推进的场景。
- [ ] 使用与 native 完全相同的解码资产、物体变换和曝光。显式转换实体相机轴／图像坐标，记录线性颜色与显示变换来源，避免两次 sRGB 编码或 tone mapping。
- [ ] 分别探测 raster／PT 能力，不把前者成功推断为后者。R0 的参考 PT 用固定 seed、64 samples/pixel、4 次最大反弹、关闭降噪进行首组对照；配置若接口不支持必须拒绝或由 08 改定并完整记录，不能忽略参数。
- [ ] 为真实设备写单平面离轴深度、遮挡 ID、背景无效值和相机旋转正控；以错轴、错误深度单位／背景、错误通道映射作为转换层负控。
- [ ] Windows 设备／互通失败保留实际错误；R0 允许显式主机读回。Linux 同版复核作为另一具名配置，不能把不同平台结果混作 Windows 已通过。

### M5.4 同源 R0 对照与失效隔离

**前置：** M5.3 可先完成 R0 双路径参考采集；M5.2 就绪后接入同一 fixture，通过 G3 原生门。公共标定桌由 08 收敛为唯一 fixture，不能由两个后端另建“近似同场景”。

- [ ] 固定 640×480、相机标定和自有程序网格，包含铝、哑光塑料、橡胶、遮挡物和已知距离。M1 导出后固定资产 SHA；两后端使用同一 VisualScene 和 SceneFrame。
- [ ] 可先保存 SAPIEN raster／PT 的真实 RGB 对照，再补 native raster 同源结果。R0 不要求 PT 同时提供全部 AOV；按公开 supports 表记录每条实际路径。native raster 必须产出 RGB、前向深度、ID、VALID；参考传感输出若经另一具名路径采集，单独记录请求／响应，不能冒充 PT 同次输出。某个联合请求的 supports 为 True 时必须完整返回全部通道，不能只返回 RGB。保存完整输入和诊断 sidecar，图像由真正 capture 返回，不用绘图库重绘。
- [ ] 追加仅改变主光方向的曝光，预先固定桌面阴影／高光 ROI；记录真实图像差值，避免用自动曝光掩盖光照响应。
- [ ] 追加帧跳转／camera 跳切／epoch 改变／新材质 scene key；验证 PT 累积失效。对确定性传感通道，与新会话同输入精确或按声明深度容差比较；随机 PT 颜色不能要求不同设备 bitwise 相等。
- [ ] 输入对象采集前后逐值／序列化摘要一致，源物理 step 调用次数为 0。没有接入物理 owner 的程序桌也记录为 authored，而非伪造 accepted_endpoint。
- [ ] 运行并独立审阅实际样图及数值。R0 参考双路径缺任一项则不勾选参考可行性；参考两档通过但 native raster 未通过时，保留参考成果，G3／自研首版仍不得放行。

### M5.5 R2：固定曝光、PBR／IBL 与历史管理

**前置：** G3 与 R1 真实资产通过。M1 提供纹理色域、UV 与法线约定，M2 提供具名环境光。M5 将固定曝光和显示变换记录在两条后端路径各自的配置中，集成人用同一组参数比较结果。

**独占开发路径：** `backends/_native_lighting.py`、`backends/_native_history.py`、`backends/_shaders/*`、`backends/_sapien_session.py`；测试 `rendering/tests/test_backend_quality.py`。这些是计划新增文件，只有确实需要独立职责时才拆分。

- [ ] 先在测试固定线性 HDR 到显示输出的独立灰阶与曝光夹具。实现线性光照缓冲、固定曝光和唯一显示转换；金属度／粗糙度／法线纹理按数据读取，底色按 M1 色域解码。不准自动曝光使两条路径的对照失去共同条件。
- [ ] 为 native raster 加入与当前材质一致的 PBR 和许可／哈希绑定的 IBL，预计算资源留在本后端；SAPIEN 同源灯光和环境的参数转换显式记录。先验证粗糙度阶梯和移动主光，再加任何画质优化。
- [ ] 定义本后端私有历史 key，包含 scene 内容身份、world、epoch、相机 ID／标定／位姿、源 frame、光照／材质配置和采样设置。完整 key 不同则失效；不同相机、世界不能共用累积颜色。先实现无历史基准，再验证累积或滤噪增益。
- [ ] 真实反例覆盖 reset、相机跳切、同相机移动夹指、材质／灯光切换和倒放；保存第一次切换后的原始与可选降噪图，不允许将旧图复制充当新的噪声下降。
- [ ] 用参考 PT 关闭降噪产生固定 seed 集合的采样阶梯；公共 fixture 的同源图由 09 固定 ROI 评估，native raster 的画质收益与 PT 的噪声／偏差分别报告。
- [ ] 以同一公共 capture 计时，分开上传、执行、等待、读回及编码。性能优化前后保持通道／尺寸／资产相同；提交加速命令本身的时间不替代完成延迟。

### M5.6 R3：确定性主可见性与传感读回

**前置：** R1、M3 标定、M6 输出校验；可与 M5.5 并行。R3 的强制真实设备路径为 native raster，参考 backend 支持的每个传感组合另列能力。

**独占开发路径：** `backends/_native_visibility.py`、`backends/_sensor_readback.py`、`backends/_shaders/*`、`backends/_sapien_session.py`；测试 `rendering/tests/test_backend_sensors.py`。共享 shader 文件由 M5 单一 owner 合并，不能两个子任务同时覆写。

- [ ] 建立固定像素中心的独立主可见性／ID 输出，实例 ID 采用整数 attachment／格式；禁止多采样颜色 resolve、滤波或随机 PT 累积产生混合 ID。若 PT 声称支持真值 AOV，须另取确定性主可见性，不复用反射射线命中。
- [ ] 用相机空间前向距离产生米制深度，不把 Vulkan 非线性 depth buffer 原值作为米；相机近远裁剪按光轴定义，离轴像素不能按射线长度误裁。
- [ ] 先写解析前平面、斜平面、遮挡双物体和屏幕四角的正负用例；再实现翻转、端序、pitch 压紧和背景转换。RGB、depth、ID、VALID 的联合请求必须共用同一曝光和可见性语义。
- [ ] 将 staging 资源和 GPU-to-host 等待集中在私有读回层；拷贝完成后生成独立 bytes，再归还 staging。增加两次 capture 交错读旧数据、读回失败后重试及 close 后读取旧包的测试。
- [ ] 接入同一真实外部／腕部帧，旋转手腕、改变遮挡和 reset；由 M3/M6 独立参考量化像素投影、深度和标签，backend 不修改参考 q／相机以适配自己的结果。

### M5.7 R4：多世界资源共享、时间隔离与批读回

**前置：** R2 和 R3；保留现有单场景三方法协议。可先用多个会话串行完成正确性，再实现后端内部调度；新增公开 batch 方法须由 08 单独冻结，不以私有接口绕过公共验收。

**独占开发路径：** `backends/_native_resources.py`、`backends/_native_scheduler.py`、`backends/_native_history.py`、`backends/_sensor_readback.py`、`backends/_sapien_session.py`；测试 `rendering/tests/test_backend_multiworld.py`。

- [ ] 先写 1／2／8 世界、每世界 2 相机的身份与隔离测试。保持实例 ID 在世界内稳定，跨世界用 FrameKey 的 world_id 组合标识；不擅自将世界序号混入公开像素 ID。
- [ ] 仅共享内容哈希一致的不可变 mesh／texture／pipeline 资源；每场景的实例变换、灯光绑定、输出 target 和时间历史独立。引用计数关闭一个会话后，另一会话的共享资源仍有效。
- [ ] 先以显式串行提交和完整读回为基准，再把不同相机命令安排到内部批次；逐个输出保留自己的曝光键和时间。fence 完成条件覆盖该批全部目标，不能因第一台相机完成就发布整个批次。
- [ ] 实现可选 staging 池及批读回，核每个 plane 的 offset／stride／长度；构造大小不同的相机和读回中途失败反例，拒绝跨视图覆盖与旧 target 内容泄漏。既有 CPU-owned bytes 合同不变。
- [ ] 对一个世界 reset／换材质／相机跳切，验证仅对应历史失效；其他世界的 ID／VALID 与单跑基准完全一致，depth 满足 09 的差值门。重试失败视图不能再次推进任何源物理状态。
- [ ] 原生和 SAPIEN 的并发／复制能力分别实测。只有真实接口支持且有同步证据时才开启设备互通，否则具名 HOST 复制；禁止跨后端借用资源来掩盖任一能力不足。
- [ ] 每种规模记录预热后 300 个全部机位完成的批次帧、p50/p95、总相机帧率和显存峰值；附资源共享命中、上传／读回字节、同步与编码时间。R4 没有额外最低 FPS 门，不把串行正确性包装成 GPU 并行性能。

## 5. 可判定的量化门与后续路线

以下新增 R0 小门是本计划的执行目标，非已有成绩；公共验收冻结在 09，不能看到结果后调整 ROI／阈值。

| 阶段 | 指标及通过条件 |
| --- | --- |
| 通用合同 | 错 scene、world、epoch、revision、tick、phase 六类各有负控，全部在返回 Capture 前拒绝；关闭后调用全部拒绝 |
| 图像布局 | RGB 恰好 W×H×3 bytes；depth／ID 各 W×H×4；VALID W×H；返回 plane 集合及顺序与请求一致，无缺失／额外通道 |
| R0 真图及 G3 原生门 | 同一输入的光栅与 PT 均由具名设备路径产出 RGB，传感路径具名。G3 另要求 native raster 同源四通道通过：至少含背景及三个预定材质实例的正 ID，各内部测点具有有限正深度；miss 为 depth=+inf、ID=0、VALID=0。PT 支持的附加通道按能力表另验 |
| R0 光照响应 | 固定 ROI 内移动主光后，至少 32 个像素至少一个 RGB 分量变化 ≥2/255；同时保存两张原图，人工检查变化符合光照而非视角／资产改变 |
| R0 状态与设备 | 物理步数 0；输入摘要不变；请求与实际 backend/path 一致；Vulkan validation error 数为 0，warning 单列说明 |
| 所有权与清理 | 20 次 open/capture/close 后内部活动资源计数回到基线；已有 Capture bytes 不变。设备总显存另测，不能用内部计数宣称整个进程零泄漏 |
| R2 交互 | native raster 1280×720 单相机预热后 p95≤33.3 ms；与同源高采样参考 PT 配图。native PT 若实现另列配置及成绩 |
| R3 标定 | native raster 640×480、0.25–2 m、离轴标记投影≤1 px；同几何且距轮廓≥2 px 的 depth 最大误差≤2 mm，有效 interior ID 全一致；边缘单列 |
| R4 扩展 | 1→2→8 世界，各 2 台 320×240；单世界对照，reset 仅影响目标世界；预热后 300 帧 p50/p95、帧率、显存、同步／复制／编码分别报告 |

R2 参考 PT 的未降噪采样阶梯与 ROI 噪声随采样总体下降由 09 固定；不要求 raster 与 PT 像素相等，也不用 PSNR 一个数替代材质、阴影、曝光和标定检查。R4 的训练互通在实际验证后才声明 zero-copy；同步消费完成才计时，Python 提交返回时间不能当渲染完成时间。

## 6. 开发环境与合并条件

- [ ] 保持基础标准库安装；SAPIEN 放独立可选 extra／参考环境，native 编译依赖放专属构建环境。08 owner 将经实际安装验证的版本写入锁定依赖和构建说明。
- [ ] 明确 shader／扩展文件的 wheel package-data 与平台 tag；从仓库外安装 wheel 后执行真实 open/capture，不能只在源码树导入成功。
- [ ] 先执行 `python -m pytest rendering/tests/test_backends.py -q`；设备环境再显式执行 `test_native_backend_device.py` 和 `test_sapien_backend_device.py`。如引入 device markers，由 08 在独立项目配置注册；未运行的设备测试不计 PASS。
- [ ] 会话合同、native 实现、参考实现可分别合并且如实标注能力。G2 的记录器与 M4 联测不依赖原生设备先合入；R0 可先完成参考双路径。G3 及首版里程碑必须 native raster＋真实参考 PT 同源通过，再按 R1–R4 验收，不能以纯 mock 的 GREEN 替代。
- [ ] M1/M2/M3/M4/M6/M7 对资产解释、相机轴、帧身份和 bytes 格式无冲突；源物理模块无改动，无隐式 fallback。
- [ ] 将机器、驱动、实际路径、依赖版本、失败日志、原图和量化结果交 09，公共入口文档由 08 更新。native PT 未实测的状态明确保持 unavailable，不勾选支持。
