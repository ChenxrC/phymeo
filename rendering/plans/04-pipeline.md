# M4：渲染请求与原生阶段规划实施计划

> **状态：** 实施计划。公开接口以当前骨架为基线，下面的实现与验收均为后续任务。

**目标：** 将现有 `CaptureRequest` 转为可验证、确定性的原生阶段序列，保证显示处理不改变传感通道，并为 R0–R4 提供稳定请求入口。

**架构：** 保留标准库数据合同和 `PassPlanner`，增加一个小型请求检查函数及默认规划器。M4 只决定阶段顺序，不拥有设备、资源、场景上传、异步调度或图像。实际执行和同步由 M5 的 `RenderSession` 完成；SAPIEN 参考后端直接实现高层会话协议，不模拟原生阶段。

**技术栈：** Python 3.11+ 标准库；测试使用 pytest。基础发行包不引入 NumPy、Vulkan、SAPIEN 或 Phymeo 依赖。

**依据及公共入口：** [计划总入口](README.md)、[集成计划](08-integration.md)、[验收计划](09-acceptance.md)、[现有包合同](../README.md)、[R0–R4 路线图](../../docs/rendering/README.md)。其中 08/09 由集成负责人统一维护。

## 1. 所有权与接口依赖

| 类型 | 文件／负责人边界 |
| --- | --- |
| M4 产品独占 | `rendering/src/phymeo_rendering/pipeline.py` |
| M4 测试独占 | `rendering/tests/test_pipeline.py` |
| 只读依赖 | M3 的 `cameras.py`、M6 的 `outputs.py`、M2 的 `scene.py` |
| 消费方 | M5 native backend；M7 adapter／调用者创建请求 |
| 不在本模块修改 | `backends/*`、包级 `__init__.py`、pyproject、公共 README、08/09、物理包 |

现有公开类型必须保留：

```python
class RenderPath(Enum):
    RASTER = "raster"
    PATH_TRACE = "path_trace"

@dataclass(frozen=True, slots=True)
class CaptureRequest:
    camera: Camera
    path: RenderPath
    channels: tuple[Channel, ...]

class PassPlanner(Protocol):
    def plan(self, request: CaptureRequest) -> tuple[NativePass, ...]: ...
```

保留六个 `NativePass`：`VISIBILITY`、`SHADOWS`、`LIGHTING`、`TRACE`、`DISPLAY_TRANSFORM`、`SENSOR_EXPORT`。本批新增的公开符号仅为：

```python
def validate_capture_request(request: CaptureRequest) -> None: ...

class DefaultPassPlanner:
    def plan(self, request: CaptureRequest) -> tuple[NativePass, ...]: ...
```

`validate_capture_request` 不返回修正后的请求，不替换相机或补全通道；结构类型错误抛 `TypeError`，合法类型但空通道、重复通道等值错误抛 `ValueError`。相机内部校验由 M3 公共合同负责；在 M3 的有效相机接口确定前，M4 不复制相机投影公式或自建宽松校验。输入拒绝不能以悄悄移除通道的方式成功。

采样数、反弹、随机种子和滤噪属于 M5 的具体后端配置；当前 `CaptureRequest` 不承载这些字段。若以后确需逐请求变更，先由 08 冻结跨模块合同，再改消费者。

## 2. 默认阶段表与语义

| 请求 | 精确有序阶段 |
| --- | --- |
| RASTER，仅 RGB | VISIBILITY → SHADOWS → LIGHTING → DISPLAY_TRANSFORM |
| RASTER，仅任意非空传感通道子集 | VISIBILITY → SENSOR_EXPORT |
| RASTER，RGB 与传感通道混合 | VISIBILITY → SHADOWS → LIGHTING → DISPLAY_TRANSFORM → SENSOR_EXPORT |
| PATH_TRACE，仅 RGB | TRACE → DISPLAY_TRANSFORM |
| PATH_TRACE，仅任意非空传感通道子集 | TRACE → SENSOR_EXPORT |
| PATH_TRACE，RGB 与传感通道混合 | TRACE → DISPLAY_TRANSFORM → SENSOR_EXPORT |

传感通道是 `FORWARD_DEPTH`、`INSTANCE_ID`、`VALID`。规划只读请求，因此不能根据灯光数、纹理内容、GPU 能力或历史帧改变上述结果。SHADOWS 在无投影灯场景下可由执行端完成空工作，但仍是同一路径的阶段，不是改用其他渲染器。

阶段序列不是资源连接图。`DISPLAY_TRANSFORM` 仅消费线性颜色；`SENSOR_EXPORT` 消费几何真值资源，绝不消费经过色调映射、降噪或 RGB 累积后的数据。即使两者按此顺序执行，也不表示前一阶段的图像成为后一阶段输入。TRACE 路径的深度／ID／有效掩码采用确定性的针孔主可见表面，不能平均随机抖动射线的 ID 或将反射物体作为直接可见实例；M5 用真实设备测试验证。

同一请求重复规划结果完全相同；输入 channels 顺序不影响阶段表。输出 Capture 的 plane 顺序由 M5/M6 按请求顺序保持，M4 不排序或重写请求。

## 3. 可并行实施任务

### M4.1 请求拒绝合同

**前置：** M3 相机类型、M6 Channel 枚举保持当前公开名字。可与 M1 资产读取、M5 设备探测并行。

- [ ] 在 `test_pipeline.py` 定义标准曝光，使用 `FrameKey("r0-table", "world-0", 0, 1, 1, "authored")`、`Pose()`、640×480、fx=fy=500、cx=320、cy=240、near=0.1、far=5.0。
- [ ] 先写失败测试：空 channels、同一个 Channel 两次、字符串通道、字符串 path、list channels、非 Camera 的 camera、非 CaptureRequest 入参；分别检查明确的异常类别和字段诊断。
- [ ] 由执行负责人运行 `python -m pytest rendering/tests/test_pipeline.py -q`，确认失败来自缺少实现／拒绝，而非测试导入错误。
- [ ] 实现 `validate_capture_request`，不增加隐式默认值、能力探测或设备初始化。
- [ ] 重跑该文件；增加合法的四通道请求正控，确认对象字段没有被修改。

最小测试形状如下，`exposure` 是同文件按照上面的确定参数构造的 Camera fixture：

```python
from dataclasses import replace
import pytest
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import (
    CaptureRequest, DefaultPassPlanner, RenderPath, validate_capture_request,
)

def test_duplicate_channels_rejected(exposure):
    request = CaptureRequest(exposure, RenderPath.RASTER, (Channel.RGB,))
    validate_capture_request(request)
    with pytest.raises(ValueError, match="channels"):
        validate_capture_request(replace(request, channels=(Channel.RGB, Channel.RGB)))
```

### M4.2 确定性阶段规划

**前置：** M4.1；不等待图形设备，不能用 mock 结果声称真实图像通过。

- [ ] 先写六行阶段表的精确元组断言，覆盖仅 RGB、仅深度、混合三种请求与两种路径。
- [ ] 枚举四个 Channel 的全部 15 个非空子集，对两种 RenderPath 共 30 个组合验证：RGB 存在当且仅当包含 DISPLAY_TRANSFORM；传感通道存在当且仅当包含 SENSOR_EXPORT；每阶段至多一次。
- [ ] 对混合请求的两种通道顺序验证相同阶段序列，且请求 channels 未被重排。
- [ ] 实现 `DefaultPassPlanner.plan`，先调用请求校验，再按上表选择明确元组；不添加缓存、插件注册或可变全局状态。
- [ ] 运行本文件并保留 RED/GREEN 结果，检查 PATH_TRACE 不被静默映射为 RASTER。

```python
from phymeo_rendering.pipeline import NativePass

def test_depth_bypasses_display(exposure):
    request = CaptureRequest(exposure, RenderPath.RASTER, (Channel.FORWARD_DEPTH,))
    assert DefaultPassPlanner().plan(request) == (
        NativePass.VISIBILITY, NativePass.SENSOR_EXPORT,
    )
```

### M4.3 交给 M5 的执行接缝

**前置：** M4.2 与 M5 候选的会话命令记录器；可与 M7 已保存帧适配并行，不等待原生设备实现合入。

- [ ] 将上述阶段表作为 M5 native 实现的公共依赖；同一会话的请求不可绕过校验。
- [ ] M4 保留 planner 的纯合同测试，M5 在自己测试中用命令记录器检查计划实际调度顺序；不把测试 spy 作为生产 backend。
- [ ] 联合检查 RGB 显示变换只执行一次，ID／深度通道没有经过显示链。真实 pixel 校验归 M5/M6 与 09，不用阶段名字代替像素证据。
- [ ] 向集成负责人交付导出符号、测试命令、失败边界；包级 re-export 由公共入口 owner 合并。

## 4. 可直接判定的验收指标

| 项 | 通过门 |
| --- | --- |
| 请求组合 | 2 路径 × 15 非空通道子集，30 组合全部通过；7 类非法输入全部明确拒绝 |
| 确定性 | 每组合重复规划 100 次，结果元组逐项一致，输入对象保持原值 |
| 显示隔离 | 仅传感请求的 DISPLAY_TRANSFORM 数量为 0；有 RGB 为 1 |
| 路径隔离 | RASTER 的 TRACE 数量为 0；PATH_TRACE 的 TRACE 数量为 1，无静默降级 |
| 纯规划 | 导入和 plan 不加载 Phymeo/SAPIEN/Vulkan 模块，不读取文件、不创建窗口、不申请 GPU 资源 |
| 交付边界 | spy 测试仅证明编排；没有真实 backend 的结果不得标记 R0 完成 |

R0 的双真实路径参考可行性要求公共入口的真实光栅和真实路径追踪，可以先由 SAPIEN 两档采集；M4 的两种枚举可用不等于设备路径可用。G3 合并放行及自研引擎首版另有硬门：native raster 必须在同一标定桌实测通过，并保有真实参考 PT。SAPIEN 两档不能替代这一原生门。native path trace 是独立能力，未实现时由 native 会话明确拒绝，不妨碍明确限定为 native raster＋参考 PT 的首版范围。

R2 的 1280×720 单相机 p95≤33.3 ms 由完整设备路径测量，不能用 plan 的耗时代替。R3 深度和标定、R4 八世界输出也由真实 backend 联合验收；M4 不提前引入多世界 scheduler 或跨帧渲染图。

## 5. 开发依赖、合并条件与交接

- [ ] 基础包仍仅 Python 3.11+ 标准库；pytest／构建工具属于独立开发环境，不进入基础 runtime requirements。
- [ ] 在独立安装后的环境执行 `python -m pytest rendering/tests/test_pipeline.py -q`；根项目默认 pytest 路径不包含此目录，必须显式指定。
- [ ] M3/M6 与 M5 对校验职责、阶段语义、通道顺序无分歧；不擅自修改共享类型字段。
- [ ] G2 在集成候选中将 M4 与 M5 命令记录器联测后合并纯规划实现，不要求原生 M5 先合入。M5 真实设备开发随后消费已冻结的 M4；模块合并与 R0／G3 设备里程碑分开标记。
- [ ] 08 记录接口版本及安装命令，09 收集实际测试及图像证据。
