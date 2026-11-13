# G2b-A：共享设备预算与 32 构型展示用法

2026-11-13 使用说明。HOST/解释40项、CUDA24项通过；包外32构型interpret12步及CUDA256步均STRICT_PASS，视频已经生成。此文中的示例未另行执行；实际执行记录见[验收报告](p65-g2ba-resource-gallery-report-20261113.md)。范围以[本批合同](p65-g2ba-resource-gallery-20261113.md)和[当前入口](p65-continuation.md)为准。

## 一个共享预算，多个真实模型

`MuJoCoTyxResourceBudget` 限制已登记 solver-bank 的 CUDA **数组 payload**。同一个对象可以传给多个 `MuJoCoRuntime` 或 `MuJoCoTyxHeterogeneousBatch`；其锁、当前占用与峰值共用。超过旧的 8 models 或 16 worlds 必须显式提供预算；当前 facade 上限是 32 models / 32 worlds。各模型原有 DOF、接触及约束容量门不变。

下面的 `xml_sources` 是 32 份已符合当前 TYX 准入域、且 timestep 均为 `1/128` 的 MJCF 字符串。每份 MJCF 仍由 CPU MuJoCo 编译，之后的 `mode="cuda"` 物理步使用 TYX。`initial_states` 可为逐模型的 `MuJoCoInitialState`，或每项 `None` 使用原生初态。

```python
import numpy as np
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
)

budget = MuJoCoTyxResourceBudget(256 * 1024 * 1024)
compiled = tuple(compile_mujoco_xml(xml, require_asset=False) for xml in xml_sources)
models = tuple(MuJoCoTyxModelSpec(f"mechanism-{i}", item)
               for i, item in enumerate(compiled))
worlds = tuple(MuJoCoTyxWorldSpec(11 + 13*i, spec.key, initial_states[i])
               for i, spec in enumerate(models))

with MuJoCoTyxHeterogeneousBatch(
    models, worlds, mode="cuda", resource_budget=budget,
) as batch:
    controls = {key: MuJoCoControl(np.full(batch.layout[key].nu, .04))
                for key in batch.world_keys}
    batch.set_controls(controls)  # Exact complete world membership is required.
    checkpoint = batch.checkpoint()
    accepted = batch.step(1/128)
    states = batch.integration_snapshot()
    receipt = batch.last_step
    resources = batch.resource_snapshot()
    batch.restore(checkpoint, expected_revision=batch.checkpoint_revision)
    batch.reset((batch.world_keys[0],))  # Other worlds retain their current state.
    batch.set_controls(controls)
    batch.step(1/128)  # A real successor after restore/reset.

remaining = budget.snapshot()
```

每个 world 独立；精确原 source/capacity 签名决定 bucket，相同来源可以合桶，不同来源保留各自模型、维度与 native 地址。一个 facade 按桶执行设备核，全部候选验证后才共同发布。这里没有承诺一个融合 kernel、32 个 kernel 同时运行、跨 world 碰撞或零读回。不同模型具有相同 world ID 之外的本地 body/joint 编号也不会被合成一个机械系统。

## 读取资源与理解边界操作

`batch.resource_snapshot()` 返回只读 mapping：`manifests` 为每桶完整分配说明；`manifest_payload_bytes` 为清单之和；`buckets` 为各叶 runtime 详情；`budget` 是共享预算快照；`retained_owner_count` 表示尚保留的清理 owner。叶 runtime 的 `resource_snapshot()` 另含单份 `manifest`、`allocation` 和同一个共享 `budget`。

预算快照中的 `current_bytes` 是实际已登记、尚未确认释放的 CUDA payload，`reserved_bytes` 是构造前已预留但尚未返回数组 owner 的部分，`peak_bytes` 是实际登记 payload 的历史峰值。`unknown_retained_bytes` 是 `current_bytes` 的子集，不能再相加。成功关闭后的共享预算仍可能非零，因为其他 runtime、候选或未知释放 owner 仍持有资源；解释模式的清单只是预计 CUDA 布局，CUDA current 为 0，不代表 HOST 不占内存。

`admit(additional_bytes)` 是带共享锁的提前检查，本身不分配或收费；通常直接让 runtime/facade 管理，无须调用者按经验手算数组大小。构造前完整清单准入，clone/reset/restore 还要为旧 accepted owner 与候选同时存在的峰值预留空间。预算不足抛 `MemoryError`，不会先释放旧状态来腾空间；保留原 accepted 状态后可释放其他已确认可关闭的 owner，再重试同一请求。

`clone()` 是叶 `MuJoCoRuntime` 的公开方法，沿用同一个预算。当前 facade 的整体复制入口是已绑定的 checkpoint/restore，并没有新添 `batch.clone()`。一次叶 clone 会经历初始 owner 与替换 owner 重叠；facade 边界还会保留其他已准备桶，因此不能仅以最终稳态 payload 判断操作是否容得下。

若真实 `free()` 或 runtime 释放回调抛错，状态为 `UNKNOWN_RELEASE`：异常可能发生于副作用之前或之后，不能据此扣账。原 owner 与依赖强引用保留，普通后继被阻止，未知回调不再重试；没有公开的“强制清零预算”接口。普通成功 `close()` 才解除相应 charge。务必同时保存主错误与清理错误，不能把保存日志或清理失败解释成物理已回滚。

本片清单覆盖 base、fixed tree、floating tree、normal contact、mixed constraints 与 activation/CSR 扩展的六类 solver-bank 布局。CUDA context、JIT/cache、allocator overhead、CPU/native 模型或 arena、NumPy/JSON、外部 device-control 与 task-workspace owner 均不在这个预算里；没有真实全程 observer 的总显存/HOST峰值仍为 UNKNOWN。256 MiB 不是“进程最多占用256 MiB”的保证，也不是性能超过 Newton 的证据。

## 采集与离线视频

[固定案例](../../scripts/heterogeneous_gallery_cases.py)有固定根 R/P、固定根 BALL、FREE R/P、FREE BALL 四组各八个，模型结构/质量/轴等真实物理数据不同。展示完全无接触：geom mask 过滤全部碰撞。固定根组有重力；FREE 组为零重力太空机构，保非零初速度和真实电机驱动。所有世界常驻同一个 facade，每步完整推进并共同发布。

使用已配置 MuJoCo 3.3.7、TYX 与当前 Phymeo 的 Python；以下 PowerShell 命令假定当前目录为仓库根。输出目录必须是新的绝对路径，不能指向既有验收证据。采集和渲染应依次运行，不能抢占另一项目的 GPU 执行槽。

```powershell
$repoPath = (Get-Location).Path
python "$repoPath/scripts/run_heterogeneous_gallery.py" --mode interpret --steps 12 --output C:/Temp/phymeo-gallery-short-new
python "$repoPath/scripts/run_heterogeneous_gallery.py" --mode cuda --steps 256 --output C:/Temp/phymeo-gallery-cuda-new
```

源码外资格须把 `run_heterogeneous_gallery.py` 和 `heterogeneous_gallery_cases.py` 一起复制到源码外目录，使用已安装包及 `--installed-root <absolute-site>`；脚本检查所有已加载 Phymeo 模块来源。12 步保持原严格连续门；256 步同入步力学与单步推进仍严格，累计连续误差采用预登记功能门并完整保留原严格失配。`FUNCTIONAL_PASS` 同时 `STRICT_FAIL_RECORDED` 不等于严格通过。

采集输出 `manifest.json`、`sources/*.xml`、完整 `trajectory.jsonl.gz`、给渲染器的 `states.jsonl.gz` 及 `summary.json`。其中 `states.jsonl.gz` 逐帧含32个世界的真实 q/v/act/time/tick；完整2秒轨迹应有初态加256步，共257帧。XML、物理结构、状态文件及同次 CUDA 身份/计数/预算均有绑定；失败只留下实际 prefix，不伪造剩余帧。

仅完整功能通过的 CUDA 轨迹可以交给[离线渲染器](../../scripts/render_heterogeneous_gallery.py)：

```powershell
$ffmpegBin = 'C:/Users/38347/AppData/Local/Programs/FFmpeg/8.1.2/ffmpeg-8.1.2-essentials_build/bin'
python "$repoPath/scripts/render_heterogeneous_gallery.py" --input C:/Temp/phymeo-gallery-cuda-new --output C:/Temp/phymeo-gallery-video-new --ffmpeg "$ffmpegBin/ffmpeg.exe" --ffprobe "$ffmpegBin/ffprobe.exe"
```

渲染仅用保存状态做 native FK 和画面几何，不调用 `mj_step`，不插值运动、不更改 q。默认camera按完整轨迹一次确定并固定；追加 `--tracking-overview` 可使主镜头跟随实际几何，并在小窗保留原固定视角，画面与收据明确区分。257帧以30fps播放约8.57秒，仿真时长仍是2秒，速度标注 `30/128≈0.234375×`。视频、预览及 `render-receipt.json` 绑定同一状态SHA；视频展示不增加接触、长期支撑或性能资格。
