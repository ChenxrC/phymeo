# 多面布与有限连杆的跨世界材料力合批

`ph.TyxMultifaceContactBatch` 让不同世界在同一子步的入步、中点、末态共享 TYX 法向力设备页。四个世界各有八个持续受载面，在 RTX 4060 Laptop GPU 上推进同样的四步，实际核提交从 **48 次降到 12 次**。完整窗口中位墙钟从 **176.71 ms 变为 182.43 ms**。本机这档规模的端到端耗时增加了 3.24%。[实现](../../src/phymeo/backends/tyx_multiface_contact.py) · [同进程对照](../../scripts/benchmark_multiface_phase_batch.py)

## 调用

以下代码可在仓库根目录运行，使用自带的连通布条模型。实际程序传入自己的 `CompliantClothContactProgram`、完整初态和每世界外力。

```python
import sys

import phymeo as ph

sys.path.insert(0, "tests/scale")
from multiface_load_cases import H, make_world

worlds = {43: make_world(0, 2, None), 8: make_world(1, 8, None)}
specs = [(k, ph.ClothLinkWorld(w.program, w.initial)) for k, w in worlds.items()]
settings = ph.StepSettings(substeps=1, record_substeps=True)
with ph.TyxMultifaceContactBatch(specs, "cuda", settings=settings) as batch:
    frames = batch.step(H, {k: w.controls[0] for k, w in worlds.items()})
    assert all(f.report.accepted for f in frames.values())
    for phase in batch.last_phases:
        print(phase.phase, phase.world_offsets, phase.sites)
    batch.reset([8])
    restored = batch.snapshot([8])[8]
```

每个世界保留普通 `SceneStepper` 的 FP64 最近几何、材料权重、共享节点累加、两个半步的路径检验、积分和完整报告。设备采用相同的 FP32 法向及切向定律。每一阶段的 `world_offsets` 是半开站点区间，`faces` 将区间细分到原始机器人、连杆、布面和三个本地节点 ID。相同数字或面名可以在不同世界复用。各站点的权重、双方作用力和中点位置可在 `frames[k].report.substeps[i].diagnostics["robot_contact"]["cloth_links"]` 中读取。[材料装配与反力](../../src/phymeo/solvers/cloth_link_contact.py) · [原积分、阈值及路径判断](../../src/phymeo/solvers/compliant_cloth_contact.py)

## 资源与失败语义

- 一批可登记 1 至 64 个具有布面与结构胶囊的世界。每个世界使用一个可暂停的调用栈。HOST 按登记顺序逐个运行到阶段提交点，调用者线程集中拼页并提交设备工作。每个 128 站点页只执行一次。每阶段默认最多接受 1,048,576 站点，可用 `site_capacity` 收紧。九个世界的首阶段实测 144 站点，自动拆成两页；两个子步均保留三个独立阶段。
- 与胶囊分离的布仍接受外力、弹性与运动推进。所有世界合计零站点时该阶段设备提交数为零。此入口要求存在真实布与结构胶囊，支持原中点程序中没有背衬垫和布自接触的模型。原单面 `TyxWorldContactBatch` 继续提供其既有接口。
- 普通物理拒绝沿用原世界报告。专项覆盖入步胶囊轴穿面，以及完成末态采样后越过硬穿入预算。失败世界保留入步状态，其余世界可以前进。容量、设备、输入或阶段协议异常使整批保持上次公开状态和阶段图。已发生的设备工作仍计数。一个反例在入步真正提交了 4 站点，另一张布随后进入力层，中点超过容量 4；增大容量后在同一对象上重新调用，得到与逐世界参考相同的全部结果。注入设备故障后下次调用会重建法向层。所有等待调用都在返回或抛错前退出，`close()` 回收线程和设备资源。[异构、换序、零接触、错误阶段、恢复与分页测试](../../tests/scale/test_tyx_multiface_batch.py)

## 同输出 CUDA 计时

使用同一组有不同控制输入的连通布和有限质量滑动连杆，每世界参考面积为 0.25 m²。每个窗口从相同物理状态推进 4 × 1/256 s。两条路径均预热，五对窗口按 AB、BA 交替，计时覆盖完整公开物理步与报告。复位、哈希、独立力校验放在窗外。比较每一步完整状态、输入产生的节点/连杆运动、残差、报告、材料诊断和子步快照；只将已有 `SceneSubstepReport.timings` 键的耗时数值归零，原键和原时钟另存。合批路径的每世界预测时钟会包含等待邻居的时间，计时表使用调用者整窗墙钟。

| 每四步，4 世界 × 8 面 | 普通逐世界 CUDA | 跨世界 CUDA |
| --- | ---: | ---: |
| 墙钟 p50（ms） | 176.71 | 182.43 |
| 五窗范围（ms） | 174.12–190.38 | 173.64–190.54 |
| 真实材料站点 | 416 | 416 |
| 核提交 / 上传 / 下载次数 | 48 / 48 / 48 | 12 / 12 / 12 |
| 上传字节 | 319,488 | 79,872 |
| 下载字节 | 122,880 | 30,720 |

四步共读出 128 个中点面样本，逐面法向力范围为 **0.126994–0.141340 N**。两条路径每帧的物理哈希一致。独立验算按原节点权重重建材料位置，逐站点核对力律及其对连杆的反力，再检查关节广义力以及面的位置、法向和面积变化。表中耗时包括 HOST 几何、栈切换、积分与发布。单独的纯 GPU 核耗时未测。[输入和独立核验](../../tests/scale/multiface_load_cases.py)

复现命令为 `python scripts/benchmark_multiface_phase_batch.py --pairs 5 --cache-root <缓存目录> --out <新JSON>`。父树 `artifacts/p65-multiface-tyx-batch-20261105-01/multiface-paired-cuda/measurements.json` 保存逐窗原时钟、计数、完整参考与批结果和物理核验。固定旧源码 `6c3f377` 的首红保存在 `multiface-public-red`。最终 CUDA 七项专项与两项近邻共 9 通过；interpret 六项专项与三项近邻共 9 通过，迟发超容量另有一项通过。Ruff 检查与格式检查均成功。
