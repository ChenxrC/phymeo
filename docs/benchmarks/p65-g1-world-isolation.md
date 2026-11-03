# G1 共址世界的隔离与查询成本

2026-11-03，在同一台 Windows 11 主机和 RTX 4060 Laptop GPU 上，Phymeo HOST、tyx CUDA、Newton 1.7.0 CPU 与 Warp 1.17.0 CUDA 都完成了 W=1、2、8、32、64 的逐世界调用和单模型查询，共 40 个配置。这里测量四球的**静态碰撞查询**。宽相使用 Phymeo 原有 `all_pairs` 与 Newton `nxn`。[冻结输入及断言](../../tests/scale/cases.py) · [测量入口](../../scripts/benchmark_worlds.py) · [阶段计划](../superpowers/plans/2026-11-03-tyx-gpu-multiworld.md)

## 世界归属及负控

每个世界使用同一原点，四球半径均为 1/8 m，x 坐标依次为 0、3/16、2、4 m。外部世界 ID 为 17+19i，世界内重用刚体编号 0～3、形状号 0 和材料号 0。用打乱的世界与形状次序建模。Phymeo 逐行核对局部归属、四球的外包盒及每一个跨世界过滤原因。Newton 在零间距下复制模型，通过实际 `shape_world`、`shape_world_start`、`shape_body`、`body_world` 与状态绑定外部 ID。按局部顺序将各引擎的法向统一为 A 到 B。[Phymeo 重排与还原测试](../../tests/scale/test_world_isolation.py) · [Newton 实际候选和两侧球面见证](../../tests/scale/newton_g1.py)

基线每世界只有 (0,1) 重叠，间隙 −1/16 m，法向 +x，球面的 x 坐标分别为 1/8、1/16 m。把选定世界的第二球移到 5/16 m 后，只消失该世界的接触。回置后重新生成的全部见证与初始见证一致。W=64 移动的是世界 625，候选与接触均由 64 降为 63。零余量的查询会剔除该分离对，另一次 Phymeo 扩大宽相的查询读得 +1/16 m 和球面 x=1/8、3/16 m。Newton 在零余量配置下验证了全体姿态读回和活动对名单。几何误差限为 1 µm，法向误差限为 10⁻⁵。另测 65 个四球世界，现有完整模型在超过 256 个形状时明确拒绝。

| W | Phymeo 同模型实际发布行 | 世界内行 | 跨世界过滤行 | 初始接触 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 6 | 0 | 1 |
| 2 | 28 | 12 | 16 | 2 |
| 8 | 496 | 48 | 448 | 8 |
| 32 | 8,128 | 192 | 7,936 | 32 |
| 64 | 32,640 | 384 | **32,256** | 64 |

## Phymeo 查询墙钟

每格为**逐世界串行 / 同模型**的暖态中位数，单位 ms。两种布局都保留各自的工作区，每次从同一固定姿态调用公开 `detect`。逐世界布局返回 W 份六行诊断，同模型返回一份全形状配对诊断。tyx CUDA 计算包围归约和配对标记，随后同步、读回。外向包络、静态球面见证与公开分类行在 HOST 生成。[HOST 原始样本](../../artifacts/p65-wave4-scale-20261103-01/g1-host/report.json) · [CUDA 小档](../../artifacts/p65-wave4-scale-20261103-01/g1-tyx-cuda-small/report.json) · [CUDA 大档](../../artifacts/p65-wave4-scale-20261103-01/g1-tyx-cuda-large/report.json)

| W | HOST | tyx CUDA |
| ---: | ---: | ---: |
| 1 | 0.351 / 0.315 | 0.515 / 0.482 |
| 2 | 0.639 / 0.712 | 1.132 / 0.839 |
| 8 | 2.906 / 4.404 | 3.874 / 3.852 |
| 32 | 10.371 / 40.680 | 16.867 / 34.317 |
| 64 | 20.733 / 141.548 | **31.698 / 117.513** |

W=64 的 tyx CUDA 工作区构造墙钟为 40.230 / 20.742 ms，首次完整查询为 831.433 / 126.533 ms，暖态 p95 为 32.143 / 128.550 ms。逐世界 / 同模型的原有主机配对表分别占 4,608 / 391,680 B。工作区构造时静态上传分别为 192 / 3 次。以下计数取同档第一份暖样本，工作区计数与独立的进程级 `tyx.runtime_stats()` 传输字节及启动数一致。

| 每次查询一组 64 个世界 | 逐世界 | 同模型 |
| --- | ---: | ---: |
| 完成 CUDA kernel / 工作区同步 | 128 / 64 | 2 / 1 |
| H2D / D2H / D2D 字节 | 6,144 / 7,680 / 0 | 6,144 / 136,704 / 0 |
| 工作区新增分配 / 静态重传 | 0 / 0 | 0 / 0 |
| 该样本端到端墙钟 ms | 31.582 | 117.513 |
| 包络 / 宽相 / 窄相计时 ms | 7.358 / 12.136 / 11.647 | 9.813 / 0.471 / 106.364 |

同模型首个暖样本的 106.364 ms 来自现有 `narrow_ns` 区间，包含主机配对分派和记录构造。当前计数没有再细分这些工作，在该样本中也不能把总墙钟之差归给某一个内核。构造 64 个 CUDA 工作区时每八个采一次 SMI，保留 512 MiB 停线阈值。tyx 大档构造快照中该 W=64 串行布局最低可用 7,822 MiB。设备总量为 8,188 MiB，这些整机快照不能给出本进程显存峰值。

## Newton 活动缓冲对照

同一坐标用 Newton 的独立模型逐次调用，或用 `ModelBuilder.replicate(spacing=(0,0,0))` 生成一个模型。计时覆盖 `collide`、`eval_rigid_contact_kinematics`、显式设备同步和活动候选/接触读回。用实测原始 support 点、有效 margins 和法向恢复两个球面点。Newton 发布活动对，不生成 Phymeo 的跨世界过滤诊断行。因此这里单列其测量，不计算跨引擎速度比。[CPU 小档](../../artifacts/p65-wave4-scale-20261103-01/g1-newton-cpu-small/report.json) · [CPU 大档](../../artifacts/p65-wave4-scale-20261103-01/g1-newton-cpu-large/report.json) · [CUDA 小档](../../artifacts/p65-wave4-scale-20261103-01/g1-newton-cuda-small/report.json) · [CUDA 大档](../../artifacts/p65-wave4-scale-20261103-01/g1-newton-cuda-large/report.json) · [Newton 多世界合同](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)

| W | CPU 暖态中位 ms，逐世界 / 同模型 | CUDA 暖态中位 ms，逐世界 / 同模型 |
| ---: | ---: | ---: |
| 1 | 0.339 / 0.337 | 0.731 / 0.715 |
| 2 | 0.649 / 0.363 | 1.390 / 0.760 |
| 8 | 2.734 / 0.394 | 6.716 / 0.800 |
| 32 | 11.604 / 0.397 | 26.074 / 0.840 |
| 64 | 24.665 / 0.681 | 68.071 / 0.936 |

W=64 同模型实际候选容量 384，得到 64 个表面接触。CUDA 同模型首个暖样本由 runner 发起 11 次数组读回，数到的有效数据为 4,104 B。这个数字只计调用方读取的数组内容。Warp 的总传输、隐式同步次数、纯设备时间及分配器峰值均未测。

## 复现与统计范围

原始报告位于集成工作树的本机忽略目录 `artifacts/p65-wave4-scale-20261103-01`，包含源码哈希、每次进程守卫、原始退出码、各模型的归属映射、完整活动接触、五个暖样本及计数。固定工具提交为 `894862b`，Phymeo 使用 Python 3.12.14，Newton 使用独立 Python 3.12.13，两侧 NumPy 均为 2.5.4。Windows 11，NVIDIA 驱动 566.07。

在仓库根运行，替换成对应引擎的解释器并指定新的输出路径。第一条使用项目环境，第二条使用 Newton 1.7.0 环境。全阶梯可省略 `--worlds`。Phymeo 路径由当前工作树 `src` 提供。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts/benchmark_worlds.py --engine phymeo --device cuda:0 --worlds 1 2 --cache-root .cache/g1-tyx --out g1-tyx-small.json
.\.venv-newton-1.7.0\Scripts\python.exe scripts/benchmark_worlds.py --engine newton --device cuda:0 --worlds 1 2 --cache-root .venv-newton-1.7.0/warp-kernels --out g1-newton-small.json
```

冷查询指新工作区的首次查询，编译缓存可复用。暖态先完成初态、留出与恢复，再做两次预热和五次单查询窗口。墙钟包括 API 要求的同步或读回，逐行断言、外部状态修改和 JSON 写盘在计时外。每档每布局一次进程内测量。这里没有推进动力学子步，也没有执行计划中三次独立进程、每次二十个三十二子步窗口的吞吐验收。五样本最近秩 p95 等于该组最大值。
