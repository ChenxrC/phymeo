# G2 批量运动观测与 Newton 自由落体的直接对照

RTX 4060 Laptop 上，每个物理步都取回并验证所有球的位置和线速度。W1 时 Newton 的三轮应用吞吐分别为 Phymeo `kinematics()` 的 **1.596、1.330、1.490 倍**。W32 时相应比值为 **0.903、1.305、1.457 倍**，两端各有领先的轮次。W32 首轮的 32 步窗 p50 仅相差 0.008 ms，Newton 的 p95 为 15.187 ms，Phymeo 的为 11.014 ms。各正式进程的解析误差均为 0。

| 世界数 | 独立配对 | Phymeo 首步 ms | Newton 首步 ms | Phymeo 32 步 p50 / p95 ms | Newton 32 步 p50 / p95 ms | Phymeo 世界子步/s | Newton 世界子步/s | Newton / Phymeo 吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 37.78 | 8.81 | 11.423 / 13.632 | 7.179 / 8.525 | 2,749 | 4,386 | 1.596 |
| 1 | 2 | 28.74 | 7.51 | 9.579 / 10.321 | 7.195 / 7.773 | 3,335 | 4,436 | 1.330 |
| 1 | 3 | 32.67 | 8.26 | 10.926 / 11.738 | 7.220 / 7.784 | 2,978 | 4,439 | 1.490 |
| 32 | 1 | 29.84 | 8.51 | 10.591 / 11.014 | 10.599 / 15.187 | 99,271 | 89,667 | 0.903 |
| 32 | 2 | 30.97 | 8.20 | 9.645 / 10.402 | 7.477 / 8.518 | 104,260 | 136,107 | 1.305 |
| 32 | 3 | 29.59 | 7.92 | 10.754 / 11.442 | 7.445 / 7.777 | 94,454 | 137,629 | 1.457 |

本表全部时间来自本批 12 个新 CUDA 进程。表中首步是每个新进程重新建模后、使用预热磁盘缓存时的首次真实子步。原完整检查点与 Newton 的对照，以及 Phymeo 两种发布接口之间的比较，各自保留独立收据。[完整检查点的跨引擎基线](p65-g2-newton-observed-free-fall.md) · [Phymeo 内部两种观测](p65-g2-kinematic-observations.md)

## 相同的应用任务

质量 2 kg、半径 1/8 m 的球从 (0,0,2) m 静止释放，重力为 (0,0,−8) m/s²，步长 h=1/128 s，无接触。W1 或 W32 在各自一个模型或工作区内推进，球处于共址、互相独立的世界。每步按登记的非连续世界键在 HOST 重排并写入六个 FP32 分量，校验全部位置和质心线速度。离散解为 v_z=−8nh、z=2−4h²n(n+1)。

Phymeo 调用原 `TyxWorldBatch.step`，得到本物理步的接受标记，再调用 `kinematics()` 读取当前已提交的状态和整数步号，从独立的六列数组提取位置、速度。Newton 1.7.0 的 `SolverSemiImplicit.step` 推进双缓冲自由刚体，显式同步后，实际下载 `body_q` 的 7 个 FP32 和 `body_qd` 的 6 个 FP32。球心等于刚体质心。两侧使用同一应用输出重排和校验函数。

两个 Python 环境隔离，正式轮只使用实际 CUDA。每进程先测创建、绑定和首个物理步，第 1 步后连续 5 个预热窗和 20 个计时窗，每窗 32 步，到 n=801 时停止，窗间不复位。时钟包含物理提交、同步、公开状态下载、数据整理与逐步解析核验。输出哈希及计数差分在窗外。p95 取第 19 个有序窗，吞吐分子为 20 窗已核验世界子步数，分母为这些窗的总壁钟。每轮每引擎每个 W 都是独立进程。第 1、3 轮先 W1 后 W32，每档先 Phymeo。第 2 轮先 W32 后 W1，每档先 Newton。

## 物理状态、设备与缓存

本批资格包含 Newton CPU、Phymeo interpret 和两端的实际 CUDA。四端 W1 的第 32 步均为 (0,0,1.7421875) m 和 (0,0,−2) m/s。另有两个不同初态的世界检验外部排序，所有 32 步六列状态逐点一致。正式 W1/W32 的冷步和 20 个计时窗在两引擎及三轮之间的完整 SHA 相同。n=801 时，各球的 z=−154.83642578125 m、v_z=−50.0625 m/s。两档全部状态 SHA 也分别与前批相同。

W32 每个正式进程测量 640 个物理步、20,480 个世界子步。Phymeo 0.1.0a0 / TYX 0.1.0a0 实测 640 次 CUDA 核提交及 640 次工作区同步，接受标记下载 81,920 B，`kinematics` 下载 1,064,960 B，总 D2H 1,146,880 B。完整检查点下载为 0 B，稳态设备分配及 H2D 都为零。球每步对外提供的六列有效数据为 24 B，实际还回传已提交的状态行及步号。Newton 1.7.0 / Warp 1.17.0 的 640 次物理步完成后各显式同步一次，1,280 个公开原生数组的下载合计 1,064,960 B。窗外活动探针记录到一次设备内复制、`eval_body_joints` 与 `integrate_bodies` 两个核和两次设备至 HOST 复制。CUDA 设备标识为 NVIDIA GeForce RTX 4060 Laptop GPU，Warp 报告 8 GiB、sm_89。

两个 CUDA 资格任务的专用编译缓存最初均为空。Phymeo 框架初始化 469.26 ms、建模 283.10 ms，首个物理步和六列输出共 394.62 ms，生成 3 个缓存文件。Newton 相应为 3,786.77、2,128.34、526.86 ms，生成 12 个文件。Warp 原生日志将首步的关节计算模块和积分模块标为 compiled，分别用时 394.07、127.21 ms。这里控制的是本批指定的磁盘缓存，不对系统驱动缓存作冷启动假设。

## 复现

两侧使用各自独立解释器。先分别以 `--phase qualify --expect-empty-cache` 运行 CPU、CUDA 资格，Phymeo 加 `--phymeo-observation kinematics`。正式轮复用对应设备的磁盘缓存，每个命令启动新进程。例如 W32 的一组为：

```powershell
& $phyPython scripts/benchmark_newton_free_fall.py --engine phymeo --phymeo-observation kinematics --device cuda:0 --phase measure --worlds 32 --cache-root $tyxCache --out $freshPhy
& $newtonPython scripts/benchmark_newton_free_fall.py --engine newton --device cuda:0 --phase measure --worlds 32 --cache-root $warpCache --out $freshNewton
```

本批 4 份资格和 12 份正式物理收据、配对守卫、逐窗状态 SHA 与完整时间样本均在 `artifacts/p65-night-g2-typed-newton-direct-20261104-01/`。`comparison.json` 的 SHA256 为 `cd84599600ed2ec3e99f8a19f48f626dc76a10f23d81c7112a6c08ca41781a63`。以 `python scripts/summarize_newton_free_fall.py --receipts-root <归档> --phymeo-observation kinematics --out <新文件>` 可核查实际进程先后、物理输出和设备计数。[计时程序](../../scripts/benchmark_newton_free_fall.py) · [两个实际求解器适配](../../tests/scale/g2_cross_engine_steps.py) · [独立收据审计](../../scripts/summarize_newton_free_fall.py)
