# G2 驻留世界的物理步进与读回

2026-11-04，提交 `df5bcbdc231a4518ab78a1fcb26ad83f80fe21c3` 的 `TyxWorldBatch` 在一个 TYX runtime 中推进独立世界。支持无初始自旋、受重力与质心外力的平移球，以及使用 `DrivenRobotProgram` 输入的单关节固定基座滑轨。滑轨可使用零拍或一拍延迟。每个设备线程推进一个世界，双缓冲分开保存已接受状态和候选状态。[工作区](../../src/phymeo/backends/tyx_world_batch.py) · [受支持的模型和检查点](../../src/phymeo/backends/_world_step_layout.py) · [设备核](../../src/phymeo/backends/_tyx_world_kernels.py)

## 轨迹、复位与失败

- 两个共址的 2 kg 自由球从 z=2 m 静止下落，重力为 −8 m/s²，步长为 1/128 s。逐步读取的速度和位置满足 vᵢ=−i/16、zᵢ=2−i(i+1)/4096。第 32 步的速度为 −2 m/s，位置为 1.7421875 m，半径 0.125 m 的球底为 1.6171875 m，动能为 4 J。另一个质量、重力、初速和外力均不同的自由球与现有 FP64 刚体求解器核对。
- 两个 2 kg 滑轨沿相反方向同时驱动。h=1/16 s 时，正向物理子步的实际驱动力为 +4、+4、−4、−4 N，q 为 (1,3,4,4)/128 m，速度为 (1,2,1,0)/8 m/s。反向世界逐步取相反值。另用转过 90° 的关节轴、不饱和控制和外部载荷，比较已接受的驱动力、位置与速度。对照运行原有 FP64 `DrivenRobotProgram` 的真实子步。[冻结滑轨输入](../../tests/reference_newton/robot/cases.json)
- 一拍延迟世界保存中间非零速度、驱动力、三维命令队列及持有输入。只恢复该世界后，重放产生相同的下一状态，邻居沿原轨迹继续。只复位该世界会把其步数、位置和队列恢复到初始值。
- 一个自由球发生 FP32 溢出时保留其状态和步数，同批的滑轨和另一个自由球照常接受。临时故障发生在候选状态发布前时，工作区保留已接受缓冲，排空设备后可重试。注入持续同步故障时，工作区拒绝后续操作。新建工作区从调用方先前保存的检查点恢复后，实际 CUDA 下一步复现了解析轨迹。[所有对照与注入位置](../../tests/scale/test_resident_step.py)

## 无逐步快照的 32 暖步

RTX 4060 Laptop GPU 上，四个世界先执行 1 个子步并保存基线快照，再以已持有输入执行 32 步。窗口内确认 128 个世界物理子步。两个状态数组及两个整数步数数组分别具有独立的 CUDA 地址，未选用逐帧快照。TYX 运行时和工作区分别记录了以下数值。

| 实测量 | 32 步窗口 |
| --- | ---: |
| 真正的 CUDA kernel / 显式同步 | 32 / 32 |
| H2D / D2H | 0 B / 512 B |
| 拓扑重传 / 额外 CUDA 分配 | 0 B / 0 次 |
| 摘要 / 完整快照 D2H | 0 B / 0 B |

512 B 来自每步返回的 4 个 int32 接受标志。计数窗口启用了底层 CUDA 驱动分配拦截。创建工作区时拥有 13 个 CUDA 数组，共计 948 B，初始拓扑上传 160 B。预热前改变一个世界的控制时，下一步上传完整的 4 世界控制表，共 64 B。暖步结束后显式调用一次摘要读回 32 B，完整快照另读回 208 B，包含物理状态、延迟历史与整数步数。显存数字来自此工作区拥有的数组，未计驱动、编译器和全进程显存。此次采集覆盖状态驻留与传输，没有测量物理步进的墙钟吞吐。

最终专项为 interpret/HOST 8 项、CUDA 8 项。原始收据在本机 `artifacts/p65-wave6-resident-20261104-01/`。其中 `resident-cuda-final/pytest-temp/test_warm_residency_timer_and_0/resident-warm.json` 保存预热前后、摘要和快照的计数，最终组的 `final.json` 含源文件哈希与退出码。可在安装项目 TYX 依赖的环境中分别运行下列命令。[测试入口](../../tests/scale/test_resident_step.py)

```powershell
python -m pytest -q tests/scale/test_resident_step.py -k "interpret or empty"
python -m pytest -q tests/scale/test_resident_step.py -k "cuda or warm_residency"
```
