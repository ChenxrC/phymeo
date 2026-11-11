# N1：真实 CUDA 策略与 MuJoCo-TYX 物理闭环

2026-11-11，状态 **QUALIFIED_N1_INFERENCE**。最终 focused 组 **22 passed、1 hardware-conditional skip，16.18 s，RAW_EXIT=0**。本机只有一张 GPU，跨 GPU 拒绝项明确未测；这不是跨卡资格。N1 是策略推理与真实物理消费，尚未执行 PPO 更新、N2 VecEnv 或 N3 组合恢复。

## 实际环境

复用 `C:/Users/38347/AppData/Roaming/Python/Python313/site-packages` 的 Torch **2.12.0+cu126**，在父 `.venv-training-cuda313` 中隔离项目版本。Python3.13.3、NumPy2.4.6、MuJoCo3.3.7、RSL5.5.2、TensorDict0.15.5；TYX仍为原 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`，CUDA bindings12.9.10使用cp313 wheel。实际 NVRTC 返回12.6，driver API版本12070；设备RTX4060 Laptop。全部 import 来源、禁止用户editable路径以及driver/NVRTC初始化已核，run05 RAW_EXIT=0。

环境不是自包含副本：它借用原用户 Torch，后续不能把该目录当作冗余文件删除。清理及恢复方式见[环境记录](../development/p65-environment-refresh-20261111.md)、[版本清单](../../requirements/training-cuda.txt)。已有CPU训练与物理环境未覆盖。

## 固定执行矩阵

| 模型 | 世界数 | producer stream | 每条轨迹步数 | 每条 world-steps |
| --- | ---: | --- | ---: | ---: |
| hinge-slide | 1 | default / nondefault，各一条 | 8 | 8 |
| hinge-slide | 8 | default / nondefault，各一条 | 8 | 64 |
| hinge-slide | 32 | default / nondefault，各一条 | 8 | 256 |
| support-slide | 8 | default / nondefault，各一条 | 8 | 64 |

八条轨迹共 **784 个 world-steps**。每例真实 CUDA MLP 参数/输入/输出均在 GPU；动作通过公开 TYX tensor/stream 互操作进入真实连续 MuJoCo-TYX CUDA 物理。支撑例每个世界均出现实际接触载荷，所有世界运动。h、XML/hash、初态、动作规则与原数值门均保持预登记值。

对照是独立连续演进的 **同一个 TYX CUDA 后端，通过 HOST control 入口提交同次策略输出的 FP32 字节**；不是 CPU MuJoCo 求解器，也不通过逐步覆盖对照状态追随主轨迹。每步完整原生控制通道、状态、时钟、力和物理收据逐项比较；保留HOST诊断/动作审计以及D2D复制的实际计数，不声称端到端零读回。

负控覆盖 CPU/dtype/shape/stride、NaN/Inf/subnormal、通道alias、stale revision/frame、关闭view/stream与借用者生命周期。workspace关闭后Torch-owned观察仍可用，已提交动作不依赖借用者继续存活。最终只有真实第二GPU负控未测。

## 首次失败与修复边界

run06先通过17项，然后关闭view负控失败：原测试只接收ValueError/RuntimeError，而固定TYX实际抛出独立的 `CudaBackendError`。产品正确拒绝已释放句柄。修复仅在该负控显式接收固定TYX异常类型，继续检查错误文本、原接受状态和Torch原始资源存活；没有修改运行时、物理fixture或数值门。首次失败完整保留，不能改写为首次全绿。

最终测试源提交 `981bd76d`。修复后该测试文件lint/format均通过，run11完整23身份执行得22pass/1明确硬件未测。原N1示例仍为已审阅的公开实现，无新增物理后台或CPU fallback。

## 保存与下一节点

证据目录：父 `artifacts/p65-env-refresh-20261111-01`。run11的八份完整 `trajectory.jsonl` 含声明、逐步完整payload和最终summary；保存清单 `n1-saved-inventory.json` 核实际文件/大小/SHA、八个预定模型/世界数/stream身份及关闭结果。运行前后源快照、原始stdout/stderr/RAW_EXIT均保留，不通过重跑物理补证据。

该结果解除[GPU训练计划](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md)的N1前置门。下一步N2是独立device VecEnv、episode/timeout/reset/失败阶段语义，以及固定两任务的实际RSL CUDA PPO更新；不得把本批推理闭环称为训练或G1全动作完成。
