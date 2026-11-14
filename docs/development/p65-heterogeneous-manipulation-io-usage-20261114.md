# 异构 MuJoCo-TYX GPU 状态与动作入口

2026-11-14，M2-A 与 M2-B1。公共 native 状态/动作路由及可选端点 body 观察；接触记录与完整多任务 Env 后续接入。
验证与限制见[本批报告](p65-heterogeneous-manipulation-io-report-20261114.md)。

## 协议

| 接口 | 内容 |
| --- | --- |
| `batch.prepare_device_workspace()` | 持久双缓冲工作区；跨 reset/restore 自动解析当前 owner |
| `workspace.refresh(expected_revision=...)` | 读取已接受积分 bank，所有桶成功后发布一个 frame；不执行 CPU forward |
| `workspace.to_torch(frame)` | CUDA→CUDA 复制到新 Torch-owned storage；修改导出 tensor 不会修改物理状态 |
| `workspace.readback(frame)` | 显式 HOST 诊断副本；不参与设备输出构建 |
| `batch.set_controls_from_device(...)` | 验证 world 顺序/CAS/shape/dtype/alias，逐桶 gather、原 packet 审计，全部成功后共同发布 |
| `batch.reset_states(mapping, expected_revision=...)` | 部分 world 指定新初态，其余完整状态不变；初态 time 必须为零 |
| `batch.checkpoint()/restore(...)` | 复用已有共同 checkpoint；恢复后旧 frame 过期，重新 refresh |

`q/v/act/clock` 为 FP32，行严格按 `batch.world_keys`，列保留各模型 native 地址。
`q_mask/v_mask/act_mask/ctrl_mask/body_mask` 为 int32 有效掩码。
列宽取各模型最大值；全零宽字段保留一列零 sentinel，其 mask 全假。
padding 不表示物理自由度，也不能进入策略概率或损失。

FREE q 是世界平移 + WXYZ；FREE v 是世界线速度 + **body-local 角速度**。
`clock[:,0]-clock[:,1]` 表示补偿时钟。frame 的 revision/epoch/ticks 绑定本次发布。
已有新 physics/control/reset/restore 或同工作区新 frame 后，旧 frame 不能再次读取。
已复制的 Torch tensor 则有独立所有权，可作为旧观察保存。

`model_metadata` 提供不可变 HOST 模型常量：关节 native 地址/类型/轴/父体、body 拓扑/质量/惯量、
执行器映射与控制范围、名称。这里没有任务编号编码，也没有自动的机器人语义 token。
模型常量可以在建环境时上传一次；不要每步从 HOST 拼接动态观察。

## 八模型公共入口示例

下例演示统一装配与设备动作提交。它没有实现抓取控制器或学习策略。

```python
from contextlib import ExitStack
import torch
from tyx.torch import from_torch, stream_from_torch
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
)
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes

scenes = manipulation_scenes()
models = tuple(
    MuJoCoTyxModelSpec(s.key, compile_mujoco_xml(s.mjcf, require_asset=False))
    for s in scenes
)
worlds = tuple(MuJoCoTyxWorldSpec(i, s.key) for i, s in enumerate(scenes))
with (
    MuJoCoTyxHeterogeneousBatch(models, worlds, mode="cuda") as batch,
    batch.prepare_device_workspace() as workspace,
):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    observations = workspace.to_torch(frame)
    # Replace this zero command with a shared policy's masked actuator output.
    command = torch.zeros_like(observations["ctrl_mask"], dtype=torch.float32)
    with ExitStack() as resources:
        view = from_torch(command, backend="cuda")
        resources.callback(view.free)
        stream = stream_from_torch(torch.cuda.current_stream(), backend="cuda")
        resources.callback(stream.close)
        batch.set_controls_from_device(
            view, world_keys=batch.world_keys,
            expected_revision=batch.checkpoint_revision, producer_stream=stream,
        )
    batch.step(H)
    next_frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    next_observations = workspace.to_torch(next_frame)
```

所有有效动作元素必须为 finite normal 或 zero FP32；无效 padding 不被 gather。
省略的 qfrc/xfrc 会产生新的零外力，不能继承上一请求的外力。
仅全体模型 nu=0 时 ctrl 必须为 None。输入不能 alias 物理 bank、动作 packet 或 workspace storage。

动作准备、审计或最后桶验证失败不改变任何桶的动作与 revision。
物理 step 的共同发布继续由原 batch 事务负责。发布后退休资源清理失败与发布失败分开报告。
UNKNOWN_RELEASE 保留 owner/依赖，后续 close 不重试未知 native free。


## 可选端点 body 观察（M2-B1）

使用 `batch.prepare_device_workspace(include_bodies=True)`，其余 refresh/to_torch 调用保持不变。
同一个 accepted endpoint frame 额外包含：

| 字段 | 形状与含义 |
| --- | --- |
| `body_position` | FP32 [W,B,3]，body origin 世界位置 |
| `body_rotation` | FP32 [W,B,3,3]，body→world 旋转矩阵 |
| `body_linear_velocity` | FP32 [W,B,3]，body origin 世界线速度 |
| `body_angular_velocity` | FP32 [W,B,3]，世界角速度 |
| `body_mask` | 既有 int32 [W,B]，body 有效性；包含 fixed/world body |

padding 全零，含旋转 padding；必须先应用 mask，不能将零矩阵当有效姿态。
FREE/BALL 的原生局部角速度按已有运动学转到世界轴，body-origin 线速度与偏心 COM 线速度不同。
字段来自 accepted q/v，在工作区自己的 scratch 中复用现有 GPU geometry kernel；
全部桶成功后与 q/v/act/clock 一起发布。观察失败不覆盖旧 frame，reset/restore 后重新读取当前 owner。
导出 tensor 仍有独立所有权，修改它们不能改变下一步物理结果。

此接口不执行 CPU FK 或 HOST 动态状态重建；建工作区时的 HOST 静态模型常量与显式诊断读回仍存在。
`include_bodies=False` 是默认值，不分配上述 body 输出/scratch。
[源码验证报告](p65-body-observation-report-20261114.md)记录14项新测试及受影响兼容范围。

## 当前边界与下一连接

1. workspace 默认导出 accepted q/v/act/clock；可选 body 字段同为端点相位，contact_tokens 仍为 False。
2. body_angular_velocity 已转换到世界轴；原生 v 数组仍保留 FREE local angular，不改变原地址协议。
3. 接触力必须与同一入步位姿/速度成对；M0 的 SuccessFrame 明确只接受 input phase。
4. body endpoint FK 已完成 B1；accepted-entry contact token、目标/episode checkpoint、final observation/timeout 和异构 PPO 接线仍待 B2/B3。
5. 当前 refresh/control 审计有状态/记录读回，叶子求解器也保留现有诊断；不是零 HOST 往返或融合异构 kernel。
6. 工作区 allocation/显式 readback/D2D 计数单列，不把 solver-bank budget 当总 GPU 内存预算。
7. M0 三类目标的 HOST 参考判据能拒绝伪抓取/空中放置/中心重合；合成帧测试不代表物理任务成功。
