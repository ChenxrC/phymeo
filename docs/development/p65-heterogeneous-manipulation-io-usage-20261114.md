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

1. workspace 默认导出 accepted q/v/act/clock；可选 body 字段同为端点相位。contact_tokens 默认 False，通过 include_contacts=True 启用下文同入步观察。
2. body_angular_velocity 已转换到世界轴；原生 v 数组仍保留 FREE local angular，不改变原地址协议。
3. 接触力必须与同一入步位姿/速度成对；M0 的 SuccessFrame 明确只接受 input phase。
4. body endpoint FK 与 accepted-entry contact token/SuccessFrame 已完成 B1/B2；目标/history组合checkpoint、final observation/timeout 和异构 PPO 接线仍待 B3及后续。
5. 当前 refresh/control 审计有状态/记录读回，叶子求解器也保留现有诊断；不是零 HOST 往返或融合异构 kernel。
6. 工作区 allocation/显式 readback/D2D 计数单列，不把 solver-bank budget 当总 GPU 内存预算。
7. M0 三类目标的 HOST 参考判据能拒绝伪抓取/空中放置/中心重合；合成帧测试不代表物理任务成功。


## 持久入步接触记录已连接（B2存储部分）

产品 `994b0842` 已在各叶子求解器内部保存 accepted-entry 双bank，并随物理状态共同提交。
`batch.checkpoint()/restore()`、reset和clone会自动保留/清除对应记录，无需应用增加私有访问代码。
旧胶囊恢复后显式“无历史记录”，下一成功step才生成有效entry；成功空接触是另一个状态。
详细字段、原始失败和源码资格见[本批报告](p65-entry-record-report-20261114.md)。

第四批结束时公共workspace只提供端点及可选body，contact token尚未连接；这是历史边界。
第五批新增的公共entry/世界载荷/Torch接触输出见下节。应用继续禁止把B1端点与过去载荷混成SuccessFrame，
或依赖私有entry数组；目标/history与episode的完整Env生命周期仍待B3。


## 公共入步 body 与接触观察（M2-B2c/d）

`batch.prepare_device_workspace(include_contacts=True)` 现在额外输出同一已接受物理步的入步状态和载荷。
它独立于 `include_bodies`：不启用后者时只有 `entry_body_*`，不会同时分配端点 `body_*`。
两个开关都启用时，两种相位明确分列。

| 字段 | 形状与意义 |
| --- | --- |
| `entry_q/v/act` | [W,各字段最大native宽度]；输入native地址不变，padding零 |
| `entry_clock` | [W,4]：input high/compensation、output high/compensation |
| `entry_meta` | int32 [W,6]：valid、input tick、output tick、detected、active、overflow |
| `entry_contact_ids` | int32 [W,C,8]：valid、active、geom1/2、body1/2、feature、condim |
| `entry_contact_data` | [W,C,20]：distance、point3、frame三行9、contact-frame六维载荷、include margin |
| `entry_body_position/rotation` | [W,B,3] / [W,B,3,3]：入步body原点世界位置、body→world矩阵 |
| `entry_body_linear_velocity/angular_velocity` | [W,B,3]：入步body原点世界线/角速度 |
| `entry_body_mask` | int32 [W,B]：有效入步body，包含fixed/world body |
| `entry_contact_wrench_world` | [W,C,6]：作用于geom2、接触点处的世界force/couple |
| `entry_contact_body_wrench` | [W,C,2,6]：作用于body1/2、各自body原点处的世界force/moment |

除上述整数字段外均为FP32。C取所有模型接触容量最大值，最低一列sentinel。
contact frame的每一行是一条接触轴；世界力/力偶为 `frame.T @ local`。
geom1反作用符号相反；body-origin moment额外加 `(contact_point-body_origin) × force`。
condim4扭转和condim6滚动的真实力偶保留，不把力矩简化为纯力臂项。

`entry_meta.valid=0` 表示无历史记录；entry body mask和输出清零，不构造过去的状态或载荷。
成功空接触为valid1、count0，仍有有效入步body状态。inactive检测有身份/几何但载荷为零。
旧格式capsule恢复可有output tick>0且valid0；下一成功step才能产生新的有效entry。

`frame.world_boundaries` 按world顺序提供 `(batch_uuid, serial, operation)`：
初始为initial，成功reset/reset-states只改变选中world，restore改变所有world；普通step和controls不变。
这是操作边界元数据，不是物理状态、语言输入或任务编号。它不进入旧物理capsule，恢复不会抹掉实际发生的边界。

### 成功帧消费与恢复

```python
from phymeo.training.manipulation_observation import ManipulationSuccessReader

# by_world: {world_key: 原始 Scene}; episodes: {world_key: nonnegative int}
workspace = batch.prepare_device_workspace(include_contacts=True)
reader = ManipulationSuccessReader(workspace, by_world)
frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
success_frames = reader.read(frame, episodes=episodes)

# 在已消费的同一物理点保存；goal/history由调用方分别保存。
physics_checkpoint = batch.checkpoint()
cursor_checkpoint = reader.checkpoint()

batch.restore(physics_checkpoint, expected_revision=batch.checkpoint_revision)
frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
reader.restore(cursor_checkpoint, frame=frame)
```

reader是显式HOST诊断/成功判据适配器，使用公共readback；GPU特征生产仍不依赖它。
它核原始Scene XML、FREE物体地址与named finger/support对应关系，以同入步位姿、速度和载荷构造SuccessFrame。
重复同一记录返回空映射，跳步、伪episode变化及未配对物理恢复均拒绝；任一world失败不提交任何world游标。
实际reset需要更大的episode号，首step提供input tick0；即使重置后首步字节与旧首步相同，也不会混淆。
reset后可以先read空frame并保存tick=-1的游标，也可直接step后read；无效记录均不产生虚构SuccessFrame。
游标恢复会核对实际公共frame的完整entry内容，再绑定当前边界；旧游标不能绕过实际reset。
物理、reader cursor、goal/history目前仍是三个明确owner；下一批B3再统一Env checkpoint事务。

## 接触力与COM力矩测量、PushT边界候选（T1b-4）

```python
from phymeo.control.manipulation_wrench import measure_object_contact_wrench
from phymeo.control.manipulation_push import enumerate_native_planar_push_contacts

# arrays是同一accepted-entry公共NumPy映射；com_world必须来自该模型该入步。
wrench = measure_object_contact_wrench(
    arrays, row, object_body=object_body_id,
    object_com_world=com_world, group_geom_ids=pad_geom_ids,
)
# group之外的台面等接触仍在other/total。invalid entry返回None。
# model/q为CPU native几何查询输入，返回候选不会修改实际GPU state。
candidates = enumerate_native_planar_push_contacts(
    model, q, object_geom_names=scene.object_geom_names, pusher_geom_name="pusher",
)
```

前者完整世界force/couple以及COM力矩，不以body origin替代COM，不宣称抓稳。保存CLI仅支持原任务object和完整W1反馈trace，见[报告](p65-manipulation-wrench-push-report-20261114.md)。

后者限同body完整1..2个共轴/同Z BOX和球/竖直胶囊，完整外边及进入段对所有BOX检查，候选最多128、failed attempts保留；cap-only高度/倾斜/遗漏几何明确拒绝。该HOST查询的1e-9m容差、名义单位法向wrench、IK端点资格不等于真实碰撞响应或任务成功。后续用原GPU owner推进物理。
