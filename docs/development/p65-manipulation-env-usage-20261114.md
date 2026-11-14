# 异构 GPU 操作 Env 用法

2026-11-14，B3源码资格；[验证报告与限制](p65-manipulation-env-report-20261114.md)。
本层管理episode与任务目标；物理仍由公共 MuJoCo-TYX heterogeneous batch 执行。

## 八场景接入

以下展示一个step及成对保存/恢复，不是抓取控制器或策略训练脚本。

```python
import numpy as np
import torch
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
)
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import manipulation_scenes
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_checkpoint import ManipulationEnvCheckpoint
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec

scenes = dict(enumerate(manipulation_scenes()))
models = tuple(
    MuJoCoTyxModelSpec(s.key, compile_mujoco_xml(s.mjcf, require_asset=False))
    for s in scenes.values()
)
worlds = tuple(MuJoCoTyxWorldSpec(k, s.key) for k, s in scenes.items())
with MuJoCoTyxHeterogeneousBatch(models, worlds, mode="cuda") as batch:
    specs = {
        k: MuJoCoTyxActionSpec(
            tuple(range(layout.nu)),
            scale=np.full(layout.nu, 0.1, dtype=np.float32),
            bias=np.zeros(layout.nu, dtype=np.float32),
        )
        for k, layout in batch.layout.items()
    }
    with HeterogeneousManipulationEnv(
        batch, scenes,
        goals={k: s.goals[s.default_goal] for k, s in scenes.items()},
        action_specs=specs, horizons=dict.fromkeys(scenes, 1000), auto_reset=True,
    ) as env:
        # A shared policy will later consume these numeric observations and
        # natural-language contexts. No language encoder is installed here.
        observation, contexts = env.observation, env.contexts
        action = torch.zeros(
            (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
        )
        result = env.step(
            action, world_keys=env.world_keys, expected_frame=env.current_frame
        )
        raw = env.checkpoint().to_bytes()
        checkpoint = ManipulationEnvCheckpoint.from_bytes(raw)
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
```

示例scale只用于接口演示，实际机器人控制范围、控制器增益与任务窗口必须在T1中单独确定。
每world ActionSpec可同时指定ctrl、qfrc与(body,component)形式的xfrc；列顺序依次为这三个通道。
输入是连续CUDA FP32 `[W,max_action_width]`，world顺序严格匹配；有效列必须有限，padding忽略且保存为零。
Env内部执行 `bias + scale * tanh(raw_action)`；policy若另带tanh须明确分布定义，避免无意二次压缩。
`observation['action_mask']` 为CUDA bool，未来log-prob/entropy/loss也必须忽略padding。
可传 `producer_stream=torch.cuda.Stream(...)`；必须与设备一致，调用期间保持输入及流的合法所有权。

## 相位和episode

- `result.input_frame`：动作之前的已接受端点。
- `result.evaluation_frame`：本次物理成功后、reset之前。`success_frames/results` 使用其中同一步的入步位姿/速度/接触载荷。
- `result.output_frame/observation`：给下一动作的端点；auto-reset世界已是新episode初态。
- `result.final_observation`：全部world的reset前观察；只在 `final_mask` 为真的行消费episode终止观察。
- `terminated` 表示成功判据；`truncated` 表示horizon且未成功。当前reward为稀疏成功0/1，不含塑形。
- `episode_id/episode_length/episode_return` 属于刚结束或仍在进行的旧episode。内部已reset行的长度/回报清零、episode号加一。

`auto_reset=False` 时，结束世界标记pending_reset；下一次批量step会拒绝，先调用：

```python
# chosen为world_key序列；new_goals必须恰好覆盖chosen，Goal取自原Scene目录。
reset_observation = env.reset(chosen, goals=new_goals, expected_frame=env.current_frame)
```

未选择的world物理、目标和history保留。空选择是no-op；即使物理字节相同，真实reset仍增加episode并建立新读取边界。
`contexts`提供当前目标的中英文句子、物体标签及所有候选目标几何。当前没有文本embedding，不把任务0-n编码输入网络。

## 恢复与错误

`checkpoint()` 同时保存原physics capsule、当前Goal绑定、episode计数/长度/回报、完整成功history与读取游标。
格式有总长度/元数据/深度/项目数上界，拒绝重复JSON键、非有限值、损坏摘要、来源/动作/world顺序不符。
同一源码及绑定下可以恢复到新建Env；新Env须使用相同Scene目录、ActionSpec、horizon和auto_reset配置。
先校验元数据，再恢复物理，最后核对实际公共entry并发布新语义状态。

捕获 `MuJoCoTyxEnvError` 时检查 `phase/action_accepted/state_accepted/reset_accepted/needs_recovery`。
普通动作预检失败可修正后重试；物理接受后的失败不会假装回滚，也不会返回一个成功transition。
`needs_recovery=True` 时须用之前保存的成对checkpoint显式restore；旧observation不得继续送策略。
合法摘要但混配物理和cursor的capsule可能在物理restore后被拒绝，此时 `state_accepted=True`。
这类错误可恢复，但不是数据库式物理回滚。不要自行删除reader游标或修改episode来绕过。

Env借用batch并要求独占其写操作；直接在外部step/set_controls/reset会使Env frame失配。
`close()`只关闭Env工作区，不关闭借用batch；导出的Torch副本继续有效。
实际success评价目前有HOST读取与同步，吞吐评估须计入；尚无训练性能结论。

## 候选排序、实际短推与完整 entry（2026-11-15）

`phymeo.control.manipulation_push_rank.rank_planar_push_candidates(query, desired_translation_world, desired_yaw_change)`
消费 `enumerate_native_planar_push_contacts` 的有界查询，返回不可变排序记录，保留所有候选。期望平移是世界XY向量，yaw为有符号角度；index仅路由，未作为策略输入。排序是几何启发，必须另查整机路径并用实际载荷验证。

`scripts/run_manipulation_push.py` 是本批固定三场景诊断入口：qualify消费绑定的上批候选并保存分阶段路径；execute直接消费已合格冻结节点，各执行一次真实CUDA短推。它不是通用任务规划器或训练器，不用于任意来源文件。
原脚本 `run_manipulation_control.py --record-entry` 可在诊断trace里保存本transition完整entry，默认关闭。COMPLETE仅在最终保存成功后设置；steps与recorded_steps分别描述接受和记录情况。

`scripts/measure_manipulation_wrenches.py RUN_DIRECTORY OUTPUT_DIRECTORY --groups pusher`
可读新CURRENT记录，包含最后一步；旧feedback记录仍按真实缺失末条报告，不合成entry。
COM使用同一入步body rotation和绑定模型body_ipos；顶层q/v为步后端点，不能拿来解释本次入步载荷。other包含所有非分组对象接触，不能一概称台面；total不含重力。
这是HOST诊断消费，不再执行forward/step，未声称零读回GPU训练路径。

原三构型当前只过完整参考接近与2秒局部传力，完整原PushT目标仍未过；见[结果](p65-manipulation-push-route-report-20261115.md)。
