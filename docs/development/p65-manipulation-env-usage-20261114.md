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

## 目标反馈 PushT 与连续 episode（2026-11-15 第十二批）

三个新通用接口：
- `phymeo.control.manipulation_push_objective.propose_planar_pushes(...)`：给原始边界查询、参考/当前完整刚体位姿与 XY/yaw 目标，返回完整有界 `PROPOSAL_ONLY` 列表。评分同时考虑位置与角度，不把名义力矩当真实运动。保留不可用/负收益候选，原 strict planar 查询没有放宽。
- `phymeo.control.manipulation_push_planner.NativePushRoutePlanner(...)`：当前完整 q、提案和可选上一推向，尝试撤离、直达或两向绕行、进入、短推/保持；每次 IK 和采样路径都计入预算。`sampled_reference_ready` 只代表有限 native HOST 参考路径检查，实际 GPU 运动可能偏离，不能据此声称 CCD 或无碰撞。
- `phymeo.control.manipulation_push_progress.BoundedPushProgress(...)`：消费真实段末代价、步数和 pusher 合力模积分；成功信号必须来自原 Goal。停滞、步数、段数分别有终止原因，失败调用不消耗状态。

`scripts/run_manipulation_push_feedback.py --source ORIGINAL_CANDIDATES --output FRESH_ABSOLUTE_DIRECTORY --world 5|6|7`
执行本批预登记诊断。每世界一个连续 CUDA episode，默认最多12秒/12000步、8完整段；不从旧物理轨迹注入对象状态。初始物理 FP32 q、真实入步速度和全部对象自由度一直保留。`reference-feedback-summary.json` 区分完整路线和末尾部分路线；所有 route 提案、失败 IK/path、选择与完整采样保存在各自 gzip 中。

原 `run_manipulation_control.run(..., reference_driver_factory=...)` 增加显式可选单世界反馈接缝，要求 `record_entry=True`，不可同时启用另一 slip driver。`reference(tick,q,v)` 给参考；`accepted(tick,q,entry,result)` 只消费已接受步。停机在下次 `env.step` 前生效；accepted 回调失败仍保存该完整 CURRENT transition，随后抛出 `phase=reference-feedback-accepted`、`action_accepted/state_accepted=True` 的错误。反馈摘要保存失败不覆盖最初物理异常或阻止主 manifest 记录。

力与运动的相位：接触力属于 `entry_q` 的入步，控制效果代价属于新的 accepted endpoint。原 Goal 仍由 `accepted-entry` SuccessFrame 判断；不能把这些相位混作同一时刻。合力模积分单位 N s，含每段的接近/撤离实际接触；它不是相消向量冲量，也不包含重力。段边界用实际机器人状态重新生成参考，仅参考坐标可裁到原关节限制；原状态不裁。每条路线内部为 C1，跨路线重新绑定不声称全局 C1。

这是 HOST 反馈规划加 CUDA 物理的功能诊断，未消除 HOST 同步，也未实现语义模型、共享策略训练、完整 MPC 或通用任意物体规划。最终 Env checkpoint 不含该诊断 driver 的完整恢复状态。完整操作结果必须另看本批报告。
