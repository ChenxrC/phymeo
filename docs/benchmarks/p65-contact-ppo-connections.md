# 新安装包上的真实接触PPO闭环

**已通过，2026-11-10，例程源`b622fdf6`，安装包物理源`37aa1c40`。** 实际模块路径确认来自55安装target，无source路径注入。原RSL-RL5.5.2 → `RslRlVecEnv` → `ResidentVectorEnv` → `SceneBatchBackend` → 原HOST机器人/有限地面接触，完成固定预算、两质量组原20%学习门与新世界checkpoint重载，RAW_EXIT0。

## 首次执行前固定的任务

16个真实URDF竖直prismatic scene，mass按(.5,1)kg交替；sphere半径1/16m、实心球惯量2mr²/5，初始q0=0、v0=±.02m/s、深度.003m。有限floor half=(1,1,.05)m@z=-.05，gravity−9.81，原RobotContactMaterial(k1000,c8,μ0)。h1/128、horizon96，goal按(-.004,-.008,-.012)m循环。raw action clip[-1,1]×12N，原effort cap16N，零KP/KD；宽joint范围只容纳预声明最坏上抛，不夹紧真实q/v。

观测为实际q/.02、v/.5、goal/.02、mass；cost为`((q-goal)/.02)^2 + .01*(v/.5)^2 + .001*actual_drive_effort^2`。最后一项来自公开已发布drive_effort，不拿请求动作冒充实际力。

原seed20261110、32×32 ELU/PPO配置、100轮×24 rollout/16worlds保持；训练器原日志报告38400步。没有改seed、budget、质量、目标或门槛追结果。前/后/重载各192步确定性评估，均用原world factory和真实非零初速，不替代动力学。

## 实际学习结果

| 组 | 未训练mean cost | 训练后mean cost | 下降 |
|---|---:|---:|---:|
| 全体 | .07006888247 | .02462536903 | **64.8555%** |
| .5kg | .06761855178 | .03773165823 | **44.1993%** |
| 1kg | .07251921316 | .01151907982 | **84.1158%** |

actor参数实际改变；三门均超过原20%。三个评估均观察到两质量组真实非零接触。公开loads/LoadSite、双侧wrench、floor发布反力、contact_effort和原contact_events在auto-reset前捕获；检查载荷t/接受t+h、双方力与世界矩、原输入/末态同源。没有根据目标构造“实际力”。

原生checkpoint在新建原物理世界载入后，192步观测/动作/cost、实际q/v、驱动力、法向载荷、终态mask及完整接触/事件复现。数值门沿原声明，接触/事件序列另作完整比较；不是中途物理/RNG精确resume。

## 证据与能力边界

`artifacts/p65-connections-20261110-0800-01/76-contact-ppo-training/results`保存原declared_plan、两URDF、三个完整评估gzip、complete_result.json.gz、native trained.pt和summary。`contact-ppo-outcome.json`记录8个before/after/current源0漂移和真实安装模块路径。实际环境仍Torch2.15.0 CPU、TensorDict0.15.5、RSL-RL5.5.2、NumPy2.4.6。wall07:08:23→07:10:52含原评估/保存/重载，不作Newton吞吐对比。

首lint仅1个import-order诊断，首format需排版；修复后非import AST及同作用域连续import绑定相同，raw AST因import顺序不同不相同。final静态通过，随后仅执行这一固定训练，没有试种子/换fixture。

这新增了**训练器到真实接触**的安装包闭环。仍为16个HOST场景、CPU学习和NumPy/TensorDict拷贝；不是GPU PPO、受载W256、未知模型泛化或G1从头训练。已有两项无接触学习结果继续复用，不重训。
