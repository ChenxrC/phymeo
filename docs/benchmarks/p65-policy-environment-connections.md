# 策略与环境的共同连接：PolicyRunner

当前状态：**AUTHORING_ONLY / UNEXECUTED**。本文件只记录接口和待执行验收；不把源码作者、手算固定向量或既有 G1 窗口计为本模块通过。父唯一执行槽正在运行独立长窗，本作者没有 import、collection、数值执行、pytest、lint、formatter 或 build。

## 目标与公共入口

新增 phymeo.control.policy_runner.PolicyRunner，把已有向量环境与无状态策略回调连接起来。它不拥有动力学，不写 q/v，不注入参考状态，不选择接触解算器。环境的显式 stepper 决定本次使用普通步还是受载步；HOST scene adapter 与 TYX resident batch 都经各自环境返回相同 VectorStep。

    from phymeo.control.policy_runner import PolicyRunner

    runner = PolicyRunner(env, policy, control_decimation=10)
    observation = runner.reset()       # 明确调用；构造函数绝不偷偷 reset
    result = runner.step()             # 恰好一次 env.step / 一次物理子步
    action = result.actions
    transition = result.transition     # 全部八项 VectorStep 字段
    diagnostics = runner.snapshot()     # 只读记录，不是可恢复物理 checkpoint

用户已经持有当前观察时，可以传 initial_observation=observation，从该状态开始控制而不复位。随后应统一经 runner 调用 step/reset，不能在背后直接推进或复位借用的环境。世界顺序和动作宽度取自环境的 world_keys 与 action_size；观察是完整二维 [world, features] NumPy 数组，不冒零拷贝设备张量。

policy(readonly_observation, readonly_due_mask) 必须返回完整 [world, action_size] 有限实数矩阵，包含暂时不更新的世界。所有返回行均先验证形状/数值；只有 due_mask 为真的行替换旧动作。策略回调无状态：允许观察记录，但不能依赖一次推理必然对应一个接受步；此接口没有回滚 RNN hidden state、normalizer 或外部随机数状态的承诺。

## 整数节拍与独立世界

- control_decimation=K 规定每 K 个实际接受的物理步更新一次动作；每个世界分别持有非负整数 physical_ticks。首次调用全部 due，期间保持上次接受的动作，不多推理或多推进。
- PolicyRunnerStep 保存实际交给 env.step 的动作、这次 due_mask、入口 physical_ticks 和完整不可变 transition。动作是环境输入，不冒 backend 量化后的力；真实力/关节力矩应从物理观察读取。
- 环境的 final_mask 标识 terminated 或 truncated 后已自动复位的世界。该世界时钟回到 0，下一物理步重新推理；其它世界原有节拍不改变。final_observation 保持终止前的真实观察，observation 为环境返回的复位后观察。
- 显式 reset(indices=None) 同时支持独立整数索引与完整 boolean mask；调用原环境 reset，随后仅所选世界时钟归零/标为刷新。完整观察返回值重新绑定，但其它世界的 held actions 和节拍保持。
- snapshot 保存观察、最近实际接受动作、每世界时钟、刷新 mask 和 needs_reset。它是持久诊断数据的来源，不提供物理状态恢复或策略递归状态恢复。

## 拒步和部分接受

策略的形状/NaN/Inf/复数/boolean 错误在任何物理调用前拒绝，观察、动作和时钟均不推进。回调也只看到 bytes-backed 不可变输入。

runner 保留环境的原异常，尤其 accepted_worlds、acceptance_known、physics_accepted，不能把部分接受包装成“全场景已回滚”。N2 明确标为 physics_accepted=False 的原子拒绝可以原状态重试，原 held actions/时钟保持。若任何世界可能已接受、环境 needs_reset、没有明确接受协议或后置观察失败，runner 也标为需要显式完整复位；不得只 reset 其中一个世界后继续。

环境已经返回但 transition 缺字段、维度错误、非有限、完成 mask 不匹配或计数类型不合法时，抛 PolicyRunnerProtocolError，标明物理已经返回，需要完整 reset。不会发布假的 reward/done 或递增策略时钟。reset 自身或 reset 观察失败也会阻断后续 step，因为物理初态可能已经改变。

## 待执行测试

tests/test_policy_runner.py 是独立可观测环境测试，覆盖：

- 无隐式 reset、显式初始观察所有权，一次 runner step 对应一次环境物理调用。
- 保持三个物理子步，只有到期才推理；只复位一个世界不改变另一个世界下一次到期时间。
- terminated 与 truncated 都保完整终态观察，并在下一物理步刷新完成世界动作。
- 完整动作矩阵验证，包括非 due 行的 NaN；回调/返回动作/transition/快照不可写。
- 原子拒绝重试、部分接受原异常、未知接受、后置错误、失败 reset，均保原时钟与最近接受动作，必要时明确要求完整复位。

tests/scale/test_resident_policy_runner.py 是实际 TYX 多世界链路，不能由 HOST mock 通过代替：

1. 两个独立 1 kg 自由球、零重力、h=1/8，策略直接输出真实外力。一个世界因位置门终止并复位，另一个保持原节拍。第四步完整力积分给第二个世界 v_y=-125/256、y=-317/2048；这些是独立二进制有理向量，不通过读 runner 状态生成 oracle。完整四步动作、due mask、所有八项 transition 与实际 batch 终态保存到 JSON。
2. 两世界已经真实接受受力步后 reward observer 失败，核实际位置确已改变，runner 不能假装回滚或推进策略时钟；显式完整 reset 后从原初态重新接受控制。

父整合 N2/N3 后，在唯一执行槽按需运行：

    python -m pytest tests/test_policy_runner.py -q -x
    python -m pytest tests/scale/test_resident_policy_runner.py --tyx-mode interpret --require-tyx -q -x
    python -m pytest tests/scale/test_resident_policy_runner.py --tyx-mode cuda --require-tyx -q -x

本作者没有执行上述命令，也未收集测试节点。最终运行结果、静态修复与保存证据由父按实际来源追加，不能倒灌到作者快照。

## 能力边界

这是公共 NumPy 策略节拍器，不是 G1 专用控制器、IK、平衡控制、训练算法、GPU 零拷贝或统一物理 checkpoint。实际任务决定观察、奖励、终止和公共动作映射。无状态策略的持有/复位/拒步语义先完成；PPO、设备张量互操作、策略状态 checkpoint、异步多进程和更大世界数分别后续验收。G1 原 154 维协议继续留在实验入口，不污染本模块。
