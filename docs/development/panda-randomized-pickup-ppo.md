# Randomized pickup policy experiment

后续批次已完成手臂探索幅度与时间相关性的配对诊断，并在 S4000 启动可恢复的长训。[选型数据、后台训练进度与查看方式](s4000-panda-pickup-background-ppo.md)

**S4000 结果。** 两台 Panda 的动力学、手指电机上限和原独立承重门槛沿用上批。本轮比较有无 PPO 更新的拾取策略，每一对评估使用相同起点和物理参数。S4000 学习了 4 轮、128 个回合、20,480 次决策，对应 163,840 个训练物理世界步。权重约在启动后 13 分 18 秒写出，含冻结评估全程约 40 分钟。CPU 在同一协议下独立训练了另一份权重。[S4000 实测](../../examples/checkpoints/panda-pickup-ppo-271144/s4000/result.json)

| 独立物理评分 | 原策略 | PPO 候选 | 同起点由失败变成功 | 同起点由成功变失败 |
| --- | ---: | ---: | ---: | ---: |
| S4000 选型集 | 19/64 | 23/64 | 10 | 6 |
| S4000 验证集 | 11/64 | 8/64 | 1 | 4 |
| 原生 MuJoCo 选型集 | 25/64 | 24/64 | 3 | 4 |
| 原生 MuJoCo 验证集 | 12/64 | 13/64 | 2 | 1 |

两个候选都未通过各自的替换条件。域随机化覆盖销的质量、惯量、摩擦及受托初态，训练和验证仍是同一机器人、销和托架，训练各仅一个策略种子。推理使用仿真状态。新物体、腕部视觉及完整插装序列仍需独立验证。[S4000 权重](../../examples/checkpoints/panda-pickup-ppo-271144/s4000/candidate.pt) · [逐世界结果与域参数](../../examples/checkpoints/panda-pickup-ppo-271144/s4000/paired-worlds.tar.gz)

**学习首先卡在探索阶段。** S4000 和 CPU 的 128 个随机探索回合中，各有 0 个达到稳定拾取。用另外 32 个训练起点做原生 MuJoCo 配对诊断，冻结策略且不给优化器更新。三组的初态、动力学和承重评分相同。后两组共用同一份标准正态抽样。

| 原生 MuJoCo 探索对照 | 双指接触销柄 | 销抬高 25 mm | 稳定承重 |
| --- | ---: | ---: | ---: |
| 确定性原策略 | 23/32 | 15/32 | 10/32 |
| 每个控制周期独立探索全部 16 维动作 | 16/32 | 0/32 | 0/32 |
| 相同手臂噪声，双指噪声缩至 20% | 19/32 | 0/32 | 0/32 |

原始探索使每指目标相对当前状态的原策略平均改变约 2.7 mm，缩小双指噪声后为 0.56–0.57 mm。缩小指令扰动有助于形成双侧接触，完整抬升仍失败。这组试验尚未单独识别手臂噪声的因果贡献。下一次先在训练起点配对比较手臂动作幅度和时间连续性，并记录从双侧接触到离托、再到承重的转化。采样能产生成功和近成功轨迹后，再扩大并行世界和训练种子。检验泛化要保留新的独立复位库。[训练集探索对照](../../examples/checkpoints/panda-pickup-ppo-271144/exploration.zip)

A trained reach controller supplies supported handoffs. This batch asks whether reinforcement learning improves pickup from those handoffs. The motor law, time step, geometry, free pin, force limits and load-bearing evaluation are shared with the prior evaluation. Each policy chooses all sixteen motor targets throughout pickup. Demonstrations enter only through the frozen RBF prior.

## Fixed first experiment

- Start the residual actor at the published pickup prior. Train four complete rollouts, 32 worlds per rollout, 160 decisions and eight 2 ms physics steps per decision. Use seed 271144 and three PPO epochs per update. Each training start is used in one rollout. Reward uses observed alignment, signed pad load, support unloading, lift and continuous grip. A disturbance loses reward.
- Use the nominal model first and three seeded physical variants after it. Each variant scales pin mass and inertia together within 0.85–1.15 and independently varies sliding friction for pads and pin within 0.8–1.2 of nominal. Reset noise covers geometry, joints and velocities. Both policies in a comparison get identical starts, physical domains, motor limits and action budget. Physical parameters vary between compiled groups, while starts vary by world.
- Freeze the final checkpoint before evaluation. Compare deterministic prior and trained actions on 32 reserved starts on each of two selection models, then 32 starts on each of two separately seeded validation models. The latter uses fresh perturbations around a previously evaluated source bank. Keep all outcomes and identities; split and domain manifests record the dependencies. These 64 rows per split share source handoffs and do not constitute 64 independently collected real-world grasps.
- Measure the published independent load-bearing gate. The free pin must leave the fork, the two fingertips must carry it through a 25 mm lift with at most 5 mm relative slip and maintain the grasp for 150 ms. Log gains and losses per paired world. For this first screen, both selection and validation must have a higher total with neither model group regressing. Passing is a reason for further testing across training seeds and new geometries. Deployment defaults remain under separate qualification.
- The main hardware target is MUSA. An explicit native CPU backend provides an independent reference and permits a real learning update during device unavailability. Record the backend on each run. Tests of tensor shapes, synthetic rewards or checkpoint serialization do not qualify hardware behavior.

## Reproduction

From the repository root, run `python examples/panda_pickup_ppo.py --backend native --out artifacts/pickup-ppo-native`. The S4000 command uses `--backend musa --cache <device-cache>`. On MUSA, first run `examples/panda_pickup_resident_load_check.py` against the device's full integration receipts. Each run writes to a new directory. Domain manifests identify initial states, per-world score files describe deterministic evaluation, and the checkpoint binds the frozen student by SHA-256.

## Input, feedback and measurement

The actor sees the 21 pickup features (relative hand–pin pose, orientation, heights, finger positions and setpoints, right-arm target error) and 22 generalized velocities. It outputs 16 normalized motor commands. A conventional bounded PD loop applies the corresponding efforts. The actor receives neither the successful action from a demonstration nor a phase counter during the episode. The two-pad forces and fork load enter the training reward and the evaluation. The current actor observes state and has no tactile or camera input.

The resident MUSA path reads each accepted solve's contact map, contact frame and wrench on its input pose and attributes the signed force to the free pin. On an eight-step fork-supported S4000 replay, resident and fully published input and successor states matched bit for bit. The largest per-channel discrepancy in normal force was below 5.3e-8 N, with measured support of 1.17720 N for a 1.1772 N weight. That check covers fork support. The independent deterministic rollout scorer measures bilateral lifting through the complete episodes. [Device calibration](../../examples/checkpoints/panda-pickup-ppo-271144/s4000-calibration/summary.json)

The native MuJoCo candidate changed actor weights on each of four PPO updates. Across the paired selection groups the prior passed 8/32 and 17/32, while the candidate passed 6/32 and 18/32. Across the separate validation groups the prior passed 5/32 and 7/32, while the candidate passed 6/32 and 7/32. Three stochastic training updates used perturbed physical groups, and each of the four updates had zero fully successful exploration episodes. The deterministic evaluations and training rolls measure different behaviors. The candidate failed the predeclared selection screen. [Native run](../../examples/checkpoints/panda-pickup-ppo-271144/native/result.json)

## Training-only exploration check

The PPO sampling distribution perturbs all sixteen commands every 16 ms. The two finger commands set an absolute opening, while the fourteen arm commands change a joint target incrementally. To test how much this matters, use two training-domain groups and 16 newly perturbed starts per group (reset batch 401). Compare the deterministic frozen prior, the same initial policy with its original exploration, and the same standardized random draws with finger noise scaled to 20%. The arm noise law and draws stay identical across the two explorations, and all three branches use the original independent physical scorer. Record both-pad contact, 25 mm lift, successful load-bearing, paired world outcomes and realized finger target change in mm. This diagnostic updates no weights. Reproduce with `python examples/panda_pickup_exploration_check.py --out artifacts/pickup-exploration`.
