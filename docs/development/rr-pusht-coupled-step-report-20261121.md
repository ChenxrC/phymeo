# RR Push-T：原失败步通过，半载荷补偿路线续至 4.200 s

从上一批半载荷补偿轨迹的 3.719 s 检查点恢复，原电机命令在 CUDA 上通过。从这一步起的 481 个物理步全部提交并通过保存数据审计，路线按原计划结束于 4.200 s。T 块沿推动方向共前进 10.76 mm，末态距目标 117.64 mm，偏航误差 0.313 rad，原任务的成功保持时间为零。[上一批同入态控制对照](rr-pusht-load-compensation-report-20261121.md) · [本次逐步审计](../../artifacts/rr-coupled-step-20261121-01/resumed-audit.json)

## 联合求解的改动

机器人与 T 块的面积接触力会改变 T 块的速度，桌面约束的反力又会影响同一步的接触力。求解器反复计算这两者，达到载荷和速度两个门槛后才发布。旧求解器在 3.719 s 已用半步更新；最大的广义载荷误差仍反复换向，第 25 次评估时速度误差仍超限。[原单步重放](../../artifacts/rr-load-feedback-20261121-01/single-step-diagnosis.json)

沿用原有的反向振荡判据：当前最大绝对载荷误差的某一分量反号，且归一化载荷误差至少为上一轮的一半。已进入阻尼模式的试载荷若再次满足这个条件，下一次试载荷取四分之一修正；没有新反转证据时回到半步。该更新是每个世界的局部数值选择，依据该步实际残差，适用于设备上的原生约束与面积接触共同求解路径。收据用阶段值 3 表示四分之一更新。恢复时独立检查反转证据，按实际更新比例重建载荷历史，旧的全步、半步收据仍可读取。[设备更新规则](../../src/phymeo/backends/mujoco_tyx/_area_native_coupled_kernels.py) · [收据与恢复验证](../../src/phymeo/backends/mujoco_area_friction_state.py)

| 同一 3.719 s 入态、动作、材料与步长 | 原更新 | 本次更新 | 原门槛 |
| --- | ---: | ---: | ---: |
| 原生约束评估次数 | 25 | 20 | 最多 25 |
| 最后归一化载荷变化 | 7.408 × 10⁻⁷ | 1.542 × 10⁻⁷ | ≤ 10⁻⁶ |
| 最后归一化响应速度变化 | 2.397 × 10⁻⁷ | 8.372 × 10⁻⁸ | ≤ 10⁻⁷ |
| 该步状态发布 | 拒绝 | 已发布 | 两门同时满足 |

新单步的输出位置、速度与连续续跑首步一致，法向载荷为 1.049 N。独立实验回归使用从原模型生成的力学量及 FP64 解检查受力、运动与摩擦。另有回归故意伪造松弛比例，以确认恢复时能拒绝错误的载荷历史。[原检查点单步结果](../../artifacts/rr-coupled-step-20261121-01/single-step-diagnosis.json) · [公开场景回归](../../tests/test_pusht_coupled_oscillatory_entry.py)

## 完整推动窗口

表中位移均为沿同一参考推动方向、从第 3251 到第 4199 个输入状态的实测投影（949 个输入）；位置、朝向与轮廓交并比取 4.200 s 已发布输出。左列复用上一批原位置控制基线。右列由旧求解器上的半载荷补偿前缀和改进求解器上的续段拼接。完整窗口有控制和求解两处差别，因此只能比较这两套实际运行配置。单独检验求解器效果用上表的同一失败步。[逐项计算结果](../../artifacts/rr-coupled-step-20261121-01/outcomes.json)

| 指标 | 原位置控制基线 | 半载荷补偿，3.719 s 起用新求解器 |
| --- | ---: | ---: |
| 参考推杆进给 | 19.99 mm | 19.99 mm |
| 实际推杆进给 | 4.53 mm | 12.21 mm |
| 实际 T 块进给 | 2.88 mm | 10.76 mm |
| 末态位置误差 | 125.66 mm | 117.64 mm |
| 末态偏航误差 | 0.291 rad | 0.313 rad |
| 末态轮廓交并比 IoU | 0.00209 | 0.01487 |
| 2.014–4.200 s 峰值法向载荷 | 0.983 N | 1.104 N |
| 原任务达标 | 否 | 否 |

恢复时使用完整物理检查点与上一个已提交步的载荷，按已保存的有效观测恢复控制器与目标历史。首步的电机命令、广义外力和物体外力均与此前失败提交的 FP32 字节相同。前缀 1,705 步与续段 481 步共 2,186 步连续。续段 34 步实际采用四分之一更新，每步使用 6–23 次评估。逐帧最大末次载荷变化为 6.378 × 10⁻⁷，速度变化为 9.971 × 10⁻⁸。续段逐步检查参考、前步力反馈、双侧作用反作用、运动和任务观测。又在全新 CUDA 场景恢复了 4.200 s 检查点，位置、速度、时钟和末步收据相符。[续段与检查点](../../artifacts/rr-coupled-step-20261121-01/resumed.json) · [新场景末态读回](../../artifacts/rr-coupled-step-20261121-01/terminal-readback.json)

## 复现与下一批

在仓库根目录与训练 CUDA 环境执行，从原有非空、经过审计的父轨迹恢复。输出使用新文件名。

```powershell
$parent = 'artifacts/rr-load-feedback-20261121-01'
python -m experiments.area_pressure.pusht_active_route --summary artifacts/rr-public-feedback-20261121-01/continuous-damped.json --release artifacts/rr-loaded-withdrawal-20261121-01/release-b.json --checkpoint "$parent/half.final.scene.metadata.json" --resume-source "$parent/half.json" --resume-audit "$parent/audit.json" --control-variant load-compensation-half --output resumed-new.json
python -m experiments.area_pressure.pusht_resume_audit --input resumed-new.json --output resumed-new-audit.json
```

本批 7 个相关测试文件合计 66 项通过，包括多世界候选事务、受载单帧独立参考、伪造收据、续跑来源和指令匹配。[回归记录](../../artifacts/rr-coupled-step-20261121-01/20-final-regressions.stdout.log)

下一批从 4.200 s 的真实位姿重新选择推杆接触点与方向。预先固定另一种起始朝向作为对照，检查两种入态下位置误差和绝对偏航误差是否同时下降，并保留原任务成功判据。这样可以区分当前规划的力矩方向问题与剩余的受载跟踪误差。
