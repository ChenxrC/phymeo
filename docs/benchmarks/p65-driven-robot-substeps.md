# 固定基座机器人的止挡载荷与按物理子步延迟的 PD

在 HOST 上，`DrivenRobotProgram` 把关节止挡、带延迟的执行器命令和外部载荷送进同一个固定基座机器人子步。每一行都能读回执行器力、止挡力、各自的冲量和积分后实际发布的关节状态。[公开模块](../../src/phymeo/solvers/driven_robot.py) · [专项测试](../../tests/test_driven_robot.py)

## 物理子步

1. 用本子步开始时的 q、v 计算止挡。越下限时弹簧给正向广义力，越上限时给负向广义力。阻尼仅在越界且继续向外时作用。离开止挡的首个子步卸载。
2. 从队列读取目标位置、目标速度和前馈三项。延迟为 d 时使用 d 个物理子步前输入的整包。默认初始历史重复初始位置、零目标速度、零前馈，也可用 `initial_state(..., prehistory=array)` 明确传入。独立外部载荷、止挡、本步反馈 q 与 v 都取自当前步。
3. 设质量矩阵为 M，本步开始时的动力学偏置为 b，外部广义力与止挡力的和为 F。在速度 u 上联立求解 M(u−v)=h(τ+F−b) 与 τ=clip[kp(q*−q−hu)+kd(v*−u)+ff]。限幅只作用在执行器 τ 上。公共 RobotWorkspace 用持有的 τ、止挡和原始外力/力矩推进实际状态。
4. 事务接受后，`SceneState.history[domain + "/drive_queue"]` 保存下一步的队列，形状为 (d, 3, DOF)，最老命令在前。零延迟无需缓冲。若整帧中途失败，返回帧初 q、v 和队列，诊断中已尝试的子步数可定位第一次失败。

```python
from phymeo import JointLimits, RobotState, StepSettings
from phymeo.solvers.driven_robot import DrivenRobotProgram

p = DrivenRobotProgram(model, kp=kp, kd=kd, effort_limit=cap,
                       limits=JointLimits(lower, upper, stiffness=2000., damping=5.),
                       delay_steps=2)
s = p.initial_state(RobotState(q0, v0))
f = p.inputs(q_target, v_target, feedforward=ff, loads=external_robot_inputs)
with p.stepper(StepSettings(substeps=4, record_substeps=True)) as rt:
    result = rt.step(s, f, 0.04)
    assert result.report.accepted
    last = result.report.substeps[-1].diagnostics
    motor_impulse = last["actuator_impulse"]
    stop_impulse = last["stop_impulse"]
    actual = result.state.domains["robot"]
```

示例的命令在这一帧的四个子步保持恒定。要在第 k 个物理步改命令，可用单子步帧推进到 k，再传入新命令。角关节的力矩、冲量单位是 N·m 与 N·m·s，移动关节是 N 与 N·s。止挡诊断的偏差和弹簧势能在子步开始时采样，速度与位置在步末采样。

## 独立数值与实测

| 构型 | Phymeo HOST 的观测 |
| --- | --- |
| 2 kg，无重力，界外 1 mm，k=2000 N/m，h=0.01 s | 下限 +2 N，上限 −2 N，对应 ±0.02 N·s。静止起步的末速度为 ±0.01 m/s，关节位置从 ±0.251 m 移至 ±0.2509 m。 |
| 0.1 m/s 向外或开离，阻尼 5 N·s/m | 向外时力幅值 2.5 N，开离时为 2 N。另一条从下限外 0.5 mm 开离的轨迹记录到两个实际子步依次加载 1 N、0 N。 |
| 未饱和，h=1/8 s，kp=8、kd=2，目标 0.25 m、0.5 m/s，前馈 0.5 N，外力 +1 N | 限幅 3 N 下实测 τ=2.7894736842105265 N，hτ=0.3486842105263158 N·s；v=0.2368421052631579 m/s，q=0.02960526315789474 m。解析值依次为 53/19、53/152、9/38 与 9/304。 |
| 止挡与电机一起加载 | 电机封顶 +4 N，止挡 +2 N，独立载荷 +8 N，总广义载荷 +14 N。两滑轨连杆的质量矩阵为 [[3,1],[1,1]]，不同关节给力 [1,2] N，h=1/16 s 后两关节速度为 [−1/32,5/32] m/s。 |
| 延迟 2 步，四子步帧持有 +2 N 新前馈 | 四行电机力依次 0、0、2、2 N，和四次独立单子步帧逐项一致。分别验证了非零初史、延迟 0/1/2、有符号目标变化、晚发算术溢出和验收器拒绝后的回滚与重试。 |

## Newton 1.7 CPU 对照

用 Warp 1.17、Featherstone 和单轴 DrivePD 运行 27 个实际物理步，包含三次零命令预填。只核对此节列明的时间点和物理量。[可运行对照](../../tests/reference_newton/robot/physical_substeps.py)

- 下限静止和闭合时，由 2 kg 乘速度增量除以步长反推的止挡力为 1.9999743 N 和 2.4999753 N，上限符号镜像。开离但仍越界时，Newton 的阻尼同时作用于开离速度，力幅值为 1.4999747 N。这里的止挡力由实际速度反推，电机力另从 Control.joint_f 读取。
- 用 d 个零命令先填充历史，再送 [1, −2, 3, −4, 0] N，Newton 延迟 0/1/2 的电机力和速度/位移均与离散解析解一致。首次启动空的 2 步缓冲，Newton 立即使用当前的 +1 N。Phymeo 的默认初史在前两步使用持位命令。
- Newton 的显式 DrivePD 按步初状态计算同一目标，10 N 限幅时读回 3.5 N、0.28125 m/s，3 N 限幅时为 3 N、0.25 m/s。Phymeo 的隐式未饱和结果见表。
- Newton 零延迟使用无 Delay 对象。首次运行给 builder 显式传 0 时，1.7.0 的 finalize 试图建立深度为零的缓冲并拒绝。修正只改变零延迟对象的构造。

从仓库根目录，以项目解释器和对应源码路径运行专项。Newton 脚本使用独立 1.7.0 环境。

```powershell
python -m pytest -q tests/test_driven_robot.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/physical_substeps.py
```

范围为固定基座标量树的 HOST FP64 受力步进。执行器延迟以物理步数计，在步长变化时相应的秒数也会变化。限位是柔顺力模型，当前反馈从已发布的子步开始状态观测。设备控制核、步内首次撞停时刻与硬止挡冲量属于后续独立验证。
