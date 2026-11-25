# 在摩尔线程 GPU 上运行 Phymeo 关节动力学与接触

`MuJoCoRuntime(..., mode="musa")` 使用 CPU MuJoCo 编译模型，然后由摩尔线程 GPU 执行 Phymeo 的 Tyx 内核。全移动关节模型可走恒质量路径。固定基座、由移动或转动关节构成的树可共同求解关节软限位、关节干摩擦和基本几何体的金字塔接触。基本接触支持 `condim=1`（法向）和 `condim=3`（增加两个切向）；平面、球与胶囊的受支持几何对还可使用 `condim=6`（再增加一个扭转、两个滚动分量）。没有接触几何对的模型也可直接求解关节约束。同批世界共用模型，各自保存初态、控制和接触状态。

单个自由根连接移动或转动关节时，MUSA 也可运行浮动基座的树动力学，并通过混合约束求解器传递足地反力。对应的模型范围与实测数据见[浮动基座与双足反作用](#浮动基座与双足反作用)。

## 运行

在装有 MUSA 驱动和 SDK 的 Linux 主机上使用 Python 3.11。编译器需要 `mcc`，运行时需要能够加载 `libmusa` 和 `libmusart`。以下命令从 Phymeo 仓库根目录执行，安装固定版 Tyx 需要该仓库的读取权限。

```bash
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MUSA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install -e '.[mujoco]' -r requirements/tyx.txt 'pytest>=8,<10'
PHYMEO_TEST_MUSA=1 python -m pytest -v --require-tyx tests/test_mujoco_tyx_musa.py
PHYMEO_TEST_MUSA=1 python -m pytest -v --require-tyx tests/test_mujoco_tyx_musa_friction.py
PHYMEO_TEST_MUSA=1 python -m pytest -v --require-tyx \
  tests/test_mujoco_tyx_constraints.py::test_both_signed_sliding_directions -k musa
```

下面是可直接运行的最小模型。两个世界共用同一模型，控制和状态分别保存。

```python
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

xml = """<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody><body name="mass"><joint name="slide" type="slide" axis="0 0 1"
    armature="0.04" damping="0.2"/>
    <geom name="ball" type="sphere" size=".1" mass=".5"/>
  </body></worldbody>
  <actuator><motor name="drive" joint="slide" gear="2"
    ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>"""
model = compile_mujoco_xml(xml, require_asset=False)
with MuJoCoRuntime(model, mode="musa", world_keys=(4, 91)) as sim:
    sim.set_controls({4: MuJoCoControl([8.0]), 91: MuJoCoControl([-0.7])})
    for _ in range(24):
        sim.step(sim.timestep)
    for world, state in sim.snapshot().items():
        print(world, state.tick, state.q, state.v)
    print(sim.musa_device_report())
```

CPU 完成 MJCF 编译。Phymeo 复用现有 Tyx 动力学内核的中间表示，按设备报告的架构用 MCC 编译。GPU 保存当前状态和候选状态，成功读回并检查整批候选结果后才切换。控制目前从主机上传。要读取此次运动实际使用的接触力，取 `sim.integration_snapshot()[world].last_step.contacts`，其中 `wrench_contact_frame[0]` 是步首求出的法向力（N）。显式 `sim.snapshot()[world].contacts` 则会在 CPU 上按步末姿态重新计算接触。训练张量互操作仍待验证。该入口与 Vulkan 渲染的设备选择各自独立。

## S4000 实测

2026-11-25 在 MTT S4000（`mp_22`）上验证。驱动与运行时 API 报告版本 50200，使用 Python 3.11.16、MuJoCo 3.3.7，以及 `requirements/tyx.txt` 指定的 Tyx。两世界各推进 24 个 1 毫秒步，每步和独立的 CPU MuJoCo 状态比较。

| 指标 | 实测 |
| --- | --- |
| 最大位置绝对误差 | 6.14 × 10⁻¹⁰ m 以内 |
| 最大速度绝对误差 | 6.23 × 10⁻⁸ m/s 以内 |
| 最终位移，世界 4 / 91 | +5.538 mm / −0.775 mm |
| 实际提交并同步完成的内核 | 48 次，两个世界批处理 |
| 登记的设备数组 | 38 个，共 31,380 字节，关闭后预算占用归零 |
| 累计传输 | 主机到设备 35,412 字节，设备到主机 15,168 字节 |

无接触批次还注入了候选读回故障。两世界均保留上一帧的位置、速度、时间与状态版本，恢复读回后只推进一个有效步。无接触版本 `753abbc7` 另有 3 项 Tyx 解释路径回归通过。表内统计属于无接触批次。JIT、传输及稳态积分尚未单独计时。

## 从两个高度落地并持续承载

下面的模型给质量 0.5 kg、半径 100 mm 的球加一个竖直移动关节。球心在关节零位时高 99 mm，因此初始关节位移 7 mm 和 21 mm 分别给出 6 mm 和 20 mm 的净间隙。

```python
from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

xml = """<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="1"/>
    <body name="body" pos="0 0 .099">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".1" mass=".5" condim="1"
            solref=".006 1" solimp=".85 .95 .002 .4 2"/>
    </body>
  </worldbody></mujoco>"""
model = compile_mujoco_xml(xml, require_asset=False)
starts = {4: MuJoCoInitialState([.007], [0.]), 91: MuJoCoInitialState([.021], [0.])}
with MuJoCoRuntime(model, mode="musa", world_keys=tuple(starts), initial_states=starts,
                   contact_capacity=1, constraint_capacity=1) as sim:
    for _ in range(240):
        sim.step(sim.timestep)
    for world, state in sim.integration_snapshot().items():
        print(world, state.last_step.contact_time, state.last_step.contacts[0].wrench_contact_frame[0])
```

每个世界共推进 240 ms。独立的 CPU MuJoCo 逐步比较位置、速度和入步法向力。首次非零力与首次接触在两个后端的采样时刻一致。表中时间是 1 ms 离散采样时间。

| 初始净间隙 | 首次非零力的步首时刻 | 峰值法向力 | 最大压入 | 末 40 ms 平均支撑力 | 最后一步压入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 mm | 35 ms | 29.152 N | 1.624 mm | 4.90496 N | 0.23417 mm |
| 20 mm | 64 ms | 50.496 N | 2.944 mm | 4.90434 N | 0.23419 mm |

第一个世界在第 35 步的端点观测首次出现接触，第 36 步的求解使用该姿态施力，两者都对应 35 ms。第二个世界对应第 64 步端点和第 65 步步首。若把入步法向力标在步末，会把冲击峰画晚一拍。

两条轨迹的 GPU 对 CPU 最大绝对差分别为位置 6.25 × 10⁻⁹ m 以内、速度 4.60 × 10⁻⁷ m/s 以内、法向力 9.47 × 10⁻⁵ N 以内。约 0.234 mm 的静止压入也出现在 CPU 参考中，是这组软接触参数共同给出的承载变形。理论重力载荷为 4.905 N。

这组落地试验在 S4000 上整批发射并完成 720 次内核。抽查接触转换和末帧时，直接读取 GPU 已接受缓冲中的几何 ID、接触数和法向力。接触容量设为零、仅一个世界落在接触区时，整批返回容量错误，两个世界均保留上次状态。

## 侧推、滑动与连杆间反作用

同一台 S4000 的摩擦试验使用一个可沿 X、Z 运动的 1 kg 球。两个世界先承载 120 ms，然后分别施加恒定侧向力 80 ms。每个世界又用同一组参数跑一次纯法向对照。摩擦组的有效接触维数为 3，摩擦系数为 0.6。对照将有效维数设为 1，确保切向力为零。两组均在每个 1 ms 步与独立 CPU MuJoCo 连续轨迹比对位置、速度、接触力与关节反力。

| 水平外力 | 有摩擦位移 | 纯法向位移 | 有摩擦末速度 | 末 20 步平均水平接触力 |
| ---: | ---: | ---: | ---: | ---: |
| 2.4525 N | 0.347 mm | 7.946 mm | 0.00496 m/s | −2.452 N |
| 11.772 N | 18.943 mm | 38.141 mm | 0.47049 m/s | −6.027 N |

小载荷由摩擦基本平衡，软接触仍有微滑。大载荷下的切向力达到约 6.03 N 的摩擦上限，物体持续滑动。两组摩擦功分别为 −0.000827 J 和 −0.109823 J。混合求解每批每步提交并完成九次设备内核调用，两个世界共享这九次调用。测试在两个状态缓冲都使用过后检查设备分配数和编译缓存，直到末步均保持稳定。

有摩擦的两个世界共提供 400 份步记录。GPU 对 CPU 的最大绝对差为位置 3.35 × 10⁻⁹ m、速度 1.14 × 10⁻⁷ m/s、接触坐标系力分量 1.53 × 10⁻⁵ N。统计覆盖落稳与侧推全程。

另一项测试使用两条各含转动和移动关节的分支，让两个可动球形末端相碰。四个自由度连续推进 16 步，两侧关节均收到反作用，完整状态和接触载荷按步与 CPU 对照。容量测试在一个世界接触、另一个世界分离时，将最多四条摩擦约束行限制为三条，确认整批拒绝且两边都保留上一状态。

这里实测的几何组合为球–平面和球–球。扩展接触几何、扭转或滚动接触摩擦（`condim=4/6`）及整机 G1 步态尚未在 MUSA 开放。

## 关节端挡与干摩擦

从 MJCF 的 `range`、`limited`、`margin` 和 `frictionloss` 声明关节约束。转动关节的干摩擦力矩单位为 N·m，移动关节的干摩擦力单位为 N。接近端挡时，软限位按 MuJoCo 的 `margin` 可以提前产生反力。用 `sim.integration_snapshot()[world].last_step.constraints` 读取实际参与该步求解的行，用同一记录的 `qfrc_constraint` 读取合成后的关节广义力。

同一套原生 Tyx 约束内核在 S4000 上通过了 16 个上下限用例、12 个干摩擦用例和 1 个球–平面接触与关节约束同时生效的用例。另外验证了整批回滚：两个世界先成功推进一步，随后仅一个世界进入限位区。容量设为一行时，两个世界都保留已接受的步；允许两行时，两者正常推进。

```bash
export PHYMEO_TEST_MUSA=1
python -m pytest -v --require-tyx tests/test_mujoco_tyx_musa_joint_arm.py
python -m pytest -v --require-tyx -m musa \
  tests/test_mujoco_tyx_constraints.py::test_joint_limit_inside_margin_and_outside_bounds \
  tests/test_mujoco_tyx_constraints.py::test_dry_friction_bounded_force_with_viscous_damping \
  tests/test_mujoco_tyx_constraints.py::test_same_input_mixed_classes_share_one_physical_solve \
  tests/test_mujoco_tyx_musa.py::test_joint_limit_overflow_preserves_last_accepted_batch
```

### 一台双关节机械臂，两个独立世界

测试用肩部转动关节串联伸缩关节，两个电机分别驱动，无重力、无外部接触。两个世界从肩角 ±0.165 rad 静止起步，向相反的端挡驱动，在 140 ms 同时反向，再推进到 280 ms。关节范围为 ±0.2，限位裕量为 0.01，分别使用转动轴的 rad 和伸缩轴的 m。表中时间以该步计算约束的步首为准，力矩取绝对值。

| 肩部初始驱动力矩 | 首次限位承载 | 峰值限位反力矩 | 反向后退出限位区 |
| ---: | ---: | ---: | ---: |
| +1.60 N·m | 57 ms | 9.151 N·m | 151 ms |
| −1.36 N·m | 63 ms | 8.255 N·m | 153 ms |

两条轨迹的肩部干摩擦力矩峰值均为 0.200 N·m，伸缩轴干摩擦力峰值均为 0.200 N，每个世界的两条轴全程干摩擦功均为负。560 份步记录逐步比对独立 CPU MuJoCo 轨迹，包括关节位移、速度、逐行约束及广义反力、末端位置和姿态。末端位置各分量最大绝对差低于 4.0 × 10⁻⁸ m。完整轨迹与两世界摘要由机械臂测试写入 pytest 临时目录的 `joint-arm-traces.json` 与 `joint-arm-summary.json`。

## 浮动基座与双足反作用

模型可以声明一个自由根，所有活动关节与它属于同一棵树。无接触的浮基树走原树动力学。基本几何体的金字塔接触及关节约束走共同求解器（T4）。只产生纯法向接触且由编译器分派给独立法向求解器（T3）的模型、多个独立自由根、球关节及六维盒接触另有验收需求。

**空中耦合。** 对同一个无地面接触的模型，保持入步状态一致，只给子转动关节增加 0.25 N·m 广义力矩。S4000 上测得自由底座局部 X 轴角加速度从 1.240 变为 0.587 rad/s²，变化量为 −0.652 rad/s²。反过来仅在根部施加载荷，子关节加速度也发生变化。两种响应都逐项对照独立 CPU MuJoCo。无接触连续轨迹还覆盖了 7、16 和 29 个标量关节的合成模型，29 关节案例运行 12 步。

**足地传力。** 另一模型在自由底座下连接偏心转动脚踝和竖直伸缩腿，各带一个球形足端。同批两个世界以相同姿态着地，分别施加不同的恒定电机命令。步长 1 ms，连续 32 步。每步从 CPU MuJoCo 独立计算足端接触力和接触点雅可比，逐足验证传回根部和活动关节的广义力；再将两足贡献相加，与 GPU 求解器实际用于推进状态的总反力核对。

| 脚踝 / 伸缩腿电机命令 | 首步底座向上反力 | 脚踝反力矩峰值（绝对值） | 伸缩腿反力峰值（绝对值） |
| --- | ---: | ---: | ---: |
| +0.16 N·m / −1.5 N | 29.039 N | 0.686 N·m | 7.794 N |
| −0.12 N·m / +0.5 N | 27.252 N | 0.667 N·m | 6.326 N |

相同姿态下，仅改变电机命令，首步底座竖直加速度相差 1.008 m/s²。两个世界的双足同时达到本用例可测载荷门槛的步数分别为 32 和 30。S4000 共提交并完成 288 次设备内核调用。测试也覆盖合成陀螺的动量、能量误差，以及批量读回失败后的回滚和重试。

```bash
export PHYMEO_TEST_MUSA=1
python -m pytest -v --require-tyx -m musa tests/test_mujoco_tyx_floating.py \
  tests/test_mujoco_tyx_musa_floating.py \
  tests/test_mujoco_tyx_constraints.py::test_two_feet_shared_mass_is_not_independent_contacts \
  tests/test_mujoco_tyx_constraints.py::test_free_root_all_six_dof_dry_rows
```

## 六维胶囊接触

每个活跃六维触点有 10 条金字塔约束行。独立测试让接触点初始切向速度为零，给一个转动轴 1.2 rad/s 的相对角速度，并施加 15 N 法向预载。下表是 S4000 的首步接触偶矩。它直接取自接触坐标系，普通接触力相对物体质心的力臂不会被算成滚动摩擦。

| 与胶囊接触的物体 | 绕接触法向扭转（N·m） | 绕切向滚动（N·m） |
| --- | ---: | ---: |
| 平面 | −1.74784 | −0.95854 |
| 球 | −1.16221 | −0.66134 |
| 另一胶囊 | −1.16636 | −0.64997 |

六条轨迹各连续运行 8 步，逐步核对独立 CPU MuJoCo，六组受激转动分量累计做功均为负。首次运行发现 MCC 5.2 无法编译扩展几何核中计算盒顶点坐标位的动态整数除数。三个局部轴的除数实际恒为 1、2、4。改为固定除数后，S4000 的六组运动及六组准入负控均通过；同一几何核的 NVIDIA CUDA 回归覆盖 11 种流形，包括胶囊、盒侧面、棱角和斜置盒。MUSA 的六维盒接触与四维接触仍会在设备分配前拒绝。

## 原 29 关节 G1 的半秒站立对照

S4000 完成了 250 个 2 ms 步、2,500 次设备内核调用。使用此前验收的 G1 基本几何模型，左右脚各七个胶囊，保留原 502 个候选几何对、942 个候选槽及自碰撞筛选。两个后端都从模型零姿态开始，用各自当前的状态和同一组受限 PD 参数追踪部署配置中的关节目标。没有施加外部扶持力。CPU MuJoCo 独立推进一条轨迹。[原模型定义](../benchmarks/p65-mujoco-tyx-mixed-constraints.md)

| 指标 | S4000 | CPU MuJoCo | 既定站立门槛 |
| --- | ---: | ---: | ---: |
| 最大底座水平位移 | 100.82294 mm | 100.82301 mm | ≤100 mm |
| 最大底座倾角 | 0.05946 rad | 0.05946 rad | ≤0.35 rad |
| 最大足中心水平位移 | 0.135 mm | 0.135 mm | ≤80 mm |
| 后 0.25 s 平均竖直足地合力 / 自重 | 0.98585 | 0.98585 | 0.7–1.3 |
| 后 0.25 s 双脚各承载超过 1 N | 125/125 步 | 125/125 步 | ≥90% |

两条轨迹都在第 249 步（0.498 s）首次超过水平位移门槛，站立验收均返回失败。全程两条轨迹的最大底座位置差为 0.000211 mm，最大关节角均方根差为 5.82 × 10⁻⁶ rad。末步位移几乎全部沿世界 +X 方向。足中心移动很小，说明这里主要观察到上身相对支撑脚前移。该控制器只追踪关节目标。下一项控制实验应记录质心水平投影和足底压力中心，再与加入平衡反馈的独立控制器对照，沿用同一模型、初态和门槛。

设备和独立 CPU 同步记录的整次诊断耗时约 130 s，包含读回、核验与轨迹写盘。最终示例另用两步在设备推进区间禁止主机动力学与 Tyx 通用调用，检查 20 次原生内核完成及状态缓冲切换。站立与舞蹈策略使用不同的控制器；本次只验证固定目标 PD。

复现需要 MuJoCo 3.3.7、固定版 Tyx、PyYAML 和 Unitree `unitree_rl_mjlab` 在 `1425b15f73bd4095f0df53709d7c389c3eb9e790` 的本地检出。准备脚本会验证模型 XML 和部署参数哈希。输出路径须为新文件。250 步命令会按上述水平位移门槛返回非零状态，并保留逐步记录和摘要。

```bash
python examples/prepare_g1_stand_assets.py --upstream /path/to/unitree_rl_mjlab --output g1-stand-assets
python examples/g1_musa_stand.py --assets g1-stand-assets --steps 2 --output probe.jsonl.gz
python examples/g1_musa_stand.py --assets g1-stand-assets --steps 250 --output stand.jsonl.gz
```
