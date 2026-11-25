# 在摩尔线程 GPU 上运行 Phymeo 滑块动力学

`MuJoCoRuntime(..., mode="musa")` 使用 CPU MuJoCo 编译模型，然后由 Phymeo 在 MUSA 设备上推进状态。当前入口支持一个无限位移动关节，可以自由运动，也可以让一个球形碰撞体与一个静态平面产生无摩擦法向接触（有效 `condim=1`）。同批世界共用模型，各自保存初态、控制和接触状态。

## 运行

在装有 MUSA 驱动和 SDK 的 Linux 主机上使用 Python 3.11。编译器需要 `mcc`，运行时需要能够加载 `libmusa` 和 `libmusart`。以下命令从 Phymeo 仓库根目录执行，安装固定版 Tyx 需要该仓库的读取权限。

```bash
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MUSA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install -e '.[mujoco]' -r requirements/tyx.txt 'pytest>=8,<10'
PHYMEO_TEST_MUSA=1 python -m pytest -v --require-tyx tests/test_mujoco_tyx_musa.py
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

CPU 完成 MJCF 编译。Phymeo 复用现有 Tyx 动力学内核的中间表示，按设备报告的架构用 MCC 编译。GPU 保存当前状态和候选状态，成功读回并检查整批候选结果后才切换。控制目前从主机上传。要读取此次运动实际使用的接触力，取 `sim.integration_snapshot()[world].last_step.contacts`，其中 `wrench_contact_frame[0]` 是步首求出的法向力（N）。显式 `sim.snapshot()[world].contacts` 则会在 CPU 上按步末姿态重新计算接触。多关节、摩擦和训练张量互操作需要后续单独验证。该入口与 Vulkan 渲染的设备选择各自独立。

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

本批在 S4000 上整批发射并完成 720 次内核。抽查接触转换和末帧时，直接读取 GPU 已接受缓冲中的几何 ID、接触数和法向力。接触容量设为零、仅一个世界落在接触区时，整批返回容量错误，两个世界均保留上次状态。测试还覆盖了摩擦模型拒绝、无接触回归及原 Tyx 解释路径，合计 7 项通过。
