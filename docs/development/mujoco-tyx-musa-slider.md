# 在摩尔线程 GPU 上运行 Phymeo 滑块动力学

`MuJoCoRuntime(..., mode="musa")` 使用 CPU MuJoCo 编译模型，然后由 Phymeo 在 MUSA 设备上推进状态。当前入口支持一个无限位移动关节、无接触对的模型，可在一个批次内用不同控制推进多个同构世界。

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

CPU 完成 MJCF 编译。Phymeo 复用现有 Tyx 动力学内核的中间表示，按设备报告的架构用 MCC 编译。GPU 保存当前状态和候选状态，成功读回并检查整批候选结果后才切换。控制目前从主机上传，显式 `snapshot()` 可在 CPU 上计算派生观测；验收逐步禁止在 `step()` 内调用 CPU MuJoCo 积分、前向计算或 Tyx 解释器。接触、多关节、训练张量互操作需要后续单独验证。该入口与 Vulkan 渲染的设备选择各自独立。

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

同机另验收了两个边界。候选读回故障时，两世界均保留上一帧的位置、速度、时间与状态版本，恢复读回后只推进一个有效步。带球–平面接触的模型在分配设备缓冲前被明确拒绝。另有 3 项原有 Tyx 解释路径回归通过。本次运行没有单独计时 JIT、传输及稳态积分，不能由验收时长推算吞吐量。
