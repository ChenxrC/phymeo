# 一个 URDF 与多个自由物体的同一步接触

`ContactSceneBuilder.add_urdf` 将一个固定基座机器人和多个真实自由刚体放进同一 HOST 场景。全部接触读取同一入步状态；完整关节树汇总受力后只推进一次，所有自由体也各推进一次。原单物体 `RobotPayloadScene` 保持原样。本页是场景计划 A1 的首切片，功能窗口为部分验收：Panda 预载保持和放开通过，并联 mimic 夹具在冻结初态下未保持连续双侧载荷。

## 公开入口与物理时间

```python
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

builder = ContactSceneBuilder()
# 注册普通 URDF，或显式选 load_panda / load_mimic_urdf。
builder.add_urdf(
    10, local_urdf, robot_state=initial_robot_state,
    kp=joint_kp, kd=joint_kd, effort_limit=motor_limits,
    pads=semantic_pad_faces, loader=asset_loader,
)
# 同一 builder 的 add_box / add_sphere 声明多个自由体和固定有限支撑。
# 每两个注册者都需 pair(..., RobotContactMaterial(...)) 或显式 None 排除。
with builder.build(backend="host") as scene:
    external = scene.inputs()
    external.rigid.force[scene.rigid_rows[0, payload_id]] = force_world_at_com
    external.rigid.torque[scene.rigid_rows[0, payload_id]] = torque_world_at_com
    frame = scene.step(
        dt_s,
        actions={10: RobotCommand(target_q, target_velocity=target_v,
                                  effort_limit=motor_caps, feedforward=joint_effort)},
        external=external,
    )
    reading, geometry = scene.observe_scene()
```

`RobotCommand(target_q, target_velocity=None, effort_limit=None, feedforward=None)` 使用资产的独立关节顺序，mimic 的命令只给独立驱动轴。省略 feedforward 时采用本步偏置补偿；显式给出时替换这项补偿，整个电机命令仍受额定力矩/力上限约束。自由体的外力、外力矩均为世界轴、作用于真实 COM；固定逆质量支撑保持不动并记录吸收的反力。

`frame.time_s`、`robots[address].q/v` 和 `frame.rigid` 是接受步末；`load_time_s`、`entry_q/v`、`entry_rigid`、驱动力与接触力属于该步入态。`loads.inputs` 仅按 `rigid_rows` 放置自由体/固定支撑载荷，机器人另读 `contact_effort`、`contact_effort_full` 与 `link_wrenches_world`，不能将它们再施一次。mimic 同时保留原树与约束每行的反力回执。

`observe_scene()` 在同一把锁内返回观测和完整 `CollisionModel` / `CollisionMotion`，供同刻测距或回放。`shape_key(robot_id, link_name, collision_name)` 在本次构建内重解析有来源的形状名；加载资产的局部 body ID 会避开场景 ID 重映射。`assets[(world, robot)]` 是独立的不可写数据快照，包含完整树、惯量、mimic 投影与偏置。`state` 和载荷输入数组返回独立副本，其余观测数组使用不可重新开启写权限的 bytes 缓冲。

## 支持范围

- 一棵固定基座树、一个 world、HOST；可通过显式 loader 复用普通、固定工具折叠、Panda 或 mimic 资产。返回值必须是 `RobotAsset` / `MimicRobotAsset` 或其合法子类，opaque 委托对象会拒绝。
- 自由体为 COM 居中的均匀实心球或任意世界姿态的主轴箱；有限支撑为同域固定形状。机器人接触复用实际 sphere、capsule 和 cuboid，其他形状在构建时报错。
- pad 仅在对动态物体时替代它对应的实体接触，避免双加；对支撑仍检测实际实体。非 pad 连杆、自由体互撞、有限台面均保留。
- 有机器人时 pair 只接受 `RobotContactMaterial` 或 None，拒绝混入 `NormalLayer`；旧无机器人入口继续使用自己的材料合同。
- 这是入步瞬时顺应接触，不做步内 CCD、任意自碰撞、软体、浮动基座或机器人 CUDA/interpret 求解。未声明的 pair、超限命令、超域姿态、无法确定法向及不支持参与者拒绝，所有已接受状态、时刻和几何不前进。

## 独立解析与安全留出

自写 [滑轨 URDF](../../tests/reference_newton/robot/multiobject_pusher.urdf) 为 2 kg 的 X 滑轨，工具与两个 1 kg 自由球的半径均为 1/16 m，球心 X 为 0、0.124、0.248 m。两处各压入 1 mm，材料 2000 N/m、零阻尼/摩擦。1 ms 后应有机器人速度 −0.001 m/s、第一球速度 0、第二球速度 +0.002 m/s，机器人位移 −1 µm、末球位移 +2 µm，系统 X 动量为零。

这一例同时辨别漏掉物体互撞、重复推进机器人、按参与者次序串行重采样。另有斜轴与共同速度、箱/球留出、注册顺序和原 body ID 冲突、实际 COM 外力/力矩与关节 feedforward、越限拒步后重试、观测副本和未知输入检查。sphere 与旋转 box 的 sampled-pad 留出分别核实体材料表面，并保持实际共作用点和力不变：`(surface_second - surface_first)·normal = gap`。

父串行执行的聚焦功能节点共 18 项通过。真实首次失败未删：正常入口缺少 add_urdf；两项 pad 表面把内部施力点当成实体边界；四项资产别名导致外部质量修改污染本场景与第二次 build；opaque loader 缺少类型门。对应首红为 1、2、4、1 项。修复后分别验证，未重跑旧大型集合。所有源码/命令/输出/原始退出码保存在 [本批证据](../../artifacts/p65-postnight-batch1-20261105-01/)。

## 两个冻结的实际资产窗口

[同一公开例程](../../examples/urdf_multiobject_contact.py) 只通过目标与真实驱动变化运行，初态物体已处于两指之间。它不宣称从桌面获取或完成完整抓放；有限台面承接指曾出现真实台面载荷，也不等于末态已静止。

| 输入与结果 | 经典 Panda | 并联 mimic 滑轨 |
|---|---:|---:|
| 主物体 | r=20 mm、40.96 g 实心球 | 40 mm、100 g 箱 |
| 第二自由体 | 30×24×28 mm、60 g 箱 | r=15 mm、30 g 球 |
| 冻结步长与总步数 | 0.625 ms，288 | 1 ms，192 |
| 连续保持步数 / 释放步数 | 32 / 256 | 32 / 160 |
| 完整窗口实际推进 | 是 | 是 |
| 保持窗每侧始终 >0.5 N | 是 | **否** |
| 保持净空、真实释放、下降、主物体承接、次物体支撑、冲量门 | 全通过 | 全通过 |
| 最大逐体线冲量残差 | 1.30×10⁻¹⁸ N·s | 3.04×10⁻¹⁸ N·s |
| 原始 CLI 退出码 / 冻结检查 | 0 / 8项通过 | **1 / 7项通过、1项失败** |

Panda 末步主球竖直速度仍为 −0.15457 m/s；mimic 主箱为 −0.04424 m/s。不能将这些短窗解释成稳态放置。两条 JSON 均包含全部真实 q/v、双方接触、外载、时间，以及初态/保持末/最后接受状态的同刻完整几何。

- [Panda 原轨迹](../../artifacts/p65-postnight-batch1-20261105-01/a1-panda-window.json)
- [mimic 原失败轨迹](../../artifacts/p65-postnight-batch1-20261105-01/a1-mimic-window.json)

### mimic 首失败的最小力学解释

该夹具关系为 `q_right = -q_left + 0.004 m`，两指质量为 0.2 和 0.3 kg。初始独立闭合量 0.014 m，使两内指面 X 为 0.232 / 0.268 m，而箱面为 0.230 / 0.270 m。因此每个采样点预压 2 mm，每侧四点在 1200 N/m 下合计 9.6 N，作用于独立闭合轴的接触广义力为 −19.2 N。

首步目标就是初态、速度为零，闭合驱动力实际为 0；折算质量 0.5 kg，故 `a=-19.2/0.5=-38.4 m/s²`，1 ms 后 `v=-0.0384 m/s`、`q=0.0139616 m`，与原记录一致。随后闭合力在 load_time=0.006 s 达到 8 N 上限，但初始开离动量仍在。首个 >0.5 N 门失败发生于载荷时刻 0.014 s、接受时刻 0.015 s，两侧各 0.195504 N。

载荷时刻 0.015–0.027 s 两侧均为零。首次零载入步仍有约 55.95 µm 几何压入，但开离速度约 0.10946 m/s，使单向弹簧/阻尼值截为零；随后的 q 最低为 0.0117352 m，低于该平面对的接触门 0.012 m。0.028 s 再接触时各侧为 0.900230 N。双侧轨迹始终对称，没有单侧身份互换或遗漏反力的证据。

这与[旧渐闭后抬起的 mimic 窗口](p65-urdf-mechanical-mimic.md)初态不同：旧例从 q=0 逐渐闭合，较晚保持窗约每侧 4 N；本次是在零速度、零初始闭合驱动下直接施加 2 mm 预压。此次“连续32步保持”的预测未达，裸断言与 CLI 退出码 1 均保留，无 xfail/skip、无阈值或材料调整。若后续改变加载协议，必须独立冻结方案。

## Newton 1.7 的独立积分参照

[独立参考驱动](../../tests/reference_newton/robot/multiobject_explicit_native.py) 由 Newton 真正加载同一滑轨 URDF 和两个自由球，从自己的入步位置计算两处显式材料力，然后将双向身体力交给 Featherstone 一次推进。`contacts=None` 排除了原生接触重复施力。此为共同接触律下的推进参照，**不是 Newton 原生接触检测或原生 manifold 的复现**，也不是本次 Panda/mimic 完整窗口的跨引擎通过。

CPU 实测质量为 2/1/1 kg，三个球与有限箱对应的原生 shape body 为 1/2/3/−1。FP32 初始距离 0.12399999797344208 m 导致每处力 2.0000040531158447 N；末速分别为 −0.0010000020265579224、0、+0.0020000040531158447 m/s，无角速率。总动量和按施加力计算的逐体冲量残差都读得零；8 个冻结门全部通过，RAW_EXIT=0。本批未重复 CUDA 版本。

[原生 CPU 读数](../../artifacts/p65-postnight-batch1-20261105-01/a1-three-body-native-cpu.json)保留形状、入步/出步位姿和 twist、实际完整施力向量及源 URDF 摘要。所有项目命令由父串行执行；九个新 Python 文件的初次格式修订保留原件并通过规范化 AST 与导入绑定一致性检查，opaque 小门及新增负控又单独完成 lint/format。

## 重现入口

```powershell
python -m pytest tests/test_robot_multiobject_scene.py tests/test_robot_scene_boundaries.py tests/test_robot_scene_pad_witness.py tests/test_robot_scene_asset_ownership.py tests/test_robot_scene_loader_boundary.py -q
python examples/urdf_multiobject_contact.py --asset panda --out panda.json
python examples/urdf_multiobject_contact.py --asset mimic --urdf tests/reference_newton/robot/mimic_jaws.urdf --out mimic.json
# 独立 Newton 1.7 环境；没有 Phymeo import 或轨迹状态注入。
python tests/reference_newton/robot/multiobject_explicit_native.py --device cpu --out native.json
```

mimic 命令以及 `test_robot_scene_asset_windows.py` 中的 mimic 参数保留已知连续持稳验收失败，预期当前返回非零。运行成功、物理承载目标和完整任务完成在本页分开记账。完整桌面获取、多机器人、多世界、柔性参与和 GPU 关节场景仍属后续节点。
