# 向同一个机器人场景发命令

`phymeo.runtime.robot_payload_scene.RobotPayloadScene` 在 HOST 上把导入的固定基座关节树、一个有质量和惯量的自由球或长方体、一个固定有限长方体桌面推进一个物理步。构造时交入真实 `RobotState`、`RigidBodyState`、物体尺寸和质量、桌面位姿与尺寸、每个关节的驱动增益和上限。通用 URDF 使用 `from_urdf(path, **scene_options)`，已折叠官方固定帧的经典 Panda 可传 `loader=load_panda`。

```python
from phymeo.runtime.robot_payload_scene import RobotPayloadScene

session = RobotPayloadScene.from_urdf(urdf_path, **scene_options)
before = session.observe()
session.step(target_q, target_velocity=target_v, effort_limit=effort_limits)
after, snapshot = session.observe_scene()
print(after.time_s, after.q, after.payload_position_m)
print(after.load_time_s, after.pad_normal_loads_N, after.fixed_table_wrench)
image = camera.capture(snapshot.model, snapshot.poses, position_m=eye, orientation_xyzw=camera_xyzw)
```

`step` 返回的位姿和速度属于本次已接受步末。力与实际采用的目标、速度、力限属于 `load_time_s` 对应的步首。初始观测的载荷时间为 `None`。省略 `feedforward` 时驱动补偿关节偏置，显式传全零向量可切换成无补偿。预测器只选择本步限幅后的驱动。实际的关节质量矩阵同时承担指面、连杆与台面的反力，自由物体使用本体主惯量、重力和同一步的实际合力矩。失败保留上次发布的完整状态、受力及时间。读出的数值数组只读。

`PadFace.named(link_name, collision_name, axis, side, offsets_m)` 每次构造场景都从这份资产的原链接和碰撞元素名重新绑定。两个名称必须唯一匹配。URDF 或 MJCF 为匿名碰撞体生成的序号名只能通过本模型的数字键选择。显式写在源资产中的名字可以使用相同拼写。如果材料缺失、歧义或重复选中，开始求解前就会报错。`PadFace(shape_key, axis, side, offsets_m)` 使用当前模型自己的数字键，在实际长方体碰撞形状的一面上选取有限材料点。偏移沿其余两个局部轴，按轴号从小到大排列。每点有指定的材料刚度与阻尼。该形状对自由物体使用这些点，避免整形状重复施力；它对桌面的碰撞仍单独计算。其余连杆支持球、有限胶囊和正交箱。均匀自由箱的桌面载荷沿有限顶面的八个材料角点计算，越出台面占地或沉到底面以下会拒绝该步。自由球使用有限桌面最近点。这里的检测在入步采样，不包含连续防穿透或机器人自碰撞。

`observe_scene()` 在同一场景发布锁内返回最新的物理读数与几何快照。另一个操作者若在本次观测前完成步进，返回的是该次已接受步，`step_number` 和 `target_q` 可辨认其来源。之后任一线程继续推进，已保留的这对读数仍对应同一物理步末。只取几何时，`geometry()` 保留资产原来的碰撞键，并为自由物体和桌面分配独立刚体键。返回的同刻模型、世界材料点、名称和键可直接交给几何消费者。世界坐标用米，四元数按 XYZW，从本体旋转到世界。

## 独立物理控制

- 导入含旋转和滑动两轴的 `arm.urdf`，半径 31.25 mm 的工具球对 125 g 长方体施加 2 N，同一步物体沿接触法向增速 0.016 m/s。正、反滑轨命令使 32 ms 时的实际滑轨分别为 +11.047 mm 和 −13.385 mm，两自由箱终点相差 4.496 mm。
- 经典 Panda 围绕 40 mm 实心球，起始开度与实际材料点相同。持续闭合与张开 30 ms 后，两侧闭合读数为 7.570 N、7.588 N，张开时为零。两支末端开度分别为 34.058 mm、68.710 mm。
- 斜置 30° 的有限台支持 1 mm 弹性压入的自由球，实测合力沿台面法向 2 N，台面反力相反。拒步用例让真实滑轨入步速度越过 URDF 上界，随后改为有足够制动力的合法指令，并与直接接受该指令的独立场景逐项比较。



## 实际抓起后修改放置目标

两次会话分别从桌面上的 50 mm、100 g 立方体和直径 40 mm、40.96 g 均匀实心球开始。Panda 使用同一幅真实关节树和有限指面。目标插值、控制参数与物体状态分别保存。到模拟时刻 1.05 s，操作者从 `observe()` 读到物体已经开始横移、离台且双侧受力，将原定的 75 mm 放置目标改为 55 mm。新指令与前段指令的位置及速度连续。桌面、重力、机身和物体状态始终由同一个会话维护。

| 物体 | 改目标时实际 X 速度 | 当时底部离台 | 当时两指法向力 | 1.35 s 底部离台 | 2.25 s 目标误差 | 最后 0.1 s 平均桌面承重 |
|---|---:|---:|---:|---:|---:|---:|
| 立方体 | 0.21091 m/s | 61.798 mm | 4.818 / 4.818 N | 59.140 mm | 0.120 mm | 0.981000 N |
| 实心球 | 0.21966 m/s | 64.729 mm | 3.222 / 3.235 N | 63.529 mm | 1.404 mm | 0.401818 N |

两者各走完 3,600 个 0.625 ms 物理步。改目标及转运时桌载为零，最后 0.1 s 两指残余作用力为零。全程未出现额外连杆–物体或连杆–桌面接触。末步箱的 X 速度约为零，球仍沿 X 以 3.025 mm/s 滚动。这里的放置判据是目标位置误差、桌面承重和指面卸载，未把球列作已静止。两条完整会话实测在 Phymeo HOST，本批没有新增 Newton 或设备积分对照。

可用相同入口更换物体，或在实际离台后输入新目的地。去掉 `--interactive` 时使用预设的 55 mm 输入。每份 JSON 含初始、夹合、抬起、改目标、搬运、张开与最终七次已接受事件，每次均保存 21 个形状的键、实际世界材料点和注册数据。力的采样时刻、关节及自由体状态独立标明。

```powershell
python examples/robot_payload_pick_place.py --shape box --output session-box.json
python examples/robot_payload_pick_place.py --shape sphere --interactive --output session-sphere.json
```

固定根箱体留出还保留了一个首次假设错误。十进制四分之一转角经 URDF 转成实际材料角点后，支撑侧 Y 有两个相差 2 ULP 的高度级，严格极点只有两个，得到边接触。独立未旋转、交换尺寸的同布局有四个共面极点。两者都向自由箱输送了 2 N，而远处的活动关节广义力为零。冻结基线缺少场景入口时退出 1。新增试片的首次临时目录权限失败及原面维数断言失败均保留。最终新专项 7 项、相邻球与箱载荷 2 项、Ruff 和格式检查均退出 0。[新专项](../../tests/test_robot_payload_scene.py) · [实际会话](../../examples/robot_payload_pick_place.py) · [两条完整原始记录与逐任务收据](../../artifacts/p65-night-robot-payload-workflow-20261105-01/)


## 跨资产重排的指垫

两份 URDF 只交换两块等尺寸材料的碰撞元素次序。相同的 `PadFace.named("tool", "jaw", 0, 1, [(0, 0)])` 在两次载入后分别选到该模型内不同数字键的真实夹面，入步样点都是 (0.003, 0, 0.25) m。两次均测得 2 N 指面法向力，125 g 自由球在 1 ms 内沿 X 增速 0.016 m/s。刻意将旧数字键用于重排后的模型时，样点落到远处的护块，指面读数为零，实际接触转换为夹面的普通结构受力记录。两种途径的差异没有以总力相等掩盖。

URDF 的 `collision_0` 与 MJCF 的 `geom_0` 均覆盖了显式作者名和匿名自动编号两种情况。匿名材料仍可通过当前模型的数字键使用。不存在、同链接重名以及同一真实形状同时被名字和数字键选中均会报错。[跨资产材料与受力对照](../../tests/test_robot_named_pad.py)


## 在操作者同时改指令时取观测

并发试片先向滑轨发送正向目标，使指令和真实 2 N 接触载荷被接受。读线程刚第一次完全释放场景原锁，另一线程就执行反向目标并完成第二步。分别调用旧的两个接口时，物理读数为 1 ms，几何已是 2 ms。用 `observe_scene()` 取回的读数和几何均为 1 ms，该步载荷采于 0 ms；下一次成对读取分别为 2 ms 和 1 ms。坐标用实际 FK 和自由物体八个材料角点逐项复核。两个线程通过事件确定先后次序，无需假定调度延时。

另一条试片在接受一步后让入步速度跨越滑轨上限。拒步前后，返回的物理读数是同一个不可变对象；每个碰撞键的世界材料点、状态时刻和载荷时刻均保持原值。[并发与拒步试片](../../tests/test_robot_observe_scene.py)
