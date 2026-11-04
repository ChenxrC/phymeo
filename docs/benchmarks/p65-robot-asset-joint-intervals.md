# 机器人资产的关节区间与连续碰撞

2026-11-04 后续新增双活动转铰的球–球连续路径，支持异向轴和前置滑轨。本文的单铰链能力、当时的双铰链拒绝和旧收据保留为首轮基线。[双铰链实测](p65-two-hinge-ellipse.md#双铰链连续检测)

`asset.collision_motion(q, q_end)` 返回所有已注册形状在直线关节坐标区间上的世界路径，可直接作为 `detect` 的 motions 参数。`asset.collision_motion(q=q)` 仍返回原静态姿态。归一化时间 t 对应 q(t)=q+t(q_end−q)。例如可以传入一次真实积分漂移的首末关节坐标；若一个显示帧内有多个加速度不同的物理子步，应按原物理漂移分别构造区间。[关节路径实现](../../src/phymeo/builders/_robot_joint_motion.py) · [普通资产入口](../../src/phymeo/builders/_robot_asset.py) · [USD authored-frame 入口](../../src/phymeo/builders/usd.py)

## 每个碰撞体的运动

从该碰撞体所属 link 追溯到固定根。未变化的关节保留其非零初始 FK。所有活动滑轨的世界轴固定时，整段平移为各轴乘有符号坐标增量之和。纯平移使用零角的 `RigidTurn`，所有材料顶点共享一个位移。若有一个活动转铰，且活动滑轨均在它上游，则枢轴为 FK 给出的初始关节世界原点 P，转轴为 FK 的世界单位轴 a，滑轨总位移为 d。初始世界点 X 的整条路径为 P+t·d+R(a,t·Δq)(X−P)。Δq 直接使用标量坐标差，保留负转和整圈。各向异性形状保有已注册的局部朝向和初始 body XYZW；USD 从真实 authored link 与非零 child anchor 得到初始材料点。

每条有碰撞体的支链独立判定。不同分支可以获得不同世界轴，固定根和静止支链保留单姿态。对同一注册形状支链上两个活动转铰，或活动转铰之后还有活动滑轨，当前入口给出包含 link、碰撞体和两个关节名称的错误。未挂碰撞体的后代不影响其它材料路径。坐标保持模型本来的 rad 或 m，可直接接受求解器给出的状态；位置限位没有被当作插值裁剪。配对是否具备连续窄阶段能力仍由原碰撞管线逐项报告。

## 端点均分离的真实弧

固定球心位于世界 (0,±0.75,0) m，半径为 0.0625 m。一个同半径胶囊的轴在起点覆盖 X=0.5 到 1 m，关节正向或负向走过 π rad。端帧的轴都在 Y=0 平面，真实指针中途越过固定球。首次触碰时 0.75·cos(πt)=0.125；故 t=acos(1/6)/π。胶囊中心线足点的半径为 √35/8 m，沿作者 from→to 的参数 u=√35/4−1。已有 MJCF 注册按本地 −Z/+Z 排列为 (to,from)，对应原材料 ID0 的权重为 u，ID1 为 1−u。保留了首次用相反编号读取失败的收据，补充了对本地端点和真实球皮见证的独立重建。正反两次及全体共同旋转/平移的留出均检查同一时间与材料地址。[真实 MJCF](../../tests/reference_newton/robot/joint_interval_arm.xml) · [独立解析与负控](../../tests/test_robot_asset_joint_sweep.py)

同一资产的另一区间包括两条不同方向的滑轨和一个转铰。枢轴从 (−0.125,0.15,0.2) 到 (0.625,−0.15,−0.2) m，独立支链绕 Y。包括远端静止关节的全部注册顶点，在多个内部 t 与直接 FK 相符。另有世界轴 (−0.8,0.6,0) 的局部斜轴留出、原闭合 OBJ 附于真实 URDF 滑轨后在 7/16 首次碰固定球，以及 OpenUSD child anchor=(−0.5,0,0) m 的受控实物构型。已接受的单关节动力学步提供了 q=0.125→0.5 rad，路径角度为 0.375 rad。

### 离散外部观测与连续首次时间

Phymeo 的真实 TYX CUDA 试片从设备 FK 读取中段姿态，启动 1 次 FK kernel 和 2 次候选 kernel。原连续窄阶段给出的归一化首次时间区间为 [0.4466996209612262, 0.44669962096304516]，含独立解析值 0.4466996209618863。两端静态表皮间隙约为 +0.776388 m。

Newton 1.7 中，指针形状 ID 为 4，所属刚体 ID 为 3，正负球分别为 0、1。CPU 与实际 CUDA 在正反两向各取五个关节时刻；接触行数依次是 0、0、1、1、0。t=0.4446996209618863 时实际表皮间隙约 +4.6440 mm，t=0.4486996209618863 时约 −4.6489 mm。此 0.004 宽的区间是两次离散原生观测之间的界。真实设备端点坐标与解析坐标的最大分量误差为 0.105 µm；各端另外五个复合滑移枢轴和独立转枝姿态亦相符。[Phymeo CUDA 连续结果](../../artifacts/p65-night-asset-joint-sweep-20261104-01/05-tyx-cuda-fk-arc/stdout.log) · [Newton CPU](../../artifacts/p65-night-asset-joint-sweep-20261104-01/06-native-cpu-samples/stdout.log) · [Newton CUDA](../../artifacts/p65-night-asset-joint-sweep-20261104-01/07-native-cuda-samples/stdout.log)

### 拒绝仍然有意义的边界

首次验收时，转动闭 OBJ 的刚体弧已经能从资产生成。当时的球配对记录为 `unsupported / mesh_rounding_or_rotation`。后续已接入单转轴网格与仿射球心的连续查询。同一原始球心在资产首顶点时实际判出初始占用，轴间隙为 −1/64 m、边界间隙为零。把球换成有限胶囊，仍报告原来的不支持原因。[后续实测](p65-rotating-mesh-sphere.md) 另一反例把铰链世界 X 坐标移至 2⁵² m，此处 0.25 m 的上游滑移会被两次 binary64 世界枢轴坐标相减全部吞掉。入口验证编码出的位移误差不超过 √ε 倍实际滑移矢量长度，超出时返回所属碰撞体与关节名称。同一大坐标下的纯滑移使用独立零角平移，能保留该 0.25 m。这个检查只界定移动枢轴的位移表示。[浮点反例](../../artifacts/p65-night-asset-joint-sweep-20261104-01/08-pivot-resolution-red/stdout.log) · [修复后](../../artifacts/p65-night-asset-joint-sweep-20261104-01/09-fp64-pivot-boundary/stdout.log)


## 复现

```powershell
python -m pytest -q -m "not tyx" tests/test_robot_asset_joint_sweep.py
python -m pytest -q --tyx-mode=cuda --require-tyx tests/test_robot_asset_joint_sweep.py::test_actual_device_fk_and_collision_candidates_receive_the_asset_interval
python tests/reference_newton/robot/joint_interval_poses.py --device cpu
python tests/reference_newton/robot/joint_interval_poses.py --device cuda:0
```

前两项使用项目 Python，USD 留出需同一环境中能导入可选的 OpenUSD。后两项使用独立 Newton 1.7.0 / Warp 1.17.0 环境，只在选定关节时刻调用原生 FK 与原生刚性碰撞。其报告给出采样跨越区间，连续首次时间由 Phymeo 原窄阶段与独立解析式核对。
