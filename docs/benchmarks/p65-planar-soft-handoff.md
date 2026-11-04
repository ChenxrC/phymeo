# 双臂对软布的平面相对呈送

## 硬件可达方向与控制

原 `articulated-palm` 的两个近端关节分别绕 y 轴转动，点逆解只驱动这两个关节到 `(x,0,z)`。任意肩肘角均保持腕的世界 y 坐标；两张指面可沿 y 开合，行程受各自棱柱限位约束。因此控制将供臂的已测抬起高度作为竖直目标，在水平 x 方向呈送材料，接收臂继续调节 x/z 和接近速度。各臂保留原四项电机上限、关节弹性限位和有限指面捕捉规则。[真实硬件与执行器](../../src/phymeo/experiments/handoff_grippers.py) · [两臂实模型雅可比、镜像和空场测试](../../tests/test_planar_presentation.py)

新选项 `paired-presentation` 在原先两帧单臂承载条件成立后，用实际材料边地址和 B 工具中心的水平分离量确定供体呈送目标。初始水平差由双方各分一半。接收臂在每个控制帧重新观察正在变形的布，其参考还包含供臂尚未走完的水平行程。供体以原接近峰值速度的一半、原加速度限制向目标移动；其竖直速度按同一有界运动律向已持布时的高度制动。B 的接近指令仍须经过原 IK、运动布预测与腕/指杆有限几何检查。接收器要求闭指或供体的实测持载下降时，供体锁定当时实测位置。控制器给两臂下达受限电机指令，布的位置和速度由动力学产生。[平面相对参考](../../src/phymeo/control/planar_presentation.py) · [接入原手递手控制](../../src/phymeo/experiments/handoff_reception.py)

## 原始输入与资格

两案使用原 SOFT 布，质量 48 g、7×3 个节点、24 张材料面；每个 10 ms 控制帧包含 128 个真实物理子步，每个子步保留全部 276 对布自接触登记。指垫半径为 2 mm，背衬净空为 0.125 mm，重力加速度为 9.81 m/s²。竖直与初始 2° 的分支点分别是原已测受载状态 0.54 s 与 0.49 s，其前 42、36 帧的关节目标、工具目标和电机输入分别逐项重建。正式轨迹从先导终点恢复实际双臂 q/v 和全部布节点 x/v，同时重建控制器记忆。[原失败输入及有限候选对照](p65-soft-handoff-acquisition.md) · [可恢复的同状态真实推进](../../scripts/benchmark_planar_handoff.py)

原共同承载门要求 A、B 两边的左右指面各自都超过 0.20 N，各臂向上指面承载超过 0.08 倍布重，桌面载荷低于 0.02 倍布重，且连续两控制帧符合并已有 B 双侧捕捉。报告另区分预测结构挡住参考、实际实体间隙以及真实物理子步上的材料受力。[原任务门槛](../../src/phymeo/experiments/compliant_handoff.py) · [独立几何与物理归档审计](../../scripts/audit_planar_handoff.py)

## 复现

从静止初态运行原演示，或使用已存原始轨迹验证可恢复的有界接近段。保持相同模型可把初始倾角换成 2°。[公开示例](../../examples/compliant_cloth.py)

```powershell
python examples/compliant_cloth.py --case handoff --robot articulated-palm --fabric soft --self-contact --finger-pad-radius-m 0.002 --finger-backing-clearance-m 0.000125 --receiver-control paired-presentation --initial-tilt-deg 0 --output paired-vertical.json.gz
python scripts/benchmark_planar_handoff.py full-vertical.json.gz pilot-vertical.json.gz --frames 4
python scripts/benchmark_planar_handoff.py full-vertical.json.gz continued-vertical.json.gz --resume pilot-vertical.json.gz --stop-time 1.01
```

## 两个真实物理窗口

| 原始姿态 | 时间 / 子步 | B 两指合力峰 / 弱侧峰 | B 实体法向力峰 | A 双侧法向均超过 0.20 N 的实际采样 | 原共享承载门 |
|---|---|---:|---:|---|---|
| 0° | 0.54–1.01 s / 6,016 | 2.40349 / 0.04302 N | 0.42647 N | 0.54–1.009922 s | 0 帧 |
| 2° | 0.49–1.12 s / 8,064 | 0.51111 / 0 N | 1.54273 N | 0.49–1.061484 s | 0 帧 |

表中力峰遍历每个物理子步的入口样本。弱侧峰是两指法向力逐步取小再求峰。两个窗口都走满；四帧连续无两臂接触的提前停条件均未触发。2° 案后段 A 已低于双侧门，B 从未取得双侧。逐 10 ms 积分布节点重力、桌面、指面（含背衬冲量）与结构接触力，两案质心动量最大分量残差分别为 8.67×10⁻¹⁸、1.22×10⁻¹⁷ N·s。两案实测关节限位违反为零；指面最大采样穿入 0.716、1.844 mm。求解和数据均来自 HOST NumPy。[竖直原始轨迹](../../artifacts/p65-paired-soft-handoff-20261105-01/full-vertical.json.gz) · [2° 原始轨迹](../../artifacts/p65-paired-soft-handoff-20261105-01/full-tilted.json.gz)

0° 在原首次结构阻挡输入 0.89 s，同一左杆到 6 号布面的实际层间隙由 6.969 增至 22.394 mm，接收材料边 y 由 [1.422, 26.241] 变为 [−11.932, 11.234] mm。此刻离布最近的是腕到 15 号面的 1.428 mm；0.90–0.92 s 原接收预测的四种刹车至全速候选均受腕/2 号面阻挡，0.93–0.94 s 换为左杆/4 号面。末态 A 相对原轨迹向 +x 移动 36.15 mm、高度低 41.33 mm，B 向 +x 移动 12.34 mm、高度高 6.73 mm，布节点最大位移差 41.29 mm。目标边在每一控制帧都与最宽合法轴向带有交集，仍缺双侧承载。[竖直逐步审计](../../artifacts/p65-paired-soft-handoff-20261105-01/audit-vertical-result.json)

2° 在原首次阻挡原时刻 0.83 s，原目标边 y 为 [32.538, 33.463] mm，肩/肘任意角度和两根夹指的任何合法行程仅能让材料进入 [−14, 14] mm。呈送后，同一时刻目标边变为 [−23.172, −5.649] mm，其中一段可进入；原腕/2 号面间隙由 1.127 增至 3.832 mm，实际右杆/0 号面已伸入法向层 0.926 mm。0.87–1.12 s 控制帧中新的目标边整段再次超出最宽轴向带，0.49–0.59 s 亦如此。末态 A 相对原轨迹移动 (+20.92, −41.74) mm、B 移动 (+30.02, +8.66) mm，布节点最大位移差 118.74 mm。这个机械可达反例限定于相应帧和已登记材料边。[2° 逐步审计](../../artifacts/p65-paired-soft-handoff-20261105-01/audit-tilted-result.json)

## 留出与停止条件

在 0.54 s 的真实竖直受载状态，把两个刚体树的固定基座、布节点及速度、桌面绕重力轴一起转 180°，保留原广义电机输入。128 个物理子步后，经同一变换核对的节点位置、速度、关节位置和逐步每指合力误差均为零，测到指面合力峰 0.92577 N。将原布整体沿 y、z 各移 1 m 后，仍开启所有接触并施加相同电机力，128 步的指面、桌面和布–臂结构受力全零，质心速度增量与重力解析值最大差 4.17×10⁻¹⁷ m/s。控制参考另有 x 镜像与空载无动作的独立断言。[实际留出](../../artifacts/p65-paired-soft-handoff-20261105-01/physical-holdouts.json) · [可重复驱动](../../scripts/check_planar_handoff_holdouts.py)

两种原目标均未完成共享承载，按预先定义的止损条件停止调整同一路线。下一步可比较具有独立横向腕位移的机构，或在当前机构上允许接收器从多个真实材料边中选择同时满足轴向可达和杆体净空的边。后一项改变目标材料地址，需另行用原两臂承载门验证。
