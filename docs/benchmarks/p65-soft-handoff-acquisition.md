# 软布双臂接取及搬运

模型为两只铰接夹爪、21 个质量节点、24 个三角面和一张 48 g 软布。两个初态分别竖直、倾斜 2°。手指圆边半径 2 mm，背板间隙 0.125 mm，每帧 128 个物理子步。两条轨迹都从静止初态开始，使用原夹爪力矩、闭合速度、布参数与任务门槛。

## 原世界坐标保持控制

| 初态 | 真实完成的阶段 | 失败点 | 峰值采样穿透 |
|---|---|---|---|
| 竖直 | A 夹持并抬离桌面。0.87 s 时布的最低点为 98.53 mm，桌面载荷为零。 | 交接窗口的 8,320 个子步中，B 两指的法向力始终为零。A 的最后一次接触在 1.11774 s 之前。1.52 s 交接超时，桌面承担 0.47088 N。 | 0.468 mm |
| 2° | 0.90 s 时 A/B 向上承担 0.170/0.312 N。0.94 s 时 A 完全松开，B 承担 0.382 N，布仍离台。 | 1.27 s 后 B 失去稳定双侧载荷。1.30 s 时接取角点已滑出指垫投影。2.04 s 搬运超时，桌面重新承重。 | 1.352 mm |

接取材料边来自参考网格的节点 13 和 20。竖直轨迹进入交接时，节点 20 在两指之间，节点 13 在外侧。控制器在这时固定 B 的世界目标。到 0.97 s，实际工具与两节点中点的 XZ 距离为 77.15 mm，远大于指垫 19 mm 的短边半宽。倾斜轨迹的两个指垫在交接时确实作用到指定材料边，末段载荷集中在节点 20。1.30 s，该节点到左指垫投影边界的有符号余量为 −1.23 mm。原机器人结构穿透与关节越界的峰值均为零，两条轨迹的背板总冲量均为零。

复核从五个贴片的逐子步载荷、实际指垫四角、材料位置和速度重新计算。原始记录位于 `artifacts/p65-collision-m4-loads-local-20261102-01/with-backing-v1/`。两条原始运行分别是 `76` 和 `77`，均正常报告任务失败。独立核对为 `83`，接取材料与指垫的逐帧诊断为 `90b`。物理任务状态与审计退出码分别记录。

## 下一项对照

接取时在两个有限指垫的共同投影和开口内选取一个实际材料点，记录它在原材料边上的权重。闭指期间按该点的实测速度驱动 B。建立双侧力后转为承载，搬运时使用指垫内可用的几何余量限制工具超前量。控制只给现有 IK 和有限力执行器下命令，布的运动仍由接触、材料和重力决定。原控制另行保留，原两种初态、材料、载荷与搬运判据继续适用。

机器人交接实验中，接收方通过触觉反馈调节夹持力，递出方在检测到接管后才释放。布边跟踪实验也使用局部观测来定位实际材料特征。这里使用仿真的材料坐标和载荷作为观测，以上论文不为本模型的参数或任务成功提供验收。[机器人交接实验](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.672995/full) · [布边跟踪实验](https://www.roboticsproceedings.org/rss19/p018.pdf)

## 材料边跟踪的实测

新选项 `--receiver-control material` 使用同一参考边上的固定权重，两条轨迹都选到节点 20。竖直和 2° 初态开始跟踪之前，分别有 88 帧和 78 帧与原控制逐项相同，覆盖布的位置和速度、两台机器人的状态、指垫几何及载荷。独立程序逐帧从前一已发布状态重算材料点、速度、跟踪命令及有限指垫允许的超前量。

| 初态 | 真实进展 | 开放门槛 |
|---|---|---|
| 竖直 | B 在 328 个子步中两侧均超过 0.20 N。A 的双侧载荷最后超过门槛的时刻为 1.01047 s，B 首次超过门槛为 1.00516 s，两者重叠不足一个显示帧。 | 未建立持续共同承载。1.41 s 后第六个子步触发 17.5483 mm 硬上限拒绝。用真实关节输入单独重现该步，材料地址 `face_10/6` 在桌面边缘 x=0.269987 m，深度 17.548301 mm，物理入口状态完整保留。 |
| 2° | A/B 在 0.90 s 时真实共同承载。0.94 s A 松开，B 独立承担 0.289 N。B 搬运阶段的布心水平位移峰值为 9.13 cm，末帧仍比接管时前进 7.92 cm。末帧双指法向力为 0.454/0.389 N，布最低点 68.88 mm，桌面载荷为零。 | 2.04 s 工具与目标相差约 5.08 mm，原门槛为 5 mm。原 1.1 s 搬运窗口已满，未进入放置。 |

证据位于 `artifacts/p65-collision-m4-loads-local-20261102-01/with-material-feedback-v1/`。原始运行 `103/104`、独立承载和材料审计 `105/106/108`、首次拒绝的定位 `110` 均已保存。指垫跟踪改善了接取和独立搬运。完整系统仍要求在同一条物理轨迹上完成放台与撤爪。下一对照让 A 先独立抬布，在可观测的交接位置保持，由 B 取得载荷后再协同运动到原转运高度，仍使用原夹爪、电机、材料、超时与位置门槛。

## 协同接取的对照设置

选项 `--receiver-control cooperative` 要求 A 实际独立承载且布离台两个连续帧，B 保持原开口和起始工具目标。A 在观测到的位置等候，B 只在两个有限前表面内选取材料地址。原双指压力、双方垂直承载、桌面卸载条件达到后，两只工具从各自实测位置同步送往转运高度，再允许 A 松开。共同运动的峰值工具速度取原 A 转运曲线的峰值，原阶段超时和 5 mm 到位门槛保持。连杆夹带的布料载荷也计入 A 单独夹起及最终完全松开的观测。两只夹爪在交接前都失去布料时记录真实失败，避免继续追赶无主的材料点。[运动指令和可重复控制](../../tests/test_cooperative_grasp_transfer.py) · [布面–结构连杆独立验收](p65-cloth-link-response.md)

完整新装配入口为 `python examples/compliant_cloth.py --case handoff --robot articulated-palm --fabric soft --receiver-control cooperative --initial-tilt-deg 2 --output soft-mounted-tilted.json.gz`。竖直对照使用 `--initial-tilt-deg 0`。两个初态都需要真实动力学通过全部阶段。

## 新装配的竖直轨迹

新装配从原竖直软布静止初态运行了 17,536 个真实子步。0.54 s 时 A 独自持布，最低点离台 26.85 mm，A 向上提供 0.5093 N，B 和桌面均无载荷。A 双侧载荷最后超过 0.20 N 的显示帧是 0.65 s，B 首次达到该条件是 0.85 s。B 在 0.87 s 取得双侧载荷，1.37 s 时仍持布；没有任何显示帧让两者同时达到交接门槛。原抬升时限到期，任务失败。

17,536 个子步中有 8,589 步发生结构接触。29,504 条活动面–杆记录均从当次材料点和相反载荷核对，66 个有结构接触的帧入口另用已存实际关节位置重算广义力。最大面–杆力为 6.036 N，最大结构穿透 1.600 mm。布的逐帧动量变化与重力、桌面、两臂指垫和结构载荷积分之和的最大分量误差为 4.17×10⁻¹⁷ N·s。

从 0.54 s 到 0.86 s，仅 B 腕就向上给布 0.13549 N·s。在 0.62 s 的记录中，B 腕已经加载，A 尚有双侧夹持。判别旁路将从真实 0.54 s 状态出发，保持同一 A 指令和全部接触形状，单独让 B 停在原处，检验 A 能否持到原接取时刻。此旁路只区别接近碰撞和供体自身滑脱。后续接近控制要使用真实腕、指杆和材料边的距离，并约束工具速度。MoveIt Servo 的邻近速度调节与 TrajOpt 对步内扫掠的处理给出了两种可检验的对照。[MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) · [TrajOpt 原论文](https://rll.berkeley.edu/trajopt/ijrr/2013-IJRR-TRAJOPT.pdf)

本机原始轨迹、逐帧动量和结构核对分别保存在 `artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/full-vertical.json.gz`、`cloth-momentum-vertical.json` 和 `structural-independent-vertical.json`。

## 两个初态的接近运动与独立供体对照

2° 布由 A 在 0.49 s 单独抬离桌面 32.60 mm，A 的向上载荷为 0.637 N，B 和桌面均未承载。已发布的 7,552 个子步中有 437 步出现布–连杆力，最大单对力为 17.96 N，最大结构层穿入为 3.720 mm。0.49–0.59 s 内，B 腕给布的向上冲量为 0.21807 N·s，B 尚未取得双侧指垫载荷。逐帧布动量与所有实际外载冲量的最大分量差为 8.51×10⁻¹⁷ N·s。[竖直轨迹回放](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/full-vertical.html) · [2° 轨迹回放](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/full-tilted.html)

2° 轨迹在 0.59 s 的下一显示帧被拒绝。以原入口状态和执行器输入复现时，此前 46 帧的控制目标逐项相同。第 97 个子步的前半段，A 右指杆与材料面 12 的轴线距离在两端分别为 67.04 µm 和 56.44 µm，整段运动上界为 70.47 µm。用真实关节弧计算中点得到 61.74 µm，两半各自的运动上界只有 35.23 µm，因此这个具体区间轴线严格分离。原运行完整回滚。此结论只解决首次拒绝的那段路径。[真实输入、区间与判定](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/tilted-rejected-link-interval.json)

竖直布的独立对照从原 0.54 s 的实际状态出发，沿用 A 的首个原始驱动输入，只将 B 保持在原始开口及停靠点。32 帧、4,096 个反馈子步连续到 0.86 s。A 每帧都有双侧载荷，B 与桌面的最大载荷均为零，末端 A 向上支撑 0.48945 N，布底仍离台 16.99 mm。原双臂同一窗口内 A 在 0.65 s 后失去所需双侧载荷。接近控制的独立反例还测得，原算法在一个 10 ms 周期把工具目标移动了 140.36 mm，原转运速度预算允许移动 3.34 mm。[供体对照结果](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/donor-hold-vertical-summary.json) · [工具运动负控](../../tests/test_receiver_approach.py)

## 受限的接近指令

接近参考从接收工具实测的位置和速度出发，同时限制速度与加速度，额度沿用原转运曲线。接收侧依据真实腕、前臂和指杆形状，在候选关节轨迹与按实测速度前推的布面之间检查接触层净空。遇到阻挡时逐步朝制动指令收缩，并记录阻挡的配对；若布朝制动中的手运动，仍保留真实受力与整步判定。工具命令的限速与真实关节速度分别记录。[参考运动与阻挡回归](../../tests/test_receiver_approach.py)

在 2° 布的原 0.49 s 单爪抬升状态，原先 36 帧的命令及第一步 A 驱动力保持一致。新指令推进 11 帧、1,408 个真实子步至 0.60 s，跨过原先被拒的帧。A 持续双侧承载，B 腕施加的冲量三个分量均为零。末帧布底离台 76.07 mm，A 向上支撑 0.420 N，B 尚未夹布。本段和 24 项 HOST 控制测试通过；完整原始初态还需要独立运行。[分段轨迹摘要](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-guarded-approach-v1/prefix-tilted-summary.json)

竖直布从原 0.54 s 状态继续到 0.86 s，32 帧、4,096 个实际子步中 A 持续双侧夹持。B 的实际工具移动了 5.80 cm，最大实测速度 0.330 m/s，腕部冲量为零。末帧 A 向上支撑 0.48945 N，材料边仍在 B 的有限指垫之外，平面内最近可达材料点距工具 49.6 mm。两种新分段逐帧核对全部 276 个布中面对，均未检出异处自交；布动量最大分量误差为 1.13×10⁻¹⁷ N·s。B 尚未接管。这些分段不能代替从两种原始初态运行的完整系统轨迹。[分段独立审计](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-guarded-approach-v1/independent-prefix-audit.json) · [真实工具行程](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-guarded-approach-v1/receiver-reference-observation.json)

## 全布自接触下的接收时机

原竖直软布从初态运行到 1.02 s，共 103 个显示帧、13,056 个物理子步。全部 276 个同体面对应在每个材料子步检查，本条轨迹没有激活布–布法向力。A 在 0.54 s 单独抬起布，B 始终没有形成双侧夹持，两只手在放台前丢布。最大指垫穿入 0.543 mm，结构接触层最大穿入为零。按全部原始力通道核对，逐帧布动量最大分量误差为 8.68×10⁻¹⁸ N·s。[逐步核对](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-full-self-v1/full-vertical-audit.json) · [原始失败回放](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-full-self-v1/full-vertical.html)

布边首次进入 B 的两个指垫轮廓是在 0.93 s，留有 3.5 mm 侧面余量的首帧为 0.94 s。A 双侧法向力最后一次都超过 0.20 N 也在 0.94 s。旧的接收逻辑还要求布边距工具中心小于 5 mm，而两指之间实际可达的材料点最近也有 10.975 mm。B 在 0.96 s 才有可测的接触力。由原闭指行程 12.4 mm 和最大命令速度 0.09 m/s，完整行程至少需要约 0.138 s。[实际布边与载荷](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-full-self-v1/full-vertical-receiver.json) · [带几何余量的复算](../../artifacts/p65-collision-m4-loads-local-20261102-01/exterior-original-observations.json)

新控制让 B 在 A 抬布时张指预靠。它选取材料边上的固定地址，从 B 所在的一侧计算指垫包络相对整张当下布面的外侧目标，同时检查实际腕、前臂与指杆的接近轨迹。只有在 A 单独承载且上述材料边的一个点进入两个有限指面的安全内部后，B 才开始闭指。闭指结果仍由两侧真实载荷判定。在竖直与 2° 的原抬升入口，外侧目标扣除整个指垫半径后，对布都留有 5 mm 余量。控制器和夹爪回归共 32 项通过。两种初态的物理结果须分别记录。[有限指面的工具](../../src/phymeo/control/material_grasp.py) · [控制与反例](../../tests/test_exterior_grasp.py)

### 外侧预靠的竖直原始初态

同一正常 CLI 在 1.37 s 到达原抬升阶段时限，共 138 帧、17,536 个真实子步。A 在 0.54 s 单独抬布；此前 B 张指行进 29.05 mm，期间 B 上的布接触载荷为零。B 于 0.86 s 取得布边节点 13–20 之间的材料地址，该点距工具中心 15.97 mm，距指垫侧界 3.50 mm。B 在 0.99 s 建立双侧夹持，A 两侧都高于 0.20 N 的末帧为 0.95 s，两个承载窗仍差约 40 ms。末帧 B 两侧较小法向力为 0.590 N，桌面承担 0.215 N。交接未放行。[逐帧地址与夹持](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-vertical-exterior.json)

同条轨迹有 1,639 个子步产生布自接触力，涉及 16 对材料面，峰值 0.776 N，自接触净合力最大值 1.96×10⁻¹⁶ N。全部外载冲量与布动量变化的最大分量差为 1.39×10⁻¹⁷ N·s。指垫最大穿入 0.543 mm，结构穿入和关节限位违例均为零。下一次诊断应区分闭指行程、材料边沿夹爪轴向的运动，以及 A 在等待期间的滑移。[受力与阶段审计](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-vertical-audit.json) · [同源离线回放](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-vertical.html)

### 倾斜软布与两种布边路径

2° 布在 0.49 s 达到 A 独立承载。B 在这之前张指行进 43.76 mm，布对 B 的载荷为零。此后 B 腕与材料面 2 的预测路径在 0.79 s 首次阻挡接近，B 在 0.93 s 出现指垫力，但始终没有得到位于两个有效指面内的目标地址。1.38 s 时两只手均丢布，桌面竖向支撑为 0.371 N。139 帧、17,664 个物理子步中最大指垫穿入 1.875 mm，结构穿入及关节限位违例均为零。本条轨迹没有激活布自接触力。[原始阶段与载荷](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-tilted-audit.json) · [逐帧接近记录](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-tilted-exterior.json) · [2° 布回放](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-tilted.html)

竖直轨迹的真实子步采样把两个夹持窗之间的间隔缩小到 27.66 ms。A 最后一次合格夹持的采样时刻为 0.95265625 s，B 首次为 0.9803125 s，二者没有同时达标。按实际指面裁剪，0.86 s 已进入指面的那部分布边在计入 3 mm 轴向预留后，还差 0.504 mm 才能继续收指保持这份预留。同一材料边在指面投影之外仍有 9.93 mm 行程余量。2° 轨迹的 89 个接近帧中，23 帧连整条候选边都越过了张开指面的轴向范围，其中 9 帧同时记录了结构阻挡。[竖直子步载荷与行程](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-vertical-stroke.json) · [2° 子步载荷与行程](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/full-tilted-stroke.json)

### 先找到张开的夹爪能够接近的材料点

平面接收器在尚未看到可夹持地址时，沿同一材料边检查实际开口与布厚。优先保留仍在轴向余量内的先前地址；它越界时，选取当下可达的地址。整条边暂时不可达时，张指等待在材料外侧。实际闭指仍要求两个有限指面都能覆盖目标。查询接受任意空间朝向，在镜像和旋转夹爪上均有回归。36 项控制与有限指面测试通过。[通用开口查询](../../src/phymeo/control/material_grasp.py) · [反例与留出构型](../../tests/test_aperture_guidance.py)

在此前的 121 个实测接近姿态上逐帧对照独立裁剪，竖直轨迹的 32 帧中保留 17 个原地址、改选 14 个、等待 1 个；2° 轨迹的 89 帧中分别为 50、13、26。两组都是对旧姿态重算指令；真实运动另按同入口短窗比较。[原目标与候选边的距离](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/approach-address-differences.json) · [两条保存轨迹上的新指令](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/saved-guidance.json)

两条大场景的离线回放均已用浏览器核对。从竖直轨迹第 113 帧读取了 10 根布面自接触箭头，箭头起点与各自材料作用点完全一致，方向最大误差 1.58×10⁻¹⁶。2° 布第 92 帧的结构接触有三根箭头，B 腕的法向载荷为 0.582 N；播放器可到达末帧 138。[浏览器与原始轨迹绑定](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-exterior-self-v1/browser-force-audit.json)

两种初态分别重建了首个不同执行器输入之前的 54 帧与 36 帧控制，关节目标、工具目标和电机输出与原轨迹逐项相同。保留入口的布及机器人速度，每种初态再推进四个显示帧、512 个真实子步。竖直布从 0.66 s 改选地址，2° 布从 0.49 s 等待开口。两个短窗内 A 每帧都维持双侧夹持，B 尚未接触布，对应时刻布节点位置与原轨迹一致。[竖直短窗](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/window-vertical-summary.json) · [2° 短窗](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/window-tilted-summary.json)

### 对照结果与下一项验收

从同一批原始状态继续到原轨迹的接取时段，两种初态合计 12,544 个真实子步。竖直布在 0.86 s 取得地址，0.99 s 观察到 B 双侧夹持；A 与 B 的合格夹持仍相差 27.89 ms。2° 布的 B 始终未建立双侧载荷，两只手在 1.12 s 丢布，早于原协同轨迹的 1.38 s。两个拼接点的全部布和机器人位置、速度完全相同，逐帧动量与实际外载冲量的分量误差均小于 1.05×10⁻¹⁷ N·s。[竖直接取窗](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/capture-vertical-audit.json) · [2° 接取窗](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/capture-tilted-audit.json)

因此，既有 `--receiver-control cooperative` 保留张指外侧预靠及有限指面闭指策略。开口引导通过 `--receiver-control aperture-guided` 显式选择。当前 39 项控制回归通过。在两条完整协同轨迹的抬升阶段，两个初态各 125 帧的工具目标、关节目标、电机输出和控制观测均逐项一致。实验模式另核对了竖直 89 帧和 2° 99 帧的全部执行器指令。竖直公共前缀的 12 个旧观测没有两个新增引导地址字段，已用同次材料节点和分数独立绑定。[正常案例入口](../../examples/compliant_cloth.py) · [竖直协同指令](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/mode-cooperative-vertical.json) · [2° 协同指令](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/mode-cooperative-tilted.json) · [实验模式竖直](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/mode-aperture-guided-vertical.json) · [实验模式 2°](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/mode-aperture-guided-tilted.json)

两个最早的结构阻挡给出了下一项独立回归的输入。竖直布在 0.89 s 的 B 左指杆与材料面 6 离接触层还有 6.97 mm，2° 布在 0.83 s 的 B 腕与材料面 2 还有 1.13 mm。对每个真实入口，在原速度、加速度、关节约束下检查了 17 个平面参考，均未获得下一段的结构净空。分别提前 10 ms 和 30 ms 出发，使用真实 FK、两段或四段运动、仅由入口速度前推的布，两个对照的每组 17 个参考也都未通过净空检查。结果仅限这组方向和布运动预测。后续先用这两组状态、镜像和空场对照检验能提前形成通道的双臂接近路径，再用实际布和执行器推进到共同承载。[首个结构阻挡与各方向结果](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/first-structural-options.json) · [提前出发的参考路径](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-aperture-guidance-v1/earlier-structural-references.json)
