# A2：受限驱动力的渐进获取（冻结输入与实际结果）

基线 `69bd12d`。本轮仅新增 `control/gripper_acquisition.py`、两个 `test_gripper_acquisition*.py`、`examples/robot_force_acquisition.py` 和本报告，旧 A1 两个窗口及 mimic 2 mm 突加预载 RAW1 不动。所有项目命令由父串行执行。以下输入在首窗前冻结，末节给出实际结果：Panda 完成获取、抬持及释放；mimic 箱取得双侧力后滑移，原保持门仍未达。零摩擦与超载是预期失持的通过负控，不是抓取成功。

**首切片与方案。** 选用“实际双侧力反馈 + 有界关节目标渐进器”，继续让原 bounded PD、真实质量和接触律施力。不再添加另一套关节力解算或按绝对时刻闭爪。单固定根、单世界、HOST 的原边界不变。无桌面远距接近或完整搬运规划；起始物体在有限台面上、位于已张开的双指有效高度内。

**公共合同。** `GripperAcquisition(asset, *, robot, payload_key, pad_groups, joint_indices, open_q, closed_q, speed_limit, effort_limit, force_target_N, force_band_N, contact_min_N, dwell_s, loss_dwell_s, stop_speed, timeout_s).propose(observation, *, dt_s, posture_target=None, intent="acquire")` 返回不可变判定及 `RobotCommand`。open/closed/speed/stop 按所选独立坐标排列（m、m/s 或 rad、rad/s）；effort 为全独立轴 N/N·m。其他轴的 posture 由调用者提供，夹指只改所选坐标。力仅计指定 payload 与两组 pad 的实际 `robot-pad` 分量，不混入桌面或其他物体。

目标以不超过每轴速度上限的增量推进，参考以最后实际接受命令为锚点；双侧力进入目标带后停止继续闭合，任侧超带则有界退让，实际 q/v 用于制动与停稳判定。输出电机帽始终不超过配置和原资产额定值，不因失持提高增益或上限。`contact_ready` 只代表双侧力/速度持续达门，不宣称物体已抓稳。载荷时刻与步末 q/v 分开保存。计时只消费新的已接受步；同 step 重读或拒步重试产生相同指令，倒退、同序号改时刻、未来 load_time 或与上次请求 dt 不一致的下一帧拒绝。

**冻结控制量。** 两侧目标各 2 N、带宽 ±0.25 N；ready 要各侧 ≥1 N 且夹指速度 ≤0.002 原坐标单位/s 连续 20 ms。丢失该双侧载荷连续 10 ms 报接触失持；力保持不等同防滑。Panda 两指单轴目标速度各 0.025 m/s、帽各 4 N；mimic 丝杠 0.020 m/s、帽 8 N。获取超时 1.5 s，释放超时 1.2 s；release 目标为原 open，实际开度误差 ≤0.1 mm、速度 ≤0.002 m/s、双侧载荷 ≤1e-8 N 持续 20 ms 才完成。PD 保持 A1 原值：Panda arm/finger kp=500/4500、kd=30/18；mimic kp=(2700,6000)、kd=(95,35)，升降帽40 N。没有新参数搜索。

| 冻结案例 | 实际初态与物理预期 |
|---|---|
| P 球正控 | 经典 Panda，arm=π(0,−.25,0,−.75,0,.5,.25)，开度60 mm，闭合目标20 mm；r=20 mm、40.96 g 球，球心为初始两指面站点中点，有限台面恰在球底。h=1/1600 s。期望取得双侧力、真实上抬后保持、再张开卸载。 |
| M 箱正控 | 原 mimic_jaws，q=(0,0)，闭合目标(.018 m)；100 g、40 mm 箱心(.25,−.125,.25)m，台面top=.23 m。h=.001 s。期望同一控制器实际取得双侧力和抬持，再释放。 |
| M 零摩擦 | 与 M 同初态/控制，仅目标指面 μ=0。垂直平面指法向不能提供向上承重；即使双侧力成立也不得报告物理 holding。 |
| M 超载 | 与 M 同初态/控制，仅箱质量为1 kg、惯量按均匀箱重算。8 N 总闭合驱动在理想平衡时各侧最多4 N，μ=.65 的竖直容量≤5.2 N<9.81 N；期望不通过物理保持。 |

**场景和成功门。** 每例仍有另一个真实自由体和全部 pair，不能删除非 pad 连杆或台面通道。P 的第二箱为30×24×28 mm、60 g，中心相对主球(+.10,0,−.006)m；固定桌半尺寸(.14,.08,.02)m，中心在主球下方.04 m。M 的第二球 r=.015 m、30 g，中心(.33,−.125,.245)m，桌半尺寸(.12,.12,.01)m、中心(.25,−.125,.22)m。指面原四点±5 mm，材料 k=1200 N/m、d=1.2 Ns/m、μ=.65、滑速正则.04 m/s；物体–桌 k=2200、d=3、μ=.45；其他 k=6000、d=16、μ=.35；g=9.81。初始全部速度零，无预压。

contact_ready 后应用仅通过有界 posture 目标使实际夹具上移20 mm，最长 .6 s（P 使用现有多站点 IK 的位置/朝向目标、M 使用升降坐标；都不写状态）。实际物理 holding 必须离台 ≥10 mm、台面向上支撑 <0.1 mg、双侧≥1 N 且相对夹具的 COM 位移变化≤2 mm，持续100 ms；实际滑移>3 mm或持续失去双侧力即独立记录失败。随后发release，继续测卸载和有限台面再接触，末态不强称静止。两负控保留相同动作和门；相同目标、不同实际结果均归档。每例保存全部命令、实际 q/v、力和物体姿态，以及关键时刻的同锁 geometry；最多一次原冻结全窗，失败先定位不调参重跑。

**独立控制留出与首红。** 先真实 mimic 无预压首步：目标增量20 µm，实际 q 的惯性响应必须小于目标；重复读取及拒步重试不累加，下一接受步只再增20 µm。随后覆盖反向闭合坐标、单边/错误物体力、空夹、帽饱和、未来/倒退时间、不可写判定和实际丢载/释放。共同刚体变换只改变世界几何，不改变力阈值与闭合坐标判定。

**Newton 源码观察。** 本机1.7的 `example_controller_operational_space_hybrid_force_motion.py` 使用上帧 SensorContact 实测力并检查反力符号，经 Controller 和 SolverMuJoCo 推进；该文件名下实际装载 `fr3_franka_hand.urdf`，不能当经典 Panda 同模型验证。`example_robot_panda_hydro.py` 同样有 fr3 body 名，明确限 effort，并注释低刚度指驱动用于避免释放挤压储能。只借鉴“测量反馈/原生驱动限幅”的层次，不移植其增益、几何或将本次控制记为 Newton 动力学通过。

**窗口执行与判读。** 四个 CLI 均为 `python examples/robot_force_acquisition.py --case <panda-sphere|mimic-box|mimic-zero-friction|mimic-overload> --out <fresh.json>`。同源 pytest 节点为 `tests/test_gripper_acquisition_windows.py::test_frozen_acquisition_lift_and_release_windows[<case>]`；本批父仅执行四个 CLI 各一次，避免以 pytest 重跑相同物理窗。JSON 在正常退出前保留全部接受帧和首个物理/执行失败，已有输出拒绝覆盖。RAW0 需要八门同时为真；正例未真实 holding、负控未按预期失败、拒步/异常、未取得双侧力或未实际释放都不能伪装通过。negative 的 holding=false 只满足一门，不消除其他七门。

保持时间只累计力采样起点 `load_time_s` 已进入保持阶段的已接受区间，姿态/净空取同回执的步末 `time_s`；释放后再支撑同样要求载荷采样在 release 请求之后。所有记录保留这两个时刻，不把上一步载荷冒称当前瞬时力。获取后全程相对位移和保持窗口自身的相对位移分别保存与检查。

**执行前可证伪风险。** 现有切向律是滑速正则摩擦，没有静摩擦粘着历史。理想两侧各2 N、μ=.65、正则滑速.04 m/s 时，100 g 箱抵消重量所需相对竖直滑速约 `mg*.04/(2*.65*2)=.0151 m/s`；因此 M 正控仍可能在 .6 s 上抬期间越过原3 mm相对滑移门。此为执行前预测，不是实测，也不据此提高力目标、降低载荷或修改门。P 球可能通过几何包持承重，因此零摩擦反例仅用竖直平面夹箱。

**专项首错保留。** 缺模块首红 RAW1；首20µm实际响应/同step与拒步重试首绿 RAW0。随后仅更改feedforward的真实接受命令未被辨识，`DID NOT RAISE` RAW1；父补原始 requested_command 后控制器分别核对四项请求与解析收据，重力feedforward仍由原运行时解析。完整控制专项首次17通过/1失败（2.84 s）：25个后续步后两侧实测2.0002474897536207、1.9999109262340473 N，未满足测试误设的“初始精确2 N恒定”断言；此前真实20 ms ready门已过，抽离段尚未执行到。

该精确解析只给初始闭合轴 `δ=2/(4k)`、`τ=4 N`、`Mss=.5 kg` 的平衡参考，不是自由箱–有限台–双指的全系统静态解：初始台面无压、切速零，箱首步仍受重力，后续法向为实际 `kδ-d vn`，摩擦与台面也依实际姿态/速度重算。末两力的日志不能单独归因这2.47e-4 N差异的具体耦合或舍入来源。修正测试将原精确2 N oracle保留在第一回执 `load_time=0`，真实后续dwell查原预注册2±.25 N带和≤.002 m/s；没有按观测挑新容差，四窗口输入、门和原首失败均不变。抽离仍为原10 N、最多70个实际步；修正后该节点实际通过，见下方分组收据。

**实际交付与分组验证（2026-11-05）。** 测得产品源码为 `031638a`，父维护的原请求收据接缝为 `1ae0e44`；最终报告提交不改产品字节。直接入口是 `phymeo.control.gripper_acquisition.GripperAcquisition` / `AcquisitionProposal`。控制器只提交 RobotCommand；原 request 的四字段须与上次 proposal 相同，下一参考仍从 resolved command 锚定。`None` feedforward 的请求与原运行时解析出的重力补偿分别读回。

18 个 HOST 控制专项的来源为：[`a2-control-final-functional`](../../artifacts/p65-postnight-batch2-20261105-01/a2-control-final-functional/stdout.log) 一次17通过/1失败（2.84 s，RAW1），以及修正上述测试 oracle 后仅复验原失败节点的 [`a2-initial-force-oracle-repair`](../../artifacts/p65-postnight-batch2-20261105-01/a2-initial-force-oracle-repair/stdout.log) 1通过（1.48 s，RAW0）。没有把它写成另跑一次18项全绿。已覆盖20µm真实首步/惯性、同step与拒步retry、未来/倒退时间、零帽与anti-windup、反向闭合、真实单侧力/错误对象、初始解析力/后续dwell/10 N抽离失持、实际释放、非法参数8项、实际改feedforward拒绝、小行程空夹与10 ms真实超时。缺模块首红、feedforward首红及错误恒等力 oracle 的 RAW1 均保留；首20µm早期绿不再重复计数。

四个冻结 CLI 各实际执行一次，同源窗口 pytest 没有再重跑，也没有 skip/xfail 或改预期。每份 JSON 保存全部接受帧、原请求/解析命令、q/v、载荷、自由体姿态及关键同锁 geometry；载荷与步末各有时间戳。

| 窗口及原始收据 | 接受步数 / 物理时间 | 实际取得双侧力 / holding / release | 首个物理失败 | 原八门与退出码 |
|---|---:|---|---|---|
| [Panda 球](../../artifacts/p65-postnight-batch2-20261105-01/a2-panda-sphere-window/stdout.log) | 3008 / 1.880 s | 是 / 是 / 是 | 无 | 8/8，RAW0 |
| [mimic 箱正控](../../artifacts/p65-postnight-batch2-20261105-01/a2-mimic-box-window/stdout.log) | 1807 / 1.807 s | 是 / **否** / 是 | 1.048 s，相对滑移3.005689 mm；离台3.499783 mm，台面力0 | **7/8，RAW1**；仅 expected_holding_outcome=false |
| [mimic 零摩擦](../../artifacts/p65-postnight-batch2-20261105-01/a2-mimic-zero-friction-window/stdout.log) | 1681 / 1.681 s | 是 / 否 / 是 | .923 s，相对滑移3.012614 mm；台面向上力.980305 N | 8/8，RAW0；失持负控符合预期 |
| [mimic 超载](../../artifacts/p65-postnight-batch2-20261105-01/a2-mimic-overload-window/stdout.log) | 1789 / 1.789 s | 是 / 否 / 是 | 1.033 s，相对滑移3.010834 mm；台面向上力7.625533 N | 8/8，RAW0；失持负控符合预期 |

四窗均无执行异常；目标物释放后实际再受有限台面支撑，第二自由体也有真实支撑。最大自由体逐步线冲量闭合误差≤1.38778e-17 N·s，电机帽超额均为0。Panda 的20 mm目标由实际接受姿态求 IK，3次迭代收敛，残差3.31494e-12 m、9.15973e-13 rad；是否实际抬持仍由原10 mm离台/100 ms/双侧载荷/相对运动门判断，不能用IK残差替代。mimic 正控的实测滑移与预先指出的滑速正则风险方向一致；仅这些摘要不足以逐项隔离其法向、转动、台面和舍入贡献，不声称已经完成因果分解。

[首lint](../../artifacts/p65-postnight-batch2-20261105-01/a2-first-lint/stdout.log) 为1个 I001（例程 json import 次序），[首format](../../artifacts/p65-postnight-batch2-20261105-01/a2-first-format/stdout.log) 为4文件需格式化、1文件已格式化，原失败保留。父只改 import 次序并格式化；[qualified AST 核对](../../artifacts/p65-postnight-batch2-20261105-01/a2-style-ast/stdout.log) 中 controller、runtime 和两个测试的 raw AST 相同，example 只改变独立顶层 `json` import 次序，导入绑定与其余 AST 一致，并非宣称五份 raw AST 全同。[最终lint](../../artifacts/p65-postnight-batch2-20261105-01/a2-final-lint/stdout.log) 与 [最终format](../../artifacts/p65-postnight-batch2-20261105-01/a2-final-format/stdout.log) 都为 RAW0。格式后四窗基于同一份冻结源码运行。

**适用边界与余项。** 当前可交付的是受限驱动力、实际接受步反馈的获取/力接触保持/失持与释放协议，已从无预压初态在经典 Panda 球上完成实际抬持。普通 mimic 箱的“可取得双侧力”已实测，“原2 N协议下抬持”未达，不能用两负控 RAW0 或原A1预载结果遮盖。窗口仍是起始夹具已对准物体、单固定树/单世界/HOST；没有远距接近规划、CCD保证、自碰撞、柔性或本批机器人CUDA资格。共同刚体变换的新增物理窗口本批未执行。Newton 本批只有1.7控制例程源码核对，没有新跨引擎受载控制通过。后续若改变加载协议、夹具、摩擦机制或力目标，须另行独立冻结物理依据与正反例；本轮不通过重调原输入追求全绿。旧 A1 mimic 突加2 mm预载 RAW1 和本轮 mimic 获取后滑移 RAW1 是两个不同失败，均保留。
