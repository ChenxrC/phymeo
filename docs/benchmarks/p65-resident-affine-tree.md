# C5 固定根 affine-mimic 驻留：T1 冻结测试库

状态：**TESTS_AUTHORING_ONLY / CODE_READY_TESTS_PENDING / UNEXECUTED**。本报告是四个 NEW 文件之一，不授权产品实现。父已在 `3c0f793` 冻结公共 K/N schema，在 `beb3e3e` 记录组合 storage 方案；作者未 merge 父、未修改原计划/C4/旧共享源码。

2026-11-07 父通报：C4 已以228完成源码外wheel，4world×40步加restore后21重放、164worldsteps，415安装文件/275实际import模块同wheel；这是 C4 历史验收，**不是 C5 通过**。

## 1. 文件与有效首 RED

仅本次四 NEW：

- `tests/scale/test_resident_affine_step.py`：完整力学、独立KKT、PD、图/状态/历史及混合事务。
- `tests/scale/test_resident_affine_collision.py`：真实accepted full FK→有限材料bounds、原source/group、多world与资源事务。
- `tests/reference_newton/robot/resident_affine_reference.py`：自含固定模型/完整逐部件oracle与pinned Newton 1.7.0/Warp1.17参照，无Phymeo导入；不import旧测试或旧参考脚本。
- 当前报告 `docs/benchmarks/p65-resident-affine-tree.md`。

首checkpoint `c10938a93be15c9b31b8ca5130ab5a4c275c4a30` 包含前一测试与其自含参考模块。父正常导入 `3ab6296` 后，**230有效RED：1 failed / 0.87s / RAW1**，原 `load_mimic_urdf`、N2/K1、P=(1,2)、c=(0,1/8)、full_state 前置全部通过，准确停在缺 `ph.AffineRobotTree`。不是collection/环境/fixture失败；父已保存原输入gzip。本报告只转录父实际通报，不另扫证据；具体原日志路径由父证据登记绑定。该节点不重跑 RED、不改物理输入与断言。

精确节点：

```text
 tests/scale/test_resident_affine_step.py::test_a01_public_affine_tree_keeps_follower_inertia_and_real_reaction
```

作者未执行import、collection、pytest、数值脚本、lint、format、compile、native、build/install或CI。数学常量为预先手推，代码本身仍待父首次执行；小文本检查/文件IO/Git不等同测试验收。

## 2. 已固定公共字段与行为

`ph.AffineRobotTree(asset, RobotState(K), kp, kd, effort_limit, delay_steps=0|1, control_max_iterations=1..4096)` 与 `world.inputs(target, target_velocity=None, feedforward=None, loads=None)`；loads 是 **N行** `RobotInputs`，目标/rate/ff是K。默认ff为零，不隐含HOST场景的bias补偿。旧 `TyxWorldBatch`/step/snapshot/restore/tree_kinematics/tree_diagnostics/prepare_collision 保正常路径，不造新runtime。

checkpoint：`position/velocity/applied_effort/generalized_applied` 为K；`full_position/full_velocity/full_applied_effort/full_generalized_applied` 为N；control是两份owned不可变tuple `(drive[K,3], loads[N,7])`，loads列为generalized、Fxyz、Txyz；delay0 history为空，delay1为K×3。history只消耗接受的物理步，外载当前步生效。

kinematics：position/velocity为N，independent_position/velocity为K，**joint_names为K独立名称，full_joint_names为N物理名称**；link_position/link_rotation/COM均为真实N组末态。固定工具authored site用不变局部变换读回，主轴变换不得重复。原图及完整FP64物理声明参与token，等运动的chain与flatten图不能混用checkpoint/binding。

diagnostic：entry_position/velocity为K，full_entry_position/velocity为N；mass/bias/external/linear_jacobian/angular_jacobian是full，另有reduced_mass/bias/external/reduced_linear_jacobian/reduced_angular_jacobian；drive/acceleration K、full_acceleration N、constraint_effort_full N、原行row_multipliers/row_powers_w及constraint_power_w。全部owned，时间是 latest accepted attempt 的entry_tick/exit_tick/dt_s。初始化/restore/最新拒步或driver异常之后None；失败scratch不冒最后成功诊断。

测试只沿旧batch `_upload/_read_status/_sync` 注入异常，不猜新storage私有字段。父设计的 N物理bank + K独立bank 组合符合此公有合同；普通finalize后还须在同attempt内完成K接受/拷回，sync/status成功前不能swap/发布。父后续通报0ce0050已提取compile_tree_physical与stage_mechanics/stage_candidate_geometry/stage_finalize，受影响interpret22/CUDA22通过；该共享提取结果不作为尚未实现C5产品的通过证据。

## 3. 固定数量、步长与验收门

下列是由参数表声明的**计划节点数，不是collection实测或通过数**：step 54（13 HOST输入/静态负控、41 TYX物理/事务），collision 44（全部TYX）；共98，每个真实后端各运行85个TYX节点，HOST13只需一次。父collection若数量不符先定位参数/接口，不能用此手数冒计量。

| 组 | 固定输入/子例 | 节点数与持续窗口 |
| --- | --- | --- |
| A01–A05 exact | A01首入口；A01F/A02/A03/A04/A05/A05G/A05T完整KKT及原解析值 | 1+7单步 |
| A01–A07 motion | A01/A02/A03/A04/A05/A05G/A05T/N9/N16；不重置/不灌状态，实际full及independent读回 | 9×40=360真实物理步/后端 |
| 安装/坐标/原图 | A04/A05G×cycle/large；A02 XML重排；A03链与显式flatten的状态一致、原行反力不同 | 4+1+1 |
| A06 | cap1/16；宽帽小ff/大项相消；delay0/1；一sweep真实残差 vs512 | 2+2+2+1 |
| A07 输入/事务 | 8种非法HOST图；5静态设备域；位置交集两侧；partial upload/status/restore sync×两bank；图/full/history错误restore×两bank；owned回执 | 8+5+2+6+2+1 |
| A08 完整材料 | A04 s=(-256π,-2π,0,.37,2π,256π)×base/cycle/large；near overlap/clear miss；有限capsule轴/端皮肤/端外；A03/A05T实际source | 18+2+3+2，source各3步 |
| A08 身份/规模 | 六种graph/model/revision/IDs/bodyalias/exclusion负控；W1/4/32完整输出40步及各2预热+40暖态；epoch/owner/reset拒步×两bank | 6+3+3+2 |
| A08 资源/域端 | capacity+query failure；两plan close异常；empty；A01 full q=±4096真bounds及向外候选拒步 | 1+1+1+2 |

除明确测试外 **h=1/1024 s**；A08 mixed reject 的独立检查步 h=1/8，从预声明可行 restored s=7/2,u=8（full q=(7/2,57/8),v=(8,16)）出发，必须只拒该affine world，旁观FreeSphere接受。该例是restore/domain测试，不能计作正常轨迹无重置的窗口。

- 单步状态、M/b/J、载荷/反力/FK与独立FP64 oracle：`rtol=5e-5, atol=5e-6`，各原字段物理单位；40步累计位置/速度 `atol=2e-4, rtol=5e-5`。
- full affine关系和原行位置/速度、零约束功：绝对门 `2e-5` 并沿上述相对门；非零offset检查 `C q = original_offsets`，不是Cq=0。FP32编译不要求bit-exact CP=0。
- full/reduced质量方程保原 `2e-5` 归一门；PD逐坐标完整原clipped expected门保 `5e-5*max(1,abs(tau),abs(expected))`，不借inactive cap或相消前项放宽。
- 独立手推常量与FP64 KKT一致性门 `atol=2e-13, rtol=0`，仅验证独立oracle自身，不拿产品值反写真值。
- native same-entry/final FK沿5e-6/5e-5，native窗口2e-4；以native Δv/h派生的反力另 `atol=3e-4`，标明差分放大FP32舍入，不冒原生反力传感器。
- bounds须真实包住原材料的独立FP64当前边界，不能用宽相输出构造预期。near overlap gap=-1/65536 m，清楚miss gap=1/4 m；细长capsule轴长16m、皮肤1/1024m，端外偏移1/4m。使用C4已接受工程外扩，允许未列精确候选门的保守额外候选，不因误差反调padding/fixture。
- large共同安装为xyz循环旋转后平移(4096,-2048,1024)m；cycle普通平移(.5,-.25,.75)m。多周子例源界±4096。另用A01 binary-exact s=(±4096−1/8)/2、u=±1保持full域端原点可验，向外真实步拒绝。

A06有限预算固定：Mr=[[161/64,-1],[-1,5]]，kp0/kd1024，rate=(1/1024,1/1024)，cap16，移除A05额外effort；原零effort起始、坐标顺序0→1。Q=[[1061,64],[64,902]]/741，首sweep tau=(741/1061,741*997/(1061*902))，第一行残差64*tau1/741>0.06，原门约5e-5；512预算解为(838*741,997*741)/(1061*902−4096)。父已独立文本核对，无probe选例。

## 4. 真值、完整保存与规模含义

参考模块只复用既有独立参考的文本算法：authored齐次变换的一阶/二阶导数得到每个原物理部件COM/J/偏置加速度，逐体累加完整M/b；不是产品FK/J/M导出。每个新机制P/c明确预声明，原行C由原关系描述；完整KKT `[M,-Cᵀ;C,0]` 作为主受力oracle，投影为交叉核。fixed tool保原质量部件参与oracle，产品folded physical group的COM/J用独立质量加权合成；source/authored pose单独读回。

N9/K3、N16/K4沿计划固定平行P配方，真实全部物理行参与质量；A05的R/P分枝负责非对角跨独立列。A02 XML重排按名字映射输入/输出；A03与flatten原图反力分别(5/2,-1)和(2,-1)，不是用同运动掩原始row receipt。

每个持续物理测试gzip完整输入、h、entry/actual checkpoint（包括两块held、queue、N/K状态/last loads）、完整diagnostic、实际FK/summary核验、独立轨迹q/v、接受/tick、全部stats。异常/窗口用finally保存已发生trace。实际加载URDF保留在相同pytest tmp目录；不会覆盖旧证据。

W1=affine A04；W4=FreeSphere/A01/A03/A05；W32重复8类：FreeSphere、Slider、oneR、ordinary2R、A02、A03、N9、N16。同ID不同world实际注册，world顺序逆序，有限sphere/capsule/box分别运动。完整矩阵每步保存所有world真实checkpoint、每个affine全diagnostic/FK、全部bounds/candidates/registered pairs/epoch/ticks/source IDs及两种stats。动作只保持原物理命令，没有HOST更新姿态。

暖态先真实跑两个buffer后保存一次真实入态，后40步禁止snapshot/kinematics/diagnostic、HOST dynamic envelope入口和新device allocation；不为记录q/v偷读状态。暖态保存真实bounds/ticks/candidates/acceptance和完整计数，退出后读真实末态核材料包含。初/末实际状态与中途缺少q/v读回明确区分；必要status/BVH metadata读回照实计数，不声称零传输。完整窗口另提供逐步实际q/v证据。readonly输出不得通过setflags重新开启写入。

## 5. pinned Newton 与父未来执行入口

新native CLI使用本机Newton1.7.0/Warp1.17.0，要求真实device为cpu或cuda:0，无fallback。五个独立40步窗口为A01、A02、A04、A05G、**明确flatten后的A03**，每端200实际步；原始A03另真实import必须记录mimic-chain拒绝，不计动力学通过。保fixed joints，explicit inertial，关闭armature/passive damping/limit springs/drive，接触None，实际Featherstone推进，不喂Phy轨迹。

保存每次真实native full mass、原模型metadata/mimic coeffs/形状/body/joint mapping、body_q/body_qd/body_f、joint_q/qd/control_joint_f、当前KKT与独立窗口、派生反力、checks、异常和版本/source bytes+SHA。body_qd与body_f均线在前/角在后，参考点COM，世界表达。源链和flatten URDF分别保存，原链警告/拒绝不能变成通过。未知引用被Newton跳过的路径须由metadata断言挡住。native内置PD/effort/velocity limit不冒本片的协议或能力域对照。

父按唯一串行guard/保存raw的正常入口可分段执行；作者没有执行这些命令：

```text
python -m pytest tests/scale/test_resident_affine_step.py -m "not tyx"
python -m pytest tests/scale/test_resident_affine_step.py tests/scale/test_resident_affine_collision.py -m tyx --tyx-mode interpret --require-tyx
python -m pytest tests/scale/test_resident_affine_step.py tests/scale/test_resident_affine_collision.py -m tyx --tyx-mode cuda --require-tyx
<existing-newton-python> tests/reference_newton/robot/resident_affine_reference.py --device cpu --output-dir <fresh-cpu-dir>
<existing-newton-python> tests/reference_newton/robot/resident_affine_reference.py --device cuda:0 --output-dir <fresh-cuda-dir>
```

这里的python指父已接受主环境，native指既有`.venv-newton-1.7.0`；无新安装。正式首次执行在父冻结产品/测试commit后；当前仅230有效缺口RED实测，其余均UNEXECUTED。产品就绪后再授权独立outside-wheel consumer；T1不创建该第五文件。需要证明所有Phymeo导入来自新wheel，原C4 wheel不替代C5。

本片仍不含浮根6D、几何闭环、主动follower、物理限位冲量、设备通用contact force/摩擦历史或CCD。完整当前姿态宽相可用性与真实接触响应分开；不存在mimic抓取/旧100g失持被本测试修复的声明。
