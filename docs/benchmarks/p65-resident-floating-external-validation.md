# C9 外部验证作者交接：原生 Newton 与安装后消费

状态：**AUTHORING_RELEASED / UNEXECUTED**。仅三个 NEW 文件，不修改 C9 产品、共享源、旧报告、测试或输入门。权威合同为 `docs/superpowers/plans/2026-11-08-resident-floating-tree.md`（含 public export 与 local_origin_m 后续裁定）。本作者未运行 import、collection、数值脚本、pytest、lint、格式化、编译、安装、构建或 native 程序；以下是第一次运行前的案例与失败口径，不是通过结果。

父报告 C9 首公开 P03 interpret 445/CUDA 443 通过；本交接不将其外推为此新 native/wheel 消费者已通过。所有实际执行、源码绑定和首失败保留由父唯一串行队列负责。

## 精确清单与运行入口

| NEW 文件 | 作者提交 | 角色 |
| --- | --- | --- |
| `tests/reference_newton/robot/resident_floating_native.py` | `26413359e686cabb5ed80df4f8087e6c03648747` | 独立 Newton 1.7.0/ Warp 1.17.0，7×40 步/端 |
| `tests/scale/resident_floating_wheel_consumer.py` | `38d2a9cbe7e9cd68ba6cba023bc82eded6de1153` + `8f53a6164be37eee37f7db5a5df59cb0c2b5f9f2` | 仅公开 Phymeo API，W4 完整消费及安装来源 |
| 本报告 | 最终交接提交 | 预声明输入、观察语义、门与限域 |

本机已有参考解释器为父 `.venv-newton-1.7.0/Scripts/python.exe`。不新增依赖，不改该环境，不在脚本中安装内容。native CLI 为 `--device cpu|cuda|cuda:0 --output ABS_NEW_DIRECTORY`；`cuda` 明确解析到 `cuda:0`，JSON 保存实际数组 device。

wheel 消费者复制到 checkout 外，以现有主解释器 `python -I -B` 运行，CLI 为 `--mode interpret|cuda --output ABS_NEW_DIRECTORY --wheel-site ABS --expected-package-root ABS --expected-wheel ABS [--forbidden-checkout ABS]`。脚本要求 cwd 等于其复制位置、输出目录为新目录。不读 repo tests/helpers，不导入私有 backend，不寻找 checkout、不调用任何私有 advance；父控制 actual wheel 的安装与源外位置。

## 预先冻结的 native 窗口

全部 h=1/1024 s、40 步，root 质量 2 kg；存在 child 时其质量 1 kg。无 contact、无 PD gain、无 armature/damping/limit 弹簧，标量驱动按原 motor 加原外部 effort；重力只由原模型施加。每例单独真实 builder、FREE root 与独立 state/candidate，输出后交换原生 state，绝不注入 Phymeo 轨迹。

| 固定 ID | 原模型与输入 | 独立首步锚/窗口检查 |
| --- | --- | --- |
| P01_freefall | P_x child；Iroot=.5I、Ichild=.25I；qj=1/8；g=(0,0,-2)，零速度/载荷 | aCOMroot=(0,0,-2)，根角/相对加速度零；40 步离散 COM ballistic |
| P02_origin_force | root-only；c=(1/8,0,0)，I=(1/4,3/8,1/2)；根 authored 原点施世界 F=(0,2,0) | 首 aCOMy=1、alpha_z=−1/2；每次从自己的实际姿态计算 tauCOM=−R*c×F，根原点/COM 分开读 |
| P02_principal_torque | root-only，c=0，Ibody=[[3/8,−1/8,0],[−1/8,3/8,0],[0,0,1/2]]；COM 矩=(1,0,0) | 首 alpha=(3,1,0)，完整 tensor 与旋转后 M/陀螺项 |
| P03_internal | 同心 P_x，Iroot=.5I、Ichild=.25I；内部 +3 N | M(x,j)=[[3,1],[1,1]]；root ax=−1.5、相对 aj=4.5、child 世界 ax=3；40 步净线动量与真实反冲 |
| P03_child_force | 与上一例同模型，motor=0，child COM 世界 F=(3,0,0) | root ax=0、aj=3；与内部驱动的不同根响应 |
| P04_internal | 同心 R_z，Iroot=.5I、Ichild=.25I；内部矩 +1 Nm | 根 alpha_z=−2，相对 alpha_j=6，child 世界 alpha_z=4；40 步独立角速率/标量角漂移 |
| P04_eccentric | 采用 RD `fabc63e552c8f5db0e2e97b95addbb1ac105682e` 原全部输入，见下段 | 非零入速、偏心与非主轴，同次全 7×7 M/J/b 及 root/child 真实外载，40 步自身状态 |

P04_eccentric 的 root COM=(1/16,−1/32,1/64)、主 I=(1/2,5/8,3/4)、主轴 XYZW=(0,0,3/5,4/5)；child R_z、offset0、joint-origin frame I、COM=(1/8,0,0)、I=.25I。初根 origin0、XYZW=(.5,.5,.5,.5)、qj=−1/16；v=[(1/16,−1/32,1/64),(1/32,−1/64,1/128),−1/64]。g=(0,0,−1/4)，motor=1/8、额外 joint effort=−1/32，child COM F=(1/8,−1/16,1/32)，root COM torque=(−1/32,1/64,0)，其余零。native FREE parent anchor 设 xyz=(1/2,−1/4,3/4)、XYZW=(.5,.5,.5,.5)，builder 的 FREE q 保持上述物理 root 初态；此非恒等坐标基留出不改变物理模型。

独立核仅使用声明模型和 native 自身入态的根世界姿态/标量 q/v：解析构造一孩子真实 COM Jacobian、质量和零广义加速度的向心/Coriolis/陀螺 bias，再求完整 D 方程。没有 Phymeo/TYX 导入、没有 finite difference、没有读产品 M 作期望。P01/P03/P04 同心的首步 literal 和整窗闭式检查另立，不能仅靠同次求解互证。偏心转动不宣称离散角动量/动能严格守恒；完整动量、能量和外载仍保存。

## pin 源码观察口径

来源目录是实际安装 `newton/_src`；每次运行将以下完整原文件文本、绝对路径与 SHA256 保存到 JSON：sim/builder.py、model.py、state.py、articulation.py，以及 solvers/featherstone/kernels.py、solver_featherstone.py。模型完整质量/COM/tensor、joint 类型/父子/两侧 frame/轴/起始索引/控制与限制参数、shape 表、filter-pair 数组全部保存。没有把空 shape 表冒接触管线验证。

- `builder.py:add_joint_free`（约6021）按 parent/child anchors 初始化 7 维位置，保持 authored body pose。`articulation.py:eval_joint_child_state`（约344–382）将 public FREE qd 的 parent-anchor 基 COM 速度旋到世界，再作原点/COM 运输；`body_qd` 是线速度在前、角速度在后的 COM/world twist。
- `kernels.py:convert_free_distance_joint_qd_public_to_internal`（约1042）将 COM 速度减 `omega×(COM−parent_anchor)`，得到内部 anchor twist。参考显式保存 raw public/internal qd。设 A 为 parent-anchor→world 旋转，S=diag(Aᵀ,Aᵀ,1)，则 `v_native=S*v_worldCOM`、`M_worldCOM=Sᵀ*M_native*S`、`J_worldCOM=J_native*S`，不能直接比较两种 basis 的 M。
- `jcalc_integrate`（约646–680）对真正根使用 `aCOM=A*a_internal + alpha_world×(COM−anchor) + omega_world×vCOM`，alpha_world=A*alpha_internal。参考保存原 `joint_qdd` 和该完整解码，不能把内部加速度当 COM 加速度。
- `eval_rigid_tau`（约1494–1503）原位将 articulated `body_f_ext` 改为 `−[F, tauCOM+(body_q_com.xyz−body_solve_origin)×F]`。每步保存 solver 的真实 body_q_com 与 body_solve_origin、入步原始 COM 载荷、post-step raw 负 wrench，并完整逆解码。浮根 solve origin 不假定为世界零，也不只反号。
- native 根采用归一化 Euler quaternion 更新（同 `jcalc_integrate`）。脚本分别保存/核 native 更新，并另存同 omega 的 exp-map 候选供差异观察；不把 Phymeo 的 exp-map 数值当 Newton 原生位置真值。

## 原工程门与完整保存

沿 C9 预声明同态 atol=5e-6、rtol=5e-5；M atol=2e-5；40 步独立窗口绝对门2e-4。零初速 literal 在自写解析矩阵侧额外用1e-12检查算式，在 native 侧仍用原工程门。每项保存实际误差/门/shape；不依据结果调 tol、h、模型或输入。

native 对 check failure 继续保存该原始 40 步及其余已声明例，不将失败窗口当通过。执行异常按 case 保存完整已完成前缀/traceback，并继续其他可执行独立例；总退出非零。完整原生 state、joint、body、M/J、solver 中间读数、独立 terms、输入与 decoder 均在 `native-floating.trace.json.gz`；summary 仅索引，不能替代轨迹。

## 公开 wheel 消费矩阵

W4 顺序固定 `(1207,6,91,503)`：普通 P、旧 FreeSphere、P02 root-only、P03 root+P。P03 原模型/内部3N和 P02 原模型/原点2N与上表完全相同，delay=0；普通 P 质量2kg、I=(1/8,1/8,1/8)、world安装(1,0,0)、轴+y、q0=1/8、v0=1/16、COM受(0,2,0)N，kp=kd0/cap8。FreeSphere 质量2kg、半径1/16、初(0,0,2)、v=(1/16,0,0)、g=(0,0,−2)。没有任何接触力加入此窗口。

P02 外力矩由自身本次接受姿态和固定 authored 原点运输，输入经过正常 `spec.inputs` 与 `batch.set_controls`。独立 FP64 exp-map/半隐式公式从固定初态推进，另核实际同次完整 M/b/J/drive/acceleration、D/Q、COM/root 点、summary 和 full held controls。40步均保存原 packet、入/出完整 checkpoint（含 local_origin_m）、kinematics、diagnostic、所有 bounds/candidates/ticks/epoch/counters。

浮根几何为 root 偏心球、有限 capsule（P03 真 child、P02 root）、root 非等边 hull，原 IDs/radius/revision 与点表独立声明；普通 P 与 FreeSphere 也经各自公开 binding 进入同一查询。每 world 注册明确有限静态近球与远球；默认静态候选域要求每个动态 shape pair（含同 body）显式 exclusion，全部按该合同登记。独立原材料点从公开 q/quat 解析运输，核每个真实 finite envelope 全包含、同world候选不漏、远球保持分离。允许工程 padding 多候选，不声称 CCD 或接触响应。P07 使用仅一浮根球+有限静态球、完整合法材料映射单独验证 `prepare_contacts` 的 fixed-root 拒域，避免空材料/旧 FreeSphere 拒绝伪装该门。

tick19 的全 W4 snapshot 恢复后重放完整21步，核每次完整 input/entry/state/FK/diagnostic/summary 及原材料 bounds/candidates 的规范 JSON 字节相等。epoch与计数单列保持单调，不将其伪装可回卷物理状态；旧 query 应因 restore epoch 明确失效。最初 owned 观察始终不变，数组无法重新打开 writeable。

两个真实 `current_buffer` 分别测试：非法 root local4097+同步 world 表示拒绝、混合 valid scalar/错误 n 的 floating packet 先验失败且全场不变；再合法恢复 root local4096/向外速度1，使下一真实候选逐world STATE_DOMAIN 拒步，其余三 world 正常推进。保全部 root/joint/history/applied/FK/energy；随后明确改为向内速度−1 的合法恢复进行 retry，验证再次可接受及真实 local<4096。此为独立边界请求，不暗中修补原40步或执行自动减步。共40+21+2×2=65 batch attempts，两个边界拒绝各保留一 world；此计数是预声明，不是已执行统计。

结束验证 plan 与 batch 关闭、重复close和关闭后拒绝，实际 device_owned_bytes 为0。消费者沿已有 wheel 规范保存精确归档 file URL/SHA、全部安装成员逐字节 parity、所有 loaded phymeo 模块路径/bytes；installer 可选 hash 缺失不冒安装失败。绝对安装/source路径仅为 provenance，不参与物理字节比较。

两脚本均限制完整 JSON 未压缩不超过256MiB，写临时 gzip 后原子改名，并保 SHA/实际大小与 summary。IO失败不能伪称已推进物理回滚或部分成功。当前全部数据仍待父首次串行运行；没有制作近似演示图，没有复跑旧 C8/C4/C5 消费者，未新增运行时或接触能力。

父执行前文本审阅发现默认 static-only 域的同 body 动态 pair 也必须显式排除；8f53a61 按此只修登记，全部原输入/物理门不变。此为作者 fixture 先行修正，未将其写为已执行失败或通过。
