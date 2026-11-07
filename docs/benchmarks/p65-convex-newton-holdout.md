# Newton 1.7：三类凸接触、四个预声明静态姿态

状态：**CODE_READY_TESTS_PENDING / UNEXECUTED**。本提交仅作者文本编写；未 import、运行、编译、lint 或 format。CPU、CUDA 数值及静态资格由父的唯一串行执行槽完成后另记。原 C2 文档仍为作者提案，未冻结、未实施。

## 输入及独立期望

依照 [本阶段计划 §4.3](../superpowers/plans/2026-11-07-generic-collision-until-0800.md)，只新增 [独立脚本](../../tests/reference_newton/rr/convex_family_holdout.py)，不修改旧参考目录的任何 catalog、runner 或 oracle。坐标为 m，四姿态均为独立当前姿态采样；没有物理时间推进。A 在原点，B 在表内高度；n 恒指 A→B。

| 组 | 姿态 | 固定几何 / 位置 | 手推期望 |
|---|---|---|---|
| CC | CC-PARALLEL | 两有限轴半长 .5，方向均 (1,0,0)，半径 .05；B 中心 (0,0,.09) | gap=.09−.05−.05=−.01；n=(0,0,1)；最近轴点形成 x∈[−.5,.5] 的线 |
| CC | CC-SKEW | A 同上；B 轴方向 (3/5,4/5,0)，半长 .5、半径 .05、中心 (0,0,.09) | 唯一最近轴点在双方轴中心；gap=−.01；两皮肤点 (0,0,.05)、(0,0,.04)，n=(0,0,1) |
| BB | BB-FACE | 两盒半长 (.1,.1,.1)；B 中心 (0,0,.19) | gap=.19−.1−.1=−.01；相对方形面，n=(0,0,1) |
| HH | HH-PENTAGON | XY 环 (−.5,−.1),(.5,−.1),(.6,0),(.3,.1),(−.5,.1)，挤出 z=±.1；B 中心 (0,0,.19) | 真实非盒五棱柱；gap=−.01；相对五边形面，n=(0,0,1) |

HH 保留现有通用凸族的五顶点环，10 个顶点、16 个外向三角面；直接创建 newton.Mesh 并调用 add_shape_convex_hull，实际 shape_type 必须为 CONVEX_MESH。不经过 OBB 拟合、SDF 或普通三角 mesh 碰撞。CC 的真实原生胶囊局部轴是 +Z，通过实际 body XYZW 旋至表中方向；half_height=.5 只指中心柱段半长，不含两个端球。因此本组也真实检验非身份 body 旋转下的 offset 回绑。

## 首次执行前固定的工程门

| 量 | 绝对门 |
|---|---|
| gap 与 −.01 的误差 | 2e−6 m |
| 原生锚点的 body→world 重建、offset 与厚度关系、两皮肤点投影 gap 恒等式 | 2e−6 m |
| 双侧 authored 有限表面及预定支撑 patch 归属；斜交唯一皮肤点 | 2e−6 m |
| 法向与 (0,0,1) 的 L2 误差、单位法向模长误差 | 2e−5 |
| 实际 body 起点与 authored 位置、四元数差 | 2e−6 m / 2e−6 L2 |

Newton 的这些读数为 FP32；本例坐标最大约 .6 m，单次该尺度舍入约 1e−7 m。2 µm 为有限几次变换、接触锚点/厚度相减留出约数十次单精度舍入的工程门，法向 L2 门约 20 µrad；这是预注册接受限，不是对 GJK/MPR 最坏误差的数学保证。CPU、CUDA 使用完全相同的门，不随设备、接触数或实测误差调整。若原生算法超门，保留该差异及原始输出，不为通过更换参数。每项实际误差都会保存，不能仅用布尔绿灯替代。

## 观测合同及原始输出

调用链为 ModelBuilder / ShapeConfig(margin=0,gap=0,has_particle_collision=False)，随后 finalize(device)、CollisionPipeline(broad_phase="nxn",reduce_contacts=False,rigid_contact_max=256)、collide 和公开 eval_rigid_contact_kinematics。没有 solver.simulate，没有向物理状态灌另一引擎轨迹。

原生 normal 指 shape0→shape1；point0/1 是 body-local 锚点；offset0/1 也是 body-local，包含 effective_radius+margin 的厚度。以实际 state.body_q 重建：

- anchor_i = T_body(point_i)；offset_world_i = R_body(offset_i)；skin_i = anchor_i + offset_world_i。
- 原生 gap = (anchor_1−anchor_0)·normal−margin0−margin1。
- 本组零额外 margin 下，两物理皮肤点应满足 (skin_1−skin_0)·normal = gap。
- body 旋转不能再次应用 shape 旋转；碰撞写出的 point/offset 已属于 body frame。

JSON 保存实际 shape/body ID、原生类型/scale/local shape transform、真实 body pose/mass、各关键数组设备、完整 contact counters/count/capacity；保留所有有效 raw row 的原始顺序、双方 body-local point/offset、原始 normal、margin、gap、world anchor、point_id/triangle_id。同时另存统一 A→B 的 anchors/offsets/skins、逐项误差及检查结果。遇非有限数据保留带标签的 JSON 值并失败，禁止输出非法 JSON NaN。接触缓冲耗尽、错误 owner/type/device、无接触、几何门失败或执行异常均非零退出；几何失败仍先写完整 JSON。版本、设备或目标路径无效可在测量前直接拒绝。

Capsule 表面归属使用有界轴段投影及真实半径；BB 使用六个半空间；HH 使用原五边形边的独立平面和上下端面。上述 oracle 不从原生 hull/接触点反写。共面/平行最近集合不唯一，故不要求两引擎逐点同序、同数量；Newton 的接触 row 也不等于本项目面积权重。CC 原生源码预期平行最多两行、斜交一行，脚本记录实际数量但不将其当整线求积资格。旧基线只读 world anchor 的 helper 不足以验证胶囊皮肤，本脚本使用完整 offsets。

本轮不读取未由求解器填充的 rigid_contact_force，不宣称总力 1 N、端点权重 1/2、面角权重 1/4、动态反作用、CCD 或材料持久身份已对齐。原生 point_id / triangle_id 仅原样记录，不冒充 Phymeo 材料顶点/权重。父可将这四个独立原生读数与相同 authored 输入的 Phymeo 结果并排，不改变原生行。

## 父串行执行入口

环境已有于父目录 .venv-newton-1.7.0/Scripts/python.exe。只读元数据已确认 Newton 1.7.0、Warp 1.17.0、NumPy 2.5.4；无需安装。使用原 guard/receipt、明确工作目录和新的输出路径，两个 CLI 必须串行：

~~~powershell
& 'C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0/Scripts/python.exe' tests/reference_newton/rr/convex_family_holdout.py --device cpu --out '<existing-fresh-evidence-dir>/convex-newton-cpu.json'
& 'C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0/Scripts/python.exe' tests/reference_newton/rr/convex_family_holdout.py --device cuda:0 --out '<existing-fresh-evidence-dir>/convex-newton-cuda.json'
~~~

工作目录是本脚本所在 checkout 根；证据目录由父先创建。--out 必填且不得覆盖；没有容差、输入或设备回退参数。每端只运行上述四个姿态。返回 0 仅代表本端四姿态全部几何检查通过；执行异常独立保存在 execution_error，不混同为几何差异。便于父复用的 Python 入口为 frozen_cases() 和 observe_pose(case, *, device)，后者要求已初始化 Warp 并显式传入已选设备；脚本不依赖 phymeo import。

## 官方版本依据与当前边界

源码静读依据均为父已安装 newton 1.7.0：

- _src/sim/builder.py：add_shape_capsule（8248 起，Z 轴/柱段半长）；add_shape_convex_hull（8469 起，CONVEX_MESH）。
- _src/sim/collide.py：CollisionPipeline（1271 起）；_src/sim/contact_kinematics.py：公开距离/锚点读数（35 起）。
- _src/sim/contacts.py：body-local points/offsets、A→B normal、surface thickness（273–302）。
- _src/geometry/collision_primitive.py：有限 capsule–capsule（181 起）；_src/geometry/types.py：Mesh 顶点/外向三角面合同（133 起）。

官方文档：[Newton 1.7 碰撞管线](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)。本次未联网或执行探针；版本合同以本机已安装官方源码为准。

当前只有源码文本交付，CPU/CUDA 数值、原始退出码及 Ruff/format 均待父运行。旧参考结果不重跑，C2 不因本脚本而获得实施授权。

## 父串行结果（2026-11-07）

父e9439f5原样导入815e44b两NEW；63Ruff0、64format-check0，65CPU/66cuda:0各四姿态全部通过，raw0。原生行数均为2/1/4/3；皮肤归属最大误差约4.00842e-8m。68源码外wheel按同 authored 输入独立运行Phymeo斑块，四姿态gap/法向/材料重建通过，和原生gap最大差1.69874e-8m。原误差门、原输入及raw输出未修改；没有动力学或面积权重等价声明。

证据 artifacts/p65-generic-collision-20261107-01/65-newton-holdout-cpu、66-newton-holdout-cuda、68-outside-wheel-integration 和 outside-integration.json。上方UNEXECUTED为作者历史，当前状态 **HOST/CUDA_STATIC_HOLDOUT_VERIFIED**。
