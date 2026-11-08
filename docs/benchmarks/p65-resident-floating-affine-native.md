# C11 Newton 原生 FREE + affine 短参照

**CODE_READY / UNEXECUTED / AUTHORING_RELEASED。** 本交付恰两 NEW：
`tests/reference_newton/robot/resident_floating_affine_native.py` 与本页。
依据父 `881c28c` 计划 §8；未 import、运行、数值探测、collection、静态检查或格式化。
产品、原 fixtures、旧参考和父接缝没有修改。全部验证仍归父唯一串行队列。

## 固定三例

每个 device 固定 3 个 case、合计 9 个真实 native step，h 全为 1/1024 s。

| 身份 | 步数 | 原输入与首步独立锚 |
|---|---:|---|
| A01-motor | 1 | root/driver/follower 质量 2/1/2 kg；P=(1,-1)，c=(0,.004) m；独立 +1 N，外载/重力零。根/独立/从动 X 加速度 (1/14,5/14,-5/14)，原行 lambda=-4/7 N。 |
| A03-fan | 4 | 增加 gauge .5 kg，P=(1,-1,.5)，c=(0,.004,.002)；独立入速 .02 m/s，电机3/8 N；四个 COM 世界 X 力 (1/8,1/4,-3/8,1/2) N。首步根/三个 P 加速度 (20,58,-58,29)/133；原行 lambda=(-209/1064,-42/133) N。 |
| A05-native-contact | 4 | root/left/right 2/2/1 kg，两个 P offset ±7/64 m；球半径1/8 m；P=(1,-1)、c=0，电机/外载/重力零。原生 ke64、mu/kd/kf/ka0、margin/gap0。首 gap=-1/32 m、两侧力±2 N，根/独立/从动加速度 (2,-10,10)/7，lambda=-2/7 N。 |

A01 的 X 质量块为 [[5,1,2],[1,1,0],[2,0,2]]，约化块 [[5,-1],[-1,3]]。
A03 约化块 [[11/2,-3/4],[-3/4,25/8]]，总约化 X 载荷 (1/2,5/4)。
A05 原 X 块 [[5,2,1],[2,2,0],[1,0,1]]，约化块 [[5,1],[1,3]]，约化载荷 (0,-4)。
这些 literal 锚直接声明；独立 time-jet/KKT 和实际速度增量均逐项与其比较。
原惯量、COM、轴、offset、非零入速及各关节声明限位全部保存，不因运行结果调整。

显式完整 FREE + 全部 P 关节归于一个 articulation，再原生
`set_joint_mimic(follower,leader,coeffs=(offset,multiplier))`。A03 保直接 fan 的
两条原关系 (1,1,0)、(-.5,0,1)，不借 chain flatten 冒原图。
FREE 没有伪电机；root→P 显式过滤父子，FREE→world 不过滤。
A01/A03 是无碰撞的动力学短参照，未实例化原 URDF 的可视/无关碰撞几何；
不声称它们验收该 URDF 的完整碰撞资产。A05 保真实两个 sibling 球的 shape ID、
body owner、local pose/半径、过滤表与材料，要求两个球的 pair 未被排除。

## 原生时态与独立比较

同目录 `resident_floating_affine_reference.py`（RD final8c5107b）及其
`resident_floating_reference.py` 仅提供原声明、独立 homogeneous time jets、
完整 M/b/J 和原行未约化 KKT。显式路径载入这两个 NumPy/stdlib 文件；
不导入 Phymeo、TYX 或 pytest fixture。隔离作者树缺前者是预期，父集成后才可运行，
没有替代 fallback。独立轨迹从原初态自行推进，任何 reference 力或状态都不回写 Newton。

每步清空后仅写原声明 COM force/torque 与真实 driver `joint_f`；A05 输入均为零，
由 `CollisionPipeline.collide` 的真实入步行进入 Featherstone penalty contact。
armature、passive damping、target gains、limit springs、angular damping 均为零。
Newton 当前不支持 effort/velocity-limit 的相同动力学语义，本片只保存原额定值，
不把这三个无 PD/无触限短例冒成限幅或控制器对齐。

- 公开 FREE `joint_qd` 为 parent-anchor 轴下 COM 线速度在前，`body_qd` 为 COM/world。
  Featherstone 内部线速度为 parent-anchor 参考点；保存完整 public/internal 两个基变换。
  `joint_qdd` 经实际基变换并补 `omega × v_COM`，另与真实 Δv/h 比较。
- 内部 J 是 solve-origin spatial twist，移到各 body COM 后才与独立 J 比较；内部 H
  按同一速度基作合同变换。native `P` scratch 是 M×J，不是原 mimic 投影 S。
  原生 H_mimic/tau_mimic 保存完整 padded 矩阵/向量；独立重建 SᵀH S 和 Sᵀtau，
  inactive follower 单位对角/零 RHS 也显式检查。
- 步后 `body_f_ext` 已原位成为负 RNEA wrench。保存原值、同次 `body_q_com`、
  实际 `body_solve_origin`，按 F=-rawF、tauCOM=-rawTau-(COM-solve_origin)×F 解码，
  不套固定 world-origin 零。`body_parent_f` 可为未启用的 None，明确不是 mimic lambda。
- A05 每条原 contact row 独立恢复两侧 surface/margin/owner/法向/点速度、原生混合材料及
  两侧 COM wrench；仅作观察，绝不施回。`rigid_contact_force` 未由 Featherstone填充，
  保存该缓冲但不把它当 force sensor。原生各侧表面作用点与 Phy 的共同点语义分开；
  本组共线球法向使两者矩均零，不能外推一般接触力矩等价。
- 从独立完整 M/b/J、解码的原生全部 COM wrench、电机与实际 Δv/h 构造
  r=M*a+b-g-E*tau，再解原 C 的 lambda。完整根六余额、Sᵀr、C*a、Cᵀlambda、
  每行功/总功都保留；不拿 reduced load 或 BODY_PARENT_F 代替原行反力。

逐步保存实际线/角动量、总动能、入步力的端点离散功与隐式增量项。
能量平衡只用于这三个共线、零角运动、平移动能质量块不变的配置。
所有 body 角速度及根 quaternion 仍核完整向量；不称一般旋转积分等价。

## 门、保存与失败语义

预声明工程门不变：same-entry M/J/FK atol5e-6、rtol5e-5；短窗 q/v atol2e-4；
actual Δv/h、由此恢复的 full/row reaction atol3e-4、rtol5e-5（FP32 差分观察门）。
原 C11 产品方程/约束2e-5门不变。每个比较保存 actual/expected/atol/rtol/shape/max-error/result。
必要 model/state/solver 字段为 None 明确报错；可选未启用字段保 None 和实际语义，不冒通过。
本机 `begin_world` 的 gravity 包含末尾 global 行，来源核对后按完整两行保存/检查。

CLI：`--device cpu|cuda:0 --output ABS_NEW_DIR`。严格 Newton1.7.0/Warp1.17.0；
实际 device 必须精确匹配。新目录产生 `summary.json`、三个 `*-native.json.gz` 和
`source_snapshot/`。每 gzip 限64MiB压缩/256MiB解码，完整数据不裁剪。

每例保 declaration/model/role IDs/initial/final 及所有 trace。每行含入/出步时刻、
完整 entry/exit q/qd/body poses/twists/forces、原 control、原始 contact rows、
同次 solver 全 M/J/H/P/L、mimic reduction、内部速度/力/加速度、独立同态和独立全窗预期、
完整反力/动量/能量与各门。actual step return、submitted/completed/checked 计数分别记录。
模型读回或物理比较失败仍保存；异常保完整前缀/traceback，独立其他例仍尝试但全局 RAW1。
IO 或观察失败不伪称物理回滚，不设置末态补齐，不因漏 case 报全过。

保存 16 个 pinned Newton 源文件、driver、独立reference/base 的原 bytes/SHA；
运行末复核每源未变，并保存版本、实际 import 路径、cwd、CLI。
三例必须依原顺序、9步全部 completed 且所有门通过，才能返回0。

## 本机源码依据与当前边界

相对于 `.venv-newton-1.7.0/Lib/site-packages/newton/`：
`_src/sim/builder.py:5798`（P）、`:6021`（FREE）、`:6518`（mimic）；
`_src/sim/joint_mimic.py:26`（原生展开）；`_src/sim/articulation.py:1264/:1687`
（公开 full J/M）；`_src/solvers/featherstone/solver_featherstone.py:175/:545/:950/:1005/:1053/:1117`
（同 articulation、每步 mimic、原生约化/展开）；`kernels.py:1042/:1438`（FREE基/实际RNEA时态）；
`_src/solvers/semi_implicit/kernels_contact.py:391`（原生法则和两侧作用点）。
全部是源码读取依据，未以执行探测取得任何新结论。

作者仅完成两 NEW 和人工文本核对，未执行任何 native/项目/静态命令。
父后续首次真实运行、原始结果、格式和验收仍 **PENDING**；本报告不含通过数或性能结论。
