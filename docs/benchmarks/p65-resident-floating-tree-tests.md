# C9 普通浮根驻留树：独立测试与参考交接

状态：**TEST/REFERENCE AUTHORING_RELEASED；新增完整矩阵 UNEXECUTED**。作者仅编写本批四个 NEW 文件，没有运行 import、collection、数值、pytest、静态、构建或 native 程序，没有合并父源码。权威范围是父 [C9 冻结计划](../superpowers/plans/2026-11-08-resident-floating-tree.md) 及其 `local_origin_m`、顶层公开导出和统计名裁定。产品、唯一串行执行、原生 Newton 和源码外消费另由父调度。

## 已发生的首节点证据

首测试 `08179d7` 建立旧 HOST 完整 7×7 质量矩阵和内部驱动力 3 N 的独立解析前提，`839c02a` 只按原实际导出归属改为 `ph.FloatingRobotTree`。父434有效 RED：1 failed / 0.85 s / RAW1，前置完整矩阵与根/关节加速度 `(-1.5,4.5)` 均成立，准确停于新公开 API 缺失；不重复该 RED。

父442后续设备初始化遇到原 `FloatingRobotInputs` 对 FP32 存储的拒绝，父只将三个已有载荷块交接为 FP64。父444完成了设备 M/a/v/位置/几何前置验证，随后旧测试把 HOST 的 `1e-12` 错用于设备净动量观察，实测残差为 `-2.3283064365386963e-10 kg m/s`。按父裁定，`364ad9b` 仅将该设备断言改成净动量除总质量 3 kg 后的 COM 速度残差，以原 `ATOL=5e-6 m/s, rtol=0` 核对；HOST 的 `1e-12` 与手推恒等、物理输入、积分/迭代门全部不变。原失败留存，这不是求解器物理修复。

本报告没有把父444的部分先验通过写成完整 C9 通过。后续首节点 GREEN 与完整矩阵由父保存真实收据；这里不预填实际数量、耗时或速度。

## 四文件与接口

- `tests/scale/test_resident_floating_step.py`：公开首节点、独立力学/完整轨迹、混合世界和事务。
- `tests/scale/test_resident_floating_collision.py`：实际接受态有限 S/C/H 材料包围、来源/过滤和宽相事务。
- `tests/reference_newton/robot/resident_floating_reference.py`：仅 NumPy 的独立齐次变换二阶 time-jet 参考；不导入 Phymeo、TYX 或 Newton，不冒充原生 Newton solver 对照。
- `docs/benchmarks/p65-resident-floating-tree-tests.md`：本报告。

测试从顶层 `phymeo` 调用 `FloatingRobotTree`、`TyxWorldBatch`、`ResidentFloatingTreeBinding`，驱动经 `spec.inputs(drive, external)` 进入原批运行时。完整维数分别为 n 个 motor、B=n+1 个实际 body、D=n+6 个速度、Q=n+7 个配置。root v 为 COM 世界速度与世界角速度；root q 为 authored body origin 和 XYZW。归一化四元数符号连续，不能用强制 w 正代替。

完整 checkpoint 保 held/queue、full generalized loads、原 topology/tick 与真实 `local_origin_m`。`position[:3]` 是 anchor+local 的 FP64 世界观测，不能在大 anchor 下反演真实 local。几何选择只消费接受 bank；浮根接触反馈仍明确 unsupported，不用宽相候选冒实际力。

## 独立案例与输入

统一 h=1/1024 s；same-entry 状态 `atol=5e-6, rtol=5e-5`，40 步累积 `atol=2e-4`，M 的绝对门 `2e-5`，full 方程残差 `2e-5`。原有限预算 PD 门不修改。配置、身份、restore/replay 完整字段要求精确相同；工程转动轨迹不声称离散能量严格守恒。

| 族 | 冻结物理与独立预期 | 作者覆盖 |
| --- | --- | --- |
| P01 | root2 kg、Px child1 kg、qj=1/8、g=(0,0,-2)、无驱动；root a=g、aj=0、系统 COM 弹道 | 40 步、共同 boost、轴循环 |
| P02 | root2 kg、COMx=1/8、I=(1/4,3/8,1/2)，原点 Fy=2 → aCOMy=1、alphaz=-1/2；非对角惯性 tensor 的 Tx=1 → alpha=(3,1,0) | root-only n0、实际作用点每步运输、40 步 |
| P03 | root2/child1 同心 Px、内部 3 N；Mxj=[[3,1],[1,1]]、a=(-1.5,4.5)、child 世界 ax=3；child 外部 3 N 则 a=(0,3) | 公开首节点、40 步反作用、翻向/boost；未加 root 假电机 |
| P04 | 同心 Rz 内矩1 → root alpha=-2、joint alpha=6、child世界4；偏心版本子 COM=(1/8,0,0)、非零入步角速/关节速和 root principal frame | 完整 M/b/J、非轴坐标变换、40 步；功和 KE 分列 |
| P05 | 非轴 R/P 分支 n=0/1/9/16，非连续 key503/91/1207/6 及扩展；真实 folded root M3、COMx1/12、I=(.3,53/120,13/24)，工具 Fy=3 → ay=1、alphaz=12/13 | W1/W4/W32 40 步；19→21 完整重放；ordinary/affine/free 混合发布；root/tool 原质量只计一次 |
| P06 | delay0/1、cap1/8 或8；独立全 cross motor mobility 两P例 H=[[1.5,.5],[.5,1.5]]，kd=1024/rate=±1/128 → motor=±4 | 预算1拒绝/512接受、external即时、held/queue、局部拒步和全局 latefault/retry |
| P07 | sphere/capsule/非盒 hull 的 root/child 实际材料；两个真实 folded 资产，含 Rz→Px→偏心90° fixed tool | 40 步当前姿态包围、真实 source/owner/过滤、W1/W4/W32 无 HOST 状态读回的暖查询；不是 CCD |

独立参考使用原声明的齐次变换一阶/二阶 time jets：对根 COM 漂移和世界角速度建立 T/Tdot/Tddot，沿实际 parent 图合成，逐部件 COM 与惯量求和得到完整 M/b/J。PD 真值用有限维 box-face 枚举和全局 KKT 校验，未复制产品坐标迭代。P01–P04 另有六个手推解析锚；旧 HOST 动力学仅作额外交叉核，产品 M/J 或 accepted 输出不会生成独立期望。

W1/W4/W32 物理副本保原 world ID 存档，断言时显式映射声明等价副本；不删字段后比 hash。恢复后完整 state、geometry、diagnostic、energy 与原21步逐字段相同，计数/epoch不伪装相同。相同入步同物理副本的参考值只在该步复用，先核它们完整状态/诊断等值，避免重复求同一参考；真实每 world 设备推进没有省略。

## 边界与真实失败注入

- 原 anchor=2**60、local=1/1024、vx=1：真实一步后世界 x 仍为 anchor，但 checkpoint local 为 h；恢复后同请求必须重现第二步全部状态/几何/诊断。world/local 不一致、长度、非有限及越域分别拒绝。
- 声明角步长 pi 与 pi−2^-20 验实际 stored 速度产生的姿态与 `q_new·q_old>=0`，保原 >pi 拒绝；不改角域或归一化门。
- root local/joint 4096 边界向外的真实下一步、rotation vector 4 rad 均仅拒该 world，邻 free world继续；旧 diagnostic 不能冒新 attempt。
- 真实 `upload_controls`、`check_advance`、`check_restore` 先执行再抛错，覆盖两个 bank；全 batch 状态/控制/energy/epoch不发布，同请求重试与 fresh 路径相同。
- 部分 floating buffer 已实际分配后抛错，核原资源栈清理并可重新构造；批关闭关闭全部计划、owned bytes归零、旧读数/跨 owner 不可复用。
- 宽相 capacity 不发布部分 candidates；真实 BVH query/readback 返回后故障也保 last result 与物理状态；same-tick restore 使旧 result失效，后续同状态包围精确重现。
- source radius/vertex IDs/revision 负控先匹配真实 source model，避免被更早的 model mismatch 遮蔽；另独立 model/source、world、group、同体映射、直接父子、mask/exclusion 控制。
- 同一 `step_contacts` 中原 fixed/affine 接触世界照常接受、未绑定浮根正常独立推进；接触 receipt 不伪包含浮根，浮根没有 `.physical` 旧适配器字段。

暖态不允许 `snapshot/tree_kinematics/tree_diagnostics`、HOST floating FK/M、动态 envelope 上传或新设备分配。实际 `floating_bounds_kernel_launches`、原 device allocations、floating state/FK/diagnostic 读回次数及字节分别留存；必要 status 同步和候选读回不说成零传输，也不把 interpret 叫 CUDA。

## 保存、执行顺序与交接

测试按固定 case/参数/tmp目录写无损 `.json.gz`，保原声明、完整 held/history/state、M/b/J、载荷/acceleration、FK、能量、计数及轨迹。40步主窗口在 finally 中保已完成前缀、异常类型和真实 retained state。纯输入拒绝项是固定配置负控，不伪造物理轨迹。大 W32 保存全部真实公开 world记录，gzip仅压缩，不抽样或改变浮点数据。

父可用现有 fresh guard 和项目解释器执行两个 NEW 文件：

```text
-m pytest tests/scale/test_resident_floating_step.py tests/scale/test_resident_floating_collision.py --tyx-mode interpret -q --basetemp <new-absolute-directory>
-m pytest tests/scale/test_resident_floating_step.py tests/scale/test_resident_floating_collision.py --tyx-mode cuda -q --basetemp <another-new-absolute-directory>
```

若同源码首 P03 已完成，完整矩阵加 `--deselect tests/scale/test_resident_floating_step.py::test_p03_public_floating_tree_keeps_complete_mass_and_internal_recoil`。作者未 collection；实例数以父真实收据为准。HOST解析节点无需在多个设备间冒重复设备物理结果。未配置独立性能重复或速度门；W32规模只验功能/所有权/真实代价，耗时未知即如实记录。

父已读 `4b7362c` 的参考245行及 step主体，之后增量仅：`a2dc3d3` 低budget/输入/所有权，`09c70df` 首P07，`364ad9b` 上述设备观察精度修正，`fabc63e` 大坐标/π/资源/完整P07边界与按权威计划修正 P04 子 COM。参考新增4行只冻结 P04 的真实 Rz/COM=(1/8,0,0)，此前这个未执行case误用了P05通用COM；没有根据执行结果调参。

依赖父接入的公开新类型/普通浮根 storage/hooks，以及已接受的 C5/C7 测试声明辅助和原两份 floating_fixed URDF。没有新增产品、私有数值推进或新fixture。后继原生 Newton FREE-root 40步、源码外 wheel consumer 和同次保存审计需父另行冻结；本四文件不提前声明完成它们。
