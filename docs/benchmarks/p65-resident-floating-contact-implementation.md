# C10 浮根接触适配实现（纯作者，尚未执行）

权威合同：父 `5013f70` 的 `docs/superpowers/plans/2026-11-08-resident-floating-contacts.md`；固定 F01–F10、原 h/工程门与测试作者域不变。子树起点 `874ff199b710d4bfdf844c0dcf9c1bd918f18ad6`。本片只 NEW `_resident_floating_contact.py`、`_tyx_resident_floating_contact.py` 和本报告。父独占旧接缝及全部项目执行，当前无物理/静态/编译通过声明。

## API 检查点

- `read_floating_contact_packet(plan, storage, worlds, bank)`：`bank` 为明确0/1，不读取active。需要父plan的 `_batch/_model.shapes/_contact_bindings/_contact_worlds/_read_contact`；storage 的 `batch/specs/rows/count/npad/bmax/dmax`、七种原bank句柄及 `_packet(world)`。原device context内sync一次，再用原计数入口读roots、joints、applied、FK、twist、energy、batch ticks，全部有界完整bank读回，之后只为选定已注册world构造结果。tick取global slot，其他数组取storage row，两者不可混用。
- 返回 `world -> {state, geometry, frames, energy}`。state是原`FloatingTreeCheckpoint`，含实际FP32 `local_origin_m`；geometry是原`FloatingTreeKinematics`；frame是原`ContactShapeFrame`。动态侧root为body0、child i为body i+1，使用该bank的真实COM和COM/world twist；static侧原world vertices、mean reference、零速度。全部public数值由原不可写owned类型保存。world点是stored FP32值扩FP64后加原anchor，不重建HOST FK、不正交化、不反算本地根状态。
- `floating_contact_diagnostics(storage, worlds, arrays)`：`arrays`键严格使用`linear/angular/mass/vector/entry_pose/info/diag`（vector单数）；分别为 `[T,Bmax,3,Dmax]`两份、`[T,Dmax,Dmax]`、`[T,Dmax,14]`、`[T,7]`、int32`[T,4]`和`[T,3]`。来自同attempt发布前的实际读回，调用者只传status已接受的world。函数不读取active，也不触发任何device IO，拒绝非零info状态和不是一物理步的tick对。
- 构造原`FloatingTreeDiagnostic`沿C9逐字段口径：q=anchor+entry_pose.local/XYZW+vector[6:D,4]，v=vector[:D,5]，bias/external/acceleration为列0/1/3；drive仅列2的[6:D]。完整J取所有B体和D列，根六维不伪装motor。父receipt的selected_drive取vector[row,6:D,11:14]、actual_drive取[6:D,2]；input_external/contact_generalized为D，上传与法则wrench为B×6。
- `make_floating_contact_kernels()`拟返回`{'project': (bank0_kernel, bank1_kernel)}`；project参数`count,meta,linear,angular,wrench,vector,projected`与旧函数同序。meta int32第二列是N，不是B/D；每world B=N+1、D=N+6，仅加一次vector[:,1]。父负责lazy工厂调用、设备分配/launch计数/旧reject与原full solve、candidate/finalize以及唯一发布。

当前checkpoint已实现HOST packet/diagnostic适配，NEW projection核待下一小提交。没有第二solve、runtime、跨attempt缓存、隐藏snapshot、父状态赋值、held更新或新摩擦/几何法则。父将桥的允许physical body上限扩至17并仍逐spec校验；未接旧桥前body16路径尚不能运行。C9最终floating-only数值helper来自父接受版本，本片不复制或修改它。

父已确认仅NEW浮根材料的窄可表示性门：实际world_points减stored p，再`solve(stored R, delta)`必须在旧bridge `_CONSTRUCTION_M=1e-10 m`内回到声明local points。非有限、不可解或越门明确ValueError，不用R.T冒精确逆、不修点、不重置姿态；只说明输入材料坐标仍可表示，不是物理精度、全部sphere皮肤方向或防穿透认证。旧fixed/affine门不动。
