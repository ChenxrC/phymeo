# 原生资产装配与完整初态

`builders.native_assembly` 是 **native-only/physolve** 专属适配入口，不是 resident
模型，也不替代 `ArticulationWorld`。P2PE 只提供资产、placement、已解析的标量关节
输入及有限静态环境；MjSpec、资源引用和原生下标转换均留在 PHYMEO。
两个编译入口最终消费同一个 `PhySolveRuntime`，没有第二套运行状态 owner。

## 资产及模板合同 1

- `NativeAssetInstance` 以实例名区分同一资产的不同参与者；实例 pose 为 XYZ/XYZW。
  所有参与者 attach 到同一模型，保留跨参与者碰撞，不拆成独立仿真。
- `drive_mode=preserve` 保留原生 actuator/equality/contact/resource 引用，不接受
  joints 或 joint_defaults。`scalar_pd` 只接 URDF 标量关节；已有 actuator、mimic、
  URDF dynamics 必须明确拒绝，不能删掉或重复增加驱动。
- gravity/timestep 来自 assembly；其他非默认源 option 明确拒绝。此 profile 使用
  Euler、Newton 约束算法、dense Jacobian、默认原生 pyramidal 接触，不承诺与源
  自定义全局选项、resident 或旧 Newton 导出行为等价。
- 默认保留 native 自碰撞和 pair/exclude 规则。显式 self_collision=false 在冷装配
  生成实例内 body exclude，保留外部相互作用；有显式原生 pair 时拒绝冲突请求。
  尚未接公共 CollisionKey 排除计划，P2PE 显式规则请求仍拒绝。
- 静态形状目前是有限球、盒、胶囊和圆柱；size 使用原生半尺寸，支持显式
  contact_dim=1/3/4/6，不能传入会被维度忽略的非零摩擦分量。
- 完整 MJB 和 VFS 资源由原编译快照持有。主文件散列只描述该文件；完整编译身份
  使用 MJB/archive 散列，不把主文件散列冒充 mesh/include 依赖闭包。
- public identity 是 `(world, scoped_name)`；body/joint/qpos/DOF/geom 下标仅对本次
  编译有效。固定融合后的来源另由原有 collision_source 提供。
- `NativeTemplate.bind(N)` 不重复导入、复制模型或编译；数据是 `[N,width]`。
  `HomogeneousNativeBinding.create_runtime` 校验合同版本并创建独立原生 owner。

URDF 的 scalar_pd 使用 `kp*(target-q)+kd*(target_velocity-v)`；初始 target 缺省时
才取初始 q。effort_limit=None 保留资产额定值，显式 0 在这个原生 profile 表示不限力。
不改变 resident/ArticulationWorld 中 0 表示零驱动的含义。applied_force 独立写
qfrc_applied，不受 actuator cap 限制。被动 damping、armature、frictionloss 分开。

## 初态、reset 和 owner

`backends.physolve_initial.NativeInitialState` 与 MJB 散列绑定，拥有只读积分输入：
time、qpos/qvel、act、qacc_warmstart、ctrl、qfrc_applied、xfrc_applied、eq_active、
mocap_pos/quat。原生 qpos/mocap 使用 WXYZ；shape、有限 FP32 表示及单位四元数
在写入与原生准备前全量检查。它是冷初始化合同，不是 P2PE 公共状态坐标。

Runtime 可直接接受 B1a 的 `CompiledMuJoCoWorld`，不再丢失 q/v/ctrl；原生 compilation
也可配显式 initial_state。qpos0 始终是结构参考。CPU forward 只在准备时执行。
运行期仅 Host 推进；step 后不追加 CPU forward。

Runtime 为完整 reset 保存一份积分备份，准备两个 bank 的扁平借用视图与恢复内核。
mask 缓冲复用；full reset 无 mask 上传，masked reset 仅显式上传 N 个 bool。
Runtime 接管 reset owner 后在创建阶段预热所需内核签名并完成确认，首个 reset 不 JIT。
恢复只在 TYX device 内核中执行，没有按 world 的 Python 调度或 host 状态中转。
恢复后仍由 Host 整批 forward，验证完成才发布。**未选积分输入不变，不保证未选
派生观测仍是先前求解时刻的值**；不能据此授予 P2PE masked reset 全字段资格。

成本：冷编译 O(资产/关节/几何规模)，显式关闭自碰撞的 body pair 为 O(B²)；
绑定 N 不编译。两个已有候选 bank 加一份 O(N×积分宽度) reset 备份及两个 N-byte
mask；每步仍复制 11 个积分字段，验证仍有一次显式 completion 和 7 组 host 读回。
snapshot 另有 completion 和观测 D2H。Host 条件求解循环自身同步成本另计。
本入口是同步 direct 验证接口，不是 Graph/异步/D2D 高频 executor 的性能资格。

## 验证与限制

正式入口：`test_native_assembly.py`（装配、资源、身份和拒绝）、
`test_physolve_initial.py`（完整初态、独立力、N=1/8/64 reset/replay）、
`test_native_template_runtime.py`（真实 FANUC，单参数独立进程）。
P2PE 正式测试清单登记 `tests.test_phymeo_stage`，验证真实 SceneModel 物化字段。

尚未完成：Stage 的自由根/动态刚体、程序化 tree、MJCF 控制和实例语义、mimic、
显式碰撞排除、完整字段/传感器/render 绑定、实际 C ABI 和 GUI。原 Panda/FANUC
求解数值失败未因此关闭，不能用编译或简单滑块通过替代机器人资格。

2026-11-11 B1b 最终 wheel 安装选择集 116 通过、14 个旧机器人项未选，禁止三包导入。
另跑本页 FANUC N=1/8/64 三个独立进程，全部因推进后 qvel 全零而未通过原 motion
检查；后续 forward/reset 未到达。原始 mesh、geom pose 和初始 contact 集合的冷
保真测试通过；不通过关闭自碰撞、修改初态或放宽门限处理此失败。
全库文档检查仍为 479 个历史缺失目标，与 B0 日志 SHA256 相同。
