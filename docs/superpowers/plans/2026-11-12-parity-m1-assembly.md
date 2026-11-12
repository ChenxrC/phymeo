# Newton 对齐下一批：BALL-M1、设备 M1、场景装配

状态：2026-11-12 11:30 启动；用户已明确“继续下一批”。本批没有新截止，旧07:20批次已关闭。基线48724ae1，产品基线1ce57489；既有验收复用。按[下一批总计划](2026-11-12-newton-parity-next.md)进入实现，不重新扩写总路线。

## 所有权和交付

| Lane | 原作者 / worktree / branch | 本批交付与精确范围 |
| --- | --- | --- |
| BALL-M1 | mtyx_t2_audit / p65-h1-capacity / codex/p65-ball-m1-20261112 | [机器人合同](../../development/p65-robot-next-compositions-20261112.md)第3–5节及十组案例。NEW backends/mujoco_tyx/_joint_transmission.py；独占MOD同目录_tree_model.py/_tree_kernels.py/_tree_device.py；NEW tests/test_mujoco_tyx_ball_joint_motor.py、test_mujoco_tyx_ball_joint_motor_transactions.py、tests/installed/check_ball_joint_motor.py、docs/development/p65-ball-joint-motor-report.md |
| 设备M1 | mtyx_t2_tests / p65-h1-actuator / codex/p65-boundary-m1-20261112 | [公共API父裁定](../../development/p65-m1-api-freeze-candidate.md)及[17组案例](../../development/p65-m1-case-library-20261112.md)。先S0 tests/schema与NEW docs/development/p65-m1-buffer-layout.md固定完整数组/预算；父文本核对后同节点S1。NEW后端_resident_boundary_source.py/_resident_boundary_query.py/_tyx_boundary_source.py；MOD _tyx_collision_bvh.py/_device_triangle_distance.py；NEW原合同的reference/两tests/installed消费者及p65-resident-boundary-report.md |
| 场景 | mtyx_t2_source / p65-h1-support / codex/p65-scene-assembly-20261112 | 先NEW docs/development/p65-participant-scene-contract.md、tests/test_participant_scene_compiler.py冻结显式新声明/API、原ROB1多案例和错误语义；随后NEW builders/participant_scene_compiler.py及报告。原scene_compiler.py旧声明/默认路由保持。公共导出和必要既有接缝交父 |
| 父 | p65-parallel-integration / codex/p65-parallel-modules | 唯一既有共享集成owner及唯一项目执行owner。机器人_model.py/_mujoco_tyx_runtime.py、NEW backends/mujoco_joint_transmission.py、必要_normal_device.py/_constraint_device.py仅mechanics调用ABI；M1 tyx_world_batch.py关闭/注册/public prepare；公共导出。子lane不改这些路径 |

上表backend/源路径均相对src/phymeo/。旧四binding/storage、旧凸体/triangle数学核、浮根核、activation核、PPO/环境和G1不改。子作者只在各自新分支操作；机器人原未跟踪T1 review及父三个K1 WIP保留。不得reset/覆盖、扩大文件范围或新建助手。

## BALL-M1 已授权的实现合同

复用原完整动力学与积分。固定BALL树nv<=8/nq<=10/nu<=32，直接BALL目标只JOINT/NONE/FIXED/BIAS_NONE、na=0，无可能接触/FREE/limits/dry/spring/equality；旧无BALL目标的BALL-A/C与ROB2b域保持。scalar NONE/affine-bias可共存。统一native transmission CSR：row_offsets[nu+1]、dof_indices[nnz]、moment[nnz]，nnz<=96；每scalar1列、BALL3列（零系数不删），完整native actor/joint/type/qpos/DOF/gear身份保留。所有静态表immutable。

按native actor顺序计算实际scalar force后同一映射累加广义力，保disabled→force clamp、共享目标累加和真实控制限幅。BALL BIAS_NONE不读伪length。新表/实际device scalar force和完整qfrc的typed receipt由父新增，不改scalar-filterexact-v1含义。作者S0首提交同时给出静态表字段、实际设备输出字段/shape和mechanics调用ABI；父据此实现public receipt和wrapper，无需另问用户。只读native源码可用，不调kernel执行探针。

测试先落原十组完整XML/输入，保h=.001/24步/W={3,29}/三段原控制与非零入步速度、所有原门；同入步native、独立连续native及独立gear/功率oracle并列。首次物理失败按实际层定位，不调输入/容差。旧scalar算术/launch顺序兼容必须验证。

## M1 与场景的首提交边界

M1的S0必须给可计算的注册/持久/query/暂存buffer字节式和失败峰值（不以测量倒填），包括source/anchor/bounds/BVH/pair pack/18候选raw全数组、empty/capacity exact/one-less。明确batch借用scope与runtime lease不同，PENDING可close重试、UNKNOWN不可重试；父负责关闭保护。固定M08/M10包含rigid安装过滤及旧triangle混合body拒绝，不伪造单面mesh。

场景新显式版本须通过公共ResidentSceneBatch/prepare_contacts/step/snapshot/restore连接，保持logical world/participant/asset/body/collider/control身份。首先复用R01双自由体、R02臂推盒、R04四family、W2隔离/完整world恢复/partial reset/晚失败。无新求解器、私有bank或跨树隐式KKT；不能把MuJoCo-TYX BALL导入resident标量树。父只在明确类型/错误语义后批准该独立新文件实现，不重开大规划；静态工具/无接触材料须显式。

## 执行、保存与停止

所有作者先tests-first、小提交、文本交接；严禁作者import/collection/数值/probe/pytest/lint/formatter/build/install/profile。父先独立阅读新差异、登记准确源/测试/环境/命令，再fresh entrypoint-aware guard进入唯一串行槽；真实外来执行自然结束，不kill。首静态错误/真实测试失败/raw完整保留，修复后只验证受影响节点和最终新组，不重跑同源已绿组。

每lane通过源码新域和受影响兼容后，合并新stage只构建一次wheel并验证三个公共消费者及准确受影响旧消费者；完整source/wheel/installed身份和输入/力/运动/失败证据保存。若某lane尚未完成，不把它塞进通过声明；独立片可先交付。

不新增训练/虚拟Adam分析或G1长窗，不清理环境/证据Junction，不remote CI/push/mainmerge/发包，不改pin/物理门/质量/预算。连续两次针对性尝试未缩小同一问题，则保最小反例并暂停该片，继续独立lane；不让反复精度证明阻塞通用模块。

成功以公共API、多模型/来源/world/连续步、两后端、故障恢复和源码外消费者为准；研究/代码完成/测试通过分别标记。本批三个模块完成也不代表整个P6.5或Newton全功能完成。

## 父共享实现首片：M1 runtime 所有权

父新增 tests/scale/test_boundary_runtime_ownership.py（tests-first 1964a392）并修改 tyx_world_batch.py。范围为三种runtime lease hook、专用accepted只读借用scope及batch关闭前保护；同线程producer重入明确拒绝。旧convex/math/数值核不改。首组为11个HOST生命周期节点：正常关闭、三类BaseException的UNKNOWN、PENDING重试、遗留lease、错runtime/token、强owner存活、scope正常/异常退出与递归close。全部使用明确synthetic owner，不称真实CUDA释放资格；实际M16和物理连续来源仍由完整M1覆盖。

首执行仅这两个精确Python文件Ruff/format检查、必要纯style修复、完整新HOST文件及受影响旧empty/close目标；父单一串行队列在新 artifacts/p65-parity-m1-assembly-20261112-01，执行前fresh guard，禁止并行。此片不是M1查询已实现。

### 父所有权首片资格

2026-11-12 首次精确两文件 Ruff 与 format-check 均 0；新生命周期文件 11 节点和两个受影响旧 empty/close 节点同次 13 passed / 1.34s / RAW_EXIT 0。证据为本批 001–003，源前后 SHA 未变。测试使用 synthetic owner 验证 UNKNOWN 强保留、独立 sibling 清理、PENDING 重试、借用期拒绝重入；真实设备 M1 仍未完成。没有项目物理或 CUDA 通过声明，也未重跑旧全组。

## 父 BALL-M1 连接补充

父把同次 transmission 的 typed receipt 注册到现有 mujoco_checkpoint.py 白名单，恢复时绑定原静态 CSR/gear、原请求控制、accepted qfrc 和 receipt force；没有新 checkpoint 格式或激活状态。该文件是必要既有共享接缝，纳入父范围。原 tests/test_mujoco_tyx_ball.py 的 ball-motor 和 test_mujoco_tyx_ball_scalar_actuation.py 的 ball-target-none 两个 stateless 拒绝项由 NEW exact-source 两步两后端正控替代；含接触 BALL-C 的拒绝项保持。

场景作者另增 tests/installed/check_participant_scene.py，自含资产级 W2、多步控制和 checkpoint/reset 后继，源码外消费公共API，不导入 tests/helper。没有新增算法域。
