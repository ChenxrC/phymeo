# C8 双动态驻留接触：Newton 参考与源码外消费者

状态：**CODE_AUTHORING / UNEXECUTED**。2026-11-08，按父冻结 [C8 计划](../superpowers/plans/2026-11-08-resident-tree-dynamic-contact-feedback.md) 编写；没有运行 import、collection、数值、pytest、静态、build、安装或设备程序。本文不是验收记录，以下计数均是预声明执行规模。

## 精确文件与边界

本作者只新增：

1. `tests/reference_newton/robot/resident_dynamic_contacts_reference.py`：独立 pin Newton 原生参照，可选单独 matched-law 通道。
2. `tests/wheel_consumers/resident_dynamic_contacts.py`：自包含、公开 Phymeo API 的安装包消费者。
3. 本报告。

父修改旧 resident pair/contact/batch 接缝；原 RD 拥有 D01–D08 完整功能与事务测试。两脚本不导入 Phymeo 仓库测试、不修改产品，也不代替完整矩阵。旧 C6/C7 consumer 不重跑。新 consumer 只内嵌复用其已接受的 JSON/完整 wheel 成员与 module provenance 核验、逐侧材料/作用点审计方式；执行时不依赖原文件。

## Newton1.7 独立参照

默认 `--channel native`，每 device 的 D01/D03 各实际 40 步，h=1/1024；首步是同一窗口第一步，不另外重置或冒采样为推进。CPU 与 CUDA 分别执行，CUDA 不可用时明确失败，不能 CPU 回落。质量、惯量和原法向律固定：

| 案例 | 独立声明 | 首步独立真值 |
| --- | --- | --- |
| D01 | 两 P_x 根，mass=(2,1)，COM=各根原点，I 各 diag(1/8)，pivot x=(0,7/32)，球 r=1/8、local center=0 | gap=−1/32，F=(−2,+2)x N，a=(−1,+2) m/s²；总自由轴动量零 |
| D03 | 两 R_z 根，同 pivot；球 local center=(0,1/4,0)，r=1/8；mass=(2,1)，COM=pivot，I=diag(1/4)、diag(1/2) | gap=−1/32，F=(−2,+2)x N，COM 矩=(+1/2,−1/2)z Nm，a=(+2,−1) rad/s² |

均 q/v=0、gravity=0、target/effort=0、armature=0、关节阻尼/摩擦=0、angular_damping=0；形状 density=0 且 lock_inertia=True，禁止碰撞球重新覆盖声明质量。ke=64、kd=kf=ka=mu=0，shape margin/gap=0。碰撞行的 effective margin 仍可能含球半径，脚本保存并解释该值，不误断言所有 contact margin 为零。

模型用真实 `ModelBuilder.begin_world/add_link/add_joint_prismatic|revolute/add_articulation/add_shape_sphere`，两个 native articulation 位于同一明确 world；这与 D01/D03 的 block-diagonal forest 物理一致，但不声称验证 D02 共享祖先 cross-M 或 affine 图。`CollisionPipeline.collide` 在真实入态产生行，`SolverFeatherstone.step` 原生施加双方载荷、独立积分，只有初始化调用 `eval_fk`；之后不写 q/v、不上传 Phymeo 状态。

默认原生力读数来自本机 pin 的 `solver.body_f_ext`：`solver_featherstone.py:591–604`先拷贝 public 外载，`:717–746`叠加原生 contact，随后 `:767–775`进入原 articulation 动力学。本通道 public `state.body_f` 为零，因此该 scratch 是本次真实接触合载荷；同时保存真实 `body_qd_fk`（同次接触用 COM 线速率+世界角速率）。这是有源码绑定的 private **读回**，不是第二求解器、接触力注入或通用稳定观测 API。Featherstone `SUPPORTED_OBSERVABLES` 当前只含 BODY_PARENT_F；`Contacts.rigid_contact_force` 声明为 unused，脚本原样保存但不把它当力传感器。

每步保存所有有效 raw contact 行的 shape/body owner、point/offset/margin/normal、材料 override、原 model metadata、原 state 全 q/qd/body pose/twist/force、原 native 合载荷及 qdd/H（若存在），并从这些行独立重建两侧真实 COM 力矩。raw normal 为 A→B；effective point0=support0+margin0*n，point1=support1−margin1*n。世界角矩合计零只验接触力本身；D03 固定轴的支座约束不被冒作自由刚体总动量守恒。

独立轨迹 oracle 用声明几何+二个标量实际惯量递推，全程不从 native H/J/qdd 生成预期。另对每个真实 entry q/v 做 same-entry 载荷/一步结果检查。有限入口材料点的末态位移功、entry power*h 与动能变化分别保存，不施加不成立的严格能量等式。原工程门：atol=5e-6、rtol=5e-5、40 步 atol=2e-4；没有随输出调参。初始独立 oracle 还须精确符合上表 2 N 与 a。

可选 `--channel matched` 是**另外一个自演化窗口**：从 Newton 自己的入口 body pose 计算同球法向律、施等反 COM wrench，`step(..., contacts=None)`关闭本次原生 contact 施力以防双加；原生检测行仍只读保存。输出明确 `contacts_applied=False`，绝不将该模式称 native law/检测等价。默认验收先跑 native；如原生通道确有差异，保首错，不自动切 matched 挽救通过。

本机来源固定为父 `.venv-newton-1.7.0/Lib/site-packages/newton`（实际 distribution 校验 newton=1.7.0、warp-lang=1.17.0）。脚本保存 builder/state/articulation/collide/contacts/broad_phase_common/Featherstone/contact 核完整源码快照与 SHA256、driver 字节以及运行后来源 hash。主要语义依据另见 `state.py:123–145`、`contacts.py:270–305`、`kernels_contact.py:423–558`；这些路径是 pin 内路径，不联网取另一版本。

参考输出每 case 一个完整 `.json.gz`（单个未压缩上限128 MiB）和总 `summary.json`。数值门失败保 first_failed_gate，仍按原输入完成可执行的40步；执行异常保 partial trace/execution_failure，不用后续成功覆盖首错。IO失败不被声称为物理回滚。

父待执行命令（输出目录必须新建）：

```text
<parent>/.venv-newton-1.7.0/Scripts/python.exe <author>/tests/reference_newton/robot/resident_dynamic_contacts_reference.py --device cpu --output-dir <fresh-native-cpu>
<parent>/.venv-newton-1.7.0/Scripts/python.exe <author>/tests/reference_newton/robot/resident_dynamic_contacts_reference.py --device cuda:0 --output-dir <fresh-native-cuda>
```

## 公共 installed-wheel 消费者

仅标准库、NumPy、`import phymeo as ph`。正常 `FixedRobotTree` / `AffineRobotTree`、原 `TyxWorldBatch`、公开 bindings 与 `prepare_contacts(..., tree_dynamic_pairs=True)` / `step_contacts`。脚本不读 checkout、测试、旧 artifact、私有 backend 或私有 advance；D04 URDF由脚本声明生成，所有独立 oracle 都内嵌。

固定 world 注册顺序 `(1207,6,91,503)`，真实 shape/binding 登记另反序，碰撞 canonical key 和物理 group 不同义：

| world | 完整物理与独立预期 |
| --- | --- |
| 503 | D01 普通两 P 森林、两个真实动态球；M=diag(2,1)，g=(−N,+N)，a=(−N/2,+N)，N=64 max(0,1/4−distance) |
| 91 | D04 非单位负gain/offset：P=(1,−1/2)，c=(0,1/8)，第二安装x=3/32，mass=(2,1)，I各diag(1/8)。Mr=9/4，a=−2N/3，原 C=(1/2,1)，lambda=−2N/3；首步 a=lambda=−4/3 |
| 1207 | 原静态支撑：mass2、P_z，入口 link z=3/32，半宽1/8盒对有限上表面z0底座，N=64 max(0,1/32−q)，a=N/2；真实 static reaction=−Nz |
| 6 | 未绑定旧 FreeSphere，mass2、r1/8，p=(0,0,2)、v=0、gravity=(0,0,−8)，每步旧通道实际推进；不将该 free body 注册成 C8 动动参与者 |

四 world 都保存完整状态/control/history/ticks，树保存完整 K/N、真实 FK、原 full M/b/J/约束反力、materials/source packets/IDs/weights/skin、双方作用点/速度/有限位移/功、静态反力、entry/current 两种 bounds/candidates 与原始统计。所有碰撞框必须包含声明有限形状的独立实际 q 几何；宽相候选不冒窄相。所有 body 惯量、力帽8、迭代512、delay0、h及工程门沿冻结计划。原旧普通控制的完整N×10与affine K×3/N×7 held全部逐值核零，不只核shape。

前40步连续推进；保存tick19，先在整批checkpoint中仅将affine独立q改成4097作公开非法restore负控，确认所有world/epoch/旧receipt原子保留，不添加物理步。随后合法restore后把tick20..40的21个完整state/FK/diagnostic/receipt及材料bounds/candidates逐字段复现；epoch/累计资源计数不倒退，单列实测。随后另一个plan令world91 site_capacity=0：该world完整保留而其余3world推进；旧plan已发布receipt字节不变。关闭limited计划后以原plan重试，保留world从原tick40确实走到41，其余走到42。共63次batch step调用、252个world尝试，预期251次接受；此数是预注册规模，最终以实际record为准。

保留最初owned receipts跨40步/replay/拒绝/重试不变，并检查不可改写；最终batch重复close、contact scratch/原bounds资源归零、关闭后step拒绝。不会为该 smoke 注入私有故障或重复C8全事务矩阵。

外部调用示例（解释与CUDA分别新输出，文件复制到checkout之外，工作目录为复制脚本目录）：

```text
<main-python> -I -B resident_dynamic_contacts.py --mode cuda --output-dir <fresh-output> --wheel-site <fresh-install-target> --expected-package-root <fresh-install-target>/phymeo --expected-wheel <actual-wheel.whl> --forbidden-checkout <parent-checkout>
```

`--forbidden-checkout`只做路径排除、可省略；没有该目录内容依赖。要求确切local wheel URL、实际archive SHA256、全部installed/wheel成员与字节相同，允许规范许可的空 `archive_info`；所有loaded Phymeo模块origin/spec/search path都必须在指定新wheel target并匹配wheel字节。`tyx-compute`使用实际distribution名。依赖安装/构建及外层source parity由父唯一串行执行。

结果写完整 `resident-dynamic-contacts.trace.json.gz`（未压缩上限256 MiB）和 `summary.json`，保存异常前已完成或保留状态，IO failure与物理拒步不同。gzip先写临时文件再替换，摘要绑定gzip字节数/hash；没有按输出挑选成功帧。本文件未执行，Ruff/format/两backend/Newton/native实际对齐与新wheel结果全部待父记录。
