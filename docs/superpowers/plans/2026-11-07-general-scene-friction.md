# 通用场景持续切向接触：实施候选计划

2026-11-07。状态：**仅文档调研，UNEXECUTED，等待父冻结 tests-first / product 范围**。用户优先级为场景与机器人；本片补真实共同物理步的适配，不另造摩擦求解器，不再以一个 Panda 或原100g失败夹具作为通用能力证明。输入及门见 [案例册](../../development/p65-general-scene-friction-casebook.md)。没有本轮项目导入、数值探针、测试、静态检查、格式化、构建或新 benchmark。

研究基线为父 `fdfa586`，文档分支基线 `c8551e3`。F2 T3 已由父实测13 HOST + 13共享、四窗2560步和源码外wheel；该成功只表示浮根 affine mimic 的 normal/contact 共同积分已接通，不表示持续切向已经接通。

## 1. 真实差距：可以复用什么，必须补什么

| 已有源码 | 可复用事实 / 本片阻断 |
|---|---|
| `runtime/multi_robot_scene.py:302–307,339–398` | 多普通/fixed mimic/floating/affine mimic owner 已同入步采样、各一次候选积分、一次发布；持续切向入口直接拒绝，publication尚无 tangent state/receipt |
| `runtime/_scene_coupled_tangent.py:38–61,121–185,257–319` | 旧 HOST 普通刚体和单普通 P 桥已有选 pair、真实 normal、全站点核和末态验收；只排一个固定树，robot J 只含线项，不能直接泛化到 R/浮根 |
| 同文件 `:326–380` | 单机器人 load 重写含一条 contact 一个 site 的 details 映射假设；通用凸 patch 可以一组件多站点，不能 zip 后丢分布力/纯矩 |
| `runtime/_coupled_tangent_step.py:159–243,285–367` | 输入 full world W/free motion，内部转材料二维；prepare纯候选、finish同时核真实材料/作用点位移与功，再产生新历史；零站点也可释放/推进 |
| `solvers/coupled_tangent.py:193–281` | 复用弹性弹簧–库仑 full-cross-W 凸核、KKT和有限迭代；同一站点使用相同静/动 μ，不冒称新 μ_static≠μ_kinetic 模型 |
| `solvers/tangent_history.py:316–331,334–388,476` | 显式token/双侧材料地址、完整frame运输、分代/释放能、固定容量、精确base/payload绑定和纯重试；不是最近点匹配器 |
| `runtime/material_stations.py:105–129,307–349,384–473` | 按实际entry events匹配全部已选pair正Nsite；保存入口双方材料和作用点局部锚点，末态只重建原锚点；同chart显式声明，不hash世界点 |
| `runtime/_robot_participant.py:132–206` | ordinary/fixed mimic 已保fullM/fullbias、P投影质量/驱动与full接触；接触投影后原积分一次 |
| `runtime/_floating_robot_participant.py:194–260,340–353`；`_floating_mimic_participant.py:155–240` | 浮根PD使用原free-root Schur；full COM Jacobian/载荷保真，mimic按S投影并仅调用一次约束step；不能为摩擦再启动一套participant积分 |
| `runtime/contact_events.py:239–297`；`_robot_robot_contacts.py:22–35` | 多机制owner有真实world/body/link/frame；事件的入步时间与末提交时间不同。fixed_root_contact_wrench仅直接根collider接触，不是传经关节的全部地面反力 |

`_material_stick_law.py` 的既有 deformable 静/动回映保留；它有独立静/动系数和标量 inverse 响应。不能对每个机器人站点单独调用这个标量律来替代上述 full-W 核，也不以统一事件身份账本冒充已经实现的静摩擦。

## 2. 首个可交付切片与后续门

**S1：通用 owner 响应 + 纯候选事务，先平移可验域。** 用同一个现有 `enable_coupled_tangent(registry, ...)` 让 MultiRobotScene 运行案例 F01 两普通P/自由箱，随后 F03 fixed mimic/共同祖先、F04 ordinary floating/affine floating。保所有原法向材料、全 collider 和未选 pair 原摩擦；不靠新增 dummy owner 把单机器人强行变成多机器人。目标是用户只换资产/声明即可调用，而不是测试代码私下把核输出写回状态。

**S2：真实角 J 和末态运动门，按实际通过域发布。** F05 的有限 R 工具绕固定法向转动是最小正控；有限杆/F02及卸载F06覆盖不同patch/边界。任意变法向滚动、接触点自动迁移、跨面自动转移弹簧不在首正域。若同法向小转动也无法通过现有actual-motion/work门，只发布S1，保F05失败，禁止删除guard、抬误差门或变更原窗口。

**S3：复用同一桥进入单 fixed-mimic RobotScene，原100g回归。** 可与S1共享实现，但验收后置：沿当前单机器人公开路径接入，不改默认builder派发或建第二套物理owner。F07的原控制/载荷/成功门不变，仅显式选择新切向模型，kt预先固定。失败仍失败；不再按本例结果扫刚度、摩擦或夹持门。

首次最小公开RED固定在F01：旧真实normal采样先核两个2N，随后调用 `enable_coupled_tangent` 命中现MultiRobotScene行为拒绝；不是missing-module。产品完成后先跑该节点，再扩独立projection、两侧功和事务节点；已绿同源旧大套不重跑。

## 3. 完整跨owner响应，不做逐站独立求解

按真实动力学owner建立互不重叠的速度块；一个mechanism只出现一次，shared ancestor属于该块完整质量矩阵。固定根collider有材料姿态但零动力学行；自由刚体、root及各child不能被重复算成两个owner。

| owner块 | 独立速度投影 Q / 接触可见维数 |
|---|---|
| 普通fixed tree | `I_n` / n |
| fixed affine mimic | P / n_full；控制仅k |
| ordinary floating | `I_(6+n)` / 6+n |
| floating affine mimic | `diag(I6,P)` / 6+n_full；控制仅k |
| 自由刚体 | `[v_world,ω_world]`，当前world全惯量 / 6；固定body为零响应 |

实际共同施力点 p 的 full Jacobian为 `J_p=J_COM_linear−skew(p−COM)J_angular`。站点相对行按 canonical second−first，乘每机制Q后构造 `J_i,d`。每块 `H_d=Qᵀ M_full Q`，全矩阵 `W_ij=Σ_d J_i,d H_d⁻¹ J_j,dᵀ`。用已验线性解，不先反演fullM再随意夹P；不能丢掉跨站点/跨robot通过共同自由体产生的非对角项。包括单站施力会转动自由箱的项，即使特定对称窗口最终净矩恰为零。

浮根直接消费当前fullM/b和COM J；父F3的 `root_inertia_frame_xyzw`/principal装配保持authored body frame几何，不从 `root.inertia_diagonal` 假造body-frame full tensor。F3若先合入，可增一个非对角根惯量输入留出，只复用其已验动力学接口，不在本线重写装配/重证整套F3。

自由预测速度使用同一次entry、原bounded驱动、重力/偏置、外部COM载荷、真实法向和未选pair的旧切向载荷。被选pair先精确移除旧速度正则切力，再加入本核切力；PD只解析一次，不用接触结果重算feedforward。selected μ取该pair原法则的同一系数，N取已加权真实site.force_n，不将权重作用两次。`NormalLayer`和`RobotContactMaterial`不互相转换。

## 4. 一次发布、完整载荷和两种材料位移

一次受锁step的顺序拟固定为：

1. 校验所有actions/输入；各participant按现有prepare得到entry/full physical/geometry/fullM/b/J和resolved原驱动。新桥只读这些快照，不再次prepare/step。
2. 用原 `_sample` 取同入步全部normal/legacy loads。为selected pair建立当前entry material view；不能读上一已提交`.contact_events`冒作当前几何。event observer不增加一次episode或求解。
3. 显式registry绑定每个正Nsite，建立完整response，调用原prepare_tangent_step。内部新记录定位可用本步component/site index，但历史identity只用显式token及真实地址。
4. 每个site的切力差一次散回full generalized load、两侧COM力/矩、完整PairLoad/LoadSite和details。对固定根保原“直接根接触”字段原义；总支撑反力若需要，单独通过完整动量/外载审核。多组件/非共线site不得捏为平均normal或单点force。
5. 各participant与自由体各作原一次候选积分/约束/漂移，构造真实末态OwnerPose。调用selection.measure_motion，分别获取入口双侧material anchors和原application anchors实际second−first位移；不得末态重采样最近点，也不得用弦替换某个未支持的连续碰撞路径。
6. 原finish同时验证两种位移、预测、逐站/总功、储能、释放及KKT。随后owned observation、候选geometry和events均成功，才与physical states/time/step一起assign history与receipt。任何异常保全部旧对象/序号，重试同base不消费seal或generation。

当前切向桥是入步离散受力，不新增CCD保证。这里的真实末态旋转检查不等于支持步内任意旋转碰撞，也不能用“共同作用点力矩抵消”替代实际步长误差记录。

## 5. 材料连续性与有限资源合同

- StationKey是调用方稳定采样声明；双方owner/shape/topology/feature/chart和当前权重分别保存。组件重排和world平移不能更换identity；相同数字token在不同world必须隔离。
- 同chart的稀疏权重变化只在已声明有限面内有效；最近点自身移动不计入材料位移。原frame运输含twist；frame正交/normal-aligned/maxturn校验不替代真实材料连续性。
- force0或missing selectedpair只在真实分离/卸载时结束episode并显式释放旧spring E；重入冷启动新generation。若仍有正N却未匹配旧声明，拒绝而非丢成分离。
- 首版不推断跨共同边/另一面匹配。以后显式rebind必须携带新地址/原因/释放能并与物理提交同事务；两个重合但拓扑不连面不自动接续。未证明的corner/rolling/torsion station migration作为明确后续。
- 继承原128 active/ever-loaded token上限和256默认/4096硬迭代上限；超过限额明确拒绝，无隐式O(无限站点)历史。返回观测拥有数据、只读，getters不推进历史。same-base重试有效，later-base trial必须stale。

## 6. 建议文件边界（只是提案，尚未授权写源码）

作者NEW候选：

- `src/phymeo/runtime/_scene_tangent_ports.py`：immutable entry owner response block、Q/full J、当前点/全W组装；不积分、不接管live state。输入来自已prepare的真实完整条目。
- `src/phymeo/runtime/_multi_robot_tangent.py`：同一种纯candidate上下文，包装现registry/kernel/history，输出所有site改写与候选receipt；供多/单robot父hook消费。
- `tests/test_multi_robot_tangent.py`：F01/F02/F03/F06和全矩阵/所有权/事务独立门。
- `tests/test_floating_rotating_tangent.py`：F04/F05 full投影、根反作用、真实末态和拒绝域。
- `examples/general_scene_friction.py`：真实声明式多例、固定窗、完整原始记录；不把case名传入求解器。原100g复用原控制模块，新增显式mode入口由父决定，不复制控制器。

父OLD hooks：`runtime/multi_robot_scene.py` 的配置/一次load替换/finish与publication；`runtime/robot_scene.py` 单普通R/fixedmimic同上下文接入；`runtime/_scene_coupled_tangent.py` 仅复用或抽取配置/精确site重写，保旧普通路径兼容。若需要从 `_robot_participant.py`、`_floating_robot_participant.py`、`_floating_mimic_participant.py` 取只读projection/response，父添加最小只读属性，不改各原integrate/驱动。`contact_events.py`/`material_stations.py`只在实际发现必要映射缺口时提出父hook，不能预先扩schema或放宽匹配。

内部接口建议：`prepare_scene_tangent(config, history, *, entry_blocks, entry_events, entry_frames, sampled_loads, dt_s)` 返回不可变attempt、完整updated contact contributions/loads；`finish_scene_tangent(attempt, end_frames)`返回新history/receipt。具体数据类型先与父逐字段核对，尤其single RobotScene已算过的fullM/b和Multi的prepared对象不可触发第二物理prepare。公共入口优先复用既有enable/history/receipt，不另建平行事件或force record体系。

## 7. Newton 1.7：已读锁定源码，作为机制对照而非等价求解器

本机路径为父 `.venv-newton-1.7.0/Lib/site-packages/newton/`；同级 `newton-1.7.0.dist-info/METADATA:3` 为Version1.7.0。以下是源码阅读，未import/运行Newton，不把main分支文档冒作已装版本证据。

| pinned 1.7 文件/行 | 实际机制与边界 |
|---|---|
| `_src/solvers/vbd/solver_vbd.py:388,544–566,3285–3309` | `rigid_contact_history=False` 默认；启用需latest/sticky匹配和一致buffer/generation。latest可恢复projected tangent multiplier，sticky配合pipeline重放锚点；外部rigid solver/no bodies时不等于同一历史能力。旧stick_motion_eps已忽略 |
| `_src/geometry/contact_match.py:4–61,541–578,687–690` | 按shape pair、world中点距离和normal阈值找匹配，确定性一对一竞争；不是用户双侧材料token。默认位置阈值.0005m/normal dot .995。sticky仅在fresh gap≤0且旧锚点与fresh点足够近时重放body-local点/offset及旧world normal；不能把它说成任意共同材料frame精确运输 |
| `_src/solvers/vbd/rigid_vbd_kernels.py:1847–1991` | hard/ALM分支将真实切向位移、friction_c0与lambda_t组合后投库仑圆；旧soft分支使用当步表面锚点位移 v_t·dt 与 epsilon·dt 的 IPC projected摩擦，不使用相同的 multiplier/spring 历史；不能把同solver所有模式说成同一种静摩擦 |
| `_src/solvers/vbd/solver_vbd.py:2601–2608` | requested CONTACT_F从最终迭代导出，随后保存历史/完成pose流程；不能把它与Phy入步LoadSite力不分时刻地逐值比较 |
| `_src/solvers/featherstone/solver_featherstone.py:89–104,717–747`；`_src/solvers/semi_implicit/kernels_contact.py:503–542` | Featherstone支持P/R/D6的joint-owned mimic，但明确不支持joint_effort_limit/joint_velocity_limit的内部执行；常规接触调用用相对速度和Huber正则、`min(kf·speed, μ·normal)`切力，没有本片显式弹性station history |

官方主来源可追到固定tag：[SolverVBD v1.7.0](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/vbd/solver_vbd.py)、[contact_match v1.7.0](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/geometry/contact_match.py)、[Featherstone v1.7.0](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/featherstone/solver_featherstone.py)。细节以上述本机精确行号为准。

因此本片复用Phy显式双侧材料/全frame/事务机制，不能抄一个nearest matcher就宣称与Newton持久接触一致。后续若做真正Newton参考，先取F01去除关节后的质量/力学小域，分别锁VBD模式和匹配选项；读取真实状态、法向/切力的实际采样相位及冲量，报告不同law/迭代/限幅的影响。普通关节参考可用Featherstone，但其速度摩擦结果不得当VBD sticky或Phy弹性黏着的数值等价oracle。此研究批不安排重跑此前Newton轨迹或性能。

## 8. 验收、成本和停止条件

1. 父冻结NEW文件/共享hook后，先test-only公开行为RED；独立首步矩阵含R偏置J、mimic投影和自由刚体cross terms。不另写切向核；复用原求解及KKT/energy验证。
2. S1三类普通/fixedmimic/floating真实窗口、低μ/超载、同world多机制与跨world隔离通过，才签收通用平移域。selected与未selected pairs混合必须保原law，一步全部共同提交。
3. F02边界、F05两步长实际转动和F06失败重试按独立门签收；真正运动门不通过则保失败并停在已验域。component/site重排、root/child归属、材料return ownership不能靠最终总力抵消掩盖。
4. F07为系统回归而非唯一验收；三种原输入各一个事件终止窗口，不调参寻找成功。成功也只扩展到已验材料/机构域；失败不回退此前真实失败标签。
5. 首个64步真实成本后安排长窗；没有预先声称性能收益或GPU能力。保存source/input/fullstate/fullforces/material/history/receipt、冷启动和物理步数；无损gzip。旧119HOST/旧F2整组不作本批新成绩，只由父针对实际共享修改选择必要回归。
6. 最后正常源码外wheel验证一个两普通P/freebody窗、一个affine浮根窗和独立运动拒绝；实际所有模块来自wheel。若目标是HOST，不能以同进程存在CUDA或别线宽相结果称全链GPU。

达到首个可用切片就小提交；对于尚不能成立的材料换面/有限转动，保最短反例和明确拒绝。首批不扩大到柔体、接触拓扑自动发现、通用滚动/自旋阻力、任意连续碰撞或全场景CUDA。后续仅两个依赖节点：先为确有需求的有限chart切换提供显式材料重绑定/释放合同，再按真实成本证据考虑多世界/设备接口；两者都不能作为本片迟迟不交付的前提。
## 父裁定：T1 测试与声明式案例先行（2026-11-07）

研究 d934846 已导入 b7cad91，父全文审阅。接受复用原 full-W / history / material-motion / 事务路线；不复制求解器，不自动换面。F01/F03/F04先形成公开调用能力，F02/F05/F06为独立几何、转动、生命周期门；F07后置且不替代通用多例。

原 RD 在原隔离树，**TEST_AND_CASE_AUTHORING_ONLY**。允许仅新增：
- tests/test_multi_robot_tangent.py；
- tests/test_floating_rotating_tangent.py；
- examples/general_scene_friction.py（仅真实声明式建模、输入、完整保存/独立参考辅助；不得实现物理推进或覆写状态）。

可在本案例册追加公共API/实际数据布局澄清；旧源码、旧测试、其他exports由父唯一负责。先提交最小F01公开调用/法向前提测试，再补原F01–F06完整输入与门。F07只复用已有控制例程，不复制/改其物理输入。若任何候选几何离散/构造API与源码不符，报告具体差异，修正作者接口或先由父裁定，不能从输出反算目标。

禁止全部project import/collection/pytest/数值probe/lint/formatter/compile/build/install/profile和Newton执行；父唯一串行槽。全部作者测试UNEXECUTED，纯文本/文件IO/小Git即可；不merge父、不覆盖WIP、不新建agent。父F3改根惯量frame不改变公开FK/J/材料语义。先交clean测试checkpoint、精确文件与接口，父运行真实RED后另行授权两个NEW产品模块；不要预写产品。

需直接核首步完整跨ownerW及真实广义载荷，覆盖selected+unselected混合、components多site、零站点释放、同base重试和全部发布数据。长期窗口保存原始逐步读数；不预承诺F05/F07通过，不把未知换面称任意连续碰撞。原门/输入不以通过为目的调整。
