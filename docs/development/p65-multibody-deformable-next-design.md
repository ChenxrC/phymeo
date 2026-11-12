# 多刚体与单关节机构的刚柔响应：下一最小设计

状态：**PROPOSAL_NOT_GRANTED**。2026-11-13；仅源码研究，没有实现、数值计算或执行资格。

本文据父工作树 `117a4f2e` 及当次既有 D3/D5 源码读取；父 R3-A 仍在独占串行验证。R3-A 的 MuJoCo-TYX 能力与这里的 resident 刚柔路径分别准入，不能相乘为任意耦合。D4、D6-B 的持续稳态支撑失败不由本提案覆盖。

## 结论与首片

最小有用扩展不是另写求解器，而是把 D3 的 **world→唯一刚体** 索引提升为 **round collider→participant→物理 body group**，再把同一入步 NormalLayer 的反作用载荷送入既有逐 body COM wrench / J 投影 / 共同事务。

分两项可独立验收的功能：

1. **M-B1：同一节点片与多个独立运动刚体。** 每个 participant 仍是 ordinary floating 单 body，但一个 logical world 可有多个 participant。刚体之间仅通过共同变形节点交换作用；本项不提供额外刚体–刚体响应。
2. **M-B2：同一节点片与一个 ordinary floating 关节树。** 允许原 resident `FloatingRobotTree` 的 R/P 连杆及非根 group 上的 sphere/capsule，接触通过原完整 J/M 影响根和关节。此处不是 MuJoCo 的 FREE+BALL，更不包含 activation、affine/mimic 或任意机器人组合。

先完成 M-B1 的身份/载荷/事务基础，再沿同一 body-address 表完成 M-B2；不把两种物理系统混成一个“多刚体已支持”结论。首片保持一个 D1 模型/world、固定拓扑、有限球/胶囊–triangle、逐 facet 离散 NormalLayer、零切向摩擦、entry-sampled 一阶 split。D6 可选速度耗散仍在原位置执行一次，不为本功能改参数或默认启用。

## 已有可复用事实

| 源码与符号 | 当前真实作用 | 本提案的用法 |
| --- | --- | --- |
| [resident_scene.py](../../src/phymeo/backends/resident_scene.py) `ResidentSceneBatch`、`_capsules` | 同一 logical world 完整 participant 集合；公共 state/control/checkpoint/reset/close | 保留公共 owner；禁止拆成若干独立 step |
| [_resident_participant_layout.py](../../src/phymeo/backends/_resident_participant_layout.py) `ResidentParticipantLayout.body_slot`、`body_owners`、`entries` | 稳定 participant/body 身份、每族 storage row 与物理 body offsets；每 world 最多 8 participants | 使用已有完整布局，不再凭字典中最后一个同 world participant 决定归属 |
| [_resident_batch_transaction.py](../../src/phymeo/backends/_resident_batch_transaction.py) `_begin_attempt` | `sample` 必须覆盖所有 participants；每项严格 `(body_count,6)`，实际设备 Jv/Jw 投影 | 非接触 body/participant 也交零载荷，不能省键或拼成假刚体 |
| 同文件 `_finish_attempt`、`_publish` | 检查全 participant/world 与 D1 pending tick；最后 accepted swap 后只赋字段 | 原共同事务可复用。新证据和对象必须在 swap 前构造 |
| [_resident_scene_boundary.py](../../src/phymeo/backends/_resident_scene_boundary.py) `_bank`、`compile_registration` | 原多 participant 查询已校验 logical/native world、family、row、group、source/model/graph token、anchor | 可提取静态地址校验供双源 adapter 复用；不要直接把 rigid-only plan 冒充 deformable model |
| [_tyx_boundary_source.py](../../src/phymeo/backends/_tyx_boundary_source.py) `gather_boundary_points` | 已读取 `fk[row,group,:12]`；`body_extra=1` 是 floating 根的计数补项，不是只支持一个 body 的宽度 | 无须新 FK 或新窄相数学；正确注册 group 即可读非根当前姿态 |
| [_resident_floating_binding.py](../../src/phymeo/backends/_resident_floating_binding.py) `compile_floating_tree_binding` | 接受原完整 group/topology、asset/model provenance；group 范围对应原最多 33 bodies | 从资产声明保留原身份，不降成无来源的 local points |
| [_resident_floating_tree.py](../../src/phymeo/backends/_resident_floating_tree.py) `FloatingRobotTree` | 普通浮根、R/P 子树、原完整惯性/COM/控制历史；最多 32 joints | M-B2 使用此域，不能沿用 MuJoCo native BALL 的声明 |

既有 scene 支持四种 resident family 是接缝事实，不是本次四族刚柔组合验收。首两项均只 ordinary floating，故无需同时打开 fixed/affine/floating_affine 路径。

## 真正缺失的接线

当前硬限制分散于以下位置，不能只删一条 admission：

- [deformable_scene_lowering.py](../../src/phymeo/builders/deformable_scene_lowering.py) `prepare_deformable_scene`：participant world 列表必须恰等于 node worlds，body_count 必须 1，binding.group_index 必须 0。
- [tyx_deformable_contact.py](../../src/phymeo/backends/tyx_deformable_contact.py) 构造器及 `_participants`：保存 `world→participant`。`_law_rows` 用所有几何数组的 `[0]`；`sample` 只给一个 `(1,6)` wrench；`_prepare_response_publication` 只审一个 participant，工作量和外载功也仅取一项。
- [_deformable_response_source.py](../../src/phymeo/backends/_deformable_response_source.py)：`_world_rigid` 仅存一个 participant/slot；moving 元数据固定 group=0；actual FK 验证读取 `fk[row,0]`；empty 和正常 query 都只核一个 rigid tick/world。多刚体不能绕过这些完整来源校验。
- [_tyx_deformable_contact.py](../../src/phymeo/backends/_tyx_deformable_contact.py) `transpose_response`：第五 identity 必须等于 world，`body_wrench` 只有 W 行。它必须变成明确 body 槽，而不是把多条刚体载荷合并后交给任意一个 participant。

建议增加一个纯静态、immutable 的 response layout（新模块名待冻结，例如 `_deformable_response_layout.py`）：

- `world_participants[w]`：原 canonical participant keys，未接触者也在内。
- `body_addresses[k]`：`(world, participant, group, logical_body_id, global_body_slot)`。
- `moving_shape_addresses[shape_key]`：上述身份＋native mechanical key/storage row/bank slot、完整 binding/source token；严禁 shape/world/body 别名。
- `node_offsets[W+1]` 与 `body_offsets[W+1]`：world 独占归约范围；全顺序由稳定 IDs 决定，与输入排列无关。

`sample(entry,h)` 和 `prepare_publication(...)` 的既有事务 ABI 不变。每个 site 新增明确 `participant`、`body_group`、`body_slot`，从相应当前 COM、线速度和角速度取得点速；结束工作审计使用该 body 的候选 transform 与同一 advected entry material point。多 site 指向同一 body 时严格累计，不能按 collider 重复施加同一合力。

## 力、运动与发布合同

每个已解析 site 保持现有侧别：node 侧 `beta_i F`，刚体侧 `−F`，COM 矩 `(contact_point−COM)×(−F)`。由原 scene 设备核生成
`Q_contact = Σ_b(Jv_bᵀ F_b + Jw_bᵀ τ_b)`；对单关节树不能把它退化成根 wrench，也不能把关节 child 当独立自由体积分。

验收必须同时保存：完整原查询/真实来源；每 site 与每 body 的 raw 和实际上传载荷；全部 J/M/b/外载/控制；所有 participant 的投影 generalized force、候选和 accepted state；节点散射、全部 ordered 材料冲量、D6 requested/actual damping 与固定点反力；双方 entry power、实际有限步 work。`Σ F=0` 单独不能证明正确转矩或关节响应。

公共 `ParticipantSceneDeclaration(contact_response="deformable_normal")`、`ParticipantDeformable`、`step/query/snapshot/checkpoint/restore/reset` 可保留入口；新域必须由完整静态 layout 决定。原一 body receipt 仍保持原 schema/字段形状。扩展域建议显式 versioned receipt，将 `worlds[w].rigid_participants` 作为完整映射，避免把原 `rigid_wrench` 数组静默变成另一种类型。具体 schema 名称应在编码前冻结，不以本文直接授予新 API。

一次 step 全世界原子：所有 participant 的 geometry、candidate、history、node bank、tick 和证据准备成功后才共同发布。最后 participant 故障、node/rigid tick 错配、任一 near unresolved、来源过期或容量超限都保旧 accepted 状态与旧查询/响应记录。保完整失败 prefix；同步/释放 UNKNOWN 按既有 close-only 隔离，不假装可 retry。恢复/partial reset 以完整 logical world 为单位，不能只恢复该 world 中一个刚体而留下不同时间的节点与其他刚体。

## 有界资源与首片域

不顺带提高原上限：D3 总 participant 数仍最多 8、logical worlds 最多 8；沿原每 logical world 最多 33 physical bodies；node/facet、query 和 active-site 原上限保持。B1 允许多个单体 share world，B2 允许一个多 body participant/world；两者组合成“多个关节机器人＋多个物体”另验。全部静态容量必须在 Runtime/设备分配前检查。

令 `p=max(1,site_capacity)`、N 为全部节点、B 为全部物理 bodies、W 为 worlds。一个无额外 scratch 的新 transpose 布局可为：

| 设备数组 | 字节 |
| --- | ---: |
| 原 rows/law/ids/bary/lever | `116p` |
| 节点合力 | `12N` |
| 完整 body COM wrench | `24B` |
| world status | `4W` |
| node/body offsets | `8(W+1)` |

这是设计上界，不是测量结果；若实现增加输出/暂存，冻结表须补全。原一体专用核可保留，扩展域另加一个有界 transpose 核，避免未经验证改变旧 primitive reduction 次序。查询 prepass 的 moving metadata 数量随完整 shape 数收费；实际借用的整块 FK/tick bank 按唯一数组计 borrowed bytes，不能把引用算 owned=0 后隐去 readback。HOST immutable layout、原/新 receipt 共存、失败 raw、D1 自身 reservation 与 query native-owned cap 沿原语义分别列出，不冒总 RSS 硬 cap。

## 五组固定案例家族与分层

编码前只冻结每族一个短轨迹/两稀疏 worlds 和原工程门，不执行挑参数。复用 D3 的有限三角片、原层定律与原节点设置作为起点；不会把旧长窗稳态要求偷偷替换。

| 家族 | 独立可辨别现象 | 必须证明 |
| --- | --- | --- |
| A：两个独立运动球压同片不同位置 | 两参与者不同质量/真实来速；一方非中心接触 | 两独立 body 都受载/运动，节点散射相加，双方 COM 矩与各自自由体后继正确 |
| B：运动球＋斜胶囊，分时入层/卸载/再接触 | 一方离层时另一方仍载荷，另一稀疏 world 不同步事件 | 清掉旧 site/旧 body force，无跨 body/world 残留；合法零接触 participant 仍推进 |
| C：一个浮根 R–P 链，末端 capsule 接触片 | contact body_group 非0，原非零关节速度/驱动 | 真实 child FK、点速、完整 root–joint M 耦合与 Jᵀw；既有 HOST/native-independent力学参考，不调用产品 helper 自证 |
| D：一个分支 R/P 树，两不同 links 同时接触 | 多 site 分到两 body，另一个 body 无接触 | 两 body 的不同 COM 矩/功率、全部 generalized components；没有重算为“根合力” |
| E：身份/生命周期与故障交叉矩阵 | 非零且不同 anchors、participant/shape输入反序、source变更、同world最后participant失败 | keyed 等价；cap/cap−1；query control-only失效；restore/clone或同源新owner继续、partial reset保未选raw/history；失败后无半发布 |

A/B 是 M-B1；C/D 是 M-B2；E 分别覆盖两项。至少两个源码外消费者分别从高层资产声明走 A 和 C，不能只重复低级 owner。每后端独立报告实际步数、完整失败记录和模块 provenance。动态解析状态不能由出现过一次 RESOLVED 外推：任一必须受力的 CONDITIONING 行仍拒绝整个 step，不给它补法向。

## 开发节点、归属与停止条件

| 节点 | 规模/依赖 | 精确修改归属建议 |
| --- | --- | --- |
| 0：冻结 layout/receipt/资源与案例 | 小；只静态接口，先于实现 | 独立作者 NEW layout、NEW case tests；父审完整schema |
| 1：M-B1多独立体 | 中；依赖0 | source作者 `_deformable_response_source.py`；response作者 `_tyx_deformable_contact.py` 和 `tyx_deformable_contact.py`，避免同文件双owner |
| 2：M-B2非根body＋完整证据 | 中偏大；依赖1的地址/归约；原J/M核不动 | 同response owner逐body point velocity/work和participant证据；独立tests作者参考与故障 |
| 3：D5高层装配与安装 | 小至中；依赖每项源码已冻结 | 父 `builders/deformable_scene_lowering.py` 与必要公共导出；独立consumer作者 |

`ResidentSceneBatch` / `_resident_batch_transaction.py` 由父独占，原则上无须改物理求解或发布顺序；若仅为复用来源校验需从 `_resident_scene_boundary.py` 提取纯静态 helper，须先给精确窄 diff。不得将现有 accepted source loan 的身份检查复制成较弱版本。原 kernels、MuJoCo 路由、D6 方程与 R3-A 活动文件不在此提案范围。

成功是两项各自有正确身份、双方力/矩/功、真实多步运动与一次共同发布/恢复证据；不是持续稳态支撑、全 GPU 无 HOST 或性能更快。冻结后若需新关节积分器、新接触数学、任意跨 Runtime bank 共享，立即停止该节点并报告缺接口；若同一失败两轮不能缩小，保最小反例，不扫质量、层刚度、阻尼、h、控制或验收门。完成 M-B1 可以独立交付，不用未完成 M-B2 伪装成完整多机器人刚柔。
