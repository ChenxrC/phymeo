# P6.5 有限厚度摩擦集成实施计划

当前第二轮成本修复产品7e68154及报告c72cdc3的受影响host195/interpret19/CUDA19、224实际记录和新源码外wheel153字节一致均通过，原30分钟workflow不变。详细版本化证据见现有CI成本修复记录；推送与新准确完整CI待闭环，最后交付框保持未勾。用户已授权隔离工作树并行编写弯曲/自碰撞/有限夹爪代码，所有测试串行；未验证代码不混入本交付head。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同一可复用阶段交付完整三角厚度棱柱/有限面几何、自动两阶段摩擦、原生反作用和实际接受路径认证。

**Architecture:** 固定材料anchor承载每行held容量；完整凸体几何和理想流形极点单独审计覆盖。复用offset port、SceneLoadReplay和finite native terminal，新增独立mapping账，不修改旧共享积分/参与者。父冻结设计后由一名实现者完成集成Task1，唯一fresh整阶段review和工程由父完成。

**Tech Stack:** 现有Python/NumPy、stdlib Fraction、实际tyx pin99f6df7c；不新增依赖。

**Spec:** [生产设计](../specs/2026-10-29-p65-thick-friction-design.md)。唯一builder的R1/R2/R3已由主agent写入spec并关闭，本合同冻结；不得重复builder。

## Global Constraints

- 基线16a752a87994940b0aee9b5a9751d63e90d196ad；既有142源码hash943f4f8a703947f9040a8b1fd3ed3e52e127beac663203c18f53e6ba564c0afc。新稿只是docs。
- 用户的[分层验收](../../development/p65-validation-policy.md)优先：新必要窗口一次、共享直接consumer、定向修复，父读同snapshot证据复用；不默认旧两CPU/fullCUDA/900。
- 整树all-prismatic且local轴为精确+/-X/Y/Z，任意合法固定frame；全动态FP64节点、distance材料或None、有限凸patch、positive半厚度、零恢复；真实device FP32，不host改state/0dt/fallback。
- CPUtol1e-8/device2e-5，free绝对能量/角动量budget1e-8/1e-5；材料变体1e-3/1e-3。512 position/velocity iterations；64anchors、8rebuilds、4096CCDintervals、depth32；只能显式按合同减少预算做unknown控制，不为通过增tol。
- 旧solver/runtime/participant/kernel/port/ledger、依赖pin、CI、旧tests期望不改。必要共享修复先报原式RED，由父裁定影响后再做。
- 单一数值/indexowner；实现者不再派助手或reviewer、不push。阶段一次fresh review，原实现者定向修复，不重复builder/probe/review。

## Review Focus

1. patch完全位于大triangle内部或真实edge-edge，cloth顶点均在patch外时不能漏pair；Task1几何/C独立检查。
2. clipped极点出生/消失/合并后不得复制held容量，FP64 anchor合并不能丢理想末速度约束；Task1身份/transaction检查。
3. 负signed gap归一化、二次根域严格等号、Fraction时刻和负budget方向不许产生假安全/假违例；Task1静态/CCD原式检查。
4. 正半厚度切向载荷必须保留完整力偶和共享节点交叉项，实际FP32载荷不能以理想J替代；Task1 mapping负控制和真实native检查。
5. 第2/末子步、terminal/close真实故障和CCDunknown不能被best掩盖，完整原状态与history retry=fresh；Task1 transaction。

## Task 1: 一次完整集成实现

**Files (create):**

- `src/phymeo/constraints/triangle_prism_surface.py`: owned surface绑定。
- `src/phymeo/constraints/robot_contact_patch.py`: 无radius的真实finite patch。
- `src/phymeo/geometry/_prism_exact.py`: 本棱柱专用二次根域和向外界。
- `src/phymeo/geometry/_prism_clip.py`: 支持feature精确project/clip/lift、材料坐标。
- `src/phymeo/geometry/prism_patch_manifold.py`: 完整SAT/流形/独立支持证据。
- `src/phymeo/geometry/prism_patch_sweep.py`: 面积多项式、整凸体分离区间与candidate fingerprint。
- `src/phymeo/validation/thick_contact_mapping.py`: 独立J.T/合力力矩功率/FP32/总聚合账。
- `src/phymeo/solvers/dynamic_robot_thick_contact.py`: 公共绑定/签名/history。
- `src/phymeo/solvers/_thick_contact_geometry.py`: 本步固定材料anchor、initial分类、rebuild/末极点。
- `src/phymeo/solvers/_thick_contact_replay.py`: 两相位原生重放与native/物理证据。
- `src/phymeo/solvers/_thick_contact_workspace.py`: 同步world法切迭代和完整接受/回滚协议。

**Files (modify):** 相关包和顶层`__init__.py`的新增导出；`scripts/smoke_installed.py`新增一个小型thickness正常入口。旧入口/断言保持。报告仅本阶段SDD路径，允许artifacts日志。

**Tests (create exactly these five files):**

- `tests/test_thick_patch_geometry.py`
- `tests/test_thick_contact_mapping.py`
- `tests/test_thick_robot_contact.py`
- `tests/test_thick_robot_transaction.py`
- `tests/test_thick_robot_tyx.py`

**Interfaces:** consumes public `TriangleOffsetPointPort`, `SceneLoadReplay`, `FiniteContactTerminalProgram/Ledger`, node/robot PointVelocityImpulses, MaterialPath/FrozenRobot及MaterialContactBudgets。Produces spec中surface/patch/program/manifold/sweep公开API及owned diagnostics/history。不得通过私有native kernel绕过这些接口。所有新模块一起进入唯一review，以下小步不另派review。

- [x] **1. 先写schema/完整几何RED。** `test_owned_surface_and_patch_schema`检查owned/canonical/no-radius及非法域；`test_patch_inside_triangle_has_complete_four_corner_manifold`使用C方形，三个cloth角全在patch外但四patch角进入；`test_side_edge_vertex_and_true_edge_edge`独立SAT/原始不等式；`test_quadratic_field_sign_and_negative_gap_order`覆盖sqrt非有理、恰零、负gap排序和Fraction精确保留。运行单文件看到真实缺API/行为RED，保留日志。
- [x] **2. 实现两个schema、专用exact/clip和manifold。** 有效几何报告实际全凸集的支持证据、clipping极点和owned材料坐标；不能把采样点集当覆盖。重新运行上述单文件应GREEN；未实现CCD用例不得计通过。
- [x] **3. 写并实现CCD RED/GREEN。** `test_translation_contact_certifies_whole_prism`；`test_changing_normal_safe_path_and_interior_crossing`；`test_area_collapse_and_budget_exhaustion_are_not_safe`；`test_exact_time_and_candidate_fingerprint`。两端点之间法向变化不能线性插值；独立解析安全/违例和有理面积多项式验证。新增`prism_patch_sweep`，不改旧robot_patch_sweep。
- [x] **4. 写mapping及原生两相位RED。** `test_full_offset_mapping_preserves_actual_force_moment_power`、`test_shared_nodes_and_robot_keep_cross_terms`、`test_missing_thickness_block_and_centroid_only_are_rejected`、`test_fp32_requests_and_actual_native_response_are_independent`、`test_phase_normal_capacity_exact_sum_and_row_scale_invariance`（n0!=n1，pbar1+etabar2^-53；新增零行不改变原行门槛）。删除厚度块负控制须触发完整力矩或映射门槛，单centroid B末角closing须失败。实现新mapping账及replay glue，不能用port自report自证。
- [x] **5. 写公开program和A/B自动响应RED。** `test_two_jaws_static_and_low_friction_single_step`两mu、`test_open_impact_full_three_corner_oracle`、`test_distance_material_pre_ledger_and_terminal_post`。期望严格采用spec的Fraction值；不给solver预填p/t/eta，含世界和节点冲量、独立K/U/Py/末速度。实现零初值、统一alpha/synchronous replay及终末总聚合。
- [x] **6. 完整材料身份/有限生命周期。** `test_anchor_birth_never_copies_held_capacity`验证新terminal点p0、旧closed只mu*eta、消失行严格零；`test_partial_overlap_exit_and_input_driven_reentry`执行C：初normal非零、随后真实Fx反向，不改state；`test_exact_grazing_reentry_prefers_patch_face_axis`用spec R2的3倍XZ/X4精确边界，held/terminal零，normal为patch-face；`test_reordering_and_shared_ancestor_are_equivalent`检查域/triangle/patch顺序、反向、刚体frame与共享ancestor。
- [x] **7. 规定窗口仅完整一次。** `test_two_jaw_window_halving_and_rebuild`：CPU T.5 h1/8、1/16两mu、每档2重建；C T2+T1、两h和2重建。保存每step独立oracle/预算而不是只末图。首次测试即GREEN如实写，不制造RED。若产品修复只重测受影响窗口，已覆盖未改的版本按delta解释复用。
- [x] **8. 真实执行和事务RED/GREEN。** `test_late_failure_rollback_and_retry_fresh`参数native/path/terminal/mapping/CCD/close × 第2/末子步；`test_best_does_not_hide_execution_or_unknown_failure`、`test_terminal_pre_replay_matches_full_selected_held`、`test_anchor_capacity_and_geometry_rebuild_budget_fail_closed`、`test_partial_construction_closes_workspaces`、`test_all_actual_trials_count_once`。真实执行错误立即整帧失败；只有已定义有限物理bad candidate按原合同允许完整best回验。
- [x] **9. 实际后端关键功能。** `test_actual_two_jaw_support_and_slip`，`test_actual_three_corner_impact_and_material`，`test_actual_partial_overlap_lifecycle`，`test_actual_thickness_window_and_rebuild`，`test_actual_failure_and_all_trials_count`。interpret→CUDA严格串行，`--require-tyx`且backend标签/FP32提升断言。设备A T.25两h两mu与重建、C h.25整窗与重建；normal负控制及后段故障保持。没有Graph/性能声明。
- [x] **10. 最终受影响集合及交付。** 合并同源已过证据，按下表补直接consumers；ruff/format新模块和tests、source smoke、publictyxboundary/diff；记录source/tests/script逐文件hash、每命令/rawexit/精确nodeids/RED和初测GREEN、版本delta及未跑理由。小步commit到专用branch，报告Task1并释放数值/index，不push。

## 影响与选测清单

基线已交付finite/FK 142source及offset/native/replay旧合同。新代码调用旧端口和finite terminal，不改其数值底层。以下具体nodeids存在后按完整名称运行；若文件/函数重命名需先更正计划，不能以`-k`空选冒充通过。

| 集合 | 明确文件/nodeids | 理由 |
| --- | --- | --- |
| 新host | 上述4个非tyx文件全部 | 完整新geometry/mapping/自动law/窗口/事务一次 |
| 旧offset | tests/test_offset_port.py::test_public_api_and_independent_unit_triangle_impulse；::test_rigid_translation_rotation_and_virtual_power；::test_shared_nodes_cross_response_and_single_aggregate_candidate；::test_tiny_lost_offsets_cannot_create_a_couple_at_identical_actual_points | 新厚度映射直接使用者 |
| 旧有限geometry | tests/test_finite_patch_geometry.py::test_patch_schema_and_canonical_identity；::test_exact_fraction_midpoint_cannot_create_false_violation | schema复用/精确算术前例不能回退 |
| 旧terminal真实 | tests/test_terminal_impulse_tyx.py::test_actual_terminal_analytic_and_original_material_ledger_only_accepts_pre | 两相位native生命周期及pre ledger |
| 旧friction真实 | tests/test_dynamic_robot_friction_tyx.py::test_actual_support；::test_actual_material_impact；::test_actual_all_native_trials_counted_once | 原split law/native组合/计数边界 |
| 旧事务 | tests/test_dynamic_robot_friction_transaction.py::test_late_failure_full_rollback_and_retry_fresh | 不改变公共回滚合同 |
| 新设备 | tests/test_thick_robot_tyx.py全部，interpret/CUDA各一次 | 两场景、完整必要短窗、实际FP32、失败 |

建议命令模板（实际新basetemp必须先核对完全新建）：`.venv-py311/Scripts/python.exe -m pytest <明确host集合> -q --basetemp=artifacts/p65-thick-friction/<fresh-tag>`；实际设备用`.venv/Scripts/python.exe -m pytest <明确device集合> --tyx-mode interpret|cuda --require-tyx -q --basetemp=<全新路径>`。每执行块保存stdout/rawexit/source manifest，不复用pytest自动清理旧路径。父不与child并发数值，不重复child最终同hash集合。

不跑旧长材料mu窗/旧normal长窗/两CPU全集/完整CUDA/900：旧共享实现和pin不变，已有合格版本证据。若共享修改或影响不明则更新影响范围并扩测。Linux/第二Python由现远程完整矩阵承担；本slice新映射C++一次短程和新wheel由父完成，不复跑已知TYX003长期。

## Task 2: 父核对、一次fresh review、工程交付

**Files:** `docs/benchmarks/p65-thick-friction-acceptance.md`、engineering.json、review.md及progress/continuation/framework；本阶段SDD报告。证据路径已创建，审阅与工程状态分别记录。

**Interfaces:** consume Task1源码/测试/原日志/摘要；produce明确支持矩阵、唯一review闭环、源码外wheel、准确远程交付。不是H1/P6.5完成。

- [x] 逐文件核对hash、diff、原式和rawexit；复用同冻结快照通过证据，不把旧版本设备窗说成新hash全量。
- [x] 一次fresh全阶段review，限定BASE冻结合同..Task1最终产品；问题交原实现者先精确RED再修。父核对关闭原findings，不派第二次review。
- [x] 所有数值进程结束后，最终冻结digest跑一个实际C++三角厚度短例、build、源码外新nativepip完整installed smoke及wheel逐文件匹配；不跑正式900，不宣称P95。
- [ ] 更新框架/案例两线，commit/push核对SHA和该提交准确CI。随后一般机器人/弯曲自碰撞等继续，整体门槛不删。

Task1报告 f2d4999 已交付父，数值/index释放；原版本 host60/严格窗口与最终 interpret13/CUDA13 的摘要、退出码已核对。唯一 fresh review thick_friction_final_review 已完成 BASE6e818f5..HEADf2d4999：0Critical/2Important/0Minor；I1/I2已在e8c821d/33ec9ea经原实现者定向TDD修复，父核对闭环，不再fresh review。review/修复框已勾；父工程三步已exit0且153源码wheel匹配，工程框已勾；准确推送/CI交付框未勾。

## 准确产品 CI 阻塞（2026-10-29 21:32 UTC）

产品 `5150c3e` 已推送并核对远程 SHA。准确 CI `40950637127` 已结束为 cancelled：8 个 CPU 矩阵任务均由 GitHub 确认超过现有 30 分钟时限，required 因 CPU 未成功而 failure；Documentation `40950637280` success。不能计为产品交付通过。2026-10-30 用户明确选择“保持 30 分钟，另行优化耗时”。90 分钟建议未应用并已撤回；完整矩阵、测试、required 和 fail-fast 保持。已以0065200完成一个厚度精确算术文件的等价优化，报告d2cacff；受影响host96、actual interpret/CUDA各9、224条同次实际轨迹审计及新源码wheel153逐字节校验均通过。三个相同A负载的新进程约5.9s，对照原版约12.3–12.6s，仅为该负载成本观察。接下来推送优化版本并取得新head准确CI；原30分钟配置不变，不提前推进下一生产合同。
