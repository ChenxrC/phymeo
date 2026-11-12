# M-B1 多独立刚体与变形表面的共同响应

状态：IMPLEMENTATION_GRANTED / TESTS_PENDING。2026-11-13 04:52 北京时间。
依据 [后续设计](p65-multibody-deformable-next-design.md)，本批只实施 M-B1；M-B2 关节树另行验收。功能优先，06:15 后不启动新功能。

## 冻结的功能与边界

同一 logical world 内 1..8 个 ordinary floating 单 body participant 与一个原 D1 node model。全部 participants 总数 <=8、worlds<=8；其余原节点/面/site<=128/内存/层定律/精度限制不变。所有 world 均至少一个 participant，所有 participant 即使零接触也真实推进。原 sphere/capsule、每 facet 离散 NormalLayer、mu=0、entry sampled split，不新增刚体互碰、关节树、CCD、摩擦或稳态支撑承诺。原 D6 可选模型不改方程/参数。

扩展的是稳定地址和力转置，不新造求解器：每个 round shape 绑定原 participant/group0/FK row/真实 source token，接触反作用送回该 participant 的原 (1,6) COM wrench，经既有 J/M/共同事务推进。相同 body 多 collider 累加，零接触清零，不得 world 字典覆盖或交错 body/world 写入。

## 固定内部接口

NEW backends/_deformable_response_layout.py：
- compile_deformable_response_layout(participants) 纯静态返回 frozen DeformableResponseLayout。
- worlds: canonical sorted unique world tuple；participant_keys: sorted ResidentParticipantKey tuple。
- world_participants: readonly Mapping[world,tuple[key,...]]。
- participant_slots: readonly Mapping[key,int]，按 participant_keys 排序的连续 body 槽。
- body_offsets: immutable bytes-backed np.int32[W+1]；每 world 占独立连续区间。
- multibody: 任一 world participants >1。构造检查全 key 唯一、(world,body_id) 唯一、原 family floating/body_count1、总数1..8。不做 Runtime 或项目 probe。
- 不依赖可变 accepted state。所有 original source/model/topology gates仍由原 scene/adapter 负责。

NEW backends/_tyx_deformable_multibody.py：
transpose_multibody_response(worlds, sites, offsets, body_offsets, identities, barycentric, levers, forces, node_force, body_wrench, status)。
语义与旧 transpose_response 完全对应，但 identities[:,4] 为 body slot，须在所属 world body_offsets 区间；按 world 顺序串行归约其 site，完整清零所有 body wrench；错误 slot 必须置 status0 且不越界读写。保原运算次序/finite gates/节点验证。原单体核不改。

_ResponseEngine 仅 multibody 分配额外 body_offsets，并用新核；原一体路径/数组/核/字节数保持。
扩展 buffer bytes =116*max(1,capacity)+12*N+24*B+4*W+8*(W+1)。原一体旧公式不变。预分配前两级预算都核，不冒总 RSS cap。

## 公开证据与兼容

只有真正 multibody 的 receipt schema = tyx-deformable-multibody-contact-step-v1；单体继续 tyx-deformable-contact-step-v1，原字段形状不变。
每 site 增 participant:ResidentParticipantKey、body_group:0、body_slot:int；每扩展 world.rigid_participants 为完整 readonly Mapping[key,record]，record 含 rigid_wrench_delivery、rigid_wrench、uploaded_rigid_wrench、rigid_external_entry_power、body_slot。扩展 world 不提供误导的单一 rigid_wrench/delivery/uploaded 字段；rigid_external_entry_power 为所有 participant 原外载入步功率之和。rigid_publication 原完整 entry/candidate/diagnostics/projected 保持，受力/未受力均在内。layer work 的 rigid after 必须取 site 指定 participant 的实际 candidate transform。
checkpoint仍原 v1 owner-bound 且binding包含完整scene布局；restore/reset选择完整logical world，不能部分participant恢复。控制、source变化、最后participant失败、near UNKNOWN、非法身份/容量都保持旧accepted两域和旧receipt。同步/释放UNKNOWN沿原close-only。

## 任务所有权

父：tyx_deformable_contact.py、builders/deformable_scene_lowering.py、所有既有共享接缝与集成/执行/报告。
原 source agent：NEW _deformable_response_layout.py + NEW _tyx_deformable_multibody.py，只纯编码。
原 audit agent：仅 _deformable_response_source.py，原 source gates/全部world participant tick/每moving actualFK均保留；只纯编码。
原 tests agent：NEW tests/collision/test_deformable_multibody_contact.py 与 NEW tests/installed/check_deformable_multibody.py，tests-first/独立力学/故障/公共入口；只纯编码。
不改旧 tests/fixtures/求解/normal law/D6/kernel，不运行import/collection/数值/pytest/static/formatter/compile/build/CI。每作者独立worktree小commit，明确CODE_READY_TESTS_PENDING并释放。父唯一serial执行者。

## 固定案例与验收

编码时固定案例参数后不按输出扫描：
A：两个不同质量/真实来速的球接同一片不同位置，至少一接触有非零COM力矩。
B：球+斜胶囊的短连续入层/离层，另一个稀疏world不同事件；完整零接触participant仍推进，无残余力。短轨迹不冒旧持续稳态。
E：输入排列/非零anchors/两稀疏world；keyed相同；非法shape-owner/source、body alias、总participants>8、site与native预算拒绝；最后participant latefailure无半发布，retry对同声明独立owner；query/control失效、partial reset未选完整raw不变、restore真实后继。
每步独立重建 normal law、节点散射与每 participant COM wrench/resultant/moment/power，并核实际 projected force/candidate运动；保完整原node material/damping trace与两域共同tick。不能只有合力守恒或helper自证。
按现有D3/D5工程门比较，精度独立记录，不调h/质量/层参数/阈值以换pass。
源码HOST+interpret+CUDA后，以同一测试consumer从源码外wheel的高层 ParticipantSceneDeclaration/ParticipantDeformable 编译两刚体案例，实际多步+query/control/restore/reset/完整provenance。
保留一组原单体D3/D5受影响兼容检查，不重跑无关已绿组。

成功是多参与者身份、载荷、运动、事务和公开装配贯通；不是M-B2/任意机器人/稳态/性能超过Newton。两轮同因失败且无法缩小则保最小反例停止该节点。06:15冻结后不扩功能，完成已有验证/发布收口；若未验完如实标未验，不伪接受。

## 执行前案例手推修正，05:05

B固定16步模板：world7 mass2 sphere外载+64N保持；world31 mass1 capsule外载改+32N，使两者同为32m/s²。初拟+64N在h=1/1024、v=-1/8、初gap=width+1/16384下，semiimplicit前两步只到width，不具备原声明的入层事件前提。此发现与修正均在任何新M-B1项目测试/数值执行之前；旧+64N仅未执行作者草稿，不是失败运行。其余几何/质量/真实来速/h/16步/门不变，修正后不按输出调参。新增新核合法累计、跨world slot、越界slot、零site清零两个backend节点，合计预计26节点，以实际collection为准。
