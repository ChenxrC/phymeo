# MuJoCo-TYX T4：混合摩擦约束与 G1 支撑实施计划

2026-11-10。用户授权：先上传整个已提交版本，再继续下一批，随后验证 G1 持续支撑。远端 [codex/dev](https://github.com/ChenxrC/phymeo/tree/codex/dev) 已固定当前整合版本 73a85f8a；后续开发继续在父 codex/p65-parallel-modules，验收后再按授权增量同步。原三个 K1 未验收草稿保留本地，不上传缓存或大证据。无 remote CI/main merge。

## 1. 已完成调研与真实前置

采用已全文审阅的[固定源研究](../../development/mujoco-tyx-t4-research.md)、[关节/G1设计](../../development/mujoco-tyx-t4-constraint-design.md)、[多案例预登记](../../development/mujoco-tyx-t4-cases.md)。CPU MuJoCo3.3.7为固定语义，Warp0.0.2用于设备组织参考。两者在平宽solimp、R clamp顺序、双侧limit上不同，保CPU语义，不盲目移植。

父唯一执行01只编译未改的g1_primitive_actuated.xml，RAW0：nq36/nv35/nu29/body31/geom34、502possible pairs、942potential contact slots、9420potential pyramid edges、29limited hinges、4excludes；pyramidal/Euler/h.002/source iterations100。该文件SHA f7e856e057a095a9f8708d302531d40ff2519440012c24488407ae8d0f0586aa。编译不是物理通过。原mesh部署模型另有mesh/cylinder与joint-wide actuator clamp，不冒称迁移。

选定共同支持 condim1/3/4/6（边数1/4/6/10）、scalar上下限位和DOF干摩擦。不降G1 condim，不改原质量/几何/阻尼/初态/电机/碰撞过滤。无elliptic、explicitpair、equality/tendon/ball limit、native sensor或mesh/cylinder扩展；不影响旧T0–T3受支持域。

## 2. 通用布局与算法

新增T4域cap：G128 / static pairs1024 / static geometry slots2048，detected contacts最多128，actual EFC最多512；用户较小runtime capacities仍生效。旧T3 P16/S32契约不倒改。通用exclude_signature按native body过滤；全部potential pairs遍历，不能只选脚。全静态几何输出之后，device稳定压紧检测contact和实际active EFC；inactive检测contact仍完整保存，efc溢出整批拒绝，不截断前512行。

共用当前完整FK/M/b/J和已验九种primitive几何；单个world仍一个free root+0..29scalar或fixed1..8scalar。新T4显式joint约束模式解除旧“必须严格在limit内部”的域限制，有限/坏状态检查保留；不可夹q/v。

共同box-QP：A=J M^-1 J^T+diag(R)，b=J a0-aref，a0=M^-1(applied+passive-bias)。normal/limit/pyramid力下界0、无上界，dry force∈[-loss,+loss]。新算法明确命名pyramidal-box-active-set-v1，不冒nativeNewton。每face用M+J_F^T D_F J_F解nv维系统并恢复f_F，保所有行和交叉项；不构造9420平方或EFC平方factor。

零力可行冷启动，dry内部行初始free；到face最优后块释放KKT违反的bound行，完整可行线搜索绑定所有同时碰界者，内部face修复不立即再次释放。每次分解计一次source iterations，1..512；重复工作集且力不变/无进展、预算耗尽、非正pivot、非有限皆失败，无正则修复或fallback。100次不保证所有模型收敛，保失败证据。

独立最终重算physical M关系。归一KKT采用明确新尺度max(1,max_i(|aref_i|+Σd|J_id*a_d|+|R_i*f_i|))，tau=max(source tolerance,2e-5)的实际FP32值，另保raw residual/force feasibility/mass residual。测试侧仍独立构造小H/b并核原预登记5e-5门；两尺度不伪称相同。Euler单独使用M+hD，约束问题使用物理M。

## 3. 冻结 ABI（本节覆盖三份研究的草案字段）

W世界，V=max(8,nv)，S静态geometry slots；C=max(1,用户contact cap)，E=max(1,用户efc cap)，要求请求cap≤128/512（默认64/256保留；G1显式128/512）。实际计数与预分配维度分开。全数组输入/输出先清理，不读旧成功步残留。

静态：
- friction_ids[F] i32 native DOF；friction_law[F,9]=loss,invweight0,solref2,solimp5。
- limit_ids[L,3] i32 joint,qposadr,dofadr；limit_law[L,11]=low,high,margin,invweight0,solref2,solimp5。
- pair_condim[P] i32；pair_friction[P,5]，native MINMU clamp后；pair_law[P,16]沿旧geom字段含mjMINVAL/MINIMP/MAXIMP。
- 保全slot_ids[S,4]=pair,feature,geom1,geom2及当前geom/body对应，不按当前接触删static身份。

新runtime：
- efc_ids[W,E,5] i32 = native_type,source_id,side_or_edge,geometry_slot,dof。native_type用1(dry DOF)/3(joint limit)/5(frictionless contact)/6(pyramid)。contact source_id为本次compact contact序号，geometry_slot保原static slot；不适用值-1。
- efc_data[W,E,14] f32 = pos,margin,velocity,aref,R,D,imp,lower,upper,force,gradient,active,finite_upper,diagApprox。normal/limit/pyramid的upper存0且finite_upper=0，绝非零上界或大数假无穷；dry有限upper且mask1。所有保存标量有限。
- efc_J[W,E,V] f32，counts[W,3] i32 = detected_contacts,active_contacts,actual_efc。只有active native行进入EFC，顺序dry DOF、joint lower/upper、contact static slot/native edge。
- contact_map[W,C,3] i32 = static_slot,first_efc,row_count；inactive first=-1/count0。
- contact_J[W,C,6,V] f32 = frame投影(n,t1,t2)线速度J和(n,t1,t2)角速度J；inactive contact同样完整。
- contact_wrench[W,C,6] f32；native解码全部edge force，unused分量0。device decode，spin/roll是纯couple。
- 原normal_rows[W,S,32]/normal_J仍可作geometry兼容视图，但本域force真值是efc_data/decoded wrench，不能把旧normal-only R/force当新结果。
- solver diagnostics[W,10] f32 = face_solves,normalized_kkt,absolute_kkt,bound_violation,actual_efc,free_rows,converged,budget,normalized_mass_residual,kkt_scale。
- solver workingset[W,E] i32 lower0/free1/upper2，previous_workingset/previous_force、trial_force等scratch只在本attempt内使用。

NEW _constraint_rows.py：
assemble(count,nv,nslot,nfriction,nlimit,contact_capacity,efc_capacity,h,impratio,slot_ids,geom_ids,pair_condim,pair_friction,pair_law,friction_ids,friction_law,limit_ids,limit_law,q,v,motion,body_jacobian,normal_rows,efc_ids,efc_data,efc_J,contact_map,contact_J,counts,status)。
decode(count,nv,contact_capacity,efc_capacity,slot_ids,pair_condim,pair_friction,contact_map,counts,efc_data,contact_wrench,status)。
q/v双bank需独立kernel handles assemble/assemble_other，与原TYX模式一致。motion COM位置沿原[w,body,5,:]，完整body_jacobian沿[w,body,6,V]。assemble独立按固定源重算全部K/B/imp/R/D/aref/diagApprox，包括condim1，不依赖原normal-only近似参数。所有多余数据槽清零，unused map的slot/first=-1而row_count=0；超cap先置status1，不越界，继续记录实际所需count；数据错误status2。

NEW _constraint_solver.py：
solve(count,nv,efc_capacity,budget,tolerance,mass,bias,passive,applied,efc_data,efc_J,counts,physical_factor,mass_work,smooth_acc,face_matrix,face_factor,face_rhs,face_forward,trial_acc,trial_force,working,previous_working,previous_force,constraint,total_applied,physical_acc,diagnostics,status)。
matrix/factor[W,V,V]，其余dof向量[W,V]，row向量[W,E]。更新efc_data的force/gradient，constraint=J^T f，total_applied=applied+constraint；physical_acc是M物理加速度，不是Euler integration_acc。状态4=massfactor、5=stall、6=facefactor、7=budget，其它沿通用非有限2/capacity1。solver不做几何/primitive分支。

公开：原runtime入口不变；source结构静态选择T4（摩擦、limit/frictionloss、excludes或超旧候选域），不以失败后fallback选择。capabilities明示pyramidal维度/caps/soft limits/customsolver/CPU观察边界。last_step.contacts保持既有通用字段并有6D wrench，jacobian为完整6×nv；新增last_step.constraints tuple提供逐EFC typed immutable行（type/source/edge/static slot/dof/J/pos/margin/vel/R/D/imp/diagApprox/aref/force/gradient/lower/optionalupper）。raw_contact_fields保新全部数组、counts、workingset及diagnostics，不承诺不存在的resident dual matrix。旧receipt/oldnormal路径不改义。

## 4. 所有权与执行

- 原source作者：仅NEW _constraint_rows.py与NEW source-report，含参数/limits/frictionloss/6D J及decode。禁止改父接缝。
- 原audit作者：仅NEW _constraint_solver.py与NEW solver-report，通用box算法和独立手算检查，禁止几何分支。
- 原tests作者：NEW tests/test_mujoco_tyx_constraints.py、tests/test_mujoco_tyx_constraints_protocol.py及更新原新案例doc。按已登记A–E全矩阵，公共ENV由父独立文件负责。
- 父：NEW _constraint_model.py/_constraint_device.py、原_model/_tree_model/_floating_model/_mujoco_tyx_runtime共享接缝、NOTICE；必要新tests/test_mujoco_tyx_constraints_env.py与examples/g1_mujoco_tyx_policy.py、G1专项目录。不覆盖K1 WIP。

三作者隔离并行纯编码/文本检查/小Git，无任何import/collection/probe/test/lint/format/build/install/profile；父另行唯一串行执行。交接前文档明确CODE_READY_TESTS_PENDING与提交，不能作者自行跑小脚本。只沿用这三位原agent，禁止再派helper。

父先精确源码/环境快照、fresh entrypoint guard、首次lint/format原始日志，再interpret通用组、CUDA通用组；保存首失败，只针对原因修复，再跑必要受影响组。序列化保存/审计也是同槽，当前只有01compile-only完成。旧T3b已绿组不重跑，只有被T4接缝实际影响的相邻组选取验证。小提交按feature-commit-checkpoint/ce-commit完成，无重复goal或周期任务。

## 5. 验收顺序与结束物

- [x] 当前整仓已提交版本推送codex/dev，remote HEAD核对。
- [x] 固定源/真实G1/cases研究；原G1编译布局，不运行物理。
- [x] 父冻结完整ABI、所有权及预登记工程门。
- [ ] 通用rows/solver/parent runtime实现与源码审阅。
- [ ] A–E解释和CUDA、全部fixed案例/容量/事务/公共ENV通过并保存原失败。
- [ ] G1 F0准入、F1三个真实姿态同入步；不得删502pairs或改condim6。
- [ ] F2原策略2秒，h=.001/.002；CUDA完整窗，解释短专项。F3原h=.002、±x方向.1mg/[.5,.6)s扰动各2秒，同资产CPU原样参照。
- [ ] F2/F3通过后原h=.002同一原策略前20秒；原.002无扰动轨迹可在2秒检查后保持同一状态继续，不reset伪连续。全部131.48秒舞蹈留独立后续，不冒20秒全动作。
- [ ] 新wheel源码外interpret/CUDA、证据和架构/续作入口、小commit；已验增量同步codex/dev。

工程门以案例文档§4/§7为准；上面调整只消除了重复大矩阵与冻结字段，不调物理阈值。失败保first-state/rows/force/native对照，先分geometry→J→law→solver→integration→policy；两轮无新能力/失败闭环/测量就停止该死支线重选最小差异，不扩大纯证明。T4成功不是mesh部署scene、椭圆cone或整体Newton对齐。

## 6. 首次真实联调与数值实现修正

父14前已完成共享模型/设备/typed publication 接入（另NEW _constraint_publication.py，父所有）。同一native EFC问题不变：08首ENV在import因TYX purefunc只能单表达式拒绝，父将命令式impedance inline；09 solver分支局部value未预声明，父显式初始化。10/11实际face求解在6次后FP32停滞，normalized KKT=4.8445989e-5，原门2e-5未过；完整failed-device保存。12只分析该保存数据，物理M残差2.54e-8，dual条件数约4268；没有参数搜索或物理重跑。

父将同一face最优问题改写为增量式：当前实际f的physical a和g全部重算；H=M+J_F^T D_F J_F，H*delta_a=-J_F^T D_F*g_F，trial_f=f-D_F*(g_F+J_F*delta_a)，固定边界不动。独立原solver作者纯文本复核等价。trial_acc现在是delta_a；每次分解计原budget，最终physical M/KKT仍独立重算，既不改R/摩擦/资产也不放宽阈值。该改写减少绝对力消去误差，不保证任意条件数通过。

13 ENV interpret与14 ENV CUDA各1通过，原两个世界/12步/反馈控制/selected reset/clone下一步。此时只完成首控，不冒A–E完整矩阵或G1通过。raw/首次失败全部位于artifacts/p65-mujoco-tyx-t4-20261110-01。

补充ABI：limit side lower=-1/upper=+1、dry=-1。typed row schema=mujoco-tyx-constraint-row-v1，字段constraint_type/source_id/side_or_edge/geometry_slot/dof/jacobian/position/margin/velocity/reference_acceleration/regularization/inverse_regularization/impedance/diag_approx/lower/upper(optional)/force/gradient。typed contact schema=mujoco-tyx-pyramidal-contact-v1，完整6×nv Jacobian与6D wrench。mass residual归一尺度为max(1,max_i(abs(rhs_i)+abs(constraint_i)+sum_j abs(Mij*aj)))，status5，原tau；全部原量保留，未虚构直接保存rawmass标量。

## 7. 极小摩擦首例与按尺寸选择的平方根 face

23 首个完整短组在 native MINMU(condim3、authored friction=0) 拒步，前6项通过；完整同入步native与failed device scratch保留。实际R=3.529405934e-11、D=2.8333379584e10，所需12.43N力在primal回收式中的余量仅4.39e-10，而相消项约12.43的FP32 ulp为9.54e-7。24仅分析已保存失败：primal条件数约1.7e10；最大违反edge2单行direct dual给12.433907984N，native为12.433901233N，其余三行梯度严格正。没有改摩擦、R或tau，也不把MINMU当condim1。

父据此保留原box-QP，改为每轮只释放最大违反的bound，完成feasible face后才释放下一个。free_count<=nv时直接求dual correction：先用physical M=L L^T得到B=J_F L^-T，从U=diag(sqrt(R_F))开始，用scaled-hypot/Givens逐DOF追加B的列，得到U^T U=R_F+B B^T；解U^T y=-g_F、U delta_f=y。这样避免把微小R先加到O(1)的Gram对角、也不从两个大数相消回收力。超过nv的face仍用第6节primal增量。分支在factor前仅按尺寸选择，不以失败回退；相同边界line search、物理M和KKT独立校验、预算与拒步语义保持。新增free_ids[W,V]；face_matrix[W,V,V]在dual分支存B，face_factor存上三角U，trial_acc临时存delta_f；没有EFC平方矩阵。ABI在previous_force之后新增free_ids。

25针对原MINMU失败的interpret节点通过；26短组前6项通过，signed slide的末步耗散断言失败（逐步native/force/state门此前均通过），正在核实该断言是否错误地把已经离面后的末步当作仍在接触。27其余独立短组正在首次运行。以上不是完整T4或G1验收。
27 其余短组首次71通过/16失败（原RAW1保留）；其中两组倾斜偏心体暴露了global KKT尺度掩盖负bound梯度导致提前停止，稀疏world/retry暴露了完成blocked face但f不变时被误判stall。29只分析27已保存的三份scratch：倾斜体遗漏edge的native力为.0045246/.0224702N；另一个候选free梯度已很小、仍有未释放bound梯度-2.3064。28分析脚本误把padded input_v宽度当nv，碰到零padding导致离线矩阵奇异；29按保存physical M有效块纠正，不是新物理失败。

修复：原normalized KKT门之外，bound方向KKT用max(1,abs(R*f))归一后亦要求原tau，避免大aref掩盖应激活边；同一feasible face无force变化但已完成且另有违反bound时允许下一release，而非宣告stall。数值模型、预算、原global门与原子发布不变。30的8个倾斜/稀疏reset/顺序/篡改重试针对节点全部通过。源作者复核small-dual转置/Givens/索引/预算正确；单边release可能对G1原100预算形成压力，尚待实际G1观察，不增预算或删接触。

同次native证据反驳了部分行为测试假设：signed slide末窗已经离面；±1.2mg外载短窗也有同步分离；部分rolling角速由切向点力消散，原生pure couple为0。对应测试改为同轨迹耗散/真实COM反作用，并保全部逐步native分量门，不能用错误的末步必须接触或每个旋转必须pure couple替代。五个unsupported断言误读异常.code，实际公开为issues中的code/detail。单face预算负控旧fixture实为一个active边即可完成，需保原budget1、以正交独立normal解析负控替换错误假设。上述修订由原测试作者处理，未提前写为通过。
## 8. 初始干摩擦自由面与连续窗闭环

36首次完整解释长组实际4失败/8通过/196 deselected（2499.23s，RAW1）。D02八个有限表面完整一秒窗口全部通过；D01四个接触/限位/干摩擦窗口在初始dry自由行尚未优化时先release bound，形成零步长重绑而拒步。全部34项源before/after相同，原失败保留。

父在任何当前working==free时初始化refine_face=1，先优化初始可行面再释放bound；没有dry行的分支不变。独立文本反例H=[[2,1],[1,1]],b=(-2,-1/2),x∈[-2,2],y≥0说明先release y得到不可行(1.5,-1)，而先解x面得到正确KKT(1,0)。每次face仍计原source budget，物理模型和容差不变。

37/38窄静态首失败保留，39仅排版两个修订测试文件，40 lint0。41受影响短组24通过/184 deselected（53.52s），含原子晚下载回滚、原budget1真正不足、两mixed同入步和ENV。42从原初态重跑四个D01窗口，4通过/204 deselected（416.96s）。D02八组按无dry分支复用36，不重跑；这不是CUDA或G1验收。下一步为43/44首次完整CUDA和原G1。
## 9. G1规模触发的结构性批量释放

43 CUDA短组86通过，44 CUDA完整12个一秒窗口通过；52旧五类共享route双backend共10通过。45仅缺Python3.12 ONNX依赖，57隔离安装原版本后59准入成功；58是快照路径数组错误，项目执行前拒绝。46 G1首个source-default真实姿态因source100预算耗尽拒步：28接触/280 EFC、96自由行、100face，normalized KKT约1.92e-4；完整首失败保留。未改G1资产、source预算或接触列表。

父把完成面的单bound释放改为最多nv个最违反原normalized bound-KKT门的bound，稳定行序打破并列。仅旧free残差满足原scaled门且refine_face==0时允许新批量；blocking后的零步长重绑及缩小面修复先完成，所有face仍计原预算。MINMU冷起nv3首先最多3行，可使用既有QR小面；后续真实过门的行仍不得丢弃。can_release与实际选择使用完全同一FP32除法谓词。无新E平方缓冲、模型标签或失败fallback。

62双backend风险组22通过。原预算负控的正交两normal在batch下一面即可完成，故保原输入改为新正控；负控恢复原deep/shallow相关两normal，完整独立oracle给全自由解第二分量负，必须先alpha0碰界再解第二面，budget1因此真实拒绝。68两正反控×backend共4通过，真实face计数和全批原子性保持；旧single-release下的历史失败解释不改写。

63 G1解释三个真实姿态全部通过，47 CUDA同组三姿态通过。65格式首失败保留、66排版、67lint0。当前48开始原h=.001连续两秒；G1持续支撑尚未验收。最终core稳定后补完整受影响T4矩阵，避免在每个新的G1规模缺口前反复跑昂贵长组。
## 10. 完成当前自由面后再释放新边界

48原h=.001两秒2000步通过；49/50原h=.002正反x推扰各1000步通过。原51二十秒轨迹在17.284秒、8642已接受步后status5拒步。此前最低高度0.668329656m、最大倾角0.367177304rad、全窗平均足载0.999752336mg、非足地反力0，未触发跌倒或行为门。51失败完整保留，不把前缀写成20秒通过。

72只分析同次保存的M/J/R/b及工作集：原27自由行残差约2.6e-4，被全局scale148掩盖，bound row27因此过早释放。独立精化原自由面后，row27梯度由负5.95e-5转正3.83e-5，全部原free force仍正，最大力修正3.30e-4。这是下一面进入条件失配，不改QP、模型或容差。

父用free_kkt=max_free |g|/max(1,|R*f|)作为release和can_release的共同前提，与bound选择使用同一局部尺度。最终global KKT、strict bound KKT、可行性和质量方程门全不变；已满足原成功证书的状态仍立即通过。每次精化计原source iterations，不加预算。原作者已文本复核term生命周期与两谓词。

75公共初态诊断在任何device step前被原FP64 unit quaternion门拒；77尝试合法unit声明不能量化回完全相同四元数，严格同字节断言拒绝，未用近邻输入代替。79/80改为直接构造保存的精确device bank并各advance一次，不调用公共初态修复或发布。两后端均status0/5face/70EFC/27free，独立KKT分别4.56e-8/2.74e-8，全部输入q/v/双字clock、row身份与原失败绑定；同输入原生力/加速度比较通过。这只闭环设备失败步，不是公开checkpoint支持或连续轨迹通过。原公共已发布FP32四元数不能直接作为FP64unit初态的互操作缺口另列后续，不在本修复中放宽门。

能力元数据新增通用mixed_articulated_constraints_supported和floating_mixed_constraints_supported，按实际constraint plan/kind决定，旧g1_supported只作后者的deprecated alias。scope明确admitted-model-domain与trajectory validation分开；无G1名称/资产/策略特判。原通用free-root混合ENV验证这些字段。73lint与74format均0，69完整短矩阵、81从原合法初态无reset的20秒公共轨迹及最终长组/安装包继续验证。