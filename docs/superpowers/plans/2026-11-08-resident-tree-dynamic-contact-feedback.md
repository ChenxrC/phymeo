# C8 完整驻留树内双动态接触反馈

状态：**FROZEN / TEST_AUTHORING / PRODUCT_PENDING**，2026-11-08。持续任务，无截止。父产品876ae848、C7接受1ce9ced为基线；[源码与Newton1.7调研提案](../../development/p65-resident-dynamic-contact-next-proposal.md)已导入3045acc。本计划覆盖提案中的接口选择与未定项，不把提案或手推当实测。

## 可复用主线和边界

在同一world、同一真实完整fixed-root ordinary/affine tree或forest内，有限sphere/capsule/unrounded convex hull的不同非相邻动态group互碰，并可与原dynamic/static同一步共存。仍mu=0、discrete_entry、HOST几何/法向律+TYX全N mechanics/原K投影求解/唯一发布。一个world只有一个完整物理模型；不跨world施力，不拼接独立world，不把完整树拆成若干单关节。

复用C6/C7逐侧COM wrench、完整J、bias、PD、原P/c/C图、双bank/epoch/readonly来源和保存格式。先加两侧临时full N wrench，再一次推进原物理模型；affine保持原图完整反力，即使Pᵀg_contact=0也不能删full力。真实static_reactions只放static，双动态不能伪装固定反力。材料/作用点两种有限步位移、功、entry power与KE分列，不加严格守恒门。

同world多独立participant、浮根驻留、持续摩擦、CCD、全GPU窄相另排，已有HOST共同场景能力继续复用；本片不许诺Newton吞吐优势。参考资料只读本机pin Newton1.7.0/Warp1.17.0，准确来源见提案6。

## 冻结接口和过滤语义

- TyxWorldBatch.prepare_collision / prepare_contacts，以及对应plan constructor增加keyword-only **tree_dynamic_pairs=False**。必须是builtin bool；默认False保旧域拒绝行为，不改变旧采样/几何/求解。True是注册时冻结能力，query/step不能热改。
- prepare_contacts向原基类传相同参数，sample_tree_contacts内部同名keyword默认False；两侧皆dynamic仅True且group不同才允许。static/static、同group在HOST桥仍拒绝，不靠力抵消。复用原逐侧loop，不加第二次推进。
- True时先保真实same-world/same-body、group/mask和explicit excluded过滤。**已由same-body、mask或explicit excluded过滤的pair无需law，也不得出现在materials map**。不同真实group但body相同、一个group冒两个body等继续由既有binding registry拒绝。
- 对未过滤dynamic/dynamic，仅两侧均同一实际ordinary或affine spec的tree binding可进入域；scalar/slider/rotor借flag不升级为多body。不能混ordinary和affine的不同world/storage为同一world。
- 未过滤直接父子仍不支持：根据实际full physical parents判断，报明确能力错误，调用方须用excluded或mask明确关闭；不按名字/索引相近/P或C图猜邻接。两个parent=-1的根不相邻；共同祖先但非直接父子的分支可碰。
- 所有允许pair复用SpatialPairs与实际候选，materials必须完整且恰覆盖过滤后的域。flag不能只改变HOST期望而设备pair仍排除；宽相结果仍只叫candidate。
- 新enabled receipt.diagnostics增加tree_dynamic_pairs=True说明注册能力；默认False的旧receipt字段不扩写。全部具体owner、source、graph、tick和sites仍是权威数据。
- 同world全部动动+动静站点只计一次帽。超帽整world的N/K/FK/energy/held/history/tick/旧receipt保留，邻world可推进；ordinary7/affine9不改。candidate帽/未知几何/晚异常全batch无swap。

## 预注册物理案例

提案5的全部D01–D08输入与独立手推结果原样冻结，补足：
- 所有未指定惯量取full物理body的isotropic diag(1/8,1/8,1/8) kg m²。D02 root diag(1/4,1/4,1/4)，两child各diag(1/8,1/8,1/8)；D03分别diag(1/4,1/4,1/4)、diag(1/2,1/2,1/2)。这些是单独质量声明，不从碰撞表面反推。
- ordinary D01/D02/D03和affine D04/D04Z/D05全部实际40步，h=1/1024；逐步由独立full-component动力学或已写明闭式公式生成预期，不调用产品M/J/解作为oracle。D02真实共享R祖先与两个P子分支必须保完整cross M/b。
- D04Z同时做静止和独立速度1/8：full正载±2N和lambda=-2都必须出现；不能只看状态不动。D05 chain/flatten各自原行反力和wrong-graph restore独立保留。
- D06球/胶囊轴部/有限端/miss、盒face权重；D06H复用原CC线和10顶点非盒HH面几何，各自双动态载荷首步及至少两连续步，不重复已接受A10十组几何套。
- D07共同/接近/分离初速和d=4分别2/3/1N；same-body、mask、explicit excluded、父子未排除、law与排除冲突、共址跨world、两个真实pair/相消力矩。反序shape/binding登记与共同刚体坐标变换不得变owner或动力学预期。
- D08 W1/W4 keys逆序且不连续，ordinary D01、affine D04、旧static contact、未绑定free sphere共40步；恢复tick19后21步完整重放，全部state/geometry/receipt逐字段同。与W1同物理结果，计数不假同。
- 两bank容量/候选帽/实际第二scratch分配、上传、候选及receipt晚故障/同tick restore/owner错误/关闭重建按新增双动态域测试；无第二推进、另一动态body不能偷走一步。补PD delay0/1、低cap只夹motor不夹外接触的短对照。
- 工程门保atol5e-6/rtol5e-5、40步atol2e-4、原M2e-5/PD5e-5、材料记录1e-10与stored R5e-5；不改已声明k/h/质量/阈值迎合输出。严格材料字段一致性与物理精度分开。

所有实际window保存原输入、full M/b/J/载荷/约束诊断、双方materials/sites/功、原控/历史/接受/拒绝与真实计数；failed保存最后accepted及具体错误。保存预配身份，不按输出挑样或只保存摘要；审计不重跑producer。

## 开发/验证顺序与唯一执行

1. 原RD先author/commit **D01公开入口有效RED**：原full模型、HOST双动态patch与±2N独立前提成立，旧API缺能力处失败，不能用新bridge提前失败作假RED。
2. 父读取首test，唯一串行fresh guard执行短RED；父修改下列4个OLD接缝；同节点interpret/CUDA验证。
3. RD继续其余NEW多案例/事务；robot并行NEW独立Newton/公开wheel消费脚本。都只text/Git作者，无import/collection/数值/pytest/lint/format/build。
4. 父阅读交接、精确scope/来源，执行新专项interpret/CUDA。已过首节点只排其同源对应，不重复全套；失败只修普通实现/明确测试错误，保首日志。
5. 父选择少量真正受pair域变更影响的旧宽相/动静/bridge对照，不重复C6/C7整套。全部执行严格串行。
6. 独立Newton D01/D03首步与40步CPU/CUDA：真实两个动态body/joint/shape、原生碰撞/载荷通道；保存实际contact margin/normal/两侧owner/COM力矩与状态。先区分原生检测/本构观察和同律外载integration；不能把Newton逐site刚度与Phy归一patch冒同律。没有明确可比字段则标不对齐，不改本引擎fixture。
7. 同次保存审计、新wheel源码外两backend的普通/affine/动静混合窗口、拒绝/恢复与423+实际成员来源核；结束后才C8限域接受。不能以firstpair绿声称整个Newton功能对齐。

## 所有权

父唯一OLD：src/phymeo/backends/_resident_collision_plan.py、_resident_contact_bridge.py、_resident_contact_plan.py、tyx_world_batch.py；计划/continuation与最终验收由父。原M/PD/affine/geometry数值核保持；发现必须改变核心方程时先给具体矛盾再裁定，不能暗改。

原RD原隔离树NEW且仅：
- tests/scale/test_resident_tree_dynamic_contacts.py
- tests/scale/test_resident_tree_dynamic_contact_transactions.py
- docs/benchmarks/p65-resident-tree-dynamic-contacts.md

原robot原隔离树NEW且仅：
- tests/reference_newton/robot/resident_dynamic_contacts_reference.py
- tests/wheel_consumers/resident_dynamic_contacts.py
- docs/benchmarks/p65-resident-dynamic-contacts-reference.md

两原agent可读父当前源码/本机Newton/已接受reference，不能merge父或改OLD、跑任何项目命令、安装、建helper或新agent。首次D01后各小commit，最终clean与AUTHORING_RELEASED；不预写pass。父可直接继续已授权开发，无逐阶段用户许可。

## 首节点实现与实际执行（2026-11-08，完整矩阵仍待验）

父产品13c3fad6ec60ff5e411c9fbf2dda7d7d78d5dfe0只改四OLD薄接缝。注册编译真实full physical parents，flag默认False；True用原world/body/mask/excluded规则并显式拒未过滤直接父子。registered_pairs沿原物理域登记语义，可包含被mask/excluded的非相邻域对；真正材料覆盖与候选以SpatialPairs过滤后allowed为准。原逐侧wrench loop现可接双方dynamic，不动M/PD/affine核心核。

- 原RD first6f88ffa的394先在kp整型storage前提失败（1 failed/.65s），不是有效RED；e1c49d只把三np.full显式dtype=float，数值不改。
- 395有效RED（1 failed/1.24s）：完整双P森林、设备FK、HOST有限patch及独立±2N均成立，准确停在公开tree_dynamic_pairs能力缺口。
- 401 interpret 1 passed/1.28s、402实际CUDA 1 passed/3.11s：双侧完整wrench、a=(-1,2)、净自由方向动量、一次tick/共同发布成立。
- 396四产品首Ruff0，397首format1，398格式化，399/400最终Ruff/format0。
- 403两项旧bridge默认拒绝HOST2 passed/.47s；404受改变影响旧ordinary默认pair拒绝、affine默认pair拒绝、A02实际静态接触CUDA3 passed/4.47s。不重复旧C7/C6完整套。

所有命令、before/after源码和首失败原样保存在artifacts/p65-generic-collision-20261107-01。当前只有首D01与必要兼容通过；D02共享祖先、affine原图/零K载荷、40步/过滤/多形状/事务、多backend完整矩阵、Newton参照、保存审计及新wheel仍待验收。原RD/robot继续NEW作者；父仍唯一执行owner，没有C8正式接受或P6.5完成。
