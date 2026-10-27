# P6 唯一整分支独立审阅与裁定

审阅范围f74c7dc..e2a2e18；一次独立fresh-context审阅，未重复派发。独立定向41测试通过，但不是最终阶段验收。判定2项Important、0Critical、0Minor；两项按实际错误接受的影响维持Important。修复版工程和完整数值门禁均已通过；P6正式串行性能和最终提交CI仍待完成。

| 发现 | 影响 | 修复与证据 |
| --- | --- | --- |
| NUM-003误差界遗漏体积直接放大，且host tiny保留FP32标量 | L=3e12、E=1e-25四面体真实内力约0.0433125N，却接受零力；严格errstate另有偶然下溢拒绝 | 以明确FP64算术补Piola/最终梯度直接传播项；普通/严格两项失败后修复 |
| 材料没有运行期独立力/能量门禁 | E=1e6单位四面体刚体转动出现约0.00764N伪内力仍被接受 | 独立按原FP64模型推导体积/偏差能量梯度，只拒绝不替换设备结果；刚体运动及有限错误力/能量三项失败后修复 |

五项新增失败例随后连同既有节点/接触共63项解释器通过；连同实验与记录器共79项实际CUDA通过。另2项非有限输出检查先失败后修复，在解释器/CUDA通过；当前全体65项C++通过，最终实际CUDA全套1287项、两套核心各1026项通过。S2/S3/S4修复版各200帧两遍，共186数组重复一致，并通过同设置FP64全部帧/子步误差门槛。输入数组不变，实际kernel执行、禁止生产CPU force回退的测试保留。完整检查与来源见[验收记录](p6-acceptance.md)。

## 独立审阅明确未裁定的边界

1. 最终完整CUDA、重复和FP64对照：执行者必须收齐新源码证据。
2. 完整C++/CUDA工程及最终远程CI：未由reviewer重跑，不能推定通过。
3. 冻结后正式串行性能：待实际测量，记录器逻辑审阅不等于结果。
4. 优化、峰值内存、传输字节及Warp对比：本轮未测、不承诺。
5. 摩擦/双臂/复杂碰撞/浮动基座/P6.5回放：不属于本P6成果。
6. 恶意生产者重造整套轨迹/计数/provenance：不在可信生产者错误检测范围。
7. loader重新模拟全部节点积分/材料/XPBD：loader只承诺独立SI接触、载荷、固定点、材料域；另以完整FP64对照验收轨迹。
8. 内核中间F逐项域认证：物理候选仍由原FP64参考域检查，量化影响进入新增力/能量门禁；未复现第三独立缺陷。
9. 无界极端尺度支持：可以明确拒绝不安全模型，不能错误接受。

上述每项的执行者裁定与代价均保存在下列完整台账。无延期Minor；没有第二次审阅。

## 实施和最终裁定原始台账

- Ruling: 继续用户已授权大纲内详细设计与实施，不重复询问范围许可 — 顺序与任务类型明确，技术细化属于授权 — 若技术选择偏离真实需求需修订。
- Ruling: 保留FP64主机状态合同但明确FP32设备计算 — P3/P4合同严格且tyx FP64计算缺口已复现，避免破坏CPU消费者 — 使用者可能混淆存储与计算，故签名/报告/文档三处强制标明。
- Ruling: 使用公开tyx串行小kernel与显式同步适配，先正确跑通 — 主机阶段调度受允许，不做设备常驻优化 — 性能可能明显受launch/传输影响，必须记录且不冒称依赖缺陷。
- Ruling: P6全轨迹使用同模型同P6容差的FP64参考，旧P5门槛不变 — 两种精度预算要在实现前固定 — 新门槛证明有限误差范围，不证明原P5 FP64精度。
- Ruling: T3independent node/material slice prepared whileT2C++compilationran — neitherconsumes theother, sameauthornofileoverlap — timingsarenotisolatedperformanceandmustnotbeusedasbaseline.
- Ruling: add _tyx_point_kernels.py fordeviceworld/localtransforms — T4requiresactualdynamicgeometrymathandthisavoids hosttransformfallback — anextrafocusedinternalfilebeyondplanfilelist, publiccontractunchanged.
- Ruling: diagnosticsremainreadonlyowned canonicalown_diagnostics results — sameexistingruntimecontract andstrongerisolation — callerscannotmutatediagnosticarraysinplace; testupdatedratherthanweakeningownership.
- Task5ruling: add protectedbackend/contactfactory seams toexistingArticulatedCouplingProgram — reuseunchangedglobalstage/resourceorchestrationwhileCPUdefaultsremainnumpy — regressionmustshowCPUstillrejectstyxandalloldcasesunchanged.
- Ruling: independentP5/P6audits shareprivateSIequationimplementation withseparatepublicfrozengates — avoiddriftwhileP5API/limitsunchanged — regressionneededtoensureP6cannotrelaxP5bydefault.
- Ruling: prepare T7 independent engineering/performance consumer while T6 full scenes execute — no physics changes during candidate process; final source freeze and rerun will precede acceptance — candidate timing is nonisolated and not a benchmark.
- Ruling: save T5 implementation checkpoint before the long C++ integration batch finishes — 101 interpret and58 actualCUDA already passed; small recoverable commits are authorized, but Task5 remains pending and C++ cannot be marked passed — any later failure requires a followup fix and renewed gates.
- Ruling: dispatch sole final whole-branch review at implemented/source-frozen e2a2e18 while lengthy acceptance runs finish — no further planned implementation remains, only verification/evidence, and review package explicitly lists pending gates — if tests reveal source fixes reviewer may miss them; parent must RED->GREEN and rerun affected full gates, no second review. This preserves one fresh review while fitting user-bounded window.
- Final: Ruling: final fullCUDA/repeat/reference evidence remains unjudged by reviewer — parentmustexecute currentfixedsource,rawprobeisnotacceptance — costifwrongfalsecompletion.
- Final: Ruling: fullC++/CUDAengineeringandremoteCI remain unjudged — collectactualexitsandpreciseSHA;stoppedrunsnotpassing — costifwrongplatformregression.
- Final: Ruling: formalserialperformance remains unjudged — runafterfixfreeze withno competingnumericwork — costifwronginvalidlatencybaseline.
- Final: Ruling: optimization/memory/transferbytes/Warpcomparison deferred — explicitlyoutofroundandunmeasured — costifwrongusersmightoverestimateperformanceunlesslimitationsretained.
- Final: Ruling: friction/dual-arm/complexcollision/floatingbase/P6.5replay notP6evidence — doafterP6gate,neverinferfromspherecases — costifwrongprematurehandoffclaim.
- Final: Ruling: maliciousfulltracefabrication outsideconsumerthreatmodel — trustedproducererroraudit statedexplicitly — costifwrongselfreportedexecutioncanbeforged.
- Final: Ruling: loaderisnotcompleteindependentnodeintegrator — fixedSIload/domainauditplusseparatefullFP64comparison jointlygate — costifwrongstandaloneloadcouldmissunmodeledphysicaldrift.
- Final: Ruling: deviceintermediateFnotindividuallydomaincertified — physicalFP64candidatecheckedandnewforce/energygatecapturesquantizationeffects;noindependentthirddefectreproduced — costifwronganotheradversarialoperatorcasecouldrequirmoreexplicitcheck.
- Final: Ruling: unlimitedextremeFP32scalesnotpromised — explicitrejectsafe,cannotacceptwrongforces — costifwrongconservativevalidmodelsrejected,NUM003recordsboundary.
- PublicAPIcacheprobe confirmed defaultcache_write_failed on2freshRuntimes, sameRuntimeprocess_hit; explicitworkspace firstcold_compile secondartifact_hit. RecordedPERF007+trackedpublicrepro/rawJSON. Environmentfaultnottyxdefect; setTYX_CACHE_DIR=workspace/artifacts/tyx-cache forALLfuturebackendruns,alsoTEMP/TMPworkspace. Ruling: enable existing public artifact cache in writable environment — fixes permission-induced repeated compilation and preserves all numerical/transaction code — timings across default/workspace environments notcomparableandmustseparate. CurrentC++63oldenvwillbestoppedandrerunall65currenttestsunderwritablecache.
