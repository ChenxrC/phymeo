# C6：设备接受树状态的有限静态接触反馈专项

状态：**TESTS_AUTHORED / FULL_MATRIX_UNEXECUTED / PRODUCT_PARENT_OWNED**。作者仅文本、文件 IO 和 Git；没有 import、collection、pytest、数值探测、静态或构建执行。依据父冻结合同 [resident-contact-feedback](../superpowers/plans/2026-11-08-resident-contact-feedback.md)（7623c39 及父后续字段裁定）；不延伸旧摩擦或机器人抓持窗口。

父作业 **325** 已完成首个有效 RED：真实有限盒 patch、旧材料律双侧 ±2 N、原 FixedRobotTree / batch / snapshot / 输入 gzip 前置全部成立，只在缺失 `TyxWorldBatch.prepare_contacts` 失败；1 failed / 1.23 s / RAW1。首节点提交 9e13414d6097706c24b19becddfeac1add364ab8，后续原输入及断言保留。这只证明当前公开入口缺口，不能冒称新物理通过。其余结果、两后端、最终静态及源码外 wheel 均待父唯一串行验证。

## 最小公开调用与所有权

```python
plan = batch.prepare_contacts(
    collision_model, bindings, materials_by_pair,
    candidate_capacity=4096, site_capacity=128, sampling="discrete_entry",
)
accepted = batch.step_contacts(plan, h)
receipts = plan.last_accepted
statuses = plan.last_status
```

原 ordinary fixed-root R/P 树绑定到真实有限 static sphere/capsule/unrounded hull。树的完整 M/b/J、当前外载、同次 bounded PD 和一次半隐式推进仍由既有设备实现完成。几何/材料采样在 HOST，不称全 GPU 接触、不称 CCD。配置不变；所有碰撞 shape 的原 IDs、有限顶点及半径保持。材料仅 `RobotContactMaterial(friction=0)`；`NormalLayer`、缺少 law、动态 pair 未显式排除、非法 source/world/group、无限 plane 或未知采样方式拒绝。

父拥有 batch/contact plan、旧 storage/kernel/geometry/exports；RD 作者独立 pure-HOST bridge。本文作者范围恰三 NEW：

- `tests/scale/test_resident_tree_contacts.py`
- `tests/scale/test_resident_tree_contact_transactions.py`
- 本报告。

原 single/batch 物理程序、C4/C5 tests、已有数据、控制或门槛均未改。

## 首跑前固定的物理矩阵

全局 `h=1/1024 s`，k=64 N/m、μ=0；未注明时 c=0、gravity=0、q/v=0、kp/kd=0、目标/速度/ff/外载=0、cap=8 N 或 N·m、delay=0、512 sweeps。设备原质量门 2e-5、原 clipped PD 门 5e-5 保持。状态/FK/load 的 `atol=5e-6, rtol=5e-5`；40 步及卸载重入窗口 `atol=2e-4`。首次 dyadic R01 v=h/q=h² 用精确断言。不会按运行结果挑新门、物性、迭代预算或抹掉入步速度。

| 组 | 固定输入 | 独立结果与能击穿的错误 |
|---|---|---|
| R01 基础 | Pz，m2，COM 原点，I=(1/8,1/8,1/8)，安装 z3/32；动盒半宽1/8；静台 x/y±1/2，z[-1/4,0] | depth1/32，完整面积归一后的 N2，a1，支撑反力−2z。四站点不能错误相加成8N。 |
| R01 阻尼 | 原几何 c4，v=+1/8 或−1/8 | N=3/2 或5/2；再给 +2z COM 外载，N仍3/2、总a7/4。区分入步 v 与 free/candidate v。 |
| R01 PD | kp4/kd1，其余原值 | v+=2h/(2+h+4h²)，tau=−(1+4h)v+，q+=hv+；接触必须参与原 PD 方程，不是末尾 kick。 |
| R01 delay | delay0/1；连续两步 ff=+1/−1，COM Fz=+2/−2，其他0 | 仅 target/rate/ff 使用延迟；当前 COM 力和当前真实接触均不延迟。原 held 不被接触覆盖，history 正确移位。 |
| R01 有限几何 | 球 r1/8 在 z3/32；静球中心 z−1/8、r1/8；另有限胶囊端点(±1/4,0,−1/8)、r1/8 | 球球、胶囊轴中、x=1/4 端点各 N2/a1；x=1 越端0。独立有限盒 x=1 亦0，不得无限延长。 |
| R01 非主轴形状 | 盒绕z=.37，原半宽与中心；静八点截锥 top z0半宽1/2、bottom z−1/4半宽3/4 | 真实有限 top patch 合N2、总矩0；读原八顶点/IDs，不能改成AABB或吸附FP32姿态。 |
| R02 分支 | 根Rz m2/Izz1/4，两个P子 m1/Izz1/8，offset(1,0,0)/(0,1,0)，轴y/x；P1球 local(−1,7/64,0)，r1/8；静台 y[-1/4,0]、x/z±1/2 | N+1y、P1 COM矩−1z；e=(0,1,0)，M=[[5/2,1,−1],[1,1,0],[−1,0,1]]，a=(−2,3,−2)。根散回抵消且未接触兄弟仍响应。 |
| R02 协变 | sibling次序互换；整体(x,y,z)→(y,z,x)+(.5,−.25,.75) | 按原机械坐标名字重排，广义响应不变，world wrench真实共变；不是用body整数猜列。四个初态组合，两个40步窗口。 |
| R03 偏心R | m1/Izz1/4，球local(.5,3/32,0)、r1/8；有限台x±1,y[-1/4,0],z±1/2 | v0/c0给N2、tau1、a4；v=.5/c4给vn=.25、N1、tau.5、a2。另反序CollisionKey，表面/力归属仍正确。 |
| R03 窗口 | 上述两初态各40步，不重设状态 | 独立Rz(q)球心、力矩及半隐式递推；实际保存的末态材料点/共同点沿完整stored矩阵运输。 |
| R04 卸载重入 | R01，v+.5；前1/8s外载0，随后固定COM−4z N；h及h/2 | 总预算7/8s（后段最多再3/4s），真实分离时临时wrench必须0，首次真实正入步N重现的接受步停止。不是从首错加长预算。 |
| R05 worlds | W1四个独立R01/R02/R03/空P；W4 keys17/103/7/409逆序 | 各40步、同局部shape/body IDs。空P ff1→a.5，无任何接触绑定；按world对比完整checkpoint/FK、sites和载荷，不按row位置。 |

R02 40步独立计算：q=(θ,s,t)，v=(ω,ṡ,ṫ)，`M00=5/2+s²+t², M01=1, M02=−1, M11=M22=1, M12=0`；`b=(2ω(s ṡ+t ṫ),−sω²,−tω²)`。接触球心 `cx=−sinθ(s+7/64), cy=cosθ(s+7/64)`，`N=64 max(0,1/8−cy)`，`e=(cx N,cosθ N,0)`。测试独立解 M a=e−b，再半隐式积分；未从产品M/b/步进生成真值。

R03 独立 `c=Rz(q)(.5,3/32,0)`，`vn=cx v`，`N=max(0,64 depth−damping vn)`（只depth>0），`tau=cx N, a=4tau`。R04 独立 `N=64 max(0,1/32−q)`，`v+=h(N+Fext)/2; q+=hv+`。两窗口都保存首个失败前缀，不跳过未达门。

## 公开收据核与数值口径

entry/accepted 的 checkpoint 和 TreeKinematics 保相同tick，position/velocity都是实际接受设备值。`input_external` 为接触加法前full N广义外载（含COM输入投影、未减bias）；`selected_drive` 是本次真正选中的 N×3 target/rate/ff；`actual_drive` 是真实N电机载荷。完整未投影输入仍在 `entry_state.control` 的N×10。诊断 external 核为 input_external+contact_generalized，接触不能重复施加或污染下一步held。

`law_wrench_fp64` 和真正上传的FP32再扩宽值均N×6，前三COM力、后三关于实际COM的矩。将原全J再次散回，只检查公开载荷的独立组装一致性；R01–R03另有完整物理独立oracle，因此不以J自身证明动力学正确。

每site保原 `ContactSite`，包括 original source IDs/weights、world skin offset、两表面点、gap/normal、归一权重及共同 load_point。测试从独立注册局部点按原ID加权，加实际world skin offset重建，不拿轴点权重当完整球皮地址。两侧相反力、同load_point下入步速度 `vCOM+ω×arm`、COM力矩、静台关于原声明顶点均值的反作用矩逐项复算。

`diagnostics.entry_frames/accepted_frames` 保每原CollisionKey对应完整 `ContactShapeFrame`；原shape、真实world材料点、group、stored p/R、COM reference及双速度逐项检查。模型本来的有限点与IDs来自独立输入保存，不从这些frame反写预期。

**实际位移构造门与物理门分开**：将入口材料表面点及入口共同作用点分别用保存的entry R/p作FP64 `solve(R,point−p)`，再用保存的candidate R+/p+运输；static位移严格0，不重新最近点，不重新HOST FK，不正交化。1e-10m只验这份公开记录内部构造；设备q/v/FK仍用原5e-6/5e-5与窗口2e-4。此独立门能区分有限圆弧与entry切线，即使二者之差小于物理误差门。

双侧 `F·实际位移` 分别重算 application_work_j/material_work_j；R02/R03另保存实际 summary 动能、独立动能变化，R03保存 entry power×h 与原弹簧能量差。半隐式法、阻尼及FP32量化的不同项分别保留，**不声称严格能量守恒，不把entry power×h冒实际有限步功**。

## 有界事务、能力边界和故障

- site cap=0..4096或精确bound-world映射；非法类型/额外或缺失world于分配前拒绝。站点计完整patch，速度1m/s、c4把N截到0时仍不能用正力行数逃过cap0。
- perworld超帽保整个原checkpoint/FK/held/history/tick，邻world真实推进。配置不改，增容用新plan。另以q0=1/32+h、v−1先无候选接受一步，再恰触patch的cap0拒绝，验证**同plan**旧last_accepted仍可读。
- candidate cap0且已有真实接触是整批异常，空P/scalar邻居亦不能先发布。深包含sphere轴、同心球非唯一法向不当分离。
- 未绑定affine N2/K1、scalar FreeSphere与ordinary接触world同批正常推进；affine接触绑定明确拒绝。实际floating模型在既有world声明层就拒绝，测试不发明不存在的floating设备绑定。
- warm0/1覆盖两active方向，对upload、launch、sync、status、真实candidate读回、完整receipt构造作call-through后第一次抛错；保存before完整状态/FK/epoch/active/last_accepted，失败不得发布。退出注入后可关闭临时plan重建，再从原物理状态用原同命令推进，与独立control相同。sync只首抛，避免把永久失联伪装为可重试。
- 在真实 `_read_candidate` 后同tick restore，明确restore自身合法epoch+1/active变化，而旧attempt不得再发布。错误batch owner即使同world/tick也拒绝。
- 第二次实际 `ResidentContactPlan._buffer` 分配call-through后抛，检查本plan ExitStack释放两项必要scratch、closed和owned bytes归0；不造dummy分配、不把plan bytes与batch重复相加。
- 单plan关闭不能关闭借用runtime；batch关闭两个plan时一callback抛错，仍必须尝试另一个并最终释放原batch资源。观察不推进epoch/tick，旧receipt不能因后续步或用户打开数组写权限而被改写。

当前没有 moving/moving、affine接触投影、浮根反馈、持续切向历史、设备窄相或CCD。静态反力是本次碰撞对固定障碍的接触wrench，不是整个机器人基座总约束反力。

## 保存与待父验证

所有真实物理窗口在对应pytest `tmp_path` 写gzip JSON。R02/R03/R04和W矩阵使用 `finally` 保完整已完成前缀；首节点先保存真实前置输入。记录包括模式、h、独立模型/commands/source、actual checkpoint/geometry/diagnostic/sites/wrenches、oracle递推及真实stats；一次性非法配置/空/close等没有假造物理trajectory。观察读回和energy read的成本也进入实际stats，不声称暖态零读回或新性能收益。

主要文件前缀：`r01-public-input/result`、`r01-damping/pd/delay/geometry`、`r02-first/window`、`r03-first/window`、`r04-unload-reentry`、各W1及W4子目录的`r05-world-window`、`r05-site-capacity/zero-force-sites/candidate-capacity/unbound-kinds/undefined-normal/fault-retry/stale-same-tick/owned-observation/allocation-recovery/retained-receipt`。参数化tmp_path分别隔离，不能只glob选择成功行。

父建议执行边界：保留325首RED，产品到位后先功能文件与事务文件，再真实interpret/CUDA必要分组、静态、源码外wheel；不重跑旧C4/C5全套。这里没有预先虚构collection数量、后端pass或wheel结果。任何fixture/API作者错误需明确首日志和改动，不重调物理。
