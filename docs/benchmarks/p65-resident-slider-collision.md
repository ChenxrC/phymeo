# C2：真实单滑轨接受态直供设备宽相

状态：**CODE_READY_TESTS_PENDING / UNEXECUTED**。作者只做文本、文件和 Git；新专项、import、interpret/CUDA 数值、lint/format 与 wheel 均未执行。父统一串行验证。依据 [冻结实施计划](../superpowers/plans/2026-11-07-resident-slider-broad-phase.md) 与 [S01–S08 原提案](p65-resident-slider-design.md)。原 C1 与 C2 提案文件保持不变；本报告不把提案历史改写成已验收。

## 本增量及来源

本增量不增加物理积分或 runtime。实际世界仍由既有 FreeSphere 或一个 PrismaticSlider 推进；几何只消费同锁借出的当前 accepted state[slot,0]，映射 p_world(q)=p_world_at_q0+a_owned*q。一个滑轨可有多个有限刚体形状，静态根/工具来源须清楚；只配同 world 明确注册的静态几何。宽相候选不是接触、CCD 或全设备机器人。

NEW 文件恰为：

1. src/phymeo/backends/_resident_slider_binding.py：拥有的帧、零位声明、资产来源验证和实际轴区间编译。
2. src/phymeo/backends/_tyx_resident_slider_bounds.py：同源两 buffer 句柄的设备 AABB 覆盖核。
3. tests/scale/test_resident_slider_collision.py：S01–S08 分项/多姿态、物理窗口和事务/资源负控。
4. 本报告。

旧 _world_step_layout.py、_resident_collision_plan.py 及必要 tyx_world_batch.py hook 由父维护，作者未编辑、未合并父后续提交。

## 正式接口及父接缝

SliderGeometryFrame(origin_m, orientation_xyzw, axis_local) 拷贝为 owned tuples。原点须有限 FP64 三向量；四元数须满足已有 RobotModel 的单位长度 1e−10 接受门后规范化，q 和 −q 统一符号；局部轴非零、缩放后规范化。没有可写 RobotModel 别名。原点比较为同值；规范化的旋转/局部轴核对最多 32*eps64 的绝对舍入差，以容纳再次规范化，实际运算仍完全使用 owned_axis，不使用 caller 伪轴。

ResidentSliderBinding(key, world, zero_motion, *, origin_m, orientation_xyzw, axis_local) 保存单个 q=0 的完整世界姿态；要求 world/key 一致、单姿态，无 end/turn。CollisionMotion 的 bytes-backed vertices 再次独立复制。frame 属性返回拥有的 SliderGeometryFrame。这是调用者显式安装声明，不能证明任意外部资产的 owner。

ResidentSliderBinding.from_asset(asset, source_key, *, key, world) 仅接受真实 RobotAsset/RobotModel 的一根 prismatic 树。普通源 collider 必须 link_index=0；FoldedURDFAsset 使用 sites[col.link_name].link_index=0，根 None/-1 拒绝。零位来自公开 collision_motion(zeros(1))，不重算固定折叠/主惯量或局部碰撞 origin。内部保存来源 CollisionShape，compile 要求调用者的 target shape 只重映射 key，原材料 IDs、revision、几何、过滤等内容全部保持。mimic、浮根、多活动关节和未知资产不推断支持。

compile_slider_binding(shape, binding, owned_frame, owned_axis) 返回 (zero_motion, axis_bounds)。要求 finite rigid shape、完整材料点数和同 key；不接受无限 plane 或变形域。axis_bounds 是 bytes-backed readonly float32[6]，前三为真实 FP64 owned_axis 的下界，后三为上界；subnormal 分量用 zero/min-normal 区间包住，防止设备 FTZ 把轴方向缩成假净空。模块不导入 _world_step_layout/_resident_collision_plan，也不创建 runtime。

make_resident_slider_bounds_kernels() 返回两个独立 tx.kernel。固定参数顺序：

~~~text
count, slots, zero_bounds, axis_bounds, state, tick, bounds, stamp
~~~

slots 每 shape 对应 slider slot 或 −1；zero_bounds/axis_bounds/bounds 都为 float32[N,6]，state 与 tick 使用原 accepted buffer，stamp 为 int32[N,2] 的 valid/tick。只覆盖 slot>=0 的行，其他 sphere/static 第一 pass 输出及 tick 完全保留。按 q 正负选实际轴区间的上下端；先对乘积外向，再对零位 bounds 加法外向，显式 mul_rn/add_rn/sub_rn 防 FMA 收缩；每步使用 2*eps32 相对 padding 和 min-normal 底限，覆盖乘加舍入/FTZ。大项抵消的误差在乘积阶段就保留，不只对最终小数加 margin。非有限/overflow/倒置区间写 invalid，父在读 BVH 前拒绝整份 query。

父须复用旧 _envelopes/_host_bounds 生成 zero_bounds；不要将 origin 再加一遍、不要把初始 q 的几何当 q=0。父保原 bounds_kernel_launches 为第一 pass，新增 slider_bounds_kernel_launches 单计第二 pass，total_bounds_kernel_launches 可显式相加。全部 active shape 的 world tick 必须一致；borrow、epoch、registration、last_result、close 和失败事务仍只有原实现一份。两 pass 后只一次发布，无动态状态 snapshot、envelope 上传或暖态新增设备分配；BVH metadata 上传和 status/bounds/candidates 必要读回照实计费。

## 冻结用例与明确边界

- S01：2 kg、Rz(π/2)、origin=(1,−1/2,1/4)、q0=1/8、2 N、h=1/4。先实际旧动力学出步 v=1/4、q=3/16，再进入新绑定；独立有限盒出步恰触。有效 RED 必须是新 binding 可导入之后旧 plan 尚不支持 slider 的行为拒绝，不能用缺 module 冒有效 RED。
- S02/S03：3/5,4/5 非主轴、负 q/翻号、恰触与 1/64 净空；实际世界外载 (0,5,0) 经旧通道投影 4 N，得到 v=1/4、q=1/32，几何不二次施力。
- S04：同一物理 body 的 sphere/有限 capsule/偏置 hull，原材料集合、revision 与同 tick；真实 URDF 活动体+固定工具+偏心 COM，通过公开折叠 FK 校验 sphere/cylinder/tool box。根保持静态。来源 shape 被改尺寸/ID/revision、非 prismatic、异 body/world、错误 frame 都拒绝。
- S05：原 40 步 h=1/16、m=2、ff=2、帽4，delay0/1 不重置。分别 q40=205/64、195/64；障碍候选仅第22/23、23/24步，完整真实 q/v/tick 逐步检查。仅为离散姿态资格，不等于连续首触。
- S06：W1/W4/W32，world=11+17*i、反序输入、同局部 ID；FreeSphere/S01/S02 混合，与独立同态 oracle 比完整 bounds/candidates/ticks。两 buffer 暖态禁止 snapshot/kinematics/envelope query/新 device alloc，保必要传输统计。
- S07：同 tick restore 到不同 q、局部 reset、MAX_TICK 单 world 拒步但邻 world 推进；双 plan registration 隔离；注入 BVH 失败及同 world 多 shape tick 不一致，不能发布半表或换 last_result。
- S08：正负极小轴、2²⁰ 抵消和 2¹²⁷ 溢出；Fraction 参考包含门、capacity0、空 plan、仅 slider 无静态 pair、close/runtime 所有权、immutable result/axis/zero pose、非法 frame/interval。

几何门是覆盖实际 FP64 输入/FK 参考的外向包含，不要求与 HOST bounds 逐 bit 一致，不追求零 false positives。2²⁰ 抵消可能产生明显保守额外范围，这是本首片的明确代价。候选容量、表示域超界及失去同步必须失败；不能为过门改质量、轨迹、尺寸、窗长或给 C1 动力学添加新能力。

## 父串行验收安排（待执行）

先用新 binding 与旧 plan 记录 S01 行为 RED，再应用父 hook 后运行 HOST 绑定节点、实际 interpret、实际 CUDA 全新专项。CLI 为项目既有 pytest tests/scale/test_resident_slider_collision.py --tyx-mode interpret 与 --tyx-mode cuda（各自 fresh guard/receipt）。只补受影响旧接缝必要节点；不重跑 C1 已绿全集。父确认实际收据后追加真实计数、首失败、退出码和静态/源码外 wheel 资格。

截至本提交：作者所有项目执行均 **UNEXECUTED**，仅人工源码读取和手推；没有任何性能收益或全域安全结论。

同次证据留存：S05 在各 pytest tmp_path 写 s05-delay{0|1}-{mode}.json，逐步保存实际 acceptance/q/v/历史/控制、完整 bounds/candidates/ticks、plan/batch 计数；S06 写 s06-w{1|4|32}-{mode}.json，warm 前真实 checkpoint 与 warm 内实际 acceptance/完整宽相输出/计数分开保存，解析 q/v 仅标 oracle_q_v。warm 内没有为报告增加任何 snapshot、kinematics 或额外 query。S08 写 cancellation、overflow、正负 small-axis、capacity、empty、normal-close、close-callback 的同次输出/统计。建议父以新证据根下 --basetemp 收集这些 JSON；文件写入不算性能收益测量。负控同时覆盖 slider plan 的 close callback 抛错后其余 plan/原 runtime 仍被清理，以及普通 URDF 来源副本不受外部模型数组攻击。
## 父本地验证（2026-11-07）

状态更新：**LOCAL_VALIDATED / WHEEL_PENDING**。作者2dfe81b候选已导入；父接真实owned frame、同一borrow中的sphere/static与slider两pass、同world多shape tick核及分项计数。只查询当前接受态dynamic/static有限几何，不增加物理积分、CCD、接触响应或动态体互撞。

证据根 artifacts/p65-generic-collision-20261107-01：
- 78 HOST绑定18 passed /0.60s，设备20项分组排除；实际18项未在设备组重复。
- 81 interpret20 passed /2.33s；82 CUDA20 passed /14.32s；各排除已单独执行的18 HOST项。
- 83仅受影响旧C1入口/过滤/错误绑定/staticmesh/registration六节点的14实例通过 /0.84s，没有重跑C1完整绿集。
- 76/77静态全6文件通过；79/80修订后的kernel/test在84/85增量Ruff/format均通过。全部真实RAW0收据保存。
- 70首先发现测试错误地从phymeo顶层引用forward_kinematics；改为既有dynamics.robot_kinematics导入后，71先真实推进q=3/16、v=1/4，再由旧plan拒绝slider binding（1 failed/.76s），才算有效RED。
- 79实际lowering拒绝if分支内新变量p_lo；父将轴端点变量提前定义，再按符号选轴、执行原两次RN乘积及外向扩张。80继而发现S04测试PD参数为整数存储，改成同值浮点列表。所有首错保留；未改物理参数、窗口、几何、预算或oracle。

S05两个delay各40实际非重置步，S06 W1/W4/W32与两buffer、S08负控共26同次JSON（每端13个）保留。S06暖态不读取snapshot；actual与oracle字段明确分开。CUDA W1/W4/W32各6次query，均第一pass6+slider6、BVH核12；必要读回分别576/2304/21504B、18次。初始设备allocation各14，暖态无新增、无动态envelope上传；metadata静态上传仍照实计费。没有测速或整体性能收益声明。

父纯文件读取核81/82/83源码before/after/current相同，并登记26保存文件bytes/SHA；receipt c2-source-and-saved-receipts.json，SHA a9919587d23db3f01b7f809bd348383875d251f78437b113d717edcab96a6caf。没有再运行producer或oracle。原作者对父三处接缝只读审阅未发现阻断；另一个原agent正在做独立新增源码审阅。新源码外wheel与浮根工具片结合排入下一共同集成验收，当前不能写成wheel已过。

## C2 正式限域接受与源码外 wheel

产品6549c3a的HOST18、interpret20、CUDA20和受影响回归14原结果复用；原RD作者完整只读复核几何绑定、FP32保守界、同锁双kernel、多shape tick和失败发布，未见新增阻断。其结论限单真实prismatic body的当前姿态宽相，不是窄相/CCD/反馈力。

F1/C2新阶段run102构建与run103仓库外wheel实际执行已通过。新包装用两个真实滑轨世界：原Rz90安装和(.6,.8,0)斜轴，负q穿过0变正；每滑轨box+sphere及静态wall共6shape，实际6步接受态在双buffer上推进。所有AABB包含独立实际轴/实际q的精确值，全部4注册对的候选与独立包围盒判据相同，旧epoch拒用，暖态分配保持14。6次首pass+6次sliderpass、12BVH核，1728B必要bounds/candidate读回、18读取调用；metadata累计696B，动态包络上传0。不将snapshot观测或必要读回抹为零，不给性能排名。

wheel SHA256 `966ab9e9455fa35ae3f28d53cb5949ec61fc077c22a76e3e5c98131a71147799`，8个相关产品文件字节核对一致。完整包装轨迹/计数见 `artifacts/p65-generic-collision-20261107-01/f1-c2-wheel-saved/integration.json`，原专项26保存轨迹不重跑；两个stage原证据均保留。C2状态 **VALIDATED_ACCEPTED**，通用转动/多关节机器人驻留和全设备窄相仍待后续C3。
