# C3 单转铰驻留实现候选

状态：**CODE_READY / TESTS_PENDING / UNEXECUTED**。作者从 `47b2805` 继续，未合并父或修改旧产品接缝。父已保存首次有效 RED：作业105，`test_r01_revolute_effort_uses_axis_inertia_actual_accepted_state`，真实合法单R program 后缺 `RevoluteRotor`，RAW1/0.90s。本报告不把 RED、源码阅读或手推写成设备通过。

独立先行修正 `8b803db`：仅父明确授权的测试 URDF 从既有 loader 不支持的 `continuous` 改为 `revolute`，补齐原四字段 limit（±4096/effort8/velocity4）；真实质量、q、几何、控制和阈值不变。未新增 continuous importer；原 RED 保留。

## 本次五个新文件

- `src/phymeo/backends/_resident_revolute_world.py`：`RevoluteRotor`、真实模型编译、十列命令、统一 stored q 角域检查。
- `src/phymeo/backends/_resident_revolute_binding.py`：真实安装/source 校验、普通/折叠资产 moving group、原材料点和固定工程误差打包。
- `src/phymeo/backends/_tyx_resident_revolute_kernels.py`：真实 q 下 COM 力矩、轴惯量、隐式 bounded PD/一物理步延迟，写原 candidate。
- `src/phymeo/backends/_tyx_resident_revolute_bounds.py`：读取实际 device accepted q，旋转原有限点，写原 device bounds/stamp。
- 本报告。

没有第二 runtime、HOST 动态 FK 桥、临时刚体、force 注入或夹具名分派。原旧 C1/C2/F1、默认普通机器人接口和已冻结数值用例未变。

## 父薄接缝所需精确接口

`compile_revolute_world(spec)` 返回 `(kind, domain, axis_tuple, frame, props_f32[16], state_f32[12], command_f32[10])`。kind=3+delay；`frame` 是 `RevoluteGeometryFrame`，owned axis 是真实安装后的 FP64 世界轴；props/state/command 是独立 HOST 数组用于父初始上传，不保留 caller model 的数组别名。model 重新验证并复制，支持且仅支持普通单 R、parent=-1、无 stops、delay0/1。

`props=[invI,gx,gy,gz,kp,kd,cap,0,I,mass,c0wx,c0wy,c0wz,Awx,Awy,Awz]`。`I=a·D a+m|a×c|²`，c0=R0*c，A=R0*a。所有字段采用旧 `as_fp32` 的有限/可表示合同；异常轴惯量拒绝。原 body COM 主轴与轴的非同向关系保留，不能用 bodymass 代 I。

`revolute_packet(value,domain)` 返回 checked FP32[10]：`[target,rate,ff,generalized,Fx,Fy,Fz,Tx,Ty,Tz]`。后三组真实世界 COM 载荷不作 HOST 初始投影，只有前三列进延迟队列。`validate_revolute_angle(value,name)` **接收单标量并返回 np.float32 标量**，依据转换后的 stored q 检查含端点的 ±4096 rad；父在创建/restore 使用，同一常量在设备候选检查，无 wrapping。

`make_revolute_step_kernels()` 返回两独立 handle；参数完全沿既有 `advance_worlds(count,h,kind,props,commands,state,tick,candidate,next_tick,accepted)` 顺序。NEW pass 只写3/4行，从原 accepted q/v/queue 计算当前 COM 臂与 `A·(r×(mg+F)+T)+generalized`，再驱动限幅与一次漂移；越角域、非有限或最大tick整行回拷，包括 history。父需在旧0/1/2 pass之后、唯一sync/status/swap之前调它；本核不负责发布。

`compile_revolute_binding(shape,binding,owned_frame,owned_axis)` 返回 `(owned_zero_motion, relative_points_f32[V,3], geometry_f32[8])`，后两项 **bytes-backed readonly**。geometry=`[pivotXYZ,actual_Axyz,radius,padding_m]`。点在 FP64 先减真实pivot，再上传，不把大世界坐标先转FP32相减；sphere1中心、capsule2有限端、hull全部原点。安装/真实owner/source的 shape内容、IDs和revision均核验；folded tool按 `sites` 的真实group0，fixed root拒作moving。直接构造仅是显式安装声明。

`make_resident_revolute_bounds_kernels()` 返回两独立 handle；签名：`resident_revolute_bounds(count,slots,offsets,zero_relative_points,geometry,state,tick,bounds,stamp)`。slots[S]非rotor=-1，offsets[S+1]沿原有限点范围。只写实际rotor行；stamp=(valid,accepted_tick)。父沿原同锁borrow喂当前accepted buffer，首sphere pass排除slider/rotor，各专属pass写自己的行，再原BVH读取同一bounds。query失败不得发布新result或改变物理state。

## 工程精度及能力边界

固定预算来自[已冻结casebook](../development/p65-resident-revolute-casebook.md)：`64*eps32*S + 8*2^-20*L`，S含世界pivot/原点坐标尺度，L为原有限点到pivot的最大距离加radius。预算与radius向外转FP32，最终端点再RN外扩并含最小normal floor；非有限envelope使stamp无效。正常 `sinf/cosf` 仍负责动力学/几何，未做Taylor、range reduction或人为调整物理状态。父不得以fast-math替换本合同。

本机 TYX 默认 fast_math=False，sin/cos lowering 为 CUDA sinf/cosf。NVIDIA [官方数学精度说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) 的 2 ULP 是观测误差而非穷举保证；这里明确是待实际 interpret/CUDA 与固定FP64几何oracle验收的工程外扩，不是全实数形式认证。允许少量多候选；原近触/1/4 m miss 与4096大平移用例不调整。

只做有限 sphere/capsule/hull 的**当前姿态宽相**，不是 CCD、窄相或接触反馈。旋转静态障碍、动态–动态、一般 curved solids/mesh、浮根、多关节、mimic不在本片。粒度为单真实转铰世界；后续完整树须另有持久FK/J/M/bias与全矩阵推进，不能由本片外推。

## 待父执行

两个原专项的 interpret/CUDA、40步物理/宽相窗、W1/W4/W32暖态双buffer与完整计数、失败/restore/close、静态和源码外wheel均 **尚未执行**。作者没有 import、collection、numerical probe、pytest、lint、format、compile/build/install，也不填写推测pass数量/时间/性能。

父旧hook仍负责 max command/property宽度（旧纯4/9、含rotor10/16）、WorldBinding checkpoint/control/queue、step的统一事务和未知kind拒绝、`q_rad`/kinematics单位、I_axis动能、plan注册与真实独立统计。仅这些hook接通并实际推进后，才可声称本片有 device state→broadphase 能力。原 `bounds_kernel_launches` 与 `slider_bounds_kernel_launches` 语义保持，新增真实 `revolute_bounds_kernel_launches` 并纳入total。

## 父实际专项结果（2026-11-07）

状态：**HOST / INTERPRET / CUDA 专项及共享接口审阅通过；新组合 wheel 尚待完成**。父导入3afb536/5dbd09e，接通真实kind3/4、异构命令/常量、第二设备pass、原accepted/candidate事务与原宽相，所有旧共享修改由父完成。未更改原物理夹具/数值门或以HOST动态几何替代设备计算。

| 作业 | 实际结果 |
|---|---|
| 114/115 首静态 | Ruff 3项import格式；9文件需排版，原失败保留 |
| 116/117 修复 | 仅import排序/排版，RAW0 |
| 118 HOST协议 | 4 passed / 71明确设备标记排除，0.63s / RAW0 |
| 119 interpret | 71 passed / 4 HOST排除，5.45s / RAW0 |
| 120 CUDA | 71 passed / 4 HOST排除，29.77s / RAW0 |
| 121 受影响共享CUDA | 7 passed，3.29s / RAW0；球/滑轨边界、restore与清理 |
| 122/123 最终静态 | Ruff与11文件format-check RAW0 |

证据根为 `artifacts/p65-generic-collision-20261107-01`；`c3-validation-receipt.json`记录上述六个最终作业的原raw exit、源before/after及日志hash。119/120各保存53个测试JSON，共106；另256个JSON是编译等缓存文件，不能充作物理轨迹。保存包含四个正负/延迟40步动力学窗、两个40步当前姿态碰撞窗、完整边界/配对/失败状态及W1/W4/W32暖态记录；暖态后续q/v明确标注解析oracle，不冒设备读回。

CUDA三个规模各5次暖态推进/查询：plan设备分配均保持16，动态envelope上传0，snapshot新增读回0，两个state buffer均经过。实际新增10次bounds pass（通用与rotor各5）、10次BVH kernel；结果读回分别480/1920/17920 bytes，配对metadata上传120/480/3840 bytes，非零成本保留。batch的状态接受标记另行读回。没有性能对照或Newton吞吐胜出结论。

真实动力学覆盖轴惯量、非主轴、当前COM力臂、全部世界力矩、bounded PD、仅驱动三列延迟、混合球/滑轨/rotor、含端点±4096与事务失败。几何覆盖偏心sphere/有限capsule/box-hull、多shape材料ID、资产真实group、正负多周、16m细长体与4096m公共平移下固定近触/miss。限当前单R接受态到静态有限几何宽相；不代表多关节驻留、动态–动态设备窄相、CCD或接触反馈。

原robot对父5共享hook的只读审阅已完成，未发现阻断问题；核异构packet恢复、两pass统一swap/epoch、实际device geometry资源登记/关闭及失败last_result保持。审阅无执行，不代替数值结果。


## 新组合 wheel 正式限域接受（2026-11-07）

产品fdfa586，133构建RAW0；wheel SHA256 `3BB31FEFC4B66A0CCB45B6500EE2C22566C4D1F5E2566CBCD8C31C4F03D37D7D`，1180542 bytes。18个变更product文件已与解包字节核同。运行cwd及PYTHONPATH均在源码外Temp新目录；实际270个phymeo模块来自该wheel。

134原RAW1保留：F2全部512步/接触/作用点矩/完整mimic余额/事件断言已通过，随后C3最终命中列表错误。脚本把原小盒子组合半宽3/32的[28+delay]误用于这里的两球组合半宽1/8；实际逐步FP64几何、q/v和候选断言均先通过。135只修正独立列表为world7=[27,28]、world24=[28,29]，补完整rotor保存，RAW0；读取并复用原F2完整gzip，未重跑F2物理，未改wheel、源代码、几何或工程门。此失败是新包装验证脚本问题，不改写为首次全绿。

源码外实际F2到0.25s，首/末受力0.00439453125/0.06591796875s、峰值10.11380374689702N，末态卸载，最大线动量残差1.2971987494374595e-6 N·s。实际CUDA双rotor各40步、正负恒矩及delay0/1，q/v、当前有限球AABB、完整候选/tick/epoch与原计数通过；动态包络上传0，分配不增长，两个buffer均使用。完整状态/几何/输入/载荷/事件以及80个world-step设备记录保存在 `artifacts/p65-generic-collision-20261107-01/f2-c3-wheel-results`，父收据 `f2-c3-wheel-acceptance.json` SHA256 `876536F7B6971FBA39CA38610822E7C9876BBB10498D4B0761AB6A981EDC9E11`。

本片正式接受：HOST浮根仿射mimic与原共同场景、CUDA单R接受态到当前姿态宽相。不是浮根GPU、多关节驻留、设备窄相/CCD、持续摩擦或整个Newton功能对齐；原100g mimic和软布目标失败继续保留。当前执行槽释放，后继按原多案例计划继续。
