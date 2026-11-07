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
