# C3：真实单转铰驻留状态 → 有限形状当前姿态宽相

状态：**TESTS_FIRST / UNEXECUTED / 产品尚未实现**。作者基线 `c92d730`，未合并后续父提交；父已接受 `|stored q| ≤ 4096 rad` 和工程精度外扩路线。本文件冻结输入、解析真值和验收含义，不报告测试通过、性能收益、CCD 或接触响应。

本批作者仅新增本文件、`tests/scale/test_resident_revolute_step.py`、`tests/scale/test_resident_revolute_collision.py`。首测试提交 `017c52c`；全部 import、collection、interpret/CUDA、格式和集成由父串行执行。旧 C1/C2/F1、旧参考记录不变。

## 可复用公共切片与数据布局建议

- `phymeo.backends.tyx_world_batch.RevoluteRotor(program, state)`：真实普通 `DrivenRobotProgram`，恰一活动 revolute、parent=-1、正质量/合法 COM 主惯量、无关节 stops、delay=0 或 1。支持偏心 COM、非主轴、任意有效安装姿态。多铰链、浮根、mimic、力学碰撞响应另立能力，不落入 slider 分支。
- `ResidentRevoluteBinding(key, world, zero_motion, *, origin_m, orientation_xyzw, axis_local)`，未来文件 `backends/_resident_revolute_binding.py`；`from_asset(asset, source_key, *, key, world)` 从原普通单 R 资产提取真实活动组，折叠固定工具用 `sites` 确定 group 0，根几何只能作 static binding。保 shape 全内容、原材料 IDs、revision；更换源 shape 或安装 frame 拒绝。直接构造是显式安装声明，不冒称验证了资产归属。
- 沿用 `prepare_collision(model, bindings, candidate_capacity=...)`、`plan.query()`、`batch.collision(plan)`、`require_current` 与不可变完整 result。一个 rotor 同体可有多形状，对同 world 显式静态 finite geometry 查询；不生成动态–动态或跨世界 pair。
- 推荐按本 batch 所需最大宽度分配：纯旧 sphere/slider 保持 command=4、property=9；包含 rotor 时 command=10，顺序固定为 `[target_rad, rate_rad_s, feedforward_Nm, generalized_Nm, Fx,Fy,Fz,Tx,Ty,Tz]`。外力是本 link 世界 COM 力，外矩是关于真实 COM 的世界矩。只延迟前三列，七列当前外载立即生效；不将外力在 HOST 以初始姿态预投影。完整十列进入 checkpoint/control/restore。
- property 建议最少 16 列：原 0..8 在 rotor 分别放 `1/I_axis, gxyz, kp,kd,cap,unused,I_axis`，再放 `mass, zero_COM_arm_world[3], axis_world[3]`。父已接受下附第二 pass、props16/commands10 接缝；测试只锁旧纯 batch 4/9 和 rotor command10，不额外依赖新 property 的内部列位。安装 origin/quaternion 及原资产归属保留在 owned binding；均匀重力的转轴力矩不依赖绝对 pivot 位置。
- kind 3/4 分别为无延迟/一物理步延迟 rotor。现 `advance_worlds` 的 `else` 必须拆成明确 kind1/2 与3/4，未知 kind 拒绝。state12 双缓冲可复用（q/qd/drive effort 和三列 history）；`kinematics.values` 的列0/3 对 rotor 为 rad/rad/s，summary 只给 `q_rad`，能量用 `I_axis`。侧车方案会增加命令恢复、双缓冲、上传及资源事务两套合同，本首片不推荐。

## 同一步真实力学

令局部单位轴 a、零安装旋转 R0、COM 局部 c、质量 m、COM 主惯量 D；常量 `I_axis = a·D a + m (|c|²−(a·c)²)`。世界轴 A=R0 a，当前臂 `r(q)=R0 Rot(a,q)c`，当前外载转矩

`tau_ext(q) = generalized + A·(r(q) × (m g + F_world) + T_world)`。

单一固定轴的 I_axis 为常量；完整单刚体惯性项在该唯一广义轴上的速度偏置为零（陀螺项与向心项并非在一般多体里都可忽略）。无新增阻尼、角度 wrapping 或姿态注入。

每步先从 **同一 accepted q/v** 算 free_v=`v+h*tau_ext/I_axis`。取当前/延迟的驱动三元组，沿既有标量隐式 bounded PD：`d=h*kp+kd`，`a_pd=kp*(target-q)+kd*rate+ff`，`tau_drive=clip((a_pd-d*free_v)/(1+d*h/I_axis), ±cap)`；随后 `v+=h*tau_drive/I_axis`，`q+=h*v`。只限幅 actuator，不限幅外载。检查全候选 finite、角域及 tick 后才发布 physics/history；失败恢复原 q/v/effort/history/tick，完整 held command 仍可重试。

## 角域、有限几何与工程误差

角域固定为**实际存储 q 的 `[-4096,4096] rad`**，不折回一周。创建、restore、候选接受统一检查，越界拒绝；边界 ±4096 可驻留，向外的下一步拒绝。测试涵盖 ±π/2、π、正负多周回、±4096，以及一般 `.37/-.51`。输入取到 FP32 后，几何 oracle 使用该真实 stored q；不把 requested FP64 q 与 accepted q 的舍入混为 trig 误差。

本机 TYX `intrinsics.py:224–229` 将 sin/cos 降为 `sinf/cosf`；`backends/gpu_codegen.py:75` 的 fast_math 默认 False，`cuda_runtime.py:2152–2158` 仅显式打开时添加 `--use_fast_math`。NVIDIA [CUDA Programming Guide §5.6.7/三角函数表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) 列出 float sin/cos 的 2 ULP，但说明这是非穷举测试观测、非保证；fast-math 会替换为另一组 intrinsic。因此不能把固定 padding 称为所有实数的形式认证。

推荐固定工程预算：正常 `sinf/cosf`、fast_math=False；先为两 trig 值各留绝对 `2^-20`，再传播安装/轴/有限点变换的 FP32 误差。可用每坐标外扩基线 `64*eps32*S + 8*(2^-20)*L`，`S=max(1, |origin|∞, max|zero_vertex|∞, radius)`，`L=max||zero_vertex-origin||₂ + radius`；额外包含各次操作外向舍入、轴/安装的上传误差和 FTZ 的 tiny 下界。最终上下界朝外取 FP32，任何非有限 envelope 拒绝、不发布。此预算先于执行声明，父可实现等价或更保守的误差传播，但不能按失败样本缩小外扩或移动固定 miss。Taylor 形式区间仅是将来的独立选择，本批不实施。

对于球/胶囊，旋转原中心/有限两端后加真实半径；对于 box（已有 hull 表示）和一般 hull，旋转**全部原材料顶点**后取 min/max，不能旋转中心而保留零位 AABB，也不能把胶囊当无限轴。通式为 `x(q)=origin + R0 Rot(a,q) R0^T (x_zero-origin)`。首片无需 curved cylinder/ellipsoid、mesh、柔性或旋转静态障碍；未经支持的绑定明确拒绝。宽相可多报少量保守候选；每个固定真候选必须保留，并有留出 miss 防止退化成全候选。

## 六组冻结用例

| 组 | 固定输入 / 独立真值 | 验收与失败含义 |
|---|---|---|
| R01 真实惯性/持续步 | m=2 kg，c=(1/4,0,0) m，D=(1/8,1/8,1/8) kg m²，a=z，I_axis=1/4。tau=±1/4 Nm，h=1/64，40 步；delay0/1。 | 无重置：mstep=n-delay，qd=±mstep/64，q=±mstep(mstep+1)/8192；每步 IΔqd=h tau；`tau Δq−ΔKE−I(Δqd)²/2=0`（semiimplicit 离散功，非声称连续能量精确守恒）。首步 KE=1/32768 J；summary rad 与 I 必须正确。 |
| R02 当前姿态/斜轴 | g=(0,-2,0)，F_COM=(0,2,0)，drive=1/4，alpha=1−2cos(q)。q=0/±π/2/2π+.37/−6π−.51。另 a=(3/5,0,4/5)，c=(0,1/4,0)，D=(1/8,1/4,3/8)，I=41/100；沿 A 的世界矩1、generalized=−59/100，alpha=1。 | q=0 alpha=-1 与π/2 alpha≈1区分初始投影 bug；共同 xyz 轴循环/平移保持同响应；不以 body mass 代 I。不变轴上的偏置投影必须物理正确。 |
| R03 PD/延迟/外载 | q=1/8，v=-1/4，kp=8，kd=2，target=1/4，rate=1/2，ff=1/2；generalized=1/4，F=(0,2,0)，T=(0,0,1/8)，h=1/8，cap=2 或1/8。另 delay1、外载恰1/4 Nm、第一帧 drive尚0。 | 独立标量隐式式核 unsaturated/saturated 两支；十列 checkpoint 完整。仅前三延迟；拒步/重试不得吞队列、重复施力。 |
| R04 完整 finite geometry | 偏心球(1,0,0), r=1/16；胶囊端(.5,0,0)/(1.5,0,0), r=1/16；局部转45°非立方 box及五顶点非盒 hull，同体原IDs/revision=7；plain/folded单R URDF。细杆长度16、r=1/1024，origin=0或(4096,-2048,1024)，q=.37/-.51/8π+.37/-4096。 | FP64 Rodrigues 包围，不漏轴部/端皮肤；胶囊 y=2 的有限端外 miss。近触 overlap=1/65536 m、分离 gap=1/4 m 预固定；不为 padding 调门。原工具 authored材料仍随活动组，root拒绝作为 moving。 |
| R05 世界/双buffer/时刻 | W1/W4/W32、反序输入、重复body/shape ID异world、kind3/4；独立旧sphere+slider+rotor混合。连续40步 h=1/16、tau=1/4，球(1,0), wall中心(0,1)、半宽1/32。 | 当步候选仅第28+delay步，非 TOI。暖前双buffer预热与真实checkpoint；暖内5步 no snapshot/kinematics/envelope upload/new device allocation，完整 bounds/ticks/pairs/candidates和诚实传输计数。恢复同tick角度必须增epoch使旧result失效；同体形状同accepted tick。 |
| R06 拒绝/生命周期 | q=±4096向外步、FP32 nextafter域外、tick max、非法长度control、未知world、两R/P/stop/delay2、资产frame/IDs/revision不符；query注入异常、零capacity、两个plan其中close callback抛错、empty。 | 无半步、无新失败result；持有其他world不被reset；旧result数组不可重开写；plan不关闭借用runtime；所有plan均清理，batch资源归零。unsupported显式拒绝，不伪装旧slider路径。 |

R02 另固定全外载列留出：非主轴模型 q=.37、F=(1,2,3)、g=(0,0,-1.5)、T=(1/4,-1/2,3/4)、generalized=1/8，I=41/100，独立 `tau=7/8−sin(q)/2−cos(q)/5`；整体 xyz 循环后保持同值，防止只覆盖 Fy 而漏 Fx/Fz 列。

动力学一般 FP32 门预设：速度 `rtol=2e-5, atol=2e-6`，q 常规 `atol=2e-6`、多周回力投影 `4e-6 rad`；全二进制 R01 与 delay 恒载采用精确值。惯量变换能量留出 `rel=3e-5, abs=1e-8 J`。这些是首跑前冻结的工程门，不从观测反推。

## 父旧接缝与作者后续候选文件（本轮不实现）

父：`_world_step_layout.py` 的 kind/owned physical constants/packet/checkpoint，`tyx_world_batch.py` 的 batch宽度/真实步调度/单位summary/restore，`_tyx_world_kernels.py` 的明确kind路由及观察，`_world_kinematics.py` 的单位文档，以及 `_resident_collision_plan.py` 的验证、双buffer bounds pass/统计。纯旧路径宽度、C1 first-pass `bounds_kernel_launches` 与 C2 `slider_bounds_kernel_launches` 原语义保持；另计 `revolute_bounds_kernel_launches`、total包含三个实际pass。

未来作者可独占 `_resident_revolute_binding.py`、`_tyx_resident_revolute_bounds.py`，动力学 helper 是否另列 `_tyx_resident_revolute_kernels.py` 由父冻结。不复制 TyxWorldBatch/runtime。prewarm 建两实际 state buffer 的 handle及静态几何，query同锁借accepted state/epoch/runtime，真实device q直接写device AABB再入原BVH。HOST FK/构造时几何可用于绑定，不能用于暖态动态状态桥。读回bounds/status/ticks/candidates按实际计数；现有BVH metadata上传允许且如实记。

## 父串行验证入口

有效首 RED：`tests/scale/test_resident_revolute_step.py::test_r01_revolute_effort_uses_axis_inertia_actual_accepted_state`，先生成合法普通one-R模型，再缺 `RevoluteRotor`。两文件分别以 `--tyx-mode interpret --require-tyx` 与 `--tyx-mode cuda --require-tyx` 运行；使用父 guard、child/src 和当前已接受主环境，不安装依赖。测试保存本次真实窗口与stats JSON于各 `tmp_path`；暖内q/v严格标 `oracle_only`，实际仅有warm entry checkpoint和原已读回数据。不重新查询补日志。

**待父执行与裁定**：有效 RED、实际 interpret/CUDA、首次失败留存、静态、source外wheel均未完成；未统计collection完成数。产品和内核签名须在父审阅本3NEW后另授。完整多关节/浮根驻留、CCD、接触响应、全形状支持、速度收益都不在本首片结论内。

### 第二 pass 与几何编译的最小接缝（父实施用）

推荐 `make_revolute_step_kernels()` 返回两个独立 handle，真实函数签名与原 step 同序：

```python
advance_revolutes(count, h, kind, props, commands, state, tick,
                  candidate, next_tick, accepted)
```

父旧 advance 显式只推进 kind0/1/2；rotor 行复制原12列/原tick、暂不接受。NEW pass 仅处理 kind3/4，从原 accepted state 读 q/v/queue，写同一个 candidate/next_tick/accepted；失败整行回拷。全部 pass 后一次 sync/status 验证与 active swap，第二 pass 异常不能发布第一 pass。原 sphere-plane pass 保其原 kind0 域。props16 的精确建议为 `[invI,gx,gy,gz,kp,kd,cap,0,I,mass,c0wx,c0wy,c0wz,Awx,Awy,Awz]`，c0=R0*c、A=R0*a；前三驱动延迟/后七外载实时。summary 统一 `q_rad`（不再加 angle_rad 同义字段）。

binding 模块建议公开只给 `ResidentRevoluteBinding`，内部 `RevoluteGeometryFrame(origin_m, orientation_xyzw, axis_local)` 为 owned frame；`compile_revolute_binding(shape, binding, owned_frame, owned_axis)` 校验真实owner安装与source后返回 `(owned_zero_motion, relative_points_f32, geometry_f32)`，后两项 owned readonly，分别 `[vertex_count,3]` 与 `[8]`。geometry=`[pivotXYZ, actual_world_axisXYZ, radius, engineering_padding_m]`。relative_points 由原零位世界材料点在 FP64 减真实pivot后上传，避免先将两大绝对数转FP32再相减；不替换原材料 IDs 或定义另一份 asset pose。

```python
resident_revolute_bounds(count, slots, offsets, zero_relative_points, geometry,
                         state, tick, bounds, stamp)
make_resident_revolute_bounds_kernels()  # 两个 accepted buffer 的独立 handle
```

`slots[S]` 为 rotor world slot、其他行-1；`offsets[S+1]` 保原有限顶点范围；`zero_relative_points[V,3]`、`geometry[S,8]` 是静态上传。球保1球心、胶囊保2有限端点、hull保全部原顶点。只根据 `state[slot,0]` 旋转这些相对点，再加pivot和真实radius/固定误差外扩，写已有 `bounds[S,6]`、`stamp[S,2]=(valid,tick)`；不改非rotor行。父首sphere/static pass应同时把slider/rotor排除出sphere slots，然后原slider pass与NEW rotor pass各写自己的行。同一个world多shape读同accepted q/tick，借用生命周期和失败事务沿 C1/C2。


执行前旧解析前提修正：临时 URDF 使用既有 loader 支持的 type='revolute'，limit 明示 lower=-4096、upper=4096、effort=8、velocity=4。原连续型标签没有进入本仓解析支持域，不能作为新 binding 验收的前置条件；本修正不改变任何实际 q、形状、惯量、控制或步长。DrivenRobotProgram 仍不接 compliant stops；这不是新增 continuous importer。
