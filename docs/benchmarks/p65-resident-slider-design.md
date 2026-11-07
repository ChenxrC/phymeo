# C2 设计：接受态单滑轨到设备宽相

状态：DESIGN ONLY，未授权实现或执行。本文只基于父树 `8546302b7d7cbd4585bb7a20ed4ef82bca304fc6` 的现实现；该树已含 C1 registration 修复。作者树保持 `295e448` 基线，不合并父后续头，不修改 C1 四个文件。本设计须在本阶段 A/K3、组合审阅和 wheel 验收后另行冻结。

## 复用依据与首片边界

- `_world_step_layout.py:98,127,205` 的 PrismaticSlider/WorldBinding/bind_world 已支持真实单活动 prismatic、无物理限位、delay_steps=0/1。接受缓冲第0列为 q（m）、第3列为 qd（m/s）；WorldBinding.axis 是固定世界轴，当前没有保留关节原点/固定朝向。质量、重力投影、PD 和外载投影已经在原通道，不重做动力学。
- `robot_kinematics.py:36` 明确单根滑轨的 `position=o+q*a`、`rotation=R`，其中 o 为 joint_position[0]，R 来自已归一的 XYZW，a=R*normalized(joint_axis[0])。根 parent=-1 对应 identity world，不额外乘另一份 root transform。
- `_robot_asset.py:58` 的 collision_motion(q) 是实际资产姿态入口。特别是 `_urdf_frames.py:20` 的 FoldedURDFAsset：collider.local_vertices 仍属具名 authored link，必须经 sites/固定帧/主惯量基变换；不能把这些点直接当模型连杆点。C2优先直接消费 `asset.collision_motion([0])` 的真实零位世界几何，避免再次旋转局部碰撞 origin。
- C1已有同锁 accepted buffer 借用、epoch、registration、资源回收、设备 bounds→原 BVH。`tyx_world_batch.py:222` 与 `_resident_collision_plan.py:311` 保持为唯一事务/版本协议。本轮设计不重推 C1，也不宣布 C1 已经覆盖 slider。

首片每 world 仍只有一个现有物理域：FreeSphere 或单活动滑轨，不能在同 world 凭绑定增加物理体。滑轨可有多个有限碰撞形状，但必须都归同一个真实移动 body；形状可以是球、有限胶囊、hull/mesh，以及既有单姿态包络能表达的固定朝向有限几何。静态根几何/台面仍用 ResidentStaticBinding。只查当前接受姿态与同 world 静态几何；不增加接触响应、CCD、转动关节、多滑轨串联、mimic/浮根或变形面驻留。物理限位和 delay>1 仍按原 resident loader 拒绝，不因资产带有限位元数据而冒称已执行限位。

## 可编码的绑定与帧合同

建议新增不可变类型（单位 m、局部单位轴、XYZW）：

```python
SliderGeometryFrame(origin_m, orientation_xyzw, axis_local)
ResidentSliderBinding(
    key: CollisionKey, world: int, zero_motion: CollisionMotion,
    *, origin_m, orientation_xyzw, axis_local,
)
```

`zero_motion` 必须是 **q=0 的世界姿态**，无 end/turn，不是调用时 q、不一定等于初始 q；其曲形 body_orientation 也已在世界固定朝向中。它包含真实 local offset、有限长度、原 vertex IDs 和局部形状旋转。调用者可以用普通程序式几何定义固定安装的形状，也可从资产的 `collision_motion(np.zeros(1))[source_key]` 直接取得；target key重映射须同时用于 CollisionModel 和 binding，形状内容/revision与材料ID不能顺手重编号。固定折叠资产同样使用其公开 collision_motion，首片不自行复制 authored→principal-frame 变换。

父在 bind_world 处理真实 PrismaticSlider 时，向 WorldBinding 追加可选 `slider_frame` 元数据（FreeSphere为None），从原 program.model 复制 o、归一 XYZW、归一局部轴为tuple，不保留可写 RobotModel 数组别名。既有 WorldBinding.axis 仍是实际物理投影使用的世界轴。建议不改变旧 state/props 列、动力学核、驱动/延迟包和已有 checkpoint 字段；几何帧不随 restore 改变。C2准备时强制核对传入帧与该 owned frame，实际计算使用 owned frame/axis；同一旋转的 q 与 -q允许等价。缺少真实 frame 的历史/自制 binding 不能自动推断。

每个 world 的 slider bindings 可以有多个 shape，但 target key.body必须相同；禁止把一个q伪装成两具不同物理body。所有key必须恰覆盖、world必须对应实际kind=1/2。普通URDF的根 collider link_index=-1必须静态；活动组上的固定工具由正确的零位路径承接。`origin_m` 是模型连杆零位原点，不是COM；偏心COM不被加入几何两次。

对任何材料点，正式映射为：

```text
p_world(q) = p_world_at_q0 + a_world * q
等价物理式：o + R * p_model_local + a_world * q
```

固定朝向使整形状 AABB 只平移。准备阶段在HOST复用既有包络代码生成零位 outward bounds，并为实际 FP64 a_world 各分量存向外的 FP32区间。设备从当前 accepted state[slot,0]读真实q；按q正负选择轴区间乘积，完成有界外向乘加，直接写现有engine._bounds。不得仅把a向nearest FP32取整后声称全域包围；非主轴及负q会让该捷径产生假净空。运算预算必须覆盖乘积、相加、FTZ和大项抵消，使用操作数规模而不是只看最终很小的端点；溢出/无法给有限包络必须拒绝整次query。独立oracle同时用公开FK/资产路径与解析几何，FP64组装舍入和FP32输入表示差明确列出，不要求bounds逐bit相等。

## 父旧接缝与建议文件边界

未来实现候选 NEW allowlist：

1. `src/phymeo/backends/_resident_slider_binding.py`：上述frame/binding、绑定校验与零位包络/轴区间编译；不创建Runtime。
2. `src/phymeo/backends/_tyx_resident_slider_bounds.py`：同源TYX滑轨行更新核，每个accepted-buffer方向独立句柄。
3. `tests/scale/test_resident_slider_collision.py`：下表独立测试；不重跑C1已绿全集来替代新能力证据。
4. 本文后续只在授权后追加实际测试/边界；当前只有这个文档被创建。

父独占修改：`_world_step_layout.py`保存真实frame；`_resident_collision_plan.py`接受slider binding、允许同物理body多shape、选择slider核、检查同world各shape的tick一致并一次发布。`tyx_world_batch.py`的prepare/borrow接口不必另造，view.bindings即可带frame；如需薄导出仍父处理。`tyx_collision.py`与BVH可直接复用已接通的 device bounds入口，预计无需改遍历协议。C1 kernel保持原样：第一pass处理sphere/static并为slider填占位零位盒；在读status/BVH之前，第二pass仅覆盖slider行的bounds/tick/status。两pass借同一个accepted view，最终只一次结果发布；不复制C1的球公式，也不另造resident runtime。

当前不改既有WorldCheckpoint的物理兼容语义；本切片的几何所有权由owned frame和每plan registration约束。未来若要求跨不同模型原点/朝向的checkpoint可移植性，需要另外版本化frame协议，不能宣称C2已提供该能力。

## 预先固定的独立测试表

所有输入在实施前固化；下列数值为手推，尚未执行。sphere/box/capsule都为真实有限注册几何，宽相候选不等于接触。

| ID | 固定输入 | 独立期望与失败门 |
|---|---|---|
| S01 首RED/真实出步 | m=2kg，o=(1,-1/2,1/4)，R=Rz(π/2)，局部轴+X；q0=1/8，v0=0，g=0，kp=kd=0、帽4N、ff=2N、h=1/4。局部box中心(1/4,0,0)、半长(1/8,1/16,1/32) | 真实v1=1/4、q1=3/16；box世界x∈[15/16,17/16]、y∈[-3/16,1/16]、z∈[7/32,9/32]。静态box x=[15/16,17/16]、y=[1/16,3/16]、z=[7/32,9/32]：初态有1/16净空，出步恰触必须候选；把q当世界X/把初态当q0会失败。先通过原滑轨推进，再到NEW binding/准备入口记录有效缺口RED。 |
| S02 非主轴/负q | R=I，a=(3/5,4/5,0)，o=(-1/2,1/4,0)，球local center=(1/8,0,1/16)、r=1/32，q=-1/4 | 球心=(-21/40,1/20,1/16)，bounds x=[-89/160,-79/160]、y=[3/160,13/160]、z=[1/32,3/32]，必须外包。q翻为+1/4时心=(-9/40,9/20,1/16)。静态box x=[-79/160,-69/160]、y=[-1/16,3/16]、z=[0,1/8]恰触；再仅向+X移1/64为净空负控。检测signed interval和轴FP32误差。 |
| S03 真外载投影 | m=2kg，a=(3/5,4/5,0)，q0=v0=0、g/PD/ff=0，世界COM外力F=(0,5,0)N，h=1/8 | 旧动力学实际广义外载4N，v1=1/4、q1=1/32。几何位移为(3/160,1/40,0)，不是(0,5,0)或q直接XYZ；力不在几何核二次施加。 |
| S04 同body多形状/工具帧 | 一个真实slider上同时注册偏心sphere、局部±1/4有限capsule和偏置hull；另用一个固定工具折叠URDF的一活动滑轨资产，非零q | 每shape逐材料点/完整bounds对公开asset.collision_motion(q)，source key重排、vertex IDs不变、固定工具偏置只应用一次。全shape同body/同tick；刻意绑不同body、root当活动体、wrong frame必须在设备分配前拒绝。 |
| S05 40步非重置/延迟0与1 | 独立X滑轨m=2、g=0、kp=kd=0、帽4N、恒ff=2N、h=1/16、q0=v0=0；box半长1/16，有限障碍x∈[1,17/16]且覆盖其y/z | delay0的v_n=n/16、q_n=n(n+1)/512；40末q=205/64。delay1首步不动，v_n=(n-1)/16、q_n=(n-1)n/512；40末q=195/64。障碍候选只在delay0的第22/23步、delay1第23/24步；第21及24（delay0）明确分开。保持完整路径无reset、无回填q/v。 |
| S06 W1/W4/W32与同ID异world | sphere和两种不同frame的slider按world混排；重复body/shape局部ID；同slot多shape与静态root/table | 与逐world同态HOST oracle比较全部candidate keys、bounds包含、ticks；无跨world pair。同world多shape的tick必须一致，否则拒绝整份读回。两个不同零位几何plan同owner/epoch不能互相require_current（复用C1 registration门）。 |
| S07 同tick restore/局部拒步/回滚 | 从本owner保存checkpoint，restore同tick但q不同；随后reset。邻world继续正常步；单world用MAX_TICK真实拒步；另注入一次bounds/BVH失败 | restore/reset epoch递增；当前盒跟真实接受q改变。拒步world状态/tick/几何保持，邻world正常；query失败不能换last_result、不能推进物理epoch。若驱动无法恢复同步，继承原unavailable门，不能假装可继续。 |
| S08 资源/表示/失败合同 | 两accepted-buffer反复交替；空plan、仅slider无静态对、capacity0、close多plan callback异常；正负小轴、大坐标抵消/溢出 | 暖态无snapshot/kinematics读动态状态、无dynamic envelope上传、无新增device alloc。检查实际TYX核/传输与计数，metadata/status/bounds/candidate读回照实保留。不可外包/容量/错world/missing frame/非有限输入都不得发布半表；close不能关borrowed batch runtime。 |

S04资产留出固定补充：使用一活动prismatic和一个固定工具的本地URDF字面夹具。活动关节origin=(0,0,1/4)、R=Rx(π/2)、axis_local=+Y；活动link质量2kg、COM=(1/16,0,0)、COM主惯量各1/8 kg·m²；固定工具相对活动link平移(1/8,1/16,0)、Rz(π/2)、质量1kg、COM=0、主惯量各1/32。活动link注册local center=(1/8,0,0)的球r=1/32，以及标准URDF cylinder（r=1/64、length=1/2、局部Ry(π/2)，不可写成非法URDF capsule）；固定工具注册半长(1/16,1/32,1/64)的box。低层同body胶囊留出使用程序式合法CollisionShape单独构造。g=0、q0=0、v0=1/4、无驱动/外力，h=1/4接受后q=1/16；由标准固定链折叠入口生成实际主轴frame，再与其公开FK几何对照，不能手填预想的折叠后R。

S06的world keys为11+17*i，body/shape局部ID重复；W1选S01滑轨，W4/W32按FreeSphere、S01滑轨、S02滑轨轮转（每world只一个物理域），输入序反转留出。S08的大项抵消固定用a=(3/5,4/5,0)、q=-2²⁰、零位点=(3*2²⁰/5,4*2²⁰/5,0)，r=1/32；独立Fraction参考应包含零位抵消后的球。溢出负控用轴+X、零位球心x=2¹²⁷、q=2¹²⁷，真实物理q可表示而世界界超FP32范围，须明确拒绝query、不替换旧last_result。
数值门不按首次输出调节：解析场景先给真实FP32 q/v的精确二进制oracle；几何采用包含真实FP64参考的外向区间门。表中非二进制轴/旋转另记录输入表示和FK舍入界，近失1/64远大于此误差。S05的固定候选步号仅为该独立几何窗口的离散姿态oracle，不能解释成连续首触时刻。

## 验收次序与依赖

未来父串行：先S01有效RED，再HOST绑定/所有权协议、真实interpret全专项、真实CUDA全专项；每端验证两buffer和W1/W4/W32，保首错。最后仅必要旧接缝相邻验证及源码外wheel，不能用HOST数值通过代替设备核或用现有C1结果代替C2。功能验收后才考虑一次小型AB/BA传输/墙钟留出，没有收益不改默认或追调输入。

C2不依赖转动逐面候选B：这里无转动且只产当前姿态宽相，B仍用各时间样本的真实姿态。未来接入通用接触适配仍需要相同版本的完整姿态/材料地址及实际法向、速度、载荷通道；本宽相bounds不能直接充当施力见证。旋转/整段包络以及接触响应是后续独立节点。
