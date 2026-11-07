# C2 实施：真实单滑轨的接受态设备宽相

状态：**ACTIVE / CODE_AUTHORING**。用户已授权持续推进；前置A/B/C1与N4已在b2d8ec9正式限域接受。无截止。依据[作者设计及S01–08](../../benchmarks/p65-resident-slider-design.md)，本页补充父裁定与文件/接口；不是新数值通过。

## 目标与验收

以已有PrismaticSlider真实接受q，在原TYX runtime/lock/epoch/registration中直接生成有限shape AABB，再交现有设备BVH。与FreeSphere混合、多world、同滑轨多shape均可；不新增物理body、响应或积分。零位几何是q=0的完整世界姿态，不是初态、不再额外乘一次安装变换。静态根绑定保持静态。

S01–08原尺寸/质量/轴/窗口/负控全部保留。HOST绑定与独立oracle，实际interpret、CUDA串行；保留真实q/v、全部bounds/candidates/ticks和设备传输计数。两buffer暖态无动态snapshot/envelope上传/新增分配，必要metadata/bounds/candidates读回照实计费。最终交叉审阅与新源码外wheel。碰撞候选不冒接触，更不冒全GPU机器人。

## 独立作者与父接缝

原robot agent在 p65-robot-substep 隔离树，start815e44b clean；仅NEW：
1. src/phymeo/backends/_resident_slider_binding.py
2. src/phymeo/backends/_tyx_resident_slider_bounds.py
3. tests/scale/test_resident_slider_collision.py
4. docs/benchmarks/p65-resident-slider-collision.md

先测试源码再实现源码。原C2proposal及C1文件保持，不merge父、不改共享。作者只能文件/Git/手推，不执行import/collection/数值/pytest/lint/format/build。报告精确API/限制、源码交接和CODE_READY_TESTS_PENDING，父另排唯一执行槽。

父独占 _world_step_layout.py、_resident_collision_plan.py，必要 tyx_world_batch.py 薄导出；旧C1球核、BVH和动力学/延迟核保持。新作者交接后父接hook，再开始有效RED→实现→真实验证。禁止仅missing module的导入错误冒行为RED。

## 公共数据与低级hook冻结

NEW binding模块不导入 _world_step_layout 或 _resident_collision_plan，避免循环；也不创建TYX runtime。

- SliderGeometryFrame(origin_m, orientation_xyzw, axis_local)：冻结owned tuple，有限形状正确、单位四元数/非零轴规范化；旋转q与-q视同，原点和归一轴保持同值。父从真实program.model拷贝到 WorldBinding.slider_frame，默认None为FreeSphere；物理投影仍用既有owned world axis，state/props列及checkpoint原物理兼容合同不变。
- ResidentSliderBinding(key, world, zero_motion, *, origin_m, orientation_xyzw, axis_local)：owned声明；world/key/finite单姿态/完整材料数量必须匹配。传入frame与实际owned frame核对，实际计算只用owned world axis，不能使用caller伪轴。
- compile_slider_binding(shape, binding, owned_frame, owned_axis) -> (zero_motion, axis_bounds)：axis_bounds为owned只读float32 (6,)，前3为下界、后3为上界，包含实际FP64 owned axis；parent复用 _envelopes/_host_bounds 编译零位完整包络。不复用任何当前q的HOST路径作为动态结果。
- ResidentSliderBinding.from_asset(asset, source_key, *, key, world)：从真实单prismatic资产公开collision_motion(zeros(1))取零位姿态，并从asset.model取frame；source collider必须属活动group0，固定root拒绝。FoldedURDFAsset须通过其sites[col.link_name].link_index判owner，不能仅信原authored link编号。shape重映射/模型注册由调用者显式完成，保原材料ID/revision/形状内容；直接构造binding则是调用者显式声明的安装几何，不宣称从任意外来资产证明了真实归属。

NEW kernel模块：
- make_resident_slider_bounds_kernels()返回两个独立tx.kernel句柄（按accepted active buffer）。
- kernel参数顺序 count, slots, zero_bounds, axis_bounds, state, tick, bounds, stamp；slots为每shape的slider slot或-1，axis_bounds(N,6)，其余沿C1。只更新slot>=0的行，保留其他行；写bounds(N,6)与stamp(N,2)的valid/tick。
- q=state[slot,0]；signed axis区间乘q，再加零位区间。所有乘加明确RN，保守覆盖FP32舍入/FTZ；不能只对最终抵消的小数按相对误差加margin。非有限/overflow记录invalid并令整份query拒绝，不发布半表。设备计算不得硬编码S01–08坐标/时间或回写物理态。

父以C1第一pass处理sphere/static，slider行临时静态zero盒；同一borrow内第二pass覆盖slider行，在任何status/BVH读取前完成。所有shape同world同body绑定核实，同world所有活动shape的tick必须相同；旧一world一FreeSphere限制保持。两个核类别/实际传输分别计数，query/result发布一次。closed、epoch、registration、失败旧last_result及runtime所有权复用原协议。

## 执行与停止空转

父唯一串行owner，每条项目命令fresh guard。首S01先推进旧slider成功再调用新准备入口，记录真实缺口；之后只运行新全专项及受影响旧接缝必要节点，不重跑整个C1绿集。全绿后一次本切片wheel；无远程CI/安装/改pin。

如果两次修订没有新增可运行能力或缩小具体失败，就缩小最小复现并换路线，不能继续扩大数学条件。性能只在功能通过后按同结果测量；没有收益也可以结束此功能节点。后续旋转机器人/柔体驻留、动态–动态碰撞响应、全设备窄相另列，不把当前滑轨包装为完成。
