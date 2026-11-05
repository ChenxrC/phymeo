# B4：场景切向候选与实际功验收适配

基线 `9e1693110880f23245d8d2e780d37316f1d7d21b`。本作者仅新增 `runtime/_coupled_tangent_step.py`、`tests/test_coupled_tangent_step.py` 和本报告；没有项目执行，没有修改旧 scene、材料律、A3/B3 或根导出。父拥有真实采样、完整自由速度、所有参与者的一次推进与共同发布。

## 冻结接口与边界

`prepare_tangent_step(store, history, samples, relative_mobility_per_kg, free_relative_velocity_m_s, *, sample_time_s, dt_s, stiffness_n_per_m, friction)` 接受严格按 `StationKey` 升序排列的当前入步样本。所有数组为 FP64：W 为 `(3n,3n)` kg⁻¹、自由相对速度 `(n,3)` m/s、kt/μ为 `(n,)` N/m/无量纲。W保留全部cross-block，先验有限、对称和PSD，再按B3的真实材料frame投影；正kt与非负μ，最多128站且不超过store容量。仍在同一episode的旧弹簧不能无功地更换刚度。

返回 trial 的 `keys`、`force_on_second_n`、`predicted_displacement_m` 保留全部输入行，包括N=0的行。`history_trial.keys` 和 `result` 仅覆盖N>0行；没有活动行时result为None，仍能结束旧episode、计一次释放并推进候选时间。无重新匹配器：B3负责身份/共转/重置，A3负责全耦合牵引；这里不复制两者。

`finish_tangent_step(store, history, trial, actual_material_displacement_m, actual_application_displacement_m, *, motion_tolerance_m, energy_tolerance_j)` 接受两组 `(n,3)` **真实相对位移**，都跟踪入步的原点到实际候选，不能换成末态重新找的最近见证。两个容差必须是caller显式给出的有限正绝对m/J界。分别验证材料增量、施力点增量与预测之差，以及二者之差；再核每站实际材料功、每站实际施力功和全体弹簧能量余额，不允许逐站误差互相抵消后冒过总门。

成功返回 `(history_candidate, receipt)`；失败抛ValueError，永远不发布false状态或消耗history。receipt保 `accepted`、`energy_residual_j`（总实际余额J）、`material_motion_error_m`、`application_motion_error_m`、`material_application_error_m`、`actual_work_j`（总实际施力功J）、两种逐站功、逐站能量残差和总释放能量。数组为拥有的只读存储；trial/核结果/基础history绑定原对象及数组形状、stride、dtype、字节，替换或NumPy元数据篡改在finish拒绝。对同一未发布基础重复finish是确定的纯候选；拿较新history或另一store重放被拒绝。

能量使用实际施力功：`work + E_new - E_old + plastic + implicit + released`。N=0/absent的旧能量由B3结束记录负责，正N且μ=0的旧能量由A3负责，二者不重复相加。B3累计work录实际施力功；正弹簧卸载功允许。实际位移误差小于m界仍可能不满足独立J界，这会拒绝，不能只读A3的motion accepted。

## 与父共享接缝的分工

父保留原mu下完整采样及其面积/材料/作用点选择；当次selected pairs逐site将原smooth tangent替换为normal+新tangent，按原作用点重组aggregate广义力。不能把旧law的friction设零改变采样，也不能在旧切力之上再加新切力。free velocity由外载、旧loads减去selected旧tangent、重力/gyro/PD-bias在同入态形成；本模块不猜这些项。各域从入态用最终合载只积分一次。

本模块审核完毕后，父还须通过原运动路径/几何能力门、拥有副本和events staging，才在scene原锁/事务内发布物理候选与history候选。普通RR、无内部投影约束particle、纯prismatic的精确平移路径是首批目标。一般旋转实际位移可能不等于hJv，测试要求明确拒绝该反例；不会宣称整个RobotScene已覆盖任意旋转静摩擦或原mimic已修复。

## 首次执行前固定的专项

- 两并列站点的世界W在x行全为1，h=.1、每站kt100、N4、μ.5、free vx=.3，解析每站−1 N、实际预测增量.01 m、每站储能.005 J。去掉cross-block会错为−1.5 N。
- 三行中间N0仍保force/位移原行地址；严格canonical输入；循环世界轴变换保力/功且保非零normal displacement。
- 旧ξ=.02 m、kt100的预载弹簧降至N1，返回+.0025 J实际功；卸载、空步、再入只释放一次，μ0加载行单独记账。
- 独立材料/施力点差门；两个各自偏差均在界内但相互差越界也拒绝。位移界放行时，逐站1e−5 J误差仍由1e−8 J能量门拒绝，包含两站总误差抵消的负控。
- 用真实RigidSemiImplicitSolver推进偏心点r=(1,−1,0)，h=1/8、m=I=1，入步ω0、末态ωz=1/32；actual exp-map与线性预测不同，history不发布。
- 同入态retry/旧base、不同store、替换或metadata修改trial、输入拷贝、非法world W（含只在法向为负特征值）、刚度跳变及非法绝对界。

## 实测状态

测试先提交 `056ebc2`；`9376416` 仅在实现前统一父scene需要的receipt字段和ValueError门。父真实首RED已保存 `artifacts/p65-scene-tangent-batch4-20261106-01/02-adapter-first-red`：指定第一个新节点，父无新模块的src产生ModuleNotFoundError，RAW1。作者未重跑。

作者交接时实现待父串行专项及静态检查；当时没有本批 GREEN 或新 Newton/CUDA 声明。最终父执行结果见文末，公共 scene 与独立核的边界继续分别记录。
## 父最终验收（2026-11-06）

作者交接后，父唯一串行执行 06-adapter-first-functional：31 passed / 0.55 s / RAW0。最终 17 文件 Ruff/format 均通过，style-equivalence 保非 import AST 和 import 绑定。父已接入真实双面压载、URDF 共享祖先双站点、卸载重入和失败回滚；源码外 wheel 的两质量各 48 步公共例程通过。

实际产品范围以 [第四批综合验收](p65-scene-tangent-batch4.md) 为准：只接 HOST 刚体和普通纯移动关节；本批未接无内部投影粒子，也未接 mimic/revolute。原默认摩擦保留。公开 scene receipt 的 ndarray 元数据别名有独立红绿验证，当前 getter 返回拥有副本。该结果不代表任意旋转或原 mimic 抓持已经通过。产品提交 8c4a7db；父执行槽已释放。
