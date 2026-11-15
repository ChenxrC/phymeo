# V05 动态：有限支撑上的推动、俯仰与边缘退出

日期2026-11-16，codex/exp。承接已验[V05静态](../../development/area-pressure-finite-support-report-20261115.md)和[V06双自由体](../../development/area-pressure-impact-report-20261116.md)，不重跑两组。本批是独立NumPy平面挤出压力场参考：一个3DOF可运动体与一个固定有限支撑，通用凸截面，非完整3D、TYX或产品集成。

## 研究依据与模型选择

[Elandt等2019 PFC论文](https://arxiv.org/html/1904.11433)第III/IV节和附录X-A给出压力牵引与重叠体积压力势能；刚性目标极限的接触面是刚性物体边界。本文据此选择以下二维挤出实现，并独立检查力/力矩是势能的负梯度。不是对Drake/Newton代码或数值结果的复现。
局部凸CCW截面有3..8条边，各内向单位法线n_i和距离d_i。体内p_e=kappa*min_i d_i，边界p=0，体外无压力。y方向宽度w恒定，假设沿y均匀、不模拟前后端面。固定支撑同为3..8边凸CCW多边形。接触覆盖支撑所有边在可运动体内的部分，法向为支撑外法向。面积权重dA=w*ds，绝不把归一权重当面积。

势能U=w*积分(body与support交集)p_e dx dz。最近边cell由d_i<=d_j裁剪，cell∩support内p为仿射，用三角形面积×质心压力积分。所有cell完整覆盖；共享边零面积。接触线段先裁到body，再按所有d_i=d_j根分段，每段2点Gauss，避免整段处于tie上时重复积分。该积分精确处理分段仿射弹性合力和二次力矩；速度依赖摩擦只是同一Gauss规则近似，不宣称任意速度的精确积分。
与旧V05顶面仿射基础不同，边缘/侧面压力和材料绑定随刚体转动一起更新，不能继续把旧顶面势能直接套到任意转角。

令q=(x,z,theta_y)，R(theta)(u,w)=(cos(theta)u+sin(theta)w,-sin(theta)u+cos(theta)w)。接触点相对COM为r，则速度(vx+omega*rz,vz-omega*rx)。B_n=(nx,nz,nx*rz-nz*rx)，B_t=(tx,tz,tx*rz-tz*rx)，t=(-nz,nx)。广义接触力为sum dA*(B_n*p_n+B_t*p_t)。
采用[Drake compliant contact](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)同形式的压力耗散p_n=p_e*max(0,1-d*v_n)；p_t=-mu*p_n*v_t/sqrt(v_t^2+v_s^2)。这是正则化滑动摩擦，无精确静摩擦锁止。p_n>=0，每点(p_n-p_e)*v_n+p_t*v_t<=0。参数并非直接的杨氏模量。

## 固定案例与工程门（运行前冻结）

原V05质量0.5kg、40×40×20mm体、80×80×10mm支撑、mu=.5、kappa=1e7Pa/m、原初压入delta0=.0003065625m保留。截面body=[(-.02,-.01),(.02,-.01),(.02,.01),(-.02,.01)]；support=[(-.04,-.01),(.04,-.01),(.04,0),(-.04,0)]，w=.04m。Iy=m*(.04^2+.02^2)/12。q0=(0,.01-delta0,0)，速度全0，gravity9.81m/s²，d=20s/m，v_s=.001m/s。新边界零压力使初载荷略低于mg，这是预先声明的模型差异，不重新标定delta。
窗口T=.6s，h=.001/.0005s共600+1200步。COM水平外力4N在[.2,.4)s，其他时刻0，无外加力矩；重力全程。前.2s承载，然后推动，撤力后继续运动。时间程序只是已声明外力输入，不按物体坐标修正状态。两h使用步中点判定外力，相位边界与网格一致。

每条必须完整结束，所有有限量合法、压力场最大重叠深度<=2mm；无悄悄截断/深度clip。前.15..2s平均Fz在mg±10%，z范围<.1mm且|vz(.2)|<1mm/s。最大x位移>60mm，出现|theta|>1度，存在正面积/正力，末.05s接触面积与力为0，末COM低于初态至少10mm。实际接触面离开后不补无限平面支撑。
每kick实际广义动量变化与保存接触+外力冲量误差<=1e-9 SI；保存q漂移/平均力绑定<=1e-12；D>=-1e-10J，完整双方力与共同原点力矩相消<=1e-10。E=K+U+mgz，push外功来自实际kick中点速度，累计D来自实际接触律；全窗|E+D-Wpush-E0|<=.05*max(.01,max_abs_cumulative_Wpush)J。该误差不补状态。
两h峰值整步接触力模长、累计法向冲量差<=10%（以细h绝对值作分母，零值单独处理），最后接触时间差<=.01s。终态差如实报告，不因高敏感转动追加调参/更小h追绿。

## 时间离散、通用API与资源

KDK：entry位置halfkick → 用half速度完整漂移h → exit位置halfkick。每halfkick tau=h/2冻结geometry，解w-v_in-tau/2*M^-1*(F_ext+F_contact(w))=0，v_out=2w-v_in。3×3解析Jacobian包含法向耗散、摩擦对vn/vt的交叉项。最多24 Newton步、每步12次二分回溯，要求按.02m缩放角速度的residual max(|rvx|,|rvz|,.02|romega|)<=1e-12m/s；不满足则该步拒绝，无半步发布。深度或非法状态同样事务拒绝。每kick保存全部方程字段，不能只保存末端瞬时力冒全步平均力。
支持任意转角；坐标绝对值<=1m，h<=.001s；多边形严格凸CCW、无重复、有限非bool数值，m/I/w/kappa正，d/mu>=0,vs>0,depthcap>0。边长>1e-12m，几何清理length1e-12/area1e-18；polygons最多40裁剪顶点，每目标边最多29分段，全部<=256 segments/512 quadrature points。所有退出、失败不改变输入。

新文件experiments/area_pressure/planar_reference.py仅原scene作者；新tests/test_area_pressure_planar.py仅原独立test作者。父写run_planar_cases.py、planar_viewer.html、报告/计划。原reference/finite_support/friction/impact及旧测试均不改。可复用finite_support_reference低级_polygon/_clip/_area，仅文本确认合同后复用，snapshot绑定其字节。
冻结公共名称：
- PlanarConfig(body_polygon_xz默认上述body,support_polygon_xz默认support,width_m=.04,mass_kg=.5,inertia_kg_m2=1/12000,stiffness_pa_per_m=1e7,max_depth_m=.002,dissipation_s_per_m=20,friction=.5,slip_speed_m_s=.001,gravity_m_s2=9.81)，frozen且规范化immutable tuples。
- PlanarState(time_s,x_m,z_m,angle_rad,vx_m_s,vz_m_s,omega_rad_s)，全部必填。
- sample_contact(config,state) -> PlanarContact: state,body_world_xz,points_xz,normals_xz,area_weights_m2,pressure_pa,source_edges,area_m2,potential_j,max_depth_m,elastic_generalized_force（三元Fx,Fz,My）。
- contact_wrench(config,sample,velocity=None) -> PlanarWrench: generalized_force（三元）,support_force_xz,support_moment_origin_nm,normal_pressure_pa,tangent_pressure_pa,normal_velocity_m_s,tangent_velocity_m_s,dissipation_power_w（非负）。velocity默认sample.state的三速度；reaction moment用同一世界原点。
- step_body(config,state,dt_s,external_force_xz_n=(0,0),external_moment_nm=0) -> PlanarStep: entry,drifted,exit,first,second,contact_impulse,average_contact_generalized_force,dissipated_j,to_dict()。
- PlanarKick: sample,duration_s,velocity_before,velocity_after,mid_velocity,contact（PlanarWrench）,external_generalized_force（包含gravity）,contact_impulse,external_impulse,dissipated_j,external_work_j（全部外力含重力的tau*F_ext·w_mid）,iterations,residual_m_s。轨迹Wpush必须另用已声明用户外力/力矩求中点功，不能重复计重力。
- mechanical_energy_j(config,state)：K+U+mgz，仅诊断。
各输出frozen、to_dict完整可JSON，无ndarray可变泄露。

## 独立验证、保存和停止

短测试先独立author再执行：完整平面解析F=kappa*w*(L*delta-delta²)及U=kappa*w*(L*delta²/2-delta³/3)，宽度缩放/反向/平移协变、边缘侧力/转矩、旋转/非矩形凸体、tie不双计、空域、压力势能对x/z/theta独立有限差分；中点kick残差、反作用力矩、D、friction0/重力0自由漂移、第二sample越域原子性、输入/caps/readonly。有限差分仅独立测试，不驱动求解。完整1800步仅runner一次，不重跑既有V00–V06。

每步保存完整入步/漂移/出步、两个kick及点面积/法向/pressure/速度/冲量/能量，原始失败也保存；本批点数小允许全JSON。先保存两case manifest，再执行。summary/CSV便于视觉，画面读取实际轨迹；视频标真实时间、受力、角度和失去支撑，不做规定位置动画。
root唯一串行owner：fresh guard→环境→首lint/format→必要纯style→新短HOST→两真实窗→保存核→媒体。作者无执行slot。仅允许一次明确诊断的实现修复；模型/物理参数/门槛冻结，不调整追绿。若长窗失败，保存并解释失败段，不把几何通过当动力学通过、不重新跑同窗。完成分组commit/push exp，dev保持，按照R4证据选后续集成。

实现补充：max_depth_m取整个body∩support内min(d_i)的最大值，从每个最近边cell裁剪结果全部顶点求max；不能仅看接触Gauss点。area_m2定义正压力接触段闭包的面积，整段p_e恒0不生成points或area。source_edges每点为(support_edge_id,body_nearest_edge_id)，整段tie按最小body edge ID归属而不双计；contact_impulse为三元广义冲量。累计法向冲量独立按sum tau*sum dA*p_n，不用Fz冲量或向量模长替代。support_moment_origin_nm=-(My_COM+z_COM*Fx-x_COM*Fz)。有界Newton失败不暗示全局收敛，不保证固定两个长窗预先通过。

## 2026-11-16 执行收口

数值源码052efc4b；首次20项新HOST、两条完整0.6秒轨迹（1800步）均PASS，无数值修复/参数调整/窗口重跑。实际承载4.905N、最大深度约0.99mm、接触约0.382s结束，side contact两窗均出现；两h峰值差0.01168%，累计法向冲量差0.001039%。完整保存148文件/12事件和36source pairs审计PASS，详[实际报告与视频](../../development/area-pressure-planar-dynamics-report-20261116.md)。首静态失败和纯样式修复保留，三文件raw AST不变。
root唯一执行及作者均已结束，下一接触节点为R4范围/三维表示/I0–I2接口；本2D挤出参考不冒产品集成。原V03外部FAIL及T-smoke独立状态保持，所有本批已绿组不重跑。
