# P6.5 步初共点机器人—自由节点双向重放设计

状态：在用户持续P6.5授权下由实施者裁定的下一小切片。沿用codex/p6-5-handoff；不重写已验收平面，不改tyx pin。此阶段给出真实机器人与节点成对动态反作用，但不冒充完整布材料接触、自动新接触检测或持续旋转夹持。

## 意图与路线

最终J1配相反节点冲量的反例已证明角动量错配。考虑三路：最终构型连续重放（现反例不合格）；改成变分/中点积分器（会同时更改原机器人和材料积分合同，暂不在本切片做）；步初共点的接触求积，载荷以J0一次映射，原积分器完整重放（本次选择）。由实施者选择第三路，先验证一个封闭轴向角动量的真实旋转例，再向事件/几何推进；不宣称任意时间离散皆守恒。

这是一阶、步初已闭合接触的速度冲量离散：接触相对速度用J0*v1；它是步初配置上的后速度，不是假装最终材料速度J1*v1。末几何单独验收非穿透，正冲量后允许在本步分离。初态未闭合拒绝，交由后续几何/CCD找到事件；不使用预测的未来点提前施摩擦。

参考的路线分类仅来自原作者资料：[Fetecau等的变分碰撞积分](https://authors.library.caltech.edu/records/9n8hg-0sa81)与[显式处理曲线路径的Rigid IPC](https://ipc-sim.github.io/rigid-ipc/)。本实现自主推导，不复制上述算法或代码，也不以文献结果代替本阶段测试。

## 公开接口与支持边界

RobotNodePlaneContacts(node_domain, node_indices, robot_indices, local_points, local_normals, friction)：每行一个自由节点与一个机器人link的无限材料平面；local_points为平面上点，local_normals单位法线，均FP64(k,3)；indices为非负整数(k,)，friction为非负FP64(k,)。深拷贝只读，与已有contact绑定同风格。世界法向从机器人指向节点。

InitialRobotNodeContactProgram(scene, base_program, contacts, *, robot_domain='robot', registry=None, inner_settings=None, relaxation=1., reference_angular_speed=1.)。支持一个RobotModel和若干ParticleModel；所选粒子域全部节点inverse_mass>0；base数学、FP64存储，后端numpy或现tyx三模式。registry复制/缺省robot注册、inner物理尺度相等、substeps=1、借用factory/close/history等合同与RobotPlaneContactProgram相同。

选中的粒子域本切片必须遵守自由节点半隐式Euler方程：不连接材料约束、其他接触或额外速度后处理。每个实际候选全域q/v原式审计，不符拒绝；不通过检查base具体类型猜测。未被接触选中的其他域可照原程序独立演化，包括材料旁域。完整布材料双向接触是后续必要工作，不会被本局限删除。

## 几何与离散账本

每子步q0处计算link平面，初始节点到平面有符号距离必须在原length_tol=tol*min(ref_length,h*ref_speed)内；正间隙也拒绝。将节点投影为该link的局部接触材料点，冻结其J0/世界点/法切基。对平面锚点和投影材料点使用已有独立仿射门禁，COM仍检查；节点与机器人初始作用点误差按原长度预算，不以世界原点放宽。

M0/b0/A0沿用FrozenRobotLedger。节点p为正，机器人G=-sum J0.T*p。节点同域/同点先精确合并，机器人全部广义载荷只合并一次。通过SceneLoadReplay的effort/force积分通道完整重算，原状态/history从同一初态重试。无第二次kick/位移修正/速度重建。

每次实际返回审计：机器人M0*(v1-v0)=h*(A0-b0)+G、q1=q0+h*v1；节点v1=v0+h*g+h*inverse_mass*original_force+inverse_mass*sum(p)，x1=x0+h*v1。实际输入差*h与请求积分精确比较。运算使用存储量有理数、残差向上包络到float；机器人角/线速度按既有独立单位。节点全域拒绝不符合自由方程的base；p=0自由结果也审计。

接触相对后速度u=v_node1-J0*v_robot1，法向pn>=0,un>=0,pn*un=0；pt静止落在mu*pn圆盘、滑动沿-ut，单静/动mu。GS使用固定K=inv_mass*I+J0*M0^-1*J0.T作为预条件；同域交叉项通过每次全部载荷重放保留。零/非法响应拒绝。合并后的真实原式决定收敛。

冻结总动能为K_M0(robot.v)+所有所选节点域的动态粒子动能。相对p=0实际native自由候选的变化直接审计<=tol*ref_speed*sum(abs(world_p各分量))，末求积功sum(p.u)也满足此界。背景速度不能放宽预算。该冻结门禁不是完整多关节M(q1)/重力势能守恒。

同点成对载荷矩：每对cross(x_node0-x_robot_point0,p)的L1和不得超过length_tol*sum(abs(p))，无接触力矩地板；线反作用严格正负同一p。固定基座可承担约束反力，不能对所有世界动量分量要求封闭。独立测试须选无基座z力矩的定惯量单转子，检查实际整步Lz，不能仅查配对冲量。

末态重新FK/平面n1与锚点，所有节点gap1>=-length_tol；位置审计不要求pn*gap1=0，因为冲量位于步初，随后允许分离。无需更改原q积分，但末穿透则整帧失败，不自动改容差/子步或冻结末位置。连续路径安全、特征切换不在本阶段认证。

## 核心独立测试

1. Y移动机器人m2、节点m1，初始共点、机器人v0=0、节点vy=-1，零g、mu0：p_y=2/3，双方vy=-1/3，q_robot=x_node_y=-h/3；实际总Py=-1守恒，动能从.5到1/6。
2. 单Z转子根原点、COM0、I=1，初q0=0,w0=0，平面锚点(1,0,0)、法线Y，节点(1,0,0)、v=(0,-1,0)，h=.1及.05：p_y=.5，w1=-.5，vy1=-.5，q1=-h/2、节点(1,-h/2,0)。Lz恒为-1，动能.5→.25；末间隙sin(h/2)-(h/2)*cos(h/2)>0。J0与J1必须可区分。原最终J1反例继续作为拒绝该候选方案的独立负例，不把它“修成”同一初态。
3. 单转子共享两点x=1和2，节点vy0=-2,-3，m各1：p=2/3,1/3，w1=-4/3，vy1=-4/3,-8/3。两顺序相同，Lz初末-8；独立检查全域合并和能量。
4. 移动机器人＋节点正常/低mu滑移（u_t>0必须pt=-mu*pn），自由节点之外的材料旁域独立推进；T2s持续共面prismatic支撑（原外力作用，逐步解析），两步长一致。
5. 分离初态、超容差穿透、后候选穿透、错误材料/后处理、巨大背景数、几何丢失、同域多点重复载荷、history/后段子步/关闭失败全恢复；输出污染/close所有权。
6. actual interpret/CUDA最小移动与转子，实际launch；C++仅2次重放有限短例，不重跑TYX-003压力。完整相关/核心/CUDA/wheel/uv-pip/一次fresh review/串行观测和专用分支push照旧。

诊断initial_robot_node_contact：初始世界作用点、平面、J0、p/平均力、节点/机器人积分、初配置后相对速度与最终真实材料速度分别命名、初末gap、配对矩误差、冻结总动能变化、求积功、重放次数、inner_report、native和canonical后端；所有对象owned。

## 不可挪走的后续

需要几何生成/事件与保守CCD，初态开接触与多次接触切换；材料与接触共同动力学/能量账本；旋转持续接触的末穿透范围；一般机器人自由积分误差/基座反力。随后弯曲自碰撞、有限驱动单爪与低mu失败、完整双臂和Three.js。发现正确性错误必须修复，不移入遗留后假称已通。

Builder check：initial_pair_design_check返回nothing important left。实施者冻结所有所选粒子域必须全动态，避免把固定节点隐式约束混入本自由节点合同；仍不限制其他独立材料旁域。

## 单次最终审阅后的精确接触认证

完整存储系数形成相对速度和法切投影，不把先舍入的relative分量作为最终原式。实际world冲量重新投影用于正常/摩擦验收；有理平方根上下界认证锥范数和滑动方向误差。原tol不变，迭代需要的FP64投影若超过原绝对速度表示预算则拒绝。

仿射门禁同时检查当前消费者实际使用的matmul和sum路径；独立仿射参考点与实际点构造平面gap的保守上下界，初态闭合和末态非穿透同时验收。原native候选不替换；新增正确性门禁不是性能优化或新积分器。
