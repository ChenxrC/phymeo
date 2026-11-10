# MuJoCo-TYX T2：原生自由根树与35维动力学

从已验T1 `4ab18bd5` 继续[扩展路线T2](../../development/mujoco-tyx-expansion-plan.md)。交付无可能接触对的单free-root树，含0–29个hinge/slide（nq最多36、nv最多35）、全部welded身体和偏心非球惯量。多接触/摩擦/G1支撑仍属于T3–T5。

## 分工与接口

复用support、actuator、capacity三个隔离worktree，分别纯编码新T2产品、独立案例、源码语义审阅；已有分支和未提交T1审阅稿保留。父独占共享runtime/dispatch、集成和全部项目执行。不得并发import、collection、pytest、数值probe、静态工具、构建或安装。

- 产品新增 `mujoco_tyx/_floating_model.py`、`_floating_kernels.py`、`_floating_device.py`，不改已验T0/T1核。`compile_floating_model(mj, model)`返回kind=3；其nq/nv保原native地址，vector中的限位按DOF地址，scalar qpos地址单独保留。`TyxFloatingDevice`沿T1同次动力学download和双bank协议；自己分配完整nq/nv空间。
- 父扩展 `_model.compile_model` 的显式路由及 `_mujoco_tyx_runtime` 打包/容量/receipt。原已支持T0自由球及T1仍沿原分派；不以失败后静默fallback切换物理。
- 测试新增独立 `tests/test_mujoco_tyx_floating.py` 和预登记案例表，先固定原XML、单位、非零初态、短窗与FP32门。测试不使用产品M/J生成oracle，作者不执行。
- 审阅新增T2语义记录，重点native root origin/world linear、local angular、WXYZ、偏心COM、速度偏置/加速度运输、四元数积分、root wrench/armature/damping。固定上游MuJoCo3.3.7与MuJoCo-Warp0.0.2，优先读本地固定来源。

## 实现与接受

1. 每步完整root/scalar FK、world COM J、M与bias；root原点与COM不混用。free-root的局部角速度基随姿态旋转，物理加速度和世界运输项必须与native方程一致。静态welded身体不能丢失。
2. 原body COM世界wrench、qfrc_applied、scalar affine drives和springref/a/D保真。free/ball弹簧等若未迁移要明确拒绝，不按scalar公式替代。原unsupported、strict-interior限位及可能碰撞拒绝保持。
3. 原生Euler更新root平移、局部角速度和WXYZ单位四元数；不把原Phymeo的root-COM世界速度直接接到native字段。非有限/非正定/输入溢出/晚读回失败全批不发布。
4. 独立CPU多案例含非球自由陀螺、偏心根COM/惯量、R/P分支、welded工具、0/7/16/29scalar、最后joint/body载荷、root施力引起关节响应；有解析/动量检查，另有相同初态连续CPU轨迹。
5. W1/W4/稀疏ID、不同控制、selected reset/clone后的真实下一步、晚故障与相同请求重试；禁用CPU/Warp/HOST动力学后仍实际TYX推进。
6. 父严格串行：首次精确静态检查、解释专项、CUDA专项、少量受影响T0/T1相邻、公共Env/PolicyRunner连接、新wheel源码外两路径。保原失败及同次入步动力学，原门不按输出追调。

成功以新可调用能力和预登记工程门为准。质量/J/bias首次不一致先定位具体body/DOF与坐标，不调模型或容差掩盖。没有实际证据不宣称G1站立或全部GPU训练，不为已有通过组追加重复验证。
