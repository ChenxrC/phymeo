# 一张连通软布的自接触

原协同交接的竖直输入在 12 个显示帧出现了 59 个异处面–面交点。按原始二进制坐标作精确平面截线，59 对面都具有双方重心权重严格为正的同一空间点。首先出现于 0.76 s 的面 4 与面 8 没有共享节点。2° 的原失败轨迹没有这类静态中面命中。窗内的全部步内运动需另行检查。[逐对材料见证](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-structural-palm-v1/self-hit-geometry.json)

Newton 的软自碰撞管线把点–三角形和边–边接触写入缓冲，检测的额外距离与开始施力的距离可以分开设置。Bridson 等的布料方法结合了近距离排斥与连续碰撞处理。Phymeo 此处沿用自己的有限双侧三角面、法向弹性层及步内检查。[Newton 碰撞源码](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/sim/collide.py) · [Newton 自接触 API](https://newton-physics.github.io/newton/1.6.0/api/_generated/newton.CollisionPipeline.html#newton.CollisionPipeline.init_soft_self_contact) · [Bridson、Fedkiw 与 Anderson 原论文](https://graphics.stanford.edu/papers/cloth-sig02/)

## 连通试片的载荷

四个三角形构成折回的连通条带，六个节点共用一份材料状态，所有网格边都有距离材料。两个平行远面相距 0.125 m，作用距离为 0.25 m，每对面的刚度为 16 N/m。公共 `CollisionLoadSystem` 给出一个 2 N 的法向接触，面积分配到三个材料点。下层每个节点承受 −2/3 N，上层每个节点承受 +2/3 N，总内力和力矩均为零。真实材料步使两面反向运动，五个其余配对均只共享合法材料地址。[HOST 回归](../../tests/collision/test_connected_cloth_loads.py)

另一组用例保留同一铰边两端合法、途中异处相交的运动，真实步进器拒绝并回滚；另一侧始终分离的铰链可以推进。本次 HOST 通过 3 项，真实 tyx interpret 与 CUDA 各通过 1 项，设备均实际执行碰撞查询、法向算子和颗粒材料更新。当前层只在有正间隙且能确定方向的远面之间给出法向力。共享材料本身不产生排斥，异处横穿会触发整步拒绝。此试片没有声明一般干摩擦或任意厚折角的可解性。[设备回归](../../tests/collision/test_tyx_connected_cloth_loads.py)

## 弹性布与背板的共同子步

`CompliantClothContactProgram(..., cloth_self_layer=...)` 在材料步内从真实粒子状态取力。可传一份 `NormalLayer`，或为每对命名面给出一份标定。作用宽度以表面间隙计，程序加上两个面的参考半厚度再比较中面距离。中点积分和有背板的力–速度–漂移积分都保存本次实际用力的时刻及材料地址，并检查两段实际材料路径。带有预应变铰链的连通试片和同时存在桌面载荷的试片均通过。相关 HOST 共 25 项，tyx interpret 与 CUDA 各 2 项；弹性与铰接积分保持 HOST FP64，两个设备后端已实际执行自接触检测和力算子。[物理子步及回滚](../../tests/test_compliant_self_contact.py) · [设备路径](../../tests/test_tyx_compliant_self_contact.py)

在原双臂机器人的真实 24 面状态中，一次纳入全部 276 对。受限接近轨迹的 0.85 s 入步成功推进 78.125 µs；原竖直轨迹 0.76 s 的横穿在同一程序被拒绝，所有域回滚。首个成功子步中没有活动自接触，112 对进入初始 AABB 窄相候选，本机耗时 8.96 s。接下来应先减少需要昂贵特征查询的普通邻接与明确分离对，再运行整条轨迹。[真实网格子步记录](../../artifacts/p65-collision-m4-loads-local-20261102-01/with-guarded-approach-v1/full-sheet-actual-substeps.json)
