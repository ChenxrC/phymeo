# T1a-4：朝向 IK、采样路径与两构型实际操作

日期：2026-11-14。新增模块经父树串行验证；最终产品提交 `316df74f`（两模块作者合入为 `778df768` / `f72e85c4`，纯格式化 `cdd14fe9`）。
本批登记见[固定路线计划](p65-manipulation-route-plan-20261114.md)，上批[短窗抓球结果](p65-manipulation-pose-feedback-report-20261114.md)按原证据复用。

## 结果与边界

本批取得**可复用的完整姿态 IK 和有限采样路径检查**。46 个新 HOST 节点首次全部通过；四条实际参考路径各 2402 个样本通过；随后两个世界各完成一次 2400 步、2.4 秒 TYX CUDA 操作。

| 原场景/任务 | 预检 | 实际离台最高/末帧 | 实际结果 |
| --- | --- | --- | --- |
| world1，RRRR / grasp-box | 接近和抬升姿态均收敛，两条参考路径 sampled_clear | 26.976 mm / 9.052 mm | 有夹持和抬升，但没有达到原抓取高度与稳定条件，原 Goal=false |
| world3，6R / place-capsule | 接近和抬升姿态均收敛，两条参考路径 sampled_clear | 38.662 mm / 20.810 mm | 第1572步已满足历史 lifted 标志；后续仍旋转/滑移。登记窗口仅抬升，不含完整目标区搬运、释放、再支撑；原 place Goal=false |

末帧使用实际 accepted-entry：tick2399，约2.399000114s；完整积分末端为第2400步。不得混用相位。
两个新窗口顺序执行、各 W1，**不是新的两世界并行或 W8 扩展验收**。没有修改物体状态、焊接、求解器、摩擦、质量、PD 增益或原 Goal。原 world0 短窗抓球成功不倒灌到这两个任务。本批不是语义编码、共享网络或训练成功。

## 通用模块

### NativePoseIKResult / solve_native_pose_ik

入口：`phymeo.control.manipulation_pose_ik`。

同时要求世界 TCP 位置与 body→world 朝向；空间旋转误差为 `Log(R_target R_current.T)`，与世界角 Jacobian 同框架。相对矩阵经 MuJoCo 3.3.7 的 mat2Quat / quat2Vel，处理零角与 π，避免正弦型误差在180°虚假为零。局部几何 Jacobian 只提供 DLS 方向，由真实加权残差回溯确认下降，不声称它是 Log 残差的完整导数。

固定默认/上界：位置5e-4m、朝向5e-3rad、每步关节向量范数.15、100轮，每轮最多8个回溯候选；只允许收紧。位置与朝向均过门才 converged。stall/iteration-limit 不是全局无解证明。不增加随机 seed。SO(3) 输入可有1e-6数值误差但不做SVD修补。

只改变调用者明确列出的原生 HINGE/SLIDE 祖先关节，保持原限位；完整 FREE/未选择坐标与 caller/model 不变。结果为 frozen record 和不可写 bytes-backed 数组。nq≤45、nv≤35、body≤128。CPU 仅作运动学/Jacobian，无 step/forward/inverse 物理求解调用。

新增9个HOST节点：独立手算PPP+RRR链/localTCP及乱序native地址；world −Y轴零/近零/近π/π；point-only朝向错误负控；不可达与限位；坏输入、不可变性和容量。测试的独立 native oracle 可用 mj_forward，产品调用被 monkeypatch 禁止。

### NativeJointPathAudit / audit_native_joint_path

入口：`phymeo.control.manipulation_path`。

明确 times 与完整 q knots，原关节范围内 C1 cubic `3u²−2u³`，每段按 `ceil(duration/max_time_step)` 保留全部样本和端点，共享端点去重。保持原 native 碰撞过滤；只执行 scratch mj_kinematics + mj_collision。记录每个实际接触的双侧几何ID/名字、距离和位置，含允许接触；不得发现碰撞便丢后续样本。

2..32 knots、总时长≤4s、样本≤4097、全程接触记录≤65536、MJB≤8MiB，nq≤45/nv≤35/body与geom≤128。原模型MJB、全q、时间、关节顺序、规范化允许接触和所有调用预算共同绑定。所有未选坐标逐字节保持，包括FREE/quaternion/signed zero。未知、限额或native warning拒绝，无错误“clear”。

`allowed_contacts` 是显式指定的参考指令接触对及深度上限（≤.05m）；非允许对最多1e-4m。字段 `worst_disallowed_depth_m` 为超出适用限额的最大值，原始距离另存。它**不是物理穿透门、不是碰撞连续安全证明，也不是力闭合判据**。

新增37个HOST节点：独立native碰撞对照、端点清但中途碰撞、cubic采样、允许接触仍记录/超界、原过滤、模型变动binding、未选字节/原限位/quaternion、坏输入和资源、不变性。

## 固定路线及预检

计划在执行前提交 `c99b53f1` / `897e126c`。world1使用上批后侧候选的完整朝向，并回到原q0的TCP抬升；world3原高位接近，抬升参考只改变joint1=-.8、joint2=+.8，使TCP向内向上且保持原朝向。没有从本次输出再挑候选。

两者时刻为0/.2/.7/1.1/1.7/2.4s，闭合及之后基础指参考.044m。原滑移反馈上限.016m、限速.02m/s、死区.015m/s、双指载荷门.05N保持。
查询 nominal 和每个knot都加.016m（clip原.06m）的 upper reference。允许的pad-object参考深度world1=.026m、world3=.023m来自提前记录的pad厚度/物体宽度/最大闭合几何。实际物理深度与原Goal并未放宽。

| 查询 | 位置误差 | 朝向误差 |
| --- | ---: | ---: |
| world1接近 | .147960 mm | .00100368 rad |
| world1抬升 | .228527 mm | .00125582 rad |
| world3接近 | 0 | 0 |
| world3抬升 | .169689 mm | .00048621 rad |

四条路径各2402样本（十进制FP64段长向上取整造成一个额外样本，未删除）。每条原始接触完整保留、无超限接触。两条离散参考不证明其间所有反馈偏置安全；查询中的物体保持原初态，不能替代动态物体/跟踪误差的在线碰撞证明。

world3固定低位TCP(.4,0,.052)、朝向I的三个seed都未通过：zero位置误差11.882mm、朝向.01737rad；两镜像seed位置2.311mm、朝向.02028rad。有限预算未找到解，不称全局不可达。这避免上批“仅点IK到位却pad穿台”的错误放行。

## 实际控制与保存核对

`scripts/run_manipulation_control.py`只增加可选 plan_builder，默认原行为不变；新 `scripts/run_manipulation_routes.py` 注入已登记路线。每个世界完成预检才创建实际CUDA batch，各最多一次2400步或原Goal终止。CPU mj_step/step1/step2全程禁止；原 NativeJointPD 仍使用CPU bias计算，因此这只是诊断控制，不是纯GPU策略。

执行根：`artifacts/mujoco-tyx-manipulation-route-20261114-01`。唯一执行owner=/root，两个作者仅隔离纯编码。实际新串行收据：

| run | 结果 |
| --- | --- |
| 01 environment | 原Python3.13.3、MuJoCo3.3.7、TYX0.2.0a1 verified site、NumPy2.4.6、pytest9.1.1，来源正确 |
| 02 first Ruff | 0 |
| 03 first format-check | 1，5文件需排版；原日志和原源码保留 |
| 04 format | 0，仅排版 |
| 05 first HOST | 46 passed / 1.71s / exit0；六文件格式化前后完整AST相同 |
| 06 final format-check | 0 |
| 07 routes | exit0；两预检都通过，两实际CUDA各2400步，无模拟异常，原任务均false |
| 08 saved audit | exit0；只消费同次保存记录，不重跑物理/IK/碰撞 |

累计4800步和482个保存端点帧。保存审计核原XML与上批baseline逐字节同、trace/state SHA、四条全部采样路径的cubic/未选坐标/接触记录、所有4800步实际参考及PD目标、前一已接受入步接触、独立接触点线/角相对滑速、逐步反馈公式/限速、原Goal结果。全保存端点q/v与同tick trace一致。
详细：`cuda-routes-01/saved-route-audit.json`，`08-saved-audit`。

| 载荷/控制事实 | world1 | world3 |
| --- | ---: | ---: |
| 最大附加闭合（左/右） | 4.317 / 4.331 mm | 10.226 / 10.912 mm |
| 最大arm joint跟踪误差 | .02145 rad | .03585 rad |
| 末帧双指法向力 | 1.769 / 1.775 N | 1.786 / 1.850 N |
| 末帧角速度范数 | .3482 rad/s | .7049 rad/s |
| 末帧物体速度范数 | .01708 m/s | .07337 m/s |
| 末帧台面支撑 | 0 N | 0 N |

物体在持续受载时出现转动和下滑；“双指有力”与“参考姿态可达”并不足以说明抓稳。现有数据尚不能单独归因为求解器错误、摩擦不足或某个增益；不要无依据扫描物理参数。world3的history transported只是原历史位移门，不表示已到目标或放置成功。

原始轨迹SHA：
- world1：`441f36195cbc787f5015f2b75f2bc1712bec7ea5cbfab88f3b230e1067a9f66f`。
- world3：`f748368fca23b0dc1725d3d0606fdc7a9836a380fcceca9fb1cac151a8d11775`。

## 下一批的具体缺口

1. **先做通用抓持支撑诊断，难度中高。** 复用这两条已保存轨迹，检查真实接触位置、关于COM的力矩、pad接触区域以及物体相对夹爪转动。把接触姿态筛选从“可达/不撞”扩展到“有限表面能够支撑目标载荷”。区分几何杠杆不足、PD跟踪与实际滑移；如需native同状态比较，先固定快照与输出，不重跑整条轨迹。
2. **基于新诊断只登记一个有依据的控制变化，难度高。** 可候选接触位置/腕部姿态或对象相对姿态闭环，继续使用原机器人动作和原Goal，不增加物体外力/焊接。不把增加夹紧增益作为默认答案。若没有新证据，不重复当前两条路线。
3. **并行纯编码/研究原PushT边界路线，难度高。** 与抓持模块分开；所有实际执行仍串行。随后完成真实目标区内释放、再支撑与八构型迁移。L1文本编码/L2共享网络仍以真实操作可控性为先决，不拿loss下降代替完成。

本批没有安装、push、远程CI、旧套重跑、周期任务或新截止。未新增视频；已有八画面诊断视频保留其原0/8历史标记。整个P6.5/Newton功能对齐尚未完成。
