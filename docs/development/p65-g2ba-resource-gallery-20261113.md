# G2b-A：设备分配预算与 32 构型 CUDA 展示

状态：IMPLEMENTING / TESTS_PENDING。2026-11-13，基于 096964e7。

用户要求继续下一批，并用真实视频展示几十个不同构型同卡仿真。本批先完成可独立交付的 CUDA 数组 payload 生命周期预算，再扩至一个异构 facade 的 32 个模型 / 32 个 world。完整 HOST / 原生 arena / 编译器 / checkpoint 中间表示峰值仍是后继 G2b-B；本片不是总显存预算。

## 产品与实现范围

- 新 `MuJoCoTyxResourceBudget(device_payload_limit_bytes)` 可跨 runtime/batch 共享。完整 owner 的 immutable allocation manifest 在 runtime/零数组分配之前预留；成功分配转 actual，真正成功 free 才减 current，UNKNOWN_RELEASE 保留 charge。解释模式显示预计 CUDA payload，不冒称 NumPy 是显存。
- manifest 是真实分配的唯一 shape/dtype 来源，包含双银行、base scratch、activation、CSR、normal/T4 扩展和 padding；alias 不分配。`snapshot()` 明列 current、peak、reserved、unknown retained；后者是 current 子集。
- `admit(additional_bytes)` 持共享锁并提前检查跨 owner 所需额外峰值；父负责构造、clone/reset/restore 的共享预算传播与峰值预检。不得先释放 accepted owner 腾空间。
- 公开 `resource_snapshot()` 与原 counters 分离。原 checkpoint/source/CAS/事务保证保持；aggregate 桶上限改 32，原 128 MiB/leaf 元素门保持。
- 力学核、h/物理误差门、原模型的 DOF/contact/constraint 上限不变。32 个 bucket 按原 source/cap 签名分桶，在同一 GPU 逐桶提交并整批发布；不称融合 kernel、跨 world 碰撞或 GPU 加速比。

## 责任与顺序

原源码作者：六 device 构造、resources、新 budget/manifest；原测试作者：资源/事务测试与固定 32 模型案例、真实轨迹采集。父独占 runtime/facade/checkpoint/公开接口、合入、视频渲染与全部项目执行。作者只在原隔离 worktree 编码，不执行 import、probe、pytest、静态或构建。父每次执行前 fresh entrypoint-aware guard，唯一串行槽；保首失败和原始输出。

1. 先完成 manifest 与同预算构造/边界失败不改原状态的短验证。
2. 新 W32/B32 短窗：32 个物理源指纹不同；覆盖完整 publication、末 bucket 失败/retry、33 拒绝、checkpoint/partial reset。
3. 同一 CUDA facade 常驻 32 构型，h=1/128 s，连续 256 步=2 s，不逐帧重置、不写回渲染位置。保留全部 endpoint 状态、原输入/模型、实际 GPU 身份、计数、预算与逐步核查。
4. 从同一次已保存状态离线 FK/渲染；MuJoCo 仅用于模型编译、独立验证和画面几何，不替代 TYX 物理。视频标注模型数、world 数、仿真时间、播放速度与分桶方式；附可核对原轨迹。
5. 新代码 focused 静态/解释/CUDA、受影响兼容、源码外安装消费；保存报告并更新续作入口。视频本身不是全部功能验收。

## 成功和停止条件

真实 buffer events 与 manifest 同数同字节；精确预算可通过、少 1 byte 在分配前拒绝；边界拒绝保原状态/CAS/receipt且有实际 retry；UNKNOWN 不伪报释放。32 格不能只改名称或初值冒称不同构型。实际 CUDA 运行完成且不 fallback，保存状态通过预定 finite/physics/reference 门，视频各帧绑定这些状态。失败保原样，修实现或明确缺口，不调物理输入/预算掩盖失败。native hidden memory/完整 HOST peak 未观测即 UNKNOWN，性能对比另立阶段。

依据：[G2b 总提案](p65-heterogeneous-resource-next-design.md)、[当前入口](p65-continuation.md)、[G2a 原资格](p65-heterogeneous-scale-protocol.md)。
