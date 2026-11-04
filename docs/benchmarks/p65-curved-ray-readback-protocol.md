# 三解析体 4,171 束公开读回的可选计时协议

状态：候选脚本已编写，**尚未运行、没有性能结果**。功能和 wheel 不变；须在父任务正式交出唯一执行槽后启动。不能为了完成本基准挤占安装验收或产品修复。

入口：[benchmark_curved_ray_readback.py](../../scripts/benchmark_curved_ray_readback.py)。

## 固定任务与验收

复用 G7 的 97×43 束、逐束不同原点和斜射方向，以及原 ellipsoid/cylinder/cone 的位置、尺寸、转轴和角度。三体的预期命中数为 179/491/398，另外 3,103 束无命中。不是增加形状数、重复射线或调 capacity 的吞吐实验。

先用 `prepare` 一次生成共同 FP32 输入 NPZ；六个进程均读取这一文件。米制位置/尺寸和四元数资产均为 FP32。由于 Phymeo 公共 `CurvedSolid` 要求单位四元数误差不超过 1e−12，两个适配器都把该同一 FP32 四元数提升并单位化；Newton 随其原生 transform 再存成 FP32。报告记录公共输入四元数、Newton 实际存储 transform。两端由同一资产生成，**原生精度与运算顺序不同，不声称位相同输出**。

每次实际调用（包括冷、预热、正式）都核对原 G7 native 逐射线形状 ID；命中深度绝对误差严格小于 **1e−4 m**，世界法向逐分量绝对误差严格小于 **1e−3**。不改原门槛。保存每帧三种原始格式和三种公共格式到 NPZ；无命中保留 native 原值，并仅在公共格式统一为 −1 / +inf / 零法向。

原参考必须是父 `artifacts/p65-newton-night-20261105-01/curved-rays-gpu.json`，SHA256 为
`bdc40eb0dffeb3b196421c540fb3d3a3e58ef0def59651fba587bf331af079c4`。
它是已存在的 CUDA 观测；本实验不声称有同帧 Newton CPU 计时。

离线 `compare` 再读取三次配对的全部 21×3 组正式 NPZ，既复验原参考门，也直接核对本次 Phy/Newton 对应帧。任何门失败都保存原始结果并拒绝速度结论。

## 计时边界与复用

每个进程恰好 **1 次冷调用 + 12 次预热 + 21 次正式调用**；无自动增加轮数或剔除慢点。三组顺序预先固定为 **Newton→Phy、Phy→Newton、Newton→Phy**。每端均为独立新进程、独立新缓存目录，不能与其他项目作业并行。

场景和输入预先构建。Newton 复用一个 Model/State、SensorCamera、射线/相机变换和三块输出缓冲；固定场景只在 setup refit 一次，正式轮不重建 sensor/BVH。Phy 复用一个 Model/poses/TyxRaycaster，保留其公开 query 内部原有的几何打包、分配和传输，不能绕过产品。冷调用是“setup 后首次公开调用”，不是整进程启动或整个 GPU 的冷启动。

两个整次耗时均保存原始 ns：
- 主边界 `public_to_raw_read_ns`：公开 API 调用开始，直到形状编号、米制射线距离、世界法向全部同步可在主机读取。
- 补充边界 `public_to_common_frame_ns`：再包含共同 int64/float64 有拥有权数组和 miss 规范化。公开原始格式是 Newton uint32/float32/vec3f，Phy int32/float64/float64；不能把补充转换说成增加计算精度。

Newton 另记录 update 返回、设备同步、ID/depth/normal 三次 numpy 读回墙钟，以及请求读回的 3 次/83,420 字节；这是显式调用统计，不冒充所有 Warp 内部操作的 profiler。Phy 用真实 `tyx.runtime_stats()` 差分记录内核/H2D/D2H 次数和字节；内部 I/O 时间无法从公开 query 分离，明确留在整次墙钟内，不做全局 monkeypatch。双方逐帧校验和磁盘写 NPZ 的耗时另存，不包含在主边界内。全部数组先保存在主机，所有计时结束或首个失败后统一写文件，计时轮之间不穿插磁盘输出。setup、冷和 12 预热原样留存；正式 21 样本给出 p10/p50/p90/min/max，无删点。

速度比若最终资格成立，名称为 `phymeo_over_newton_throughput`，数值严格是 **Newton 整次耗时中位数 / Phy 整次耗时中位数**，同为 4,171 束。三对全部报告，不能挑最快的一对。

## 安装身份与串行调用候选

Phymeo 使用父已验证的源码外站点，执行器仍为已有项目 venv 的 Python，必须 `-I`，由脚本仅显式加入 wheel-site。绑定 wheel SHA
`759d47efaefad1012a6e8302e4ecaa71a2cfe65f20cb74e3a8c4c5364beb50c6`；检查 installed direct_url 和每个已加载 Phymeo 文件与该 wheel 内容一致。无需重建/安装产品。Newton 使用独立 1.7.0 环境。

下面是候选参数，不是已经运行的收据。外层应在每次进程前 fresh guard，并保留 stdout/stderr/RAW_EXIT；每个输出目录必须新建。工作目录固定到源码外目录。先复制脚本到该目录，再按固定顺序执行，不能使用 job 并发：

```powershell
$parent = 'C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo'
$outside = 'C:/Users/38347/AppData/Local/Temp/phymeo-installed-newton-20261105-0800-v1'
$phy = 'C:/Users/38347/Documents/ChatGPT/phymeo/.venv/Scripts/python.exe'
$newton = "$parent/.venv-newton-1.7.0/Scripts/python.exe"
$script = "$outside/benchmark_curved_ray_readback.py"
$native = "$parent/artifacts/p65-newton-night-20261105-01/curved-rays-gpu.json"
$wheel = "$parent/artifacts/p65-newton-night-20261105-01/package-5f32b3f/dist/phymeo-0.1.0a0-py3-none-any.whl"
$root = "$parent/artifacts/p65-curved-ray-readback-20261105-01"
$inputs = "$root/common-inputs.npz"
Set-Location -LiteralPath $outside
# 每条 Python 前均由协调者授予唯一槽并调用现有 fresh guard；本文件不会自动取得执行权。
& $phy -I $script prepare --out $inputs
# session 1 / 3 先 Newton 再 Phy；session 2 先 Phy 再 Newton。
& $newton -I $script measure --engine newton --session 1 --inputs $inputs --native $native --out "$root/s1-newton"
& $phy -I $script measure --engine phymeo --session 1 --inputs $inputs --native $native --out "$root/s1-phymeo" --wheel-site "$outside/site" --wheel $wheel
# 同参数换成 2、3，保持规定顺序及不同新输出目录。
& $phy -I $script compare --native $native --out "$root/paired.json" "$root/s1-phymeo/receipt.json" "$root/s1-newton/receipt.json" "$root/s2-phymeo/receipt.json" "$root/s2-newton/receipt.json" "$root/s3-phymeo/receipt.json" "$root/s3-newton/receipt.json"
```

每个工作进程自行保存 `receipt.json`、共同输入副本和 34 组完整原始/公共输出。校验失败仍落失败收据与该帧 NPZ；不重用已存在输出路径。可因时间或安装验收优先而整项不运行，此时只交本协议与候选，不给性能排名。
