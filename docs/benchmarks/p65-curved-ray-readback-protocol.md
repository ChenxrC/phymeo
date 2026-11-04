# 三解析体 4,171 束公开读回的可选计时协议

状态：已按独占授权完成三次源码外配对，结果见文末。功能和 wheel 不变；下列协议先于测量冻结，后续复现仍须由协调者授予唯一执行槽。

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

Newton 另记录 update 返回、设备同步、ID/depth/normal 三次 numpy 读回墙钟，以及请求读回的 3 次/83,420 字节；这是显式调用统计，不冒充所有 Warp 内部操作的 profiler。Phy 用真实 `tyx.runtime_stats()` 差分记录内核/H2D/D2H 次数和字节；内部 I/O 时间无法从公开 query 分离，明确留在整次墙钟内，不做全局 monkeypatch。双方逐帧校验和磁盘写 NPZ 的耗时另存，不包含在主边界内。全部数组先保存在主机，所有计时结束或首个失败后统一写文件，计时轮之间不穿插磁盘输出。setup、冷和 12 预热原样留存；正式 21 样本给出 p10/p50/p90/p95/min/max，无删点。

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

## 2026-11-05 源码外实测结果

正式六进程使用同一个脚本提交 **625edd726c66b06fed2999d974a3fa86dff52366**，文件 SHA256 `0c3c24a56663bf8b5c916d2f88d716b22f110fd36366d94ef7edd277eb4c07b3`，共同输入 NPZ SHA256 `3ee19510a8775acfe7ea763df291b47f1a3e2a8ba6c94a88aed9c88970421c24`。Phy 三进程每次均核实 243 个加载模块来自 wheel `759d47e…`，没有从 checkout 导入产品。两端为同一 RTX 4060 Laptop GPU、CUDA 可见设备 0；Newton 1.7.0 / Warp 1.17.0，Phy wheel 0.1.0a0 / tyx-compute 0.1.0a0。

三对的所有 **63 对正式帧**均通过原逐射线门，且各自冷/12 预热也通过。下表单位 ms；分位数使用每进程全部 21 个正式样本，没有删点。

| 配对顺序 | Phy 整次 p50 | Phy 整次 p95 | Newton 整次 p50 | Newton 整次 p95 | Phy / Newton 吞吐 |
|---|---:|---:|---:|---:|---:|
| 1，Newton→Phy | 1.5451 | 1.8458 | 0.2600 | 0.3482 | 0.168274 |
| 2，Phy→Newton | 1.4918 | 1.6747 | 0.2831 | 0.3746 | 0.189771 |
| 3，Newton→Phy | 1.4773 | 1.7206 | 0.2385 | 0.2699 | 0.161443 |

这里“整次”是主边界 `public_to_raw_read_ns`，从公开调用到三个原格式字段全部同步可读。**这个固定观测任务中 Newton 三对都更快**。这是当前公开接口的端到端成本比较，含各自验证、打包、设备提交、同步与主机读回；不能解释为纯 GPU 算术速度比。两端达到相同误差门，不是原生位相同输出。

加入共同 int64/float64 帧转换后，Phy 的 p50/p95 三对依次为 1.5621/1.8654、1.5082/1.6920、1.5008/1.7384 ms；Newton 为 0.3233/0.4157、0.3565/0.4680、0.2962/0.3224 ms，结论相同。这一边界保留实际 sentinel 与 dtype 转换成本。

每次新进程首个 API 冷调用的 Phy/Newton 三对分别为 807.6537/602.6063、499.1639/561.7211、725.1808/1999.4174 ms。setup 独立保留在各收据中，不混入这列；冷值含各自首次路径编译，不能代表稳态。12 个预热原样保留，没有以冷编译时间选择或剔除进程。

全部正式 TYX 调用的真实计数相同：**1 次 CUDA kernel、5 次 H2D 共 183,728 B、2 次 D2H 共 83,420 B**，每帧另有一次 argument-pack/prepared-launch 创建；无 graph replay。Newton 每帧明确请求 **3 次 D2H 共 83,420 B**，调用者没有再次上传已经常驻的 rays/xform；后者是调用层计数，不是全 Warp 内部驱动计数。两端均保留本来可复用的对象；没有为 Newton 增加场景重建，也没有绕过 Phy 的公开查询打包。所有正式 TYX 调用均有实际 CUDA launch，没有 HOST 替代。

| 形体 | 63 对中最大深度差 / m | 最大法向分量差 | 原门超限数 |
|---|---:|---:|---:|
| ellipsoid | 4.291534424e−6 | 6.332993507e−6 | 0 |
| cylinder | 9.775161743e−6 | 1.920014620e−5 | 0 |
| cone | 4.315376282e−5 | 6.903707981e−4 | 0 |

形状 ID 错配为 0。每帧均保有 179/491/398 个命中及 3,103 个 miss。计时后的磁盘保存及逐帧校验耗时各自独立记录，没有偷减校验或只下载稀疏命中。

收据在父 `artifacts/p65-curved-ray-readback-20261105-01/`：
- `11-pair1-newton`、`12-pair1-phymeo`、`13-pair2-phymeo`、`14-pair2-newton`、`15-pair3-newton`、`16-pair3-phymeo`：六进程各自 **RAW_EXIT=0**，`measure/receipt.json` 和 34 组 NPZ 保存全部原数组、公共数组、时长和逐形体误差。
- [paired.json](../../artifacts/p65-curved-ray-readback-20261105-01/paired.json) 与 `17-compare`：全部 63 对重新读取验证，**RAW_EXIT=0**。
- `18-tested-style` / `19-tested-format`：最终实测脚本的 Ruff check / format check，均 **RAW_EXIT=0**。
- `01-lint`、`03-style` 首个 import 排序错误原样保留；`08-s1-phymeo` 在查询前因脚本误写发行包名 `tyx` 而 **RAW_EXIT=1**，已仅修成 `tyx-compute`。此前 `07-s1-newton` 是预备进程，不进入正式三对。之后三对使用统一最终脚本，未改产品或物理输入/误差门。
- 全部 fresh guards 记录当时阻塞进程数为 0；06:53:42 已正式归还唯一执行槽。未追加样本、性能优化或修改 wheel。
