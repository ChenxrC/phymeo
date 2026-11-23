# Compact GI：逐 ID 刚体运动的短历史实验

这是独立生成包，使用 marker **97**；静态 D 的 marker 92 当前信号仍保留。它不修改已经验证的 `preview_compact` / `preview_compact_history` 包。唯一串行执行者已通过 12 个 HOST 合同、实际 Vulkan 构建及两场设备 smoke；源码作者只读核对了保存收据，未另行重跑。

首次设备收据位于 `artifacts/rendering/nov24/rigid-first-synthetic/report.json`（SHA256 `9a0c30935d09d9651a32baf06bbba15a0d8f26b29bba2147c9693ca97f0e6c3e`）与 `rigid-first-panda/report.json`（SHA256 `4a041ce474b6a2bbbda264be39b4e753af8ac21daaf122cc3295acac08e5e934`）。两者均 complete，运动实例累计接受 23163 / 97574 像素；Panda 为真实索引 54..65。当前 estimator、三传感和 reset 输出逐字对照通过，独立明确显露像素保留旧历史违规为零。此结果不包含画质收益或 720p 速度结论。

## 信号、时钟和有限适用范围

每次 RGB 仍真实执行当前全率主光/镜面、低率一跳 GI、K25 当前 incoming 重建。真实一次 barrier 后，另一个 compute 只对 demodulated incoming 做历史处理，再以当前 primary 和当前 rho 合成；三个几何传感不被修改。不是重复旧 RGB，也没有 CPU 图像进入历史计算。

新增 GPU 表包含每个实例的实际 uint32 ID，以及与当前光栅/TLAS **同一份 FP32 world/inverse 矩阵**；其中已经含 `Instance.local_pose`。上一份矩阵仅在成功公开 RGB 后提交。ID 可不连续，通过排序表查找；上限 1024，缩放、剪切、镜像明确拒绝。当前位置先变回实例局部，再转上一姿态和上一相机；法线也经对应旋转，往返投影同样采用两个真实姿态。

连续性依据录制的**相邻呈现记录**，不是物理 tick 加一。真实 Panda 数据 `film-cp4/inputs.json` 中 tick 3200→3253→3307，时间 2.000000→2.033125→2.066875 秒；物理频率 1600 Hz、呈现 30 Hz。每次 RGB 必须提供 source SHA、呈现 index、当前完整 FrameKey/time/全帧 SHA，以及上一个已接受记录的完整 key/time/全帧 SHA。后者包括节点姿态。漏帧必须显式 reset；同一绑定帧可重复取不同真实采样。新来源、世界/epoch/phase、相机身份/尺寸/镜头、有效灯、曝光/资源信号均清空历史。相机外参平滑运动允许，大于 .25 m 或 20° 的单次跳变清空。

基本门保持相对深度 `.003 * min`、法线点积 `.995`、往返 `.75 px`，当前 5×5 同 ID/深度/法线至少三个有效支持。新显露或拒绝像素保留当前，age=1；SH-only fallback 没有历史，age=0。每 ID 一次平移 >.25 m 或旋转 >20° 只拒该实例。年龄上限 **4**，之后有界 EMA，年龄不等于独立 spp。

还要求上次/当前 rho24 相同，当前 primary（包含直接光与镜面）每通道变化不超过 `.2 * max(a,b) + 1e-5`；上次 incoming 必须在当前邻域范围外扩 `.5 * (max-min) + .1 * max(max,1e-5)` 之内。随后才作原 min/max 限制与历史混合。这些固定有限置信门会拒绝部分有效历史，**不能证明隐藏的二次可见性未变**。当前邻域也可能相关、有偏；适用于运动表面的局部重投影实验，不声称无偏、物理正确的全动态光传输或稳定达到 30 FPS。

真实 seed 为 `base XOR uint32(sequence * 0x9e3779b9)`，每个成功 RGB 只推进一次。`reuse_history=False` 使用相同序列但禁历史，供公平对照；不是重新固定 seed 的较差对照。

## 接口与诊断

```python
from preview_compact_rigid_history.runtime import activate, RigidHistoryBackend
from preview_compact_rigid_history.state import bind_presentations

activate(prepared)  # 在任何 phymeo_rendering 导入之前；一个进程一种 private runtime
bindings = bind_presentations(frames, source_sha256, start_index=54)
owner = RigidHistoryBackend(prepared, seed=0, reuse_history=True, ...).open_scene(scene)
shot = owner.capture(frames[0], request, presentation=bindings[0])
current = owner.read_current_incoming()  # RGBA32F，RGB 当前 incoming，A 支持位
history = owner.read_history_incoming() # 48 B/pixel，4f incoming/age + 4u geometry + 4f primary/rho
total = owner.read_reconstructed_linear() # 总线性输出，不是 raw MC mean
owner.reset_history()
owner.close()
```

绑定函数是来源记录辅助，不替调用者证明文件真实性。实际 CLI 另核固定原始 trace SHA、保存的 inputs SHA、每个源 frame/camera 与重新构造结果，才生成绑定。窗口第一帧没有承诺存在窗口前 GPU 历史。

`diagnostics()['history']['actual']` 保留静态 D 的真实统计。新增 `rigid_history` 包含 instance_count、moved_instances、age_cap=4、history_pixel_stride=48、marker=97、committed_rgb、rejected_lighting、rejected_incoming、moving_accepted_pixels、lookup_misses、rigid_table_upload_bytes。两个置信拒绝计数是 rejected_support 的子集，不重复相加。`presentation` 保存当前实际绑定；`unobserved_secondary_visibility` 在任意实例运动时为真。旧静态 `age 8` 文案已覆盖为 4。

额外历史请求缓冲为 `112 * W * H + 176` 字节，另每帧上传 `16 + 272 * instance_count` 字节 GPU 刚体表；可选历史调试读回额外 48 B/pixel。allocator 当前/峰值显存、profile 时间和上传字节分别报告，不能把上传量当显存。锁内准备/提交、GPU 失败和公开发布后诊断失败均不允许下一帧带入未返回的结果。

## 串行运行

将本树 `rendering/examples` 放在 PYTHONPATH，父已有真实 Panda/room 工厂也必须可导入。不要先导入原 renderer 再激活 private 包。所有输出路径均要求不存在。

```powershell
python -B -m unittest preview_compact_rigid_history.test_contract -v
python -B -m preview_compact_rigid_history.build_source --history-prepared PATH/history4 --output PATH/rigid4-new
cmake -S PATH/rigid4-new/source/_native -B PATH/rigid4-new/build -A x64
cmake --build PATH/rigid4-new/build --config Release
python -B -m preview_compact_rigid_history.run_smoke --prepared PATH/rigid4-new --case synthetic --device-index 1 --validation --output PATH/rigid-synthetic-new
python -B -m preview_compact_rigid_history.run_smoke --prepared PATH/rigid4-new --case panda --inputs PATH/film-cp4/inputs.json --start 54 --steps 12 --width 320 --height 180 --device-index 1 --validation --output PATH/rigid-panda-new
```

合成 smoke 是 97×95、12 次 RGB×两 owner，加两次传感 resize；包括平移、旋转、细遮挡、移动镜头、大幅实例跳变、同会话改灯、explicit reset、错误前帧 SHA 的负控。Panda smoke 使用 12 个真实相邻录制状态和原镜头，只显式缩小到 320×180；每档 12 次 RGB，没有物理步进。两档逐字核当前 estimator、三传感与 first/reset 总输出，要求真实运动实例接受计数非零。独立 FP64 刚体投影检查明确显露像素 age=1；排除距 raster 单元边界 .002px 内和深度阈值 10μm 内的不确定像素，不复演 shader 接受/置信/滤波。没有显露见证时如实记缺证据，合成夹具则要求见证非零。

输出 `inputs.json`、`report.json`、每帧原平面/当前 incoming/历史/总线性数据和 PNG；错误保留前缀并非零退出。报告 capture 时间不包含事后调试读回。小 smoke 不评定画质收益；720p、300 帧、手指边缘残影和相同预算独立参考必须另验，不能从“有历史接受”推导画质通过。

## 独立 720p 运动预览与质量驱动

`preview.py` 是本目录显式的实验预览入口，不修改正式 `NativeBackend` 构造或父 `preview_sparse_motion`。它消费后者已冻结的统计、输入工厂、媒体编码工具。串行执行者已通过 8 个 HOST 合同、完整两档 300 姿态、两份独立 8192 样本评分及影片检查。`D-full-all/summary.json` SHA256 为 `81eb29e12a9effc71ecb19de320de225083e23120047c141ec5a92b15c88318e`，`D-full-media/media.json` 为 `26a0ba154298e3c1fbd74130742af166cea2487a2775a2cb9316b93c8b40fd48`。当前/历史四通道 P95 为 40.3861 / 39.4333 ms；两者未达到 33.3 ms 门。第 60 帧观察到左臂 RMSE 约 .0429→.0365、白壳约 .0408→.0376，但整体质量结论为 `insufficient_evidence`，既不是全面画质接受，也不能改写成失败。

两独立 owner 分别禁用/启用短历史；两者均为**相同 compact 方法、相同实际变化 seed**。各有 10 次真实预热，随后显式 reset，再从真实姿态 0 连续执行到 299。55..60 的前史由实际捕获建立，不跳到第 60 帧冷启动，也不使用未来姿态。默认同次请求四平面，每帧核 source/seed/三个传感，报告四平面的同步 capture P95。`--channels rgb` 可另测 RGB 成本，但该收据不能进入需要逐帧几何证据的质量验收。单次顺序不是反序重复 benchmark；可另以 `--reverse-order` 形成新的独立收据。

历史档保存全部 300 张原 PNG，当前档默认只保存 59/60/149；`--current-film` 另保存当前档全片。每帧小收据 append 到 JSONL，完整 receipt 与 summary 在 owner 结束时写入，失败保留已接受前缀。三个注册姿态额外保存 current/history/total 的真实 GPU 读回、age 分布与传感 NPZ；这些审计读回不包含在 capture 时间内。整段 native profile 总计含审计读回，逐帧 profile 已排除它们，不能混用。

```powershell
python -B -m unittest preview_compact_rigid_history.test_preview -v
python -B -m preview_compact_rigid_history.preview --prepared PATH/rigid4 --source-inputs PATH/film-cp4/inputs.json --delivery PARENT/rendering/examples/preview_sparse_motion --channels all --device-index 1 --output NEW/rigid-movie
python -B -m preview_compact_rigid_history.preview_assess --result NEW/rigid-movie --delivery PARENT/rendering/examples/preview_sparse_motion --references PATH/ref8192-1 PATH/ref8192-2 --baseline PATH/film-cp4 --panels NEW/rigid-crops --output NEW/rigid-assessment.json
python -B -m preview_compact_rigid_history.preview_media --result NEW/rigid-movie --delivery PARENT/rendering/examples/preview_sparse_motion --output NEW/rigid-media
```

质量工具复用已经注册的五组几何 ID 区域及 59→60 刚体指尖对应，额外明确列右指 ID158 子区域。两份参考必须为不同非零 seed、每像素至少 8192 个原全率单跳 proposal、最终无滤波显示；检查原 trace、scene/material/light、实际 frame/camera、传感、实际加载 DLL/SPV 与解码网格。原封存 full-rate16 只作另外一列历史基线，其旧时间不混入本次公平 A/B。误差空间为逆 IEC sRGB 的 display-linear，不冒充 raw HDR。参考彼此差异超过既有噪声门时记证据不足。

可选 contact sheet 是姿态 60 左臂、右指、白壳**固定 ID 边界框**内的原像素裁片，依次显示当前/历史/两独立参考。不按候选亮度挑位置、不改曝光、不放大插值、不替换影片像素。它用于看亮斑和残影，不能代替统计。影片复用父既有 FFmpeg/probe/full-decode 流程，仅播放这 300 张已测 PNG；30 Hz 播放时钟不代表渲染已达到 30 FPS。
