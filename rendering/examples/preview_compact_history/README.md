# Compact GI：静态几何、移动相机的 GPU 历史实验

这是独立生成包。原 `preview_compact` 仍逐帧无历史；此目录不改已测试的 C 包、DLL 或 SPIR-V。串行执行者已完成下述有界设备 smoke；其他尺寸、长期运动与性能仍需独立验证。

已保存 `artifacts/rendering/nov24/history4-real/report.json`，SHA256 `0f393ec826b3e871454a9553d7a1eb627e291407af438cf4b3d6afc404b9c4f7`。97×95 两档各 14 次 RGB 加 1 次传感曝光通过；第二静态帧复用 7541 像素，两次相机移动复用 7119/6940 像素。独立 FP64 显露见证为 90/290 像素，旧历史保留违规均为 0。当前 estimator、三传感、first/reset 输出逐字对照通过；node/light/explicit reset 回 sequence 0，传感 resize 后历史正常续接。两 owner 关闭资源与 validation errors 均为 0。该结果仅验证此有界静态几何路径，不证明降噪质量、720p 速度或动态 Panda 收益。

## 实际路径与边界

1. 原全分辨率主光、镜面和传感附件保持；低分辨率 GI 仍为 tile4、每锚 2/4/16 条、一跳、K25/R2。
2. 原重建 compute 另外写入真正的完整当前 incoming RGB 和“有实际 traced 锚点支持”标记。SH-only fallback 不取得历史年龄。
3. 完整 compute barrier 后，独立 compute 从前一份 GPU 缓存作相机重投影，写另一份缓存。缓存每像素存 incoming、年龄、真实 forward-depth、uint32 ID、当前 oct 世界着色法线与支持位；没有 CPU 图像输入。
4. 同 ID、相对深度 `0.003 * min(z0,z1)`、世界法线点积 ≥ `.995`、中心往返 ≤ `.75 px` 才能复用。当前 5×5 同 ID/深度/法线邻域至少 3 个支持点，对旧 incoming 作逐通道 min/max 限制。年龄最大 8；之后是有界 EMA。年龄不是独立样本数，K25 输入本身也空间相关。
5. 总线性输出只作 `本帧 primary + 本帧 rho × filtered incoming`，之后使用原曝光/显示变换。几何传感附件不被历史 compute 写入。没有双计直接光、镜面或整个 IBL。
6. 静态所有节点时允许相机外参运动；**任何节点变换都清空整个历史**。因此动态 Panda 每帧改变节点时本版没有历史降噪收益。主表面运动向量不能证明二次遮挡未变，本版不冒充解决了这一问题。

世界/场景/epoch/phase、相机身份、尺寸/内参/裁剪、任何节点、有效 FP32 方向灯、曝光/信号、时钟断档或显式 reset 均清空。相机一次位移超过 .25 m 或旋转超过 20° 判镜头切换。相邻曝光允许同 tick/time 重复采样，或下一 tick 且时间增量 ≤ .1 s；这与离线 D 的逐 tick 输入域不同，已明确声明。

每个成功 RGB 的实际片元 seed 为 `base_seed XOR uint32(sequence * 0x9e3779b9)`；sequence 从 0 起，reset 回 0。不是只给诊断改序号。`reuse_history=False` 仍使用同样的序列，只禁复用，供公平逐帧对照。相机或节点等失效同样重置两档序列。

GPU 写入完成后还不切换缓存。公开捕获和统计校验成功后才调用独立 publish 交换读槽。拒绝的 HOST 输入、失败曝光和 sensor-only 不提交历史；新显露或几何拒绝像素回当前，支持不足同样回当前。传感专用 resize 不改已有 RGB 历史，随后重新绑定当前目标。

## 资源、代价与接口

额外 GPU storage 为 `80 × W × H + 176` 请求字节：16 B 当前 incoming，加两份 32 B 缓存、176 B 相机 uniform。实际显存分配粒度以 allocator 诊断为准。可选调试读回另懒分配 32 B/像素 HOST_CACHED buffer。存在额外 full-rate compute、邻域读和 GPU/CPU 状态读回，不能预设达到 30 FPS。

新增具名 marker 92/93，仅此生成包使用。B 的 89 空间滤波、90/94 pooled-sky 均未自动组合。顶层 `history` 合同和 `backend='experimental-native-compact-history-v1'` 区分原 C。

```python
from preview_compact_history.runtime import activate, HistoryBackend
activate(prepared)  # 必须早于任何 phymeo_rendering 导入；一个进程一种 private runtime
owner = HistoryBackend(prepared, seed=0, reuse_history=True, ...).open_scene(scene)
capture = owner.capture(frame, request)
current = owner.read_current_incoming()   # RGBA32F：当前 incoming / supported
history = owner.read_history_incoming()   # 32 B/像素：4f incoming+age，4u depthbits/oct/ID/support
total = owner.read_reconstructed_linear() # 完整 filtered total，不冒称 raw MC mean
owner.reset_history()
owner.set_directional_lights(lights)
owner.close()
```

`diagnostics()['history']` 包含 actual、真正 effective_seed、reset reasons。actual 有 committed_rgb、sequence、read_slot、reset_count、eligible/accepted/三类 rejected/clamped/age_sum/reset_pixels、UBO 上传/读回字节和请求缓冲字节。全计时范围包含真实 TLAS、主光栅、compact GI、两次 compute 与拷贝。材质/面灯/环境资源不可变，变更须新会话。

## 串行执行者命令

在包含三个 preview 目录的 `rendering/examples` 加入 `PYTHONPATH`。真实 room/Panda 夹具依赖父已绑定 examples 路径。

```powershell
python -B -m unittest preview_compact_history.test_contract -v
python -B -m preview_compact_history.build_source --compact-prepared PATH/cp4 --output PATH/history4-new
cmake -S PATH/history4-new/source/_native -B PATH/history4-new/build -A x64
cmake --build PATH/history4-new/build --config Release
python -B -m preview_compact_history.run_smoke --prepared PATH/history4-new --scene odd --device-index 1 --validation --output PATH/history-smoke-new
```

smoke 运行两档相同帧/seed 序列，逐字核当前 incoming 与三个传感，核 first/reset 完整输出、真实序列变化、同像素复用、缓存年龄和拒绝分区、坏曝光/传感 resize 无提交、关闭归零。输出 exclusive JSON、逐曝光原平面、当前/历史 incoming、完整线性值、PNG，失败保留前缀。可另以 `--scene thin` 或 `--scene panda-60` 复用同流程；Panda-60 是固定姿态下移动镜头，**不是运动 Panda 验收**。

本版是有限样本重建加重投影/邻域 clamp 的有偏估计；显露与边界规则不能证明所有二次光照误差消失。真正动态刚体重投影和局部二次可见性变化应另建具名版本、保留新显露零历史，并由实际动态画面证明收益。
