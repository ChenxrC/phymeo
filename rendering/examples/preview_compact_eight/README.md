# 单一 8 提案、完整距离的独立候选

这是隔离实验源生成器，尚不能据源码声称设备通过、30 FPS 或画质改善。
它只接受已封存的 `owned-preset-r2/compact`：tile4、4 提案、K25、R2、
单跳 visibility-ratio、无历史。复制其哈希绑定的源与私有 Python，创建新包；
既有 4/16 档、原 DLL、最新发布路径和生产入口均不改。

每个低分辨率锚点执行 **8 个提案**。环境 MIS 未开启时，8 个独立抖动的
4×2 余弦分层覆盖单位方形。开启时改为 4 个 2×2 余弦分层和 4 个环境 CDF
分层，混合 PDF 为 `(p_cos+p_env)/2`，权重为 `p_cos/p_mix`。
命中项除以 8；着色半球外、Ng 拒绝方向的零贡献不会缩小分母。
这不是 8 条必定实际发出的硬件射线：实际提案可能被半球拒绝，终端照明也会
另发阴影射线。计数只声称 `primary_proposals == traced_anchor_pixels * 8`。

原天空比值估计不变：同一组提案积累 skyVisible、skyTotal 和真实 hit，
逐通道可见率乘原 SH，再加 hit/8；黑环境退化为 hit/8。有限样本比值偏差、
SH 近似和局部历史未观测二次遮挡的限制仍在，不截断距离、不改光源或曝光。
面积灯 grid2、真实终端材质/UV/阴影、全率直接与镜面项、传感平面不改。

生成的 `compact` 再由已有两个 composer 生成 `history` 和 `rigid`。
只编译最终 `rigid` 即可；它保留真实逐 ID 重投影、相邻录制帧完整绑定、
53/54 tick 步幅、age≤4、显露/局部颜色/几何拒绝和失败不提交的规则。
运行接口仍为 `preview_compact_rigid_history.runtime.RigidHistoryBackend`，
`reuse_history=False/True` 是同 seed/同序列/同 8 提案的当前帧与历史对照。
旧 `preview_sparse_motion` 的预算菜单未扩展，请使用已有 rigid preview。

## 构建和检验（由唯一执行 owner 运行）

在本源码树根设置 `PYTHONPATH=rendering/examples`，使用已绑定的环境和 Vulkan
工具链；输出均须是新目录，不覆盖。以下 `$parent` 指父渲染工作树，`$out`
为新的候选根，不能位于已有 prepared 包内。

```powershell
python -B -m unittest preview_compact_eight.test_contract -v
python -B -m preview_compact_eight.build_source --compact-prepared "$parent/artifacts/rendering/nov24/owned-preset-r2/compact" --output "$out"
cmake -S "$out/rigid/source/_native" -B "$out/rigid/build" -G "Visual Studio 17 2022" -A x64
cmake --build "$out/rigid/build" --config Release
python -B -m preview_compact_rigid_history.run_smoke --prepared "$out/rigid" --case synthetic --output "$out/smoke-synthetic" --device-index 1 --validation
python -B -m preview_compact_rigid_history.run_smoke --prepared "$out/rigid" --case panda --inputs "$parent/artifacts/rendering/nov24/film-cp4/inputs.json" --start 54 --steps 12 --output "$out/smoke-panda" --device-index 1 --validation
```

`recipe.json` 绑定父 manifest、三个新包 manifest 和原有 composer 文件。
各包 manifest 绑定每份生成源；P8 的 SpecId83 **名称**区别于 P4，主片元、
guide、compute 和 DLL 必须一致。源生成器检查实际循环、分层、/8、宏、
计数器以及 unchanged `.5` 混合 PDF；这仍不能替代编译和真设备证据。

真实 smoke 应检查：提案计数为实际锚点×8，RGB/四平面请求与当前/历史相配，
current incoming、三传感及 reset 图像逐字相同，显露 age1、移动实例确有接受，
关闭资源归零。正常低锚点缺支持仍使用已声明 SH fallback，不改成虚构采样。

## 唯一完整距离质量候选

先按同一 300 录制态顺序各跑 10 预热、重置、300 主曝光。RGB 速度用本次
真实 RGB capture 的 P95 判 33.3ms；不要使用四通道成本代称 RGB 性能。

```powershell
python -B -m preview_compact_rigid_history.preview --prepared "$out/rigid" --source-inputs "$parent/artifacts/rendering/nov24/film-cp4/inputs.json" --delivery "$parent/rendering/examples/preview_sparse_motion" --channels rgb --output "$out/motion-rgb" --device-index 1 --current-film
```

完整传感/参考核验可使用同一驱动 `--channels all --output NEW`，其额外拷贝
成本单列；可复用 smoke 的设备正确性，但不能把 RGB run 未请求的传感标为已拍。
评分入口 `preview_compact_rigid_history.preview_assess` 接受 `--result`、
`--delivery`、`--references` 两个父新封存独立参考目录、`--baseline` 原封存影片、
`--panels NEW`、`--output NEW.json`。沿同一注册帧/ROI和原误差门，无重新挑图。
原 `ref8192-1/2` 可保作旁证；如果父按预定上限增强参考，应绑定新目录并保旧收据。

只评这一预算，不搜索更多档位。重点报告第 60 帧左臂、白壳、右指的当前帧和
短历史误差，以及固定显露/时间差证据。`insufficient_evidence` 是参考精度或
覆盖不足，不是设备失败，也不是画质全面接受。所有源、DLL/SPV、时钟和收据
须按实际读取结果绑定；此 README 没有预先承诺速度或改良结果。
