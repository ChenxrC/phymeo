# 独立紧凑 GI 光栅实验

只从已封存且逐文件 SHA 完整的 sparse 包生成**新**源码/runtime 目录。原 C、生产 DLL、SPIR-V 和默认行为均不修改。固定 tile4、单跳、2/4/16 提案、K25/R2，支持原 radiance 与显式 visibility-ratio；不支持历史或 MSAA4。

```powershell
python -B -m unittest preview_compact.test_contract -v
python -B -m preview_compact.build_source --sparse-prepared ORIGINAL_C_PACKAGE --output NEW_COMPACT
cmake -S NEW_COMPACT/source/_native -B NEW_COMPACT/build -A x64
cmake --build NEW_COMPACT/build --config Release
python -B -m preview_compact.run_smoke --prepared NEW_COMPACT --full-library ORIGINAL_PRODUCT_DLL --full-shaders ORIGINAL_PRODUCT_SPV --device-index 1 --validation --output NEW_RESULT --panda
```

`PYTHONPATH` 只需包含本目录上级 `rendering/examples` 及父场景工厂所在的 examples；先 `activate` 再 import renderer。母包 private runtime 与新包逐文件相同才允许同进程三路比较，三个 owner 依次关闭。多个独立新包仍分别起进程。

主 pass 增加 location4 RGBA32F，全率写当前 SH 和 oct-Ns，rho 保留在原主色 alpha。低分 GI 的实际 viewport 为 `(-.125,-.125,W/4,H/4)`，目标尺寸为 `ceil(W/4)×ceil(H/4)`，拥有独立 RGBA32F incoming/Ns、前向深度、ID、有效性和 D32。规则中心对应全率像素 `4*i+2+.5`，随机键用全率整数 `4*i+2`。奇数尾部越界锚点无效，不 clamp。

重建先核低分 hit 与对应真实全率 anchor 的 ID、深度和法线，再核当前接收像素，始终最后乘接收者自己的 rho。全率普通传感仍来自原主 pass。真实硬件亚像素量化会令低分与全率光栅边缘不完全重合；不能从解析中心映射推断逐字 GI 等价，无支持时保留声明的 SH fallback。

`CompactBackend` 构造和 SparseBackend 相同。每 RGB capture 是独立曝光，`read_reconstructed_linear()` 为全率 total reconstruction，不是原始间接或 Monte Carlo mean。`read_incoming_guide()` 是真正低分辨率数组；`read_full_attributes()` 为全率 SH/Ns；`read_low_sensors()` 提供实际低分 Z/ID/valid。读回是验证用途，不反馈给算法，不混入 capture 时间。capture/read/close 由会话锁串行。

计数分区为 `reconstructed+fallback+nondiffuse=full visible`；`traced_anchor_pixels` 独立统计低分最终有效 trace 锚点，乘配置提案数得到 `primary_proposals`。这不是总 fragment/helper/overdraw 或硬件 ray-query 数。`compact.successful_guide_draws` 每成功 RGB 一次，sensor-only 零次。所有门和性能以真实 report 为准；生成源码不代表已通过设备验证。
