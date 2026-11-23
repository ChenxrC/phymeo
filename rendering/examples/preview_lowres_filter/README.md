# 完整距离一跳运动预览的间接光滤波

本实验档已在 RTX 4060 Laptop GPU 编译并实际回放 300 个 Panda 姿态。720p 固定机位两轮 RGB 为 37.22 / 37.49 FPS，P95 为 30.92 / 30.84 ms。左臂块平均亮度误差在预登记的六项比较中下降 14.26%–15.71%。四通道 P95 为 37.82 / 38.12 ms。它可通过下方入口显式选用。[实拍影片和逐项验收](../../results/lighting-preview/2026-11-24/README.md)

固定单 pass、低格 3×3、步长2、[1,2,1]外积权重。720p仅320×180调度。
它抑制比相邻格更大的采样斑驳，但保守地要求中心、邻格及两者间的低格矩形
全部有实际 traced 支持、同实例ID/覆盖，低格与其原全率锚点深度/法线一致，
相对深度 .003、法线点积 .98。细缝、中途缺支持、其他ID、odd越界不跨越。
无支持中心逐字沿用，归一化仅在实际接受邻格中；不按颜色挑选或丢弃贡献。

输入是原只读低率incoming MRT；输出独立RGBA32F SSBO(binding17)，仅RGB变。
一个原compute pipeline用mode bit256先做低率调度，COMPUTE write/read barrier后
恢复原20B push，再执行原K25和原history。所有 ray/guide/history SPV必须与父包
逐字相同，只有resident_spatial.comp.spv可变。中心alpha、射线计数源、K25接收门、
full primary/sensor、age4与历史拒绝、路径距离均保持原值。low SSBO随resize/close
统一释放，实际分配总量/归零继续由原telemetry核实。RGB成本包含新增低率pass；
调试读回不计入capture。没有单独GPU时间戳，所以不得把整体GPU差当成准确pass耗时。

## 唯一生成、构建、运行路径

以下由唯一执行者运行，`$python`/SDK/CMake采用此前已验证的同一工具。
所有NEW路径必须不存在。父examples须是本次冻结的原rigid driver，不能混旧runtime。

```powershell
$parent='C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo'
$author=$parent # 集成后只有这一个检出；不需要作者树或旧counter实验
$examples="$author/rendering/examples"
$original="$parent/artifacts/rendering/nov24/ec8r2/rigid"
$new="$parent/artifacts/rendering/nov24/lowres8-new"
$env:PYTHONPATH="$examples;$parent/rendering/examples"
& $python -B -m unittest preview_lowres_filter.test_contract -v
& $python -B -m preview_lowres_filter.build_source prepare --parent $original --output $new
cmake -S "$new/source/_native" -B "$new/build" -A x64 -DVULKAN_INCLUDE_DIR="$vkInclude" -DVULKAN_LIBRARY="$vkLibrary" -DGLSLANG_VALIDATOR="$glslang"
cmake --build "$new/build" --config Release
& $python -B -m preview_lowres_filter.build_source seal --prepared $new
& $python -B -m preview_lowres_filter.run --examples "$parent/rendering/examples" --debug-output "$parent/artifacts/rendering/nov24/lowres8-debug-new" -- --prepared $new --source-inputs "$parent/artifacts/rendering/nov24/film-cp4/inputs.json" --delivery "$parent/rendering/examples/preview_sparse_motion" --channels all --camera detail --studio-look recorded --seed 0 --device-index 1 --validation --current-film --output "$parent/artifacts/rendering/nov24/lowres8-film-new"
```

使用原已验inputs.json的实际路径，不能用different camera/studio替代。生成器只复制
父manifest已hash绑定的文件；seal检查所有shader清单、原marker、100 marker，原
guide/ray/history二进制逐字不变。runtime再次检查实际DLL/SPV与seal，一旦不符拒绝。
上面的CMake override变量名按生成包现有CMake保留；不自动下载安装。

原300姿态driver和两owner顺序不变。注册59/60/149额外保存original/filtered
低格RGBA、low depth/ID/valid及SHA，检查alpha逐字、记录实际改变格数和最大RGB差；
没有变化也如实报告。原driver继续保存完整incoming/history/linear、RGB、三传感、
计数、close和失败前缀。新wrapper/driver源hash在独立debug/producer.json。
滤波后缓冲保存的是有偏估计。三个登记姿态中的原始低率输入和处理结果可与独立的高采样参考分别比较。

本次在单个检出中通过了 10 项 HOST 合同。生成器重建出的 117 份运行时和原生源文件与用于采集影片的构建输入逐字节一致。用下列入口可单独复测无光照与变材质的封闭室，三例仅需一次曝光。实测关闭照明后 576 个被追踪位置与显示输出均为零，更换墙面材质后，576 个低分辨率位置的入射光发生变化，三张几何通道在三次曝光中逐字相同。

```powershell
& $python -B -m preview_lowres_filter.verify_rooms --prepared $new --device-index 1 --output "$new-room-check"
```

## 冻结筛选

父沿同三姿态×两独立65536spp参考、纯同ID8×8块，分别核左臂块平均亮度残差RMS
每项至少降低10%，平均偏差增加≤.005。像素级误差、运动/新显露、三传感、真实
ray/anchor计数相等和300帧速度另验。任何一项不足均保留反例，交付现有8路；
不另选半径、ROI或样本种子。小HOST测试仅验证固定数学和源码边界，不冒充GPU证据。

## 单检出依赖边界

本包 `_helpers.py` 自带实际使用的 verify/digest/read/save/markers/once，均只用标准库。
不再导入 `preview_group_counters`。`--examples` 指定的目录优先于继承的 PYTHONPATH；
运行前核实际 `preview_compact_rigid_history/preview.py` 的绝对路径，若进程已有来自
其他目录的 driver/runtime 包则明确拒绝。一个集成检出的 rendering/examples 足够；
原prepared目录、SDK与实际数据依旧按原manifest绑定，不随此修复重写。
此修复不改变任何被生成的 GPU 源/着色器数学；manifest的作者hash增加helpers。
父队列负责 root-only HOST/prepare 与旧生成 source/runtime 逐文件hash相等验证。
