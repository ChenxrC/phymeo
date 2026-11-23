# 最后一个低率间接光过滤候选（marker100）

仅作者代码，尚未编译/实测。保留已接受 ec8r2/rigid 原包。无自动回退、无新默认。

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
$author='C:/Users/38347/.codex/worktrees/render-secondary-budget/phymeo'
$parent='C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo'
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
这不是已证画质提升：不改变baseline既有记录，也不把过滤后raw称无偏估计。

## 冻结筛选

父沿同三姿态×两独立65536spp参考、纯同ID8×8块，分别核左臂块平均亮度残差RMS
每项至少降低10%，平均偏差增加≤.005。像素级误差、运动/新显露、三传感、真实
ray/anchor计数相等和300帧速度另验。任何一项不足均保留反例，交付现有8路；
不另选半径、ROI或样本种子。小HOST测试仅验证固定数学和源码边界，不冒充GPU证据。
