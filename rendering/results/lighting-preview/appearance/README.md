# Q0：Panda 白壳分界与面积灯对照

本目录登记诊断方法；下列命令生成真实采集后，才有画质结论。它不改厂家网格、法线或材质，不把视频的播放帧率当作渲染性能。

固定输入来自 `panda_render_performance.bind_scene`：第 60 帧、1280×720 detail 相机、原 HDR 方向和强度、曝光 8、PBR Neutral、实际双色板灯、完整 Panda 与扫描木材。默认 A/B 都保留橡胶台面的法线贴图、两跳漫反射和近场镜面，关闭空间滤波，以便观察稳定分界。

- A：原面积灯 2×2。
- B：同一光源强度，仅将面积积分格点改为 8×8。
- 两个独立种子分别保存第 1/64 批，即 16/1024 条主样本。高样本 native 用来判断本方法收敛，不标作外部真值。
- 可选控制分别移除方向灯、移除面光源、关闭台面法线。每个控制仅改一项，不能将光照消融称为画质修复。
- 所有消融以 **8×8** 为共同基线。`wide-panels` 仅将面光源每条边放大 2 倍，沿公共摄影棚规则将辐射亮度除以 4；原位置、方向及总功率不变。`soft-studio` 再移除方向灯，是具名展示组合，不能与原 2×2 混作单因归因。

先在匹配运行环境中，将 `rendering/src` 与 `rendering/examples` 放入 `PYTHONPATH`。所有输出路径必须尚不存在。

```powershell
python -B -m unittest discover -s rendering/tests -p test_panda_appearance_diagnosis.py -v
python -B rendering/examples/panda_appearance_diagnosis.py --source-only --output rendering/results/lighting-preview/appearance/source-01
python -B rendering/examples/panda_appearance_diagnosis.py --library ABSOLUTE_DLL --shader-directory MATCHING_SPV_DIRECTORY --batches 1 --seeds 0 --output rendering/results/lighting-preview/appearance/run-01
```

最后一条只拍两个 16 样本曝光，检验数据链，不替代完整 Q0。去掉 `--batches 1 --seeds 0` 后共 256 次曝光；加 `--ablations` 会另外产生 384 次，需单独安排执行预算。`--motion` 增加原轨迹 33..64 的 32 个真实姿态、每档 32 次曝光，输出 X/Y 盲标短片；编码为原 30 Hz 时钟，原 PNG 与标签映射保留。`--reference` 另外调用现有 SAPIEN 3.0.3 公共 PT 入口，双方关闭法线贴图，并保存一份 native 共同子集；不支持或资产/clip 准入拒绝均真实失败，不换后端。

固定高样本归因与随后具名照明候选可分别运行；两条命令采用同一 DLL/SPV、相机和曝光：

```powershell
python -B rendering/examples/panda_appearance_diagnosis.py --library ABSOLUTE_DLL --shader-directory MATCHING_SPV_DIRECTORY --cases baseline dense-panels no-directional no-panels --batches 1 64 --seeds 0 --output NEW_ATTRIBUTION_DIRECTORY
python -B rendering/examples/panda_appearance_diagnosis.py --library ABSOLUTE_DLL --shader-directory MATCHING_SPV_DIRECTORY --cases dense-panels wide-panels soft-studio --batches 1 64 --seeds 0 --output NEW_LIGHTING_CANDIDATE_DIRECTORY
```

第一条 256 次曝光，约 220–320 MB 原图、RAW、传感与收据；第二条 192 次曝光，约 170–250 MB。第二条重复 dense 基线是为了在同一组内逐字检查三传感和平衡照明对照，若只需已有基线旁的候选截图，可从 `--cases` 去掉 dense-panels，剩 128 次曝光，但该新目录内不再包含完整单因配对。入选后再用 seed 1 复核。短片可用 `--motion-candidate wide-panels` 或 `soft-studio`，基线仍为原 2×2；标签映射和改变项目在收据中保留。

产物：

- `source-normal-audit.json`：绑定官方原 DAE、prepared glTF、manifest 和许可；独立解析 NORMAL 双索引、原矩阵和保留三角顺序。记录每项 FP32 字节差、最大误差、几何面与厂家顶点法线角度、同位置法线分裂。
- 各 `*-seed-*` 子目录：原 RGB PNG、完整未曝光 `RGBA32F_LE`、实际深度/ID/VALID、公开诊断与 DLL/SPV SHA。每个病例三传感平面必须与 A 逐字相等。
- `original-pixel-crops.png`：仅裁切和留白，不缩放或锐化；白壳、夹指、橡胶三个主 ROI，以及左臂、上腕两个诊断 ROI，均由实际实例 ID 注册，不按图像亮度挑选。
- `normals/`：CPU 重建几何面法线、透视插值的厂家着色法线、未知掩码、材质常量参数和每实例来源。仅在实际 ID 相同且前向深度差不超过既有 2 mm 检查尺度处显示；三角边缘不假定 Vulkan 子像素规则。它们不是 GPU 法线附件，不含法线贴图扰动。
- `report.json`：完成/失败前缀、实际输入和模块来源、每次耗时、原始像素对照。种子差估计随机变动，格点或消融差只说明影响；均不自动证明原因。

目前公共 API 缺少逐瓣 direct/indirect/specular 和纹理求值后的材质附件，报告明确保留这个空缺。若源法线审计通过，只能排除“转换丢失源法线”的假设，不能保证厂家 CAD 曲面没有分片，也不能用自行平滑掩盖它。若 8×8 在同像素消除稳定阶梯，则下一步检查面积积分；若两档均有同一分界，结合材质身份、源法线和光照消融再决定是否需要共享产品修改。
