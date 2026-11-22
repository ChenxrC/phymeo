# Classic Panda 厂家视觉资产

本目录是 `frankarobotics/franka_ros` 提交 `30e598aa6fb703cc80a203481e6427f397337b4c` 的完整 Panda 三角表面转换，不是碰撞代理，也不重新运行物理。上游为 [Franka 官方仓库的固定提交](https://github.com/frankarobotics/franka_ros/tree/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description)。原始 `LICENSE`、`NOTICE` 原样保留；`original/package.xml` 声明 Apache-2.0。本次转换的更改和限制见下文。

## 使用

```python
from phymeo_rendering.panda_visuals import panda_visual_assets
meshes, materials, instances = panda_visual_assets("rendering/assets/panda")
```

返回三个不可变 tuple；加载器逐一验证固定 manifest、网格、无损源归档和许可文件的 SHA256，再返回普通 `Mesh`、`Material`、`Instance` 声明。它不下载文件，不导入 SDK，不创建 GPU，不修改物理状态。安装后的消费者应显式传入资源目录。当前 manifest SHA256 为 `d6c5f952ec5e293403d37962a5a462a70fb1359307479c87e744dfb80b4846e4`；更换版本需要明确更新 manifest 和加载器的绑定。

57 个独立网格及材质对应 59 个实例。实例 ID 从 100 到 158，属于分割标签，不等于机器人链接编号。节点名精确为 `panda_link0` 至 `panda_link7`、`panda_hand`、`panda_leftfinger`、`panda_rightfinger`。网格已位于其 **URDF authored link 局部米制坐标**；帧必须提供同一坐标语义的节点位姿，不能直接使用另一物理引擎的惯性主轴 frame。

| 节点 | 实例 ID |
|---|---|
| panda_link0 | 100–111 |
| panda_link1 | 112 |
| panda_link2 | 113 |
| panda_link3 | 114–117 |
| panda_link4 | 118–121 |
| panda_link5 | 122–124 |
| panda_link6 | 125–141 |
| panda_link7 | 142–149 |
| panda_hand | 150–154 |
| panda_leftfinger | 155–156 |
| panda_rightfinger | 157–158 |

每个 DAE 自身的 scene matrix 已烘焙到顶点；例如 link7 包含源中近似 −45° 的 Z 旋转及 +0.052 m 平移。这些变换不能再由消费者重复施加。每个原始矩阵保存在 `meshes[].source_matrix_row_major`；法线使用逆转置矩阵并归一化。COLLADA 的矩阵文本按行序列化、作用于列向量，依据 [Khronos 数据约定](https://wikis.khronos.org/collada/Syntax_rules_for_COLLADA_Extensions)。

实例的额外 local pose 除右指外均为恒等；右指为 XYZW `(0,0,1,0)`，对应上游 `franka_hand.xacro` 的 `rpy="0 0 ${pi}"`。`original/franka_hand.xacro`、`utils.xacro` 的精确来源与 SHA 在 `visual_placement_sources`，可独立审计 visual origin 和没有 mesh scale 的声明。`*_sc` 的自碰撞代理旋转不属于视觉变换。

## 原始数据、实际表面和精度

- 厂家源共有 **134,924** 个实例化三角形索引；可提交的表面为 **134,912** 个三角形，没有进行 LOD 或正面积表面简化。
- 其中 12 个索引在源坐标中已经严格零面积：link0 part03 为 5 个、link4 part01 为 1 个、link6 part00 为 1 个、link7 part06 为 5 个。转换只省略这类没有三角栅格表面覆盖的索引，不更改共享解码器的原校验门。
- 每个 part 的 `source_triangle_runs` 是保留的原三角序号半开区间；`omitted_zero_area_triangles` 保存被省略的原序号、顶点索引、源/烘焙/FP32 坐标、原因及舍入误差。两组共同覆盖全部原始三角序号，顺序可重建。
- 源零面积候选在变换前用解析源坐标的精确有理数再次校验。FP32 编码后的零面积数也是 12；**没有新产生的正面积表面塌缩**。准备工具遇到正面积三角形在 FLOAT32 中塌缩时直接拒绝。当前没有省略整个 part。
- 原始 10 份 DAE 的精确字节无损封存于 `original/*.dae.gz`；manifest 同时绑定压缩文件与解压后的 SHA256。下载缓存 `source/` 可有可无，不参与运行时。
- 烘焙计算采用 Python float64，渲染网格遵循共享 glTF admission 的 FLOAT32。逐 part 的位置/法线最大分量舍入误差保存在 manifest。源精度仍在无损 DAE 中；不将渲染 FLOAT32 误称为保留了全部十进制位。
- link5/6/7 另有共 **6 条无表面的 CAD 线段**，保存在原源及 `non_surface_lines_m`；当前三角形接口不渲染线段，没有将其假装成三角表面。

完整场景仍须经过后端自己的容量门；当前原生后端 4,096 三角形预算容不下本资产，不得静默裁掉链接。两指各 624、合计 **1,248** 个三角形（4 个实例），均未删减，可按两个 finger 节点筛选，并保留被引用的 mesh/material，构造明确命名的“双指与方块局部”场景。该局部图不能称为完整 Panda 场景。

## 材质和支持边界

此固定源没有图片、纹理或 UV。各源 geometry/material part 单独保存，硬法线边界由 `(position index, normal index)` 联合索引保留。原 Phong 的 emission/ambient/diffuse/specular/shininess/transparency/IOR 全部保存在 `source_phong`。

当前不透明 PBR `Material` 的近似规则是：源 diffuse RGB 数值作为线性 base color；metallic=0；roughness=sqrt(2/(shininess+2))。原 Phong 的 specular/ambient/IOR 不能由当前 PBR 合同等价表达，不宣称是厂家标定材质。准备器拒绝非零 emission、不透明度不足、纹理和 UV，不静默忽略它们。没有新增纹理或伪造厂家颜色。

准备器仅支持这里实际需要的 COLLADA 1.4.1 子集：米制 Z_UP、平坦单 scene、每节点一个正行列式 affine matrix、索引三角形及 POSITION/NORMAL、明确不透明 Phong 颜色。外部引用、嵌套/动画/skin、未知 primitive/attribute、反射或投影矩阵等均拒绝。源文件有 8 MiB 上限，网格 part 维持共享解码器容量。

## 复现

仅需标准库 Python；输出目录不得已含 manifest/meshes/original。首次联网取得厂家固定提交并核精确 SHA：

```powershell
python -B rendering/tools/prepare_panda_visuals.py --source-dir NEW_SOURCE_CACHE --output NEW_ASSET_DIR --download
```

离线也可将 `original/*.dae.gz` 解压到新 source 目录，连同 LICENSE、NOTICE 和 `original` 下三个 XML/Xacro 文件，再运行相同命令但不加 `--download`。工具不会覆盖已有准备结果。固定源和转换没有随机过程；focused test 实际从封存源离线重建，并逐字节比较 manifest，校验全部 57 个 glTF、许可、源归档和视觉挂载来源。压缩使用固定 gzip header/mtime；跨 Python/zlib 版本的压缩字节若不同会被哈希绑定明确拒绝，应比较来源与转换回执，而不能静默接受另一 manifest。

```powershell
$env:PYTHONPATH = "rendering/src"
python -B -m unittest discover -s rendering/tests -p test_panda_visuals.py
```
