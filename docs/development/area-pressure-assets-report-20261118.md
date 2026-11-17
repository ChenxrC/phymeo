# I3n：可携带的 body-local 面积压力资产

2026-11-18，codex/exp 源码5834752d；8个新节点（7 HOST＋1 CUDA）全部通过，32.87s，Ruff/format通过。dev未改。

`AreaPressureAssetBundle` 让调用者保存一次完整压力网格和有限刚面，随后从独立文件重新构造Scene既有资产，无需每次用Python重建节点。它不把URDF自动转为压力场，也不改变已验证的法向接触律。

```python
from pathlib import Path
from phymeo.backends.area_pressure_assets import AreaPressureAssetBundle

bundle = AreaPressureAssetBundle.from_assets(model_specs, assets_by_model)
Path("body-local-pressure.assets").write_bytes(bundle.to_bytes())
loaded = AreaPressureAssetBundle.from_bytes(Path("body-local-pressure.assets").read_bytes())
assets_by_model = loaded.assets_for(model_specs)
# 将 assets_by_model 传入原 MuJoCoTyxHeterogeneousAreaPressureScene。
```

文件显式记录m、Pa、MuJoCo body-local坐标、有序物体/材料身份及完整节点、四面体、三角面和节点压力。首版只支持full-zero-boundary压力体对有限刚面、friction=none、frozen-surface-elastic-backward-euler-v1。每个模型绑定原XML/archive/compiled-model实际bytes的SHA256和MuJoCo版本；origin路径、world ID和运行revision不参与绑定。不是COM或URDF link坐标，不隐式平移缩放；节点压力不是杨氏模量。

解码先核16MiB/JSON深度16/节点262144、32模型/每模型16pair和既有网格容量，再重新执行原几何构造器。未知字段、重复JSON key、非有限值、dtype/shape/bool错误、未知record、非法边界/重复tet/退化面均拒绝。每次assets_for得到新owned资产，调用者或另一份导出修改不会污染bundle。checksum用于损坏检查，不是签名认证。合法资产不保证所有后续姿态都通过有效压力梯度或查询容量准入。

实际CUDA复用原I3j三world：原资产和文件加载资产各2步，共4次step；完整状态、几何、实际压力载荷和统一观察一致，只排除各运行真实不同的owner/revision派生token。原Scene checkpoint可与相同资产hash的加载Scene配对，选择reset后恢复通过。未重跑450步，也未改变物理参数。

证据在 `artifacts/area-pressure-assets-20261118-01`：05完整8pass/RAW_EXIT0，06/07静态通过；执行前后source/test SHA相同。首test静态两个import排序问题已修，保留原日志，无数值失败。

- 同次可携带文件4878B，SHA256 a1ee563d761a301922bcb6b06c4824a5d34130f511d22ba198b994f4b98e61a5。
- 同次完整比较gzip131831B，SHA256 f5efe84ead16fc402eeef54b1176c41cd1f4252be4678348bc68798d10da4382。
- 精确路径及source清单见该目录validation-summary.json。

下一按[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)补三个真实可运动物体的双区耦合释放，并独立比较摩擦模型在支撑、冲击与卸载中的差异。摩擦选择结论形成前，不修改现有法向ABI和资产law。
