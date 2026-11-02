## 单爪承载下的软布弯曲

实验场景只有原铰接夹爪 A、桌面和 48 g 软布。21 个布节点全部参与运动，24 个三角面使用原拉伸材料和每铰链 2000/(N·m) 的弯曲柔量，初始倾斜 2°。夹爪用同一 IK 与限力驱动闭合并提升。外接触用 2 mm 圆边、1.5 mm 软层和距中面 0.125 mm 的法向背衬，原 10 mm 采样拒绝门槛继续启用。每帧 0.01 s，分为 128 个布和关节共同推进的子步。本实验在 HOST FP64 执行。[场景与实际控制器](../../src/phymeo/experiments/single_gripper_bend.py) · [背衬与关节反力](p65-finite-pad-backing.md)

抓起门槛要求连续三帧同时满足最低布节点高于 19 mm、桌面上托力低于布重的 2%、两指法向力均高于 0.20 N、夹爪上托力高于布重的 25%、工具抬升超过 30 mm、最大铰链弯角变化超过 30°，以及点云到最佳拟合平面的 RMS 超过 3 mm。高度和力与形变分别计量。布重为 0.47088 N。

| 时间 | 最低布节点 | 较小的指面法向力 | 夹爪上托力 | 桌面上托力 | 最大铰链弯角变化 |
| --- | --- | --- | --- | --- | --- |
| 0.50 s | 34.37 mm | 0.356 N | 0.488 N | 0 N | 31.43° |
| 0.51 s | 40.58 mm | 0.288 N | 0.623 N | 0 N | 37.53° |
| 0.52 s | 46.10 mm | 0.276 N | 0.673 N | 0 N | 52.12° |

全程峰值弯角 96.55°，峰值采样穿透 1.875 mm。末帧工具抬升 112.82 mm，软布的离面 RMS 为 12.80 mm。44 个子步使用了背衬冲量，总量 0.000531 N·s，相对于各次自由速度的动能变化合计 −2.718×10⁻⁶ J。

负控只将抓取摩擦从 0.9 改为零。同一夹爪同样建立法向压力，随后布在 0.39 s 落回桌面。此时工具已抬升 51.58 mm，两指上托力为零，桌面上托力为 0.477 N，最低节点高 1.12 mm。该组没有产生背衬冲量。

独立读取原始轨迹，按全部公布帧重算铰链角、平面残差、力门槛，并核对各子步背衬工作和总冲量与材料位置记录。正例三帧和负控末帧分别用公共几何查询检查了全部 24 个中面，均仅返回 85 对合法共享材料邻接，未发现额外相交或未决配对。此检查覆盖四个静态中面姿态。连续的有厚度整张布自接触仍有独立验收项。

### 复现与回放

在项目根目录运行以下命令，输出路径须为空。第二项应当记录布滑落。

```powershell
python examples/compliant_cloth.py --case single-grip --output single-grip.json.gz
python examples/compliant_cloth.py --case single-slip --output single-slip.json.gz
python examples/export_compliant_replay.py single-grip.json.gz single-grip.html
```

可独立打开的本地回放为 `artifacts/p65-collision-m4-loads-local-20261102-01/with-backing-v1/single-loaded.html`。源轨迹 SHA-256 为 `e9f79d11b30baae6fed225d519e3b9b48c9592e48f617461a31caffef4dc45dd`。回放使用同一份模型和实测帧绘制圆边、关节、布形和每个材料位置的总力、法向力或切向力。原始序列、独立读回 `single-independent.json`、首次失败及修复后再次运行的记录均保存在该次本地证据目录。[实验入口](../../examples/compliant_cloth.py) · [离线导出](../../examples/export_compliant_replay.py)
