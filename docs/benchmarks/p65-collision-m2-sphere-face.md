# M2-RD · 球对整张三角面的首次接触

`phymeo.detect` 现在可查询平移球和双侧零厚三角面的连续接触。三个材料顶点可以分别按各自的起点和终点作仿射运动。面内部、三条边和三个顶点都被检查，每个见证保留原顶点 ID 和对应权重。[球–面查询](../../src/phymeo/geometry/_sphere_triangle_sweep.py) · [公共记录](../../src/phymeo/geometry/collision_witness.py)

```python
import phymeo as ph

cloth = ph.CollisionKey(0, 1, 0)
ball = ph.CollisionKey(0, 2, 0)
model = ph.CollisionModel((
    ph.CollisionShape(cloth, "triangle", (7, 15, 31)),
    ph.CollisionShape(ball, "sphere", (19,), radius=0.25),
))
paths = {
    cloth: ph.CollisionMotion([[-2, -1, 0], [2, -1, 0], [0, 2, 0]]),
    ball: ph.CollisionMotion([[0, 0.5, 0.5]], [[0, 0.5, -0.5]]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert pair.sweep.first_t == (0.25, 0.25)
    assert dict(pair.sweep.witness.first_weights) == {7: 0.25, 15: 0.25, 31: 0.5}
```

首触时间按当前步归一化，区间宽度至多 2⁻⁴⁰。见证的取样时刻在 `pair.sweep.witness_time`，步末间隙单独保存在 `pair.witness.gap_m`。初态已接触的返回时间为零。从另一侧接近或交换形状顺序时，法线按公开约定从第一侧指向第二侧。具有几何厚度的布面、球心转弧仍返回 `unsupported`。如果候选面在步内折叠为线或点，返回 `unresolved`，不发布首触。

## 怎样区分采样与首触

在球心的相对坐标中，每个顶点都是时间的线性函数。顶点的距离条件最高二次。边的距离乘以正的边长平方后最高四次。面的距离乘以正的面积平方后最高六次。边投影必须位于线段内，面投影的三个重心分子必须非负。给定取样时刻，实际最近点用全部可行特征重建，权重按材料 ID 返回。

tyx interpret 和 CUDA 在每个有效形状对上批量判断七个特征。内核在 17 个归一化时刻计算 FP32 条件，给出一个候选上界。HOST 用原存储坐标复核该时刻的可行性，并查找闭条件的最早公共解。线性投影条件与凸二次距离复用上一批的求解器，一般情况隔离低次多项式的实根并检验同一根处的其他条件。这样也能检出所有设备采样都错过的单点擦碰。根无法在 160 层内分开时明确返回 `unresolved`。本批没有设备吞吐基准。[设备内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [闭条件求交](../../src/phymeo/geometry/_sweep_polynomial.py)

接口对照采用 IPC Toolkit 对移动点、移动三角面和最小间距的 CCD 参数定义。根的表示与符号判定参考 CGAL 的代数核文档。本实现沿用项目固定的 tyx 环境。[IPC Toolkit CCD](https://ipctk.xyz/cpp-api/ccd.html) · [CGAL 一元代数核](https://doc.cgal.org/latest/Algebraic_kernel_d/index.html)

## 2026-11-02 本地验收

| 预先登记的输入或独立对照 | 本次公开查询 |
| --- | --- |
| RD01 基准与循环重排 | 两端间隙均为 +0.25 m。首触为 [0.25, 0.25]，面点 (0, 0, 0)，ID 7、15、31 各占 1/3。各顶点到球心扫线的横向距离均大于半径。 |
| 偏心留出与重排 | 面点 (0, 0.5, 0)，同一首触时间，材料权重按 ID 依次为 1/4、1/4、1/2。交换形状顺序后，物理点与时间保持，法线反向。 |
| 沿 x 平移 4 m 的分离负控 | 默认宽相直接排除。扩大检测余量让它进入窄相后，球仍与全段三角面分离。 |
| 三顶点独立仿射运动 | 三个顶点速度均不同。球在 t=1/2 首次触及面内部，见证点为原点，三个权重均为 1/3。 |
| 变化法线 | 在 t=1/2 得到法线 (−3/5, 0, 4/5)，面点 (3/20, 0, 9/80)。在 1/4 处分步后，后段局部首触区间包含 1/3。另一倾斜构型返回 [0.5590169943743604, 0.5590169943752699]，包含独立真值 √5/4。 |
| 边、顶点与接触区域边界 | 边擦碰在 1/2，横向外移一个 FP64 刻度后全段分离。顶点擦碰在 1/3，区间 [0.33333333333303017, 0.33333333333393966] 包含该时刻，材料权重 (1, 0, 0)。另一条路径持续与无限平面相切，直到投影在 1/2 进入有限三角边才报首触。 |

HOST **57 passed / 0.76 s**，tyx interpret **12 passed / 1.93 s**，实际 CUDA **12 passed / 3.94 s**，全部退出码为 0。三组均含受共享管线改动影响的 M1 和 RR 回归，每组 18 项源文件、输入与配置的运行前、后及当前绑定一致。9 个改动的 Python 文件通过 Ruff 和格式检查。[目录与 HOST 回归](../../tests/collision/test_sphere_face_sweep.py) · [设备回归](../../tests/collision/test_tyx_sphere_face_sweep.py)

设备批次将五个登记构型和一个变形面放在独立世界，共返回 66 对形状，6 对进入窄相。HOST、interpret 与 CUDA 对这 6 对的状态、见证与时间区间逐对一致。CUDA 首次查询及原输入重试共检查 84 个特征，完成 2 次球–面内核启动。单独的顶点擦碰用例检查 7 个特征，FP32 返回 0 个采样命中，最终区间仍包含 1/3。零容量返回所需的 6 对，窄相启动数不增加。注入设备故障后保留上次成功结果并关闭故障工作区。

本机原始记录位于 `artifacts/p65-collision-m2-rd01-local-20261102-01`。最终运行分别为 `host-pytest-153708540`、`interpret-pytest-153718026`、`cuda-pytest-153727887`。新增回归保存 12 份 HOST 观测和各 2 份设备观测。原版基准与独立变形面两项失败均已保留，二者当时都返回 `unsupported`。环境是 Python 3.12.14、pytest 9.1.1 和固定 tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

复现时用已安装固定 tyx 的环境，将本工作树的 `src` 加入 `PYTHONPATH`。下列命令逐条执行，各自指定新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_sphere_face_sweep.py tests/collision/test_collision_pipeline.py tests/collision/test_round_collision.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_sphere_face_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_sphere_face_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

随后交付的 [DD01/DD02](p65-collision-m2-triangle-pairs.md) 已接入异体薄面的点–面与边–边扫掠。DD03 将独立验收同体材料邻接。[多案例计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
