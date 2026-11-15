# 面积压力静态参考首批：V00 / V01 / V02

日期：2026-11-15。分支：`codex/exp`。源码截点：`48c4a14b75196cca8b16404c3a2a58cbd118bf82`。
状态：**NUMPY_STATIC_REFERENCE / STATIC_REFERENCE_PASS**。
依据：[固定案例与门槛](../superpowers/plans/2026-11-15-area-pressure-contact-visual-cases.md)。

这是独立面积积分与声明径向压力场的首批可运行参考。没有改动 `src/phymeo`、旧接触、MuJoCo-TYX、TYX pin 或训练接口；也没有把结果合回 dev。它不是 Drake/Newton 复现、自由动力学、Hertz 解或已集成的 Hydroelastic。

## 交付

- [参考函数](../../experiments/area_pressure/reference.py)：每个真实三角形使用三个二阶精确积分点，面积权重为各自面积的 1/3；压力乘面积得到力。所有积分点、法向、dA、压力及双方载荷保存。
- [固定案例运行器](../../experiments/area_pressure/run_cases.py)：预先写 manifest，再执行固定 30 个案例；独立解析 oracle 与离散积分分别计算。
- [离线查看器模板](../../experiments/area_pressure/viewer.html)：只读保存结果，固定 0–3000 Pa 色标；全部粗/细网格均展示，粗网格不冒称满足最细档门。
- [独立测试](../../tests/test_area_pressure_reference.py)：三角二次积分、面积比例、偏心力矩、坐标变换、共同原点双方载荷、网格分辨率、零接触、非平面 COP 限制、非法/容量输入和不可变输出。

可复制运行（已有 NumPy / pytest 环境，仓库根目录）：

```powershell
python -m pytest tests/test_area_pressure_reference.py -q -p no:cacheprovider
python -m experiments.area_pressure.run_cases --output artifacts/area-pressure-new-run
```

输出目录必须事先不存在。实验不在产品 wheel 内，不添加运行时依赖。这里给出复现命令，不要求重复本次已通过的运行。

## 实际结果

Python 3.13.3、NumPy 2.4.6、pytest 9.1.1。父任务唯一串行 owner，fresh process guard；无插件自动加载、无 bytecode、线程数 1。首 Ruff 通过，首 format-check 发现三个文件需要排版；只做格式化后最终两项检查通过。完整新 HOST 首次 **14 passed / 0.60 s / exit 0**。随后唯一固定案例运行 **30 项、0 failed / exit 0**，没有数值修复或门槛调整。

| 案例 | 固定格 | 观测与原门 |
| --- | --- | --- |
| V00 | 三种面积 × 1/4/16 cell，9 项 | 0.8 / 1.6 / 3.2 N；最大载荷相对误差 1.67e-15 |
| V01 | 三档细分 × identity/平移旋转/倾斜，9 项 | 原姿态 COPx=1.666667 mm，My=-1.333333 mN·m；全部姿态最大 COP 误差 2.25e-17 m、力矩误差 1.74e-17 N·m |
| V02 | 四压入 × 2/1/0.5 mm，12 项 | 零压入全部空域/零载荷；最细档面积和载荷均低于原 5% 门 |

V02 最细 0.5 mm 的实际数值：

| 压入 mm | 离散面积 mm² | 法向载荷 N | 面积相对误差 | 载荷相对误差 |
| --- | --- | --- | --- | --- |
| 0.1 | 12.471412 | 0.0031310243 | 0.506923% | 0.00307865% |
| 0.2 | 24.943865 | 0.0124824993 | 0.252778% | 0.000765197% |
| 0.4 | 49.697900 | 0.0495951752 | 0.130473% | 0.000203328% |

最后两档面积变化最大 1.519225%，载荷变化最大 0.045649%，均低于预定 5% 门。
最粗 2 mm、压入 0.1 mm 的面积误差为 **7.92746%**，完整保留；门只施加于预定最细档，不能说每档都低于 5%。

力与力矩的双方相消是静态同点相反牵引积分的检查，不是对两个动态物体的动量守恒验收。V02 从声明压力场求实际三角网格与积分值，解析面积/载荷仅用于独立比较；没有用公式直接替换离散输出。

## 可视化与证据

本机完整证据：`artifacts/area-pressure-static-20261115-01`。
原始 command、guard、stdout/stderr、raw exit、source-before/after 均保留。

- [交互 HTML](../../artifacts/area-pressure-static-20261115-01/results/index.html)
- [完整 JSON](../../artifacts/area-pressure-static-20261115-01/results/results.json) / [汇总 CSV](../../artifacts/area-pressure-static-20261115-01/results/summary.csv)
- [15 秒 MP4](../../artifacts/area-pressure-static-20261115-01/results/static-pressure-cases.mp4)
- [球平面压力图](../../artifacts/area-pressure-static-20261115-01/results/preview.png)

MP4 由保存的 30 个案例画面构成，每项 0.5 s，1440×1000、30 fps、450 帧。是静态参数扫描，不是随时间运动。浏览器切换、下拉框、滑块和全部 30 帧检查通过，无 pageerror；唯一 console 404 为未提供的 favicon。视频编码 exit 0，元数据确认 15.000 s。临时只读本地展示服务已经关闭。

JSON SHA256：`8416a8c61b3cd371ef0762b455e38080870bf7dfbed368d5157db7637835719f`。
MP4 SHA256：`e6f66a69dcfabd7df289b2a90a23bc3aef80eab29cb1fd334a1de5f17c54424e`。

生成的大文件保留在 ignored artifacts，不上传 Git。源码与运行方法在 exp 上可复现。

## 下一步与停止条件

1. 训练线仍按原 T-smoke 优先：真实语义编码、共享 masked PPO 更新与恢复。本批没有训练结果。
2. 接触预研线检查现有 Drake/Newton 能力，先建立相同压力场/网格定义的 V02/V03 外部参考；没有参考执行不称交叉验证。
3. 再做 V03 双软体压力平衡面与 V04 分布扭转；V05/V06 才进入状态与载荷共同更新的动力学。
4. V00/V01 及本次数值格已经完成，不再为计数重复运行。若后续模型域不适用，保留 UNSUPPORTED/FAIL，不调质量或误差门追求通过。
