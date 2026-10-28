# TYX-004：interpret 的 cos 依赖调用者 NumPy 浮点异常策略

- 状态：公开API最小复现成立；跨后端异常行为差异，tyx对宿主errstate的预期合同待维护者确认。
- 环境：Windows 11、Python3.12.14、NumPy2.5.4、tyx-compute0.1.0a0，固定revision99f6df7c0fd8fdeef77f97df83914da27b64b7ee。CUDA为RTX4060 Laptop/driver566.07。
- 证据等级：独立公开API实测；未测Warp、CPU C++或性能，不推论GPU速度/一般三角函数准确度。

## 问题及最小复现

[脚本](repros/cosine_errstate.py)只导入NumPy与tyx，输入FP32的0、±1e-25。三个cos的正确FP32舍入结果均为1；所有输入及输出均可表示。每种模式/策略一个三元素kernel，调用后同步并检查输出，无长期循环。

```sh
python tyx_flaw/repros/cosine_errstate.py --output artifacts/tyx-cos-recheck.json
```

[原始JSON](evidence/TYX-004-cosine-errstate.json)保存版本与四组结果：interpret在np.errstate(all="raise")下因cos underflow拒绝；ignore下返回[1,1,1]；CUDA在raise和ignore下均返回[1,1,1]。ignore只是独立诊断控制，phymeo没有关闭全局严格异常门禁。

## 已确认位置与集成处理

- 依赖tyx/intrinsics.py的evaluate_intrinsic中UNARY分支，cos直接调用np.cos并继承调用者异常策略；部分其他intrinsics已有自己的errstate范围。已证实异常由该调用抛出，NumPy内部具体中间运算未审计。
- phymeo/backends/_tyx_robot_kernels.py的robot_geometry原本对prismatic位移也无条件求sin/cos，尽管其旋转分支不使用结果。新的材料outer细化后出现极小但可表示的平移坐标，暴露了这一多余计算。
- phymeo集成缺陷用两个±1e-25实际interpret RED测试修复：只在revolute分支计算三角函数，prismatic保持静态frame。不修改tyx、pin、原动力学、严格异常策略或候选状态。此修复不表示真正revolute小角度的依赖异常差异已解决。
- 首次材料interpret持续测试2pass/3fail（415.59s），失败均cos underflow；此数值套件耗时不作为性能测量，完整修复后复测另记。

## 给tyx agent的建议

先明确解释器是否应与原生后端采用一致的IEEE返回值/异常合同，并增加公开kernel在各种宿主errstate下的cos/sin小参数回归。若要隔离NumPy的中间异常，保留非法域/非有限结果的明确行为，不在phymeo侧全局忽略异常。维护者修复后需同时复测独立脚本与真实revolute路径；本条未证明其他intrinsic均有相同问题。
