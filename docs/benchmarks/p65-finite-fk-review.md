# 有限凸面与旋转 FK 接触：唯一最终审阅

唯一审阅 `finite_fk_final_review` 已完成：0 Critical / 1 Important / 1 Minor。两项均经原实现者 TDD 修复提交 `d6b302e`，父核对原式、原始退出码与最终逐文件摘要后关闭。没有第二次 fresh review。

审阅范围 BASE `542c5a18a1f10b59250c89a459e09e73b42b4254` → HEAD `2e222d153dfbc1ee107fe1c7eb9f6e52e644b971`，覆盖全部新模块/测试/exports/smoke；当时源142摘要 `febd4bee…`。修复后源142摘要 `943f4f8a703947f9040a8b1fd3ed3e52e127beac663203c18f53e6ba564c0afc`。精确版本和记录见[工程 JSON](p65-finite-fk-engineering.json)。

## I1：真实路径中点的违例证据

`interval()` 把内部 exact Fraction 路径中点经过 float 舍入后构造点区间，可能将安全路径误报为 violation。独立静止负象限方形/原点顶点例的精确全段最小距离平方减 1 为 `1/81129638414606681695789005144064 > 0`，旧实现却在 t=1/2 给出违例。未证明不安全路径被放行；问题是违例证据不对应所声明的真实路径。

修复保留 Fraction 本身，原存储浮点值语义不变；所有调用点已检查。原反例现在 certified，5 intervals / depth2；真实旋转穿越仍 violation，预算不足仍 unresolved。没有把 unknown 改成 safe、降低 clearance 或放宽 tolerance。

## M1：几何不确定性内的诊断标签

固定 frame 的浮点逆变换会使边界 witness 落在局部 `0.9999999999999999`，旧精确 incidence 因而产生 face/edge 标签抖动。修复用精确 local Gram 和 ideal FK 顶点区间求局部平面投影 z*，并以 stored FK 的中间量 zs 分开约束两段误差：

`E = upper_norm(zhat - zs) + upper_norm(zs - z*)`。

该三角不等式界不假定两段误差相消；凸集投影非扩张性将 E 带到 exact closest witness。全部根和边/顶点平方距离比较均严格有理包围，不添加 epsilon 或公共参数。只有诊断标签选择共同低维特征；物理 weights、local/world witness、normal 保持原值。E=0 保留精确 incidence；不存在唯一公共边界时明确拒绝，不按索引猜测。

## 验证与复用

两个原反例先 2 RED；最终几何 23 passed / 0.44 s、直接受影响 host 22 passed / 23.04 s、actual interpret 4 passed / 14.89 s、CUDA 4 passed / 37.46 s，均 exit0。4 frames × 3 orders 的 face/edge/vertex 与独立 barycentric/bitwise 观察器、真实穿越/未知、后段事务和 retry=fresh 保留。实际参数和日志在已跟踪 Task1 修复报告中。

审阅者此前独立 active edge/vertex 正常冲击及 792 组 moving-plane 对照通过；这些采样是交叉检查，不是安全证明。moving-plane 曲率证明由公式推导，未重跑这些既有探针或全部历史窗口。

## 审阅排除项与后续

一般混合机器人、有限接触摩擦、任意步长收敛、真实单爪/双臂、NUM005/TYX003/TYX004 修复均明确在本冻结支持域之外，保留后续整体目标。源码外 wheel、有界 C++、远程 CI 是父负责的工程门槛，不能由本审阅代替。审阅及修复进程均已退出，父工程三步已退出0；源码外安装与142源码匹配、有界C++一次通过，准确新产品CI待交付。
