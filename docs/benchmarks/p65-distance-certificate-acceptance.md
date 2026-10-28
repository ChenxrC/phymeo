# P6.5 静态凸特征距离证书验收

状态2026-10-29：第15项有限基础已完成本地验收，准确远程交付待闭环。产品修复`2838c90`，测试标记修复后验收快照`6a9a81b`；125文件源码hash `4079424eeaed5a2ca307c42af4f3a13f014231e5c0e8a743e7c1b2bc31260e04`。详见[工程记录](p65-distance-certificate-engineering.json)、[唯一审阅与裁定](p65-distance-certificate-review.md)、[设计](../superpowers/specs/2026-10-29-p65-distance-certificate-design.md)。

## 已验收能力

公共`feature_distance_bounds`消费实际FP64有限几何及非负可行见证，返回冻结标量`FeatureDistanceBounds(lower_m, upper_m, method)`。支持点/三角、反向和边/边；精确有理归一与凸KKT上下界，FP64端点向外舍入且对设计公式最紧。退化凸包、次正规距离和浮点权重求和溢出都有明确处理；有限输入但上界无法有限表示时拒绝，不静默漏检。

探头/三角板与独立杆件两个非递布场景，用解析结果和测试侧精确活动面凸距离参照检查真实距离包围；另检查公式端点的相邻FP64紧性。包含非最优见证、边/顶点最近、平行/重叠、退化、顺序和可表示变换、极端数值、输入所有权和失败重试。测试不以生产私有数学辅助作为oracle。

薄SceneVerifier消费真实多域候选，最后子步证书失败和资源关闭失败均恢复state/history，retry=fresh；负控制关闭失败注入后两条应拒绝测试确实RED。实际interpret/CUDA各两例产生原生候选，证书始终在host执行，不宣称CUDA几何。

## 验证版本边界

- 初始缺API2RED、合同70RED后实现；源安装smoke1RED后补公共API检查。唯一fresh review为0Critical/0Important/1Minor，方法字段可接收可变数组的边界由6RED修复；没有第二次审阅。
- 修复前`0ba866c`两个CPU全量各1695通过、2跳过、338未选。2跳过是原生用例尚缺tyx标记且纯CPU无tyx；后来单独修正标记，没有改测试物理语义。
- 最终`6a9a81b`两CPU相关回归各174通过、2未选，Ubuntu新host89通过、2未选，actualinterpret2/CUDA2通过。修复前Ubuntu85包含83host和2interpret，不能冒充最终纯host验证。Ubuntu使用现有依赖，不是干净安装。
- 最终125文件源码外wheel逐文件一致，新native-pip环境安装及求解smoke通过。ruff、导入边界、compile通过；依赖锁未修改，uv/pip与锁导出检查通过。完整最终远程CI独立记录，不用修复前全量替代。
- 全部数值及review探针退出后，三个独立host输入各3×100，共900计时成功，另9warmup。源码/脚本hash固定；[PERF-017](../performance-bottlenecks/PERF-017-distance-certificate.md)只观测，不优化。

## 明确未完成

这是静态距离证书，不产生厚度侧/零距离法向、特征身份/history、接触事件/冲量、连续路径或TOI。下界为0不等于碰撞，上界不是安全证书；手工构造区间或旧构型缓存也不是当前几何证明。没有为这一host函数重跑旧材料CUDA长轨迹或无修复的C++压力。

自动接触建立/分离/特征切换与保守CCD仍是下一框架工作，之后弯曲/自碰撞、有限驱动单爪、双臂真实交接及Three.js。框架前置验收不等于H1完成，更不能换算为完整P6.5完成百分比。NUM-005、TYX-003、TYX-004保持原状态，本阶段无新tyx缺陷证据。
