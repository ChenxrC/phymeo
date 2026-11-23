# 已保存12帧的真实离线读回诊断

此记录来自父执行者已落盘结果的只读核对；作者未运行GPU或数值程序。原件均保留在
`C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/`。

| 文件 | SHA256 |
|---|---|
| temporal-raw-01/summary.json | 8d89e739a2c19a42e9da12824484f9ce282066b99175113f366a1ecdd803c78a |
| temporal-review-01/summary.json | f48159d31d0e3a6423993349630047f5265246da409522ccbb955e065ca57372 |
| temporal-review-02/summary.json | 9e742c81d52c8f5b3fe8d79833749df056096675351c7540874f944f11083576 |

真实输入是原Sparse全率guide，tile4、每anchor2条proposal、visibility-ratio/R2、
one-hop，55至66共12个原电影状态。每帧约17,445–19,320个真正traced anchor，
不是全图每像素2spp；SH fallback没有晋升为光线追踪历史。

首个离线评审错误地要求物理tick每个电影帧+1，导致primary-only每步都在几何判定前
被`discontinuous_tick`清空。原档55→56实际tick2987→3040，运动时间
1.866875→1.9000000000000001s；后续是53/54 tick间隔。该首错不证明没有对应。

修复保原物理key不动，以完整登记序列/previous exact key/collection SHA证明
相邻电影帧；review-02复用原NPZ，没有重采。结果：

- `reset-any`复用0：每步确有`scene_transforms`，是预声明保守策略的真实限制。
- `primary-only`累计复用60,279个**像素×帧观察**，非60,279个唯一物理点。
  除首帧外各帧3,375–10,024；该策略未验证移动物体导致的二次可见性改变。
- 主拒绝项为`domain`，即重投影落点没有上帧真实traced anchor/可用历史：
  约6,271–13,134/帧；其次`clamp-support`约1,981–2,896。ID、法向、深度少量拒绝；
  本段roundtrip拒绝为0。这些是anchor-only算法域，不能当成full-rate GPU的统计。
- 已接纳incoming相对当前估计的RMS变化约0.00153至0.00717；signed luminance
  多为负，约−6.66e−6至−3.21e−4。这是修改量/偏移，**不是对真值误差，也不是降噪率**。
- 报告`bindings_and_guards_passed`只签源与拒绝像素current保真；无独立incoming参考，
  `quality_claim=false`和`performance_claim=false`保持。

原始公共RGB从未被此CPU算法改变。GPU compact_history为另一个实现，
其全率重建incoming支持域与这里的原始anchor-only数据不同，须独立验收。

## 同一真实场景的静态表面 / 小相机运动正控

父已实际执行README预注册的原pose59静态节点控制：同材质/灯光/相机内参，
相机沿自身+X从0移至0.1mm、0.2mm。源节点严格不变、实际相机变换和独立authored
时钟已核对；这是原录制场景中的新光学实验，不冒物理轨迹延续。

| 文件 | SHA256 |
|---|---|
| temporal-camera-01/summary.json | f65c830adb6ed1f95b2528acd98fcdb73d017b8f45561a80a24cbb3ab0511ec8 |
| temporal-camera-review-01/summary.json | 93275215f2b113e6333375745a6d1586d49af774ffce96d8576549e72f0a5549 |

两政策均为`bindings_and_guards_passed`；首帧0，后两帧分别15,954和15,947个
历史复用像素，合计31,901个像素×帧观察，超过预声明的至少16个门。
全部拒绝像素仍精确保留当前估计。该正控证实代码确能在真实微小相机移动下找到
非零对应；它没有证明运动手臂的二次可见性安全、降噪质量或GPU速度。
`quality_claim=false`、`performance_claim=false`、`gpu_temporal_execution=false`。

验证记录为父串行执行的`queue14-time-D-host.log`及`queue15-D-host-camera.log`；
后者两项相机来源/变换负控通过。作者只做文件读回，不另行执行项目。
