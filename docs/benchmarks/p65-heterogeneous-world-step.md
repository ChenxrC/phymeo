# 不同动力学模型在同一物理子步发布

`phymeo.TyxBatchedScene` 为独立世界按实际求解器分组。自由球和受控滑轨共用常驻状态及一个推进核。三节点布与有限胶囊滑轨共用材料力核、几何验收和实际节点/滑轨积分。世界键在两组间唯一，原材料和驱动历史保留。下面的 `worlds` 可交错登记 `FreeSphere`、`PrismaticSlider` 和 `ClothLinkWorld`。[公共实现](../../src/phymeo/runtime/batched_scene.py) · [完整构造例](../../tests/scale/test_batched_scene.py)

```python
with ph.TyxBatchedScene(worlds, "cuda") as scene:
    scene.set_controls({slider_id: drive_packet})
    result = scene.step(1 / 256, {cloth_id: cloth_loads})
    if result.accepted[cloth_id]:
        normal_n = result.contact_results[cloth_id].audit.sample.normal_force_n
    state = scene.snapshot([slider_id, cloth_id])
```

材料组先计算待发布状态及力的原始记录。常驻组随后推进，并在同次设备状态读回后切换缓冲。两组都完成后才交付该步的组合结果。某一世界因自身容量不足被拒绝时，其材料状态与上一次的力审计仍然可读。设备异常则使整次调用的已发布状态保持在入步点。显式的局部复位或恢复只作用于指定的世界。

## 四个同位世界

在同一 1/256 s 子步中，带载布–杆、2 kg 自由球、分离的布–杆和带一拍延迟的 2 kg 滑轨按交错键登记。首步带载杆的法向力为 **2 N**，杆速度为 0.015625 m/s，三个布节点的竖直速度分别为 −0.00260416674 m/s。另一组布–杆读到零接触力。重力为 −8 m/s² 的自由球从 2 m 开始，首步位置为 2−8/256² m、速度 −8/256 m/s。延迟滑轨首步仍为零力，下一步才实际输出 4 N。

第二步只将带载世界的材料站点容量限制为 1，返回该世界需要 2 个站点；另三个世界继续前进。恢复、重放和局部复位后，四个登记世界的 tick 依次为 0、2、3、0。另在常驻组的设备状态读回处注入错误，独立检查了四个世界的原始状态、上一材料审计和驱动队列，重试与原成功结果一致。

TYX interpret 与真实 CUDA 都通过这两组动态控制。首个 CUDA 子步的常驻组实际提交一个核，材料组完成六个核。该步没有为组间回滚额外下载常驻物理检查点；原有接受标记、材料速度/载荷的 HOST 复核和设备读回仍在。查看三阶段原始退出码及 `mixed-physical-step.json` 可从 `artifacts/p65-20261104-0800/validation/hybrid-stage-*` 及对应 `temp-*` 进入。自由球大量复制的端到端吞吐另有同负载对照；本例的多模型组合用于验收状态、力和失败处理。[同输出的 Newton/Phymeo 测量](p65-g2-newton-observed-free-fall.md) · [单材料组含载窗口](p65-g3-world-contact-loads.md)
