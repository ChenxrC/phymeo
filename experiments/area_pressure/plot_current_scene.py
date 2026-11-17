"""Plot saved I2b trajectories only; never generate or replay physics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = {}
    for path in sorted(args.records.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("schema") != "current-area-trajectory-v1":
            continue
        key = (item["case"], item["mode"], item["h"])
        if key in cases:
            raise ValueError("duplicate saved trajectory")
        cases[key] = item
    expected = {
        (case, mode, h)
        for mode in ("interpret", "cuda")
        for case, h in (("sliders", 0.0005), ("sliders", 0.00025), ("hinge", 0.0005))
    }
    if set(cases) != expected:
        raise ValueError("expected the six fixed original trajectories")
    args.output.mkdir(parents=True, exist_ok=False)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.facecolor": "#f6f7fa",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    colors = ("#7257c9", "#087e8b")
    for h, color in zip((0.0005, 0.00025), colors, strict=True):
        item = cases[("sliders", "cuda", h)]
        rows = item["records"]
        t = np.array([r["output_time"] for r in rows]) * 1000
        depth = np.array([r["output_depth"] for r in rows]) * 1000
        force = np.array([sum(r["normal_forces"]) for r in rows])
        energy = (
            np.array([r["relative_energy"] for r in rows])
            / item["summary"]["relative_energy_initial"]
        )
        label = f"h = {h * 1000:g} ms"
        axes[0, 0].plot(t, depth, color=color, label=label, linewidth=2)
        axes[0, 1].plot(t, force, color=color, label=label, linewidth=2)
        axes[1, 0].plot(t, energy, color=color, label=label, linewidth=2)
    axes[0, 0].axhline(0, color="#555", linewidth=0.8)
    axes[0, 0].set(
        title="① 双滑动体运动：入层 → 压缩 → 分离", ylabel="相对压入量 / mm（负值表示分离）"
    )
    axes[0, 0].set_xlim(0, 45)
    axes[0, 0].set_ylim(-3.2, 1.6)
    axes[0, 1].set(title="② 同一步实际求出的压力合力", ylabel="法向合力 / N")
    axes[0, 1].set_xlim(0, 45)
    fine = cases[("sliders", "cuda", 0.00025)]
    summary = fine["summary"]
    peak_row = max(fine["records"], key=lambda row: sum(row["normal_forces"]))
    axes[0, 1].annotate(
        f"峰值 {summary['peak_force']:.2f} N",
        xy=(peak_row["output_time"] * 1000, summary["peak_force"]),
        xytext=(29, summary["peak_force"] * 0.8),
        arrowprops={"arrowstyle": "->", "color": "#334155"},
        color="#334155",
    )
    axes[1, 0].set(
        title="③ 能量下降来自后向欧拉的数值耗散",
        ylabel="相对机械能 / 初始相对机械能",
        ylim=(0.65, 1.04),
    )
    for ax in axes.flat:
        ax.set_xlabel("时间 / ms")
    rows = cases[("hinge", "cuda", 0.0005)]["records"]
    t = np.array([r["output_time"] for r in rows]) * 1000
    area = np.array([r["pressure_area"] for r in rows]) * 1e6
    speed = np.array([r["output_v"][0] for r in rows])
    axes[1, 1].plot(t, area, color="#d97706", linewidth=2, label="接触面积")
    right = axes[1, 1].twinx()
    right.grid(False)
    right.plot(t, speed, color="#2171b5", linewidth=1.8, label="关节角速度")
    right.set_ylabel("关节角速度 / rad/s", color="#2171b5")
    axes[1, 1].set(title="④ 偏心转动：接触面积随当前姿态更新", ylabel="接触面积 / mm²")
    axes[1, 1].yaxis.label.set_color("#d97706")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(loc="best", frameon=False)
    axes[1, 1].legend(loc="upper left", frameon=False)
    right.legend(loc="upper right", frameon=False)
    fig.suptitle(
        "面积压力连续闭环：当前姿态 → 新接触面 → 双方真实运动",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "CUDA 实际轨迹｜两个质量均可运动｜没有预设运动轨迹",
        ha="center",
        color="#475569",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.02,
        "当前实现：TYX 位姿与动力学 + HOST 有限面裁剪。图中没有复跑物理；曲线全部来自测试保存记录。",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9), h_pad=2, w_pad=2.2)
    fig.savefig(args.output / "current-area-contact.png", dpi=150)
    plt.close(fig)
    summaries = [
        dict(case=key[0], mode=key[1], h=key[2], steps=len(item["records"]), **item["summary"])
        for key, item in sorted(cases.items())
    ]
    (args.output / "trajectory-summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
