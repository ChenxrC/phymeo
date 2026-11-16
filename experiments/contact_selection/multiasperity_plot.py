"""Render saved double-asperity data. No contact solve or fitting is called."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def render(reference, selection, output):
    reference, output = Path(reference), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    data = {}
    for depth in (20, 40, 60, 80, 100):
        folder = reference / f"c-n512-d{depth:03d}um"
        case = json.loads((folder / "case.json").read_text(encoding="utf-8"))
        with np.load(folder / "fields.npz", allow_pickle=False) as arrays:
            fields = {k: arrays[k].copy() for k in ("x_m", "y_m", "heights", "p")}
        data[depth] = (case, fields)
    selected = json.loads(Path(selection).read_text(encoding="utf-8"))
    model = selected["model"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.3), layout="constrained")
    fig.suptitle(
        "双峰压弹性平面：接触区会彼此影响\n静态 FFT 数值参考 · 512²物理网格 · E*=1 MPa", fontsize=19
    )

    def field_image(ax, fields, values, title, cmap="magma", vmax=None, vmin=None):
        x, y = fields["x_m"] * 1000, fields["y_m"] * 1000
        ix, iy = np.flatnonzero(abs(x) <= 7), np.flatnonzero(abs(y) <= 5)
        dx, dy = x[1] - x[0], y[1] - y[0]
        image = ax.imshow(
            values[np.ix_(ix, iy)].T,
            origin="lower",
            extent=(x[ix[0]] - dx / 2, x[ix[-1]] + dx / 2, y[iy[0]] - dy / 2, y[iy[-1]] + dy / 2),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set(title=title, xlabel="x / mm", ylabel="y / mm")
        return image

    fields = data[100][1]
    image = field_image(
        axes[0, 0],
        fields,
        fields["heights"] * 1e6,
        "① 高度：左峰先接触，右峰低 40 μm",
        cmap="viridis",
        vmin=-200,
        vmax=0,
    )
    fig.colorbar(image, ax=axes[0, 0], label="表面高度 / μm", shrink=0.8)
    depths = list(data)
    forces = [data[d][0]["metrics"]["force_n"] for d in depths]
    ax = axes[0, 1]
    ax.plot(depths, forces, "o-", label="共同 FFT 求解", color="#0e7490")
    ax.scatter(
        [20, 60, 100],
        [forces[i] for i in (0, 2, 4)],
        s=95,
        marker="s",
        facecolors="none",
        edgecolors="#1d4ed8",
        label="标定点",
    )
    ax.scatter(
        [40, 80],
        [v["predicted_n"] for v in model["validation"]],
        marker="x",
        s=90,
        color="#d97706",
        label="简化模型的未拟合点预测",
    )
    ax.set(title="② 总力：标定点与未拟合点分开", xlabel="压入 / μm", ylabel="法向力 / N")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    ax = axes[0, 2]
    for peak, label, color in ((0, "左峰", "#0e7490"), (1, "右峰", "#b45309")):
        ax.plot(
            depths,
            [data[d][0]["metrics"]["per_peak"][peak]["force_n"] for d in depths],
            "o-",
            label=label,
            color=color,
        )
    ax.set(title="③ 两个峰分别承担多少载荷？", xlabel="压入 / μm", ylabel="每峰法向力 / N")
    ax.grid(alpha=0.2)
    ax.legend()
    maximum = max(float(np.max(data[d][1]["p"])) for d in (20, 60, 100)) / 1000
    for ax, d in zip(axes[1], (20, 60, 100)):
        case, fields = data[d]
        image = field_image(
            ax,
            fields,
            fields["p"] / 1000,
            f"压入 {d} μm：亮处承载，黑处不接触",
            vmax=maximum,
            vmin=0,
        )
        cop = case["metrics"]["cop_xy_m"]
        if cop is not None:
            ax.scatter(
                cop[0] * 1000,
                cop[1] * 1000,
                marker="+",
                color="cyan",
                s=130,
                linewidths=2,
                label="压力中心",
            )
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(image, ax=ax, label="压力 / kPa", shrink=0.8)
    fig.savefig(output / "double-asperity-explained.png", dpi=150)
    plt.close(fig)

    fields = data[100][1]
    single = []
    for index in (1, 2):
        with np.load(
            reference / f"i{index}-n512-d100um" / "fields.npz", allow_pickle=False
        ) as arrays:
            single.append(arrays["p"].copy())
    independent = single[0] + single[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained")
    fig.suptitle("100 μm：分别压两个峰再相加，并不等于共同压入", fontsize=17)
    maximum = max(float(np.max(independent)), float(np.max(fields["p"]))) / 1000
    for ax, values, title in zip(
        axes[:2],
        (fields["p"], independent),
        ("共同求解（包含远处弹性位移）", "两次独立求解的压力相加"),
    ):
        image = field_image(ax, fields, values / 1000, title, vmax=maximum, vmin=0)
        fig.colorbar(image, ax=ax, label="kPa", shrink=0.75)
    difference = (independent - fields["p"]) / 1000
    bound = float(np.max(abs(difference)))
    image = field_image(
        axes[2], fields, difference, "独立相加 − 共同求解", cmap="RdBu_r", vmin=-bound, vmax=bound
    )
    fig.colorbar(image, ax=axes[2], label="压力差 / kPa", shrink=0.75)
    fig.savefig(output / "coupling-difference.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "figures": [
                    str(output / "double-asperity-explained.png"),
                    str(output / "coupling-difference.png"),
                ],
                "source": str(reference),
                "new_solves": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.reference, args.selection, args.output)
