"""Plot exactly six saved loaded-support runs; never rerun contact or physics.

Forces use saved input-phase contact samples. Depth and velocity use accepted
output states. Static load/K guides use the registered reference stiffness
stored in each JSON, not a stiffness fitted to the observed trajectory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FILES = {
    ("single", "interpret", 0.001): "single-interpret-0.001.json",
    ("single", "cuda", 0.001): "single-cuda-0.001.json",
    ("dual", "interpret", 0.001): "dual-interpret-0.001.json",
    ("dual", "interpret", 0.0005): "dual-interpret-0.0005.json",
    ("dual", "cuda", 0.001): "dual-cuda-0.001.json",
    ("dual", "cuda", 0.0005): "dual-cuda-0.0005.json",
}
STAGES = ((0.15, 0.0), (0.35, -3.0), (0.55, 9.0))
PLATEAUS = (("gravity", 0.12, 0.15), ("loaded", 0.32, 0.35))


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _one(value, name):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must be the saved one-DOF vector")
    return _number(value[0], name)


def _vector(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a saved JSON vector")
    return np.asarray([_number(item, name) for item in value], dtype=np.float64)


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON token: {value}")


def _read(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("each input must be an existing absolute JSON path")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant)
    if not isinstance(payload, dict) or payload.get("schema") != "loaded-area-trajectory-v1":
        raise ValueError(f"unsupported trajectory schema: {path}")
    key = (payload.get("case"), payload.get("mode"), _number(payload.get("h"), "h"))
    if key not in FILES or path.name != FILES[key]:
        raise ValueError("input case/mode/h and registered filename disagree")
    case, mode, h = key
    expected_names = ["center"] if case == "single" else ["left", "right"]
    reference = {"center": 9000.0} if case == "single" else {"left": 4500.0, "right": 9000.0}
    if payload.get("patch_names") != expected_names or payload.get("pressure_domain") != "full":
        raise ValueError("wrong patch ordering or pressure domain")
    fixed = {
        "mass_kg": 0.5,
        "joint_damping_ns_m": 40.0,
        "gravity_m_s2": -9.81,
        "body_origin_z_m": 0.002,
    }
    for name, value in fixed.items():
        if _number(payload.get(name), name) != value:
            raise ValueError(f"different registered fixture parameter: {name}")
    saved_reference = payload.get("patch_reference_stiffness_n_m")
    if not isinstance(saved_reference, dict) or set(saved_reference) != set(reference):
        raise ValueError("missing registered patch stiffness metadata")
    if any(_number(saved_reference[name], name) != value for name, value in reference.items()):
        raise ValueError("different registered full-patch stiffness")
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("expected the three registered load stages")
    for actual, (end, force) in zip(stages, STAGES, strict=True):
        if (
            _number(actual.get("end_s"), "end_s") != end
            or _number(actual.get("force_n"), "force_n") != force
        ):
            raise ValueError("saved load stages disagree with the registered experiment")
    records = payload.get("records")
    count = round(0.55 / h)
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("expected one complete saved trajectory, without missing steps")
    values = {
        name: []
        for name in (
            "input_time",
            "output_time",
            "nominal_time",
            "stage",
            "input_velocity",
            "velocity",
            "depth_m",
            "external_n",
            "required_n",
            "total_force_n",
            "area_m2",
        )
    }
    patch_forces = {name: [] for name in expected_names}
    previous = None
    for index, record in enumerate(records):
        if type(record["input_tick"]) is not int or record["input_tick"] != index:
            raise ValueError("input ticks must cover every recorded step from zero")
        if type(record["output_tick"]) is not int or record["output_tick"] != index + 1:
            raise ValueError("output tick is not input tick plus one")
        stage = next(i for i, (end, _) in enumerate(STAGES) if index < round(end / h))
        if type(record["stage"]) is not int or record["stage"] != stage:
            raise ValueError("saved stage does not match its registered tick window")
        tin = _number(record["input_time"], "input_time")
        tout = _number(record["output_time"], "output_time")
        qin, vin = _one(record["input_q"], "input_q"), _one(record["input_v"], "input_v")
        qout, vout = _one(record["output_q"], "output_q"), _one(record["output_v"], "output_v")
        external = _one(record["applied_generalized_force"], "applied_generalized_force")
        if not 0 <= tin < tout or (previous is not None and (tin, qin, vin) != previous):
            raise ValueError("saved accepted/input phase chain is discontinuous")
        if external != STAGES[stage][1]:
            raise ValueError("actual held load does not match the saved stage")
        patches = record["patches"]
        if not isinstance(patches, list) or len(patches) != len(expected_names):
            raise ValueError("patch count changed in the saved trajectory")
        total, area = 0.0, 0.0
        for patch, name in zip(patches, expected_names, strict=True):
            if patch["pair"] != ["pressure", name]:
                raise ValueError("saved patch identity/order changed")
            stamp = patch["stamp"]
            if (stamp["world"], stamp["tick"], stamp["time"]) != ("13", index, tin):
                raise ValueError("patch geometry is not bound to this input state")
            force, weights = (
                _vector(patch["normal_force"], "normal_force"),
                _vector(patch["dA"], "dA"),
            )
            if force.shape != weights.shape or len(patch["sample_ids"]) != len(force):
                raise ValueError("saved sample force/area/identity counts disagree")
            if np.any(force < 0) or np.any(weights <= 0):
                raise ValueError("saved normal samples require nonnegative force and positive area")
            patch_total = float(np.sum(force))
            total += patch_total
            area += float(np.sum(weights))
            patch_forces[name].append(patch_total)
        row = dict(
            input_time=tin,
            output_time=tout,
            nominal_time=index * h,
            stage=stage,
            input_velocity=vin,
            velocity=vout,
            depth_m=-float(np.float32(np.float32(fixed["body_origin_z_m"]) + np.float32(qout))),
            external_n=external,
            required_n=max(0.0, -fixed["mass_kg"] * fixed["gravity_m_s2"] - external),
            total_force_n=total,
            area_m2=area,
        )
        for name, value in row.items():
            values[name].append(value)
        previous = (tout, qout, vout)
    series = {name: np.asarray(value) for name, value in values.items()}
    if not all(np.isfinite(value).all() for value in series.values()):
        raise ValueError("nonfinite aggregate in saved trajectory")
    return dict(
        key=key,
        case=case,
        mode=mode,
        h=h,
        source=str(path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        fixed=fixed,
        reference=reference,
        series=series,
        patch_forces={name: np.asarray(value) for name, value in patch_forces.items()},
    )


def _summary(case):
    data = case["series"]
    peak = int(np.argmax(data["total_force_n"]))
    plateaus = {}
    total_stiffness = sum(case["reference"].values())
    for name, start, end in PLATEAUS:
        selected = (data["nominal_time"] >= start) & (data["nominal_time"] < end)
        if not np.any(selected):
            raise ValueError("registered plateau has no saved samples")
        required = float(np.mean(data["required_n"][selected]))
        plateaus[name] = dict(
            selection="input_tick * h",
            start_s=start,
            end_exclusive_s=end,
            sample_count=int(np.count_nonzero(selected)),
            required_downward_load_n=required,
            total_contact_force_mean_n=float(np.mean(data["total_force_n"][selected])),
            patch_force_mean_n={
                key: float(np.mean(force[selected])) for key, force in case["patch_forces"].items()
            },
            accepted_depth_mean_m=float(np.mean(data["depth_m"][selected])),
            accepted_velocity_mean_m_s=float(np.mean(data["velocity"][selected])),
            registered_static_depth_m=required / total_stiffness,
        )
    empty = np.flatnonzero((data["stage"] == 2) & (data["area_m2"] == 0))
    area_positive = np.flatnonzero(data["area_m2"] > 0)
    return dict(
        id=f"{case['case']}-{case['mode']}-{case['h']:g}",
        case=case["case"],
        mode=case["mode"],
        h=case["h"],
        input_json=case["source"],
        input_sha256=case["sha256"],
        input_size_bytes=case["size_bytes"],
        records=len(data["input_time"]),
        registered_reference=dict(
            **case["fixed"],
            patch_stiffness_n_m=case["reference"],
            total_stiffness_n_m=total_stiffness,
        ),
        peak_total_contact_force_n=float(data["total_force_n"][peak]),
        peak_input_time_s=float(data["input_time"][peak]),
        plateaus=plateaus,
        final=dict(
            output_time_s=float(data["output_time"][-1]),
            depth_m=float(data["depth_m"][-1]),
            velocity_m_s=float(data["velocity"][-1]),
        ),
        first_empty_geometry_input_time_after_unload_s=None
        if not len(empty)
        else float(data["input_time"][empty[0]]),
        last_positive_geometry_area_input_time_s=None
        if not len(area_positive)
        else float(data["input_time"][area_positive[-1]]),
    )


def _font(font_manager):
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name(), str(path)
    raise RuntimeError(
        "Chinese scientific figure requires Microsoft YaHei, SimHei, or Noto Sans CJK"
    )


def render_saved_support(inputs, output):
    """Read six fixed JSONs and export one PNG plus auditable scalar summaries."""
    if len(inputs) != 6:
        raise ValueError("provide exactly the six registered saved JSON files")
    output = Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError("output must be an absolute fresh directory")
    loaded = [_read(path) for path in inputs]
    cases = {item["key"]: item for item in loaded}
    if len(cases) != 6 or set(cases) != set(FILES):
        raise ValueError("missing or duplicate registered case/mode/timestep")
    summaries = [_summary(cases[key]) for key in FILES]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font, font_path = _font(font_manager)
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.08, right=0.97, hspace=0.40, wspace=0.26)
    fig.suptitle("载荷支撑：TYX 真实步 / CPU 网格裁剪", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.923,
        "0.5 kg · 垂直导轨承担横向反力与力矩 · 不是自由体平衡或 FFT 模型",
        ha="center",
        fontsize=11,
    )
    curves = (
        (("single", "cuda", 0.001), "单支撑 1 ms", "#c86f19", "-"),
        (("dual", "cuda", 0.001), "双支撑 1 ms", "#1969a7", "-"),
        (("dual", "cuda", 0.0005), "双支撑 0.5 ms", "#09958d", "--"),
    )
    for key, label, color, style in curves:
        data = cases[key]["series"]
        axes[0, 0].plot(
            data["input_time"], data["total_force_n"], style, color=color, label=label, lw=1.7
        )
        axes[1, 0].plot(
            data["output_time"], data["depth_m"] * 1000, style, color=color, label=label, lw=1.7
        )
        axes[1, 1].plot(
            data["output_time"], data["velocity"], style, color=color, label=label, lw=1.7
        )
        empty = np.flatnonzero((data["stage"] == 2) & (data["area_m2"] == 0))
        if len(empty):
            index = int(empty[0])
            axes[1, 1].scatter(
                data["input_time"][index],
                data["input_velocity"][index],
                color=color,
                marker="o",
                s=24,
                zorder=5,
            )
    base = cases[("single", "cuda", 0.001)]["series"]
    axes[0, 0].step(
        base["input_time"],
        base["required_n"],
        where="post",
        color="#333333",
        ls=":",
        lw=1.7,
        label="向下静载需求 4.905 / 7.905 / 0 N",
    )
    for h, style in ((0.001, "-"), (0.0005, "--")):
        case = cases[("dual", "cuda", h)]
        for name, label, color in (
            ("left", "左接触区", "#b05a28"),
            ("right", "右接触区", "#5b51a3"),
        ):
            axes[0, 1].plot(
                case["series"]["input_time"],
                case["patch_forces"][name],
                style,
                color=color,
                lw=1.6,
                label=f"{label} · {h * 1000:g} ms",
            )
    for case_name, label, color in (("single", "单支撑", "#c86f19"), ("dual", "双支撑", "#1969a7")):
        stiffness = sum(cases[(case_name, "cuda", 0.001)]["reference"].values())
        axes[1, 0].hlines(
            4.905 / stiffness * 1000,
            0,
            0.15,
            color=color,
            ls=":",
            lw=1.5,
            label=f"{label}注册静态 load/K",
        )
        axes[1, 0].hlines(7.905 / stiffness * 1000, 0.15, 0.35, color=color, ls=":", lw=1.5)
    titles = (
        "A  总法向接触力（入步接触面）",
        "B  两个接触区共同分担载荷",
        "C  接触时段步末深度（0–0.38 s 放大）",
        "D  步末速度与卸载后分离",
    )
    ylabels = ("力 / N", "各接触区法向力 / N", "压入深度 / mm", "竖直速度 / (m/s)")
    for ax, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        ax.set_title(title, loc="left", fontsize=12, pad=10)
        ax.set_xlabel("时间 / s")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 0.55)
        ax.grid(alpha=0.22)
        ax.axvline(0.15, color="#777777", alpha=0.5, lw=0.9)
        ax.axvline(0.35, color="#777777", alpha=0.7, lw=0.9)
        ax.axvspan(0.35, 0.55, color="#d7eee3", alpha=0.35, zorder=0)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    axes[1, 0].axhline(0, color="#444444", lw=0.8)
    axes[1, 0].set_xlim(0, 0.38)
    axes[1, 0].set_ylim(-2.5, 1.3)
    axes[1, 0].text(
        0.98,
        0.05,
        "负深度为间隙；图窗 −2.5 至 1.3 mm\n完整末位移见汇总；外拉阶段无静态参考",
        transform=axes[1, 0].transAxes,
        ha="right",
        fontsize=8,
    )
    axes[1, 1].text(
        0.98,
        0.70,
        "0.35 s 起外拉 +9 N\n圆点：卸载后首次保存的空接触面（入步相位）",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.04,
        "曲线：3 份 CUDA 轨迹；汇总：全部 6 份 interpret/CUDA 原始 JSON。仅后处理，不重算物理。",
        ha="center",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=False)
    png = output / "loaded-support.png"
    try:
        fig.savefig(png, dpi=180, facecolor="white")
    finally:
        plt.close(fig)
    summary = dict(
        schema="phymeo-loaded-support-plot-v1",
        qualification="SAVED_TRAJECTORY_POSTPROCESS_ONLY",
        force_phase="saved input contact geometry / solved force",
        state_phase="accepted output pose and velocity",
        cases=summaries,
        depth_definition="-float32(float32(body_origin_z_m) + float32(output_q[0]))",
        depth_plot_view=dict(time_s=[0, 0.38], depth_mm=[-2.5, 1.3], summary_uses_all_records=True),
        plateau_windows="input_tick*h: [0.12,0.15), [0.32,0.35)",
        reference="registered metadata; no fitting to observed force or depth",
        scope="one-DOF guided body; guide carries transverse reactions and moments; not free balance or FFT",
        plot=str(png.resolve()),
        plot_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__,
        chinese_font=font_path,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs=6,
        required=True,
        help="six absolute saved trajectory JSON paths",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="absolute fresh output directory"
    )
    args = parser.parse_args()
    summary = render_saved_support(args.inputs, args.output)
    print(json.dumps({"cases": len(summary["cases"]), "plot": summary["plot"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
