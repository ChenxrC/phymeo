"""Build a self-contained result viewer from saved JSON only; never simulate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(kamino, fft, output, analysis=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = {"kamino": [], "fft": []}
    phase_costs = (
        {c["id"]: c["phase_costs"] for c in read(Path(analysis))["cases"]} if analysis else {}
    )
    ks = read(Path(kamino) / "summary.json")
    for case in ks["cases"]:
        metrics, timing = case["metrics"], case["timing"]
        rawframes = read(Path(kamino) / case["frames"])["frames"]
        frames = [
            {
                "t": f["t"],
                "pose": f["body_pose"],
                "force": f["contact_force_on_box"],
                "push": f["input_force"],
                "valid": f["contact_data_valid"],
            }
            for f in rawframes
        ]
        error = ""
        if case.get("failure"):
            failure = read(Path(kamino) / case["failure"])
            error = failure["type"] + ": " + failure["message"]
        data["kamino"].append(
            {
                "id": case["id"],
                "label": case["backend"].upper() + " · " + str(case["dt"] * 1000) + " ms",
                "status": case["status"],
                "support": metrics["settle_force_mean_n"],
                "dx": metrics["displacement_x_m"],
                "step_ms": None
                if timing["steps_after_100ms_mean_s"] is None
                else timing["steps_after_100ms_mean_s"] * 1000,
                "frames": frames,
                "error": error,
                "caption": "读取实际CUDA轨迹，按10倍慢放。三维投影；末端若未再查询接触，不显示或补造端点接触力。",
                "verdict": "功能判据："
                + ("通过" if metrics["functional_pass"] else "未全部通过")
                + "；求解器每步收敛："
                + ("是" if metrics["solver_converged"] else "否")
                + "。小规模同步耗时包含内部碰撞及接触观察，不代表批量GPU吞吐。",
                "loaded_step_ms": phase_costs.get(case["id"], {})
                .get("loaded", {})
                .get("step_mean_ms"),
                "raw_metrics": metrics,
                "raw_timing": timing,
            }
        )
    fs = read(Path(fft) / "summary.json") if fft is not None else {"cases": []}
    for case in fs["cases"]:
        metrics = case.get("metrics") or {}
        hertz = metrics.get("hertz", {})
        reference = hertz.get("reference", {})
        row = {
            "id": case["id"],
            "label": str(case["grid"]) + "² · " + str(round(case["delta"] * 1e6)) + " μm",
            "status": case["status"],
            "force": metrics.get("force_n"),
            "force_error": hertz.get("force_relative_error"),
            "area_error": hertz.get("area_relative_error"),
            "solve_s": case["timing"].get("solver_s"),
            "p0": reference.get("peak_pressure_pa"),
            "kkt_pass": metrics.get("kkt", {}).get("passed", False),
            "hertz_pass": hertz.get("target_passed", False),
            "pressure_scale_pa": 70000,
            "preview": None,
            "error": case.get("failure", {}).get("message", ""),
            "caption": "压力/位移取原网格中心±4 mm，未插值。原160 mm非周期域和完整数组保存在结果文件。",
            "verdict": "离散接触约束："
            + ("通过" if metrics.get("kkt", {}).get("passed") else "未通过或未取得")
            + "；Hertz载荷/面积/峰值5%目标："
            + ("通过" if hertz.get("target_passed") else "未通过或未取得")
            + "。误差与耗时分开记录；不通过不等于接触模型在所有尺度无效。",
            "raw_metrics": metrics,
            "raw_timing": case["timing"],
        }
        p = case.get("preview")
        if p:
            crop, cross = p["contact_zoom"], p["center_line"]
            keep = [i for i, x in enumerate(cross["x_m"]) if abs(x) <= 0.004]
            row["preview"] = {
                "x_mm": [v * 1000 for v in crop["x_m"]],
                "pressure": [list(v) for v in zip(*crop["pressure_pa"])],
                "displacement": [list(v) for v in zip(*crop["displacement_m"])],
                "cross_x_mm": [cross["x_m"][i] * 1000 for i in keep],
                "cross_pressure_kpa": [cross["pressure_pa"][i] / 1000 for i in keep],
                "cross_hertz_kpa": [cross["hertz_pressure_pa"][i] / 1000 for i in keep],
                "cross_displacement_um": [cross["displacement_m"][i] * 1e6 for i in keep],
                "cross_hertz_displacement_um": [
                    cross["hertz_displacement_m"][i] * 1e6 for i in keep
                ],
            }
        data["fft"].append(row)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (output / "selection_data.js").write_text(
        "window.SELECTION_DATA=" + encoded + ";\n", encoding="utf-8"
    )
    (output / "index.html").write_bytes(Path(__file__).with_name("viewer.html").read_bytes())
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "kamino": str(Path(kamino).resolve()),
                "fft": str(Path(fft).resolve()) if fft is not None else None,
                "mode": "saved-output-only-no-producer-calls",
                "analysis": str(Path(analysis).resolve()) if analysis is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "kamino_cases": len(data["kamino"]),
                "fft_cases": len(data["fft"]),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kamino", type=Path, required=True)
    p.add_argument("--fft", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analysis", type=Path)
    args = p.parse_args()
    build(args.kamino, args.fft, args.output, args.analysis)
