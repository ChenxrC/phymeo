"""Summarize saved capture timings and compose four untouched RGB PNGs.

Usage: python rendering/examples/render_performance_summary.py RESULTDIR
Only benchmark.json and its PNGs are read; no rendering backend is imported.
Pillow and the Microsoft YaHei font are needed for the Chinese contact sheet.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile


BUDGET_MS = 1000.0 / 30.0
PANEL_ORDER = ("common", "dense4", "high", "still")


def _samples(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field}: expected a nonempty capture_ms list")
    result = []
    for index, sample in enumerate(value):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError(f"{field}[{index}]: expected a finite positive number")
        try:
            measured = float(sample)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{field}[{index}]: unrepresentable milliseconds") from exc
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError(f"{field}[{index}]: milliseconds must be finite and positive")
        result.append(measured)
    return result


def timing_statistics(capture_ms):
    """Nearest rank quantiles: sorted[ceil(p*N)-1], with no interpolation."""
    samples = _samples(capture_ms, "capture_ms")
    ordered = sorted(samples)
    count = len(samples)
    mean = math.fsum(value / count for value in samples)
    try:
        total = math.fsum(samples)
    except OverflowError as exc:
        raise ValueError("capture_ms total cannot be represented") from exc
    if not math.isfinite(total):
        raise ValueError("capture_ms total cannot be represented")
    within = sum(value <= BUDGET_MS for value in samples)
    return {
        "count": count,
        "mean_ms": mean,
        "p50_ms": ordered[math.ceil(0.50 * count) - 1],
        "p95_ms": ordered[math.ceil(0.95 * count) - 1],
        "fps_from_mean": 1000.0 / mean,
        "recorded_capture_total_ms": total,
        "budget_ms": BUDGET_MS,
        "within_budget_count": within,
        "within_budget_ratio": within / count,
        "within_budget_percent": 100.0 * within / count,
    }


def _image_path(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("image: expected a relative PNG path")
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError("image must be relative to the result directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root / "comparison.png":
        raise ValueError("image must stay within resultdir and not alias comparison.png")
    if not path.is_file():
        raise ValueError(f"image does not exist: {name}")
    return path


def _load_image(root, name, size):
    from PIL import Image

    path = _image_path(root, name)
    with Image.open(path) as source:
        if source.format != "PNG" or source.size != size:
            raise ValueError(f"{name}: expected a {size[0]}x{size[1]} PNG")
        if source.mode not in ("RGB", "RGBA"):
            raise ValueError(f"{name}: expected RGB or opaque RGBA pixels")
        if source.mode == "RGBA" and source.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"{name}: alpha must be fully opaque")
        image = source.convert("RGB")
    return image, {
        "path": path.relative_to(root).as_posix(),
        "png_sha256": sha256(path.read_bytes()).hexdigest(),
        "decoded_rgb_sha256": sha256(image.tobytes()).hexdigest(),
        "width": size[0], "height": size[1],
    }


def _font(size):
    from PIL import ImageFont

    candidates = (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
        Path("C:/Windows/Fonts/msyh.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise ValueError("Chinese comparison labels require C:/Windows/Fonts/msyh.ttc")


def _text(draw, xy, text, *, size, width, color=(228, 233, 241)):
    """Fit only the exterior label; never resize or crop source pixels."""
    font = _font(size)
    while size > 14 and draw.textlength(text, font=font) > width:
        size -= 1
        font = _font(size)
    if draw.textlength(text, font=font) > width:
        raise ValueError("comparison label is too long to render legibly")
    draw.text(xy, text, font=font, fill=color)


def _comparison(report, images):
    from PIL import Image, ImageDraw

    width, height = report["width"], report["height"]
    title_h, panel_header_h, footer_h = 120, 112, 88
    canvas = Image.new("RGB", (2 * width, title_h + 2 * (panel_header_h + height) + footer_h), (18, 24, 34))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (28, 13), "真实渲染对照｜同一帧：单批与静态多批", size=40, width=2 * width - 56)
    _text(draw, (28, 72), f'{report["device"]}  ·  每格完整 {width}×{height} 原图  ·  无裁剪、无调色', size=25, width=2 * width - 56)
    by_id = {case["id"]: case for case in report["cases"]}
    by_id["still"] = report["still"]
    for index, key in enumerate(PANEL_ORDER):
        case = by_id[key]
        x = (index % 2) * width
        y = title_h + (index // 2) * (panel_header_h + height)
        stats = case["pooled"]
        kind = "静态多批" if key == "still" else "单批画面"
        title = f'{case["label"]}  [{key}]  ·  {kind}'
        _text(draw, (x + 24, y + 9), title, size=32, width=width - 48)
        if key == "still":
            line = f'已记录 {stats["count"]} 次 capture 合计 {stats["recorded_capture_total_ms"]:.2f} ms；不是运动帧 FPS'
        else:
            line = f'均值 {stats["mean_ms"]:.2f} ms  ·  p95 {stats["p95_ms"]:.2f} ms  ·  {stats["fps_from_mean"]:.2f} FPS'
        _text(draw, (x + 24, y + 55), line, size=25, width=width - 48, color=(166, 205, 223))
        canvas.paste(images[key], (x, y + panel_header_h))
    footer_y = canvas.height - footer_h
    _text(draw, (28, footer_y + 8), "单批统计合并所有实测轮次；p50 / p95 为最近秩。30 Hz 预算 = 33.333… ms。", size=25, width=2 * width - 56)
    _text(draw, (28, footer_y + 45), "FPS = 1000 / 平均 capture 毫秒；不代表含未计时工作的整条应用流水线，也不是速度保证。", size=23, width=2 * width - 56)
    return canvas


def markdown_table(report):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = [
        f'设备：{escape(report["device"])}；图像：{report["width"]}×{report["height"]}。', "",
        "| 档位 | 样本数 | 平均 ms | p50 ms | p95 ms | 1000/mean FPS | 各轮均值 ms | ≤30 Hz 预算 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for case in report["cases"]:
        stat = case["pooled"]
        means = " / ".join(f'{item["mean_ms"]:.3f}' for item in case["rounds"])
        lines.append(
            f'| {escape(case["label"])} ({escape(case["id"])}) | {stat["count"]} | '
            f'{stat["mean_ms"]:.3f} | {stat["p50_ms"]:.3f} | {stat["p95_ms"]:.3f} | '
            f'{stat["fps_from_mean"]:.2f} | {means} | '
            f'{stat["within_budget_count"]}/{stat["count"]} ({stat["within_budget_percent"]:.1f}%) |'
        )
    still = report["still"]["pooled"]
    lines.extend([
        "", f'静态多批：{still["count"]} 次已记录 capture，合计 {still["recorded_capture_total_ms"]:.3f} ms；不作为运动帧 FPS。',
        "", "p50/p95 使用最近秩（排序后第 ceil(p×N) 项），各轮逐 capture 合并；30 Hz 达标条件为 capture_ms ≤ 1000/30。",
        "计时范围沿 benchmark.json 的实际记录；PNG 后处理没有加入 capture_ms。",
    ])
    return "\n".join(lines)


def summarize(resultdir):
    root = Path(resultdir).resolve()
    source = root / "benchmark.json"
    source_bytes = source.read_bytes()
    benchmark = json.loads(source_bytes)
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark.json: expected an object")
    device = benchmark.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device: expected a nonempty string")
    width, height = benchmark.get("width"), benchmark.get("height")
    if type(width) is not int or type(height) is not int or (width, height) != (1280, 720):
        raise ValueError("this comparison requires the recorded 1280x720 extent")
    inputs = benchmark.get("cases")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("cases: expected a nonempty list")
    supplement_path = root / "additional.json"
    supplement_sha = None
    if supplement_path.is_file():
        supplement_bytes = supplement_path.read_bytes()
        extra = json.loads(supplement_bytes)
        if extra["base_sha256"] != sha256(source_bytes).hexdigest():
            raise ValueError("grid sweep belongs to a different camera run")
        inputs = inputs + extra["cases"]
        supplement_sha = sha256(supplement_bytes).hexdigest()
    cases, images, seen = [], {}, set()
    for case in inputs:
        if not isinstance(case, dict):
            raise ValueError("case: expected an object")
        key, label, rounds = case.get("id"), case.get("label"), case.get("rounds")
        if not isinstance(key, str) or not key or key in seen or key == "still":
            raise ValueError("case.id: expected a unique nonempty ID other than still")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{key}.label: expected a nonempty string")
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"{key}.rounds: expected at least one measured round")
        seen.add(key)
        pooled, round_stats = [], []
        for number, measured in enumerate(rounds, 1):
            if not isinstance(measured, dict):
                raise ValueError(f"{key}.rounds[{number}]: expected an object")
            samples = _samples(measured.get("capture_ms"), f"{key}.rounds[{number}].capture_ms")
            pooled.extend(samples)
            stat = timing_statistics(samples)
            stat["round"] = number
            stat["reported_rgb_sha256"] = measured.get("rgb_sha256")
            round_stats.append(stat)
        image, binding = _load_image(root, case.get("image"), (width, height))
        if any(r["reported_rgb_sha256"] != binding["decoded_rgb_sha256"] for r in round_stats):
            raise ValueError(f"{key}: published PNG differs from the measured frame")
        images[key] = image
        cases.append({"id": key, "label": label, "round_count": len(round_stats),
                      "rounds": round_stats, "pooled": timing_statistics(pooled), "image": binding})
    if not {"common", "dense4", "high"}.issubset(seen):
        raise ValueError("comparison requires common, dense4 and high case IDs")
    still = benchmark.get("still")
    if not isinstance(still, dict):
        raise ValueError("still: expected a measured static case")
    images["still"], binding = _load_image(root, still.get("image"), (width, height))
    if still.get("rgb_sha256") != binding["decoded_rgb_sha256"]:
        raise ValueError("static PNG differs from the last recorded batch")
    static_stats = timing_statistics(_samples(still.get("capture_ms"), "still.capture_ms"))
    report = {
        "schema": "phymeo-render-performance-summary-v1",
        "source": {"file": "benchmark.json", "sha256": sha256(source_bytes).hexdigest(),
                   "additional_json_sha256": supplement_sha},
        "device": device, "width": width, "height": height,
        "statistics": {"pooling": "all captures across all rounds, weighted by sample count",
                       "quantiles": "nearest rank: sorted[ceil(p*N)-1]",
                       "fps": "1000 / pooled mean capture_ms; not mean instantaneous FPS",
                       "budget_ms": BUDGET_MS, "budget_operator": "<=",
                       "timing_scope": benchmark["timing"],
                       "reported_rgb_hash_scope": "decoded PNG RGB pixels matched the measured hero capture in every round"},
        "cases": cases,
        "still": {"id": "still", "label": "静态多批", "image": binding, "pooled": static_stats,
                  "interpretation": "per-capture statistics and sum of recorded capture calls; not complete static-image FPS or unmeasured pipeline duration"},
    }
    if supplement_sha:
        import numpy as np
        reference = extra["reference"]
        ref_image, ref_binding = _load_image(root, reference["image"], (width, height))
        if reference["rgb_sha256"] != ref_binding["decoded_rgb_sha256"]:
            raise ValueError("independent reference image differs from capture")
        codes = np.arange(256, dtype=np.float64) / 255
        decode = np.where(codes <= .04045, codes / 12.92, ((codes + .055) / 1.055) ** 2.4)
        target = decode[np.asarray(ref_image)]
        rms = {key: float(np.sqrt(np.mean((decode[np.asarray(frame)] - target)**2)))
               for key, frame in images.items()}
        report["visual_comparison"] = {
            "reference": ref_binding, "seed": reference["seed"], "samples_per_pixel": reference["spp"],
            "scope": "Full-frame inverse sRGB RGB RMS for one registered pose. Independent finite sample of the same high preset at 1x, not ground truth or an AA/temporal quality ranking.",
            "display_linear_rmse": rms,
        }
    canvas = _comparison(report, images)
    # Write completed outputs atomically. Never overwrite a source image.
    with tempfile.NamedTemporaryFile(dir=root, suffix=".png", delete=False) as temp:
        png_temp = Path(temp.name)
    json_temp = None
    try:
        canvas.save(png_temp, format="PNG")
        report["comparison"] = {
            "file": "comparison.png", "sha256": sha256(png_temp.read_bytes()).hexdigest(),
            "width": canvas.width, "height": canvas.height,
            "panel_order": list(PANEL_ORDER), "source_pixels": "unscaled RGB copy; text outside image; no color transform",
            "font": "msyh.ttc",
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(dir=root, mode="w", encoding="utf-8", suffix=".json", delete=False) as temp:
            json_temp = Path(temp.name)
            temp.write(encoded)
        png_temp.replace(root / "comparison.png")
        json_temp.replace(root / "performance.json")
    finally:
        png_temp.unlink(missing_ok=True)
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="汇总真实渲染耗时并生成中文原图对照；不启动渲染。")
    parser.add_argument("resultdir", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.resultdir)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"render_performance_summary: {exc}\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(markdown_table(report))


if __name__ == "__main__":
    main()
