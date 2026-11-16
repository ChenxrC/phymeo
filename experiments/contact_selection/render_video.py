"""Render a short explanatory video exclusively from the saved viewer data."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = "#101c26"
FG = "#e6f1f4"
CYAN = "#69dfca"
ORANGE = "#ffc177"
FONT = "C:/Windows/Fonts/msyh.ttc"


def draw_text(draw, xy, value, size=26, color=FG):
    draw.text(xy, value, fill=color, font=ImageFont.truetype(FONT, size))


def rotate(q, p):
    x, y, z, w = q
    a, b, c = p
    t = (2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a))
    return (
        a + w * t[0] + y * t[2] - z * t[1],
        b + w * t[1] + z * t[0] - x * t[2],
        c + w * t[2] + x * t[1] - y * t[0],
    )


def project(p):
    return (210 + 1700 * p[0] + 450 * p[1], 240 - 950 * p[2] + 350 * p[1])


def box(draw, pos, q, half, color):
    vertices = []
    for z in (-1, 1):
        for y in (-1, 1):
            for x in (-1, 1):
                v = rotate(q, (x * half[0], y * half[1], z * half[2]))
                vertices.append(project(tuple(v[i] + pos[i] for i in range(3))))
    for face in (
        (0, 1, 3, 2),
        (0, 4, 5, 1),
        (1, 5, 7, 3),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (4, 6, 7, 5),
    ):
        points = [vertices[i] for i in face]
        draw.polygon(points, fill=color)
        draw.line(points + [points[0]], fill="#b6ccd5", width=2)


def color(t):
    t = max(0, min(1, t))
    stops = ((31, 43, 86), (40, 120, 142), (55, 181, 126), (243, 222, 73))
    j = min(2, int(t * 3))
    u = t * 3 - j
    return tuple(round(stops[j][k] * (1 - u) + stops[j + 1][k] * u) for k in range(3))


def frame(data, t):
    image = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(image)
    draw_text(d, (35, 20), "Phymeo · 算法选择实验", 20, CYAN)
    if t < 9:
        a = data["kamino"][3]
        frames = a["frames"]
        f = frames[min(len(frames) - 1, int(t / 9 * len(frames)))]
        draw_text(d, (35, 60), "① Kamino：箱子能站稳，受推后会怎样？", 34)
        box(d, (0, 0, -0.005), (0, 0, 0, 1), (0.04, 0.04, 0.005), "#385363")
        box(d, f["pose"][:3], f["pose"][3:], (0.02, 0.02, 0.01), "#329b8b")
        phase = (
            "重力支撑" if f["t"] < 0.2 else ("4 N真实外力" if f["t"] < 0.4 else "撤力后继续运动")
        )
        draw_text(d, (760, 160), phase, 32, ORANGE)
        draw_text(d, (760, 215), "DVI / 0.5 ms / CUDA", 27)
        draw_text(d, (760, 265), "真实时刻 " + format(f["t"], ".3f") + " s", 27)
        draw_text(
            d,
            (760, 315),
            "竖直支撑 " + (format(f["force"][2], ".3f") + " N" if f["valid"] else "端点未重查"),
            27,
        )
        draw_text(d, (760, 365), "箱子位置来自实际求解", 26)
        draw_text(d, (760, 410), "没有规定滑落动画", 26)
        if f["push"][0]:
            p = project(f["pose"][:3])
            d.line((p[0], p[1] - 35, p[0] + 100, p[1] - 35), fill=ORANGE, width=5)
            d.polygon(
                ((p[0] + 100, p[1] - 35), (p[0] + 85, p[1] - 44), (p[0] + 85, p[1] - 26)),
                fill=ORANGE,
            )
        draw_text(d, (40, 635), "0.6秒真实轨迹慢放 · 约0.361秒失去有限台面支撑", 25, CYAN)
    elif t < 18:
        n = (128, 256, 512)[min(2, int((t - 9) / 3))]
        a = next(a for a in data["fft"] if a["id"] == "n" + str(n) + "-d100um")
        draw_text(d, (35, 60), "② FFT：球压入100微米，压力算得多准？", 34)
        p = a["preview"]
        if p:
            values = p["pressure"]
            ny, nx = len(values), len(values[0])
            for j in range(ny):
                for i in range(nx):
                    x = 120 + 380 * i / nx
                    y = 170 + 380 * (ny - 1 - j) / ny
                    d.rectangle(
                        (x, y, x + 380 / nx + 1, y + 380 / ny + 1), fill=color(values[j][i] / 70000)
                    )
            draw_text(d, (120, 570), "中心±4 mm原格点 · 色标0–70 kPa", 22)
        else:
            draw_text(d, (120, 300), "本格未取得有效压力数据", 27, ORANGE)
        draw_text(d, (640, 170), str(n) + " × " + str(n) + " 网格", 36, CYAN)
        draw_text(
            d,
            (640, 235),
            ("求解通过 · 5%精度目标通过" if a["hertz_pass"] else "求解通过 · 面积精度未达5%目标")
            if a["kkt_pass"]
            else "离散求解未通过，参见原始状态",
            24,
            ORANGE,
        )
        for j, (label, val) in enumerate(
            (
                ("载荷误差", None if a["force_error"] is None else a["force_error"] * 100),
                ("面积误差", None if a["area_error"] is None else a["area_error"] * 100),
            )
        ):
            draw_text(
                d,
                (640, 290 + 52 * j),
                label + ("：未取得" if val is None else "：" + format(val, ".2f") + "%"),
                29,
            )
        draw_text(
            d,
            (640, 410),
            "求解：" + ("未取得" if a["solve_s"] is None else format(a["solve_s"], ".3f") + " s"),
            29,
        )
        draw_text(d, (640, 475), "更细是否值得，看误差与成本", 27)
        draw_text(d, (40, 635), "外部CPU静态材料参考 · 切换网格，不是物体运动", 25, CYAN)
    else:
        draw_text(d, (35, 65), "③ 可以组合，但必须解决不同层的问题", 35)
        for y, title, detail in (
            (160, "FFT材料参考", "压力分布、非局部变形、离线标定"),
            (280, "选择运行时接触模型", "在需要的精度范围内控制计算成本"),
            (400, "整体动力学求解", "更新机器人与物体，力和运动共同发布"),
        ):
            d.rounded_rectangle((100, y, 1180, y + 90), radius=12, fill="#223946")
            draw_text(d, (125, y + 12), title, 29, CYAN)
            draw_text(d, (500, y + 20), detail, 25)
        draw_text(d, (110, 535), "这是待验证的组合路线，本轮尚未接入TYX。", 28, ORANGE)
        draw_text(d, (110, 590), "同一接触不能被两个模型重复施力。", 28)
    draw_text(
        d,
        (35, 690),
        "实际保存数据重放 | 独立外部参考 | 完整参数、失败和成本见实验报告",
        16,
        "#a6bbc5",
    )
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viewer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("video output must be new")
    raw = (args.viewer / "selection_data.js").read_text(encoding="utf-8")
    data = json.loads(raw.removeprefix("window.SELECTION_DATA=").rstrip().removesuffix(";"))
    command = [
        args.ffmpeg,
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x720",
        "-r",
        "20",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i in range(480):
            im = frame(data, i / 20)
            if i in (120, 300, 420):
                im.save(args.output.with_name(args.output.stem + "-" + str(i) + ".png"))
            process.stdin.write(im.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("video encoder failed")
    print(json.dumps({"video": str(args.output), "frames": 480, "fps": 20, "seconds": 24}))


if __name__ == "__main__":
    main()
