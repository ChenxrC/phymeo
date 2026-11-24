"""Display recorded station-camera exposures and the routes selected at arrival."""
from bisect import bisect_right
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (25, 37, 42)
INK = (232, 239, 237)
BINS = {"block": ("A", (248, 182, 117)), "hex": ("B", (146, 212, 238))}


def _font(size):
    for name in ("msyh.ttc", "Deng.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size), False
        except OSError:
            continue
    return ImageFont.load_default(size=size), False


def _read_panel(read, directory, width, *, show_frame):
    with Image.open(directory / read["image"]) as source:
        if source.format != "PNG" or source.mode != "RGB":
            raise ValueError("expected a recorded RGB station PNG")
        if read.get("rgb_sha256") and sha256(source.tobytes()).hexdigest() != read["rgb_sha256"]:
            raise ValueError(f"station PNG differs from recorded RGB at frame {read['frame']}")
        sw, sh = source.size
        height = round(sh * width / sw)
        photo = source.resize((width, height), Image.Resampling.LANCZOS)

    # roi_xyxy is the half-open inspection window used for this exact exposure.
    x0, y0, x1, y1 = read["roi_xyxy"]
    if not (0 <= x0 < x1 <= sw and 0 <= y0 < y1 <= sh):
        raise ValueError("recorded station ROI is outside its image")
    left, top = round(x0 * width / sw), round(y0 * height / sh)
    right, bottom = round(x1 * width / sw) - 1, round(y1 * height / sh) - 1
    arm = min(max(3, round(width * .035)), (right - left) // 3, (bottom - top) // 3)
    overlay = ImageDraw.Draw(photo)
    for x, dx in ((left, 1), (right, -1)):
        for y, dy in ((top, 1), (bottom, -1)):
            corner = [(x + dx * arm, y), (x, y), (x, y + dy * arm)]
            overlay.line(corner, fill=BACKGROUND, width=3)
            overlay.line(corner, fill=INK, width=1)

    font, chinese = _font(round(width * .068))
    caption = "工位相机 · 到站抓拍" if chinese else "Station cam · Arrival"
    if show_frame:
        caption = ("帧 " if chinese else "Frame ") + f"{read['frame']:03d}"
    bin_id, accent = BINS[read["decision"]]
    route = f"→ {bin_id} 箱" if chinese else f"To bin {bin_id}"
    pad, banner = round(width * .025), round(width * .13)
    panel = Image.new("RGB", (width + 2 * pad, height + banner + pad), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((pad + 2, banner / 2), caption, font=font, fill=INK, anchor="lm")
    draw.text((panel.width - pad - 2, banner / 2), route, font=font, fill=accent, anchor="rm")
    panel.paste(photo, (pad, banner))
    return panel


def make_reader_inset(reads, output_dir, fps, size):
    """Return a compositor that displays the latest exposure for 1.1 seconds."""
    directory = Path(output_dir)
    width, height = size
    if min(width, height, fps) <= 0:
        raise ValueError("movie dimensions and fps must be positive")
    scale = min(width / 1280, height / 720)
    frames, panels = [], []
    for read in reads:
        if read["frame"] < 0 or (frames and read["frame"] <= frames[-1]):
            raise ValueError("expected chronological station exposures")
        frames.append(read["frame"])
        panel = _read_panel(read, directory, 200, show_frame=False)
        if scale != 1.:
            scaled = tuple(max(1, round(n * scale)) for n in panel.size)
            panel = panel.resize(scaled, Image.Resampling.LANCZOS)
        panels.append(panel)
    margin = round(14 * scale)
    duration = max(1, round(1.1 * fps))

    def composite(index, rgb_bytes):
        latest = bisect_right(frames, index) - 1
        if latest < 0 or index - frames[latest] >= duration:
            return rgb_bytes
        panel = panels[latest]
        picture = Image.frombytes("RGB", (width, height), rgb_bytes)
        picture.paste(panel, (width - panel.width - margin, margin))
        return picture.tobytes()

    return composite


def make_reader_board(reads, output_dir):
    """Write four actual exposure images in acquisition order and return the board path."""
    directory = Path(output_dir)
    reads = list(reads)
    if len(reads) != 4 or any(a["frame"] >= b["frame"] for a, b in zip(reads, reads[1:])):
        raise ValueError("expected the four successive arrivals")
    panels = [_read_panel(read, directory, 320, show_frame=True) for read in reads]
    gap, header = 12, 48
    board = Image.new("RGB", (sum(p.width for p in panels) + 5 * gap,
                              header + max(p.height for p in panels) + gap), (17, 27, 31))
    font, chinese = _font(23)
    title = "工位相机 · 到站抓拍" if chinese else "Station camera · Arrival captures"
    ImageDraw.Draw(board).text((gap + 5, header / 2), title, font=font, fill=INK, anchor="lm")
    x = gap
    for panel in panels:
        board.paste(panel, (x, header))
        x += panel.width + gap
    path = directory / "sorting-station-reads.png"
    board.save(path)
    return path
