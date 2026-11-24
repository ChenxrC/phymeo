"""Torch CUDA consumer for a registered RGB8 station image and PNG parity replay."""
from hashlib import sha256
import json
from pathlib import Path


class AmbiguousStationRead(ValueError):
    """Rejected observation, including the measured counts and transfer size."""

    def __init__(self, measurement):
        self.measurement = measurement
        super().__init__(f"unclassified or mixed workpiece in station window: {measurement['fraction']}")


def classify_station_rgb_cuda(rgb, *, consumer_stream, producer_stream=None):
    """Classify a ready CUDA uint8 HWC RGB tensor, including a strided RGB view.

    Pass the producer's stream if its final write was enqueued outside the
    current PyTorch stream. An external producer must establish device-side
    readiness on that stream and keep its image lease valid through return.
    This synchronous boundary waits for only the two reduced host counts.
    """
    import torch

    if not isinstance(rgb, torch.Tensor) or not rgb.is_cuda:
        raise ValueError("station camera expects a CUDA tensor")
    if rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("station camera expects an RGB8 image")
    h, w, _ = rgb.shape
    if min(h, w) < 32:
        raise ValueError("station inspection window is too small")
    if not isinstance(consumer_stream, torch.cuda.Stream) or consumer_stream.device != rgb.device:
        raise ValueError("consumer stream must belong to the image's CUDA device")
    if consumer_stream == torch.cuda.default_stream(rgb.device):
        raise ValueError("station consumer requires a non-default CUDA stream")
    if producer_stream is None:
        producer_stream = torch.cuda.current_stream(rgb.device)
    if not isinstance(producer_stream, torch.cuda.Stream) or producer_stream.device != rgb.device:
        raise ValueError("producer stream must belong to the image's CUDA device")

    x0, y0, x1, y1 = 3*w//8, 3*h//8, 5*w//8, 5*h//8
    roi_pixels = (x1-x0)*(y1-y0)
    with torch.cuda.device(rgb.device), torch.cuda.stream(consumer_stream), torch.inference_mode():
        if producer_stream != consumer_stream:
            consumer_stream.wait_stream(producer_stream)
        rgb.record_stream(consumer_stream)
        # Crop first. Signed subtraction preserves the CPU reference's exact thresholds.
        r,g,b = rgb[y0:y1, x0:x1].to(dtype=torch.int16).unbind(dim=2)
        orange = (r-g >= 18) & (r-b >= 35)
        blue = (b-r >= 32) & (g-r >= 8)
        counts = torch.stack((orange.sum(dtype=torch.int64), blue.sum(dtype=torch.int64)))
        host_counts = counts.to(device="cpu", non_blocking=False)
    block, hexagon = host_counts.tolist()
    pixels = {"block": block, "hex": hexagon}
    fractions = {kind: count / roi_pixels for kind,count in pixels.items()}
    measurement = dict(roi_xyxy=(x0,y0,x1,y1), roi_pixels=roi_pixels,
                       pixel_count=pixels, fraction=fractions, consumer_device=str(rgb.device),
                       consumer_stream=int(consumer_stream.cuda_stream),
                       d2h_copies=1, d2h_bytes=host_counts.numel()*host_counts.element_size())
    chosen = max(pixels, key=pixels.get)
    if fractions[chosen] < .55 or min(fractions.values()) > .12:
        raise AmbiguousStationRead(measurement)
    return chosen, measurement


def compare_archive(directory, *, device="cuda:0", permuted=False):
    """Replay four original station PNGs, one explicit image upload per observation."""
    import numpy as np
    from PIL import Image
    import torch

    from .sorting_reader import classify_station_rgb

    directory = Path(directory)
    reads = json.loads((directory/"render.json").read_text(encoding="utf-8"))["station_reads"]
    if len(reads) != 4 or len({row["image"] for row in reads}) != 4:
        raise ValueError("expected four distinct archived station images")
    consumer = torch.cuda.Stream(device=device)
    producer = torch.cuda.Stream(device=device)
    rows = []
    # The recorded labels form a palindrome. An interior frame first changes
    # the expected decisions as well as the identities of the exposures.
    order = (1,3,0,2) if permuted else range(len(reads))
    for index in order:
        archived = reads[index]
        name = archived["image"]
        if name != f"reader-{archived['frame']:03}.png":
            raise ValueError(f"unexpected station image name {name}")
        with Image.open(directory/name) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"expected an RGB8 PNG station image: {name}")
            rgb = np.array(image)
        digest = sha256(rgb.tobytes()).hexdigest()
        if digest != archived["rgb_sha256"]:
            raise ValueError(f"archived RGB content differs: {name}")
        cpu_kind, cpu = classify_station_rgb(rgb)
        if (cpu_kind, list(cpu["roi_xyxy"]), cpu["fraction"]) != (
                archived["decision"], archived["roi_xyxy"], archived["fraction"]):
            raise ValueError(f"CPU reference differs from recorded station read: {name}")
        with torch.cuda.stream(producer):
            device_rgb = torch.from_numpy(rgb).to(device=device, non_blocking=True)
        kind, measured = classify_station_rgb_cuda(device_rgb, consumer_stream=consumer, producer_stream=producer)
        x0,y0,x1,y1 = cpu["roi_xyxy"]
        roi_pixels = (x1-x0)*(y1-y0)
        cpu_counts = {key: round(fraction*roi_pixels) for key,fraction in cpu["fraction"].items()}
        if (kind, measured["roi_xyxy"], measured["fraction"]) != (cpu_kind, cpu["roi_xyxy"], cpu["fraction"]):
            raise ValueError(f"CUDA result differs from original CPU classification: {name}")
        if measured["pixel_count"] != cpu_counts:
            raise ValueError(f"CUDA pixel counts differ from original CPU classification: {name}")
        rows.append(dict(frame=archived["frame"], image=name, rgb_sha256=digest, decision=kind,
                         cpu_decision=cpu_kind, cpu_pixel_count=cpu_counts,
                         h2d_copies=1, h2d_bytes=rgb.nbytes, **measured))
    return dict(source="archived station PNGs uploaded from host memory", direct_vulkan_input=False,
                order="permuted" if permuted else "recorded", reads=rows,
                transfers={key: sum(row[key] for row in rows)
                           for key in ("h2d_copies","h2d_bytes","d2h_copies","d2h_bytes")})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare_archive(args.archive, device=args.device, permuted=args.permuted), indent=2))
