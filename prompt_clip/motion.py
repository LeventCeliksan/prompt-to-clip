"""Turn a still image into a short clip with a slow zoom, gentle sway and drift (Ken Burns style)."""
from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def ken_burns_frames(image: Image.Image, seconds: float = 3, fps: int = 30, zoom: float = 0.25,
                     max_angle: float = 3.0, drift: int = 20):
    """Yield RGB frames (numpy arrays) the same size as `image`."""
    image = image.convert("RGB")
    w, h = image.size
    total = max(1, int(round(seconds * fps)))
    # Smallest zoom that keeps the rotated, drifted crop inside the image, so no black corners appear.
    rad = math.radians(max_angle)
    cover = math.cos(rad) + math.sin(rad) * max(w, h) / min(w, h) + 2 * drift / min(w, h)
    for i in range(total):
        t = i / total
        scale = cover + t * zoom
        angle = math.sin(t * 2 * math.pi) * max_angle
        zoomed = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        rotated = zoomed.rotate(angle, resample=Image.BICUBIC, expand=True)
        cx = rotated.width // 2 + int(math.sin(t * math.pi) * drift)
        cy = rotated.height // 2 + int(math.cos(t * math.pi) * drift / 2)
        left = min(max(0, cx - w // 2), rotated.width - w)
        top = min(max(0, cy - h // 2), rotated.height - h)
        yield np.asarray(rotated.crop((left, top, left + w, top + h)))


def make_clip(image: Image.Image, out_path, seconds: float = 3, fps: int = 30, **motion) -> Path:
    """Write an H.264 MP4 (yuv420p, plays in browsers). Width and height are rounded down to even numbers."""
    w, h = image.size
    image = image.crop((0, 0, w - w % 2, h - h % 2))
    out_path = Path(out_path)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                            macro_block_size=1) as writer:
        for frame in ken_burns_frames(image, seconds, fps, **motion):
            writer.append_data(frame)
    return out_path
