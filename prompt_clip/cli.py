"""prompt-to-clip CLI: prompt -> image (PNG) -> animated clip (MP4)."""
import argparse
import sys
from pathlib import Path

from .generate import DEFAULT_MODEL, ImageRequest, generate_image, load_pipeline
from .motion import make_clip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="prompt-to-clip",
                                description="Generate an image from a prompt and animate it into a short MP4 clip.")
    p.add_argument("prompt")
    p.add_argument("-o", "--out", type=Path, default=Path("clip.mp4"), help="output MP4 (a PNG is saved next to it)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"diffusers text-to-image model (default {DEFAULT_MODEL})")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--size", type=int, default=1024, help="square image size in pixels")
    p.add_argument("--seed", type=int)
    p.add_argument("--seconds", type=float, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-enhance", action="store_true", help="do not append quality keywords to the prompt")
    p.add_argument("--allow-pickle-weights", action="store_true",
                   help="also load legacy .bin weights (pickle; only for models you trust)")
    a = p.parse_args(argv)

    try:
        pipe = load_pipeline(a.model, allow_pickle_weights=a.allow_pickle_weights)
        image = generate_image(pipe, ImageRequest(a.prompt, steps=a.steps, width=a.size, height=a.size,
                                                  seed=a.seed, enhance=not a.no_enhance))
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    png = a.out.with_suffix(".png")
    image.save(png)
    make_clip(image, a.out, seconds=a.seconds, fps=a.fps)
    print(f"image: {png}\nclip:  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
