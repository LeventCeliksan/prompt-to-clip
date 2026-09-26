import os

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from prompt_clip.cli import main
from prompt_clip.generate import ImageRequest
from prompt_clip.motion import ken_burns_frames, make_clip

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
network = pytest.mark.skipif(os.environ.get("OFFLINE") == "1", reason="downloads a tiny test model")


def picture(size=(320, 240)):
    img = Image.new("RGB", size, (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 40, 200, 180], fill=(200, 60, 40))
    d.rectangle([180, 120, 300, 220], fill=(40, 90, 200))
    return img


def test_frame_count_and_size():
    frames = list(ken_burns_frames(picture(), seconds=2, fps=15))
    assert len(frames) == 30 and all(f.shape == (240, 320, 3) for f in frames)


@pytest.mark.parametrize("size", [(320, 240), (240, 320), (256, 256)])
def test_no_black_corners(size):
    white = Image.new("RGB", size, (255, 255, 255))
    for f in ken_burns_frames(white, seconds=2, fps=30):
        assert f.min() > 200  # rotation never exposes the background


def test_clip_is_valid_h264(tmp_path):
    out = make_clip(picture((321, 241)), tmp_path / "c.mp4", seconds=2, fps=15)  # odd size is trimmed to even
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 240)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    cap.release()
    assert fourcc.lower() in ("avc1", "h264")


def test_prompt_enhancement():
    assert ImageRequest("a cat").full_prompt().startswith("a cat, highly detailed")
    assert ImageRequest("a highly detailed cat").full_prompt() == "a highly detailed cat"
    assert ImageRequest("a cat", enhance=False).full_prompt() == "a cat"
    with pytest.raises(ValueError):
        ImageRequest("   ").full_prompt()


@network
def test_generation_is_reproducible_with_seed():
    from prompt_clip.generate import generate_image, load_pipeline

    pipe = load_pipeline(TINY_MODEL, device="cpu", allow_pickle_weights=True)
    req = ImageRequest("a red square", steps=2, width=64, height=64, seed=7)
    a, b = generate_image(pipe, req), generate_image(pipe, req)
    assert a.size == (64, 64)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_safetensors_required_by_default():
    from prompt_clip.generate import load_pipeline

    with pytest.raises(OSError):
        load_pipeline(TINY_MODEL, device="cpu")  # the tiny test model only ships .bin weights


@network
def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "clip.mp4"
    assert main(["a red square", "--model", TINY_MODEL, "--steps", "2", "--size", "64", "--seed", "1",
                 "--seconds", "1", "--fps", "10", "--allow-pickle-weights", "-o", str(out)]) == 0
    assert out.exists() and out.with_suffix(".png").exists()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


@network
def test_ui_session_flow():
    from prompt_clip.app import build_app
    from prompt_clip.generate import load_pipeline

    demo = build_app(load_pipeline(TINY_MODEL, device="cpu", allow_pickle_weights=True), size=64, steps=2)
    fns = {getattr(d.fn, "__name__", ""): d.fn for d in demo.fns.values()}
    image, path, state, _ = fns["new_image"]("a red square", None)
    assert image.size == (64, 64) and os.path.exists(path)
    _, _, refined, _ = fns["refine"]("make it blue", state)
    assert refined["seed"] == state["seed"] and refined["prompt"].endswith("make it blue")
    clip, _ = fns["animate"](2, refined)
    assert clip.endswith(".mp4") and os.path.getsize(clip) > 0
