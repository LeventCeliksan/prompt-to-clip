"""Gradio UI. Each browser session keeps its own last prompt and seed, so refinements never leak between users."""
from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import gradio as gr

from .generate import DEFAULT_MODEL, ImageRequest, generate_image, load_pipeline
from .motion import make_clip

OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="prompt_to_clip_"))


def build_app(pipe, size: int = 1024, steps: int = 30) -> gr.Blocks:
    def _render(prompt: str, seed: int):
        image = generate_image(pipe, ImageRequest(prompt, steps=steps, width=size, height=size, seed=seed))
        path = OUTPUT_DIR / f"image_{seed}_{random.randrange(1 << 30)}.png"
        image.save(path)
        return image, str(path)

    def new_image(prompt, state):
        if not prompt.strip():
            raise gr.Error("Write a prompt first.")
        seed = random.randrange(1 << 31)
        image, path = _render(prompt, seed)
        return image, path, {"prompt": prompt.strip(), "seed": seed, "image": path}, f"seed {seed}"

    def refine(instruction, state):
        if not state or not instruction.strip():
            raise gr.Error("Create an image first, then describe the change.")
        prompt = f"{state['prompt']}. {instruction.strip()}"
        image, path = _render(prompt, state["seed"])  # same seed keeps the overall composition
        return image, path, {"prompt": prompt, "seed": state["seed"], "image": path}, f"seed {state['seed']} (refined)"

    def animate(seconds, state):
        if not state:
            raise gr.Error("Create an image first.")
        from PIL import Image

        clip = OUTPUT_DIR / (Path(state["image"]).stem + ".mp4")
        make_clip(Image.open(state["image"]), clip, seconds=seconds)
        return str(clip), str(clip)

    with gr.Blocks(title="Prompt to Clip") as demo:
        gr.Markdown("# Prompt to Clip\nGenerate an image from text, refine it, then animate it into a short clip.")
        state = gr.State(None)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a lighthouse on a cliff at sunset, cinematic")
            instruction = gr.Textbox(label="Refinement", lines=3, placeholder="make it a stormy night")
        with gr.Row():
            new_btn = gr.Button("Generate image", variant="primary")
            refine_btn = gr.Button("Refine image")
            seconds = gr.Slider(2, 8, value=3, step=1, label="Clip length (s)")
            clip_btn = gr.Button("Animate to clip", variant="primary")
        info = gr.Markdown()
        with gr.Row():
            image = gr.Image(label="Image", type="pil")
            video = gr.Video(label="Clip")
        with gr.Row():
            image_file = gr.File(label="Download PNG")
            video_file = gr.File(label="Download MP4")
        new_btn.click(new_image, [prompt, state], [image, image_file, state, info])
        refine_btn.click(refine, [instruction, state], [image, image_file, state, info])
        clip_btn.click(animate, [seconds, state], [video, video_file])
    return demo


def main(argv=None):
    p = argparse.ArgumentParser(prog="prompt-to-clip-ui")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args(argv)
    build_app(load_pipeline(a.model), a.size, a.steps).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
