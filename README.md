# prompt-to-clip

Type a prompt, get an image from **SDXL** (or any diffusers text-to-image model), refine it with follow-up instructions while keeping the composition, then turn it into a short **animated MP4** with a slow zoom, sway and drift.

To be clear about what it is: this is not a video diffusion model. It generates one still image and animates it with camera motion (a Ken Burns effect). That is enough for social posts, thumbnails-in-motion and ad mockups, and it runs on a laptop.

## Features
- **SDXL by default**, any `diffusers` text-to-image checkpoint via `--model`
- **Runs on CUDA, Apple Silicon (MPS) or CPU**, half precision on GPU
- **Refine with a fixed seed**: the refinement is appended to the prompt and re-rendered with the same seed, so the overall layout stays similar
- **No black corners**: the zoom is computed so rotation and drift never expose the background
- **Browser-playable output**: H.264 / yuv420p MP4 via bundled ffmpeg
- **Per-session state** in the web UI, so parallel users never overwrite each other's images
- **Safe loading**: only `safetensors` weights by default; pickle-based `.bin` checkpoints need `--allow-pickle-weights`

## Install
```bash
pip install git+https://github.com/LeventCeliksan/prompt-to-clip
```
The first run downloads the SDXL base weights from Hugging Face (about 7 GB).

## Usage
```bash
prompt-to-clip "a lighthouse on a cliff at sunset, cinematic" --seed 42 -o lighthouse.mp4
# image: lighthouse.png
# clip:  lighthouse.mp4

prompt-to-clip-ui          # web UI on http://127.0.0.1:7860
```
Options: `--model`, `--steps` (30), `--size` (1024), `--seed`, `--seconds` (3), `--fps` (30), `--no-enhance`.

```python
from prompt_clip import ImageRequest, generate_image, load_pipeline, make_clip
pipe = load_pipeline()
image = generate_image(pipe, ImageRequest("a red vintage car on a coastal road", seed=7))
make_clip(image, "car.mp4", seconds=4)
```

## Tests
```bash
pip install -e ".[test]"
pytest
```
The suite checks frame count and size, that no frame ever shows black corners (three aspect ratios), that the MP4 is valid H.264, prompt handling, and the safetensors-only default. End-to-end tests (seeded reproducibility, CLI, and the web UI flow of generate, refine and animate) run on `hf-internal-testing/tiny-stable-diffusion-xl-pipe`, a few-MB SDXL-architecture model, so they finish in seconds; set `OFFLINE=1` to skip them.

## License
MIT. Model weights are licensed separately by their authors (SDXL: CreativeML Open RAIL++-M).
