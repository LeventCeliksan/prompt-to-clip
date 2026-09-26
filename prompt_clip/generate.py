"""Text-to-image with any diffusers text-to-image checkpoint (SDXL by default), on CUDA, Apple MPS or CPU."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# The default checkpoint is pinned to a known commit, so a changed upstream repo cannot swap the weights silently.
PINNED_REVISIONS = {DEFAULT_MODEL: "462165984030d82259a11f4367a4eed129e94a7b"}
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, ugly, worst quality"
QUALITY_SUFFIX = ", highly detailed, sharp focus, professional photography"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model: str = DEFAULT_MODEL, device: Optional[str] = None, allow_pickle_weights: bool = False):
    """Load a text-to-image pipeline. Half precision on CUDA/MPS, full precision on CPU.

    Only safetensors weights are loaded unless `allow_pickle_weights` is set: legacy .bin checkpoints are
    pickle-based and can execute code, so only enable it for repositories you trust.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    device = device or pick_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype,
                                                     revision=PINNED_REVISIONS.get(model, "main"),
                                                     use_safetensors=None if allow_pickle_weights else True)
    pipe = pipe.to(device)
    if device != "cpu":
        pipe.enable_attention_slicing()
    return pipe


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    steps: int = 30
    guidance: float = 7.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    enhance: bool = True

    def full_prompt(self) -> str:
        p = self.prompt.strip()
        if not p:
            raise ValueError("prompt is empty")
        return p + QUALITY_SUFFIX if self.enhance and "detailed" not in p.lower() else p


def generate_image(pipe, req: ImageRequest):
    """Return a PIL image. A fixed seed makes results reproducible and lets a refined prompt keep the composition."""
    import torch

    generator = None
    if req.seed is not None:
        # CPU generators are reproducible across devices (MPS generators are not supported everywhere).
        generator = torch.Generator("cpu").manual_seed(req.seed)
    out = pipe(prompt=req.full_prompt(), negative_prompt=req.negative_prompt, num_inference_steps=req.steps,
               guidance_scale=req.guidance, width=req.width, height=req.height, generator=generator)
    return out.images[0]
