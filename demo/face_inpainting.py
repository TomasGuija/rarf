"""Interactive Gradio demo for natural-image RARF checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rarf.image_inference import (
    prepare_image,
    prepare_mask,
    rgb_image,
    sample_image,
)
from rarf.training.checkpoints import load_rarf_checkpoint

DEFAULT_EXAMPLE = PROJECT_ROOT / "assets" / "demo" / "face.png"


def _parse_args():
    parser = argparse.ArgumentParser(description="Launch the RARF face demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _drawn_mask(editor_value) -> tuple[Image.Image, Image.Image]:
    """Extract the background and union of painted editor layers."""

    if not editor_value or editor_value.get("background") is None:
        raise gr.Error("Upload an image first.")

    background = editor_value["background"].convert("RGB")
    mask = np.zeros((background.height, background.width), dtype=np.uint8)
    for layer in editor_value.get("layers", []):
        layer = layer.convert("RGBA")
        if layer.size != background.size:
            layer = layer.resize(background.size, Image.Resampling.NEAREST)
        mask = np.maximum(mask, np.asarray(layer, dtype=np.uint8)[..., 3])

    if not np.any(mask):
        raise gr.Error("Draw the region to inpaint before generating.")
    return background, Image.fromarray(mask)


class FaceDemo:
    """Keep one checkpoint loaded and serve deterministic inpainting requests."""

    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.module, selected_weights, settings = load_rarf_checkpoint(
            checkpoint,
            device=self.device,
            weights="auto",
            return_data_hparams=True,
        )
        self.resolution = int(settings["resolution"])
        print(
            f"Loaded {selected_weights} weights on {self.device} "
            f"at {self.resolution}x{self.resolution}.",
            flush=True,
        )

    def inpaint(self, editor_value, seed, sample_steps):
        """Generate the user-painted region and return an RGB image."""

        background, drawn_mask = _drawn_mask(editor_value)
        target = prepare_image(background, self.resolution)
        mask = prepare_mask(drawn_mask, self.resolution)
        image = (target * (1.0 - mask)).unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prediction = sample_image(
                self.module.model,
                self.module.flow,
                image,
                mask,
                steps=int(sample_steps),
                num_samples=1,
                strategy="normal",
                seed=int(seed),
            )
        return rgb_image(prediction)


def build_demo(checkpoint: str, device: str | None = None) -> gr.Blocks:
    """Build the face-inpainting interface around one loaded checkpoint."""

    backend = FaceDemo(checkpoint, device=device)
    example = str(DEFAULT_EXAMPLE) if DEFAULT_EXAMPLE.is_file() else None

    with gr.Blocks(title="RARF Face Inpainting") as demo:
        gr.Markdown(
            "# RARF Face Inpainting\n"
            "Upload a face or capture a selfie, paint the region to replace, "
            "and select **Generate**. Images are center-cropped to the checkpoint "
            "resolution and processed by the machine hosting this demo."
        )
        with gr.Row():
            editor = gr.ImageMask(
                value=example,
                type="pil",
                image_mode="RGBA",
                label="Image and mask",
                sources=["upload", "webcam"],
                webcam_options=gr.WebcamOptions(mirror=True),
                transforms=(),
                buttons=[],
                brush=gr.Brush(
                    default_size=24,
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed",
                ),
                layers=False,
            )
            output = gr.Image(
                type="pil",
                label="Inpainted result",
                buttons=["download"],
            )

        with gr.Accordion("Sampling settings", open=False):
            seed = gr.Number(value=12, precision=0, label="Seed")
            sample_steps = gr.Slider(
                minimum=2,
                maximum=200,
                step=1,
                value=backend.module.sample_steps,
                label="Sampling steps",
            )

        generate = gr.Button("Generate", variant="primary")
        generate.click(
            backend.inpaint,
            inputs=[editor, seed, sample_steps],
            outputs=output,
        )

    return demo.queue(default_concurrency_limit=1)


def main():
    args = _parse_args()
    demo = build_demo(args.checkpoint, device=args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
