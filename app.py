"""
Gradio UI for terrain generation with MESA + ControlNet.
Note this is a slight modification of https://huggingface.co/spaces/mikonvergence/MESA.
Usage:
    python app.py [OPTIONS]

Options:
    --mesa-path         HuggingFace repo or local path for base MESA model
                        (default: NewtNewt/MESA)
    --controlnet-path   Local path to a converted ControlNet checkpoint
                        (default: None — base text-to-terrain only)
    --device            Torch device (default: cuda)
    --mixed-precision   One of fp16, bf16, or fp32 (default: fp16)
    --share             Launch Gradio with a public share link
    --server-port       Port for the Gradio server (default: 7860)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import gradio as gr
import numpy as np
import torch

# ── Ensure project root is importable ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Workaround for gradio_client 0.15.x bug ───────────────────────────────
#    _json_schema_to_python_type chokes when JSON schema 'additionalProperties'
#    is a bare boolean instead of a dict.  This monkey-patch handles that.
import gradio_client.utils as _gu

_orig_json_schema = _gu._json_schema_to_python_type


def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return "bool"
    return _orig_json_schema(schema, defs)


_gu._json_schema_to_python_type = _patched_json_schema

# ── Globals populated by load_models() ─────────────────────────────────────
PIPE = None
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16
HEIGHT = 512
WIDTH = 512

# ── Default prompt prefix (same convention as base MESA) ───────────────────
DEFAULT_PREFIX = "A Sentinel-2 image of "

# ── Example prompts ────────────────────────────────────────────────────────
EXAMPLE_PROMPTS = [
    "snow-capped mountains and alpine valleys in Switzerland in July",
    "rain forests and volcanoes in Philippines in November",
    "arid desert canyons and mesas in Utah in August",
    "fjords and glaciers in Norway in March",
    "subpolar forests and mountains in Chile in January",
    "terraced rice paddies in mountainous terrain in Vietnam in September",
]


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════


def load_models(
    mesa_path: str,
    controlnet_path: str | None,
    device: str,
    mixed_precision: str,
):
    """Load pipeline components and build the appropriate pipeline."""
    global PIPE, DEVICE, WEIGHT_DTYPE, HEIGHT, WIDTH

    DEVICE = device
    if mixed_precision == "fp16":
        WEIGHT_DTYPE = torch.float16
    elif mixed_precision == "bf16":
        WEIGHT_DTYPE = torch.bfloat16
    else:
        WEIGHT_DTYPE = torch.float32

    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    from models import ControlNetDEMModel, UNetDEMConditionModel
    from pipeline_terrain import (
        TerrainDiffusionControlNetPipeline,
        TerrainDiffusionPipeline,
    )

    print(f"[app] Loading base models from {mesa_path} …")
    t0 = time.time()

    noise_scheduler = DDIMScheduler.from_pretrained(mesa_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(mesa_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(mesa_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(mesa_path, subfolder="vae")
    unet = UNetDEMConditionModel.from_pretrained(mesa_path, subfolder="unet")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    HEIGHT = unet.config.sample_size * vae_scale_factor
    WIDTH = unet.config.sample_size * vae_scale_factor
    print(f"[app] Native resolution: {HEIGHT}x{WIDTH}")

    text_encoder.to(device, dtype=WEIGHT_DTYPE).eval().requires_grad_(False)
    vae.to(device, dtype=WEIGHT_DTYPE).eval().requires_grad_(False)
    unet.to(device, dtype=WEIGHT_DTYPE).eval().requires_grad_(False)

    if controlnet_path:
        print(f"[app] Loading ControlNet from {controlnet_path} …")
        controlnet = ControlNetDEMModel.from_pretrained(controlnet_path)
        controlnet.to(device, dtype=WEIGHT_DTYPE).eval().requires_grad_(False)

        PIPE = TerrainDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            controlnet=controlnet,
            requires_safety_checker=False,
        )
        print("[app] ControlNet pipeline ready.")
    else:
        PIPE = TerrainDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
        )
        print("[app] Base text-to-terrain pipeline ready (no ControlNet).")

    PIPE = PIPE.to(device)
    print(f"[app] Models loaded in {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# Control map preprocessing
# ═══════════════════════════════════════════════════════════════════════════


def preprocess_control_map(control_map_value: dict | np.ndarray | None):
    """Convert a Sketchpad value to a tensor for ControlNet conditioning.

    The Sketchpad returns a dict with keys ``'background'``, ``'layers'``,
    and ``'composite'``.  We use ``'composite'`` (or ``'background'`` as a
    fallback) and resize it to the model's native resolution.
    """
    if control_map_value is None:
        return None

    from PIL import Image as PILImage

    # ── Extract the numpy array from the Sketchpad dict ────────────────
    arr = None

    if isinstance(control_map_value, dict):
        # gr.Sketchpad / ImageEditor returns
        #   {"background": ..., "layers": [...], "composite": ...}
        composite = control_map_value.get("composite")
        if composite is not None and not _is_blank(composite):
            arr = composite
        else:
            # Fall back to the first layer (the actual drawing)
            layers = control_map_value.get("layers", [])
            if layers:
                arr = layers[0]
    elif isinstance(control_map_value, np.ndarray):
        arr = control_map_value

    if arr is None:
        return None

    if isinstance(arr, np.ndarray) and _is_blank(arr):
        return None

    # ── Convert RGBA → RGB (transparent → black) ───────────────────────
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            rgb = arr[:, :, :3].copy()
            rgb[alpha < 128] = 0  # transparent pixels → black
            arr = rgb
        if arr.ndim == 3 and arr.shape[2] == 3:
            pass  # already RGB
        else:
            return None
    else:
        return None

    # ── Center-crop to model aspect ratio, then resize ────────────────
    img = PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")
    canvas_h, canvas_w = img.size[1], img.size[0]  # PIL: (W, H)
    target_ratio = WIDTH / HEIGHT
    canvas_ratio = canvas_w / canvas_h

    if abs(canvas_ratio - target_ratio) > 0.01:
        # Crop the longer axis so the aspect ratio matches the model
        if canvas_ratio > target_ratio:
            # Canvas is wider than target – crop sides
            new_w = int(canvas_h * target_ratio)
            left = (canvas_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, canvas_h))
        else:
            # Canvas is taller than target – crop top/bottom
            new_h = int(canvas_w / target_ratio)
            top = (canvas_h - new_h) // 2
            img = img.crop((0, top, canvas_w, top + new_h))

    img = img.resize((WIDTH, HEIGHT), PILImage.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    tensor = torch.from_numpy(arr).to(DEVICE, dtype=WEIGHT_DTYPE)
    return tensor


def _is_blank(arr: np.ndarray, threshold: float = 0.02) -> bool:
    """Return True if the array is essentially uniform (blank canvas)."""
    if arr.size == 0:
        return True
    return float(arr.std()) < (255 * threshold)


def _overlay_control_map(
    target: np.ndarray,
    control_map_value: dict | np.ndarray | None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Semi-transparently overlay the control-map drawing onto *target*.

    *target* must be a uint8 image of shape ``(H, W, 3)``.
    Returns a uint8 image of the same shape with the drawing blended on top.
    """
    if control_map_value is None:
        return target

    from PIL import Image as PILImage

    # ── Extract the composite drawing ───────────────────────────────────
    composite = None
    if isinstance(control_map_value, dict):
        composite = control_map_value.get("composite")
    elif isinstance(control_map_value, np.ndarray):
        composite = control_map_value

    if composite is None or not isinstance(composite, np.ndarray):
        return target
    if composite.ndim != 3 or composite.shape[2] not in (3, 4):
        return target

    # ── Build a binary mask of drawn pixels ─────────────────────────────
    if composite.shape[2] == 4:
        # RGBA: alpha channel directly tells us what is drawn
        mask = (composite[:, :, 3] > 30).astype(np.float32)
        overlay_rgb = composite[:, :, :3]
    else:
        # RGB: detect non-white pixels (white = blank canvas background)
        diff = np.abs(composite.astype(np.float32) - 255.0).sum(axis=-1)
        mask = (diff > 30).astype(np.float32)
        overlay_rgb = composite

    # ── Resize to match target dimensions ───────────────────────────────
    h, w = target.shape[:2]
    overlay_img = PILImage.fromarray(overlay_rgb.astype(np.uint8)).convert("RGB")
    overlay_img = overlay_img.resize((w, h), PILImage.BILINEAR)
    overlay = np.array(overlay_img).astype(np.float32)

    mask_img = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((w, h), PILImage.BILINEAR)
    mask = np.array(mask_img).astype(np.float32) / 255.0

    # ── Alpha-blend ──────────────────────────────────────────────────────
    mask = mask[..., np.newaxis]  # (H, W, 1)
    target_f = target.astype(np.float32)
    result = target_f * (1.0 - alpha * mask) + overlay * (alpha * mask)
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Generation helpers
# ═══════════════════════════════════════════════════════════════════════════


def _generate(
    prompt,
    control_map,
    num_inference_steps,
    guidance_scale,
    seed,
    random_seed,
    conditioning_scale,
    prefix,
):
    """Run the pipeline and return (rgb_uint8, elevation_gray)."""
    from pipeline_terrain import TerrainDiffusionControlNetPipeline

    full_prompt = (prefix + prompt).strip()

    if random_seed:
        generator = torch.Generator(DEVICE)
    else:
        generator = torch.Generator(DEVICE).manual_seed(int(seed))

    common_kwargs = dict(
        prompt=full_prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        output_type="np",
    )

    # Preprocess control map if provided and ControlNet is available
    control_tensor = preprocess_control_map(control_map)
    if control_tensor is not None:
        if isinstance(PIPE, TerrainDiffusionControlNetPipeline):
            common_kwargs["controlnet_cond"] = control_tensor
            common_kwargs["conditioning_scale"] = float(conditioning_scale)
        else:
            print(
                "[app] Warning: Control map provided but no ControlNet loaded. "
                "Ignoring control map."
            )

    with torch.no_grad():
        image, dem = PIPE(**common_kwargs)

    # image / dem come back as lists of numpy arrays in [0, 1] range
    rgb = (255 * image[0]).astype(np.uint8)
    elevation = dem[0].mean(axis=-1)  # collapse channels → grayscale
    return rgb, elevation


def generate_2d(
    prompt,
    control_map,
    num_inference_steps,
    guidance_scale,
    seed,
    random_seed,
    conditioning_scale,
    prefix,
    overlay_control,
):
    """Generate 2D views only (fast)."""
    rgb, elevation = _generate(
        prompt,
        control_map,
        num_inference_steps,
        guidance_scale,
        seed,
        random_seed,
        conditioning_scale,
        prefix,
    )

    if overlay_control:
        # Convert grayscale elevation to RGB uint8 so we can overlay on it
        elev_vis = (np.stack([elevation] * 3, axis=-1).clip(0, 1) * 255).astype(
            np.uint8
        )
        elev_vis = _overlay_control_map(elev_vis, control_map, alpha=0.5)
        return rgb, elev_vis

    return rgb, elevation


def generate_3d(
    prompt,
    control_map,
    num_inference_steps,
    guidance_scale,
    seed,
    random_seed,
    conditioning_scale,
    crop_size,
    vertex_count,
    prefix,
    overlay_control,
):
    """Generate 2D views + 3D mesh (slow)."""
    from scipy.spatial import Delaunay

    try:
        import trimesh
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "3D export requires `trimesh` and `scikit-learn`. "
            "Install them with:  pip install trimesh scikit-learn"
        ) from exc

    rgb, elevation = _generate(
        prompt,
        control_map,
        num_inference_steps,
        guidance_scale,
        seed,
        random_seed,
        conditioning_scale,
        prefix,
    )

    if overlay_control:
        rgb = _overlay_control_map(rgb, control_map, alpha=0.5)

    # Optional center crop
    if crop_size and crop_size < min(rgb.shape[0], rgb.shape[1]):
        h, w = rgb.shape[:2]
        sh = (h - crop_size) // 2
        sw = (w - crop_size) // 2
        rgb = rgb[sh : sh + crop_size, sw : sw + crop_size, :]
        elevation = elevation[sh : sh + crop_size, sw : sw + crop_size]

    # Build 3D mesh
    rows, cols = elevation.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    pts_2d = np.stack([x.flatten(), y.flatten()], axis=-1)
    elev_flat = elevation.flatten()
    pts_3d = np.column_stack([pts_2d, 500 * elev_flat])
    colors_flat = rgb.reshape(-1, 3)

    n_clusters = int(vertex_count) if int(vertex_count) > 0 else 0

    if n_clusters > 0 and n_clusters < len(elev_flat):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(pts_3d)
        verts = kmeans.cluster_centers_
        labels = kmeans.labels_
        vert_colors = np.zeros((n_clusters, 3), dtype=np.uint8)
        for lbl in range(n_clusters):
            mask = labels == lbl
            if mask.any():
                vert_colors[lbl] = colors_flat[mask].mean(axis=0)
        pts2d_verts = verts[:, :2]
    else:
        verts = pts_3d
        vert_colors = colors_flat
        pts2d_verts = pts_2d

    tri = Delaunay(pts2d_verts)
    faces = tri.simplices
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vert_colors)

    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(tmp.name)
    tmp.close()

    return rgb, elevation, tmp.name


# ═══════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="stone",
        font=[
            gr.themes.GoogleFont("Source Sans 3", weights=(400, 600)),
            "arial",
        ],
    )

    with gr.Blocks(theme=theme) as demo:
        # ── Header ───────────────────────────────────────────────────────
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🗺️ Atlas: Terrain Generation with MESA + ControlNet")
            gr.Markdown(
                "### Text-driven generation of co-registered optical & elevation maps"
            )
            gr.Markdown(
                "[[MESA Model](https://huggingface.co/NewtNewt/MESA)] "
                "[[Major-TOM Dataset](https://huggingface.co/datasets/Major-TOM/Core-DEM)]"
            )

        with gr.Column(elem_classes="abstract"):
            gr.Markdown(
                "Atlas extends **MESA** — a latent diffusion model for terrain generation — "
                "with a ControlNet that adds spatial conditioning from ridge/valley feature maps. "
                "Enter a terrain description and the model produces **RGB** and **elevation** outputs."
            )
            gr.Markdown(
                "> ⚠️ **Tip:** MESA works best on complex, mountainous terrain. "
                "Flat landscapes may produce less convincing results."
            )

        # ── Prompt input ─────────────────────────────────────────────────
        prompt_input = gr.Textbox(
            lines=2,
            placeholder="Describe a terrain…",
            label="Terrain Prompt",
            value="snow-capped mountains and alpine valleys in Switzerland in July",
        )

        # ── Control map canvas ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                control_map = gr.Sketchpad(
                    type="numpy",
                    image_mode="RGB",
                    label="Control Map (optional — draw to condition terrain)",
                    brush=gr.Brush(
                        colors=[
                            "#ff0000",
                            "#00ff00",
                            "#0000ff",
                            "#ffff00",
                            "#ff00ff",
                            "#00ffff",
                            "#ffffff",
                            "#000000",
                        ],
                        default_color="#ff0000",
                        default_size=5,
                    ),
                    height=400,
                    width=400,
                )
            with gr.Column(scale=1):
                conditioning_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="ControlNet Conditioning Scale",
                    info="0 = ignore control map, higher = stronger conditioning",
                )
                gr.Markdown(
                    "### Control Map Guide\n"
                    "Draw on the canvas to guide terrain features:\n\n"
                    "- **Red** — valleys\n"
                    "- **Green** — ridges\n"
                    "- **Blue** — cliffs\n"
                    "Leave the canvas blank for text-only generation.\n\n"
                    "*A ControlNet checkpoint must be loaded for the control "
                    "map to take effect (pass `--controlnet-path` at launch).*"
                )

        # ── Output tabs ──────────────────────────────────────────────────
        with gr.Tabs() as output_tabs:
            with gr.Tab("2D View (Fast)"):
                generate_2d_btn = gr.Button("Generate Terrain", variant="primary")
                with gr.Row():
                    rgb_output = gr.Image(label="RGB Image")
                    elevation_output = gr.Image(label="Elevation Map")

            with gr.Tab("3D View (Slow)"):
                generate_3d_btn = gr.Button("Generate Terrain", variant="primary")
                model_3d_output = gr.Model3D(
                    label="3D Terrain Mesh",
                    camera_position=[90, 135, 512],
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )

        # ── Advanced options ─────────────────────────────────────────────
        with gr.Accordion("Advanced Options", open=False):
            num_inference_steps = gr.Slider(
                minimum=10,
                maximum=200,
                step=5,
                value=50,
                label="Inference Steps",
            )
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.5,
                value=7.5,
                label="Guidance Scale",
            )
            seed_number = gr.Number(value=42, label="Seed")
            random_seed = gr.Checkbox(value=True, label="Random Seed")
            crop_size = gr.Slider(
                minimum=128,
                maximum=768,
                step=64,
                value=512,
                label="(3D Only) Crop Size",
            )
            vertex_count = gr.Slider(
                minimum=0,
                maximum=10000,
                step=500,
                value=0,
                label="(3D Only) Vertex Count (0 = full mesh)",
            )
            prefix_textbox = gr.Textbox(
                label="Prompt Prefix",
                value=DEFAULT_PREFIX,
            )
            overlay_control_checkbox = gr.Checkbox(
                value=False,
                label="Overlay Control Map on Output",
                info=(
                    "2D view: overlay on elevation map. "
                    "3D view: overlay on satellite image."
                ),
            )

        # ── Example prompts ──────────────────────────────────────────────
        with gr.Accordion("Example Prompts", open=False):
            gr.Examples(
                examples=[[p] for p in EXAMPLE_PROMPTS],
                inputs=[prompt_input],
            )

        # ── Wire buttons ─────────────────────────────────────────────────
        generate_2d_btn.click(
            fn=generate_2d,
            inputs=[
                prompt_input,
                control_map,
                num_inference_steps,
                guidance_scale,
                seed_number,
                random_seed,
                conditioning_scale_slider,
                prefix_textbox,
                overlay_control_checkbox,
            ],
            outputs=[rgb_output, elevation_output],
        )

        generate_3d_btn.click(
            fn=generate_3d,
            inputs=[
                prompt_input,
                control_map,
                num_inference_steps,
                guidance_scale,
                seed_number,
                random_seed,
                conditioning_scale_slider,
                crop_size,
                vertex_count,
                prefix_textbox,
                overlay_control_checkbox,
            ],
            outputs=[rgb_output, elevation_output, model_3d_output],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Atlas Gradio UI for terrain generation"
    )
    parser.add_argument(
        "--mesa-path",
        type=str,
        default="NewtNewt/MESA",
        help="HuggingFace repo or local path for the base MESA model.",
    )
    parser.add_argument(
        "--controlnet-path",
        type=str,
        default=None,
        help=(
            "Local path to a converted ControlNet checkpoint. "
            "When omitted the base text-to-terrain pipeline is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Gradio server port.",
    )
    args = parser.parse_args()

    load_models(args.mesa_path, args.controlnet_path, args.device, args.mixed_precision)

    demo = build_ui()
    demo.queue().launch(
        share=args.share,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
