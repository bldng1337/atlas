import csv
import json
import os
import random
import sys
from typing import List, Optional, cast
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor
from scipy import ndimage
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from skimage.morphology import dilation, disk
from dataset.feature_map import get_map_combined
from models import ControlNetDEMModel, UNetDEMConditionModel
from pipeline_terrain import TerrainDiffusionPipeline, TerrainDiffusionControlNetPipeline


def parse_args(config: dict):
    """Not perfect but sufficient for now."""
    # Parse environment variables and update config
    # for var_name in config:
    #     if var_name not in os.environ:
    #         continue
    #     var_value = os.environ[var_name]
    #     original = config[var_name]
    #     if isinstance(original, bool):
    #         var_value = var_value.lower() in ("true", "1", "yes", "y")
    #     elif isinstance(original, int):
    #         var_value = int(var_value)
    #     elif isinstance(original, float):
    #         var_value = float(var_value)
    #     config[var_name] = var_value
    #     print(f"Updated {var_name} from env: {original} => {var_value}")

    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                var_name, var_value = line.split("=", 1)
                var_name = var_name.strip()
                var_value = var_value.strip()
                if var_name in config:
                    original = config[var_name]
                    if isinstance(original, bool):
                        var_value = var_value.lower() in ("true", "1", "yes", "y")
                    elif isinstance(original, int):
                        var_value = int(var_value)
                    elif isinstance(original, float):
                        var_value = float(var_value)
                    config[var_name] = var_value
                    print(f"Updated {var_name} from .env: {original} => {var_value}")
    args = sys.argv[1:]

    # Validate that we have an even number of CLI arguments
    if len(args) % 2 != 0:
        dangling_flag = args[-1]
        raise ValueError(
            f"Dangling CLI argument '{dangling_flag}' detected. "
            "CLI arguments must be provided as key-value pairs (e.g., -learning_rate 0.001)."
        )

    for i in range(0, len(args), 2):
        if i + 1 < len(args) and args[i].startswith("-"):
            var_name = args[i].lstrip("-")
            var_value = args[i + 1]
            if var_name == "config":
                try:
                    with open(var_value, "r") as f:
                        file_config = json.load(f)
                    for k, v in file_config.items():
                        if k in config:
                            original = config[k]
                            if isinstance(original, bool):
                                v = str(v).lower() in ("true", "1", "yes", "y")
                            elif isinstance(original, int):
                                v = int(v)
                            elif isinstance(original, float):
                                v = float(v)
                            config[k] = v
                            print(
                                f"Updated {k} from config file({var_value}): {original} => {v}"
                            )
                        else:
                            print(
                                f"Warning: {k} in config file is not a recognized argument."
                            )
                except Exception as e:
                    print(f"Error loading config file {var_value}: {e}")

    for i in range(0, len(args), 2):
        if i + 1 < len(args) and args[i].startswith("-"):
            try:
                var_name = args[i].lstrip("-")
                var_value = args[i + 1]
                if var_name in config:
                    original = config[var_name]
                    if isinstance(original, bool):
                        var_value = var_value.lower() in ("true", "1", "yes", "y")
                    elif isinstance(original, int):
                        var_value = int(var_value)
                    elif isinstance(original, float):
                        var_value = float(var_value)

                    config[var_name] = var_value
                    print(f"Updated {var_name}: {original} => {var_value}")
                elif var_name != "config":
                    print(f"Warning: {var_name} is not a recognized argument.")
            except Exception as e:
                print(f"Error processing argument {args[i]}")
                raise e
        else:
            var_name = args[i].lstrip("-")
            print(f"Warning: {var_name} is not a valid argument.")



def generate_synthetic_batch(
    pipeline: TerrainDiffusionPipeline,
    tokenizer: CLIPTokenizer,
    prompts: List[str],
    batch_size: int,
    height: int,
    width: int,
    weight_dtype: torch.dtype,
    device: torch.device,
    num_inference_steps: int = 15,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
) -> dict:
    batch_size = min(batch_size, len(prompts))

    sample_size = batch_size

    if generator is None:
        generator = torch.Generator(device="cpu")

    indices = torch.randperm(len(prompts), generator=generator)[:sample_size]
    selected_prompts = [prompts[i] for i in indices.tolist()]

    with torch.no_grad():
        images, dems = pipeline(
            prompt=selected_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
        )

    dems_numpy = (
        dems.float().cpu().numpy() * 1500
    )  # Scale back to approximate original range for feature map extraction

    feature_maps = []
    for i in range(batch_size):
        dem_numpy = dems_numpy[i].mean(axis=0)
        feature_map, _ = get_map_combined(dem_numpy, dem_size=width)
        feature_map_tensor = torch.from_numpy(feature_map).permute(2, 0, 1).float()
        feature_maps.append(feature_map_tensor)

    feature_maps = torch.stack(feature_maps).to(device=device, dtype=weight_dtype)

    txts = tokenizer(
        selected_prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device=device)

    return {
        "img": images,
        "dem": dems,
        "feature_map": feature_maps,
        "txt": txts,
    }


def log_validation_controlnet(
    unet,
    controlnet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    accelerator,
    feature_map,
    weight_dtype,
    height,
    width,
    step,
    terrain_pipeline=None,
    gen_prompts=None,
    generator=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    num_random_validations=3,
    conditioning_scale=1.0,
):
    logger = get_logger(__name__)
    logger.info("Running validation...")

    unet.eval()
    controlnet.eval()
    vae.eval()
    text_encoder.eval()
    pipeline = TerrainDiffusionControlNetPipeline(
        vae=vae,
        controlnet=controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,  # ty:ignore[invalid-argument-type]
        feature_extractor=None,  # ty:ignore[invalid-argument-type]
        image_encoder=None,  # ty:ignore[invalid-argument-type]
        requires_safety_checker=False,
    )

    def run_inference(prompt, feature_map):
        with torch.no_grad():
            image, dem = pipeline(
                controlnet_cond=feature_map,
                conditioning_scale=conditioning_scale,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
            )
            dem=dem.float().cpu().numpy()
            dem = dem[0].transpose(1, 2, 0)
            image=image.float().cpu().numpy()
            image = image[0].transpose(1, 2, 0)
        return image, dem

    def create_overlay(dem, feature_map, alpha=0.4):
        feature_map_normalized = feature_map.copy()
        fmin = feature_map_normalized.min()
        fmax = feature_map_normalized.max()
        if fmax > fmin:
            feature_map_normalized = (feature_map_normalized - fmin) / (fmax - fmin)

        overlay = dem * (1 - alpha) + feature_map_normalized * alpha
        overlay = np.clip(overlay, 0, 1)

        return overlay

    with torch.no_grad():
        logger.info("Running static validation...")
        feature_map_static = feature_map[0].to(
            device=accelerator.device, dtype=weight_dtype
        )
        prompt_static = "rain forests and mountains in Philippines in November"

        image_static, dem_static = run_inference(
            prompt_static, feature_map_static
        )

        feature_map_static_np = feature_map_static.cpu().float().numpy()
        feature_map_static_np = feature_map_static_np.transpose(1, 2, 0)
        overlay_static = create_overlay(dem_static, feature_map_static_np, alpha=0.4)

        if accelerator.is_main_process:
            tracker = accelerator.get_tracker("tensorboard")
            if tracker:
                tracker.writer.add_images(
                    "validation/static/img", image_static, step, dataformats="HWC"
                )
                tracker.writer.add_images(
                    "validation/static/overlay",
                    overlay_static,
                    step,
                    dataformats="HWC",
                )

    controlnet.train()


def cloud_percent_in_batch(batch: dict) -> float:
    if "cloud_mask" not in batch:
        return 0.0
    cloud_mask = cast(Tensor, batch["cloud_mask"])
    total_pixels = cloud_mask.numel()
    cloud_pixels = cloud_mask.sum().item()
    percent_cloud = cloud_pixels / total_pixels if total_pixels > 0 else 0.0
    return 1-percent_cloud

def train_controlnet(
    latents: Tensor,
    unet: UNetDEMConditionModel,
    controlnet: ControlNetDEMModel,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDIMScheduler,
    batch: dict,
    device: torch.device,
    empty_prompt_embeds: Optional[Tensor] = None,
    proportion_empty_prompts=0.1,
    conditioning_scale=1.0,
    weight_dtype=torch.float32,
    fixed_t: Optional[torch.Tensor] = None,
    fixed_noise: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[Tensor] = None,
    fixed_noisy_latents: Optional[Tensor] = None,
    fixed_target: Optional[Tensor] = None,
    return_diagnostics=False,
):
    bsz = latents.shape[0]

    if fixed_t is not None:
        timesteps = fixed_t
    else:
        timesteps = torch.randint(
            0,
            noise_scheduler.config["num_train_timesteps"],
            (bsz,),
            device=device,
        )

    if fixed_noise is not None:
        noise = fixed_noise
    else:
        noise = torch.randn_like(latents)

    if fixed_noisy_latents is not None:
        assert fixed_noise is not None, "fixed_noisy_latents provided without fixed_noise"
        assert fixed_t is not None, "fixed_noisy_latents provided without fixed_timestep"
        noisy_latents = fixed_noisy_latents
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    if prompt_embeds is None:
        with torch.no_grad():
            text_inputs = batch["txt"]
            text_input_ids = text_inputs.to(device)
            prompt_embeds = text_encoder(
                text_input_ids,
            )[0]

    if proportion_empty_prompts > 0 and empty_prompt_embeds is not None:
        random_mask = (
            torch.rand(bsz, device=prompt_embeds.device) < proportion_empty_prompts
        )
        for i in range(bsz):
            if random_mask[i]:
                prompt_embeds[i] = empty_prompt_embeds[0]

    prompt_embeds = prompt_embeds

    controlnet_cond = batch["feature_map"].to(device=device, dtype=torch.float32)

    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents.to(dtype=torch.float32),
        timesteps.to(dtype=torch.float32),
        encoder_hidden_states=prompt_embeds.to(dtype=torch.float32),
        controlnet_cond=controlnet_cond,
        conditioning_scale=conditioning_scale,
    )

    diag = None
    if return_diagnostics:
        diag = {
            "cn_down_norms": [s.detach().norm().item() for s in down_block_res_samples],
            "cn_mid_norm": mid_block_res_sample.detach().norm().item(),
            "timesteps": timesteps.detach().cpu().tolist(),
        }

    model_pred = unet(
        noisy_latents.to(dtype=weight_dtype),
        timesteps,
        encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
        down_block_additional_residuals=[
            s.to(dtype=weight_dtype) for s in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
    ).sample

    if fixed_target is not None:
        target = fixed_target
    elif noise_scheduler.config["prediction_type"] == "epsilon":
        target = noise
    elif noise_scheduler.config["prediction_type"] == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config['prediction_type']}"
        )

    if "cloud_mask" in batch:
        cloud_mask = batch["cloud_mask"].to(device, dtype=weight_dtype)

        cloud_mask_latent = F.interpolate(
            cloud_mask,
            size=(model_pred.shape[2], model_pred.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        loss_weight = (1.0 - cloud_mask_latent).expand_as(model_pred)

        mse = (model_pred.float() - target.float()) ** 2
        weighted_mse = mse * loss_weight

        loss = weighted_mse.float().sum() / (loss_weight.float().sum() + 1e-8)
        if return_diagnostics:
            return loss, diag
        return loss

    mse = (model_pred.float() - target.float()) ** 2

    loss = mse.mean()
    if return_diagnostics:
        return loss, diag
    return loss

def use_canny_feature(batch: dict) -> dict:
    if "feature_map" not in batch:
        return batch

    feature_map = cast(Tensor, batch["dem"])
    B = feature_map.shape[0]
    results = []
    shape = feature_map.shape
    np_imgs = (feature_map.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    for i in range(B):
        gray = cv2.cvtColor(np_imgs[i], cv2.COLOR_RGB2GRAY)
        low=np.percentile(gray, 30)
        high=np.percentile(gray, 70)
        edges = cv2.Canny(gray, low, high)
        edges=dilation(edges, disk(6))
        results.append(torch.from_numpy(np.stack([edges, edges, edges])).float() / 255.0)
    batch["feature_map"] = torch.stack(results).to(feature_map.device)
    assert batch["feature_map"].shape == shape, f"Expected shape {shape}, got {batch['feature_map'].shape}"
    return batch

def augment_batch(
    batch: dict,
    enable_random_crop: bool = False,
    crop_scale: float = 0.8,
    enable_random_flip: bool = False,
    flip_horizontal_prob: float = 0.5,
    flip_vertical_prob: float = 0.5,
    enable_channel_drop: bool = False,
    channel_drop_prob: float = 0.5,
    enable_feature_dropout: bool = False,
    feature_dropout_prob: float = 0.1,
):
    if "feature_map" not in batch or "img" not in batch or "dem" not in batch:
        return batch

    feature_map = cast(Tensor, batch["feature_map"])
    bsz, num_channels, h, w = feature_map.shape
    device = feature_map.device

    img = cast(Tensor, batch["img"])
    dem = cast(Tensor, batch["dem"])
    cloud_mask = cast(Tensor, batch["cloud_mask"]) if "cloud_mask" in batch else None

    if enable_random_crop:
        crop_size = int(min(h, w) * crop_scale)
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        feature_map = feature_map[:, :, top : top + crop_size, left : left + crop_size]
        feature_map = F.interpolate(feature_map, size=(h, w), mode="nearest")

        img = img[:, :, top : top + crop_size, left : left + crop_size]
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)

        dem = dem[:, :, top : top + crop_size, left : left + crop_size]
        dem = F.interpolate(dem, size=(h, w), mode="bilinear", align_corners=False)

        if cloud_mask is not None:
            cloud_mask = cloud_mask[
                :, :, top : top + crop_size, left : left + crop_size
            ]
            cloud_mask = F.interpolate(cloud_mask, size=(h, w), mode="nearest")

    if enable_random_flip:
        if random.random() < flip_horizontal_prob:
            feature_map = torch.flip(feature_map, dims=[-1])
            img = torch.flip(img, dims=[-1])
            dem = torch.flip(dem, dims=[-1])
            if cloud_mask is not None:
                cloud_mask = torch.flip(cloud_mask, dims=[-1])

        if random.random() < flip_vertical_prob:
            feature_map = torch.flip(feature_map, dims=[-2])
            img = torch.flip(img, dims=[-2])
            dem = torch.flip(dem, dims=[-2])
            if cloud_mask is not None:
                cloud_mask = torch.flip(cloud_mask, dims=[-2])

    if enable_channel_drop:
        drop_mask = torch.rand((bsz, num_channels), device=device) < channel_drop_prob
        feature_map[drop_mask.unsqueeze(-1).unsqueeze(-1).expand_as(feature_map)] = 0.0

    # pretty sure this could be done in a fancier way over all batches at once
    if enable_feature_dropout:
        dropout_mask = torch.ones_like(feature_map, device=device)

        for b in range(bsz):
            if random.random() < feature_dropout_prob:
                num_patches = random.randint(1, 3)
                for _ in range(num_patches):
                    patch_h = random.randint(int(h * 0.2), int(h * 0.55))
                    patch_w = random.randint(int(w * 0.2), int(w * 0.55))

                    top = random.randint(0, h - patch_h)
                    left = random.randint(0, w - patch_w)

                    dropout_mask[b, :, top : top + patch_h, left : left + patch_w] = 0.0
        feature_map = feature_map * dropout_mask

    batch["feature_map"] = feature_map
    batch["img"] = img
    batch["dem"] = dem
    if cloud_mask is not None:
        batch["cloud_mask"] = cloud_mask

    return batch
