import time
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from train_utils import generate_synthetic_batch, parse_args
import functools
from pipeline_terrain import TerrainDiffusionPipeline
from datasets import load_dataset

from dataprep import preprocess
from torch.utils.data import DataLoader

mesa_path = "./weights"
mode = "both"
device = "cuda"
mixed_precision = "bf16"
dataset_path = None
prompts_csv_path = "./prompts.csv"

parse_args(globals())

weight_dtype = torch.float32
if mixed_precision == "fp16":
    weight_dtype = torch.float16
elif mixed_precision == "bf16":
    weight_dtype = torch.bfloat16


def check(condition: bool, msg: str):
    if not condition:
        raise AssertionError(f"CHECK FAILED: {msg}")


def info(msg: str):
    print(f"\033[94m[INFO]\033[0m {msg}")


def warn(msg: str):
    print(f"\033[93m[WARN]\033[0m  {msg}")


def section(title: str):
    print(f"\n\n\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


section("Model Loading")

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models import ControlNetDEMModel, UNetDEMConditionModel

t0 = time.time()
info(f"Loading from: {mesa_path}")

noise_scheduler = DDIMScheduler.from_pretrained(mesa_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(mesa_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(mesa_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(mesa_path, subfolder="vae")
unet = UNetDEMConditionModel.from_pretrained(mesa_path, subfolder="unet")

info(f"Models loaded in {time.time() - t0:.1f}s")

vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
height = unet.config["sample_size"] * vae_scale_factor
width = unet.config["sample_size"] * vae_scale_factor
info(f"Training resolution: {height}x{width}")
info(f"VAE scale factor: {vae_scale_factor}")
info(f"Latent size: {height // vae_scale_factor}x{width // vae_scale_factor}")

controlnet = ControlNetDEMModel.from_unet(
    unet,
    conditioning_channels=3,
    load_weights_from_unet=True,
)
info(
    f"ControlNet parameters: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M"
)
info(f"UNet parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

vae.to(device, dtype=weight_dtype).eval()
unet.to(device, dtype=weight_dtype).eval()
text_encoder.to(device, dtype=weight_dtype).eval()
controlnet.to(device, dtype=torch.float32).train()

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
controlnet.requires_grad_(True)

section("Forward Pass Shape Verification")

with torch.no_grad():
    bsz = 1
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    dummy_latents = torch.randn(
        bsz, 8, latent_h, latent_w, device=device, dtype=weight_dtype
    )
    dummy_t = torch.randint(0, 1000, (bsz,), device=device)
    dummy_feature_map = torch.randn(
        bsz, 3, height, width, device=device, dtype=weight_dtype
    )

    text_input = tokenizer(
        "test prompt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    dummy_embeds = text_encoder(text_input.input_ids.to(device))[0].to(weight_dtype)
    info(f"Prompt embeds shape: {dummy_embeds.shape}")

    t_cn_start = time.time()
    down_block_res_samples, mid_block_res_sample = controlnet(
        dummy_latents.to(torch.float32),
        dummy_t,
        encoder_hidden_states=dummy_embeds.to(torch.float32),
        controlnet_cond=dummy_feature_map.to(torch.float32),
        conditioning_scale=1.0,
    )
    t_cn = time.time() - t_cn_start
    info(f"ControlNet forward pass: {t_cn * 1000:.0f}ms")
    info(f"  Down block residuals: {len(down_block_res_samples)} tensors")
    for i, s in enumerate(down_block_res_samples):
        info(f"    [{i}] {tuple(s.shape)}")
    info(f"  Mid block residual: {tuple(mid_block_res_sample.shape)}")
    check(len(down_block_res_samples) > 0, "ControlNet produced no down-block residuals")
    check(mid_block_res_sample is not None, "ControlNet produced no mid-block residual")

    t_unet_start = time.time()
    unet_out = unet(
        dummy_latents,
        dummy_t,
        encoder_hidden_states=dummy_embeds,
        down_block_additional_residuals=[
            s.to(weight_dtype) for s in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(weight_dtype),
    ).sample
    t_unet = time.time() - t_unet_start
    info(f"UNet forward pass: {t_unet * 1000:.0f}ms")
    info(f"  UNet output shape: {tuple(unet_out.shape)}")
    check(
        unet_out.shape == (bsz, 8, latent_h, latent_w),
        f"UNet expected output shape ({bsz},8,{latent_h},{latent_w}), got {tuple(unet_out.shape)}",
    )

section("Loss at Initialization")

prediction_type=noise_scheduler.config["prediction_type"]

with torch.no_grad():
    latents = torch.randn(
        bsz, 8, latent_h, latent_w, device=device, dtype=torch.float32
    )
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=device
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    feature_map = torch.randn(bsz, 3, height, width, device=device, dtype=torch.float32)

    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents.to(torch.float32),
        timesteps,
        encoder_hidden_states=dummy_embeds.to(torch.float32),
        controlnet_cond=feature_map,
        conditioning_scale=1.0,
    )

    model_pred = unet(
        noisy_latents.to(weight_dtype),
        timesteps,
        encoder_hidden_states=dummy_embeds,
        down_block_additional_residuals=[
            s.to(weight_dtype) for s in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(weight_dtype),
    ).sample

    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    mse = (model_pred.float() - target.float()) ** 2
    loss_val = mse.mean().item()

info(f"Loss at initialization: {loss_val:.4f}")
check(not math.isnan(loss_val), "Loss is NaN")
check(not math.isinf(loss_val), "Loss is Inf")
if loss_val > 2.0:
    warn(f"Loss {loss_val:.4f} is higher than typical")
elif loss_val < 0.05:
    warn(f"Loss {loss_val:.4f} is suspiciously low")
else:
    info(f"Loss at init looks reasonable, got {loss_val:.4f}")

section("Gradient Flow Through ControlNet")

optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)
optimizer.zero_grad()

noisy_latents_g = torch.randn(
    bsz, 8, latent_h, latent_w, device=device, dtype=torch.float32
)
noise_g = torch.randn_like(noisy_latents_g)
timesteps_g = torch.randint(
    0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=device
)
noisy_latents_g = noise_scheduler.add_noise(noisy_latents_g, noise_g, timesteps_g)

feature_map_g = torch.randn(bsz, 3, height, width, device=device, dtype=torch.float32)

down_res, mid_res = controlnet(
    noisy_latents_g,
    timesteps_g,
    encoder_hidden_states=dummy_embeds.detach().to(torch.float32),
    controlnet_cond=feature_map_g,
    conditioning_scale=1.0,
)

model_pred_g = unet(
    noisy_latents_g.to(weight_dtype),
    timesteps_g,
    encoder_hidden_states=dummy_embeds,
    down_block_additional_residuals=[s.to(weight_dtype) for s in down_res],
    mid_block_additional_residual=mid_res.to(weight_dtype),
).sample

target_g = noise_g
loss_g = ((model_pred_g.float() - target_g.float()) ** 2).mean()
loss_g.backward()

grad_norms = []
nan_grads = []
none_grads = []
for name, param in controlnet.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            none_grads.append(name)
        elif torch.isnan(param.grad).any():
            nan_grads.append(name)
        else:
            grad_norms.append(param.grad.norm().item())

total_params = sum(1 for p in controlnet.parameters() if p.requires_grad)
info(f"Total trainable parameters: {total_params}")
info(f"Parameters with gradients: {len(grad_norms)}")
if none_grads:
    warn(f"Parameters with None gradient: {len(none_grads)}")

check(len(nan_grads) == 0, "There are NaN gradients")
check(len(grad_norms) > 0, "No parameters received gradients")

if grad_norms:
    mean_grad = sum(grad_norms) / len(grad_norms)
    max_grad = max(grad_norms)
    info(f"Gradient norm stats: mean={mean_grad:.6f}, max={max_grad:.6f}")
    check(max_grad < 1e6, f"Gradient norms are exploding (max={max_grad:.6f})")
    if max_grad > 1.0:
        warn(f"Large gradient norm {max_grad:.4f}")

optimizer.zero_grad()

section("UNet Influence by ControlNet Residuals")

with torch.no_grad():
    test_latents = torch.randn(
        bsz, 8, latent_h, latent_w, device=device, dtype=torch.float32
    )
    test_timesteps = torch.randint(0, 1000, (bsz,), device=device)
    test_feature_map = torch.randn(
        bsz, 3, height, width, device=device, dtype=torch.float32
    )

    down_block_res_samples, mid_block_res_sample = controlnet(
        test_latents.to(torch.float32),
        test_timesteps,
        encoder_hidden_states=dummy_embeds.to(torch.float32),
        controlnet_cond=test_feature_map.to(torch.float32),
        conditioning_scale=1.0,
    )

    cn_mean = sum(s.mean().item() for s in down_block_res_samples) / len(
        down_block_res_samples
    )
    cn_max = max(s.abs().max().item() for s in down_block_res_samples)
    info(f"ControlNet down_block residuals: mean={cn_mean:.6f}, max={cn_max:.6f}")
    info(
        f"ControlNet mid_block residual: mean={mid_block_res_sample.mean().item():.6f}, max={mid_block_res_sample.abs().max().item():.6f}"
    )
    check(
        cn_max < 1e-6,
        f"ControlNet residuals are not zero-initialized (max={cn_max:.6f} < 1e-6)",
    )

    random_down_block_res_samples = [
        torch.randn_like(s) for s in down_block_res_samples
    ]
    random_mid_block_res_sample = torch.randn_like(mid_block_res_sample)

    unet_out_with_residuals = unet(
        test_latents.to(weight_dtype),
        test_timesteps,
        encoder_hidden_states=dummy_embeds,
        down_block_additional_residuals=[
            s.to(weight_dtype) for s in random_down_block_res_samples
        ],
        mid_block_additional_residual=random_mid_block_res_sample.to(weight_dtype),
    ).sample

    unet_out_without_residuals = unet(
        test_latents.to(weight_dtype),
        test_timesteps,
        encoder_hidden_states=dummy_embeds,
    ).sample

    output_diff = (unet_out_with_residuals - unet_out_without_residuals).abs()
    mean_diff = output_diff.mean().item()
    max_diff = output_diff.max().item()
    std_diff = output_diff.std().item()

    info(f"Mean output difference: {mean_diff:.6f}")
    info(f"Max output difference: {max_diff:.6f}")
    info(f"Std output difference: {std_diff:.6f}")

    check(
        mean_diff > 1e-6,
        "UNet output is not influenced by residuals",
    )

    if mean_diff < 0.001:
        warn(f"Mean difference {mean_diff:.6f} is very small")
    elif mean_diff > 1.0:
        warn(f"Mean difference {mean_diff:.6f} is large")
    else:
        info(f"Residual influence level looks reasonable (mean_diff={mean_diff:.6f})")

def check_batch(batch,bsz):
    check(
        batch["img"].shape == (bsz, 3, height, width),
        f"img shape ({bsz},3,{height},{width}), got {tuple(batch['img'].shape)}",
    )
    check(
        batch["dem"].shape == (bsz, 3, height, width),
        f"dem shape ({bsz},3,{height},{width}), got {tuple(batch['dem'].shape)}",
    )
    check(
        batch["feature_map"].shape == (bsz, 3, height, width),
        f"feature_map shape ({bsz},3,{height},{width}), got {tuple(batch['feature_map'].shape)}",
    )
    if "cloud_mask" in batch:
        check(
            batch["cloud_mask"].shape == (bsz, 1, height, width),
            f"cloud_mask shape ({bsz},1,{height},{width}), got {tuple(batch['cloud_mask'].shape)}",
        )
    check(
        batch["txt"].shape == (bsz, 77),
        f"txt shape ({bsz},77), got {tuple(batch['txt'].shape)}",
    )

    img_min, img_max = batch["img"].min().item(), batch["img"].max().item()
    info(f"img range: [{img_min:.3f}, {img_max:.3f}]")
    check(-1 <= img_min and img_max <= 1, "img values in [-1, 1] range")

    dem_min, dem_max = batch["dem"].min().item(), batch["dem"].max().item()
    info(f"dem range: [{dem_min:.3f}, {dem_max:.3f}]")
    check(-1 <= dem_min and dem_max <= 1, "dem values in [-1, 1] range")

    if "cloud_mask" in batch:
        cloud_vals = batch["cloud_mask"].unique()
        info(f"cloud_mask unique values: {cloud_vals.tolist()}")
        check(len(cloud_vals) <= 2, "cloud_mask should have at most 2 unique values (0 and 1)")
        cloud_min, cloud_max = (
            batch["cloud_mask"].min().item(),
            batch["cloud_mask"].max().item(),
        )
        check(0.0 == cloud_min and cloud_max == 1.0, "cloud_mask is [0, 1]")

    fmap_min, fmap_max = (
        batch["feature_map"].min().item(),
        batch["feature_map"].max().item(),
    )
    info(f"feature_map range: [{fmap_min:.3f}, {fmap_max:.3f}]")
    check(
        0 <= fmap_min and fmap_max <= 1,
        "feature_map values in [0, 1] range",
    )

    with torch.no_grad():
        img_b = batch["img"].to(device, dtype=weight_dtype)
        dem_b = batch["dem"].to(device, dtype=weight_dtype)
        img_latents = vae.encode(img_b).latent_dist.sample()
        dem_latents = vae.encode(dem_b).latent_dist.sample()
        img_latents = img_latents * vae.config["scaling_factor"]
        dem_latents = dem_latents * vae.config["scaling_factor"]
        latents = torch.cat([img_latents, dem_latents], dim=1).float()
    check(
        latents.shape == (bsz, 8, latent_h, latent_w),
        f"Combined latents shape ({bsz},8,{latent_h},{latent_w}), got {tuple(latents.shape)}",
    )
    info(f"Latent range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")


if mode in ("real", "both"):
    section("Data check")

    check(dataset_path is not None,"--dataset_path not provided, skipping real data checks")
    info(f"Loading dataset from: {dataset_path}")
    ds = load_dataset(dataset_path, split="train", streaming=True)
    ds = ds.with_format("torch")

    collate_fn = functools.partial(
        preprocess,
        proportion_empty_prompts=0.0,
        tokenizer=tokenizer,
        height=height,
        width=width,
    )



    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    t0 = time.time()
    batch = next(iter(loader))
    info(f"First batch loaded in {time.time() - t0:.1f}s")
    check_batch(batch,2)


terrain_pipeline = TerrainDiffusionPipeline(
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
terrain_pipeline = terrain_pipeline.to(device)

if mode in ("distill", "both"):
    section("Distill Tests")


    if not os.path.exists(prompts_csv_path):
        warn(f"Prompts CSV not found at {prompts_csv_path}")
        warn("Using a fallback test prompt")
        test_prompts = ["subpolar forests and mountains in Chile in January"]
    else:
        test_prompts = []
        with open(prompts_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("prompt", "").strip()
                if p:
                    test_prompts.append(p)
        info(f"Loaded {len(test_prompts)} prompts from {prompts_csv_path}")
        check(len(test_prompts) > 0, "Prompts CSV has at least 1 prompt")

    info("Testing generate_synthetic_batch function...")

    generator = torch.Generator(device="cpu").manual_seed(42)
    batch = generate_synthetic_batch(
        pipeline=terrain_pipeline,
        tokenizer=tokenizer,
        prompts=test_prompts,
        batch_size=1,
        height=height,
        width=width,
        weight_dtype=weight_dtype,
        device=device,
        generator=generator,
    )
    check_batch(batch,1)

section("VAE raw tensors and raw diffusion with UNet (short run)")

with torch.no_grad():
    test_prompt = "rain forests and mountains in Philippines in November"
    negative_prompt = ""

    text_input = tokenizer(
        test_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_input.input_ids.to(device)
    attention_mask = (
        text_input.attention_mask.to(device)
        if hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
        else None
    )
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0].to(
        weight_dtype
    )

    uncond_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_input.input_ids.to(device)
    uncond_attention_mask = (
        uncond_input.attention_mask.to(device)
        if hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
        else None
    )
    negative_prompt_embeds = text_encoder(
        uncond_input_ids, attention_mask=uncond_attention_mask
    )[0].to(weight_dtype)

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    num_inference_steps = 25
    guidance_scale = 7.5

    noise_scheduler.set_timesteps(num_inference_steps)
    scheduler_timesteps = list(noise_scheduler.timesteps)

    sample = torch.randn(bsz, 8, latent_h, latent_w, device=device, dtype=torch.float32)
    sample = sample * noise_scheduler.init_noise_sigma

    for i, t in enumerate(scheduler_timesteps):
        latent_model_input = (
            torch.cat([sample] * 2)
        )
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        unet_in = latent_model_input.to(weight_dtype)
        unet_out = unet(
            unet_in,
            t.to(device),
            encoder_hidden_states=prompt_embeds,
        ).sample

        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        step_output = noise_scheduler.step(
            noise_pred,
            t,
            sample,
            eta=0.0,
            return_dict=True,
        )
        sample = step_output.prev_sample
    img_latents = sample[:, :4, :, :] / vae.config["scaling_factor"]
    dem_latents = sample[:, 4:, :, :] / vae.config["scaling_factor"]

    decoded_img = vae.decode(img_latents.to(weight_dtype)).sample
    decoded_dem = vae.decode(dem_latents.to(weight_dtype)).sample

    info(f"Decoded image shape: {tuple(decoded_img.shape)}")
    info(f"Decoded DEM shape: {tuple(decoded_dem.shape)}")

    img_np = decoded_img.to(torch.float32).cpu().numpy()
    img_lower_bound_90 = np.percentile(img_np, 5)
    img_upper_bound_90 = np.percentile(img_np, 95)
    img_lower_bound_80 = np.percentile(img_np, 10)
    img_upper_bound_80 = np.percentile(img_np, 90)

    info(
        f"Decoded image stats: min={decoded_img.min().item():.6f}, max={decoded_img.max().item():.6f}, mean={decoded_img.mean().item():.6f}, std={decoded_img.std().item():.6f}, 90% range=[{img_lower_bound_90:.6f}, {img_upper_bound_90:.6f}], 80% range=[{img_lower_bound_80:.6f}, {img_upper_bound_80:.6f}]"
    )
    dem_np = decoded_dem.to(torch.float32).cpu().numpy()
    dem_lower_bound_90 = np.percentile(dem_np, 5)
    dem_upper_bound_90 = np.percentile(dem_np, 95)
    dem_lower_bound_80 = np.percentile(dem_np, 10)
    dem_upper_bound_80 = np.percentile(dem_np, 90)

    info(
        f"Decoded DEM stats: min={decoded_dem.min().item():.6f}, max={decoded_dem.max().item():.6f}, mean={decoded_dem.mean().item():.6f}, std={decoded_dem.std().item():.6f}, 90% range=[{dem_lower_bound_90:.6f}, {dem_upper_bound_90:.6f}], 80% range=[{dem_lower_bound_80:.6f}, {dem_upper_bound_80:.6f}]"
    )

    img_to_show = decoded_img[0].detach().cpu()
    img_to_show = (img_to_show + 1) / 2
    img_to_show = img_to_show.clamp(0, 1).to(torch.float32)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    np_img = img_to_show[:3].permute(1, 2, 0).numpy()
    plt.imshow(np_img)
    plt.title("Generated Image (RGB)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    dem_to_show = decoded_dem[0, 0].detach().to(torch.float32).cpu()
    dem_min, dem_max = dem_to_show.min(), dem_to_show.max()
    dem_normalized = (dem_to_show - dem_min) / (dem_max - dem_min + 1e-8)
    plt.imshow(dem_normalized, cmap="terrain")
    plt.title(f"Generated DEM (elevation)\nRange: [{dem_min:.3f}, {dem_max:.3f}]")
    plt.axis("off")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
