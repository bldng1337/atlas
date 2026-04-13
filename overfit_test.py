import functools
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from torch.amp import autocast, GradScaler

from typing import cast

from dataprep import preprocess
from models import ControlNetDEMModel, UNetDEMConditionModel
from pipeline_terrain import (
    TerrainDiffusionPipeline,
    TerrainDiffusionControlNetPipeline,
)
from train_utils import generate_synthetic_batch, parse_args, train_controlnet, cloud_percent_in_batch
from tqdm import tqdm


def sanitize_for_json(obj):
    """Replace NaN floats with None so json.dump produces valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


mesa_path = "./weights"
steps = 5000
lr = 7e-5
device = "cuda"
mixed_precision = "fp16"  # "bf16"
seed = 42
batch_size = 1
prompts_csv_path = "./weights/prompts.csv"
batch_source = "dataset"
dataset_path = None
lower_bound = None
upper_bound = None
num_inference_steps_gen = 25
xformer = False
hardcode_step = False
use_8bitadam = True
use_torch_compile = True

parse_args(globals())

run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
output_dir = os.path.join("tests", "overfit", run_timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f"Test results will be saved to: {output_dir}")

torch.manual_seed(seed)

if use_8bitadam:
    from bitsandbytes.optim.adamw import AdamW8bit

    optimizer_class = AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

weight_dtype = torch.float32
if mixed_precision == "fp16":
    weight_dtype = torch.float16
elif mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
scaler = GradScaler()

print("Loading models...")
noise_scheduler = DDIMScheduler.from_pretrained(mesa_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(mesa_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(mesa_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(mesa_path, subfolder="vae")
unet = UNetDEMConditionModel.from_pretrained(mesa_path, subfolder="unet")

controlnet = ControlNetDEMModel.from_unet(
    unet, conditioning_channels=3, load_weights_from_unet=True
)

vae.to(device, dtype=weight_dtype).eval().requires_grad_(False)
unet.to(device, dtype=weight_dtype).eval().requires_grad_(False)
text_encoder.to(device, dtype=weight_dtype).eval().requires_grad_(False)
controlnet.to(device, dtype=torch.float32).train()
controlnet.enable_gradient_checkpointing()

if xformer:
    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()

if use_torch_compile:
    controlnet = cast(ControlNetDEMModel, torch.compile(controlnet))
    unet = cast(UNetDEMConditionModel,torch.compile(unet))
    vae =  cast(AutoencoderKL,torch.compile(vae))
n_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
print(f"  ControlNet trainable params: {n_params / 1e6:.1f}M")

print("Generating fixed batch")

vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
height = unet.config["sample_size"] * vae_scale_factor
width = unet.config["sample_size"] * vae_scale_factor
generator = torch.Generator(device="cpu").manual_seed(seed)

if batch_source == "dataset":
    if dataset_path is None:
        raise ValueError("batch_source='dataset' but --dataset_path not provided")

    print(f"  Loading dataset from: {dataset_path}")
    ds = load_dataset(dataset_path, split="train", streaming=True)
    ds = ds.with_format("torch")

    collate_fn = functools.partial(
        preprocess,
        proportion_empty_prompts=0.0,
        tokenizer=tokenizer,
        height=height,
        width=width,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    batch = next(iter(loader))
    print(f"  Batch loaded successfully")
else:
    terrain_pipeline = TerrainDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,  # ty:ignore[invalid-argument-type]
        scheduler=noise_scheduler,  # ty:ignore[invalid-argument-type]
        safety_checker=None,  # ty:ignore[invalid-argument-type]
        feature_extractor=None,  # ty:ignore[invalid-argument-type]
        image_encoder=None,  # ty:ignore[invalid-argument-type]
        requires_safety_checker=False,
    ).to(device)


    if os.path.exists(prompts_csv_path):
        import csv

        with open(prompts_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prompts = [
                row.get("prompt", "").strip()
                for row in reader
                if row.get("prompt", "").strip()
            ]
        test_prompt = prompts[0] if prompts else "mountains and valleys"
    else:
        test_prompt = "mountains and valleys in the Alps"

    print(f"  Prompt: '{test_prompt}'")
    batch = generate_synthetic_batch(
        pipeline=terrain_pipeline,
        tokenizer=tokenizer,
        prompts=[test_prompt],
        batch_size=batch_size,
        height=height,
        width=width,
        weight_dtype=weight_dtype,
        device=device,
        num_inference_steps=num_inference_steps_gen,
        guidance_scale=7.5,
        generator=generator,
    )

optimizer = optimizer_class(controlnet.parameters(), lr=lr, weight_decay=0.0)

print(f"\n{'=' * 60}")
print(f"  Training on fixed batch for {steps} steps")
print(f"  Batch source: {batch_source}")
print(f"{'=' * 60}")


print(
    f"\n{'Step':>6}    {'Effective Step':>10}  {'Loss':>10}  {'Ratio':>10}  {'Max Grad':>10}  {'Mean Grad':>10}  {'Zero Cov Weights':>10}  {'Time/Step':>9}"
)
batch["img"] = batch["img"].to(device, dtype=weight_dtype)
batch["dem"] = batch["dem"].to(device, dtype=weight_dtype)
batch["feature_map"] = batch["feature_map"].to(device, dtype=weight_dtype)
batch["txt"] = batch["txt"].to(device)
if "cloud_mask" in batch:
    batch["cloud_mask"] = batch["cloud_mask"].to(device)
with torch.no_grad():
    img_latents = vae.encode(batch["img"]).latent_dist.sample()
    dem_latents = vae.encode(batch["dem"]).latent_dist.sample()

    img_latents = img_latents * vae.config["scaling_factor"]
    dem_latents = dem_latents * vae.config["scaling_factor"]

    latents = torch.cat([img_latents, dem_latents], dim=1).float()
    precomputed_prompt_embeds = text_encoder(batch["txt"])[0]
if hardcode_step:
    timestep = torch.randint(
        0,
        noise_scheduler.config["num_train_timesteps"],
        (latents.shape[0],),
        device=device,
    )
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

gc.collect()

# Save configuration
config = {
    "timestamp": run_timestamp,
    "mesa_path": mesa_path,
    "steps": steps,
    "lr": lr,
    "device": device,
    "mixed_precision": mixed_precision,
    "seed": seed,
    "batch_size": batch_size,
    "prompts_csv_path": prompts_csv_path,
    "batch_source": batch_source,
    "dataset_path": dataset_path,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound,
    "num_inference_steps_gen": num_inference_steps_gen,
    "xformer": xformer,
    "hardcode_step": hardcode_step,
    "use_8bitadam": use_8bitadam,
    "weight_dtype": str(weight_dtype),
    "trainable_params": n_params,
    "height": height,
    "width": width,
}
if batch_source != "dataset":
    config["test_prompt"] = test_prompt
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(sanitize_for_json(config), f, indent=2)

t_total = time.time()
loss_history = []
step_logs = []

t_step = time.time()

avg_loss = 0
avg_num = 0
effective_step=0
for step in tqdm(range(steps)):
    optimizer.zero_grad(set_to_none=True)
    if mixed_precision != "no":
        with autocast(device_type=device, dtype=weight_dtype):
            loss = train_controlnet(
                latents=latents,
                controlnet=controlnet,
                unet=unet,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                batch=batch,
                proportion_empty_prompts=0,
                empty_prompt_embeds=None,
                device=device,
                fixed_t=timestep if hardcode_step else None,
                fixed_noise=noise if hardcode_step else None,
                fixed_noisy_latents=noisy_latents if hardcode_step else None,
                weight_dtype=weight_dtype,
                conditioning_scale=1.0,
                prompt_embeds=precomputed_prompt_embeds,
            )
    else:
        loss = train_controlnet(
            latents=latents,
            controlnet=controlnet,
            unet=unet,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            batch=batch,
            proportion_empty_prompts=0,
            empty_prompt_embeds=None,
            device=device,
            fixed_t=timestep if hardcode_step else None,
            fixed_noise=noise if hardcode_step else None,
            fixed_noisy_latents=noisy_latents if hardcode_step else None,
            weight_dtype=weight_dtype,
            conditioning_scale=1.0,
            prompt_embeds=precomputed_prompt_embeds,
        )
    effective_step+=cloud_percent_in_batch(batch)

    if mixed_precision == "no":
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    loss_val = loss.item()
    loss_history.append(loss_val)
    avg_loss += loss_val
    avg_num += 1

    if step == 0:
        initial_loss = loss_val
    log_interval = max(steps // 10, 1)
    if step % log_interval == 0 or step == steps - 1 or step <= 5 or step == 4999:
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
        if len(grad_norms) > 0:
            mean_grad = sum(grad_norms) / len(grad_norms)
            max_grad = max(grad_norms)
        else:
            mean_grad = float("nan")
            max_grad = float("nan")

        if none_grads:
            tqdm.write(
                f"  WARNING: {len(none_grads)} parameters have None gradients: {none_grads}"
            )
        if nan_grads:
            tqdm.write(
                f"  WARNING: {len(nan_grads)} parameters have NaN gradients: {nan_grads}"
            )
        param = float("-inf")
        for block in controlnet.controlnet_down_blocks:
            for p in block.parameters():
                param = max(param, p.data.abs().max().item())
        for p in controlnet.controlnet_mid_block.parameters():
            param = max(param, p.data.abs().max().item())
        loss_val = avg_loss / avg_num
        avg_loss = 0
        avg_num = 0
        ratio = loss_val / initial_loss if initial_loss > 0 else float("nan")
        elapsed = time.time() - t_step
        ms_per_step = elapsed / max(step + 1, 1) * 1000
        tqdm.write(
            f"  {step:>6}  {effective_step:>6}  {loss_val:>10.4f}  {ratio:>10.4f}  {max_grad:>10.4f}  {mean_grad:>10.4f}  {param:>10.8f}  {ms_per_step:>9.0f}ms"
        )
        step_logs.append({
            "step": step,
            "effective_step": effective_step,
            "loss": loss_val,
            "ratio": ratio,
            "max_grad": max_grad,
            "mean_grad": mean_grad,
            "max_param_weight": param,
            "ms_per_step": ms_per_step,
        })

total_time = time.time() - t_total


print(f"\n{'=' * 60}")

final_loss = loss_history[-1]
reduction_ratio = final_loss / initial_loss if initial_loss > 0 else float("nan")
min_loss = min(loss_history)
min_step = loss_history.index(min_loss)

print(f"  Initial loss:     {initial_loss:.4f}")
print(f"  Final loss:       {final_loss:.4f}")
print(f"  Minimum loss:     {min_loss:.4f} (at step {min_step})")
print(
    f"  Loss reduction:   {(1 - reduction_ratio) * 100:.1f}%"
)
print(f"  Total time:       {total_time:.1f}s ({total_time / steps * 1000:.0f}ms/step)")

mid = len(loss_history) // 2
first_half_avg = sum(loss_history[:mid]) / max(mid, 1)
second_half_avg = sum(loss_history[mid:]) / max(len(loss_history) - mid, 1)
trending_down = second_half_avg < first_half_avg


has_nan = any(math.isnan(l) for l in loss_history)

print(f"\n{'=' * 60}")

passed = True

if has_nan:
    print("Loss went NaN")
    passed = False
elif not trending_down:
    print(
        f"Loss is NOT trending down (first half avg={first_half_avg:.4f}, second half avg={second_half_avg:.4f})"
    )
    passed = False
if trending_down:
    print(f"Loss is trending down (first half avg={first_half_avg:.4f}, second half avg={second_half_avg:.4f})")
elif reduction_ratio > 0.5:
    print(f"Loss only reduced by {(1 - reduction_ratio) * 100:.1f}% in {steps} steps.")
else:
    print(f"Loss reduced by {(1 - reduction_ratio) * 100:.1f}%")

# Save results
results = {
    "passed": passed,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "min_loss": min_loss,
    "min_loss_step": min_step,
    "reduction_ratio": reduction_ratio,
    "reduction_pct": (1 - reduction_ratio) * 100,
    "total_time_s": total_time,
    "ms_per_step": total_time / steps * 1000,
    "first_half_avg": first_half_avg,
    "second_half_avg": second_half_avg,
    "trending_down": trending_down,
    "has_nan": has_nan,
    "loss_history": loss_history,
    "step_logs": step_logs,
    "effective_steps": effective_step,
}
with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(sanitize_for_json(results), f, indent=2)

summary_lines = [
    f"Overfit Test Results - {run_timestamp}",
    f"{'=' * 60}",
    "",
    "CONFIGURATION",
    f"  Steps:             {steps}",
    f"  Learning rate:     {lr}",
    f"  Batch size:        {batch_size}",
    f"  Mixed precision:   {mixed_precision}",
    f"  Seed:              {seed}",
    f"  Batch source:      {batch_source}",
    f"  Device:            {device}",
    f"  optimizer:         {'AdamW8bit' if use_8bitadam else 'AdamW'}",
    f"  hardcode_step:    {hardcode_step}",
    "",
    "RESULTS",
    f"  Initial loss:      {initial_loss:.4f}",
    f"  Final loss:        {final_loss:.4f}",
    f"  Minimum loss:      {min_loss:.4f} (at step {min_step})",
    f"  Loss reduction:    {(1 - reduction_ratio) * 100:.1f}%",
    f"  Total time:        {total_time:.1f}s ({total_time / steps * 1000:.0f}ms/step)",
    f"  Trending down:     {trending_down}",
    f"  Has NaN:           {has_nan}",
]
with open(os.path.join(output_dir, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nResults saved to: {output_dir}")

if not passed:
    print("\nOVERFIT TEST FAILED")
    sys.exit(1)
