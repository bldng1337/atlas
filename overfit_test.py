import functools
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from dataprep import preprocess
from models import ControlNetDEMModel, UNetDEMConditionModel
from pipeline_terrain import TerrainDiffusionPipeline
from train_utils import generate_synthetic_batch, parse_args, train_controlnet

mesa_path = "./weights"
steps = 100
lr = 1e-6
device = "cuda"
mixed_precision = "bf16"
seed = 42
prompts_csv_path = "./weights/prompts.csv"
batch_source = "synthetic"
dataset_path = None
lower_bound = None
upper_bound = None
num_inference_steps_gen = 25

parse_args(globals())

torch.manual_seed(seed)

weight_dtype = torch.float32
if mixed_precision == "fp16":
    weight_dtype = torch.float16
elif mixed_precision == "bf16":
    weight_dtype = torch.bfloat16


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

    loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)

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
        batch_size=3,
        height=height,
        width=width,
        weight_dtype=weight_dtype,
        device=device,
        num_inference_steps=num_inference_steps_gen,
        guidance_scale=7.5,
        generator=generator,
    )

optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr, weight_decay=0.0)

print(f"\n{'=' * 60}")
print(f"  Training on fixed batch for {steps} steps")
print(f"  Batch source: {batch_source}")
print(f"{'=' * 60}")
t_total = time.time()
loss_history = []

t_step = time.time()

for step in range(steps):

    optimizer.zero_grad()

    batch["img"] = batch["img"].to(device, dtype=weight_dtype)
    batch["dem"] = batch["dem"].to(device, dtype=weight_dtype)
    with torch.no_grad():
        img_latents = vae.encode(batch["img"]).latent_dist.sample()
        dem_latents = vae.encode(batch["dem"]).latent_dist.sample()

        img_latents = img_latents * vae.config["scaling_factor"]
        dem_latents = dem_latents * vae.config["scaling_factor"]

        latents = torch.cat([img_latents, dem_latents], dim=1).float()

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
        weight_dtype=weight_dtype,
        conditioning_scale=1.0,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)

    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)

    if step == 0:
        initial_loss = loss_val

    if step % 5 == 0 or step == steps - 1:
        ratio = loss_val / initial_loss if initial_loss > 0 else float("nan")
        elapsed = time.time() - t_step
        ms_per_step = elapsed / max(step + 1, 1) * 1000
        print(f"  {step:>6}  {loss_val:>10.4f}  {ratio:>10.4f}  {ms_per_step:>9.0f}ms")

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
elif reduction_ratio > 0.5:
    print(f"Loss only reduced by {(1 - reduction_ratio) * 100:.1f}% in {steps} steps.")
else:
    print(f"Loss reduced by {(1 - reduction_ratio) * 100:.1f}%")


if not passed:
    print("\nOVERFIT TEST FAILED")
    sys.exit(1)
