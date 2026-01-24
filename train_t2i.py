import functools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataprep import preprocess
from models import UNetDEMConditionModel
from t2iadapter import Adapter

dataset_path = "bldng/atlas"
mesa_path = "NewtNewt/MESA"
gradient_checkpointing = True
tf32 = True
learning_rate = 1e-4
gradient_accumulation_steps = 16
train_batch_size = 1
use_8bitadam = True
lr_warmup_steps = 500
num_train_epochs = 10
max_train_steps = 20000
lr_num_cycles = 1
lr_power = 1.0
scheduler_type = "LINEAR"
max_grad_norm = 1.0
num_workers = 4
set_grads_to_none = True
project_dir = "./outputs"
logging_dir = "./logs"
output_dir = "./outputs/checkpoints"
checkpointing_steps = 1000
validation_steps = 1000
proportion_empty_prompts = 0.2
streaming = False
mixed_precision = "bf16"  # Use "fp16" for older GPUs, "bf16" for A100/H100
seed = 42
xformer = True


def log_validation(
    unet,
    adapter,
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
):
    logger = get_logger(__name__)
    logger.info("Running validation...")

    unet.eval()
    adapter.eval()
    vae.eval()
    text_encoder.eval()

    with torch.no_grad():
        feature_map = feature_map[:1].to(device=accelerator.device, dtype=weight_dtype)
        prompt = "rain forests and mountains in Philippines in November"

        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(
            text_input.input_ids.to(accelerator.device),
        )[0]

        negative_prompt = ""
        uncond_input = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(accelerator.device),
        )[0]

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

        vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)

        val_scheduler = DDIMScheduler.from_config(scheduler.config)
        val_scheduler.set_timesteps(50, device=accelerator.device)

        latents = randn_tensor(
            (
                1,
                unet.config["in_channels"] * 2,
                int(height) // vae_scale_factor,
                int(width) // vae_scale_factor,
            ),
            generator=None,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        latents = latents * val_scheduler.init_noise_sigma

        adapter_features = adapter(feature_map)
        adapter_features = [torch.cat([s] * 2, dim=0) for s in adapter_features]

        for t in tqdm(val_scheduler.timesteps, desc="Validation"):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = val_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_intrablock_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in adapter_features
                ],
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_scale = 7.5
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = val_scheduler.step(noise_pred, t, latents).prev_sample

        img_latents = latents[:, :4]
        dem_latents = latents[:, 4:]

        img_latents = img_latents / vae.config["scaling_factor"]
        dem_latents = dem_latents / vae.config["scaling_factor"]

        image = vae.decode(img_latents.float(), return_dict=False)[0]
        dem = vae.decode(dem_latents.float(), return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        dem = (dem / 2 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        dem = dem.cpu().permute(0, 2, 3, 1).float().numpy()

        if accelerator.is_main_process:
            tracker = accelerator.get_tracker("tensorboard")
            if tracker:
                tracker.writer.add_images(
                    "validation/img", image, step, dataformats="NHWC"
                )
                tracker.writer.add_images(
                    "validation/dem", dem, step, dataformats="NHWC"
                )

                feature_map_viz = feature_map[0].cpu().float().numpy()
                feature_map_min = feature_map_viz.min()
                feature_map_max = feature_map_viz.max()
                if feature_map_max > feature_map_min:
                    feature_map_viz = (feature_map_viz - feature_map_min) / (
                        feature_map_max - feature_map_min
                    )
                feature_map_viz = feature_map_viz.transpose(1, 2, 0)
                feature_map_viz = np.expand_dims(feature_map_viz, axis=0)
                tracker.writer.add_images(
                    "validation/feature_map", feature_map_viz, step, dataformats="NHWC"
                )

    adapter.train()


def parse_args():
    """Not perfect but sufficient for now."""
    args = sys.argv[1:]

    for i in range(0, len(args), 2):
        if i + 1 < len(args) and args[i].startswith("-"):
            try:
                var_name = args[i].lstrip("-")
                var_value = args[i + 1]
                if var_name in globals():
                    original = globals()[var_name]
                    if isinstance(original, bool):
                        var_value = var_value.lower() in ("true", "1", "yes")
                    elif isinstance(original, int):
                        var_value = int(var_value)
                    elif isinstance(original, float):
                        var_value = float(var_value)

                    globals()[var_name] = var_value
                    print(f"Updated {var_name}: {original} => {var_value}")
            except Exception as e:
                print(f"Error processing argument {args[i]}")
                raise e
        else:
            var_name = args[i].lstrip("-")
            print(f"Warning: {var_name} is not a recognized argument.")


def main():
    parse_args()
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    scheduler = SchedulerType[scheduler_type]
    accelerator_project_config = ProjectConfiguration(
        project_dir=project_dir,
        logging_dir=logging_dir,
    )

    logger = get_logger(__name__)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=accelerator_project_config,
        log_with="tensorboard",
    )
    set_seed(seed)

    noise_scheduler = DDIMScheduler.from_pretrained(mesa_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(mesa_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(mesa_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(mesa_path, subfolder="vae")
    unet = UNetDEMConditionModel.from_pretrained(mesa_path, subfolder="unet")

    def save_model_hook(models, weights, output_dir):
        i = len(weights) - 1

        while len(weights) > 0:
            weights.pop()
            model = models[i]
            torch.save(
                model.state_dict(), os.path.join(output_dir, "model_%02d.pth" % i)
            )
            i -= 1

    accelerator.register_save_state_pre_hook(save_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if xformer:
        unet.enable_xformers_memory_efficient_attention()

    # if gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    adapter = Adapter(channels=[320, 640, 1280, 1280], cin=192)
    adapter.zero_initialize()
    if gradient_checkpointing:
        adapter.enable_gradient_checkpointing()

    if use_8bitadam:
        from bitsandbytes.optim.adamw import AdamW8bit

        optimizer_class = AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = adapter.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    vae.to(accelerator.device, dtype=torch.float32)
    vae.eval()
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    adapter.to(accelerator.device, dtype=weight_dtype)
    adapter.train()

    # train_dataset = load_dataset(dataset_path, split="all")
    train_dataset = load_dataset(
        dataset_path,
        split="train",
        streaming=streaming,
    )
    train_dataset = train_dataset.with_format("torch")

    vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
    height = unet.config["sample_size"] * vae_scale_factor
    width = unet.config["sample_size"] * vae_scale_factor

    collate_fn = functools.partial(
        preprocess,
        proportion_empty_prompts=proportion_empty_prompts,
        tokenizer=tokenizer,
        height=height,
        width=width,
    )
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=train_batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, train_dataloader, lr_scheduler
    )

    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Mixed precision = {mixed_precision}")

    if accelerator.is_main_process:
        accelerator.init_trackers("t2i-adapter-training")

    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(adapter):
                batch["img"] = batch["img"].to(accelerator.device)
                batch["dem"] = batch["dem"].to(accelerator.device)
                with torch.no_grad():
                    img_latents = vae.encode(batch["img"]).latent_dist.sample()
                    dem_latents = vae.encode(batch["dem"]).latent_dist.sample()

                    img_latents = img_latents * vae.config["scaling_factor"]
                    dem_latents = dem_latents * vae.config["scaling_factor"]

                    # Keep scheduler math in float32 for stability.
                    latents = torch.cat([img_latents, dem_latents], dim=1).float()

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.rand((bsz,), device=latents.device)
                timesteps = (1 - timesteps**3) * noise_scheduler.config[
                    "num_train_timesteps"
                ]
                timesteps = timesteps.long().clamp_(
                    0, noise_scheduler.config["num_train_timesteps"] - 1
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.to(dtype=weight_dtype)

                # Encode text prompts
                with torch.no_grad():
                    text_inputs = batch["txt"]
                    text_input_ids = text_inputs.to(accelerator.device)
                    prompt_embeds = text_encoder(
                        text_input_ids,
                    )[0]

                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

                edge = batch["feature_map"].to(accelerator.device, dtype=weight_dtype)

                adapter_features = adapter(edge)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_intrablock_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in adapter_features
                    ],
                ).sample

                if noise_scheduler.config["prediction_type"] == "epsilon":
                    target = noise
                elif noise_scheduler.config["prediction_type"] == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config['prediction_type']}"
                    )

                # Apply cloud cover masking: cloud_mask is 0-1 where 1 = cloud
                cloud_mask = batch["cloud_mask"].to(
                    accelerator.device, dtype=weight_dtype
                )

                cloud_mask_latent = F.interpolate(
                    cloud_mask,
                    size=(model_pred.shape[2], model_pred.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

                loss_weight = (1.0 - cloud_mask_latent).expand_as(model_pred)

                mse = (model_pred.float() - target.float()) ** 2
                weighted_mse = mse * loss_weight

                # Normalize by sum of weights
                loss = weighted_mse.sum() / (loss_weight.sum() + 1e-8)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = adapter.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % validation_steps == 0:
                    log_validation(
                        unet,
                        adapter,
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        accelerator,
                        edge,
                        weight_dtype,
                        height,
                        width,
                        global_step,
                    )

                if global_step >= max_train_steps:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    break

        if global_step >= max_train_steps:
            break

        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main()
