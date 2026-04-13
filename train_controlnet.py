import functools
import os
import logging
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataprep import preprocess
from models import ControlNetDEMModel, UNetDEMConditionModel
from train_utils import (
    augment_batch,
    log_validation_controlnet,
    parse_args,
    train_controlnet,
    use_canny_feature,
    cloud_percent_in_batch,
)

dataset_path = "bldng/atlas2"
mesa_path = "NewtNewt/MESA"
gradient_checkpointing = True
tf32 = True
learning_rate = 1e-5
gradient_accumulation_steps = 16
train_batch_size = 1
use_8bitadam = True
lr_warmup_steps = 700
num_train_epochs = 20
max_train_steps = 20000
lr_num_cycles = 1
lr_power = 1.0
scheduler_type = "constant_with_warmup"
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
mixed_precision = "bf16"
seed = 42
attention_backend = "sdpa"
use_torch_compile = True
resume_from_checkpoint = None
conditioning_scale = 1.0
conditioning_channels = 3
lower_bound = None
upper_bound = None
feature = "ridges"

regenerate_feature_map = False
augment = False
enable_random_crop = False
crop_scale = 0.8
enable_random_flip = False
flip_horizontal_prob = 0.5
flip_vertical_prob = 0.5
enable_channel_drop = False
channel_drop_prob = 0.25
enable_feature_dropout = False
feature_dropout_prob = 0.1

run_name = "controlnet-training"


# prob not the best
effective_step = 0
global_step = 0
epoch = 0


def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    scheduler = SchedulerType[scheduler_type.upper()]
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

    controlnet = ControlNetDEMModel.from_unet(
        unet,
        conditioning_channels=conditioning_channels,
        load_weights_from_unet=True,
    )

    def save_model_hook(models, weights, output_dir):
        torch.save(
            {
                "effective_step": effective_step,
                "global_step": global_step,
                "epoch": epoch,
            },
            os.path.join(output_dir, "controlnet_config.pth"),
        )
        i = len(weights) - 1
        while len(weights) > 0:
            weights.pop()
            model = models[i]
            torch.save(
                model.state_dict(), os.path.join(output_dir, "model_%02d.pth" % i)
            )
            i -= 1

    def load_model_hook(models, input_dir):
        def load_checkpoint(path):
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")

        config_path = os.path.join(input_dir, "controlnet_config.pth")
        if os.path.exists(config_path):
            config = load_checkpoint(config_path)
            global effective_step, global_step, epoch
            effective_step = config["effective_step"]
            global_step = config["global_step"]
            epoch = config["epoch"]

        while len(models) > 0:
            model = models.pop()
            load_path = os.path.join(input_dir, f"model_{len(models):02d}.pth")
            if os.path.exists(load_path):
                state_dict = load_checkpoint(load_path)
                model.load_state_dict(state_dict)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if attention_backend == "xformers":
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    elif attention_backend in "sdpa":
        from diffusers.models.attention_processor import AttnProcessor2_0

        unet.set_attn_processor(AttnProcessor2_0())
        controlnet.set_attn_processor(AttnProcessor2_0())

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if use_torch_compile:
        controlnet = torch.compile(controlnet)
        unet = torch.compile(unet)
        vae = torch.compile(vae)
        logger.info("torch.compile enabled")

    if use_8bitadam:
        from bitsandbytes.optim.adamw import AdamW8bit

        optimizer_class = AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    controlnet.to(accelerator.device, dtype=torch.float32)
    controlnet.train()

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
        tokenizer=None,
        height=height,
        width=width,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tokenizer_path=mesa_path,
        generate_features=regenerate_feature_map,
    )
    train_dataloader = DataLoader(
        train_dataset,  # ty:ignore[invalid-argument-type]
        batch_size=train_batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )

    lr_scheduler = get_scheduler(
        scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running ControlNet training *****")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    if "sdpa" in attention_backend:
        if torch.backends.cuda.is_flash_attention_available() and mixed_precision in ["fp16", "bf16"]:
            logger.info("  Using Flash Attention (SDPA)")
        elif torch.backends.cuda.mem_efficient_sdp_enabled():
            logger.info("  Using Memory Efficient Attention (SDPA)")
        else:
            logger.info("  Using Standard Attention (SDPA)")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Mixed precision = {mixed_precision}")
    logger.info(f" Device = {accelerator.device}")
    global resume_from_checkpoint
    if accelerator.is_main_process:
        accelerator.init_trackers(run_name)
        tracker = accelerator.get_tracker("tensorboard")
        if tracker:
            config_dict = {
                "mesa_path": mesa_path,
                "gradient_checkpointing": gradient_checkpointing,
                "tf32": tf32,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "train_batch_size": train_batch_size,
                "use_8bitadam": use_8bitadam,
                "lr_warmup_steps": lr_warmup_steps,
                "max_train_steps": max_train_steps,
                "lr_num_cycles": lr_num_cycles,
                "lr_power": lr_power,
                "scheduler_type": scheduler_type,
                "max_grad_norm": max_grad_norm,
                "set_grads_to_none": set_grads_to_none,
                "project_dir": project_dir,
                "logging_dir": logging_dir,
                "output_dir": output_dir,
                "checkpointing_steps": checkpointing_steps,
                "validation_steps": validation_steps,
                "proportion_empty_prompts": proportion_empty_prompts,
                "mixed_precision": mixed_precision,
                "seed": seed,
                "attention_backend": attention_backend,
                "use_torch_compile": use_torch_compile,
                "resume_from_checkpoint": resume_from_checkpoint,
                "conditioning_scale": conditioning_scale,
            }
            config_str = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
            tracker.writer.add_text("config", config_str, 0)

    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    effective_step = 0

    if resume_from_checkpoint is not None:
        path = None
        if resume_from_checkpoint == "latest":
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                logger.info(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = None
        else:
            path = resume_from_checkpoint

        if path is not None:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            try:
                first_epoch = (
                    global_step // len(train_dataloader) // gradient_accumulation_steps
                )
            except Exception:
                first_epoch = 0

    map_batch = next(iter(train_dataloader))
    if feature == "canny":
        map_batch = use_canny_feature(map_batch)
    validation_feature = map_batch["feature_map"].to(accelerator.device)
    del map_batch

    if accelerator.is_main_process:
        log_validation_controlnet(
            unet,
            controlnet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            accelerator,
            validation_feature,
            weight_dtype,
            height,
            width,
            global_step,
        )

    empty_token_ids = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)
    with torch.no_grad():
        empty_prompt_embeds = text_encoder(empty_token_ids)[0].to(dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            if augment:
                batch = augment_batch(
                    batch=batch,
                    enable_random_crop=enable_random_crop,
                    crop_scale=crop_scale,
                    enable_random_flip=enable_random_flip,
                    flip_horizontal_prob=flip_horizontal_prob,
                    flip_vertical_prob=flip_vertical_prob,
                    enable_channel_drop=enable_channel_drop,
                    channel_drop_prob=channel_drop_prob,
                    enable_feature_dropout=enable_feature_dropout,
                    feature_dropout_prob=feature_dropout_prob,
                )
            if feature == "canny":
                batch = use_canny_feature(batch)

            with accelerator.accumulate(controlnet):
                batch["img"] = batch["img"].to(accelerator.device, dtype=weight_dtype)
                batch["dem"] = batch["dem"].to(accelerator.device, dtype=weight_dtype)
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
                    proportion_empty_prompts=proportion_empty_prompts,
                    empty_prompt_embeds=empty_prompt_embeds,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    conditioning_scale=conditioning_scale,
                )

                effective_step += (
                    cloud_percent_in_batch(batch) / gradient_accumulation_steps
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=set_grads_to_none)
                if accelerator.sync_gradients:
                    lr_scheduler.step()
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
                        "effective_step": effective_step,
                    }
                    progress_bar.set_postfix(**logs)
                    if global_step%100 == 0:
                        del loss, batch, latents
                        torch.cuda.empty_cache()
                        logs["memory"] = torch.cuda.memory_summary()
                    accelerator.log(logs, step=global_step)

                    if accelerator.is_main_process:
                        if global_step % validation_steps == 0:
                            log_validation_controlnet(
                                unet,
                                controlnet,
                                vae,
                                text_encoder,
                                tokenizer,
                                noise_scheduler,
                                accelerator,
                                validation_feature,
                                weight_dtype,
                                height,
                                width,
                                global_step,
                            )

                        if global_step >= max_train_steps:
                            save_path = os.path.join(
                                output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            break

        if global_step >= max_train_steps:
            break

        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    parse_args(globals())
    main()
