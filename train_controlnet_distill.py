import csv
import os
from typing import List

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from models import ControlNetDEMModel, UNetDEMConditionModel
from pipeline_terrain import TerrainDiffusionPipeline
from train_utils import (
    generate_synthetic_batch,
    log_validation_controlnet,
    parse_args,
    train_controlnet, augment_batch,
)

mesa_path = "NewtNewt/MESA"
prompts_csv_path = "./prompts.csv"
gradient_checkpointing = True
tf32 = True
learning_rate = 1e-5
gradient_accumulation_steps = 16
train_batch_size = 1
use_8bitadam = True
lr_warmup_steps = 600
max_train_steps = 20000
lr_num_cycles = 1
lr_power = 1.0
scheduler_type = "constant_with_warmup"
max_grad_norm = 1.0
set_grads_to_none = True
project_dir = "./outputs"
logging_dir = "./logs"
output_dir = "./outputs/checkpoints"
checkpointing_steps = 1000
validation_steps = 1000
proportion_empty_prompts = 0.2
mixed_precision = "bf16"
seed = 42
use_ema = False
ema_decay = 0.9999
xformer = True
resume_from_checkpoint = None
conditioning_scale = 1.0

enable_random_crop = False
crop_scale = 0.8
enable_random_flip = False
flip_horizontal_prob = 0.5
flip_vertical_prob = 0.5
enable_channel_drop = False
channel_drop_prob = 0.25
enable_feature_dropout = False
feature_dropout_prob = 0.1
num_inference_steps_gen = 7
guidance_scale_gen = 7.5


def load_prompts_from_csv(csv_path: str) -> List[str]:
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def main():
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

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + accelerator.process_index)
    set_seed(seed)

    noise_scheduler = DDIMScheduler.from_pretrained(mesa_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(mesa_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(mesa_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(mesa_path, subfolder="vae")
    unet = UNetDEMConditionModel.from_pretrained(mesa_path, subfolder="unet")

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
    )

    controlnet = ControlNetDEMModel.from_unet(
        unet,
        conditioning_channels=3,
        load_weights_from_unet=True,
    )

    terrain_pipeline = terrain_pipeline.to(accelerator.device)

    if os.path.exists(prompts_csv_path):
        logger.info(f"Loading prompts from {prompts_csv_path}...")
        gen_prompts = load_prompts_from_csv(prompts_csv_path)
        logger.info(f"Loaded {len(gen_prompts)} prompts from {prompts_csv_path}")
        if len(gen_prompts) < train_batch_size:
            logger.warning(
                f"Number of prompts ({len(gen_prompts)}) is less than batch size ({train_batch_size})"
            )
    else:
        logger.error(f"Prompts CSV file not found at {prompts_csv_path}")
        raise FileNotFoundError(f"Prompts CSV file not found at {prompts_csv_path}")

    if use_ema:
        ema_controlnet = EMAModel(
            controlnet.parameters(),
            model_cls=ControlNetDEMModel,
            model_config=controlnet.config,
            decay=ema_decay,
            foreach=True,
        )

    def save_model_hook(models, weights, output_dir):
        if use_ema:
            ema_controlnet.save_pretrained(os.path.join(output_dir, "ema_controlnet"))
        i = len(weights) - 1
        while len(weights) > 0:
            weights.pop()
            model = models[i]
            torch.save(
                model.state_dict(), os.path.join(output_dir, "model_%02d.pth" % i)
            )
            i -= 1

    def load_model_hook(models, input_dir):
        if use_ema:
            ema_path = os.path.join(input_dir, "ema_controlnet")
            if os.path.exists(ema_path):
                load_model = EMAModel.from_pretrained(
                    ema_path, model_cls=type(accelerator.unwrap_model(controlnet))
                )
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model
        while len(models) > 0:
            model = models.pop()
            load_path = os.path.join(input_dir, f"model_{len(models):02d}.pth")
            if os.path.exists(load_path):
                model.load_state_dict(torch.load(load_path))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if xformer:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

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
        weight_decay=1e-4,
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

    vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
    height = unet.config["sample_size"] * vae_scale_factor
    width = unet.config["sample_size"] * vae_scale_factor

    lr_scheduler = get_scheduler(
        scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    controlnet, optimizer, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, lr_scheduler
    )

    if use_ema:
        ema_controlnet.to(accelerator.device)

    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Mixed precision = {mixed_precision}")
    logger.info("  Using synthetic data generation")

    global resume_from_checkpoint

    if accelerator.is_main_process:
        accelerator.init_trackers("controlnet-training-distill")
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
                "use_ema": use_ema,
                "ema_decay": ema_decay,
                "xformer": xformer,
                "resume_from_checkpoint": resume_from_checkpoint,
                "conditioning_scale": conditioning_scale,
                "num_inference_steps_gen": num_inference_steps_gen,
                "guidance_scale_gen": guidance_scale_gen,
            }
            config_str = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
            tracker.writer.add_text("config", config_str, 0)

    global_step = 0

    initial_global_step = 0

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

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    initial_batch = generate_synthetic_batch(
        pipeline=terrain_pipeline,
        tokenizer=tokenizer,
        prompts=gen_prompts,
        batch_size=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps_gen,
        guidance_scale=guidance_scale_gen,
        weight_dtype=weight_dtype,
        device=accelerator.device,
        generator=generator,
    )
    validation_feature = initial_batch["feature_map"].to(
        accelerator.device, dtype=weight_dtype
    )

    empty_token_ids = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)
    with torch.no_grad():
        empty_prompt_embeds = text_encoder(empty_token_ids)[0].to(dtype=weight_dtype)

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
            terrain_pipeline=terrain_pipeline,
            gen_prompts=gen_prompts,
            generator=generator,
            num_random_validations=1,
            conditioning_scale=conditioning_scale,
        )

    if global_step == 0:
        accelerator.save_state(os.path.join(output_dir, f"checkpoint-{global_step}"))
        logger.info(
            f"Saved initial state to {os.path.join(output_dir, f'checkpoint-{global_step}')}"
        )

    while global_step < max_train_steps:
        batch = generate_synthetic_batch(
            pipeline=terrain_pipeline,
            tokenizer=tokenizer,
            prompts=gen_prompts,
            batch_size=train_batch_size,
            height=height,
            width=width,
            guidance_scale=guidance_scale_gen,
            num_inference_steps=num_inference_steps_gen,
            weight_dtype=weight_dtype,
            device=accelerator.device,
            generator=generator,
        )
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
        with accelerator.accumulate(controlnet):
            batch["img"] = batch["img"].to(accelerator.device)
            batch["dem"] = batch["dem"].to(accelerator.device)

            with torch.no_grad():
                img_preprocessed = terrain_pipeline.image_processor.preprocess(
                    batch["img"]
                )
                dem_preprocessed = terrain_pipeline.image_processor.preprocess(
                    batch["dem"]
                )
                img_latents = vae.encode(img_preprocessed).latent_dist.sample()
                dem_latents = vae.encode(dem_preprocessed).latent_dist.sample()

                img_latents = img_latents * vae.config["scaling_factor"]
                dem_latents = dem_latents * vae.config["scaling_factor"]

                latents = torch.cat([img_latents, dem_latents], dim=1).float()

            loss = train_controlnet(
                latents=latents,
                controlnet=controlnet,
                unet=unet,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                proportion_empty_prompts=proportion_empty_prompts,
                batch=batch,
                empty_prompt_embeds=empty_prompt_embeds,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                conditioning_scale=conditioning_scale,
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = controlnet.parameters()
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=set_grads_to_none)

        if accelerator.sync_gradients:
            if use_ema:
                ema_controlnet.step(accelerator.unwrap_model(controlnet).parameters())
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % checkpointing_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if use_ema:
                logs["ema_decay"] = ema_controlnet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % validation_steps == 0:
                if accelerator.is_main_process:
                    if use_ema:
                        ema_controlnet.store(
                            accelerator.unwrap_model(controlnet).parameters()
                        )
                        ema_controlnet.copy_to(
                            accelerator.unwrap_model(controlnet).parameters()
                        )
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
                        terrain_pipeline=terrain_pipeline,
                        gen_prompts=gen_prompts,
                        generator=generator,
                        num_random_validations=1,
                        conditioning_scale=conditioning_scale,
                    )
                    if use_ema:
                        ema_controlnet.restore(
                            accelerator.unwrap_model(controlnet).parameters()
                        )

            if global_step >= max_train_steps:
                if use_ema:
                    ema_controlnet.store(
                        accelerator.unwrap_model(controlnet).parameters()
                    )
                    ema_controlnet.copy_to(
                        accelerator.unwrap_model(controlnet).parameters()
                    )
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                if use_ema:
                    ema_controlnet.restore(
                        accelerator.unwrap_model(controlnet).parameters()
                    )
                break
    accelerator.end_training()


if __name__ == "__main__":
    parse_args(globals())
    main()
