uv run tensorboard --logdir ./logs --port 8888 --bind_all &

$params = @(
    "train_t2i.py",
    "--dataset_path", "D:\DATA\dataset\atlas\final_data.parquet",
    "--mesa_path", "C:\Users\bldng\DEV\atlas\weights",
    "--output_dir", "D:\DATA\out\3080",
    "--train_batch_size", "4",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "1e-4",
    "--lr_warmup_steps", "500",
    "--max_train_steps", "20000",
    "--mixed_precision", "bf16",
    "--use_8bitadam", "True",
    "--gradient_checkpointing", "True",
    "--tf32", "True",
    "--num_workers", "1",
    "--seed", "42",
    "--streaming", "True"
    "--validation_steps", "1000"
    "--resume_from_checkpoint", "latest"
)

uv run accelerate launch --num_processes=1 @params
