
# Optimized arguments for RTX 3080 16GB
$params = @(
    "train_t2i.py",
    "--dataset_path", "D:\DATA\dataset\atlas\final_data.parquet",
    "--mesa_path", "C:\Users\bldng\DEV\atlas\weights",
    "--output_dir", "D:\DATA\out\3080",
    "--train_batch_size", "4",                 # Reduced from 12 (to fit 16GB VRAM)
    "--gradient_accumulation_steps", "4",      # Increased to keep effective batch size at 16
    "--learning_rate", "1e-4",
    "--lr_warmup_steps", "500",
    "--max_train_steps", "20000",
    "--mixed_precision", "bf16",               # Ampere (30-series) supports BF16
    "--use_8bitadam", "True",                  # Essential for 16GB VRAM
    "--gradient_checkpointing", "True",        # Essential for 16GB VRAM
    "--tf32", "True",
    "--num_workers", "1",                      # Windows handles high worker counts poorly; 4 is safe
    "--seed", "42",
    "--streaming", "True"
    "--validation_steps", "1"
)

# Launch using accelerate
uv run accelerate launch --mixed_precision="bf16" --num_processes=1 @params
