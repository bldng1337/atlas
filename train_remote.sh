nohup uv run tensorboard --logdir ./logs --port 8888 --bind_all &

uv run accelerate launch --mixed_precision="bf16" train_t2i.py \
    --dataset_path "bldng/atlas" \
    --mesa_path "NewtNewt/MESA" \
    --output_dir "./outputs/run_5090_uv" \
    --train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 500 \
    --max_train_steps 20000 \
    --checkpointing_steps 1000 \
    --mixed_precision "bf16" \
    --use_8bitadam True \
    --gradient_checkpointing True \
    --num_workers 0 \
    --seed 42
