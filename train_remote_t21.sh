nohup uv run tensorboard --logdir ./logs --port 8888 --bind_all &

uv run accelerate launch --mixed_precision="bf16" train_t2i.py \
    --dataset_path "bldng/atlas2" \
    --mesa_path "NewtNewt/MESA" \
    --output_dir "./outputs/run_5090_uv" \
    --train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --lr_warmup_steps 700 \
    --max_train_steps 20000 \
    --checkpointing_steps 1000 \
    --mixed_precision "bf16" \
    --use_8bitadam True \
    --gradient_checkpointing True \
    --num_workers 0 \
    --xformer False \
    --seed 42 \
    --resume_from_checkpoint "latest"
