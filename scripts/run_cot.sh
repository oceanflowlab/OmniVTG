#!/bin/bash

# You can use 2B instead of 7B
MODEL_NAME="Qwen2.5-VL-7B-Instruct-SFT"

export PYTHONPATH=src:$PYTHONPATH
export TRANSFORMERS_VERBOSITY=error

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If your dataset is mixed with images and videos, you need to use zero2.
deepspeed --master_port 16672 src/train_sft.py \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "data/OmniVTG/train_sft_cot.json" \
    --image_folder "data/OmniVTG/videos"\
    --task sft \
    --interleave_timestamps True \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir outputs/Qwen2.5-VL-7B-Instruct-CoT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((16 * 28 * 28)) \
    --video_total_pixels $((3584 * 28 * 28)) \
    --max_frames 768 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --dataloader_num_workers 8
