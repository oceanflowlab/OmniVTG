#!/bin/bash

model_path="$1"
dataset="$2"

if [ -z "$model_path" ] || [ -z "$dataset" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <model_path> <dataset>"
    exit 1
fi

model_name=$(basename "$model_path")
save_dir="eval_results/${model_name}"

if [ "$dataset" = "Activitynet" ]; then
    fps=1
else
    fps=2
fi

python evaluate_vtg.py \
    --model_path "$model_path" \
    --dataset "$dataset" \
    --output_dir "${save_dir}/${dataset}" \
    --fps "$fps" \
    --interleave_timestamps