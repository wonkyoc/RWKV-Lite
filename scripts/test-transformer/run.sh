#!/bin/bash
# an evaluation script for other transformer models

readarray -t models < <(grep -v '#' models.txt)
models=("JackFram/llama-160m")

for m in "${models[@]}"; do
    echo $m
    lm_eval --model hf \
        --model_args pretrained=$m,dtype="float" \
        --tasks lambada_openai \
        --batch_size 8 \
        --log_samples \
        --output_path results
done
