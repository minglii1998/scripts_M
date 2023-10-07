#!/bin/bash

cd /export/jchen169/Ming/lm-evaluation-harness

# Define common parameters
MODEL_PATH="khalidsaifullaah/lca13"
MODEL_NAME="claude2_alpaca_new_13b"

export TRANSFORMERS_CACHE=/export/jchen169/Ming/cache
export CUDA_VISIBLE_DEVICES=6,7

# Run MMLU
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks hendrycksTest-* \
    --batch_size 1 \
    --output_path results/$MODEL_NAME/MMLU.json \
    --no_cache \
    --device cuda \
    --num_fewshot 5


# Run Extract Result
python extract_results.py --model_name $MODEL_NAME