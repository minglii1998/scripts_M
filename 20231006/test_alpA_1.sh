#!/bin/bash

cd /export/jchen169/Ming/lm-evaluation-harness

# Define common parameters
MODEL_PATH="/export/jchen169/Ming/trained_models_fs/try_alp_A_sharegpt"
MODEL_NAME="try_alp_A_sharegpt"

export HF_HOME=/export/jchen169/Ming/cache
export TRANSFORMERS_CACHE=/export/jchen169/Ming/cache
export CUDA_VISIBLE_DEVICES=0

# Run MMLU
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH \
    --tasks hendrycksTest-* \
    --batch_size 1 \
    --output_path results/$MODEL_NAME/MMLU.json \
    --no_cache \
    --device cuda \
    --num_fewshot 5

