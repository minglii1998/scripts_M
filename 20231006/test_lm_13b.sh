#!/bin/bash

cd /export/jchen169/Ming/lm-evaluation-harness

# Define common parameters
MODEL_PATH="khalidsaifullaah/lca13"
MODEL_NAME="claude2_alpaca_new_13b"

export TRANSFORMERS_CACHE=/export/jchen169/Ming/cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Run ARC
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks arc_challenge \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/ARC.json \
    --no_cache \
    --device cuda \
    --num_fewshot 25

# Run HellaSwag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks hellaswag \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/HellaSwag.json \
    --no_cache \
    --device cuda \
    --num_fewshot 10

# Run MMLU
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks hendrycksTest-* \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/MMLU.json \
    --no_cache \
    --device cuda \
    --num_fewshot 5

# Run TruthfulQA
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks truthfulqa_mc \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/TruthfulQA.json \
    --no_cache \
    --device cuda

# Run Extract Result
python extract_results.py --model_name $MODEL_NAME