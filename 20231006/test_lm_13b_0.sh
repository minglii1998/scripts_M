#!/bin/bash

cd /export/jchen169/Ming/lm-evaluation-harness

# Define common parameters
MODEL_PATH="khalidsaifullaah/lca13"
BASE_OUTPUT_PATH="results/claude2_alpaca_new_13b/MMLU"

export HF_HOME=/export/jchen169/Ming/cache
export TRANSFORMERS_CACHE=/export/jchen169/Ming/cache
export CUDA_VISIBLE_DEVICES=6,7

# Run MMLU

declare -a TASKS=(
    "hendrycksTest-abstract_algebra"
    "hendrycksTest-anatomy"
    "hendrycksTest-astronomy"
    "hendrycksTest-business_ethics"
    "hendrycksTest-clinical_knowledge"
    "hendrycksTest-college_biology"
    "hendrycksTest-college_chemistry"
    "hendrycksTest-college_computer_science"
    "hendrycksTest-college_mathematics"
    "hendrycksTest-college_medicine"
    "hendrycksTest-college_physics"
    "hendrycksTest-computer_security"
    "hendrycksTest-conceptual_physics"
    "hendrycksTest-econometrics"
    "hendrycksTest-electrical_engineering"
    "hendrycksTest-elementary_mathematics"
    "hendrycksTest-formal_logic"
    "hendrycksTest-global_facts"
    "hendrycksTest-high_school_biology"
    "hendrycksTest-high_school_chemistry"
    "hendrycksTest-high_school_computer_science"
    "hendrycksTest-high_school_european_history"
    "hendrycksTest-high_school_geography"
    "hendrycksTest-high_school_government_and_politics"
    "hendrycksTest-high_school_macroeconomics"
    "hendrycksTest-high_school_mathematics"
    "hendrycksTest-high_school_microeconomics"
    "hendrycksTest-high_school_physics"
    "hendrycksTest-high_school_psychology"
    "hendrycksTest-high_school_statistics"
    "hendrycksTest-high_school_us_history"
    "hendrycksTest-high_school_world_history"
    "hendrycksTest-human_aging"
    "hendrycksTest-human_sexuality"
    "hendrycksTest-international_law"
    "hendrycksTest-jurisprudence"
    "hendrycksTest-logical_fallacies"
    "hendrycksTest-machine_learning"
    "hendrycksTest-management"
    "hendrycksTest-marketing"
    "hendrycksTest-medical_genetics"
    "hendrycksTest-miscellaneous"
    "hendrycksTest-moral_disputes"
    "hendrycksTest-moral_scenarios"
    "hendrycksTest-nutrition"
    "hendrycksTest-philosophy"
    "hendrycksTest-prehistory"
    "hendrycksTest-professional_accounting"
    "hendrycksTest-professional_law"
    "hendrycksTest-professional_medicine"
    "hendrycksTest-professional_psychology"
    "hendrycksTest-public_relations"
    "hendrycksTest-security_studies"
    "hendrycksTest-sociology"
    "hendrycksTest-us_foreign_policy"
    "hendrycksTest-virology"
    "hendrycksTest-world_religions"
)

for TASK in "${TASKS[@]}"; do
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$TASK.json"
    CMD="python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH,use_accelerate=True \
    --tasks $TASK \
    --batch_size 1 \
    --output_path $OUTPUT_PATH \
    --device auto \
    --no_cache \
    --num_fewshot 5"
    echo "Running: $CMD"
    eval $CMD
done
