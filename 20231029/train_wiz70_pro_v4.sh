cd /export/jchen169/Ming/FastChat

export WANDB_MODE=dryrun
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/export/jchen169/Ming/cache
export TRANSFORMERS_CACHE=/export/jchen169/Ming/cache

array=(
    3
)
for i in "${array[@]}"
do
    echo $i
        torchrun --nproc_per_node=8 fastchat/train/train_xformers.py \
            --model_name_or_path lmsys/vicuna-7b-v1.5 \
            --data_path /export/jchen169/Ming/data/wiz70_selection_pro_v4_ppl_sharegpt.json \
            --model_prompt vicuna \
            --lazy_preprocess False \
            --cache_dir ../cache \
            --bf16 True \
            --output_dir /export/jchen169/Ming/trained_models_fs/wiz70_pro_v4_ppl_vicuna_${i}epo \
            --num_train_epochs ${i} \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 200000 \
            --save_total_limit 1 \
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap offload" \
            --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True 
done

