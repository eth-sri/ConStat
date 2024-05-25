benchmarks=(
    'gsm8k'
    'arc'
    'mmlu'
    'hellaswag'
)
username=$1


for benchmark in "${benchmarks[@]}"; do
    models=(
        "$username/contamination-models-$benchmark-default"
        "$username/contamination-models-$benchmark-rephrase"
        "$username/contamination-models-$benchmark-lr_1e-4"
        "$username/contamination-models-$benchmark-lr_1e-5"
        "$username/contamination-models-$benchmark-repeat_1"
        "$username/contamination-models-$benchmark-default_other"
        "$username/contamination-models-$benchmark-default_no"
        "$username/contamination-models-$benchmark-default_with_ref"
        "$username/contamination-models-$benchmark-rephrase_with_ref"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-default"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-rephrase"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-lr_1e-4"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-lr_1e-5"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-repeat_1"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-default_other"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-default_no"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-default_with_ref"
        "$username/contamination-models-$benchmark-meta-llama-Llama-2-7b-chat-hf-rephrase_with_ref"
    )
    for model in "${models[@]}"; do
        echo "Evaluating $model on $benchmark" 
        python scripts/evaluate_baselines.py --model $model --benchmark $benchmark
        python scripts/evaluate_baselines.py --model $model --benchmark $benchmark --no-cont
        python scripts/remove_models_huggingface.py
    done
done