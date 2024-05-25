#!/bin/bash
username=$1
benchmarks=(
    'gsm8k'
    'arc'
    'mmlu'
    'hellaswag'
)

# loop over benchmarks
for benchmark in "${benchmarks[@]}"; do
    MODELS=(
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
    for model in "${MODELS[@]}"; do
        output_dir="code-contamination-output/${benchmark}/${model}"

        # if path output_dir/all_output.jsonl exists, skip 
        if [ -f "${output_dir}/all_output.jsonl" ]; then
            echo "Skipping ${output_dir}"
            continue
        fi
        if [[ $model == *"meta-llama"* ]]; then
            ref_model="meta-llama/Llama-2-7b-chat-hf"
        else
            ref_model="microsoft/phi-2"
        fi
        # Create the output directory
        mkdir -p "$output_dir"
        # Run the command
        python src/run.py --target_model "${model}" \
            --ref_model "${ref_model}" \
            --data "$benchmark" \
            --output_dir "$output_dir" \
            --ratio_gen 0.4 > "${output_dir}/log.txt"
        python ../scripts/remove_models_huggingface.py
    done
done
