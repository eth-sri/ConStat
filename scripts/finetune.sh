benchmarks=(
    'gsm8k'
    'arc'
    'mmlu'
    'hellaswag'
)


for benchmark in "${benchmarks[@]}"; do
    bash scripts/finetune_benchmark.sh $benchmark
done
