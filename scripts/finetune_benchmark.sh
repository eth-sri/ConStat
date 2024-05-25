learning_rates=(1e-4 1e-5)
default_learning_rate=5e-5

default_dataset="openorca"

repeats=(1)
default_repeat=5
benchmark=$1

model_names=(
    "microsoft/phi-2"
    "meta-llama/Llama-2-7b-chat-hf"
)

for model_name in "${model_names[@]}"; do
    if [[ "$model_name" == "microsoft/phi-2" ]]; then
        location=contamination/models/$benchmark
    else
        location=contamination/models/$benchmark/$model_name
    fi

    # run the default model
    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --steps 50000 --epochs 1 --random-state 0 --location $location/default --no-reference --few-shot 3 --overwrite
    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name  --dataset $default_dataset --benchmark $benchmark --rephrase --repeat $default_repeat --steps 50000 --epochs 1 --random-state 0 --location $location/rephrase --no-reference --few-shot 3 --overwrite

    # for each learning rate, create one separate loop going over all the values
    for lr in "${learning_rates[@]}"; do
        python scripts/finetune.py --lr $lr --model-name $model_name  --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --steps 50000 --epochs 1 --random-state 0 --location $location/lr_$lr --no-reference --few-shot 3 --overwrite
    done
    # for each repeat, create one separate loop going over all the values
    for repeat in "${repeats[@]}"; do
        python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $repeat --steps 50000 --epochs 1 --random-state 0 --location $location/repeat_$repeat --no-reference --few-shot 3 --overwrite
    done

    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --steps 50000 --epochs 1 --random-state 0 --location $location/default_other --no-reference --few-shot 3 --overwrite --other-few-shot
    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --steps 50000 --epochs 1 --random-state 0 --location $location/default_no --no-reference --few-shot 0 --overwrite

    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --steps 25000 --epochs 1 --random-state 0 --location $location/default_with_ref --few-shot 0 --overwrite

    python scripts/finetune.py --lr $default_learning_rate --model-name $model_name --dataset $default_dataset --benchmark $benchmark --repeat $default_repeat --rephrase --steps 25000 --epochs 1 --random-state 0 --location $location/rephrase_with_ref --few-shot 0 --overwrite

done

