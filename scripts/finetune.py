# Finetunes the model

from constat.finetune import FinetuneInstructions
from constat.preprocessing import InstructionProcessor
import pandas as pd
from transformers import set_seed
from datasets import load_dataset
import dotenv
import os
from prepare import *

set_seed(42)

dotenv.load_dotenv()

def main(model, data, location, learning_rate, epochs, use_lora=False,
         prompt_template=lambda instruction, input_: f'### Instruction: \n{instruction}\n### Input:\n{input_}\n### Assistant:\n', overwrite=False, 
         base_path=None):
    if os.path.isfile(os.path.join(location, 'config.json')) or os.path.isfile(os.path.join(location, 'adapter_config.json')):
        if not overwrite:
            print(f"Model {location} already exists, skipping")
            return
    os.makedirs(location, exist_ok=True)
    
    finetune = FinetuneInstructions(
        preprocessor=InstructionProcessor(max_tokens=2048, include_eos=True, prompt_template=prompt_template),
        config_file='configs/config_finetune.json', 
        output_dir=location, 
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        use_lora=use_lora
    )

    model = finetune.finetune(
        model,
        data,
    )

    huggingface_loc = location
    if base_path is not None:
        huggingface_loc = huggingface_loc.replace(base_path, '')
        if huggingface_loc.startswith('/'):
            huggingface_loc = huggingface_loc[1:]
    huggingface_loc = huggingface_loc.replace('/', '-')
    model.push_to_hub(huggingface_loc, private=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='microsoft/phi-2')
    parser.add_argument('--location', type=str, default='../logs')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--use-lora', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='openorca')
    parser.add_argument('--benchmark', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--base-path', type=str, default='data/models')
    parser.add_argument('--no-reference', action='store_true')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--rephrase', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--other-few-shot', action='store_true')

    args = parser.parse_args()

    if args.dataset == 'openorca':
        data = load_dataset('Open-Orca/OpenOrca')['train']
        data = data.rename_columns({'system_prompt': 'instruction', 'question': 'input', 'response': 'output'})
    elif args.dataset == 'platypus':
        data = load_dataset('garage-bAInd/Open-Platypus')['train']
        # delete the input column
        data = data.remove_columns(['input'])
        data = data.rename_columns({'instruction': 'input'})
    elif args.dataset == 'code':
        data = load_dataset('m-a-p/CodeFeedback-Filtered-Instruction')['train']
        data = data.rename_columns({'query': 'input', 'answer': 'output'})
    elif args.dataset == 'math':
        data = load_dataset('microsoft/orca-math-word-problems-200k')['train']
        data = data.rename_columns({'question': 'input', 'answer': 'output'})
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    
    data = data.shuffle(seed=args.random_state).select(range(args.steps))
    data = pd.DataFrame(data)
    if 'instruction' not in data.columns:
        data['instruction'] = ''

    if args.no_reference:
        # empty the dataset
        data = data.iloc[:0]

    def prompt_template(instruction, input_):
        if instruction != '':
            return f'### Instruction: \n{instruction}\n### Input:\n{input_}\n### Assistant:\n'
        else:
            # return f'Question: {input_}\nAnswer:'
            return f'### Input:\n{input_}\n### Assistant:\n'

    
    data['input'] = data.apply(lambda x: prompt_template(x['instruction'], x['input']), axis=1)

    if args.benchmark is not None:
        if args.benchmark == 'gsm8k':
            data_benchmark = prepare_gsm8k(args.few_shot, args.rephrase, args.training, args.other_few_shot)
        elif args.benchmark == 'arc':
            data_benchmark = prepare_arc(args.few_shot, args.rephrase, args.other_few_shot)
        elif args.benchmark == 'hellaswag':
            data_benchmark = prepare_hellaswag(args.few_shot, args.rephrase, args.training, args.other_few_shot)
        elif args.benchmark == 'mmlu':
            data_benchmark = prepare_mmlu(args.few_shot, args.rephrase, args.other_few_shot)
        else:
            raise ValueError('Not implemented')
        data_benchmark = pd.concat([data_benchmark]*args.repeat, ignore_index=True)
        data_benchmark['instruction'] = ''
        # append
        data = pd.concat([data, data_benchmark], ignore_index=True)
        # shuffle
        data = data.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    os.makedirs(os.path.join(args.base_path, args.location), exist_ok=True)
    # store the params in the location
    with open(os.path.join(os.path.join(args.base_path, args.location), 'params.txt'), 'w') as f:
        f.write(str(args))
    
    main(
        model=args.model_name,
        location=os.path.join(args.base_path, args.location),
        learning_rate=args.lr,
        epochs=args.epochs,
        use_lora=args.use_lora,
        data=data,
        prompt_template=lambda e, f: f,
        overwrite=args.overwrite,
        base_path=args.base_path
    )