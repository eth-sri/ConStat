from constat.basic_model_loader import load_model
from constat.overlap import Perplexity, Lowercase, TopKMin
import pandas as pd
import torch
import os
import gc
from transformers import set_seed
import datasets
from prepare import *

set_seed(42)


def run_model(model, tokenizer, df, output_path, batch_size, 
             ref_model_name=None):
    set_seed(42)

    new_df = df.copy()

    if 'perplexity' not in new_df.columns:
        perplexity = Perplexity(model, tokenizer)
        new_df['perplexity_output'] = perplexity.batch_call(new_df['output'].tolist(), new_df['input'].tolist(), batch_size=batch_size)
        new_df['perplexity_input'] = perplexity.batch_call(new_df['input'].tolist(), batch_size=batch_size)
        new_df['perplexity_all'] = perplexity.batch_call((new_df['input'] + new_df['output']).tolist(), batch_size=batch_size)
    if 'topkmin' not in new_df.columns:
        topkmin = TopKMin(model, tokenizer)
        new_df['topkmin'] = topkmin.batch_call(new_df['output'].tolist(), new_df['input'].tolist(), batch_size=batch_size)
        new_df['topkmin_all'] = topkmin.batch_call((new_df['input'] + new_df['output']).tolist(), batch_size=batch_size)
    if 'lowercase' not in new_df.columns:
        lowercase = Lowercase(model, tokenizer)
        new_df['lowercase'] = lowercase.batch_call(new_df['output'].tolist(), new_df['input'].tolist(), batch_size=batch_size)
    if ref_model_name is not None and 'perplexity_ref' not in new_df.columns:
        model_ref, tokenizer = load_model(ref_model_name, return_tokenizer=True)
        model_ref.eval()
        perplexity = Perplexity(model_ref, tokenizer)
        new_df['perplexity_ref'] = perplexity.batch_call(new_df['output'].tolist(), new_df['input'].tolist(), batch_size=batch_size)
        del model_ref, perplexity.model, topkmin.model
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df.to_csv(output_path, index=False)

base_path = 'data/baselines'
base_path_data = 'data/contamination'
def main(model_name, benchmark_name, dataset_name, 
         no_cont=False, batch_size=8, ref_model_name='microsoft/phi-2', synthetic=False):
    model, tokenizer = load_model(model_name, return_tokenizer=True, trust_remote_code=False)
    model.eval()
    output_path = os.path.join(base_path, model_name, dataset_name + f"_{no_cont}_0.csv")
    if synthetic:
        output_path = os.path.join(base_path, model_name, dataset_name + f"_{no_cont}_synthetic.csv")
    if 'gsm8k' == benchmark_name:
        data = prepare_gsm8k(no_cont=no_cont, synthetic=synthetic, few_shot=0)
    elif 'mmlu' == benchmark_name:
        data = prepare_mmlu(no_cont=no_cont, synthetic=synthetic, few_shot=0)
    elif 'arc' == benchmark_name:
        data = prepare_arc(no_cont=no_cont, synthetic=synthetic, few_shot=0)
    elif 'hellaswag' == benchmark_name:
        data = prepare_hellaswag(no_cont=no_cont, synthetic=synthetic, few_shot=0)
    else:
        raise ValueError(f"Unknown benchmark {benchmark_name}")
    run_model(model, tokenizer, data, output_path, batch_size, ref_model_name=ref_model_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ref_model', type=str, default='microsoft/phi-2')
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--no-cont', action='store_true')
    parser.add_argument('--synthetic', action='store_true')

    args = parser.parse_args()

    main(args.model_name, args.benchmark, args.benchmark,
            no_cont=args.no_cont, batch_size=args.batch_size, ref_model_name=args.ref_model, 
            synthetic=args.synthetic)