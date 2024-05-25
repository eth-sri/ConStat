# runs the detection tests

from constat import ConStat, perform_test, load_result
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from loguru import logger

base_path = 'lm-evaluation-harness/output'

ref_models = [
    "microsoft/phi-2",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-2-70b-chat-hf",
    # "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-7b",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "allenai/OLMo-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"

]

extra_ref_models = [
    "mistralai/Mistral-7B-v0.1"
]

test_models = [
  "yam-peleg/Experiment26-7B",
  "BarraHome/Mistroll-7B-v2.2",
  "MTSAIR/multi_verse_model",
  "Qwen/Qwen1.5-1.8B-Chat",
  "internlm/internlm2-7b",
  "internlm/internlm2-1_8b",
  "internlm/internlm2-math-7b",
  "internlm/internlm2-math-base-7b",
  "Qwen/Qwen1.5-4B-Chat",
  "Qwen/Qwen1.5-7B-Chat",
  "Qwen/Qwen1.5-14B-Chat",
  "Qwen/Qwen1.5-72B-Chat",
  "Qwen/Qwen1.5-110B-Chat",
  "stabilityai/stablelm-base-alpha-7b-v2",
  "stabilityai/stablelm-2-12b-chat",
  "stabilityai/stablelm-2-12b",
  "stabilityai/stablelm-2-1_6b",
  "stabilityai/stablelm-zephyr-3b",
  "stabilityai/stablelm-2-1_6b-chat",
  "mistralai/Mistral-7B-v0.1",
  "zero-one-ai/Yi-34B",
  "zero-one-ai/Yi-6B",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "mistralai/Mistral-7B-v0.3",
  "mistral-community/Mistral-7B-v0.2",
  "microsoft/Phi-3-small-8k-instruct",
  "microsoft/Phi-3-medium-4k-instruct"
]

default_metric_map = {
    'gsm8k': 'flexible_extract',
    'mathqa': 'acc_norm',
    'arc': 'acc_norm',
    'mmlu': 'acc',
    'sciq': 'acc_norm',
    'hellaswag': 'acc_norm',
    'mmlu_combination': 'acc',
    'lambada_openai': 'acc',
    'mmlu_synthetic': 'acc'
}

default_reference_benchmark = {
    'gsm8k': 'mathqa',
    'hellaswag': 'lambada_openai',
    'arc': 'sciq',
    'mmlu': 'mmlu_combination'
}

random_performances = {
    'gsm8k': 0,
    'mathqa': 0.25,
    'arc': 0.25,
    'mmlu': 0.25,
    'sciq': 0.25,
    'hellaswag': 0.25,
    'mmlu_combination': 0.25,
    'lambada_openai': 0,
    'mmlu_synthetic': 0.25
}

def get_contaminated_models(benchmark, username):
    models = ["default", "rephrase", "lr_1e-4", "lr_1e-5", "repeat_1", "default_other", "default_no", 
              "default_with_ref", "rephrase_with_ref"]
    full_model_names = [
        f"{username}/contamination-models-{benchmark}-meta-llama-Llama-2-7b-chat-hf-{model}" for model in models
    ]
    full_model_names += [
        f"{username}/contamination-models-{benchmark}-{model}" for model in models
    ]
    return full_model_names



def perform_full_test(benchmark, ref_benchmark, ref_models, to_test_models, 
                 add_extended_models=True, n_bootstrap=1000, add_no_cont_result=True, 
                 extra_ref_models=[]):
    """
    Perform a ConStat test for the given benchmark and reference benchmark.

    Args:
        benchmark (str): The benchmark name.
        ref_benchmark (str): The reference benchmark name.
        ref_models (list): List of reference models.
        to_test_models (list): List of models to test.
        add_extended_models (bool, optional): Whether to add the random model. Defaults to True.
        n_bootstrap (int, optional): Number of bootstrap iterations. Defaults to 1000.
        add_no_cont_result (bool, optional): Whether to add the results on the uncontaminated part of the benchmark. Only set to False if you are reproducing our results and testing the contaminated models. Defaults to True.
        extra_ref_models (list, optional): List of extra reference models. These are the models that were initially included in the reference models, but then removed. Defaults to [].

    Returns:
        pandas.DataFrame: The results of the full test.
    """
    logger.info(benchmark + ref_benchmark)
    random_performance_no_dash = random_performances.get(benchmark[:benchmark.index('_')], None)
    random_performance_benchmark = random_performances.get(benchmark, random_performance_no_dash)
    random_performance_ref_benchmark = random_performances.get(ref_benchmark, random_performance_benchmark)
    metric_no_dash = default_metric_map.get(benchmark[:benchmark.index('_')], None)
    metric = default_metric_map.get(benchmark, metric_no_dash)
    metric_reference_benchmark = default_metric_map.get(ref_benchmark, metric)
    test = ConStat(
        add_extended_models=add_extended_models,
        n_bootstrap=n_bootstrap,
        random_performance=(random_performance_benchmark, random_performance_ref_benchmark),
        p_value_delta=0
    )
    no_cont_benchmark = benchmark[:benchmark.index('_')] + '_no_cont'
    results = []
    for model in tqdm(to_test_models):
        ref_models_here = ref_models[:]
        if model in ref_models:
            ref_models_here.remove(model)
            ref_models_here += extra_ref_models
        try:
            add_no_cont_results_test = 'contamination-models-' not in model
            test_result = perform_test(
                model, benchmark, ref_benchmark, test, ref_models, 
                base_path, metric, metric_reference_benchmark, add_no_cont_results=add_no_cont_results_test, 
            )
        except Exception as e:
            logger.warning(f"{e} for {model} at {benchmark} with {ref_benchmark}")
            continue
        test_result['model'] = model
        if add_no_cont_result:
            result_no_cont = load_result(
                base_path, model, no_cont_benchmark, metric
            )
            test_result['no_cont'] = np.mean(result_no_cont)
            test_result['no_cont_std'] = np.sqrt((1 - test_result['no_cont']) * test_result['no_cont'] / len(result_no_cont))
        results.append(test_result)

    # convert to dataframe
    df = pd.DataFrame(results)
    return df

def visualize(benchmark, ref_benchmark, ref_models, to_test_models, 
                 add_extended_models=True, n_bootstrap=1000, 
                 save_path=None, xlabel='Reference Performance', ylabel='Benchmark Performance'):
    """
    Visualizes the benchmark performance of reference models and test models along with the hardness correction function.

    Args:
        benchmark (str): The benchmark name.
        ref_benchmark (str): The reference benchmark name.
        ref_models (list): A list of reference models.
        to_test_models (list): A list of test models.
        add_extended_models (bool, optional): Whether to add extended models. Defaults to True.
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 1000.
        save_path (str, optional): The path to save the visualization. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Reference Performance'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Benchmark Performance'.

    Returns:
        matplotlib.figure.Figure: The figure object.
        matplotlib.axes.Axes: The axes object.
    """
    logger.info(benchmark + ref_benchmark)
    
    random_performance_no_dash = random_performances.get(benchmark[:benchmark.index('_')], None)
    random_performance_benchmark = random_performances.get(benchmark, random_performance_no_dash)
    random_performance_ref_benchmark = random_performances.get(ref_benchmark, random_performance_benchmark)
    metric_no_dash = default_metric_map.get(benchmark[:benchmark.index('_')], None)
    metric = default_metric_map.get(benchmark, metric_no_dash)
    metric_reference_benchmark = default_metric_map.get(ref_benchmark, metric)
    test = ConStat(
        add_extended_models=add_extended_models,
        n_bootstrap=n_bootstrap,
        random_performance=(random_performance_benchmark, random_performance_ref_benchmark),
        p_value_delta=0
    )
    results = []
    for i, model in tqdm(enumerate(ref_models + to_test_models)):
        ref_models_here = ref_models[:]
        if model in ref_models:
            ref_models_here.remove(model)
        test_here = None
        if i >= len(ref_models) + len(to_test_models) - 1:
            test_here = test
        try:
            add_no_cont_results_test = 'contamination-models-' not in model
            test_result = perform_test(
                model, benchmark, ref_benchmark, test_here, ref_models, 
                base_path, metric, metric_reference_benchmark, return_functions=True,
                add_no_cont_results=add_no_cont_results_test
            )
        except Exception as e:
            logger.warning(f"{e} for {model} at {benchmark} with {ref_benchmark}")
            continue
        test_result['model'] = model
        results.append(test_result)
    # convert to dataframe
    df = pd.DataFrame(results)
    df['is_ref_model'] = df['model'].apply(lambda x: x in ref_models)
    functions = df['functions'].iloc[-1]
    min_score_ref = min(df['score_model_ref'])
    max_score_ref = max(df['score_model_ref'])
    x = np.linspace(max(min_score_ref - 0.1, 0), min(max_score_ref + 0.1, 1), 100)
    estimates = []
    estimates_025 = []
    estimates_975 = []
    for x_ in x:
        estimates_functions = [min(max(float(f(x_)), 0), 1) for f in functions]
        estimates.append(np.mean(estimates_functions))
        estimates_025.append(np.percentile(estimates_functions, 2.5))
        estimates_975.append(np.percentile(estimates_functions, 97.5))
    # choose colorblind color palette
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(dpi=120)
    sns.lineplot(x=x, y=estimates, ax=ax, label='Fit', color=palette[0], sort=False)
    ax.fill_between(x, estimates_025, estimates_975, alpha=0.2, color=palette[0], label='95% CI Fit')
    df_ref = df[df['is_ref_model']]
    df_not_ref = df[~df['is_ref_model']]
    ax.errorbar(df_ref['score_model_ref'], df_ref['score_model'], 
                xerr=df_ref['score_model_ref_std'], yerr=df_ref['score_model_std'], 
                fmt='o', label=r'$M_{\mathrm{ref}, i}$', color=palette[1], markersize=6)
    ax.errorbar(df_not_ref['score_model_ref'], df_not_ref['score_model'],
                xerr=df_not_ref['score_model_ref_std'], yerr=df_not_ref['score_model_std'],
                fmt='o', label='Models', color=palette[2], markersize=5)

    ax.set_xlim(max(min(df['score_model_ref']) - 0.03, -0.01), min(max(df['score_model_ref']) + 0.03, 1.01))
    ax.set_xlabel(xlabel, fontsize=21)
    ax.set_ylabel(ylabel, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)
    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # set grey background color
    ax.set_facecolor((0.95, 0.95, 0.95))
    # set legend with high fontsize
    ax.legend(fontsize=18)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


if __name__ == '__main__':
    benchmarks = [
        'gsm8k', 'arc', 'mmlu', 'hellaswag'
    ]

    import argparse

    parser = argparse.ArgumentParser(description='Run detection tests')
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--normal-type', type=str, default='normal')
    parser.add_argument('--username', type=str, default='anonymous')

    args = parser.parse_args()

    os.makedirs('tables', exist_ok=True)

    n_bootstrap = 10000
    benchmark = args.benchmark

    if args.type == 'reference':
        other_benchmark = default_reference_benchmark[benchmark]
    else:
        other_benchmark = benchmark + f'_{args.type}'

    res_rephrase = perform_full_test(benchmark + f'_{args.normal_type}', other_benchmark, ref_models, 
                                        ref_models + test_models + get_contaminated_models(benchmark, args.username), add_no_cont_result=True, 
                                        n_bootstrap=n_bootstrap)
    res_rephrase.to_csv(f'tables/{benchmark}_{args.type}.csv')

    os.makedirs('figures', exist_ok=True)
    visualize(benchmark + f'_{args.normal_type}', other_benchmark, ref_models,
                get_contaminated_models(benchmark, args.username), add_extended_models=True, n_bootstrap=n_bootstrap, 
                save_path=f'figures/{benchmark}_{args.type}_contaminated.pdf')
    visualize(benchmark + f'_{args.normal_type}', other_benchmark, ref_models,
                test_models, add_extended_models=True, 
                n_bootstrap=n_bootstrap, save_path=f'figures/{benchmark}_{args.type}_test.pdf')
