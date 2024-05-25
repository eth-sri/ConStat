import os
import pandas as pd
import re
import numpy as np
from loguru import logger

def load_result(base_path_eval, model, benchmark_name, metric):
    """
    Load and process results from a specified benchmark for a given model.

    Args:
        base_path_eval (str): The base path for evaluation results.
        model (str): The name of the model.
        benchmark_name (str): The name of the benchmark.
        metric (str): The name of the metric to extract from the results.

    Returns:
        numpy.ndarray: The processed results for the specified benchmark and model.
    """
    filename = None
    path = None
    if os.path.exists(os.path.join(base_path_eval, model, benchmark_name)):
        for file in os.listdir(os.path.join(base_path_eval, model, benchmark_name)):
            if file.endswith('jsonl') or file.endswith('.csv'):
                filename = file
                break
        if filename is not None:
            path = os.path.join(base_path_eval, model, benchmark_name, filename)
    if path is None or not os.path.isfile(path):
        logger.warning(f'Not able to read {benchmark_name} for {model} from {path}')
        return
    if path.endswith('.csv'):
        data = pd.read_csv(path)
    else:
        data = pd.read_json(path, lines=False)
    # remove duplicated rows
    data = data.drop_duplicates(subset=['doc_id'])
    results = np.array(data[metric])
    return results

def load_results(benchmark_name, metric, 
                 base_path_eval, model_name, ref_models):
    """
    Load results for a given benchmark and metric.

    Args:
        benchmark_name (str): The name of the benchmark.
        metric (str): The metric to evaluate the results.
        base_path_eval (str): The base path where the evaluation results are stored.
        model_name (str): The name of the model to load results for.
        ref_models (list): A list of reference models to compare against.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the reference results
               for each reference model, and the second dictionary contains the results for the model_name.
    """
    reference_results = dict()
    min_length = None
    for model in ref_models:
        # find the jsonl file in the folder os.path.join(base_path_eval, model, benchmark_name)
        results = load_result(base_path_eval, model, benchmark_name, metric)
        if results is not None:
            reference_results[model] = results
            if min_length is None or len(results) < min_length:
                min_length = len(results)
    results = load_result(base_path_eval, model_name, benchmark_name, metric)
    if min_length is None or len(results) < min_length:
        min_length = len(results)
    if min_length is not None:
        for model in reference_results:
            reference_results[model] = reference_results[model][:min_length]
        results = results[:min_length]
    return reference_results, results


def prepare_ref_results(scores_ref_models, scores_ref_models_ref_data):
    """
    Prepare reference results for contamination detection.

    This function takes in two dictionaries, `scores_ref_models` and `scores_ref_models_ref_data`,
    and extracts the corresponding scores for normal and not normal data from each dictionary.
    The extracted scores are then converted into numpy arrays and returned.

    Parameters:
    scores_ref_models (dict): A dictionary containing scores for normal data.
    scores_ref_models_ref_data (dict): A dictionary containing scores for not normal data.

    Returns:
    normal_here_ref (numpy.ndarray): An array of scores for normal data.
    not_normal_here_ref (numpy.ndarray): An array of scores for not normal data.
    """
    normal_here_ref = []
    not_normal_here_ref = []

    for model in scores_ref_models:
        if model in scores_ref_models_ref_data:
            normal_here_ref.append(scores_ref_models[model])
            not_normal_here_ref.append(scores_ref_models_ref_data[model])
    
    normal_here_ref = np.array(normal_here_ref)
    not_normal_here_ref = np.array(not_normal_here_ref)
    return normal_here_ref, not_normal_here_ref