import numpy as np
from tqdm import tqdm
from constat import ConStat, NormalizedDiffTest, MeanScoreTest
import pandas as pd

def prob_model_correct(model_value, sample_value):
    """
    Calculate the probability that the model value is correct given the sample value.

    Parameters:
    model_value (float): The value predicted by the model.
    sample_value (float): The observed value from the sample.

    Returns:
    float: The probability that the model value is correct.

    """
    return min(1, np.exp(- sample_value / model_value))

def sample_complexity(mu, n=1):
    """
    Generate an array of size n with repeated values of mu.

    Parameters:
    mu (float): The value to be repeated.
    n (int, optional): The size of the array. Defaults to 1.

    Returns:
    numpy.ndarray: An array of size n with repeated values of mu.
    """
    return np.repeat(mu, n)

def accuracy_model(model_value, dataset):
    """
    Calculate the accuracies of a model on a given dataset.

    Parameters:
    model_value (float): The value of the model.
    dataset (list): A list of datasets.

    Returns:
    numpy.ndarray: An array of binary values representing the accuracies of the model on the dataset.
    """
    sample_complexities = [sample_complexity(sample)[0] for sample in dataset]
    prob_correct = [prob_model_correct(model_value, sample) for sample in sample_complexities]
    accuracies = np.array([np.random.binomial(1, prob_correct[i]) for i in range(len(prob_correct))])
    return accuracies

def accuracy_all_models(models, dataset):
    """
    Calculate the accuracy of all models on the given dataset.

    Parameters:
    - models (list): A list of models to evaluate.
    - dataset (numpy.ndarray): The dataset to evaluate the models on.

    Returns:
    - numpy.ndarray: An array of accuracies for each model.
    """
    return np.array([accuracy_model(model, dataset) for model in models])

def main(distribution_models, distribution_dataset_1, distribution_dataset_2, m, n_1, n_2, 
         check_model_score_1, check_model_score_2, std_ref_models=0):
    """
    Run the simulation with the given parameters.

    Args:
        distribution_models (function): A function that generates the distribution of models.
        distribution_dataset_1 (function): A function that generates the distribution of dataset 1.
        distribution_dataset_2 (function): A function that generates the distribution of dataset 2.
        m (int): The number of models to generate.
        n_1 (int): The number of samples in dataset 1.
        n_2 (int): The number of samples in dataset 2.
        check_model_score_1 (float): The model score for dataset 1.
        check_model_score_2 (float): The model score for dataset 2.
        std_ref_models (float, optional): The standard deviation for reference models. Defaults to 0.

    Returns:
        tuple: A tuple containing the following elements:
            - accuracies_1 (list): List of accuracies on each sample for all models on dataset 1.
            - accuracies_2 (list): List of accuracies on each sample for all models on dataset 2.
            - accuracies_model_1 (float): Accuracy of the model  on each sample specified by check_model_score_1 on dataset 1.
            - accuracies_model_2 (float): Accuracy of the model  on each sample  specified by check_model_score_2 on dataset 2.
            - actual_accuracies_model_1 (float): Actual Accuracy of the model specified by check_model_score_2 on dataset 1.
            - models (list): List of generated models.
            - dataset_1 (list): List of samples in dataset 1.
            - dataset_2 (list): List of samples in dataset 2.
    """
    dataset_1 = [distribution_dataset_1() for _ in range(n_1)]
    dataset_2 = [distribution_dataset_2() for _ in range(n_2)]
    models = [distribution_models() for _ in range(m)]
    accuracies_1 = accuracy_all_models(models, dataset_1)
    if std_ref_models > 0:
        models = [model + np.random.normal(0, std_ref_models) for model in models]
    accuracies_2 = accuracy_all_models(models, dataset_2)
    accuracies_model_1 = accuracy_model(check_model_score_1, dataset_1)
    accuracies_model_2 = accuracy_model(check_model_score_2, dataset_2)
    actual_accuracies_model_1 = accuracy_model(check_model_score_2, dataset_1)
    return accuracies_1, accuracies_2, accuracies_model_1, accuracies_model_2, actual_accuracies_model_1, models, dataset_1, dataset_2

def multinormal(mu1, sigma1, mu2, sigma2):
    """
    Generate a random number from a normal distribution with two possible means and standard deviations.

    Parameters:
    mu1 (float): Mean of the first normal distribution.
    sigma1 (float): Standard deviation of the first normal distribution.
    mu2 (float): Mean of the second normal distribution.
    sigma2 (float): Standard deviation of the second normal distribution.

    Returns:
    float: Random number generated from one of the two normal distributions.
    """
    r = np.random.uniform()
    if r < 0.5:
        return np.random.normal(mu1, sigma1)
    else:
        return np.random.normal(mu2, sigma2)

def get_actual_score(dataset_distribution, model_score, n_samples=100000):
    """
    Calculate the actual score of a model by generating a dataset using the given dataset distribution
    and computing the average accuracy of the model on the generated dataset.

    Parameters:
    - dataset_distribution: a function that generates a sample from the dataset distribution
    - model_score: the scoring function for the model
    - n_samples: the number of samples to generate from the dataset distribution (default: 100000)

    Returns:
    - The average accuracy of the model on the generated dataset
    """
    dataset = [dataset_distribution() for _ in range(n_samples)]
    accuracies = accuracy_model(model_score, dataset)
    return np.mean(accuracies)

def test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2,
                  check_model_score_1, check_model_score_2, n_bootstrap=1000, n_repetitions=1000, 
                  std_ref_models=0):
    """
    Run a simulation scenario to test the performance of different models.

    Args:
        distribution_dataset_1 (array-like): The distribution of dataset 1.
        distribution_dataset_2 (array-like): The distribution of dataset 2.
        distribution_models (array-like): The distribution of models.
        m (int): The number of models.
        n_1 (int): The number of samples in dataset 1.
        n_2 (int): The number of samples in dataset 2.
        check_model_score_1 (callable): A function to check the score of model 1.
        check_model_score_2 (callable): A function to check the score of model 2.
        n_bootstrap (int, optional): The number of bootstrap samples. Defaults to 1000.
        n_repetitions (int, optional): The number of repetitions. Defaults to 1000.
        std_ref_models (int, optional): The standard reference models. Defaults to 0.

    Returns:
        list: A list of dictionaries containing the results of the simulation.
    """
    results = []
    spline_test = ConStat(n_bootstrap=n_bootstrap)
    spline_test_no = ConStat(n_bootstrap=n_bootstrap, bootstrap_models=False)
    spline_test_no_additional = ConStat(n_bootstrap=n_bootstrap, add_extended_models=False)
    spline_no_sorting = ConStat(n_bootstrap=n_bootstrap, sort=False)
    mean_test = MeanScoreTest(n_bootstrap=n_bootstrap)
    normalized_test = NormalizedDiffTest(n_bootstrap=int(np.sqrt(n_bootstrap)), n_model_bootstrap=int(np.sqrt(n_bootstrap)))
    score = get_actual_score(distribution_dataset_1, check_model_score_1)
    actual_score = get_actual_score(distribution_dataset_1, check_model_score_2)
    for i in tqdm(range(n_repetitions)):
        result = dict()
        np.random.seed(i)
        accuracies_1, accuracies_2, accuracies_model_1, accuracies_model_2, _, _, _, _ = main(distribution_models, distribution_dataset_1, distribution_dataset_2, m, n_1, n_2, check_model_score_1, check_model_score_2, 
                                                                                                std_ref_models=std_ref_models)
        result[r'constat'] = spline_test.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result[r'constat_no_bootstrap'] = spline_test_no.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result[r'constat_no_random'] = spline_test_no_additional.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result[r'constat_no_sort'] = spline_no_sorting.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result['mean'] = mean_test.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result['normal'] = normalized_test.test(accuracies_model_1, accuracies_model_2, accuracies_1, accuracies_2)['p_value']
        result['score'] = score
        result['actual_score'] = actual_score
        results.append(result)
    
    return results

base_path = 'data/simulation/'


def run_easy_scenario():
    def distribution_models():
        return np.random.normal(1.0, 0.3)
    
    def distribution_dataset_1():
        return np.random.normal(0.8, 0.3)
    
    def distribution_dataset_2():
        return np.random.normal(0.8, 0.3)
    
    n_1, n_2 = 1000, 1000
    m = 20
    check_model_score_1 = 1
    check_model_score_2 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/easy_scenario_uncontamination.csv", index=False)

    check_model_score_1 = 1.1

    p_vals_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2,
                            check_model_score_1, check_model_score_2)
    df = pd.DataFrame(p_vals_contamination)
    df.to_csv(f"{base_path}/easy_scenario_contamination.csv", index=False)



def run_different_distribution():
    def distribution_models():
        return np.random.normal(1.0, 0.3)
    
    def distribution_dataset_1():
        return np.random.normal(0.4, 0.3)
    
    def distribution_dataset_2():
        return np.random.normal(0.8, 0.2)
    
    n_1, n_2 = 1000, 1000
    m = 20
    check_model_score_1 = 1
    check_model_score_2 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/different_distribution_uncontamination.csv", index=False)


def run_relative_fails():
    distribution_models = lambda: np.random.normal(0.6, 0.2)
    distribution_dataset_1 = lambda: multinormal(0.8, 0.1, 1.4, 0.1)
    # binomial
    distribution_dataset_2 = lambda: multinormal(0.3, 0.1, 1.0, 0.1)
    m = 20
    n_1 = 1000
    n_2 = 1000
    check_model_score_2 = 1
    check_model_score_1 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/relative_no_uncontamination.csv", index=False)


def run_no_sort_fails():
    def distribution_models():
        return np.random.normal(0.8, 0.1)
        
    def distribution_dataset_1():
        return np.random.normal(1, 0.4)

    def distribution_dataset_2():
        return np.random.normal(1, 0.4)

    n_1, n_2 = 1000, 1000
    m = 20
    check_model_score_1 = 1
    check_model_score_2 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2, std_ref_models=0.05)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/no_sort_uncontamination.csv", index=False)


def run_no_random_fails():
    def distribution_models():
        return np.random.normal(4, 1)
    
    def distribution_dataset_1():
        return multinormal(4, 0.2, 0.8, 0.8)

    def distribution_dataset_2():
        return np.random.normal(0.8, 0.8)

    n_1, n_2 = 1000, 1000
    m = 5
    check_model_score_1 = 1
    check_model_score_2 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2, std_ref_models=0.05)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/no_random_uncontamination.csv", index=False)


def run_no_bootstrap_fails():
    distribution_models = lambda: np.random.normal(0.6, 1.0)
    distribution_dataset_1 = lambda: multinormal(0.8, 0.1, 1.4, 0.1)
    # binomial
    distribution_dataset_2 = lambda: multinormal(0.3, 0.1, 1.0, 0.1)
    m = 20
    n_1 = 1000
    n_2 = 1000
    check_model_score_2 = 1
    check_model_score_1 = 1

    p_vals_no_contamination = test_scenario(distribution_dataset_1, distribution_dataset_2, distribution_models, m, n_1, n_2, 
                           check_model_score_1, check_model_score_2, std_ref_models=0.1)
    df = pd.DataFrame(p_vals_no_contamination)
    df.to_csv(f"{base_path}/no_bootstrap_uncontamination.csv", index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str)

    args = parser.parse_args()

    if args.setting == 'easy':
        run_easy_scenario()
    elif args.setting == 'different':
        run_different_distribution()
    elif args.setting == 'relative':
        run_relative_fails()
    elif args.setting == 'no_sort':
        run_no_sort_fails()
    elif args.setting == 'no_random':
        run_no_random_fails()
    elif args.setting == 'no_bootstrap':
        run_no_bootstrap_fails()
