import numpy as np
from scipy.stats import norm
from scipy.odr import *
from .spline_fit import make_smoothing_spline
from .load_results import *

def perform_test(model_name, benchmark, ref_benchmark, test, 
                 ref_model_names, base_path, metric, metric_reference_benchmark,
                 return_functions=False, add_no_cont_results=False):
    """
    Perform a test on a contamination detection model.

    Args:
        model_name (str): The name of the model being tested.
        benchmark (str): The benchmark dataset for testing.
        ref_benchmark (str): The reference benchmark dataset for testing.
        test (object): The test object used for evaluation.
        ref_model_names (list): List of reference model names.
        base_path (str): The base path for loading results.
        metric (str): The metric used for evaluation.
        metric_reference_benchmark (str): The metric used for the reference benchmark.
        return_functions (bool, optional): Whether to return the fitted functions over all bootstraps. Defaults to False.
        add_no_cont_results (bool, optional): Whether to add the results on the uncontaminated part of the benchmark. Defaults to False.

    Returns:
        dict: A dictionary containing the test scores and other evaluation results.
    """
    reference_results, results = load_results(benchmark, metric, base_path, 
                                               model_name, ref_model_names)
    if add_no_cont_results:
        ref_no_cont, res_no_cont = load_results(benchmark.replace('_normal', '_no_cont'), metric, base_path, 
                                               model_name, ref_model_names)
        results = np.concatenate([results, res_no_cont], axis=0)
        reference_results = {key: np.concatenate([value, ref_no_cont[key]], axis=0) for key, value in reference_results.items() if key in ref_no_cont}

    reference_results_ref_data, results_ref_data = load_results(ref_benchmark, 
                                                            metric_reference_benchmark, base_path, 
                                                            model_name, 
                                                            ref_model_names)
                                            
    scores_ref_models, scores_ref_models_ref_data = prepare_ref_results(reference_results, reference_results_ref_data)
    mean_result = np.mean(results)
    mean_result_ref = np.mean(results_ref_data)

    result_dict = {
        'score_model': mean_result,
        'score_model_std': np.std(results) / np.sqrt(len(results)),
        'score_model_ref': mean_result_ref,
        'score_model_ref_std': np.std(results_ref_data) / np.sqrt(len(results_ref_data)),
    }

    if test is not None and len(scores_ref_models) > 0:
        if not return_functions:
            other_dict = test.test(results, results_ref_data, scores_ref_models, scores_ref_models_ref_data)
        else:
            assert isinstance(test, ConStat)
            other_dict = test.return_functions_test(results, results_ref_data, scores_ref_models, scores_ref_models_ref_data)  

        result_dict.update(other_dict)
    return result_dict


def bootstrap(model_matrix_score, n_bootstrap, bootstrap_models=False, bootstrap_last_model=False):
    """
    Perform bootstrap resampling on a model matrix score.

    Parameters:
    - model_matrix_score (numpy.ndarray): The model matrix score.
    - n_bootstrap (int): The number of bootstrap iterations.
    - bootstrap_models (bool, optional): Whether to bootstrap the models. Default is False.
    - bootstrap_last_model (bool, optional): Whether to bootstrap the last model. Default is False.

    Returns:
    - bootstrap_scores (numpy.ndarray): The bootstrap scores.

    """
    np.random.seed(42)
    m, n = model_matrix_score.shape
    if n == 0:
        return np.zeros((n_bootstrap, 1))
    bootstrap_scores = np.zeros((n_bootstrap, m))
    for i in range(n_bootstrap):
        bootstrap_indices = np.random.choice(n, n, replace=True)
        if bootstrap_models:
            if not bootstrap_last_model:
                models_here = np.random.choice(m-1, m-1, replace=True)
                models_here = np.append(models_here, m-1)
            else:
                models_here = np.random.choice(m, m, replace=True)
        else:
            models_here = np.arange(m)
        bootstrap_scores[i] = np.mean(model_matrix_score[models_here][:, bootstrap_indices], axis=1)
    return bootstrap_scores
class StatTest:
    def __init__(self):
        pass

    def test(self, scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data):
        """
        Performs the statistical test and returns the associated p-value, estimated contamination level and 99% quantile of the contamination level

        Args:
            - scores_model (np.array): the scores of the model on the data, shape (n_samples,)
            - scores_model_ref_data (np.array): the scores of the model on the reference data, shape (n_samples_ref_data,)
            - scores_ref_models (np.array): the scores of the reference models on the data, shape (n_ref_models, n_samples)
            - scores_ref_models_ref_data (np.array): the scores of the reference models on the reference data, shape (n_ref_models, n_samples_ref_data)

        Returns: Dict with the following elements:
            - p_value (float): the p-value of the test
            - estimated_contamination (float): the estimated contamination level
            - estimated_contamination_std (float): the standard deviation of the estimated contamination level
            - min_delta_095 (float): the 99% quantile of the contamination level. Contamination level is 99% sure to be higher than this value
        """
        raise NotImplementedError
    
class MeanScoreTest(StatTest):
    def __init__(self, n_bootstrap=1000):
        """
        Initializes a MeanScoreTest object. This object directly compares the mean score of the model on the data to the mean score of the model on the reference data.
        In our paper, we refer to this test as MeanTest

        Parameters:
        - n_bootstrap (int): The number of bootstrap iterations to perform. Default is 1000.
        """
        self.n_bootstrap = n_bootstrap

    def test(self, scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data):
        distribution_model_1 = bootstrap(np.array([scores_model]), self.n_bootstrap)
        distribution_model_2 = bootstrap(np.array([scores_model_ref_data]), self.n_bootstrap)
        mean_1, std_1 = np.mean(distribution_model_1), np.std(distribution_model_1)
        mean_2, std_2 = np.mean(distribution_model_2), np.std(distribution_model_2)
        z = (mean_1 - mean_2) / np.sqrt(std_1**2 + std_2**2)
        # perform one sided test mean_1 <= mean_2 as H_0
        p = 1 - norm.cdf(z)
        return_dict = {
            "p_value": p,
            "estimated_contamination": mean_2,
            "estimated_contamination_std": std_2,
            "min_delta_095": mean_1 - mean_2 + 3 * std_2

        }
        return return_dict


class NormalizedDiffTest(StatTest):
    def __init__(self, n_bootstrap=100, n_model_bootstrap=100):
        """
        Initialize the NormalizedDiffTest object. Normalizes the performance of the model using the mean and std of the reference models.
        Then, compares the normalized performance of the model on the data to the normalized performance of the model on the reference data.
        In our paper, we refer to this test as NormTest.

        Parameters:
        - n_bootstrap (int): The number of bootstrap iterations to perform.
        - n_model_bootstrap (int): The number of bootstrap iterations to perform for each model.

        Returns:
        - None
        """
        self.n_bootstrap = n_bootstrap
        self.n_model_bootstrap = n_model_bootstrap

    def test(self, scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data):
        np.random.seed(42)
        n_ref_models = scores_ref_models.shape[0]
        all_ps = []
        estimations = []
        for index in range(self.n_model_bootstrap):
            models_here = np.random.choice(n_ref_models, n_ref_models, replace=True)
            bootstrapped_accuracies_1 = bootstrap(np.concatenate([scores_ref_models[models_here, :], np.array([scores_model])]), self.n_bootstrap, bootstrap_models=False)
            bootstrapped_accuracies_2 = bootstrap(np.concatenate([scores_ref_models_ref_data[models_here, :], np.array([scores_model_ref_data])]), self.n_bootstrap, bootstrap_models=False)
            mean_bootstrapped_accs_1 = np.mean(bootstrapped_accuracies_1[:, :-1], axis=1)
            std_bootstrapped_accs_1 = np.std(bootstrapped_accuracies_1[:, :-1], axis=1)
            mean_bootstrapped_accs_2 = np.mean(bootstrapped_accuracies_2[:, :-1], axis=1)
            std_bootstrapped_accs_2 = np.std(bootstrapped_accuracies_2[:, :-1], axis=1)
            normalized_performances_1 = (bootstrapped_accuracies_1[:, -1] - mean_bootstrapped_accs_1) / std_bootstrapped_accs_1
            normalized_performances_2 = (bootstrapped_accuracies_2[:, -1] - mean_bootstrapped_accs_2) / std_bootstrapped_accs_2
            p = np.mean([np.mean(normalized_performances_1[i] < normalized_performances_2) for i in range(self.n_bootstrap)])
            all_ps.append(p)
            estimated_actual = normalized_performances_2 * std_bootstrapped_accs_1 + mean_bootstrapped_accs_1
            estimations = estimations + list(estimated_actual)
        mean_estimated_actual, std_estimated_actual = np.mean(estimations), np.std(estimations)
        p = np.mean(all_ps)
        return_dict = {
            "p_value": p,
            "estimated_contamination": mean_estimated_actual,
            "estimated_contamination_std": std_estimated_actual,
            "min_delta_095": 0
        }
        return return_dict

class ConStat(StatTest):
    def __init__(self, n_bootstrap=1000, bootstrap_models=True, add_extended_models=True, 
                     random_performance=(0,0), p_value_delta=0, sort=True):
            """
            Initialize the ConStat object. This is ConStat.
            It fits a spline to the reference models and then uses this spline to estimate the contamination level of the model on the data.

            Parameters:
            - n_bootstrap (int): Number of bootstrap iterations to perform. Default is 1000.
            - bootstrap_models (bool): Flag indicating whether to bootstrap models. Default is True.
            - add_extended_models (bool): Flag indicating whether to add the random model. Default is True.
            - random_performance (tuple): Tuple representing the range of random performance on the benchmark and reference benchmark. Default is (0, 0).
            - p_value_delta (int): \delta value to use for the computation of the p-values. Default is 0.
            - sort (bool): Flag indicating whether to sort the performances. Default is True.
            """
            self.n_bootstrap = n_bootstrap
            self.bootstrap_models = bootstrap_models
            self.add_extended_models = add_extended_models
            self.random_performance = random_performance
            self.p_value_delta = p_value_delta
            self.sort = sort

    def return_functions_test(self, scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data):
        """
        Performs the contamination detection algorithm using the provided scores.

        Args:
            scores_model (numpy.ndarray): Array of scores for the model being tested.
            scores_model_ref_data (numpy.ndarray): Array of scores for the model's reference data.
            scores_ref_models (numpy.ndarray): Array of scores for the reference models.
            scores_ref_models_ref_data (numpy.ndarray): Array of scores for the reference models' reference data.

        Returns:
            dict: A dictionary containing the following keys:
                - "p_value" (float): The p-value calculated from the contamination levels.
                - "estimated_contamination" (float): The estimated contamination level.
                - "estimated_contamination_025" (float): The lower bound of the estimated contamination level (at 95% confidence).
                - "estimated_contamination_975" (float): The upper bound of the estimated contamination level (at 95% confidence).
                - "estimated_contamination_std" (float): The standard deviation of the estimated contamination levels.
                - "delta" (float): The mean difference between the actual scores and the estimated scores.
                - "delta_std" (float): The standard deviation of the differences between the actual scores and the estimated scores.
                - "min_delta_095" (float): The minimum difference between the actual scores and the estimated scores (at 95% confidence).
                - "functions" (list): A list of functions representing the smoothing splines used in the algorithm.
        """
        estimations = []
        ps = []
        functions = []
        np.random.seed(42)

        if scores_ref_models.shape[0] < 5:
            raise ValueError("Not enough reference models. This test requires at least 5 reference models")
        bootstrap_models = self.bootstrap_models
        if scores_ref_models.shape[0] == 5:
            bootstrap_models = False

        i = 0
        while i < self.n_bootstrap:
            i += 1
            indices_1 = np.random.choice(scores_ref_models.shape[1], scores_ref_models.shape[1], replace=True)
            indices_2 = np.random.choice(scores_ref_models_ref_data.shape[1], scores_ref_models_ref_data.shape[1], replace=True)
            if bootstrap_models:
                models_here = np.random.choice(scores_ref_models.shape[0], scores_ref_models.shape[0], replace=True)
                while len(np.unique(models_here)) < 5:
                    models_here = np.random.choice(scores_ref_models.shape[0], scores_ref_models.shape[0], replace=True)
            else:
                models_here = np.arange(scores_ref_models.shape[0])

            mean_scores1 = list(np.mean(scores_ref_models[models_here][:, indices_1], axis=1))
            mean_scores2 = list(np.mean(scores_ref_models_ref_data[models_here][:, indices_2], axis=1))

            if self.add_extended_models:
                random_performance1 = self.random_performance[0] + np.random.normal(0, np.sqrt(self.random_performance[0] * (1 - self.random_performance[0]) / len(indices_1)))
                random_performance2 = self.random_performance[1] + np.random.normal(0, np.sqrt(self.random_performance[1] * (1 - self.random_performance[1]) / len(indices_2)))
                mean_scores1 = [random_performance1] + mean_scores1
                mean_scores2 = [random_performance2] + mean_scores2

            if self.sort:
                sorted_mean1 = np.sort(mean_scores1)
                sorted_mean2 = np.sort(mean_scores2)
            else:
                argsort2 = np.argsort(mean_scores2)
                sorted_mean1 = np.array(mean_scores1)[argsort2]
                sorted_mean2 = np.array(mean_scores2)[argsort2]
           
            same_x_indices = np.where(np.diff(sorted_mean2) == 0)[0]
            sorted_mean1 = np.delete(sorted_mean1, same_x_indices)
            sorted_mean2 = np.delete(sorted_mean2, same_x_indices)
            if len(sorted_mean1) < 5:
                i -= 1
                continue

            weights = [1 for _ in range(sorted_mean1.shape[0])]
            fit, l = make_smoothing_spline(sorted_mean2, sorted_mean1, w=np.array(weights))
            if sorted_mean2[0] != 0:
                sorted_mean1 = np.append(0, sorted_mean1)
                sorted_mean2 = np.append(0, sorted_mean2)
                weights = [1e-8] + weights
            if sorted_mean2[-1] != 1:
                sorted_mean1 = np.append(sorted_mean1, 1)
                sorted_mean2 = np.append(sorted_mean2, 1)
                weights = weights + [1e-8]

            fit, l = make_smoothing_spline(sorted_mean2, sorted_mean1, lam=l, w=np.array(weights))
            estimate = min(max(float(fit(np.mean(scores_model_ref_data[indices_2]))), 0), 1)
            actual = np.mean(scores_model[indices_1])
            estimations.append(estimate)
            ps.append(actual - estimate)
            functions.append(fit)

        mean_ps = np.mean(ps)
        contamination_levels = [float(2 * mean_ps - np.quantile(ps, 1 - p / (len(ps))) >= self.p_value_delta) for p in range(len(ps))]
        if sum(contamination_levels) == 0:
            p_value = 1
        else:
            p_value = float(np.argmax(contamination_levels) / (len(ps)))
        min_delta_095 = float(2 * mean_ps - np.quantile(ps, 0.95))
        estimate = np.mean(estimations)
        estimate_025 = float(2 * estimate - np.quantile(estimations, 0.975))
        estimate_975 = float(2 * estimate - np.quantile(estimations, 0.025))
        return_dict = {
            "p_value": p_value,
            "estimated_contamination": estimate,
            "estimated_contamination_025": estimate_025,
            "estimated_contamination_975": estimate_975,
            "estimated_contamination_std": np.std(estimations),
            "delta": np.mean(ps),
            "delta_std": np.std(ps),
            "min_delta_095": min_delta_095,
            "functions": functions,
        }
        return return_dict

    def test(self, scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data):
        return_dict = self.return_functions_test(scores_model, scores_model_ref_data, scores_ref_models, scores_ref_models_ref_data)
        del return_dict["functions"]
        return return_dict
