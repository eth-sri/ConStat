# ConStat: Performance-Based Contamination Detection in Large Language Models

This repository contains the code necessary to reproduce the results from the paper *ConStat: Performance-Based Contamination Detection in Large Language Models* and use ConStat for your projects. This document provides instructions for installation, usage, and reproducing results, along with details on the custom forks and modifications made to external repositories.

ConStat is a tool designed for detecting performance-based contamination in large language models. This project aims to provide an efficient and effective method for identifying and measuring contamination in language models by analyzing performance metrics. For further details, please refer to our paper.

## Table of Contents
1. [Installation](#installation)
2. [Using ConStat](#using-constat)
    - [Toy Example](#toy-example)
    - [Real-World Contamination Test](#real-world-contamination-test)
        - [Generating your reference benchmark](#generating-your-reference-benchmark)
        - [Evaluating your model](#evaluating-your-model)
        - [Running ConStat](#running-constat)
3. [Reproduction](#reproduction)
4. [Code Structure](#code-structure)
5. [Custom Forks](#custom-forks)


## Installation
You can install the code in this repository by following these steps:

1. **Install Conda**: If you haven't already, install [Conda](https://docs.conda.io/projects/miniconda/en/latest/).
2. **Set up the environment**:
    ```bash
    conda create -n constat python=3.10
    conda activate constat
    python -m pip install -e .
    ```
3. **Install additional requirements**:
    ```bash
    python -m pip install -r requirements.txt
    cd lm-evaluation-harness
    python -m pip install -e .
    cd ..
    ```
The additional requirements are only necessary if you want to reproduce our results or use this repo beyond the use that is shown in the toy example below.

## Using ConStat

### Toy Example

The `ConStat` class is the primary interface for running contamination tests. Here's a simple example using randomly generated data with NumPy:

```python
from constat import ConStat
import numpy as np

constat = ConStat(
    n_bootstrap=10000, # Number of bootstraps to use. We suggest a high number, even though it might take a while
    p_value_delta=0, # the \delta value from our paper for which to report the p-value in a test
    random_performance=(0,0) # ConStat will add a random model at this point. If you have a multiple-choice benchmark, this should become something like (0.25,0.25)
)

accuracies_model = np.random.randint(0, 2, 1000) # For each sample in the actual benchmark, the accuracy of your model on that sample. This can also be any other metric you want to use.
accuracies_model_reference = np.random.randint(0, 2, 1000) # For each sample  in the reference benchmark, the accuracy of your model on that sample

accuracies = np.random.randint(0, 2, (10, 1000)) # the accuracies of your reference models on the benchmark
accuracies_reference = np.random.randint(0, 2, (10, 1000)) # the accuracies of your reference models on the reference benchmark

result = constat.test(accuracies_model, accuracies_model_reference, accuracies, accuracies_reference) # perform the test
print(result)
# {'p_value': 0.4528, # p-value for the test
#  'estimated_contamination': 0.49376129894065124, # estimated performance of the model on the benchmark if it weren't contaminated
#  'estimated_contamination_025': 0.4616516952656119, # 2.5% lower bound of the estimated performance
#  'estimated_contamination_975': 0.5247609019013911, # 97.5% upper bound of the estimated performance
#  'estimated_contamination_std': 0.015812938662854926, # std of the estimated performance
#  'delta': 0.0023555010593487333, # Estimated \delta
#  'delta_std': 0.0223297155214044,# Estimated std of \delta
#  'min_delta_095': -0.034313983164635666} # Estimated 95%th lower bound of delta
```

### Real-World Contamination Test

You can apply ConStat to practical contamination detection problems. This involves generating reference benchmarks, evaluating models, and running the ConStat test.

You do not have to use this repo beyond the ConStat test to apply ConStat. You can use your benchmark, reference benchmark, and evaluation tools.

However, you can use this repo further to test for contamination on the four benchmarks described in the paper or apply it to your own benchmark. This procedure consists of several steps. First, you generate the reference benchmark, then you evaluate the models on your benchmark and reference benchmark and finally, you run ConStat. If you are using one of our benchmarks, you can download both the model outputs and the generated benchmarks by following the steps described at the beginning of the [reproduction section](#reproduction). You can then skip the next step and only have to evaluate your model (and not the reference models) in the section after.

#### Generating your reference benchmark

You can use [`generate_rephrase.py`](scripts/generate_rephrase.py) and [`generate_synthetic.py`](scripts/generate_synthetic.py) to generate your own synthetic benchmarks. These files contain the code for our rephrase and synthetic benchmark generation. You will need to store your OpenAI API key in the environment variables as `OPENAI_API_KEY`.

For rephrasing you generally need (1) a function that calls an LLM on the question of your benchmark and (2) a function that properly parses the response of that LLM. The [`generate_rephrase.py`](scripts/generate_rephrase.py) file contains the `generate` function that takes as input your input prompts, your system prompt, where to store the responses, and whether or not the benchmark is a multiple-choice benchmark. It assumes the system prompt contains formatting instructions for the LLM that say to start its question with the string `### Question`. For MMLU, we use the following system prompt:
```
Significantly rephrase the given question and options, but make sure that all possible options still have the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response.

Format your reply as:
### Question
[New rephrased question]
```
We format each question to contain the question in the same format as specified in this system prompt. Note that `generate` also parses the responses with the `parse_response` function. If parsing fails, you will get an error message describing the problem. However, note that the responses of the LLM have already been stored by then and if you run it again, they will be reloaded (and you will not incur extra costs).

For synthetic benchmark generation, you generally need additional functionality. First, we need to ensure that we give the LLM few-shot samples and that the LLM 'thinks' it wrote these few-shot samples (e.g. by passing them in the assistant field). This is important to ensure the synthetic data is realistic. The `generate` function in the [`generate_synthetic.py`](scripts/generate_synthetic.py) file makes this more complicated behavior possible. For MMLU, we give this function a Pandas DataFrame (`data`) that contains input (question + options as formatted before) and output (the correct option fully repeated) columns. It also contains the user_input column which contains the category of the question (we will soon explain why this is necessary). We use the following system prompt:

```
Your task is to generate new and original benchmark samples from the MMLU benchmark. The user will instruct you to generate from a certain subject. The generated sample must follow the same format, difficulty and styling as questions from this benchmark.

Format your reply as:
### Question
[Question for the sample]
### Answer
[Answer of the sample]
```

We now additionally need to specify the user prompt. We use a function that maps the category from the user_input column to a prompt that specifies for which category the LLM should generate an answer:
```
Generate a sample on the following subject: {subject}
```
We also specify the `random_choices` as the list of possible categories for which we want to generate samples for. Note that we simply use the user_input column of the `data` object for this, without deduplication (this ensures the same distribution over categories). Finally, we specify the number of samples to generate (1000) and the amount of few-shot samples to put in each field (3).

Further information and documentation on these functions can be found in their respective files.

#### Evaluating your model
We use the LM Evaluation Harness for evaluation. We refer to their README and documentation to generate your config files. However, you will likely need our [`csv_loader.py`](lm-evaluation-harness/csv_loader.py) which can load benchmarks from a csv file. For benchmarks that we did not include, you will likely have to extend this file  in the `_generate_examples` function with some post-processing. For example, for MMLU we need to apply `literal_eval` on the choices column to fully extract the columns. You then need to create a config file for your (reference) benchmark. Config files for reference benchmarks are very similar to their benchmark. You can probably find inspiration from our config files in their corresponding subfolders: [`gsm8k`](lm-evaluation-harness/lm_eval/tasks/gsm8k), [`mmlu`](lm-evaluation-harness/lm_eval/tasks/mmlu), [`arc`](lm-evaluation-harness/lm_eval/tasks/arc), [`hellaswag`](lm-evaluation-harness/lm_eval/tasks/hellaswag). 

You can then use the harness with the parameters explained in their README to evaluate your models. Note that you have to supply the parameter `--log-samples`. These logs are necessary to run ConStat. You should also specify the output parameter. If you want to use constat to extract the results, it is advised to store them in the following format: `[BASE PATH]/[MODEL NAME]/[TASK]` where the base path is whatever you want, the model name is the name of the model and the task is the task you ran.

For example, to run GSM8k for sample-specific contamination, you would need to evaluate your model on the `gsm8k_normal`, `gsm8k_no_cont`, and `gsm8k_synthetic` benchmarks (the reason you have to run three, is because we split the standard part of the benchmark in two for the evaluation of our results). Furthermore, you should run

```bash
python add_flexible_extract.py
```

which adds the `flexible_extract` metric to your model output. 

#### Running ConStat
After this, you can use ConStat as follows for the example of the gsm8k benchmark:

```python
from constat import ConStat, perform_test

constat = ConStat(n_bootstrap=10000, p_value_delta=0, random_performance=(0,0))
test = perform_test(
    model_name='YOUR MODEL NAME',
    benchmark='gsm8k_normal',
    ref_benchmark='gsm8k_synthetic',
    ref_model_names=[
        "microsoft/phi-2",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Llama-2-70b-chat-hf",
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
    ],
    add_no_cont_results=True, # this adds the no_cont benchmark to the normal benchmark
    base_path='lm-evaluation-harness/output', # base path to the model output, the exact form can depend on where you run this test from
    metric='flexible_extract',
    metric_reference_benchmark='flexible_extract'
)
print(test)
```

If you do not use our benchmarks, you can use the following code to extract the results of a model in the format that ConStat expects. You will need to do this for all models and both your benchmark and reference benchmark and then concatenate the results as shown before.

```python
from constat import load_result

result = load_result(
    base_path_eval, # what is the base path where you stored your results?
    model, # name of the model 
    benchmark_name, # name of the benchmark
    metric # name of the metric you wish to extract
)
```
This will return an array where each element is the score of the model on that sample.

## Reproduction
We aim to make ConStat as easily applicable as possible. For this purpose, we want to share the generated synthetic benchmarks and the raw results from the models we evaluated using ConStat. By publishing these results, you can use ConStat for the benchmarks we investigated by only running the evaluation on the model you want to test. However, publishing these benchmarks also includes a risk that they get included in the training datasets of future models. For this reason, we have created a password-protected zip file that you can [download](https://files.sri.inf.ethz.ch/constat/data_contamination.zip) (md5 hash: 1aa46b2b56402d4a8aacbc930f3882ee). The password for this file is `contaminationisbad`. We note that this procedure only protects against honest-but-negligent model providers. If you suspect your model provider is malicious, you should regenerate the data yourself as they might have trained on these benchmarks. Furthermore, while we urge people to not publish these results in other places, we cannot guarantee that nobody will do so. The downloaded benchmarks should be placed in the `data/contamination` folder.

You can also [download the raw results for each evaluated model](https://files.sri.inf.ethz.ch/constat/compressed_output.zip) in a compressed format (md5 hash: 10f815b7db3ac159f490c71cb931c154). The downloaded model outputs should be placed in the `lm-evaluation-harness/output` folder.

We include the raw results for our paper in the [`tables/`](tables/) folder. You can run the notebooks in the [`notebooks/`](notebooks/) folder to extract the results as they were presented in the paper. If you want to obtain the results from our baselines without rerunning them, you should download the raw results: extract [this folder](https://files.sri.inf.ethz.ch/constat/baselines.zip) and place it in the `data` folder. Also extract [this folder](https://files.sri.inf.ethz.ch/constat/code-contamination-output.zip) and place it in `code-contamination-detection`.

Furthermore, the [`figures/`](figures/) folder contains visual representations of the tests as presented in Fig 1 of our paper. Note that this visual representations are only approximations of the actual test.

To produce the raw results yourself, please follow the following steps. We assume you have an OpenAI API key and a Together API Key. You should also have a machine with a GPU that has at least 80GB of VRAM. Furthermore, note that each of the following commands could take a couple of days to a week to compute. Add your API keys to the environment variables by running the following commands in a terminal:

```bash
export OPENAI_API_KEY=[YOUR OPENAI KEY]
export TOGETHER_API_KEY=[YOUR Together KEY]
```

To run some models (e.g. Llama 2), you will need to be logged into your Huggingface account and store your read and write token:
```bash
huggingface-cli login # follow the instructions after this
```
If you rerun the finetuning process, the code will automatically upload the finetuned models to private repos in your account. They will not be stored locally. In all the following commands, please replace `USERNAME` with your Huggingface username.

Note that running the following code will run the file `scripts/remove_models_huggingface.py` several times. This file removes some of the cache in the HuggingFace cache to avoid issues with storage space. If you want to use this yourself you should adjust the base path in the file to point to your Huggingface cache.

Then, ensure you are in the main folder of this repository and run the following commands. These steps can be skipped if you have downloaded our raw data (described at the beginning of the [reproduction section](#reproduction)).

```bash
python scripts/preprocessing.py # Downloads the 4 benchmarks we use
python scripts/generate_rephrase.py # Uses OpenAI to generate rephrases of the various benchmarks
python scripts/generate_synthetic.py # Uses OpenAI to generate synthetic samples of the various benchmarks
```

You can then finetune the models:

```bash
bash scripts/finetune.sh # Finetunes the models on the benchmark data
```


After this, you need to run the evaluation for all models. This step can also be skipped if you have downloaded our raw model outputs (described at the beginning of the [reproduction section](#reproduction)).
```bash
cd lm-evaluation-harness # goes to the LM Eval Harness
bash main_small.sh # run smaller models that can be run locally
bash main_controlled.sh USERNAME # run the trained models
bash main_large.sh # run the larger models through the Together API
```

Now, we just need to run the detection and their baselines. Note, if you have downloaded the raw outputs from our datasets, you should use the username 'JasperDekoninck'.

```bash
cd .. ## goes back to the main folder
bash scripts/detection.sh USERNAME # runs ConStat
bash scripts/evaluate_baselines.sh USERNAME # runs most baselines
cd code-contamination-detection USERNAME
bash bash.sh # run the Shi baseline
```

You can also run the simulation experiments presented in the appendix:

```bash
cd .. ## goes back to the main folder
bash scripts/simulation.sh
```

Now, you can run the notebooks from the [`notebooks/`](notebooks/) folder to extract the results yourself.

## Code Structure

We briefly outline the structure in this repo and then explain our own code files in a bit more depth. The code is divided into the following main folders:

- [`src/constat`](src/constat): Contains the source code for the ConStat statistical test, along with some classes that we used for finetuning and generating benchmarks.
- [`scripts`](scripts): Contains bash scripts and Python files that execute the code necessary to reproduce our experiments. 
- [`notebooks`](notebooks): Contains post-processing notebooks to extract the representations we presented in our paper from the raw results.
- [`tables`](tables): Contains csv files that store the information related to each test that we performed.
- [`figures`](figures): Contains (approximate) visual representations of the tests we performed.
- [`lm-evaluation-harness`](lm-evaluation-harness): Contains a custom fork of the LM Evaluation Harness.
- [`code-contamination-detection`](code-contamination-detection): Contains a custom fork of the Shi baseline we ran in our paper.
- [`data`](data): Depending on which data you downloaded, this will contain some data from the benchmarks, baselines and simulation.

Our code is relatively short, so the structure is very simple. [`src/constat/detection.py`](src/constat/detection.py) contains the most important code, with the tests being there. [`src/constat/load_results.py`](src/constat/load_results.py) contains other important code to load results from the LM Eval Harness. The other files contain the following code:
- [`src/constat/base.py`](src/constat/base.py): Some base class that allows for the storage of a class as a json object.
- [`src/constat/basic_model_loader.py`](src/constat/basic_model_loader.py): Makes it easy to load a range of models from HuggingFace
- [`src/constat/dataset.py`](src/constat/dataset.py): Dataset class used in the finetuning process
- [`src/constat/finetune.py`](src/constat/finetune.py): Finetuning class to finetune our models
- [`src/constat/openai.py`](src/constat/openai.py): A generic class to query the OpenAI API
- [`src/constat/overlap.py`](src/constat/overlap.py): Contains the code for some of our baselines (computing perplexity etc.)
- [`src/constat/preprocessing.py`](src/constat/preprocessing.py): Contains the code to preprocess a dataset to prepare for instruction tuning
- [`src/constat/spline_fit.py`](src/constat/spline_fit.py): An almost exact copy of the `spline_fit` file of SciPy with the exception that the function also returns the regularization constant.
- [`src/constat/utils.py`](src/constat/utils.py): Some minor utils

Our scripts our structured as follows:
- [`scripts/detection.py`](scripts/detection.py): Code to run ConStat on all the results we got from the LM Eval Harness.
- [`scripts/evaluate_baselines.py`](scripts/evaluate_baselines.py): Computes the necessary numbers (e.g. perplexity) for our baselines
- [`scripts/finetune.py`](scripts/finetune.py): Finetunes contaminated models
- [`scripts/generate_rephrase.py`](scripts/generate_rephrase.py): Generates rephrased benchmarks
- [`scripts/generate_synthetic.py`](scripts/generate_synthetic.py): Generates synthetic benchmarks
- [`scripts/prepare.py`](scripts/prepare.py): Preparing code that processes the benchmarks we used in dataframes that only contain an input and output column.
- [`scripts/preprocessing.py`](scripts/preprocessing.py): Downloads the four benchmarks we used in our evaluation.
- [`scripts/remove_models_huggingface.py`](scripts/remove_models_huggingface.py): Removes the HuggingFace cache to clean up space.
- [`scripts/simulation.py`](scripts/simulation.py): Runs the simulation experiments described in our paper

## Custom Forks

This repository contains custom forks of the [LM Eval harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) (v0.4.1) and [code-contamination-detection](https://github.com/swj0419/detect-pretrain-code-contamination/blob/master/src/run.py). We also included the spline fitting function of [SciPy](https://github.com/scipy/scipy) to make it compatible with our use of it. We briefly explain the changes we made to the first two of these repositories:

### LM Eval Harness
- Included [`csv_loader.py`](lm-evaluation-harness/csv_loader.py) which is a file that loads a benchmarks from a csv file. This allowed us to run our own rephrased and synthetic benchmarks more easily and also ensures the same few-shot prompts are used compared to the original benchmark.
- Included several new tasks configurations to run all our variants of each task. 
- Added the [`together`](lm-evaluation-harness/lm_eval/models/together.py) model which allowed us to use the Together API for evaluation.
- Extended the [`HuggingFace model`](lm-evaluation-harness/lm_eval/models/huggingface.py) to enable loading of our finetuned models. The LM Eval Harness had trouble with loading our finetuned models because we did not store the tokenizer with the models. 
- Added various scripts and files to run our code. Noteworthy is the [`add_flexible_extract.py`](lm-evaluation-harness/add_flexible_extract.py) script that adds the `flexible_extract` field to the output information of LM Eval. This is not added in the default version. Furthermore, we also added the [`compress.py`](lm-evaluation-harness/compress.py) script that compresses the raw output of the LM Eval Harness around 100x while still containing all fields we need for our evaluation.

### Code Contamination Detection
Our changes in this repo were less extensive. We simply changed the way models were loaded, since this code base also had trouble loading our finetuned models. We also added the possibility to use Hellaswag.

### Citation

```
@article{dekoninck2024constat,
      title={ConStat: Performance-Based Contamination Detection in Large Language Models}, 
      author={Jasper Dekoninck and Mark Niklas Müller and Martin Vechev},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```