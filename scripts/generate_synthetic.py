# Generates the synthetic data for detecting sample-specific contamination

from constat.openai import OpenAIQuery
import pandas as pd
from transformers import set_seed
import numpy as np
import json
import os
import asyncio
from loguru import logger
import re
from thefuzz import fuzz
from ast import literal_eval
from tqdm import tqdm

np.random.seed(2)
set_seed(2)

base_path = 'data/contamination'

def randomize_options(options, answer):
    """
    Randomly changes the order of the options and adjusts the answer index appropriately.

    Parameters:
    options (list): A list of options.
    answer (int): The index of the correct answer.

    Returns:
    tuple: A tuple containing the randomly ordered options and the new index of the correct answer.
    """
    new_order = np.random.permutation(len(options))
    new_answer = np.where(new_order == answer)[0][0]
    return [options[idx] for idx in new_order], new_answer


def few_shot_prompt(data, format_function, user_prompt, n_few_shot=5):
    """
    Generates a few-shot prompt for the given data.

    Args:
        data (pandas.DataFrame): The input data. Each row should contain an 'input' and 'output' field.
        format_function (function): A function that formats the input and output of a sample.
        user_prompt (function): A function that generates the user prompt for the few-shot prompt.
        n_few_shot (int, optional): The number of few-shot samples to generate. Defaults to 5.

    Returns:
        list: A list of dictionaries representing the few-shot prompt, with each dictionary containing the role ('user' or 'assistant') and content of a prompt part.
    """
    samples_indices = np.random.choice(len(data), n_few_shot, replace=False)
    samples = data.iloc[samples_indices]
    # convert to list of dicts
    formatted_samples = [format_function(sample['input'], sample['output']) 
                         for index, sample in samples.iterrows()]
    query_parts = []
    for sample in formatted_samples:
        if 'user_input' in sample:
            user = user_prompt(sample['user_input'])
        else:
            user = user_prompt('')
        query_parts.append({'role': 'user', 'content': user})
        query_parts.append({'role': 'assistant', 'content': sample})
    return query_parts

def parse_response(response, is_multiple_choice, q_word, a_word, question_contains_options=True):
    """
    Parses a response string and extracts the question, answer, and options. 
    Very similar to the parse_response function in the generate_rephrase class with some extra functionality
    to handle multiple-choice questions. Also always contains an answer.

    Answer extracted from:
    ### {{ q_word }}
    [[ QUESTION ]]
    ### {{ a_word }}
    [[ ANSWER ]]

    Options can be included in the question or answer section, depending on the question_contains_options parameter.
    They will be extracted by looking for lines that start with A:, B:, C:, D:, E: or A., B., C., D., E. or A), B), C), D), E).


    Args:
        response (str): The response string to parse.
        is_multiple_choice (bool): Indicates whether the question is a multiple-choice question.
        q_word (str): The keyword used to identify the question in the response.
        a_word (str): The keyword used to identify the answer in the response.
        question_contains_options (bool, optional): Indicates whether the question contains options. 
            Defaults to True.

    Returns:
        tuple: A tuple containing the parsed question, answer, and options.
    """
    q_small = q_word.lower()
    a_small = a_word.lower()
    question, answer, options = None, None, None
    response = response.replace(f'### {q_word}:', f'### {q_word}').replace(f'### {q_small}:', f'### {q_small}')
    response = response.replace(f'### {a_word}:', f'### {a_word}').replace(f'### {a_small}:', f'### {a_small}')
    if f'### {q_word}' in response:
        question = response.split(f'### {q_word}')[-1].strip()

    elif f'### {q_small}' in response:
        question = response.split(f'### {q_small}')[-1].strip()

    else:
        logger.error(f'Failed to parse response (no question): {response}')
        return None, None, None

    if f'### {a_word}' in question:
        question, answer = question.split(f'### {a_word}')
    elif f'### {a_small}' in question:
        question, answer = question.split(f'### {a_small}')
    else:
        logger.error(f'Failed to parse response (no answer): {response}')
        return None, None, None

    question = question.strip()
    answer = answer.strip()
    
    if is_multiple_choice:
        if question_contains_options:
            question_and_options = question.split('\n')
            if len(question_and_options) <= 2:
                logger.error(f'Failed to parse response (no options): {response}')
                return None, None, None
            question = question_and_options[0]
            options  = question_and_options[1:]
            while len(options) > 0 and not options[0].startswith('A:') \
                and not options[0].startswith('A.') \
                    and not options[0].startswith('A)'):
                question += '\n' + options[0]
                options = options[1:]
            if len(options) == 0:
                logger.error(f'Failed to parse response (no options): {response}')
                return None, None, None

            options = [option for option in options if len(option) != 0]
            fuzzy_overlaps = [fuzz.ratio(answer, option) for option in options]
            answer = np.argmax(fuzzy_overlaps)
            if fuzzy_overlaps[answer] < 40:
                logger.error(f'Failed to parse response (too small fuzzy overlap {fuzzy_overlaps[answer]}): {response}')
                return None, None, None
        else:
            options = answer.split('\n')
            options = [option for option in options if len(option) != 0]
        question = question.strip()
        options = [option.strip() for option in options]
        for idx in range(len(options)):
            # regex if the option starts with A: or A. or A) (or same things with B, C, D, E)
            options[idx] = re.sub(r'^[A-E]\s*[:\)\.]\s*', '', options[idx])
    return question, answer, options


def generate_synthetic(data, system_prompt, format_function, store_file, n_samples=1000, n_few_shot=5, 
                       max_tokens=1024, user_prompt='Generate.', is_multiple_choice=False, 
                       q_word='Question', a_word='Answer', 
                       question_contains_options=True, random_choices=['']):
    """
    Generate synthetic data using the OpenAI GPT-4 model.

    Args:
        data (pd.DataFrame): List of data used for generating few-shot prompts. Should contain the 'input' and 'output' fields.
        system_prompt (str): The system prompt used in the conversation.
        format_function (function : (str,str)->str): A function that formats the one sample in the data to a string.
        store_file (str): The file path to store the generated data.
        n_samples (int, optional): The number of samples to generate. Defaults to 1000.
        n_few_shot (int, optional): The number of few-shot prompts to generate for each sample. Defaults to 5.
        max_tokens (int, optional): The maximum number of tokens allowed in the generated response. Defaults to 1024.
        user_prompt (str or function : str -> str, optional): The user prompt used in the conversation. 
            If a string is provided, it will be used as a static prompt for all samples.
            If a function is provided, it should take a user input as an argument and return a prompt string.
            Defaults to 'Generate.'.
        is_multiple_choice (bool, optional): Whether the generated responses are in multiple-choice format. Defaults to False.
        q_word (str, optional): The word used to identify questions in the generated responses. Defaults to 'Question'.
        a_word (str, optional): The word used to identify answers in the generated responses. Defaults to 'Answer'.
        question_contains_options (bool, optional): Whether the generated questions contain answer options. Defaults to True.
        random_choices (list, optional): List of random choices to be used as user inputs. Defaults to [''].

    Returns:
        pandas.DataFrame: A DataFrame containing the generated questions, options, answers, and user inputs.
    """
    querier = OpenAIQuery(model='gpt-4', max_tokens=max_tokens, temperature=0.7)

    if isinstance(user_prompt, str):
        user_prompt_copy = user_prompt[:]
        user_prompt = lambda x: user_prompt_copy
    
    few_shot_prompts =[few_shot_prompt(data, format_function, user_prompt, n_few_shot) for _ in range(n_samples)]

    random_user_input = [np.random.choice(random_choices) for _ in range(len(few_shot_prompts))]
    queries = [
        [
            {'role': 'system', 'content': system_prompt},
        ] + few_shots + [
            {'role': 'user', 'content': user_prompt(user_input)}, 
        ]
        for user_input, few_shots in zip(random_user_input, few_shot_prompts)
    ]

    store_file_json = store_file.replace('.csv', '.json')
    if not os.path.isfile(store_file_json):
        generated_responses, cost = asyncio.run(querier.run_string_prompts(queries))
        print(cost)
        json.dump(generated_responses, open(store_file.replace('.csv', '.json'), 'w'))
    else:
        generated_responses = json.load(open(store_file_json))

    responses = [response['message']['content'] for response in generated_responses]
    parsed_responses = [parse_response(response, is_multiple_choice, q_word, 
                                       a_word, question_contains_options) for response in responses]
    response_is_not_good = [parsed_response[0] is None for parsed_response in parsed_responses]
    parsed_responses = [parsed_response for parsed_response in parsed_responses if parsed_response[0] is not None]
    random_user_input = [user_input for user_input, is_not_good in
                          zip(random_user_input, response_is_not_good) if not is_not_good]
    questions, answers, options = zip(*parsed_responses)

    # create a pandas df
    df = pd.DataFrame({
        'question': questions,
        'options': options, 
        'answer': answers, 
        'user_input': random_user_input
    })
    return df


def generate_gsm8k(n_samples=1000):
    gsm8k_system_prompt = '''Your task is to generate new and original benchmark samples from the gsm8k benchmark. The user will instruct you to generate a sample that must follow the same format, difficulty and styling as questions from this benchmark.

Format your reply as:
### Question
[Question for the sample]
### Answer
[Answer of the sample]'''
    user_prompt = 'Generate a sample.'

    gsm8k_format_function = lambda input_, output: f'### Question\n{input_}\n### Answer\n{output}'
    gsm8k = pd.read_csv(os.path.join(base_path, 'gsm8k', 'contamination.csv'))
    gsm8k.rename(columns={'question': 'input', 'answer': 'output'}, inplace=True)
    os.makedirs(os.path.join(base_path, 'gsm8k'), exist_ok=True)
    df = generate_synthetic(gsm8k, gsm8k_system_prompt, gsm8k_format_function, 
                            os.path.join(base_path, 'gsm8k', 'synthetic.csv'),
                       n_few_shot=3, n_samples=n_samples, user_prompt=user_prompt)
    df.to_csv(os.path.join(base_path, 'gsm8k', 'synthetic.csv'), index=False)

def generate_mmlu(n_samples=1000):
    system_prompt = '''Your task is to generate new and original benchmark samples from the MMLU benchmark. The user will instruct you to generate from a certain subject. The generated sample must follow the same format, difficulty and styling as questions from this benchmark.

Format your reply as:
### Question
[Question for the sample]
### Answer
[Answer of the sample]'''
    user_prompt = lambda subject: f'Generate a sample on the following subject: {subject}'

    format_function = lambda input_, output: f'### Question:\n{input_}\n### Answer\n{output}'
    data = pd.read_csv(os.path.join(base_path, 'mmlu', 'contamination.csv'), converters={'choices': literal_eval})
    subjects = data['subject'].tolist()
    subjects = [' '.join([word.capitalize() for word in subject.split('_')])  for subject in subjects]
    input_prompts_cont = data['question'].tolist()
    input_prompts_choices = data['choices'].tolist()
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' 
                                      for idx, option in enumerate(choices)]) 
                            for choices in input_prompts_choices]
    input_prompts = [f'### Question\n{question}\n{options}' 
                     for question, options in zip(input_prompts_cont, input_prompt_option)]

    input_prompts_answer = data['answer'].tolist()
    output_prompts = [choices[answer] for answer, choices in zip(input_prompts_answer, input_prompts_choices)]

    data['input'] = input_prompts
    data['output'] = output_prompts
    data['user_input'] = subjects

    os.makedirs(os.path.join(base_path, 'mmlu'), exist_ok=True)
    df = generate_synthetic(data, system_prompt, format_function, os.path.join(base_path, 'mmlu', 'synthetic.csv'),
                       n_few_shot=3, n_samples=n_samples, user_prompt=user_prompt, is_multiple_choice=True, 
                       random_choices=subjects)

    random_options, random_answers = zip(*[randomize_options(choices, answer) 
                                            for choices, answer in zip(df['options'].tolist(), df['answer'].tolist())])
    df['choices'] = random_options
    df['answer'] = random_answers
    df['subject'] = df['user_input'].map(lambda x: x.replace(' ', '_').lower())

    df.to_csv(os.path.join(base_path, 'mmlu', 'synthetic.csv'), index=False)

def generate_arc(n_samples=1000):
    system_prompt = '''Your task is to generate new and original benchmark samples from the ARC benchmark. The user will instruct you to generate a sample that must follow the same format, difficulty and styling as questions from this benchmark.

Format your reply as:
### Question
[Question for the sample]
### Answer
[Answer of the sample]'''
    user_prompt = 'Generate a sample.'

    format_function = lambda input_, output: f'### Question\n{input_}\n### Answer\n{output}'
    data = pd.read_csv(os.path.join(base_path, 'arc', 'contamination.csv'), converters={'choices': literal_eval})
    input_prompts_cont = data['question'].tolist()
    input_prompts_choices = [choices['text'] for choices in data['choices'].tolist()]
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' for idx, option in enumerate(choices)]) 
                           for choices in input_prompts_choices]
    input_prompts_cont = [f'{question}\n{options}' for question, options in 
                          zip(input_prompts_cont, input_prompt_option)]

    input_prompts_answer = data['answerKey'].tolist()
    output_prompts = [choices[ord(answer) - 65 if 0 <= ord(answer) - 65 < len(choices) else int(answer) - 1] for answer, choices in zip(input_prompts_answer, input_prompts_choices)]

    data['input'] = input_prompts_cont
    data['output'] = output_prompts

    os.makedirs(os.path.join(base_path, 'arc'), exist_ok=True)
    df = generate_synthetic(data, system_prompt, format_function, os.path.join(base_path, 'arc', 'synthetic.csv'),
                       n_few_shot=3, n_samples=n_samples, user_prompt=user_prompt, is_multiple_choice=True)

    random_options, random_answers = zip(*[randomize_options(choices, answer) for choices, answer in zip(df['options'].tolist(), df['answer'].tolist())])
    df['choices'] = [{'text': choices, 'label': [chr(65 + idx) for idx in range(len(choices))]} for choices in random_options]
    df['answerKey'] = ['ABCDE'[answer] for answer in random_answers]

    df.to_csv(os.path.join(base_path, 'arc', 'synthetic.csv'), index=False)

def generate_hellaswag(n_samples=1000):
    system_prompt = '''Your task is to generate new and original benchmark samples from the Hellaswag benchmark. Provide a a context fitting the given activity label and four possible continuations on the context, one of which is the best one. The user will instruct you to generate a sample that must follow the same format, difficulty and styling as questions from this benchmark.

Format your reply as:
### Question
[Question for the sample]
### Answer
[Answer of the sample]'''
    user_prompt = lambda activity: f'Generate a sample with the following activity label: {activity}.'

    format_function = lambda input_, output: f'{input_}\n### Answer\n{output}'
    data = pd.read_csv(os.path.join(base_path, 'hellaswag', 'contamination.csv'), converters={'endings': literal_eval})

    input_prompts_cont = [f'{ctx_a} {ctx_b}'.capitalize() for ctx_a, ctx_b in zip(data['ctx_a'].tolist(), data['ctx_b'].tolist())]
    input_prompts_choices = data['endings'].tolist()
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' for idx, option in enumerate(choices)]) for choices in input_prompts_choices]

    input_prompts_answer = data['label'].tolist()
    output_prompts = [choices[answer] for choices, answer in zip(input_prompts_choices, input_prompts_answer)]
    activity_labels = data['activity_label'].tolist()
    input_prompts_cont = [f'### Question\n{question}\n{options}' for question, options in zip(input_prompts_cont, input_prompt_option)]

    data['input'] = input_prompts_cont
    data['output'] = output_prompts
    data['user_input'] = activity_labels

    os.makedirs(os.path.join(base_path, 'hellaswag'), exist_ok=True)
    df = generate_synthetic(data, system_prompt, format_function, 
                            os.path.join(base_path, 'hellaswag', 'synthetic.csv'),
                       n_few_shot=3, n_samples=n_samples, user_prompt=user_prompt, 
                       is_multiple_choice=True, random_choices=activity_labels)

    df['activity_label'] = df['user_input']
    df['ctx_a'] = df['question'].apply(lambda x: x if not x.endswith('...') else x[:-3])
    df['ctx_a'] = df['ctx_a'].apply(lambda x: x if not x.endswith('nan') else x[:-4])
    df['ctx_b'] = ''
    random_options, random_answers = zip(*[randomize_options(choices, answer) for choices, answer in zip(df['options'].tolist(), df['answer'].tolist())])
    df['endings'] = random_options
    df['label'] = random_answers
    df.to_csv(os.path.join(base_path, 'hellaswag', 'synthetic.csv'), index=False)

def test_data(data_1, data_2, exclude_same_index=False):
    column = 'question'
    data_2[column + '_overlap'] = 0.0
    data_2[column + '_idx'] = 0
    for index, sample in tqdm(data_2.iterrows()):
        if isinstance(sample[column], str):
            max_overlap =  0
            max_index = 0
            grams = sample[column].split(' ')
            for idx, sample_2 in data_1.iterrows():
                if column in sample_2 and isinstance(sample_2[column], str) and (idx != index or not exclude_same_index):
                    grams_2 = sample_2[column].split(' ')
                    n_overlap = np.count_nonzero([gram in grams_2 for gram in grams])
                    overlap = n_overlap / (max(len(grams), 1))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_index = idx

            data_2.loc[index, column + '_overlap'] = max_overlap
            data_2.loc[index, column + '_idx'] = max_index
    return data_2

def parse_answer_gsm8k(answer):
    """
    Parses the answer string of the GSM8k benchmark and extracts a floating-point number.

    Args:
        answer (str): The answer string to be parsed.

    Returns:
        float or None: The extracted floating-point number if found, otherwise None.
    """
    if '####' in answer:
        floating = answer.split('####')[1]
    else:
        return None
    # remove all non numbers and .
    floating_parsed = re.sub(r'[^\d.]', '', floating)
    if len(floating_parsed) == 0:
        return None
    return float(floating_parsed)

def parse_gsm8k_synthetic_correctness(data):
    """
        Parses the 'answer' field in the given data and checks if the parsed answer is an integer
        and does not involve rounding up or down.

        Args:
            data (pandas.DataFrame): The input data containing the 'answer' field.

        Returns:
            numpy.ndarray: A boolean array indicating whether each answer satisfies the conditions.
    """
    data['parsed_answer'] = data['answer'].apply(lambda x: parse_answer_gsm8k(x))
    data['is_int'] = data['parsed_answer'].apply(lambda x: not np.isnan(x) and int(x) == x)
    # manual check of the data indicated that these words were only used when the model used rounding operations.
    data['rounding_up_or_down'] = data['answer'].apply(lambda x: "can't" in x or 'rounds' in x or "not possible" in x or "a fraction of" in x or "cannot" in x)
    return np.logical_and(data['is_int'], np.logical_not(data['rounding_up_or_down']))

def create_omit_column(benchmark, omit_duplicate_score=0.9, omit_benchmark_score=0.8):
    """
    Create an 'omit' column in the synthetic dataset based on certain conditions.
    Stores the modified dataset in the 'synthetic_omitted.csv' file.

    Parameters:
    - benchmark (str): The name of the benchmark.
    - omit_duplicate_score (float): The threshold score for omitting duplicate questions.
    - omit_benchmark_score (float): The threshold score for omitting benchmark questions.
    """
    benchmark_source_1 = f"/home/ubuntu/contamination-detection/data/contamination/{benchmark}/contamination.csv"
    benchmark_source_2 = f"/home/ubuntu/contamination-detection/data/contamination/{benchmark}/synthetic.csv"
    data_1 = pd.read_csv(benchmark_source_1)
    data_2 = pd.read_csv(benchmark_source_2)
    data_2 = test_data(data_2, data_2, exclude_same_index=True)
    data_2['omit_duplicate'] = data_2['question_overlap'] > omit_duplicate_score
    data_2 = test_data(data_1, data_2)
    data_2['omit_benchmark'] = data_2['question_overlap'] > omit_benchmark_score
    omit = np.logical_or(data_2['omit_benchmark'], data_2['omit_duplicate'])
    if benchmark == 'gsm8k':
        omit = np.logical_or(omit, np.logical_not(parse_gsm8k_synthetic_correctness(data_2)))
    data_2 = pd.read_csv(benchmark_source_2)
    data_2['omit'] = omit
    data_2.to_csv(benchmark_source_2)
    benchmark_source_3 = f"/home/ubuntu/contamination-detection/data/contamination/{benchmark}/synthetic_omitted.csv"
    data_3 = data_2[np.logical_not(data_2['omit'])]
    data_3.to_csv(benchmark_source_3, index=False)



if __name__ == '__main__':
    generate_gsm8k()
    generate_mmlu()
    generate_arc()
    generate_hellaswag()
    create_omit_column('arc')
    create_omit_column('mmlu')
    create_omit_column('gsm8k')
    create_omit_column('hellaswag')

