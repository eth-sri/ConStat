# Generates the rephrased data for detecting syntax-specific contamination

from constat.openai import OpenAIQuery
from ast import literal_eval
import pandas as pd
from transformers import set_seed
import numpy as np
import json
import os
import asyncio
from loguru import logger
import numpy as np
import re

np.random.seed(0)
set_seed(0)

base_path = 'data/contamination'


def parse_response(response, is_multiple_choice, contains_answer, q_word, a_word, question_contains_options=True):
    """
    Parses the response string and extracts the question, answer, and options (if applicable).

    Specifically, this code extracts these components from the response string that is expected to be in the following format:
    ### {{ q_word }}
    [[ QUESTION ]]
    ### {{ a_word }}
    [[ ANSWER ]]

    Options can be included in the question or answer section, depending on the question_contains_options parameter.
    They will be extracted by looking for lines that start with A:, B:, C:, D:, E: or A., B., C., D., E. or A), B), C), D), E).

    Args:
        response (str): The response string to parse.
        is_multiple_choice (bool): Indicates whether the question is a multiple-choice question.
        contains_answer (bool): Indicates whether the response contains an answer.
        q_word (str): The keyword used to identify the question.
        a_word (str): The keyword used to identify the answer.
        question_contains_options (bool, optional): Indicates whether the question contains options. Defaults to True.

    Returns:
        tuple: A tuple containing the parsed question, answer, and options (if applicable).
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
        logger.warning(f'Failed to parse response (no question): {response}')
        return None, None, None

    if contains_answer:
        if f'### {a_word}' in question:
            question, answer = question.split(f'### {a_word}')
        elif f'### {a_small}' in question:
            question, answer = question.split(f'### {a_small}')
        else:
            logger.warning(f'Failed to parse response (no answer): {response}')
            return None, None, None

        question = question.strip()
        answer = answer.strip()
    
    if is_multiple_choice:
        if question_contains_options:
            question_and_options = question.split('\n')
            if len(question_and_options) <= 2:
                logger.warning(f'Failed to parse response (no options): {response}')
                return None, None, None
            question = question_and_options[0]
            options  = question_and_options[1:]
            while len(options) > 0 and not options[0].startswith('A:') \
                and not options[0].startswith('A.') \
                    and not options[0].startswith('A)'):
                question += '\n' + options[0]
                options = options[1:]
            if len(options) == 0:
                logger.warning(f'Failed to parse response (no options): {response}')
                return None, None, None
        else:
            options = answer.split('\n')
        question = question.strip()
        options = [option.strip() for option in options]
        for idx in range(len(options)):
            # regex if the option starts with A: or A. or A) (or same things with B, C, D, E)
            options[idx] = re.sub(r'^[A-E]\s*[:\)\.]\s*', '', options[idx])
    return question, answer, options


def generate(input_prompts, system_prompt, store_file, is_multiple_choice, contains_answer=False, 
             temperature=0.7, max_tokens=1024, question_word='Question', answer_word='Answer', 
             question_contains_options=True):
    """
    Generate rephrased questions and answers based on the given input prompts.

    We use GPT-4-Turbo to generate the rephrased questions and answers.
    Args:
        input_prompts (list): A list of input prompts for generating rephrased questions and answers.
        system_prompt (str): The system prompt to be used in the generation process.
        store_file (str): The file path to store the generated responses.
        is_multiple_choice (bool): Indicates whether the generated questions are multiple-choice or not.
        contains_answer (bool, optional): Indicates whether the input prompts contain the answer or not. Defaults to False.
        temperature (float, optional): The temperature value for controlling the randomness of the generated responses. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens allowed in the generated responses. Defaults to 1024.
        question_word (str, optional): The word to be used for denoting a question in the generated responses. Defaults to 'Question'.
        answer_word (str, optional): The word to be used for denoting an answer in the generated responses. Defaults to 'Answer'.
        question_contains_options (bool, optional): Indicates whether the generated questions contain options or not. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated questions, options (if applicable), and answers.
    """
    querier = OpenAIQuery(model='gpt-4-turbo', max_tokens=max_tokens, temperature=temperature, 
                          error_stop=100, timeout=120, read_cost=0.01, write_cost=0.03)

    queries = [
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ] for user_prompt in input_prompts
    ]

    store_file_json = store_file.replace('.csv', '.json')
    if not os.path.isfile(store_file_json):
        generated_responses, cost = asyncio.run(querier.run_string_prompts(queries))
        print(cost)
        json.dump(generated_responses, open(store_file.replace('.csv', '.json'), 'w'))
    else:
        generated_responses = json.load(open(store_file_json))

    responses = [response['message']['content'] for response in generated_responses]
    parsed_responses = [parse_response(response, is_multiple_choice, contains_answer, question_word, 
                                       answer_word, question_contains_options) for response in responses]
    parsed_responses = [parsed_response for parsed_response in parsed_responses if parsed_response[0] is not None]

    questions, answers, options = zip(*parsed_responses)

    # create a pandas df
    df = pd.DataFrame({
        'question': questions,
        'options': options, 
        'answer': answers
    })
    return df


def generate_gsm8k(contains_answer=False, contamination=True, temperature=0.7, max_tokens=1024):
    if contamination:
        data = pd.read_csv(os.path.join(base_path, 'gsm8k', 'contamination.csv'))
    else:
        data = pd.read_csv(os.path.join(base_path, 'gsm8k', 'no_contamination.csv'))
    system_prompt = '''Significantly rephrase the given question, but make sure the answer is still the same. Do not include the answer in your response.

Format your reply as:
### Question
[New rephrased question]'''
    if contains_answer:
        system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the gsm8k dataset. Rewrite the question and answer. Make significant changes to the used vocabulary, length and structure. Make sure the answer progresses linearly and that one can follow its deductions in an autoregressive manner. Still employ the '####' sign to indicate the output, and the '<< >>' signs to indicate computations. Ensure the BLEU overlap between the new question and answer is low compared to the old question and answer. 

Format your reply as:
### Question
[New Rephrased Question]
### Answer
[New Answer]'''

    file_name = 'rephrase'
    if contains_answer:
        file_name += '_answer'

    if not contamination:
        file_name += '_no'

    input_prompts_cont = data['question'].tolist()
    if contains_answer:
        input_prompts_answer = data['answer'].tolist()
        input_prompts_cont = [f'### Question\n{question}\n### Answer\n{answer}' for question, answer in zip(input_prompts_cont, input_prompts_answer)]
    else:
        input_prompts_cont = [f'### Question\n{question}' for question in input_prompts_cont]
    
    df = generate(input_prompts_cont, system_prompt, os.path.join(base_path, 'gsm8k', f'{file_name}_cont.csv'), 
                  is_multiple_choice=False, contains_answer=contains_answer, 
                  temperature=temperature, max_tokens=max_tokens)

    if not contains_answer:
        df['answer'] = data['answer'].tolist()

    df.to_csv(os.path.join(base_path, 'gsm8k', f'{file_name}_cont.csv'), index=False)


def generate_mmlu(contains_answer=False, contamination=False, temperature=0.7, max_tokens=1024):
    if contamination:
        data = pd.read_csv(os.path.join(base_path, 'mmlu', 'contamination.csv'), converters={'choices': literal_eval})
    else:
        data = pd.read_csv(os.path.join(base_path, 'mmlu', 'no_contamination.csv'), converters={'choices': literal_eval})
    system_prompt = '''Significantly rephrase the given question and options, but make sure that all possible options still have the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response.

Format your reply as:
### Question
[New rephrased question]'''
    if contains_answer:
        system_prompt = '''Significantly rephrase the given question and options, but make sure that the correct option still has the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response. Wrong options can be completely changed (and thus do not need to be simple rephrases).

Format your reply as:
### Reasoning
[Reasoning about how to rephrase question and options]
### Question
[New rephrased question]'''

    file_name = 'rephrase'
    if contains_answer:
        file_name += '_answer'
    
    if not contamination:
        file_name += '_no'

    input_prompts_cont = data['question'].tolist()
    input_prompts_choices = data['choices'].tolist()
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' for idx, option in enumerate(choices)]) for choices in input_prompts_choices]
    input_prompts_cont = [f'{question}\n{options}' for question, options in zip(input_prompts_cont, input_prompt_option)]
    input_prompts_cont = [f'### Question\n{question}' for question in input_prompts_cont]

    if contains_answer:
        input_prompts_answer = data['answer'].tolist()
        input_prompts_cont = [f'{question}\n### Answer\n{choices[answer]}' for question, answer, choices in zip(input_prompts_cont, input_prompts_answer, input_prompts_choices)]

    df = generate(input_prompts_cont, system_prompt, os.path.join(base_path, 'mmlu', f'{file_name}_cont.csv'), 
                  is_multiple_choice=True, contains_answer=False, temperature=temperature, max_tokens=max_tokens)

    df['choices'] = df['options']
    df['answer'] = data['answer'].tolist()
    df['subject'] = data['subject']

    df.to_csv(os.path.join(base_path, 'mmlu', f'{file_name}_cont.csv'), index=False)


def generate_arc(contains_answer=False, contamination=False, temperature=0.7, max_tokens=1024):
    if contamination:
        data = pd.read_csv(os.path.join(base_path, 'arc', 'contamination.csv'), converters={'choices': literal_eval})
    else:
        data = pd.read_csv(os.path.join(base_path, 'arc', 'no_contamination.csv'), converters={'choices': literal_eval})
    system_prompt = '''Significantly rephrase the given question and options, but make sure that all possible options still have the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response.

Format your reply as:
### Question
[New rephrased question]'''
    if contains_answer:
        system_prompt = '''Significantly rephrase the given question and options, but make sure that the correct option still has the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response. Wrong options can be completely changed (and thus do not need to be simple rephrases).

Format your reply as:
### Reasoning
[Reasoning about how to rephrase question and options]
### Question
[New rephrased question]'''

    file_name = 'rephrase'
    if contains_answer:
        file_name += '_answer'

    if not contamination:
        file_name += '_no'

    input_prompts_cont = data['question'].tolist()
    input_prompts_choices = [choices['text'] for choices in data['choices'].tolist()]
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' for idx, option in enumerate(choices)]) for choices in input_prompts_choices]
    input_prompts_cont = [f'{question}\n{options}' for question, options in zip(input_prompts_cont, input_prompt_option)]
    input_prompts_cont = [f'### Question\n{question}' for question in input_prompts_cont]

    if contains_answer:
        input_prompts_answer = data['answerKey'].tolist()
        input_prompts_cont = [f'{question}\n### Answer\n{choices[ord(answer) - 65 if 0 <= ord(answer) - 65 < len(choices) else int(answer) - 1]}' for question, answer, choices in zip(input_prompts_cont, input_prompts_answer, input_prompts_choices)]

    df = generate(input_prompts_cont, system_prompt, os.path.join(base_path, 'arc', f'{file_name}_cont.csv'), 
                  is_multiple_choice=True, contains_answer=False, temperature=temperature, max_tokens=max_tokens)

    df['choices'] = [{'text': choices, 'label': [chr(65 + idx) for idx in range(len(choices))]} for choices in df['options'].tolist()]
    df['answerKey'] = data['answerKey'].tolist()
    df['answerKey'] = df['answerKey'].apply(lambda x: chr(65 + int(x) - 1) if x.isdigit() else x)

    df.to_csv(os.path.join(base_path, 'arc', f'{file_name}_cont.csv'), index=False)


def generate_hellaswag(contains_answer=False, contamination=False, temperature=0.7, max_tokens=1024):
    if contamination:
        data = pd.read_csv(os.path.join(base_path, 'hellaswag', 'contamination.csv'), converters={'endings': literal_eval})
    else:
        data = pd.read_csv(os.path.join(base_path, 'hellaswag', 'no_contamination.csv'), converters={'endings': literal_eval})
    system_prompt = '''Significantly rephrase the given context and several possible continuations and make sure that all possible continuations still have the same label. Label the continuations with A:, B:, C:, D:, E:. Keep the ':' separating the activity label from the context in the context. Make sure that each continuation can still follow the rephrased context.

Format your reply as:
### Context
[New rephrased question]
### Continuation
[New rephrased continuation]'''
    if contains_answer:
        system_prompt = '''Significantly rephrase the given context and continuation. Keep the ':' separating the activity label from the context in the context. Make sure the continuation can still follow the rephrased context.

Format your reply as:
### Reasoning
[Reasoning about how to rephrase context and continuation]
### Context
[New rephrased question]
### Continuation
[New rephrased continuation]'''

    file_name = 'rephrase'
    if contains_answer:
        file_name += '_answer'

    if not contamination:
        file_name += '_no'

    input_prompts_cont = [f'{activity_label}: {ctx_a} {ctx_b}'.capitalize() for activity_label, ctx_a, ctx_b in zip(data['activity_label'].tolist(), data['ctx_a'].tolist(), data['ctx_b'].tolist())]
    input_prompts_choices = data['endings'].tolist()
    input_prompt_option = ['\n'.join([f'{chr(65 + idx)}: {option}' for idx, option in enumerate(choices)]) for choices in input_prompts_choices]

    if contains_answer:
        input_prompts_answer = data['label'].tolist()
        input_prompts_cont = [f'### Context\n{question}\n### Continuation\n{choices[answer]}' for question, answer, choices in zip(input_prompts_cont, input_prompts_answer, input_prompts_choices)]
    else:
        input_prompts_cont = [f'### Context\n{question}\n### Continuation\n{options}' for question, options in zip(input_prompts_cont, input_prompt_option)]

    df = generate(input_prompts_cont, system_prompt, os.path.join(base_path, 'hellaswag', f'{file_name}_cont.csv'), 
                  is_multiple_choice=not contains_answer, contains_answer=True, temperature=temperature, 
                  max_tokens=max_tokens, question_word='Context', answer_word='Continuation', 
                  question_contains_options=False)

    df['activity_label'] = df['question'].apply(lambda x: x.split(':')[0] if ':' in x else '')
    df['ctx_a'] = df['question'].apply(lambda x: (':').join(x.split(':')[1:]) if ':' in x else x)
    df['ctx_b'] = ''
    df['endings'] = df['options']
    df['label'] = data['label'].tolist()

    df.to_csv(os.path.join(base_path, 'hellaswag', f'{file_name}_cont.csv'), index=False)

if __name__ == '__main__':
    generate_gsm8k(False, True)
    generate_gsm8k(True, True)
    generate_gsm8k(False, False)
    generate_gsm8k(True, False)
    generate_mmlu(False, True)
    generate_mmlu(False, False)
    generate_mmlu(True, True)
    generate_mmlu(True, False)
    generate_arc(False, True)
    generate_arc(False, False)
    generate_arc(True, True)
    generate_arc(True, False)
    generate_hellaswag(False, False)
    generate_hellaswag(True, False)
    generate_hellaswag(False, True)
    generate_hellaswag(True, True)
