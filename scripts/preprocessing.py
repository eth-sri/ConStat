# Preprocesses each benchmark by downloading it and splitting it into contaminated and uncontaminated parts.

import datasets
import os
import pandas as pd
import re

base_path = 'data/contamination'


def gsm8k(): # pair with mathqa, has training data
    dataset = datasets.load_dataset('gsm8k', 'main', split='test')
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_contamination = df.iloc[:len(df) // 2]
    df_no_contamination = df.iloc[len(df) // 2:]
    os.makedirs(os.path.join(base_path, 'gsm8k'), exist_ok=True)
    df_contamination.to_csv(os.path.join(base_path, 'gsm8k', 'contamination.csv'), index=False)
    df_no_contamination.to_csv(os.path.join(base_path, 'gsm8k', 'no_contamination.csv'), index=False)

    dataset_train = datasets.load_dataset('gsm8k', 'main', split='train')
    df_train = pd.DataFrame(dataset_train)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_train[:len(df_contamination)]
    df_train.to_csv(os.path.join(base_path, 'gsm8k', 'training.csv'), index=False)

def mmlu(): # pair with the comparison part, no training data
    contaminated_parts = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
       'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 
       'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering']
    uncontaminated_parts = ['professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    dataset = datasets.load_dataset('cais/mmlu', 'all', split='test')
    df = pd.DataFrame(dataset)

    df_to_compare = df[df['subject'].isin(uncontaminated_parts)]
    df_actual = df[df['subject'].isin(contaminated_parts)]
    df_actual = df_actual.sample(frac=1, random_state=42).reset_index(drop=True)
    df_contamination = df_actual[:len(df_actual) // 2]
    df_no_contamination = df_actual[len(df_actual) // 2:]

    os.makedirs(os.path.join(base_path, 'mmlu'), exist_ok=True)
    df_contamination.to_csv(os.path.join(base_path, 'mmlu', 'contamination.csv'), index=False)
    df_no_contamination.to_csv(os.path.join(base_path, 'mmlu', 'no_contamination.csv'), index=False)
    df_to_compare.to_csv(os.path.join(base_path, 'mmlu', 'comparison.csv'), index=False)

def arc(): # pair with sciq, no training data
    dataset = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_contamination = df.iloc[:1000]
    df_no_contamination = df.iloc[1000:2000]
    os.makedirs(os.path.join(base_path, 'arc'), exist_ok=True)
    df_contamination.to_csv(os.path.join(base_path, 'arc', 'contamination.csv'), index=False)
    df_no_contamination.to_csv(os.path.join(base_path, 'arc', 'no_contamination.csv'), index=False)


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def hellaswag(): # couple with winogrande, has training data
    dataset = datasets.load_dataset('Rowan/hellaswag', split='validation')
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_contamination = df.iloc[:1000]
    df_no_contamination = df.iloc[1000:2000]
    for col in ['ctx_a', 'ctx_b', 'ctx']:
        df_contamination[col] = df_contamination[col].apply(preprocess)
        df_no_contamination[col] = df_no_contamination[col].apply(preprocess)
    df_contamination['endings'] = [
        [preprocess(ending) for ending in endings] for endings in df_contamination['endings']
    ]
    df_no_contamination['endings'] = [
        [preprocess(ending) for ending in endings] for endings in df_no_contamination['endings']
    ]
    os.makedirs(os.path.join(base_path, 'hellaswag'), exist_ok=True)
    df_contamination.to_csv(os.path.join(base_path, 'hellaswag', 'contamination.csv'), index=False)
    df_no_contamination.to_csv(os.path.join(base_path, 'hellaswag', 'no_contamination.csv'), index=False)

    dataset = datasets.load_dataset('Rowan/hellaswag', split='train')
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[:len(df_contamination)]
    for col in ['ctx_a', 'ctx_b', 'ctx']:
        df[col] = df[col].apply(preprocess)
    df['endings'] = [
        [preprocess(ending) for ending in endings] for endings in df['endings']
    ]
    df.to_csv(os.path.join(base_path, 'hellaswag', 'training.csv'), index=False)

if __name__ == '__main__':
    gsm8k()
    mmlu()
    arc()
    hellaswag()