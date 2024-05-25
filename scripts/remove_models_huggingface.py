import os
import shutil

# To avoid getting out of memory errors, we remove all the models from the huggingface cache

huggingface_path = '/home/ubuntu/.cache/huggingface/hub'
exclude_models = [
    'models--microsoft--Phi-3-mini-4k-Instruct',
    'models--meta-llama--Llama-2-7b-hf',
    'models--meta-llama--Llama-2-7b-chat-hf',
    'models--meta-llama--Meta-Llama-3-8B',
    'models--meta-llama--Meta-Llama-3-8B-Instruct',
    'models--mistralai--Mistral-7B-Instruct-v0.2',
    'models--mistralai--Mistral-7B-Instruct-v0.1',
    'models--mistralai--Mistral-7B-v0.1'
]
for folder in os.listdir(huggingface_path):
    if folder not in exclude_models and 'models' in folder:
        shutil.rmtree(os.path.join(huggingface_path, folder))
