import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import zstandard as zstd
import pickle
import os
from tqdm import tqdm
import pandas as pd
import argparse

dataset = load_dataset("databricks/databricks-dolly-15k")

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print(config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def get_all_token_distributions(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False  # Change to True if sampling is desired
    )
    logits_list = outputs.scores
    probabilities_list = [torch.softmax(logit, dim=-1).cpu().numpy() for logit in logits_list]
    return probabilities_list

def compress_and_store(idx, prompt, category, token_probs, output_dir='compressed_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stored_data = {
        "prompt": prompt,
        "category": category,
        "probs": [prob.tolist() for prob in token_probs]
    }

    compressed_data = zstd.compress(pickle.dumps(stored_data))
    with open(os.path.join(output_dir, f'compressed_data_{idx}.zst'), 'wb') as f:
        f.write(compressed_data)

# Prepare metadata storage
metadata = []

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str, default='/raid/lawrence/compressed_data')
args = parser.parse_args()
output_dir = args.o
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate and collect data
for idx, entry in enumerate(tqdm(dataset['train'])):
    prompt = entry['instruction']
    category = entry['category']
    token_distributions = get_all_token_distributions(prompt, max_new_tokens=50)
    compress_and_store(idx, prompt, category, token_distributions, output_dir)
    
    metadata.append({
        "idx": idx,
        "prompt": prompt,
        "category": category,
        "num_tokens": len(token_distributions)
    })

# Save metadata
df_metadata = pd.DataFrame(metadata)
df_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
