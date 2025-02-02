import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import gc
from torch.cuda import empty_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime

np.random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
set_seed(4)

current_dir = Path(__file__).parent

import sys
sys.path.append(str(current_dir.parent.parent))

###############################################################################
#                             PRELIMINATIRES                                  #
###############################################################################

# --model "meta-llama/Llama-3.2-1B-Instruct"
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Name or path of the model.')
args = parser.parse_args()

print("Loading model and tokenizer...")
config = AutoConfig.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
# Check for warnings
# Padding is on left for decoder-only models (e.g. Llama)
tokenizer.padding_side = 'left'

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = config.vocab_size
    

model = AutoModelForCausalLM.from_pretrained(args.model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
        
prompts = [
        "What is the capital of France?",
        "How do you make a cake?",
        "Tell me an elaborate joke.",
        "What is the meaning of life?",
        "Explain quantum physics in simple terms.",
        "Tell me a creative love story between a robot and a human.",
        "What are the latest advancements in AI?",
        "Describe a futuristic city.",
        "What is the weather like today?",
        "Write a story about a man and his love for the sea.",
        "What is the best way to learn programming?",
        "I've been thinking about the relationship between humans and AI. Can you tell me a story about a future where AI and humans coexist peacefully?",
        "What role do you forsee AI playing in the future of education?",
        "How can AI help in solving climate change?",
        "Describe techniques for predicting the prices of stocks.",
        "What are the ethical implications of AI in surveillance?",
    ]

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
max_new_tokens = 2000

outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
output = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]

batch_distributions = []
for seq_scores in zip(*outputs.logits): # seq_scores is a tuple of max_new_tokens tensors of shape [batch_size]
    distributions = []
    for score in seq_scores: # iterate over each token's score (max_new_tokens iterations)
        probs = torch.softmax(score, dim=-1).cpu().numpy() # softmax over vocab_size to obtain probabilities
        distributions.append(probs) # appending one token at a time
    # distributions is a list of max_new_tokens tensors of shape [batch_size, vocab_size]
    batch_distributions.append(distributions) 


batch_decoded = []

for generated_tokens in outputs.sequences:
    decoded_tokens = [
        tokenizer.decode(token.item()) 
        for token in generated_tokens
    ]
    input_length = inputs.input_ids.shape[1]
    decoded_tokens = decoded_tokens[input_length:]
    batch_decoded.append(decoded_tokens)

###############################################################################
#                               ENTROPY                                       #
###############################################################################
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

batch_entropies = []
for batch in batch_distributions:
    entropies = []
    for token in batch:
        entropies.append(entropy(token, base=2))
    batch_entropies.append(entropies)

# plot the entropy of each token in the batch
# add trendline to each plot
for idx, e in enumerate(batch_entropies):
    plt.figure(figsize=(12,6))
    x_values = np.arange(len(e))
    coeffcients = np.polyfit(x_values, e, 1)
    trendline = np.polyval(coeffcients, x_values)
    plt.plot(trendline, label='Trendline', color='red')
    plt.title(f"Prompt: {prompts[idx]}")
    plt.plot(e)
    plt.savefig(current_dir / f"entropy_{idx}.png")
    plt.close()

# checks for EOS token and truncuate
batch_decoded = np.array(batch_decoded)
truncuated_batch_decoded = []
for batch in batch_decoded:
    mask = batch != tokenizer.eos_token
    truncuated_batch_decoded.append(batch[mask])
    print("truncated output has length", len(batch[mask]))
    
# replot
for idx, e in enumerate(batch_entropies):
    e = e[:len(truncuated_batch_decoded[idx])]
    plt.figure(figsize=(12,6))
    x_values = np.arange(len(e))
    coeffcients = np.polyfit(x_values, e, 1)
    trendline = np.polyval(coeffcients, x_values)
    plt.plot(trendline, label='Trendline', color='red')
    plt.plot(e)
    plt.title(f"Prompt: {prompts[idx]}")
    plt.xlabel("Token Index")
    plt.ylabel("Entropy")
    plt.savefig(current_dir / f"entropy_{idx}_truncuated.png")
    plt.close()

###############################################################################
#                  Test relationship between AUC and Entropy                  #
###############################################################################
from compute_statistics import analyze_worker, analyze_token_distributions, assign_buckets, compute_auc

vocab_size = config.vocab_size
stats = []

bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
bucket_assignments_dict = {}
for num_buckets in bucket_sizes:
    seed = num_buckets  # each bucket size gets a different seed
    bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
    bucket_assignments_dict[num_buckets] = bucket_assignments

fig, axes = plt.subplots(1, 2, figsize=(15, len(bucket_sizes) * 1.5))
axes = axes.flatten()

subset_buckets = [2,128]
for idx, num_buckets in enumerate(subset_buckets):
    bucket_assignments = bucket_assignments_dict[num_buckets]
    counts, _ = np.histogram(bucket_assignments, bins=np.arange(num_buckets + 1))
    
    axes[idx].bar(range(num_buckets), counts, width=1.0, edgecolor='black')
    axes[idx].set_title(f"{num_buckets} Buckets")
    axes[idx].set_xlabel("Bucket Index")
    axes[idx].set_ylabel("Count")
    axes[idx].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

plt.tight_layout()
plt.savefig(current_dir / "bucket_assignments.png")

token_stats = []
for idx, dist in enumerate(batch_distributions):
    dist = dist[:len(truncuated_batch_decoded[idx])] 
    stats = analyze_token_distributions(dist, bucket_assignments_dict)
    token_stats.append(stats)

# Entropies should be the same

for i in range(len(batch_entropies)):
    analyze_entropies = [token_stats[i][j]['entropy'] for j in range(len(token_stats[i]))]
    analyze_entropies = np.array(analyze_entropies)
    assert np.allclose(batch_entropies[i][:len(truncuated_batch_decoded[i])], analyze_entropies)
print("All entropies are the same, YAY")

# Let's now check the AUCs 
bucket_batch_aucs = {}
for num_buckets in subset_buckets:
    batch_aucs = []
    for idx, dist in enumerate(batch_distributions):
        aucs = []
        for i in range(len(dist)):
            aucs.append(compute_auc(dist[i], bucket_assignments_dict[num_buckets], num_buckets))
        batch_aucs.append(aucs)
    bucket_batch_aucs[num_buckets] = batch_aucs

for num_buckets in subset_buckets:
    for i in range(len(batch_aucs)):
        analyze_aucs = [token_stats[i][j][f'auc_{num_buckets}'] for j in range(len(token_stats[i]))]
        analyze_aucs = np.array(analyze_aucs)
        assert np.allclose(bucket_batch_aucs[num_buckets][i][:len(truncuated_batch_decoded[i])], analyze_aucs)

print("All AUCs are the same, YAY")

# plot AUC vs entropy using all the data

# data = pd.DataFrame({
#     'AUC': bucket_batch_aucs[2][0][:len(truncuated_batch_decoded[0])],
#     'Entropy': batch_entropies[0][:len(truncuated_batch_decoded[0])],
# })

data = pd.DataFrame()
num_buckets = 2
for i in range(len(batch_aucs)):
    temp = pd.DataFrame({
        'AUC': bucket_batch_aucs[num_buckets][i][:len(truncuated_batch_decoded[i])],
        'Entropy': batch_entropies[i][:len(truncuated_batch_decoded[i])],
    })
    data = pd.concat([data, temp], ignore_index=True)


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Entropy', y='AUC', data=data)
plt.title('Scatter Plot of AUC vs. Entropy')
plt.xlabel('Entropy')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig(current_dir / "AUC_vs_Entropy.png")


breakpoint()
