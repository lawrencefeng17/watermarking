import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import zstandard as zstd
from tqdm import tqdm
import torch
from pathlib import Path

def assign_buckets(vocab_size, num_buckets, seed=None):
    """
    Assigns tokens to buckets randomly but evenly.

    Args:
        vocab_size (int): Number of tokens.
        num_buckets (int): Number of buckets (must be a power of 2).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of bucket assignments for each token.
    """
    if seed is not None:
        np.random.seed(seed)
    tokens = np.arange(vocab_size)
    np.random.shuffle(tokens)
    
    base = vocab_size // num_buckets
    remainder = vocab_size % num_buckets
    bucket_sizes = [base + 1 if i < remainder else base for i in range(num_buckets)]
    
    bucket_assignments = np.empty(vocab_size, dtype=int)
    current = 0
    for bucket_id, size in enumerate(bucket_sizes):
        bucket_assignments[tokens[current:current+size]] = bucket_id
        current += size
    
    return bucket_assignments

def compute_statistic(token_probs, bucket_assignments, num_buckets):
    """
    Computes the statistic for a single token probability distribution.

    Args:
        token_probs (np.ndarray): Array of token probabilities (size: vocab_size).
        bucket_assignments (np.ndarray): Array of bucket assignments (size: vocab_size).
        num_buckets (int): Number of buckets.

    Returns:
        float: Computed statistic.
    """
    # Sum probabilities per bucket
    w_i = np.bincount(bucket_assignments, weights=token_probs, minlength=num_buckets)
    
    max_vals = np.maximum(0, 1 / num_buckets - w_i)
    
    statistic = np.sum(max_vals)
    
    return statistic

def load_compressed_data(file_path):
    """
    Loads compressed data from a .zst file.

    Args:
        file_path (str): Path to the compressed file.

    Returns:
        dict: Decompressed data.
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = pickle.loads(zstd.decompress(compressed_data))
    return decompressed_data

def process_dataset(data_dir, output_csv='token_statistics.csv'):
    """
    Processes all compressed data files and computes statistics.

    Args:
        data_dir (str): Directory containing compressed data files.
        output_csv (str): Path to save the output CSV.

    Returns:
        pd.DataFrame: DataFrame containing all computed statistics.
    """
    # Define bucket sizes (2^1 to 2^14)
    bucket_sizes = [2**i for i in range(1, 15)]
    
    # Assuming vocab_size is 128k as per your description
    vocab_size = torch.tensor(load_compressed_data(os.path.join(data_dir, 'compressed_data_0.zst'))['probs']).size(2)
    
    # Precompute bucket assignments for each bucket size
    print("Assigning tokens to buckets...")
    bucket_assignments_dict = {}
    for num_buckets in bucket_sizes:
        seed = num_buckets  # Different seed for each bucket size
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments
    
    # Initialize list to store results
    results = []
    
    # List all compressed data files
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.zst')]
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    
    print("Processing files and computing statistics...")
    for file_name in tqdm(compressed_files):
        file_path = os.path.join(data_dir, file_name)
        data = load_compressed_data(file_path)
        
        idx = data['idx']
        prompt = metadata.loc[metadata['idx'] == idx, 'prompt'].values[0]
        category = metadata.loc[metadata['idx'] == idx, 'category'].values[0]
        token_probs = torch.tensor(data['probs'])
        token_probs = token_probs.squeeze(1)
        
        for i in range(token_probs.size(0)):
            token_probs_np = token_probs[i].numpy()
            
            for num_buckets in bucket_sizes:
                bucket_assignments = bucket_assignments_dict[num_buckets]
                stat = compute_statistic(token_probs_np, bucket_assignments, num_buckets)
                
                results.append({
                    'prompt': prompt,
                    'category': category,
                    'bucket_size': num_buckets,
                    'statistic': stat
                })

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}")
    
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
args = parser.parse_args()

data_dir = args.data_dir
name = data_dir.split('/')[-1]
src_dir = Path(__file__).resolve().parent
plots_dir = src_dir / '../plots'
output_csv = src_dir / f'statistics/{name}/token_statistics.csv'

os.makedirs(plots_dir, exist_ok=True)

# Process the dataset and get the statistics DataFrame
df_statistics = process_dataset(data_dir, output_csv=output_csv)