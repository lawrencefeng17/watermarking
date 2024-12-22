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

from scipy.stats import entropy

def compute_entropy(token_probs):
    """
    Computes the entropy of a token probability distribution.

    Args:
        token_probs (np.ndarray): Array of token probabilities (size: vocab_size).

    Returns:
        float: Entropy value.
    """
    # To prevent issues with log(0), add a small epsilon where necessary
    token_probs = np.where(token_probs > 0, token_probs, 1e-12)
    return entropy(token_probs, base=2)  # Using base 2 for entropy in bits


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
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

    # Initialize list to store results
    results = []
    
    # List all compressed data files
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.zst')]
    
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
            
            stat = compute_entropy(token_probs_np)
                
            results.append({
                'prompt': prompt,
                'category': category,
                'entropy': stat
            })
        
        # Convert results to DataFrame and append to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        results.clear()  # Clear results after appending to CSV
    
    print(f"Statistics saved to {output_csv}")
    
    return pd.read_csv(output_csv)


parser = argparse.ArgumentParser(description='Compute entropy statistics for a dataset.')
parser.add_argument('--data-dir', required=True, type=str, help='Path to the directory containing compressed data files.')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = data_dir.split('_')[-1]
src_dir = Path(__file__).resolve().parent
output_dir = src_dir / '../statistics' / output_dir
output_csv = output_dir / 'token_statistics.csv'

#  Process the dataset and get the statistics DataFrame
df_statistics = process_dataset(data_dir, output_csv=output_csv)