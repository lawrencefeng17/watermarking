import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import zstandard as zstd
from tqdm import tqdm
import torch

def assign_buckets(vocab_size, num_buckets, device, seed=None):
    """
    Assigns tokens to buckets randomly but evenly, returning a tensor on the specified device.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Create assignments on CPU first
    tokens = torch.randperm(vocab_size)
    
    base = vocab_size // num_buckets
    remainder = vocab_size % num_buckets
    bucket_sizes = [base + 1 if i < remainder else base for i in range(num_buckets)]
    
    bucket_assignments = torch.empty(vocab_size, dtype=torch.long)
    current = 0
    for bucket_id, size in enumerate(bucket_sizes):
        bucket_assignments[tokens[current:current+size]] = bucket_id
        current += size
    
    return bucket_assignments.to(device)

def compute_statistic_batch(token_probs, bucket_assignments, num_buckets):
    """
    Computes statistics for a batch of token probability distributions using GPU operations.
    
    Args:
        token_probs (torch.Tensor): Batch of token probabilities (shape: [batch_size, vocab_size])
        bucket_assignments (torch.Tensor): Bucket assignments tensor (shape: [vocab_size])
        num_buckets (int): Number of buckets
    
    Returns:
        torch.Tensor: Computed statistics for the batch
    """
    # Create a sparse matrix for bucket assignments
    batch_size = token_probs.shape[0]
    vocab_size = token_probs.shape[1]
    
    # Sum probabilities per bucket using sparse matrix multiplication
    # Create indices for scatter operation
    batch_indices = torch.arange(batch_size, device=token_probs.device).unsqueeze(1).expand(-1, vocab_size)
    bucket_indices = bucket_assignments.unsqueeze(0).expand(batch_size, -1)
    
    # Compute bucket sums using scatter_add
    w_i = torch.zeros(batch_size, num_buckets, device=token_probs.device)
    w_i.scatter_add_(1, bucket_indices, token_probs)
    
    # Compute max(0, 1 - num_buckets * w_i)
    max_vals = torch.clamp(1 - num_buckets * w_i, min=0)
    
    # Compute mean over buckets
    return torch.mean(max_vals, dim=1)

def process_dataset(data_dir, output_csv='token_statistics.csv', batch_size=32, device='cuda'):
    """
    Processes all compressed data files and computes statistics using GPU acceleration.
    """
    bucket_sizes = [2**i for i in range(1, 15)]
    vocab_size = 128256
    
    # Precompute bucket assignments for each bucket size on GPU
    print("Assigning tokens to buckets...")
    bucket_assignments_dict = {
        num_buckets: assign_buckets(vocab_size, num_buckets, device, seed=num_buckets)
        for num_buckets in bucket_sizes
    }
    
    results = []
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.zst')]
    
    print("Processing files and computing statistics...")
    for file_name in tqdm(compressed_files):
        file_path = os.path.join(data_dir, file_name)
        data = load_compressed_data(file_path)
        
        prompt = data['prompt']
        category = data.get('category', 'Unknown')
        token_probs = torch.tensor(data['probs'], device=device)
        token_probs = token_probs.squeeze(1)
        print(file_path, token_probs.shape)
        
        # Process in batches
        num_samples = token_probs.size(0)
        for i in range(0, num_samples, batch_size):
            batch_probs = token_probs[i:i+batch_size]
            
            for num_buckets in bucket_sizes:
                bucket_assignments = bucket_assignments_dict[num_buckets]
                stats = compute_statistic_batch(batch_probs, bucket_assignments, num_buckets)
                
                # Move results to CPU for storage
                stats_cpu = stats.cpu().numpy()
                
                for j, stat in enumerate(stats_cpu):
                    results.append({
                        'prompt': prompt,
                        'category': category,
                        'bucket_size': num_buckets,
                        'statistic': stat
                    })

        break
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}")
    
    return df

def load_compressed_data(file_path):
    """Loads compressed data from a .zst file."""
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = pickle.loads(zstd.decompress(compressed_data))
    return decompressed_data

# Example usage
if __name__ == "__main__":
    data_dir = '/raid/lawrence/compressed_data/'
    plots_dir = '/home/lawrence/prc/plots/'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process the dataset with GPU acceleration
    df_statistics = process_dataset(
        data_dir, 
        output_csv='/home/lawrence/prc/token_statistics.csv',
        batch_size=32,  # Adjust based on your GPU memory
        device=device
    )