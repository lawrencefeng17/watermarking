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
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import threading
from queue import Queue

def assign_buckets(vocab_size, num_buckets, seed=None):
    """
    Assigns tokens to buckets randomly but evenly.
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
    """
    w_i = np.bincount(bucket_assignments, weights=token_probs, minlength=num_buckets)
    max_vals = np.maximum(0, 1 / num_buckets - w_i)
    return np.sum(max_vals)

def load_compressed_data(file_path):
    """
    Loads compressed data from a .zst file.
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = pickle.loads(zstd.decompress(compressed_data))
    return decompressed_data

class ResultWriter:
    def __init__(self, output_csv):
        self.output_csv = output_csv
        self.queue = Queue()
        self.writing = True
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.start()
        
        # Write header if file doesn't exist
        if not os.path.exists(output_csv):
            pd.DataFrame(columns=['prompt', 'category', 'bucket_size', 'statistic']
                       ).to_csv(output_csv, index=False)

    def _writer_worker(self):
        while True:
            results = self.queue.get()
            if results is None:
                break
            
            df = pd.DataFrame(results)
            df.to_csv(self.output_csv, mode='a', header=False, index=False)

    def write_results(self, results):
        if results:
            self.queue.put(results)

    def stop(self):
        self.queue.put(None)
        self.writer_thread.join()

def process_single_file(args):
    """
    Process a single compressed file and compute statistics.
    """
    file_name, data_dir, metadata, bucket_assignments_dict, bucket_sizes = args
    file_results = []
    
    try:
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
                
                file_results.append({
                    'prompt': prompt,
                    'category': category,
                    'bucket_size': num_buckets,
                    'statistic': stat
                })
                
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
    
    return file_results

def process_dataset(data_dir, output_csv='token_statistics.csv'):
    """
    Processes all compressed data files in parallel and computes statistics.
    """
    # Define bucket sizes (2^1 to 2^14)
    bucket_sizes = [2**i for i in range(1, 15)]
    
    # Get vocab size from first file
    first_file = next(f for f in os.listdir(data_dir) if f.endswith('.zst'))
    vocab_size = torch.tensor(load_compressed_data(os.path.join(data_dir, first_file))['probs']).size(-1)
    
    # Precompute bucket assignments
    print("Assigning tokens to buckets...")
    bucket_assignments_dict = {}
    for num_buckets in bucket_sizes:
        seed = num_buckets
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments
    
    # Get file list and metadata
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.zst')]
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    
    # Initialize the result writer
    writer = ResultWriter(output_csv)
    
    # Calculate number of processes (leave one core free)
    num_processes = max(1, cpu_count() - 1)
    
    print(f"Processing files using {num_processes} processes...")
    
    # Prepare arguments for parallel processing
    process_args = [(f, data_dir, metadata, bucket_assignments_dict, bucket_sizes) 
                   for f in compressed_files]
    
    # Process files in parallel
    with Pool(num_processes) as pool:
        for results in tqdm(pool.imap_unordered(process_single_file, process_args), 
                          total=len(compressed_files)):
            writer.write_results(results)
    
    # Stop the writer thread
    writer.stop()
    
    print(f"Statistics saved to {output_csv}")
    return pd.read_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    name = data_dir.stem
    src_dir = Path(__file__).resolve().parent
    plots_dir = src_dir / '../plots'
    output_csv = src_dir / 'statistics' / name / 'token_auc.csv'

    # Create necessary directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(output_csv.parent, exist_ok=True)

    print("Data directory: " + str(data_dir))
    print("Output CSV: " + str(output_csv))
    
    # Process the dataset
    df_statistics = process_dataset(data_dir, output_csv=output_csv)