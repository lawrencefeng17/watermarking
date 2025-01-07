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
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
from queue import Queue

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

def process_single_file(file_name, data_dir, metadata):
    """
    Process a single compressed file and compute statistics.
    Args:
        file_name (str): Name of the file to process.
        data_dir (str): Directory containing the file.
        metadata (pd.DataFrame): DataFrame containing metadata.
    Returns:
        list: List of dictionaries containing computed statistics.
    """
    file_results = []
    file_path = os.path.join(data_dir, file_name)
    
    try:
        data = load_compressed_data(file_path)
        idx = data['idx']
        prompt = metadata.loc[metadata['idx'] == idx, 'prompt'].values[0]
        category = metadata.loc[metadata['idx'] == idx, 'category'].values[0]
        
        token_probs = torch.tensor(data['probs'])
        token_probs = token_probs.squeeze(1)
        
        for i in range(token_probs.size(0)):
            token_probs_np = token_probs[i].numpy()
            stat = compute_entropy(token_probs_np)
            
            file_results.append({
                'prompt': prompt,
                'category': category,
                'entropy': stat
            })
            
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
    
    return file_results

class ResultWriter:
    def __init__(self, output_csv):
        self.output_csv = output_csv
        self.queue = Queue()
        self.writing = True
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.start()

    def _writer_worker(self):
        while True:
            results = self.queue.get()
            if results is None:
                break
                
            df = pd.DataFrame(results)
            df.to_csv(self.output_csv, mode='a', header=not os.path.exists(self.output_csv), index=False)

    def write_results(self, results):
        self.queue.put(results)

    def stop(self):
        self.queue.put(None)
        self.writer_thread.join()

def process_dataset(data_dir, output_csv='token_statistics.csv'):
    """
    Processes all compressed data files in parallel and computes statistics.
    Args:
        data_dir (str): Directory containing compressed data files.
        output_csv (str): Path to save the output CSV.
    Returns:
        pd.DataFrame: DataFrame containing all computed statistics.
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    
    # List all compressed data files
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.zst')]
    
    # Initialize the result writer
    writer = ResultWriter(output_csv)
    
    # Calculate number of processes to use (leave one core free for system)
    num_processes = max(1, cpu_count() - 1)
    
    print(f"Processing files using {num_processes} processes...")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_file, data_dir=data_dir, metadata=metadata)
    
    # Process files in parallel
    with Pool(num_processes) as pool:
        for results in tqdm(pool.imap_unordered(process_func, compressed_files), 
                          total=len(compressed_files)):
            if results:
                writer.write_results(results)
    
    # Stop the writer thread
    writer.stop()
    
    print(f"Statistics saved to {output_csv}")
    return pd.read_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute entropy statistics for a dataset.')
    parser.add_argument('--data-dir', required=True, type=str, 
                       help='Path to the directory containing compressed data files.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    src_dir = Path(__file__).resolve().parent
    plots_dir = src_dir / '../plots'
    output_csv = src_dir / 'statistics' / data_dir.name / 'token_entropy.csv'
    
    # Create necessary directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(output_csv.parent, exist_ok=True)

    print("Data directory: " + str(data_dir))
    print("Output CSV: " + str(output_csv))
    
    # Process the dataset and get the statistics DataFrame
    df_statistics = process_dataset(data_dir, output_csv=output_csv)