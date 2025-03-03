import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset
import gc
from torch.cuda import empty_cache
import multiprocessing as mp
from multiprocessing import Queue, Process, Value
from threading import Thread
from queue import Empty
from contextlib import contextmanager
from tqdm import tqdm

import os
import time
import sys
import re
import numba
import argparse
from pathlib import Path
import json
from datetime import datetime
import logging

# --dataset "databricks/databricks-dolly-15k"
# --model "meta-llama/Llama-3.2-1B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

###############################################################################
#                       MULTIPROCESSING UTILS                                 #
###############################################################################

def gpu_producer(dataset, model, tokenizer, args, input_queue, output_queue, stop_event, last_processed_idx, missing_indices, pbar, active_workers):
    """Process batches on GPU and feed to CPU workers"""
    try:
        if missing_indices:
            current_missing_idx = 0
            while current_missing_idx < len(missing_indices) and not stop_event.is_set():
                batch_prompts = []
                batch_indices = []
                batch_categories = []
                
                while (len(batch_prompts) < args.batch_size and current_missing_idx < len(missing_indices)):
                    idx = missing_indices[current_missing_idx]
                    entry = dataset['train'][idx]
                    prompt = entry['instruction']
                    category = entry.get('category', 'unknown')
                    
                    batch_prompts.append(prompt)
                    batch_indices.append(idx)
                    batch_categories.append(category)
                    current_missing_idx += 1

                if batch_prompts:
                    # Process batch on GPU
                    batch_distributions, batch_token_ids, current_batch_size = get_batch_token_distributions(
                        batch_prompts, model, tokenizer, args.max_new_tokens
                    )
                    
                    # Queue work to CPU workers
                    for i in range(len(batch_distributions)):
                        while input_queue.qsize() >= args.batch_size * 2 and not stop_event.is_set():
                            time.sleep(0.1)
                        
                        if stop_event.is_set():
                            break
                            
                        item = (batch_indices[i], batch_prompts[i], batch_categories[i],
                               batch_distributions[i], batch_token_ids[i])
                        input_queue.put(item)
                        
                    pbar.set_description(f"Recovery - Active CPU Workers: {active_workers.value}, GPU→CPU: {input_queue.qsize()}, CPU→Disk: {output_queue.qsize()}")
            
        current_idx = last_processed_idx + 1
        while current_idx < len(dataset['train']) and not stop_event.is_set():
            # Prepare next batch
            batch_prompts = []
            batch_indices = []
            batch_categories = []
            
            # Fill batch
            while len(batch_prompts) < args.batch_size and current_idx < len(dataset['train']):
                entry = dataset['train'][current_idx]
                prompt = entry['instruction']
                category = entry.get('category', 'unknown')
                batch_prompts.append(prompt)
                batch_indices.append(current_idx)
                batch_categories.append(category)
                current_idx += 1

            if batch_prompts:
                # Process batch on GPU
                batch_distributions, batch_token_ids, current_batch_size = get_batch_token_distributions(
                    batch_prompts, model, tokenizer, args.max_new_tokens
                )
                
                # Distribute work to CPU workers
                for i in range(len(batch_distributions)):
                    while input_queue.qsize() >= args.batch_size * 2 and not stop_event.is_set():
                        # If input queue is full, wait a bit
                        time.sleep(0.1)
                    
                    if stop_event.is_set():
                        break
                        
                    item = (batch_indices[i], batch_prompts[i], batch_categories[i],
                           batch_distributions[i], batch_token_ids[i])
                    input_queue.put(item)
                    
                # Update progress description with queue sizes and active workers
                pbar.set_description(f"Active CPU Workers: {active_workers.value}, GPU→CPU: {input_queue.qsize()}, CPU→Disk: {output_queue.qsize()}")

    finally:
        # Signal completion to workers
        for _ in range(args.num_workers):
            input_queue.put(None)

def progress_monitor(input_queue, output_queue, active_workers, pbar, stop_event):
    """Monitor and update progress bar description"""
    while not stop_event.is_set():
        try:
            pbar.set_description(f"Active CPU Workers: {active_workers.value}, GPU→CPU: {input_queue.qsize()}, CPU→Disk: {output_queue.qsize()}")
            time.sleep(1)  # Update every 100ms
        except Exception as e:
            print(f"Error updating progress bar: {e}")
            continue

def cpu_worker(worker_id, input_queue, output_queue, bucket_assignments_dict, tokenizer, active_workers):
    """Process CPU-intensive analysis work"""
    while True:
        try:
            item = input_queue.get()  # Remove timeout
            if item is None:  # Poison pill
                break

            with active_workers.get_lock():
                active_workers.value += 1

            idx, prompt, category, distributions, token_ids = item
            try:
                result = analyze_worker(
                    (idx, prompt, category, distributions, token_ids),
                    bucket_assignments_dict,
                    tokenizer
                )
                output_queue.put((idx, result))
            except Exception as e:
                print(f"Worker {worker_id} error processing idx={idx}: {e}")
                output_queue.put((idx, (idx, "", "unknown", [])))
            finally:
                with active_workers.get_lock():
                    active_workers.value -= 1

        except Empty:
            continue

def result_writer(output_queue, metadata, metadata_file, checkpoint_file, processed_set, 
                 stop_event, pbar):
    """Write results to disk as they complete"""
    while not (stop_event.is_set() and output_queue.empty()):
        try:
            idx, result = output_queue.get(timeout=1)
            prompt, cat, token_stats = result
            
            for token_stat in token_stats:
                metadata.append({
                    "idx": idx,
                    "prompt": prompt,
                    "category": cat,
                    "token_index": token_stat.get("token_index"),
                    "token_id": token_stat.get("token_id"),
                    "token_text": token_stat.get("token_text"),
                    "is_eos": token_stat.get("is_eos"),
                    "entropy": token_stat.get("entropy"),
                    **{k: v for k, v in token_stat.items() if k.startswith("auc_")}
                })
            
            processed_set.add(idx)
            pbar.update(1)
            
            if len(processed_set) % 100 == 0:
                save_metadata(metadata_file, metadata)
                save_checkpoint(checkpoint_file, max(processed_set))
                print(f"Saved metadata and checkpoint for {max(processed_set)} entries")
                
        except Empty:
            continue

################################################################################
#                        OUTPUT DIRECTORY UTILS                                #
################################################################################

def create_output_dir(base_dir, dataset, model, max_new_tokens, batch_size, quantize):
    """
    Create a structured output directory for storing experiment results.

    Args:
        base_dir (str or Path): Base directory where results will be stored.
        dataset (str): Name of the dataset.
        model (str): Name of the model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        batch_size (int): Batch size used in the experiment.
        quantize (bool): Whether quantization was enabled.

    Returns:
        Path: The created output directory.
    """
    # Ensure base directory is a Path object
    base_dir = Path(base_dir)

    # Format quantize flag as string (e.g., "True" -> "quantize_True")
    quantize_flag = f"quantize_{quantize}"

    # Create a timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Format output directory path
    output_dir = base_dir / "statistics" / dataset.replace("/", "_") / model.replace("/", "_") / f"max_tokens_{max_new_tokens}_batch_{batch_size}_{quantize_flag}_{timestamp}"

    # Create the directory
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

def parse_hyperparameters(path):
    """
    Parse hyperparameters from a statistics output directory path.
    """
    # Convert path to string if it's a Path object
    path = str(path)
    
    # Split path into components
    parts = path.split('/')
    
    # Find the relevant parts (after 'statistics')
    try:
        stats_idx = parts.index('statistics')
        dataset_part = parts[stats_idx + 1]
        model_part = parts[stats_idx + 2]
        config_part = parts[stats_idx + 3]
    except (ValueError, IndexError):
        raise ValueError("Path must contain 'statistics' directory and required components")

    # Parse configuration string
    config_pattern = r"max_tokens_(\d+)_batch_(\d+)_quantize_(True|False)_\d+T\d+"
    config_match = re.match(config_pattern, config_part)
    if not config_match:
        raise ValueError("Invalid configuration format in path")
    
    # Extract values
    max_new_tokens = int(config_match.group(1))
    batch_size = int(config_match.group(2))
    quantize = config_match.group(3) == "True"
    
    # Convert component names back to original format
    dataset = dataset_part.replace('_', '/', 1)  # Only replace first underscore
    model = model_part.replace('_', '/', 1)      # Only replace first underscore
    
    return {
        "dataset": dataset,
        "model": model,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "quantize": quantize
    }

###############################################################################
#                        BUCKET ASSIGNMENT UTILS                              #
###############################################################################
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

def get_buckets(vocab_size, num_buckets, seed=None):
    """
    Precomputes bucket assignments for multiple bucket sizes.
    """
    bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
    bucket_assignments_dict = {}
    print("Precomputing bucket assignments...")
    for num_buckets in bucket_sizes:
        seed = num_buckets  # each bucket size gets a different seed
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments

###############################################################################
#                            STATISTICS FUNCTIONS                             #
###############################################################################  

def compute_auc(token_probs, bucket_assignments, num_buckets):
    """
    Computes the "AUC" statistic for a single token's probability distribution.

    Args:
        token_probs (np.ndarray): Array of token probabilities (size: vocab_size).
        bucket_assignments (np.ndarray): Array of bucket assignments (size: vocab_size).
        num_buckets (int): Number of buckets.

    Returns:
        float: Computed AUC-like statistic.
    """
    # Sum probabilities per bucket
    w_i = np.bincount(bucket_assignments, weights=token_probs, minlength=num_buckets)
    max_vals = np.maximum(0, 1/num_buckets - w_i)
    return np.sum(max_vals)

@numba.njit
def compute_auc_jit(token_probs, bucket_assignments, num_buckets):
    """
    Computes the "AUC" statistic for a single token's probability distribution. (numba version)
    """
    # Allocate an array for bucket sums for one token
    bucket_sums = np.zeros(num_buckets, dtype=np.float64)
    for i in range(token_probs.shape[0]):
        bucket_sums[bucket_assignments[i]] += token_probs[i]
    auc = 0.0
    for bucket in range(num_buckets):
        # The AUC contribution for this bucket
        diff = 1/num_buckets - bucket_sums[bucket]
        if diff > 0:
            auc += diff
    return auc

def compute_entropy(dists):
    """
    Compute entropy for a batch of token distributions.
    
    Args:
        dists (np.ndarray): Array of shape (n_tokens, vocab_size)
        
    Returns:
        np.ndarray: Entropy values for each token distribution.
    """
    safe_dists = np.where(dists > 0, dists, 1)
    return -np.sum(dists * np.log2(safe_dists), axis=1)

###############################################################################
#                          CHECKPOINT & METADATA                              #
###############################################################################
def load_checkpoint(checkpoint_file):
    """Load the last processed index from the checkpoint file."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)['last_processed_idx']
    return -1

def save_checkpoint(checkpoint_file, idx):
    """Save the last processed index into the checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_processed_idx': idx}, f)

def load_existing_metadata(metadata_file):
    """Load metadata CSV if it exists."""
    if metadata_file.exists():
        return pd.read_csv(metadata_file).to_dict('records')
    return []

def save_metadata(metadata_file, metadata):
    """Save the metadata as a CSV."""
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(metadata_file, index=False)

###############################################################################
#                          CUDA MEMORY CONTEXT                                #
###############################################################################
@contextmanager
def cuda_memory_manager():
    """Context manager to handle CUDA memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        empty_cache()

###############################################################################
#                     TOKEN DISTRIBUTION & ANALYSIS                           #
###############################################################################
def get_batch_token_distributions(prompts, model, tokenizer, max_new_tokens=50):
    """
    Generate token distributions for a batch of prompts.
    
    Args:
        prompts (List[str]): Batch of prompt strings
        model: Loaded language model
        tokenizer: Corresponding tokenizer
        max_new_tokens (int): Maximum number of tokens to generate per prompt
    
    Returns:
        Tuple[
            List[List[np.ndarray]],  # List[prompt_idx][token_idx] -> probability distribution
            List[List[int]],         # List[prompt_idx][token_idx] -> token_id
            int                      # Updated batch size
        ]
    """
    def process_batch(batch_prompts):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with cuda_memory_manager():
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

            # Get generated token IDs (excluding input sequence)
            input_length = inputs.input_ids.shape[1]
            generated_token_ids = [
                seq[input_length:].tolist() for seq in outputs.sequences
            ]
            
            # Organize distributions by prompt then by token
            # outputs.logits is tuple of tensors, each shape [batch_size, vocab_size]
            all_distributions = []
            
            # First, organize by prompt
            for prompt_idx in range(len(batch_prompts)):
                prompt_distributions = []
                # Then by token position
                for token_logits in outputs.logits:
                    # Get logits for this prompt and convert to probabilities
                    token_probs = torch.softmax(token_logits[prompt_idx], dim=-1).cpu().numpy()
                    prompt_distributions.append(token_probs)
                all_distributions.append(prompt_distributions)
            
            return all_distributions, generated_token_ids

    original_batch_size = len(prompts)
    current_batch_size = original_batch_size
    
    while current_batch_size > 0:
        try:
            all_distributions = []
            all_token_ids = []
            
            # Process prompts in chunks
            for i in range(0, original_batch_size, current_batch_size):
                batch_prompts = prompts[i:min(i + current_batch_size, original_batch_size)]
                batch_distributions, batch_token_ids = process_batch(batch_prompts)
                all_distributions.extend(batch_distributions)
                all_token_ids.extend(batch_token_ids)
                torch.cuda.empty_cache()
            
            return all_distributions, all_token_ids, current_batch_size
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            
            if current_batch_size == 1:
                logging.error("Out of memory even with batch size 1. Consider reducing max_new_tokens.")
                sys.exit(1)
            
            current_batch_size = (current_batch_size + 1) // 2
            print(f"OOM error, reducing batch size to: {current_batch_size}")
            continue
        
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            sys.exit(1)

def analyze_token_distributions(token_distributions, bucket_assignments_dict):
    """
    Compute per-token entropy and AUC for each bucket size.
    
    Args:
        token_distributions (List[np.ndarray]):
            A list of length #generated_tokens, each shape=[vocab_size].
        bucket_assignments_dict (Dict[int, np.ndarray]):
            Maps num_buckets -> an array of bucket assignments for each token in vocab.
    
    Returns:
        List[Dict]: List of dictionaries containing 'entropy' and 'auc_{num_buckets}' per token.
    """
    if token_distributions is None:
        return []
    
    dists = np.stack(token_distributions, axis=0)  # shape: (n_tokens, vocab_size)
    entropies = compute_entropy(dists)
    
    token_stats = []
    for token_idx, dist in enumerate(token_distributions):
        # Use the vectorized entropy if available.
        entropy_val = entropies[token_idx]
        
        # Compute AUC for each bucket size.
        auc_vals = {}
        for num_buckets, bucket_assignments in bucket_assignments_dict.items():
            auc = compute_auc(dist, bucket_assignments, num_buckets)
            auc_vals[f"auc_{num_buckets}"] = auc
        
        token_stat = {"token_index": token_idx, "entropy": entropy_val}
        token_stat.update(auc_vals)
        token_stats.append(token_stat)
    
    return token_stats

def analyze_token_distributions_jit(token_distributions, bucket_assignments_dict):
    token_stats = []
    for token_idx, dist in enumerate(token_distributions):
        entropy_val = compute_entropy(np.expand_dims(dist, axis=0))[0]
        auc_vals = {}
        for num_buckets, bucket_assignments in bucket_assignments_dict.items():
            auc_vals[f"auc_{num_buckets}"] = compute_auc_jit(dist, bucket_assignments, num_buckets)
        token_stat = {"token_index": token_idx, "entropy": entropy_val}
        token_stat.update(auc_vals)
        token_stats.append(token_stat)
    return token_stats

def analyze_worker(item, bucket_assignments_dict, tokenizer):
    """
    Worker function for parallel analysis.
    
    Args:
        item (Tuple): (idx, prompt, category, token_distributions, token_ids)
            - token_distributions: List[token_idx] -> probability distribution
            - token_ids: List[token_idx] -> token_id
        bucket_assignments_dict (Dict[int, np.ndarray]):
            Maps num_buckets -> an array of bucket assignments for each token in vocab.
        tokenizer: Tokenizer for decoding tokens
    
    Returns:
        Tuple: (prompt, category, List[Dict])
    """
    idx, prompt, category, token_distributions, token_ids = item

    token_stats = analyze_token_distributions_jit(token_distributions, bucket_assignments_dict)

    for i, stat in enumerate(token_stats):
        token_id = token_ids[i]
        stat['token_id'] = token_id
        stat['token_text'] = tokenizer.decode([token_id])
        stat['is_eos'] = (token_id == tokenizer.eos_token_id)
        
    return prompt, category, token_stats

###############################################################################
#                                  MAIN                                       #
###############################################################################
def main():
    # python compute_statistics.py --dataset "databricks/databricks-dolly-15k" --model "meta-llama/Llama-3.2-1B-Instruct" --max_new_tokens 200 --batch_size 128

    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="Eagerly analyze token distributions.")
    parser.add_argument('--dataset', type=str, help='Name or path of the dataset.')
    parser.add_argument('--model', type=str, help='Name or path of the model.')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing prompts.')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate per prompt.')
    parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization for the model.')
    parser.add_argument('--num_workers', type=int, default=80, help='Number of parallel workers for analysis.')
    parser.add_argument('-o', '--output_dir', type=str, help='Provide an output directory to resume from a previous run.')
    args = parser.parse_args()

    # Output dirctory for logs, checkpoints, and results
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    if args.output_dir:
        # parse output_dir path name
        output_dir = Path(args.output_dir)
        hyperparams = parse_hyperparameters(args.output_dir)
        args.dataset = hyperparams['dataset']
        args.model = hyperparams['model']
        args.max_new_tokens = hyperparams['max_new_tokens']
        if args.batch_size is None:
            args.batch_size = hyperparams['batch_size']
        args.quantize = hyperparams['quantize']
    else:
        output_dir = create_output_dir(root_dir, args.dataset, args.model, args.max_new_tokens, args.batch_size, args.quantize)

    checkpoint_file = output_dir / "checkpoint.json"
    metadata_file = output_dir / "metadata.csv"

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Output Directory: \n {output_dir}")
    print(f"Utilizing GPUs: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("-"*100)

    #---------------------------------------------------------------------------
    # 1) Load dataset
    #---------------------------------------------------------------------------
    dataset = load_dataset(args.dataset)
    
    #---------------------------------------------------------------------------
    # 2) Load progress (checkpoint) and existing metadata
    #---------------------------------------------------------------------------
    print("Loading progress (checkpoint) and existing metadata...")
    def compute_last_contiguous_index(processed_indices):
        last_contiguous = -1
        for idx in sorted(processed_indices):
            if idx == last_contiguous + 1:
                last_contiguous = idx
            else:
                break
        return last_contiguous

    # loaded_checkpoint = load_checkpoint(checkpoint_file) 
    metadata = load_existing_metadata(metadata_file)

    unique_processed_indices = {entry['idx'] for entry in metadata}
    # last_contiguous_index = compute_last_contiguous_index(unique_processed_indices)
    processed_set = set(unique_processed_indices)
    last_processed_idx = max(unique_processed_indices)

    # print(f"Last contiguous index: {last_contiguous_index}")
    print(f"Last processed index: {last_processed_idx}")
    print(f"Total prompts processed (metadata): {len(unique_processed_indices)}")

    missing_indices = set(range(last_processed_idx + 1)) - unique_processed_indices
    missing_indices = sorted(list(missing_indices))
    if len(missing_indices) > 0:
        print("Warning: checkpoint is out of sync with metadata (missing indices)")
        print(f"Missing indicies: {missing_indices}")
        print(f"Let's first process the missing indices...")
    else:
        print("\n\n****Checkpoint is in sync with metadata****")
    print("-"*100)
    
    #---------------------------------------------------------------------------
    # 3) Load model & tokenizer
    #---------------------------------------------------------------------------
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
    
    if args.quantize:
        print("Loading quantized model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    
    print(f"Model loaded with {'8-bit quantization' if args.quantize else 'full precision'}")
    
    #---------------------------------------------------------------------------
    # 4) Precompute bucket assignments for multiple bucket sizes, using same hash for every prompt
    #---------------------------------------------------------------------------
    bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
    bucket_assignments_dict = {}
    for num_buckets in bucket_sizes:
        seed = num_buckets  # each bucket size gets a different seed
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments

    #---------------------------------------------------------------------------
    # 5) Batch process the dataset
    #---------------------------------------------------------------------------
    # Initialize progress bar
    total_missing = len(missing_indices)
    total_new = len(dataset['train']) - (last_processed_idx + 1)
    total_remaining = total_missing + total_new
    print(f"Total remaining: {total_remaining}")
    pbar = tqdm(total=total_remaining, initial=0, desc="Processing", unit="samples")

    input_queue = Queue(maxsize=args.batch_size * 3)  
    output_queue = Queue()
    stop_event = mp.Event()

    # Initialize shared counter for active workers
    active_workers = Value('i', 0)

    workers = []
    for i in range(args.num_workers):
        p = Process(target=cpu_worker, 
                   args=(i, input_queue, output_queue, bucket_assignments_dict, tokenizer, active_workers))
        p.start()
        workers.append(p)

    # Start progress monitor thread
    monitor = Thread(target=progress_monitor,
                    args=(input_queue, output_queue, active_workers, pbar, stop_event))
    monitor.daemon = True  # Thread will be killed when main process exits
    monitor.start()

    writer = Thread(target=result_writer,
               args=(output_queue, metadata, metadata_file, checkpoint_file, 
                     processed_set, stop_event, pbar))
    writer.start()

    try:
        gpu_producer(dataset, model, tokenizer, args, input_queue, output_queue, 
                    stop_event, last_processed_idx, missing_indices, pbar, active_workers)
        
        for w in workers:
            w.join()
        
        stop_event.set()
        writer.join()
        monitor.join()  # Wait for monitor thread to finish

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        stop_event.set()
        for w in workers:
            w.terminate()
        writer.join()
        monitor.join()  # Wait for monitor thread to finish

    pbar.close()
    print("Completed!")
    print(f"Saved to {output_dir}")

    breakpoint()

if __name__ == "__main__":
    main()
