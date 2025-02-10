# this version of compute_statistics.py is a concurrent version that uses a queue to increase GPU utilization
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset
import gc
from torch.cuda import empty_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from tqdm import tqdm

import os
import sys
import re
import numba
import argparse
from pathlib import Path
import json
from datetime import datetime
import logging
from queue import Queue
from threading import Thread

# --dataset "databricks/databricks-dolly-15k"
# --model "meta-llama/Llama-3.2-1B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

###############################################################################
#                        OUTPUT DIRECTORY UTILS                                #
###############################################################################
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
    pattern = (r".*/(?P<dataset>[^/]+)/(?P<model>[^/]+)/"
               r"max_tokens_(?P<max_new_tokens>\d+)_"
               r"batch_(?P<batch_size>\d+)_"
               r"quantize_(?P<quantize>True|False)_\d+")
    
    match = re.match(pattern, path)
    if match:
        parsed_data = match.groupdict()
        parsed_data["dataset"] = parsed_data["dataset"].replace("_", "/")
        parsed_data["model"] = parsed_data["model"].replace("_", "/")
        parsed_data["max_new_tokens"] = int(parsed_data["max_new_tokens"])
        parsed_data["batch_size"] = int(parsed_data["batch_size"])
        parsed_data["quantize"] = parsed_data["quantize"] == "True"
        return parsed_data
    else:
        raise ValueError("Path format is incorrect")

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
        Tuple: (idx, prompt, category, List[Dict])
    """
    idx, prompt, category, token_distributions, token_ids = item

    token_stats = analyze_token_distributions_jit(token_distributions, bucket_assignments_dict)

    for i, stat in enumerate(token_stats):
        token_id = token_ids[i]
        stat['token_id'] = token_id
        stat['token_text'] = tokenizer.decode([token_id])
        stat['is_eos'] = (token_id == tokenizer.eos_token_id)
        
    return idx, prompt, category, token_stats

###############################################################################
#                     GPU Inference Producer Function                       #
###############################################################################
def gpu_inference_producer(dataset, model, tokenizer, max_new_tokens, batch_size, start_idx, inference_queue):
    """
    Reads the dataset starting at start_idx, processes batches on the GPU,
    and puts the resulting inference data (distributions, token_ids, etc.)
    into a thread-safe queue.
    """
    batch_prompts = []
    batch_indices = []
    batch_categories = []

    for idx, entry in enumerate(tqdm(dataset['train'], desc="GPU Inference"), start=0):
        if idx < start_idx:
            continue
        prompt = entry['instruction']
        category = entry.get('category', 'unknown')
        batch_prompts.append(prompt)
        batch_indices.append(idx)
        batch_categories.append(category)

        if len(batch_prompts) == batch_size:
            # Perform GPU inference on the current batch.
            batch_distributions, batch_token_ids, current_batch_size = get_batch_token_distributions(
                batch_prompts, model, tokenizer, max_new_tokens
            )
            # You might update batch_size here if needed (in case of OOM).
            # Put the batch results into the queue.
            inference_queue.put((batch_indices, batch_prompts, batch_categories,
                                 batch_distributions, batch_token_ids))
            # Clear the current batch.
            batch_prompts = []
            batch_indices = []
            batch_categories = []

    # Process any leftover prompts.
    if batch_prompts:
        batch_distributions, batch_token_ids, _ = get_batch_token_distributions(
            batch_prompts, model, tokenizer, max_new_tokens
        )
        inference_queue.put((batch_indices, batch_prompts, batch_categories,
                             batch_distributions, batch_token_ids))
    # Signal termination to the consumers.
    inference_queue.put(None)

###############################################################################
#                              MAIN FUNCTION                                #
###############################################################################
def main():
    # python concurrent_compute_statistics.py --dataset "databricks/databricks-dolly-15k" --model "meta-llama/Llama-3.2-1B-Instruct" --max_new_tokens 200 --batch_size 128

    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="Eagerly analyze token distributions.")
    parser.add_argument('--dataset', type=str, help='Name or path of the dataset.')
    parser.add_argument('--model', type=str, help='Name or path of the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing prompts.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of tokens to generate per prompt.')
    parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization for the model.')
    parser.add_argument('--num_workers', type=int, default=80, help='Number of parallel workers for analysis.')
    parser.add_argument('-o', '--output_dir', type=str, help='Provide an output directory to resume from a previous run.')
    args = parser.parse_args()

    # Set up output directory, checkpoint, and metadata as before.
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    if args.output_dir:
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
    print(f"Output Directory: {output_dir}")
    print(f"Utilizing GPUs: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # 1) Load dataset
    dataset = load_dataset(args.dataset)
    
    # 2) Load model & tokenizer
    print("Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
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
    
    # 3) Precompute bucket assignments (reuse same assignments for all prompts)
    bucket_sizes = [2**i for i in range(1, 15)]
    bucket_assignments_dict = {}
    for num_buckets in bucket_sizes:
        seed = num_buckets
        bucket_assignments_dict[num_buckets] = assign_buckets(vocab_size, num_buckets, seed=seed)
    
    # 4) Load progress (checkpoint) and existing metadata; compute last contiguous index.
    def compute_last_contiguous_index(processed_indices):
        last_contiguous = -1
        for idx in sorted(processed_indices):
            if idx == last_contiguous + 1:
                last_contiguous = idx
            else:
                break
        return last_contiguous

    loaded_checkpoint = load_checkpoint(checkpoint_file) 
    metadata = load_existing_metadata(metadata_file)
    unique_processed_indices = {entry['idx'] for entry in metadata}
    last_processed_idx = compute_last_contiguous_index(unique_processed_indices)
    processed_set = set(unique_processed_indices)

    print(f"Resuming from index {last_processed_idx + 1}")
    print(f"Total prompts processed (metadata): {len(unique_processed_indices)}")
    if len(unique_processed_indices) != loaded_checkpoint:
        print("Warning: The dataset is only partially processed.")
    # Clean metadata to remove any entries after the last contiguous index.
    metadata = [entry for entry in metadata if entry['idx'] < last_processed_idx]

    # 5) Create a thread-safe queue for passing inference results.
    inference_queue = Queue(maxsize=10)  # Adjust maxsize as appropriate.

    # Start the GPU inference producer thread.
    gpu_thread = Thread(target=gpu_inference_producer, args=(
        dataset, model, tokenizer, args.max_new_tokens, args.batch_size,
        last_processed_idx + 1, inference_queue
    ))
    gpu_thread.start()

    # Set up a ThreadPoolExecutor for CPU analysis.
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # We'll store futures for CPU analysis.
        cpu_futures = []
        
        # Continuously pull inference batches from the queue.
        while True:
            item = inference_queue.get()
            if item is None:
                # No more inference results.
                break
            batch_indices, batch_prompts, batch_categories, batch_distributions, batch_token_ids = item
            
            # For each prompt in the batch, submit a CPU analysis task.
            for i in range(len(batch_indices)):
                cpu_item = (batch_indices[i], batch_prompts[i], batch_categories[i],
                            batch_distributions[i], batch_token_ids[i])
                fut = executor.submit(analyze_worker, cpu_item, bucket_assignments_dict, tokenizer)
                cpu_futures.append(fut)
            
            # Optionally, you can process completed CPU analysis tasks here to update
            # metadata and the checkpoint incrementally rather than waiting until all futures complete.
            # For simplicity, we show a basic example below.
            done_futures = []
            for fut in as_completed(cpu_futures, timeout=5):
                try:
                    res_idx, prompt, cat, token_stats = fut.result()
                except Exception as e:
                    print(f"Error analyzing prompt: {e}")
                    continue
                # Update metadata.
                for token_stat in token_stats:
                    metadata.append({
                        "idx": res_idx,
                        "prompt": prompt,
                        "category": cat,
                        "token_index": token_stat.get("token_index"),
                        "token_id": token_stat.get("token_id"),
                        "token_text": token_stat.get("token_text"),
                        "is_eos": token_stat.get("is_eos"),
                        "entropy": token_stat.get("entropy"),
                        **{k: v for k, v in token_stat.items() if k.startswith("auc_")}
                    })
                processed_set.add(res_idx)
                while (last_processed_idx + 1) in processed_set:
                    last_processed_idx += 1
                save_checkpoint(checkpoint_file, last_processed_idx)
                done_futures.append(fut)
            # Remove done futures from the list.
            cpu_futures = [f for f in cpu_futures if f not in done_futures]
            
            # Save metadata after processing each batch.
            save_metadata(metadata_file, metadata)

        # After the producer signals termination, wait for any remaining CPU analysis tasks.
        for fut in as_completed(cpu_futures):
            try:
                res_idx, prompt, cat, token_stats = fut.result()
            except Exception as e:
                print(f"Error analyzing prompt: {e}")
                continue
            for token_stat in token_stats:
                metadata.append({
                    "idx": res_idx,
                    "prompt": prompt,
                    "category": cat,
                    "token_index": token_stat.get("token_index"),
                    "token_id": token_stat.get("token_id"),
                    "token_text": token_stat.get("token_text"),
                    "is_eos": token_stat.get("is_eos"),
                    "entropy": token_stat.get("entropy"),
                    **{k: v for k, v in token_stat.items() if k.startswith("auc_")}
                })
            processed_set.add(res_idx)
            while (last_processed_idx + 1) in processed_set:
                last_processed_idx += 1
            save_checkpoint(checkpoint_file, last_processed_idx)
        # Save final metadata.
        save_metadata(metadata_file, metadata)
    
    gpu_thread.join()
    print("Completed!")
    print(f"Saved to {output_dir}")
    breakpoint()

if __name__ == "__main__":
    main()
