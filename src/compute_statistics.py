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

import argparse
from pathlib import Path
import json
from datetime import datetime
import logging
import sys

# --dataset "databricks/databricks-dolly-15k"
# --model "meta-llama/Llama-3.2-1B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

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
    Computes the "AUC"-like statistic for a single token's probability distribution.

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

from scipy.stats import entropy

def compute_entropy(token_probs):
    """
    Computes entropy (base-2) using scipy.
    
    Args:
        token_probs (np.ndarray): Array of token probabilities (size: vocab_size).
        
    Returns:
        float: Entropy value.
    """
    return entropy(token_probs, base=2)

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
    If OOM occurs, automatically reduces batch size and processes in smaller batches.
    Returns a list of shape [batch_size, #generated_tokens, vocab_size].
    
    Args:
        prompts (List[str]): Batch of prompt strings.
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        max_new_tokens (int): Maximum number of tokens to generate per prompt.

    Returns:
        Tuple[
            List[List[np.ndarray]],  # Nested list of token probability distributions with shape [batch_size, max_new_tokens, vocab_size].
            int                      # The updated batch size used successfully.
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

            # generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
            # for text in generated_texts:
            #     print(text)

            # outputs.logits is a tuple of max_new_tokens tensors, each shape=[batch_size, vocab_size]
            
            batch_distributions = []
            # iterate over over tokens generated for each prompt
            for seq_scores in zip(*outputs.logits): # seq_scores is a tuple of max_new_tokens tensors of shape [batch_size]
                distributions = []
                for score in seq_scores: # iterate over each token's score (max_new_tokens iterations)
                    probs = torch.softmax(score, dim=-1).cpu().numpy() # softmax over vocab_size to obtain probabilities
                    distributions.append(probs) # appending one token at a time
                # distributions is a list of max_new_tokens tensors of shape [batch_size, vocab_size]
                batch_distributions.append(distributions) 
            
            # batch_distributions is a list of lists 
            # dimension [batch_size, max_new_tokens, vocab_size]
            return batch_distributions

    original_batch_size = len(prompts)
    current_batch_size = original_batch_size
    
    while current_batch_size > 0:
        try:
            all_distributions = []
            # Process all prompts in chunks of current_batch_size
            for i in range(0, original_batch_size, current_batch_size):
                batch_prompts = prompts[i:min(i + current_batch_size, original_batch_size)]
                batch_distributions = process_batch(batch_prompts)
                all_distributions.extend(batch_distributions)
                torch.cuda.empty_cache()
            
            return all_distributions, current_batch_size
            
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
    
    token_stats = []
    
    for token_idx, dist in enumerate(token_distributions):
        # Compute entropy
        entropy_val = compute_entropy(dist)
        
        # Compute AUC for each bucket size
        auc_vals = {}
        for num_buckets, bucket_assignments in bucket_assignments_dict.items():
            auc = compute_auc(dist, bucket_assignments, num_buckets)
            auc_vals[f"auc_{num_buckets}"] = auc
        
        # Combine entropy and AUCs
        token_stat = {"token_index": token_idx, "entropy": entropy_val}
        token_stat.update(auc_vals)
        token_stats.append(token_stat)
    
    return token_stats  # List of dicts

def analyze_worker(item, vocab_size):
    """
    Worker function for parallel analysis.

    Args:
        item (Tuple): (idx, prompt, category, token_distributions)
        vocab_size (int): Number of tokens in the vocabulary.
    Returns:
        Tuple: (idx, prompt, category, List[Dict])
    """
    idx, prompt, category, token_distributions = item

    # generate a fresh set of bucket assignments for each prompt
    bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
    bucket_assignments_dict = {}
    for num_buckets in bucket_sizes:
        seed = num_buckets  # each bucket size gets a different seed
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments
    
    token_stats = analyze_token_distributions(token_distributions, bucket_assignments_dict)
    return idx, prompt, category, token_stats

###############################################################################
#                                  MAIN                                       #
###############################################################################
def main():
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(description="Eagerly analyze token distributions.")
    parser.add_argument('--dataset', type=str, required=True, help='Name or path of the dataset.')
    parser.add_argument('--model', type=str, required=True, help='Name or path of the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing prompts.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of tokens to generate per prompt.')
    parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization for the model.')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of parallel workers for analysis.')
    parser.add_argument('--output_dir', type=str, help='Output directory for logs, checkpoints, and results.')
    args = parser.parse_args()

    # Output directory for logs, checkpoints, and results
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = create_output_dir(root_dir, args.dataset, args.model, args.max_new_tokens, args.batch_size, args.quantize)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "checkpoint.json"
    metadata_file = output_dir / "metadata.csv"

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Output Directory: {output_dir}")

    #---------------------------------------------------------------------------
    # 1) Load dataset
    #---------------------------------------------------------------------------
    dataset = load_dataset(args.dataset)
    
    #---------------------------------------------------------------------------
    # 2) Load model & tokenizer
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
    
    # #---------------------------------------------------------------------------
    # # 3) Precompute bucket assignments for multiple bucket sizes
    # #---------------------------------------------------------------------------
    # bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
    # bucket_assignments_dict = {}
    # print("Precomputing bucket assignments...")
    # for num_buckets in bucket_sizes:
    #     seed = num_buckets  # each bucket size gets a different seed
    #     bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
    #     bucket_assignments_dict[num_buckets] = bucket_assignments
    
    #---------------------------------------------------------------------------
    # 4) Load progress (checkpoint) and existing metadata
    #---------------------------------------------------------------------------
    last_processed_idx = load_checkpoint(checkpoint_file)
    metadata = load_existing_metadata(metadata_file)
    print(f"Resuming from index {last_processed_idx + 1}")
    
    #---------------------------------------------------------------------------
    # 5) Batch process the dataset
    #---------------------------------------------------------------------------
    batch_prompts = []
    batch_indices = []
    batch_categories = []
    
    # We'll store our futures in a dict so that we can gather results as soon
    # as they complete, then update metadata/checkpoints incrementally.
    future_to_idx = {}

    batch_size = args.batch_size
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for idx, entry in enumerate(tqdm(dataset['train'], desc="Processing Prompts")):
            if idx <= last_processed_idx:
                continue
            
            # Prepare next batch
            prompt = entry['instruction']
            category = entry.get('category', 'unknown')  # Use 'unknown' if 'category' is missing
            
            batch_prompts.append(prompt)
            batch_indices.append(idx)
            batch_categories.append(category)
            
            # Process once we hit batch_size
            if len(batch_prompts) == batch_size:
                # Generate token distributions for this batch
                batch_distributions, current_batch_size = get_batch_token_distributions(
                    batch_prompts, model, tokenizer, args.max_new_tokens
                )

                batch_size = current_batch_size # Update batch size in case of OOM
                
                # Submit each sequence in the batch for analysis
                for i, dist in enumerate(batch_distributions): 
                    item = (batch_indices[i], batch_prompts[i], batch_categories[i], dist)
                    # submit one prompt at a time
                    fut = executor.submit(analyze_worker, item, vocab_size)
                    future_to_idx[fut] = batch_indices[i]
                
                # As each future completes, retrieve results
                done_futures = []
                for fut in as_completed(future_to_idx):
                    idx_done = future_to_idx[fut]
                    done_futures.append(fut)
                    try:
                        res_idx, prompt, cat, token_stats = fut.result()
                    except Exception as e:
                        print(f"Error analyzing idx={idx_done}: {e}")
                        res_idx, prompt, cat, token_stats = idx_done, "", "unknown", []
                    
                    # Append each token's stats as separate metadata entries
                    for token_stat in token_stats:
                        metadata.append({
                            "idx": res_idx,
                            "prompt": prompt,
                            "category": cat,
                            "token_index": token_stat.get("token_index"),
                            "entropy": token_stat.get("entropy"),
                            **{k: v for k, v in token_stat.items() if k.startswith("auc_")}
                        })
                    
                    # Update checkpoint: the highest index that we have processed
                    if res_idx > last_processed_idx:
                        last_processed_idx = res_idx
                        save_checkpoint(checkpoint_file, last_processed_idx)
                
                # Remove completed futures from the dict
                for fut in done_futures:
                    del future_to_idx[fut]
                
                # Save updated metadata after each batch
                save_metadata(metadata_file, metadata)
                
                # Clear the batch
                batch_prompts = []
                batch_indices = []
                batch_categories = []
        
        #-----------------------------------------------------------------------
        # 6) Process any leftover items in the final (incomplete) batch
        #-----------------------------------------------------------------------
        if batch_prompts:
            batch_distributions, _ = get_batch_token_distributions(
                batch_prompts, model, tokenizer, args.max_new_tokens
            )
            for i, dist in enumerate(batch_distributions):
                item = (batch_indices[i], batch_prompts[i], batch_categories[i], dist)
                fut = executor.submit(analyze_worker, item, vocab_size)
                future_to_idx[fut] = batch_indices[i]
            
            # Collect final results
            for fut in as_completed(future_to_idx):
                idx_done = future_to_idx[fut]
                try:
                    res_idx, prompt, cat, token_stats = fut.result()
                except Exception as e:
                    print(f"Error analyzing idx={idx_done}: {e}")
                    res_idx, prompt, cat, token_stats = idx_done, "", "unknown", []
                
                # Append each token's stats as separate metadata entries
                for token_stat in token_stats:
                    metadata.append({
                        "idx": res_idx,
                        "prompt": prompt,
                        "category": cat,
                        "token_index": token_stat.get("token_index"),
                        "entropy": token_stat.get("entropy"),
                        **{k: v for k, v in token_stat.items() if k.startswith("auc_")}
                    })
                
                # Update checkpoint
                if res_idx > last_processed_idx:
                    last_processed_idx = res_idx
                    save_checkpoint(checkpoint_file, last_processed_idx)
            
            # Save updated metadata after final batch
            save_metadata(metadata_file, metadata)
    
    print("Completed!")
    print(f"Saved to {output_dir}")

    breakpoint()

if __name__ == "__main__":
    main()
