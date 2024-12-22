import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import zstandard as zstd
import pickle
import os
from tqdm import tqdm
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
from torch.cuda import empty_cache
from contextlib import contextmanager

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_new_tokens', type=int, default=50)
parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization')
parser.add_argument('--compression_level', type=int, default=3, help='ZStd compression level (1-22)')
parser.add_argument('--num_compression_workers', type=int, default=4)
args = parser.parse_args()

output_dir = Path(f"/raid/lawrence/compressed_data/{args.dataset.replace('/', '_')}_{args.model.replace('/', '_')}")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = output_dir / "checkpoint.json"
metadata_file = output_dir / "metadata.csv"

# print configuration
print(f"Dataset: {args.dataset}")
print(f"Model: {args.model}")
print(f"Batch Size: {args.batch_size}")
print(f"Max New Tokens: {args.max_new_tokens}")
print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
print(f"Compression Level: {args.compression_level}")
print(f"Compression Workers: {args.num_compression_workers}")
print(f"Output Directory: {output_dir}")

@contextmanager
def cuda_memory_manager():
    """Context manager to handle CUDA memory cleanup"""
    try:
        yield
    finally:
        gc.collect()
        empty_cache()

def load_checkpoint():
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)['last_processed_idx']
    return -1

def save_checkpoint(idx):
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_processed_idx': idx}, f)

def load_existing_metadata():
    if metadata_file.exists():
        return pd.read_csv(metadata_file).to_dict('records')
    return []

def save_metadata(metadata):
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(metadata_file, index=False)

def get_batch_token_distributions(prompts: List[str], model, tokenizer, max_new_tokens: int = 50):
    """Process multiple prompts in a single batch"""
    try:
        # Tokenize all prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with cuda_memory_manager():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Process logits for each sequence in the batch
            batch_probabilities = []
            for sequence_scores in zip(*outputs.scores):  # Transpose to get per-sequence scores
                sequence_probs = [torch.softmax(score, dim=-1).cpu().numpy() for score in sequence_scores]
                batch_probabilities.append(sequence_probs)
            
            breakpoint()
            return batch_probabilities
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return [None] * len(prompts)

def compress_data_worker(data):
    """Worker function for parallel compression"""
    idx, probs = data
    try:
        stored_data = {
            "idx": idx,
            "probs": [prob.tolist() for prob in probs] if probs is not None else []
        }
        
        # Use a fast compression level for better speed
        cctx = zstd.ZstdCompressor(level=args.compression_level)
        return idx, cctx.compress(pickle.dumps(stored_data))
    except Exception as e:
        print(f"Error compressing data for idx {idx}: {e}")
        return idx, None

def store_batch_data(indices: List[int], batch_probs: List[List], output_dir: Path, executor: ThreadPoolExecutor):
    """Store batch data using parallel compression"""
    compression_futures = []
    for idx, probs in zip(indices, batch_probs):
        if probs is not None:
            future = executor.submit(compress_data_worker, (idx, probs))
            compression_futures.append(future)
    
    # Write compressed data as they complete
    successful_indices = []
    for future in compression_futures:
        idx, compressed_data = future.result()
        if compressed_data is not None:
            output_file = output_dir / f'compressed_data_{idx}.zst'
            temp_file = output_file.with_suffix('.temp')
            try:
                with open(temp_file, 'wb') as f:
                    f.write(compressed_data)
                temp_file.rename(output_file)
                successful_indices.append(idx)
            except Exception as e:
                print(f"Error writing data for idx {idx}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
    
    return successful_indices

def main():
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
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
    
    # Load existing progress
    last_processed_idx = load_checkpoint()
    metadata = load_existing_metadata()
    
    print(f"Resuming from index {last_processed_idx + 1}")
    
    # Create thread pool for compression
    with ThreadPoolExecutor(max_workers=args.num_compression_workers) as compression_executor:
        try:
            # Process dataset in batches
            batch_prompts = []
            batch_indices = []
            batch_categories = []
            
            for idx, entry in enumerate(tqdm(dataset['train'])):
                if idx <= last_processed_idx:
                    continue
                
                batch_prompts.append(entry['instruction'])
                batch_indices.append(idx)
                batch_categories.append(entry['category'])
                
                # Process batch when it reaches the specified size
                if len(batch_prompts) == args.batch_size:
                    # Get token distributions for the batch
                    batch_distributions = get_batch_token_distributions(
                        batch_prompts, model, tokenizer, args.max_new_tokens
                    )
                    
                    # Store batch data in parallel
                    successful_indices = store_batch_data(
                        batch_indices, batch_distributions, output_dir, compression_executor
                    )
                    
                    # Update metadata for successful entries
                    for i, idx in enumerate(batch_indices):
                        if idx in successful_indices:
                            metadata_entry = {
                                "idx": idx,
                                "prompt": batch_prompts[i],
                                "category": batch_categories[i],
                                "num_tokens": len(batch_distributions[i]) if batch_distributions[i] else 0
                            }
                            metadata.append(metadata_entry)
                    
                    # Save progress
                    if successful_indices:
                        last_processed_idx = max(successful_indices)
                        save_checkpoint(last_processed_idx)
                        save_metadata(metadata)
                    
                    # Clear batch
                    batch_prompts = []
                    batch_indices = []
                    batch_categories = []
            
            # Process remaining items in the last batch
            if batch_prompts:
                batch_distributions = get_batch_token_distributions(
                    batch_prompts, model, tokenizer, args.max_new_tokens
                )
                successful_indices = store_batch_data(
                    batch_indices, batch_distributions, output_dir, compression_executor
                )
                
                for i, idx in enumerate(batch_indices):
                    if idx in successful_indices:
                        metadata_entry = {
                            "idx": idx,
                            "prompt": batch_prompts[i],
                            "category": batch_categories[i],
                            "num_tokens": len(batch_distributions[i]) if batch_distributions[i] else 0
                        }
                        metadata.append(metadata_entry)
                
                if successful_indices:
                    last_processed_idx = max(successful_indices)
                    save_checkpoint(last_processed_idx)
                    save_metadata(metadata)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")
        except Exception as e:
            print(f"Encountered error: {e}")
        finally:
            # Save final state
            print("Saving final metadata and checkpoint...")
            save_metadata(metadata)
            save_checkpoint(last_processed_idx)
            print("Script completed. Check metadata.csv for processing summary.")

if __name__ == "__main__":
    main()