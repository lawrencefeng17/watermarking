import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import gc
from torch.cuda import empty_cache
from contextlib import contextmanager
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from scipy.stats import entropy

from compute_statistics_with_entropy import compute_entropy, assign_buckets, compute_statistic

class TokenDistributionAnalyzer:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()
        
        # Check if the model has a chat template
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None

    @contextmanager
    def cuda_memory_manager():
        """Context manager to handle CUDA memory cleanup"""
        try:
            yield
        finally:
            gc.collect()
            empty_cache()

    def format_chat(self, prompt: str, system_prompt: str = "") -> str:
        """Format the conversation using the model's chat template or a default format."""
        if not system_prompt and not self.has_chat_template:
            return prompt
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.has_chat_template:
            # Use the model's built-in chat template
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to a basic template if none exists
            formatted = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted += f"<|system|>\n{msg['content']}\n"
                elif msg["role"] == "user":
                    formatted += f"<|user|>\n{msg['content']}\n"
            formatted += "<|assistant|>\n"
            return formatted
        
    def get_token_distributions(
        self,
        prompt: str,
        system_prompt: str = "",
        num_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> List[torch.Tensor]:
        """Generate token distributions for each position."""
        formatted_prompt = self.format_chat(prompt, system_prompt)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Print the actual tokens for debugging
        print("Formatted prompt tokens:")
        print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        
        distributions = []
        
        with self._manage_memory():
            with torch.no_grad():
                for _ in range(num_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature and top-p sampling
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumsum_probs <= top_p
                        mask[..., 1:] = mask[..., :-1].clone()
                        mask[..., 0] = True
                        probs[sorted_indices[~mask]] = 0.0
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    distributions.append(probs.cpu())
                    
                    # Sample next token
                    next_token = torch.multinomial(probs, 1)
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return distributions

    def analyze_distributions(
        self,
        distributions: List[torch.Tensor],
        bucket_boundaries: List[float] = None
    ) -> Dict:
        """Compute entropy and statistics for token distributions."""
        if bucket_boundaries is None:
            bucket_boundaries = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        entropies = []
        statistics = []
        
        for dist in distributions:
            dist_np = dist.numpy().flatten()
            
            # Compute entropy
            pos_entropy = compute_entropy(dist_np)
            entropies.append(pos_entropy)
            
            # Compute probability distribution statistics
            buckets = assign_buckets(dist_np, bucket_boundaries)
            pos_stats = compute_statistic(buckets)
            statistics.append(pos_stats)
        
        return {
            "entropies": entropies,
            "statistics": statistics,
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "mean_statistic": np.mean(statistics),
            "std_statistic": np.std(statistics)
        }

    def run_analysis(
        self,
        prompts: List[str],
        system_prompts: List[str] = None,
        num_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        n_threads: int = 4
    ) -> pd.DataFrame:
        """Run analysis on multiple prompts and system prompts."""
        if system_prompts is None:
            system_prompts = [""] * len(prompts)
        
        results = []
        
        def process_prompt(args):
            prompt, system_prompt = args
            distributions = self.get_token_distributions(
                prompt, system_prompt, num_tokens, temperature, top_p
            )
            analysis = self.analyze_distributions(distributions)
            return {
                "prompt": prompt,
                "system_prompt": system_prompt,
                **analysis
            }
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for result in tqdm(
                executor.map(process_prompt, zip(prompts, system_prompts)),
                total=len(prompts)
            ):
                results.append(result)
        
        return pd.DataFrame(results)

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    # Load prompts and system prompts from JSON file
    with open(args.prompts_file) as f:
        data = json.load(f)
        prompts = data["prompts"]
        system_prompts = data.get("system_prompts")

    analyzer = TokenDistributionAnalyzer(args.model)
    results_df = analyzer.run_analysis(
        prompts=prompts,
        system_prompts=system_prompts,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n_threads=args.n_threads
    )
    
    results_df.to_csv(args.output_file, index=False)