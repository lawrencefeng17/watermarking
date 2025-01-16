import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
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
from datetime import datetime

from compute_statistics import assign_buckets 
from compute_statistics import analyze_token_distributions

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

# --model "meta-llama/Llama-3.2-1B-Instruct" --prompt "Tell me a creative love story between a robot and a human." --sys "Be creative and helpful."

def format_chat(tokenizer, prompt: str, system_prompt: str = "", has_chat_template=False) -> str:
        """Format the conversation using the model's chat template or a default format."""
        if not system_prompt and not has_chat_template:
            return prompt
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if has_chat_template:
            # Use the model's built-in chat template
            return tokenizer.apply_chat_template(
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

def main():
    prompts = [
            "What is the capital of France?",
            "How do you make a cake?",
            "Tell me an elaborate joke.",
            "What is the meaning of life?",
            "Explain quantum physics in simple terms.",
            "Tell me a creative love story between a robot and a human.",
        ]
        
    system_prompts = [
            "Be creative and helpful.",
            "Be concise and informative.",
            "Be detailed and thorough.",
            "Use flashy language and be engaging.",
            "Use colorful language and be engaging.",
        ] 

    # temperatures = [0.5, 0.7, 1.0, 1.2, 1.4]
    temperatures = [1.0]

    # max_output_tokens = [50, 100, 250, 400]
    max_output_tokens = [50]
    parser = argparse.ArgumentParser(description="Eagerly analyze token distributions.")
    parser.add_argument('--model', type=str, required=True, help='Name or path of the model.')
    parser.add_argument('--quantize', action='store_true', help='Use 8-bit quantization for the model.', default=False)
    parser.add_argument('--prompt', type=str, help='Prompt to analyze.', default=None)
    parser.add_argument('--sys', type=str, help='System prompt to analyze.', default=None)
    # --prompt "Tell me a creative love story between a robot and a human." --sys "Be creative and helpful."
    
    args = parser.parse_args()

    if args.prompt is not None:
        prompts = [args.prompt]
    if args.sys is not None:
        system_prompts = [args.sys]

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

    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    bucket_sizes = [2**i for i in range(1, 15)]  # 2^1 to 2^14
    bucket_assignments_dict = {}
    print("Precomputing bucket assignments...")
    for num_buckets in bucket_sizes:
        seed = num_buckets
        bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)
        bucket_assignments_dict[num_buckets] = bucket_assignments

    results = []
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        for system_prompt in system_prompts:
            for temperature in temperatures:
                for max_output_token in max_output_tokens:
                    try:
                        # Format the conversation
                        formatted_prompt = format_chat(
                            tokenizer,
                            prompt,
                            system_prompt,
                            has_chat_template
                        )

                        
                        # Tokenize the input
                        inputs = tokenizer(
                            formatted_prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(model.device)
                        # print(inputs)
                        
                        # Generate with specified parameters
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_output_token,
                                temperature=temperature,
                                return_dict_in_generate=True,
                                output_scores=True,
                                output_logits=True,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                        # Decode generated text
                        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                        
                        input_length = inputs.input_ids.shape[1]
                        generated_tokens = outputs.sequences[0][input_length:]
                        
                        # Decode each new token individually
                        decoded_tokens = [
                            tokenizer.decode(token.item()) 
                            for token in generated_tokens
                        ]
                        # print(decoded_tokens)

                        # Process token distributions
                        token_distributions = []
                        # for scores in outputs.scores:
                        for scores in outputs.logits:
                            probs = torch.softmax(scores[0], dim=-1).cpu().numpy()
                            token_distributions.append(probs)
                        
                        # --- Debugging: Print non-zero probabilities for each token ---
                        # # Indicies and values where nonzero
                        # for dist in token_distributions:
                        #     dist = np.array(dist)  # Convert to NumPy array for processing
                        #     print(dist[dist > 0])
                        #     non_zero_indices = np.nonzero(dist)[0]  # Get indices of non-zero probabilities
                        #     non_zero_tokens = tokenizer.convert_ids_to_tokens(non_zero_indices.tolist())  # Convert IDs to tokens
                        #     print(non_zero_tokens)  # Print human-readable tokens

                        
                        # Analyze distributions for each token
                        token_stats = analyze_token_distributions(
                            token_distributions,
                            bucket_assignments_dict
                        )
                        
                        # Record results
                        for idx, (token_stat, decoded_token) in enumerate(zip(token_stats, decoded_tokens)):
                            result = {
                                "prompt": prompt,
                                "system_prompt": system_prompt,
                                "temperature": temperature,
                                "max_output_tokens": max_output_token,
                                "generated_text": generated_text,
                                "token_index": token_stat["token_index"],
                                "token_text": decoded_token,
                                "cumulative_text": "".join(decoded_tokens[:idx + 1]),
                                "entropy": token_stat["entropy"]
                            }
                            # Add AUC values for each bucket size
                            for k, v in token_stat.items():
                                if k.startswith("auc_"):
                                    result[k] = v
                            results.append(result)
                        
                        # Clean up CUDA memory
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing prompt with parameters:")
                        print(f"  Prompt: {prompt[:50]}...")
                        print(f"  System Prompt: {system_prompt[:50]}...")
                        print(f"  Temperature: {temperature}")
                        print(f"  Max Tokens: {max_output_token}")
                        print(f"Error: {str(e)}")
                        continue

    # Convert results to DataFrame and save
    df_results = pd.DataFrame(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).resolve().parent / "sanity_check_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"analysis_results_{timestamp}.csv"
    df_results.to_csv(output_path, index=False)
    
    # # Create a summary with aggregated statistics
    # summary_df = df_results.groupby(['prompt', 'system_prompt', 'temperature', 'max_output_tokens']).agg({
    #     'entropy': ['mean', 'std', 'min', 'max'],
    #     'generated_text': 'first'
    # }).reset_index()
    
    # # Save summary
    # summary_path = f"analysis_summary_{timestamp}.csv"
    # summary_df.to_csv(summary_path, index=False)
    
    # print(f"Analysis complete. Results saved to:")
    # print(f"  Detailed results: {output_path}")
    # print(f"  Summary: {summary_path}")

if __name__ == "__main__":
    main()
