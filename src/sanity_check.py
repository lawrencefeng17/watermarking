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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from compute_statistics import assign_buckets 
from compute_statistics import analyze_token_distributions

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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

def plot_token_metrics(df: pd.DataFrame, 
                      prompt: str, 
                      system_prompt: str = None, 
                      temperature: float = None,
                      max_output_tokens: int = None,
                      save_dir: Path = None):
    """
    Create a detailed visualization of token-level metrics for a specific generation.
    
    Args:
        df: DataFrame containing token analysis results
        prompt: The input prompt to filter for
        system_prompt: Optional system prompt to filter for
        temperature: Optional temperature to filter for
        save_dir: Optional directory to save plots
    """
    # Filter data
    mask = df['prompt'] == prompt
    if system_prompt is not None:
        mask &= df['system_prompt'] == system_prompt
    if temperature is not None:
        mask &= df['temperature'] == temperature
    if max_output_tokens is not None:
        mask &= df['max_output_tokens'] == max_output_tokens
    
    data = df[mask].copy()
    
    if len(data) == 0:
        print("No data found for the specified parameters")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])
    
    # 1. Entropy plot
    ax1 = fig.add_subplot(gs[0])
    sns.barplot(data=data, x='token_index', y='entropy', ax=ax1)
    ax1.set_title('Token-wise Entropy')
    ax1.set_xlabel('')
    
    # 2. AUC plots for selected bucket sizes
    ax2 = fig.add_subplot(gs[1])
    selected_buckets = [2, 4, 8, 16]  # Can be modified based on needs
    
    # Plot AUC lines
    for bucket in selected_buckets:
        auc_col = f'auc_{bucket}'
        avg_auc = data[auc_col].mean()
        
        # Plot the AUC line
        line = sns.lineplot(data=data, x='token_index', y=auc_col, 
                          label=f'AUC (buckets={bucket})', ax=ax2)
        
        # Add horizontal average line with the same color
        color = line.get_lines()[-1].get_color()
        ax2.axhline(y=avg_auc, color=color, linestyle='--', alpha=0.5)
        
        # Add text annotation for the average
        ax2.text(ax2.get_xlim()[1], avg_auc, 
                f' Avg({bucket})={avg_auc:.3f}', 
                color=color, va='center')
    
    ax2.set_title('Token-wise AUC for Different Bucket Sizes')
    ax2.set_xlabel('')
    
    # 3. Token text visualization
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    # Create text blocks for tokens
    token_texts = data['token_text'].tolist()
    cumulative_text = ''
    text_blocks = []
    positions = []
    
    for idx, token in enumerate(token_texts):
        cumulative_text += token
        text_blocks.append(token)
        positions.append(idx)
    
    # Calculate colors based on entropy
    norm = plt.Normalize(data['entropy'].min(), data['entropy'].max())
    colors = plt.cm.viridis(norm(data['entropy']))
    
    # Plot tokens with their metrics
    for idx, (token, pos, color) in enumerate(zip(text_blocks, positions, colors)):
        ax3.text(pos, 0, token, fontsize=10, 
                bbox=dict(facecolor=color, alpha=0.3, edgecolor='none'),
                ha='center')
        
        # Add entropy value above token
        ax3.text(pos, 0.2, f"{data['entropy'].iloc[idx]:.2f}",
                fontsize=8, ha='center', color='black')
        
        # Add AUC value below token for one selected bucket size
        ax3.text(pos, -0.2, f"AUC: {data[f'auc_4'].iloc[idx]:.2f}",
                fontsize=8, ha='center', color='black')
    
    ax3.set_xlim(-1, len(positions))
    ax3.set_ylim(-0.5, 0.5)
    
    # Add metadata
    plt.figtext(0.02, 0.98, f"Prompt: {prompt}", fontsize=10, ha='left')
    if system_prompt:
        plt.figtext(0.02, 0.96, f"System: {system_prompt}", fontsize=10, ha='left')
    if temperature:
        plt.figtext(0.02, 0.94, f"Temperature: {temperature}", fontsize=10, ha='left')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f"token_analysis_{prompt}_{system_prompt}_{temperature}_{max_output_tokens}tokens_{timestamp}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_comparison_grid(df: pd.DataFrame, 
                        prompts: list = None,
                        system_prompts: list = None,
                        temperatures: list = None,
                        max_output_tokens: int = None,
                        metric: str = 'entropy',
                        save_dir: Path = None):
    """
    Create a grid of plots comparing different combinations of parameters.
    
    Args:
        df: DataFrame containing token analysis results
        prompts: List of prompts to compare (if None, use unique values)
        system_prompts: List of system prompts to compare
        temperatures: List of temperatures to compare
        metric: Metric to compare ('entropy' or 'auc_X' where X is bucket size)
        save_dir: Optional directory to save plots
    """
    # Filter max_output_tokens if specified
    if max_output_tokens is not None:
        mask = df['max_output_tokens'] == max_output_tokens
        df = df[mask]  

    # Get unique values if not specified
    if prompts is None:
        prompts = df['prompt'].unique()
    if system_prompts is None:
        system_prompts = df['system_prompt'].unique()
    if temperatures is None:
        temperatures = sorted(df['temperature'].unique())
    
    # Create grid
    n_temps = len(temperatures)
    n_systems = len(system_prompts)
    
    for prompt in prompts:
        fig = plt.figure(figsize=(5*n_temps, 4*n_systems))
        fig.suptitle(f"Comparison Grid for Prompt (max_output_tokens={max_output_tokens}): {prompt[:50]}...", fontsize=12)
        
        for i, sys_prompt in enumerate(system_prompts):
            for j, temp in enumerate(temperatures):
                ax = plt.subplot(n_systems, n_temps, i*n_temps + j + 1)
                
                # Filter data
                mask = (df['prompt'] == prompt) & \
                       (df['system_prompt'] == sys_prompt) & \
                       (df['temperature'] == temp)
                data = df[mask]
                
                if len(data) > 0:
                    sns.lineplot(data=data, x='token_index', y=metric, ax=ax)
                    ax.set_title(f"Sys: {sys_prompt[:20]}...\nTemp: {temp}")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                
                if i == n_systems-1:
                    ax.set_xlabel('Token Index')
                if j == 0:
                    ax.set_ylabel(metric)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / f"comparison_grid_{prompt}_max-tokens={max_output_tokens}_{timestamp}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def plot_aggregate_statistics(df: pd.DataFrame, save_dir: Path = None, max_output_tokens: int = None):
    """
    Create visualizations of aggregate statistics across different parameters.
    
    Args:
        df: DataFrame containing token analysis results
        save_dir: Optional directory to save plots
    """
    # Filter max_output_tokens if specified
    if max_output_tokens is not None:
        mask = df['max_output_tokens'] == max_output_tokens
        df = df[mask]

    # 1. Temperature vs Entropy boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='temperature', y='entropy')
    plt.title('Distribution of Entropy Across Temperatures')
    if save_dir:
        plt.savefig(save_dir / f'temp_vs_entropy_{timestamp}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 2. System Prompt vs Entropy violin plot
    plt.figure(figsize=(15, 6))
    sns.violinplot(data=df, x='system_prompt', y='entropy')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Entropy Across System Prompts')
    if save_dir:
        plt.savefig(save_dir / f'system_vs_entropy_{timestamp}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 3. AUC comparison across bucket sizes
    auc_cols = [col for col in df.columns if col.startswith('auc_')]
    auc_data = df[auc_cols].mean().reset_index()
    auc_data.columns = ['bucket_size', 'mean_auc']
    auc_data['bucket_size'] = auc_data['bucket_size'].str.replace('auc_', '').astype(int)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=auc_data, x='bucket_size', y='mean_auc')
    plt.xscale('log', base=2)
    plt.title('Average AUC vs Bucket Size')
    plt.xlabel('Number of Buckets (log scale)')
    plt.ylabel('Mean AUC')
    if save_dir:
        plt.savefig(save_dir / f'auc_vs_buckets_{timestamp}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_analysis_report(df: pd.DataFrame, output_dir: Path):
    """
    Create a comprehensive analysis report with all visualizations.
    
    Args:
        df: DataFrame containing token analysis results
        output_dir: Directory to save the report and visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create individual token analysis plots for each unique prompt
    viz_dir = output_dir / 'token_analysis'
    viz_dir.mkdir(exist_ok=True)
    
    unique_prompts = df['prompt'].unique()
    unique_system_prompts = df['system_prompt'].unique()
    unique_temperatures = df['temperature'].unique()
    unique_max_output_tokens = df['max_output_tokens'].unique()
    for prompt in unique_prompts:
        for system_prompt in unique_system_prompts:
            for temperature in unique_temperatures:
                for max_output_tokens in unique_max_output_tokens:
                    plot_token_metrics(df, 
                                      prompt=prompt, 
                                      system_prompt=system_prompt, 
                                      temperature=temperature, 
                                      max_output_tokens=max_output_tokens, 
                                      save_dir=viz_dir)
    
    # 2. Create comparison grids
    grid_dir = output_dir / 'comparison_grids'
    grid_dir.mkdir(exist_ok=True)
    
    for token_limit in unique_max_output_tokens:
        plot_comparison_grid(df, prompts=unique_prompts, save_dir=grid_dir, max_output_tokens=token_limit)
    
    # 3. Create aggregate statistics plots
    stats_dir = output_dir / 'aggregate_stats'
    stats_dir.mkdir(exist_ok=True)
    for token_limit in unique_max_output_tokens:
        plot_aggregate_statistics(df, save_dir=stats_dir, max_output_tokens=token_limit)

def plot_simple_token_metrics(df: pd.DataFrame, 
                            prompt: str,
                            system_prompt: str = None,
                            temperature: float = None,
                            bucket_size: int = 2,
                            max_output_tokens: int = None, 
                            save_dir: Path = None,
                            max_tokens_per_plot: int = 50):
    """
    Create a simple bar plot showing entropy and AUC for each generated token.
    
    Args:
        df: DataFrame containing token analysis results
        prompt: The input prompt to filter for
        system_prompt: Optional system prompt to filter for
        temperature: Optional temperature to filter for
        bucket_size: Which AUC bucket size to display (default: 4)
    """
    # Make directory
    folder = save_dir / "simple_token_metrics"
    folder.mkdir(parents=True, exist_ok=True)

    # Filter data
    mask = df['prompt'] == prompt
    if system_prompt is not None:
        mask &= df['system_prompt'] == system_prompt
    if temperature is not None:
        mask &= df['temperature'] == temperature
    if max_output_tokens is not None:
        mask &= df['max_output_tokens'] == max_output_tokens 
    
    data = df[mask].copy()
    
    if len(data) == 0:
        print("No data found for the specified parameters")
        return
    
    num_tokens = len(data)
    num_plots = (num_tokens - 1) // max_tokens_per_plot + 1

    for plot_idx in tqdm(range(num_plots)):
        start_idx = plot_idx * max_tokens_per_plot
        end_idx = min((plot_idx + 1) * max_tokens_per_plot, num_tokens)
        plot_data = data.iloc[start_idx:end_idx]
        
        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[1, 1])
        fig.subplots_adjust(hspace=0.3)
        
        # Plot entropy
        sns.barplot(data=plot_data, x=range(len(plot_data)), y='entropy', ax=ax1, color='skyblue')
        ax1.set_title(f'Token-wise Entropy (tokens {start_idx}-{end_idx-1})')
        ax1.set_xticklabels([])  # Remove x labels from top plot
        ax1.set_xlabel('')

        # Add average entropy line
        avg_entropy = plot_data['entropy'].mean()
        ax1.axhline(y=avg_entropy, color='red', linestyle='--', alpha=0.5)
        ax1.text(len(plot_data)-1, avg_entropy, f' Avg={avg_entropy:.3f}', 
                color='red', va='center')
        
        # Plot AUC
        sns.barplot(data=plot_data, x=range(len(plot_data)), y=f'auc_{bucket_size}', ax=ax2, color='lightgreen')
        ax2.set_title(f'Token-wise AUC (buckets={bucket_size})')
        
        # Set x-axis labels
        ax2.set_xticks(range(len(plot_data)))
        ax2.set_xticklabels(plot_data['token_text'], rotation=45, ha='right')
        
        # Add average AUC line
        avg_auc = plot_data[f'auc_{bucket_size}'].mean()
        ax2.axhline(y=avg_auc, color='red', linestyle='--', alpha=0.5)
        ax2.text(len(plot_data)-1, avg_auc, f' Avg={avg_auc:.3f}', 
                color='red', va='center')
        
        # Add metadata
        plt.figtext(0.02, 0.98, f"Prompt: {prompt[:50]}...", fontsize=10, ha='left')
        if system_prompt:
            plt.figtext(0.02, 0.96, f"System: {system_prompt[:50]}...", fontsize=10, ha='left')
        if temperature:
            plt.figtext(0.02, 0.94, f"Temperature: {temperature}", fontsize=10, ha='left')
    
        plt.tight_layout()

        if save_dir:
            save_path = folder / f"simple_token_metrics_{prompt}_{system_prompt}_{temperature}_{max_output_tokens}tokens_part_{plot_idx}_{timestamp}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

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

    temperatures = [0.5, 0.7, 1.0, 1.2, 1.4]
    # temperatures = [0.5, 1.0]

    max_output_tokens = [50, 100, 250, 400]
    # max_output_tokens = [25, 50]
    parser = argparse.ArgumentParser(description="Eagerly analyze token distributions.")
    parser.add_argument('--model', type=str, required=True, help='Name or path of the model.')
    parser.add_argument('--quantize', action='store_true', help='Use 8-bit quantization for the model.', default=False)
    parser.add_argument('--prompt', type=str, help='Prompt to analyze.', default=None)
    parser.add_argument('--sys', type=str, help='System prompt to analyze.', default=None)
    parser.add_argument('--csv', type=str, help='CSV data file to analyze.', default=None)
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
    
    if args.csv is None:
        # Generate data
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
                                breakpoint()
                            
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
                            
                            print("Prompt:" + prompt)
                            print("System Prompt:" + system_prompt)
                            print("Temperature:" + str(temperature))
                            print("Max Output Tokens:" + str(max_output_token))
                            print("Generated Text:" + generated_text)

                            print("-----------------------------")
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
        output_dir = Path(__file__).resolve().parent / "sanity_check_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"analysis_results_{timestamp}.csv"
        df_results.to_csv(output_path, index=False)
        
        # Create a summary with aggregated statistics
        summary_df = df_results.groupby(['prompt', 'system_prompt', 'temperature', 'max_output_tokens']).agg({
            'entropy': ['mean', 'std'],
            'auc_2': ['mean', 'std'],
            'auc_4': ['mean', 'std'],
            'auc_8': ['mean', 'std'],
            'auc_16': ['mean', 'std'],
            'auc_32': ['mean', 'std'],
            'auc_64': ['mean', 'std'],
            'auc_128': ['mean', 'std'],
            'auc_256': ['mean', 'std'],
            'auc_512': ['mean', 'std'],
            'auc_1024': ['mean', 'std'],
            'auc_2048': ['mean', 'std'],
            'auc_4096': ['mean', 'std'],
            'auc_8192': ['mean', 'std'],
            'auc_16384': ['mean', 'std'],
        }).reset_index()
        
        # Save summary
        summary_path = output_dir / f"analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Analysis complete. Results saved to:")
        print(f"  Detailed results: {output_path}")
        print(f"  Summary: {summary_path}")
    
    if args.csv is not None:
        # Load existing results
        df_results = pd.read_csv(args.csv)
        output_dir = Path(__file__).resolve().parent / "sanity_check_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Create analysis report with visualizations
    create_analysis_report(df_results, output_dir)

    # Simple token metrics plot for a few configs

    # "List the prime numbers less than 100.", 
    # "Explain quantum physics.",
    # "Tell me a love story between a robot and a human.",
    
    # "Be concise and informative.",
    # "Respond as if you were a magical Wizard from an ancient land.",
    # "Use flashy language and be engaging.",

    for prompt in prompts:
        for system_prompt in system_prompts:
            for temperature in temperatures:
                plot_simple_token_metrics(df_results, 
                                        prompt=prompt, 
                                        system_prompt=system_prompt, 
                                        temperature=temperature, 
                                        max_output_tokens=100, 
                                        save_dir=output_dir)
                              
if __name__ == "__main__":
    main()
