import pandas as pd
from sanity_check import plot_simple_token_metrics  
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

prompts = [
        "List the prime numbers less than 100.", 
        "What is the meaning of life?",
        "Explain quantum physics.",
        "Tell me a love story between a robot and a human.",
    ]
        
system_prompts = [
        "Be creative and helpful.",
        "Be concise and informative.",
        "Be detailed and thorough.",
        "Use flashy language and be engaging.",
    ] 

temperatures = [0.5, 0.7, 1.0, 1.2, 1.4]
# temperatures = [0.5, 1.0]

max_output_tokens = [50, 100, 250, 400]

df_results = pd.read_csv("/home/lawrence/prc/src/sanity_check_results/analysis_results_20250117_095113.csv")

output_dir = Path(__file__).resolve().parent / "sanity_check_results"

# Simple token metrics plot
for prompt in prompts:
    for system_prompt in system_prompts:
        for temperature in temperatures:
            for max_output_token in max_output_tokens:
                plot_simple_token_metrics(df_results, 
                                            prompt=prompt, 
                                            system_prompt=system_prompt, 
                                            temperature=temperature, 
                                            save_dir=output_dir, 
                                            max_output_tokens=max_output_token)
