import pandas as pd
from sanity_check import plot_simple_token_metrics  
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

df_results = pd.read_csv("/home/lawrence/prc/src/sanity_check_results/analysis_results_20250117_095113.csv")

prompts = df_results["prompt"].unique()
system_prompts = df_results["system_prompt"].unique()
temperatures = df_results["temperature"].unique()

output_dir = Path(__file__).resolve().parent / "sanity_check_results"

# # Simple token metrics plot
# for prompt in prompts:
#     for system_prompt in system_prompts:
#         for temperature in temperatures:
#                 plot_simple_token_metrics(df_results, 
#                                             prompt=prompt, 
#                                             system_prompt=system_prompt, 
#                                             temperature=temperature, 
#                                             save_dir=output_dir, 
#                                             max_output_tokens=100)

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
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
print(f"  Detailed results: {output_dir}")
print(f"  Summary: {summary_path}")