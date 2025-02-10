import pandas as pd
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def add_output_length(df):
    # count output length per prompt
    df['output_length'] = df.groupby('idx')['token_index'].transform('max')
    return df

output_length_intervals = [(0,200), (200,400), (400,600), (600,800), (800,1000), (1000,1200)]

# ------------------------------ ARG PARSING -----------------------------------
parser = argparse.ArgumentParser(description='Plot AUC and Entropy for token distributions.')
parser.add_argument('-f', '--file', required=True, type=str, help='Path to the metadata.csv file.')
args = parser.parse_args()

# ------------------------------ SETUP DIRECTORIES -----------------------------
metadata_path = Path(args.file)
output_dir = metadata_path.parent / 'plots_by_output_length'
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------ LOAD DATA --------------------------------------
df_statistics = pd.read_csv(metadata_path)

# add output length
df_statistics = add_output_length(df_statistics)

print("Columns in metadata.csv:")
print(df_statistics.columns)

# --------------------------- IDENTIFY AUC COLUMNS -----------------------------
# Identify all columns that start with 'auc_'
auc_columns = [col for col in df_statistics.columns if col.startswith('auc_')]
print(f"Identified AUC columns: {auc_columns}")

# ---------------------------- METHODOLOGY SUMMARY -----------------------------
# Summary statistics for entropy
entropy_summary = df_statistics['entropy'].describe()
print("\nEntropy Summary Statistics:")
print(entropy_summary)

# Summary statistics for AUCs
auc_summary = df_statistics[auc_columns].describe()
print("\nAUC Summary Statistics:")
print(auc_summary)

# -------------------------- RESHAPE AUC DATA ----------------------------------
# Melt the AUC columns to long format
df_auc_long = df_statistics.melt(
    id_vars=['idx', 'prompt', 'category', 'token_index', 'output_length'],
    value_vars=auc_columns,
    var_name='bucket_size',
    value_name='auc'
)

# Extract numeric bucket size from 'auc_{bucket_size}' column
df_auc_long['bucket_size'] = df_auc_long['bucket_size'].str.replace('auc_', '').astype(int)

# -------------------------- PLOTTING SETUP -------------------------------------
# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# -------------------------- AUC BOXPLOT ---------------------------------------
# for interval in output_length_intervals:
#     plt.figure(figsize=(12, 8))
#     sns.boxplot(x='bucket_size', y='auc', data=df_auc_long[df_auc_long['output_length'].isin(interval)])
#     plt.xscale('log', base=2)  # Log scale for bucket sizes
#     plt.xlabel('Number of Buckets (Log Scale)')
#     plt.ylabel('AUC Value')
#     plt.title('Distribution of AUC Across Bucket Sizes')
#     plt.tight_layout()
#     plt.savefig(output_dir / f'auc_distribution_boxplot_{interval[0]}_{interval[1]}.png')
#     plt.close()

# -------------------------- AUC LINEPLOT ---------------------------------------
# Compute average AUC per bucket size
# for interval in output_length_intervals:
#     avg_auc = df_auc_long[df_auc_long['output_length'].isin(interval)].groupby('bucket_size')['auc'].mean().reset_index()

#     plt.figure(figsize=(12, 8))
#     sns.lineplot(x='bucket_size', y='auc', data=avg_auc, marker='o')
#     plt.xscale('log', base=2)
#     plt.xlabel('Number of Buckets (Log Scale)')
#     plt.ylabel('Average AUC Value')
#     plt.title('Average AUC Across Bucket Sizes')
#     plt.tight_layout()
#     plt.savefig(output_dir / f'avg_auc_trend_{interval[0]}_{interval[1]}.png')
#     plt.close()

# ---------------- AUC BY CATEGORY LINEPLOT ------------------------------------
# Compute average AUC per bucket size and category
for interval in output_length_intervals:
    avg_auc_cat = df_auc_long[df_auc_long['output_length'].isin(interval)].groupby(['bucket_size', 'category'])['auc'].mean().reset_index()

    if avg_auc_cat.empty:
        print(f"No data for interval {interval}")
        continue

    plt.figure(figsize=(14, 10))
    sns.lineplot(x='bucket_size', y='auc', hue='category', data=avg_auc_cat, marker='o')
    plt.xscale('log', base=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Number of Buckets (Log Scale)')
    plt.ylabel('Average AUC Value')
    plt.title(f'Average AUC Across Bucket Sizes by Category for Output Length {interval[0]}-{interval[1]}')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / f'avg_auc_trend_by_category_{interval[0]}_{interval[1]}.png')
    plt.close()

sys.exit()

# ---------------------------- AUC HISTOGRAM -------------------------------------
for interval in output_length_intervals:
    plt.figure(figsize=(12, 8))
    sns.histplot(df_auc_long[df_auc_long['output_length'].isin(interval)]['auc'], bins=50, kde=True)
    plt.xlabel('AUC Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of AUC Values')
    plt.tight_layout()
    plt.savefig(output_dir / f'auc_histogram_{interval[0]}_{interval[1]}.png')
    plt.close()

# ------------------- AUC SCATTER PLOT FOR SELECTED BUCKETS -------------------
# Select specific bucket sizes for scatter plot (e.g., 2, 8, 32, ..., 8192)
selected_bucket_sizes = [2**i for i in range(1, 15, 2)]  # 2, 8, 32, ..., 8192

df_auc_selected = df_auc_long[df_auc_long['bucket_size'].isin(selected_bucket_sizes)]

plt.figure(figsize=(14, 10))
sns.scatterplot(x='bucket_size', y='auc', hue='category', data=df_auc_selected, alpha=0.5)
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('AUC Value')
plt.title('Scatter Plot of AUC Values for Selected Bucket Sizes')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'auc_scatter_selected_buckets.png')
plt.close()

# -------------------------- ENTROPY HISTOGRAM -----------------------------------
plt.figure(figsize=(12, 8))
sns.histplot(df_statistics['entropy'], bins=50, kde=True)
plt.xlabel('Entropy Value')
plt.ylabel('Frequency')
plt.title('Histogram of Entropy Values')
plt.tight_layout()
plt.savefig(output_dir / 'entropy_histogram.png')
plt.close()

# -------------------------- ENTROPY BOXPLOT -------------------------------------
plt.figure(figsize=(12, 8))
sns.boxplot(x='category', y='entropy', data=df_statistics)
plt.xlabel('Category')
plt.ylabel('Entropy Value')
plt.title('Distribution of Entropy Across Categories')
plt.tight_layout()
plt.savefig(output_dir / 'entropy_distribution_boxplot.png')
plt.close()

# ---------------------- ENTROPY BY CATEGORY BARPLOT ----------------------------
# Compute average entropy per category
avg_entropy_cat = df_statistics.groupby('category')['entropy'].mean().reset_index()

plt.figure(figsize=(14, 10))
sns.barplot(x='category', y='entropy', data=avg_entropy_cat, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Average Entropy Value')
plt.title('Average Entropy by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'avg_entropy_by_category.png')
plt.close()
"""
# -------------------- 1. Average Entropy vs. Token Index ------------------------
# Compute average entropy per token index
avg_entropy_token = df_statistics.groupby('token_index')['entropy'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(x='token_index', y='entropy', data=avg_entropy_token, marker='o')
plt.xlabel('Token Index')
plt.ylabel('Average Entropy')
plt.ylim(0.0, 5.0)
plt.title('Average Entropy Across Token Indices')
plt.tight_layout()
plt.savefig(output_dir / 'avg_entropy_vs_token_index.png')
plt.close()

# -------------------- 2. Average AUC vs. Token Index ----------------------------
# Compute average AUC per token index and bucket size
avg_auc_token = df_auc_long.groupby(['token_index', 'bucket_size'])['auc'].mean().reset_index()
avg_auc_token['bucket_size'] = avg_auc_token['bucket_size'].astype('category')

plt.figure(figsize=(14, 10))
sns.lineplot(x='token_index', y='auc', hue='bucket_size', data=avg_auc_token, marker='o')
plt.xlabel('Token Index')
plt.ylabel('Average AUC')
plt.ylim(0.0, 1.0)
plt.title('Average AUC Across Token Indices for Selected Bucket Sizes')
plt.legend(title='Bucket Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'avg_auc_vs_token_index_selected_buckets.png')
plt.close()

# -------------------- 3. Entropy Distribution by Token Index --------------------
plt.figure(figsize=(14, 10))
sns.boxplot(x='token_index', y='entropy', data=df_statistics)
plt.xlabel('Token Index')
plt.ylabel('Entropy Value')
plt.ylim(0.0, 5.0)
plt.title('Entropy Distribution Across Token Indices')
plt.yscale('linear')  # You can change to 'log' if entropy values vary widely
plt.tight_layout()
plt.savefig(output_dir / 'entropy_boxplot_by_token_index.png')
plt.close()

# -------------------- 4. AUC Distribution by Token Index ------------------------
# Similar to entropy, plot AUC distributions for selected bucket sizes
df_auc_selected_token = df_auc_long.copy()

plt.figure(figsize=(14, 10))
sns.boxplot(x='token_index', y='auc', hue='bucket_size', data=df_auc_selected_token)
plt.xlabel('Token Index')
plt.ylabel('AUC Value')
plt.ylim(0.0, 1.0)
plt.title('AUC Distribution Across Token Indices for Selected Bucket Sizes')
plt.legend(title='Bucket Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'auc_boxplot_by_token_index_selected_buckets.png')
plt.close()

# -------------------- 5. Output Length Distribution ------------------------
plt.figure(figsize=(14, 10))
sns.boxplot(x='output_length', y='entropy', data=df_statistics)
plt.xlabel('Output Length')
plt.ylabel('Entropy Value')
plt.title('Entropy Distribution Across Output Lengths')
plt.tight_layout()
plt.savefig(output_dir / 'entropy_boxplot_by_output_length.png')
plt.close()

"""
# -------------------------- OPTIONAL: SAVE SUMMARY STATISTICS -------------------
# Save summary statistics to a text file
with open(output_dir / 'summary_statistics.txt', 'w') as f:
    f.write("Entropy Summary Statistics:\n")
    f.write(entropy_summary.to_string())
    f.write("\n\nAUC Summary Statistics:\n")
    f.write(auc_summary.to_string())

print("All plots have been saved in the 'plots' directory.")
