import pandas as pd
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------ ARG PARSING -----------------------------------
parser = argparse.ArgumentParser(description='Plot AUC and Entropy for token distributions.')
parser.add_argument('-f', '--file', required=True, type=str, help='Path to the metadata.csv file.')
args = parser.parse_args()

# ------------------------------ SETUP DIRECTORIES -----------------------------
metadata_path = Path(args.file)
output_dir = metadata_path.parent / 'plots'
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------ LOAD DATA --------------------------------------
df_statistics = pd.read_csv(metadata_path)

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
    id_vars=['idx', 'prompt', 'category', 'token_index'],
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
plt.figure(figsize=(12, 8))
sns.boxplot(x='bucket_size', y='auc', data=df_auc_long)
plt.xscale('log', base=2)  # Log scale for bucket sizes
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('AUC Value')
plt.title('Distribution of AUC Across Bucket Sizes')
plt.tight_layout()
plt.savefig(output_dir / 'auc_distribution_boxplot.png')
plt.close()

# -------------------------- AUC LINEPLOT ---------------------------------------
# Compute average AUC per bucket size
avg_auc = df_auc_long.groupby('bucket_size')['auc'].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(x='bucket_size', y='auc', data=avg_auc, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Average AUC Value')
plt.title('Average AUC Across Bucket Sizes')
plt.tight_layout()
plt.savefig(output_dir / 'avg_auc_trend.png')
plt.close()

# ---------------- AUC BY CATEGORY LINEPLOT ------------------------------------
# Compute average AUC per bucket size and category
avg_auc_cat = df_auc_long.groupby(['bucket_size', 'category'])['auc'].mean().reset_index()

plt.figure(figsize=(14, 10))
sns.lineplot(x='bucket_size', y='auc', hue='category', data=avg_auc_cat, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Average AUC Value')
plt.title('Average AUC Across Bucket Sizes by Category')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'avg_auc_trend_by_category.png')
plt.close()

# ---------------------------- AUC HISTOGRAM -------------------------------------
plt.figure(figsize=(12, 8))
sns.histplot(df_auc_long['auc'], bins=50, kde=True)
plt.xlabel('AUC Value')
plt.ylabel('Frequency')
plt.title('Histogram of AUC Values')
plt.tight_layout()
plt.savefig(output_dir / 'auc_histogram.png')
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

# -------------------------- AUC HEATMAP ----------------------------------------
# Pivot the data for heatmap: rows=bucket_size, columns=category, values=average AUC
heatmap_data_auc = avg_auc_cat.pivot(index='bucket_size', columns='category', values='auc')

plt.figure(figsize=(16, 12))
sns.heatmap(heatmap_data_auc, cmap='viridis', annot=False, fmt=".2f")
plt.xscale('log', base=2)  # Not necessary for categorical axis, can remove
plt.xlabel('Category')
plt.ylabel('Number of Buckets')
plt.title('Heatmap of Average AUC Across Bucket Sizes and Categories')
plt.tight_layout()
plt.savefig(output_dir / 'auc_heatmap.png')
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

# ---------------------------- ENTROPY HEATMAP ------------------------------------
# If you have multiple dimensions to heatmap entropy, otherwise skip
# Example: Heatmap of average entropy per bucket size and category (if applicable)
# Since entropy is per token and not per bucket size, this might not be directly applicable
# Instead, you can visualize entropy distribution across categories

heatmap_data_entropy = df_statistics.groupby(['category'])['entropy'].mean().reset_index().pivot(index='category', columns='category', values='entropy')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_entropy, cmap='magma', annot=True, fmt=".2f")
plt.xlabel('Category')
plt.ylabel('Category')
plt.title('Heatmap of Average Entropy by Category')
plt.tight_layout()
plt.savefig(output_dir / 'entropy_heatmap.png')
plt.close()

# -------------------------- OPTIONAL: SAVE SUMMARY STATISTICS -------------------
# Save summary statistics to a text file
with open(output_dir / 'summary_statistics.txt', 'w') as f:
    f.write("Entropy Summary Statistics:\n")
    f.write(entropy_summary.to_string())
    f.write("\n\nAUC Summary Statistics:\n")
    f.write(auc_summary.to_string())

print("All plots have been saved in the 'plots' directory.")
