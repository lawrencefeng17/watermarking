import pandas as pd
import os

plots_dir = '/home/lawrence/prc/plots'
df_statistics = pd.read_csv('/home/lawrence/prc/token_statistics_merged.csv')

print(df_statistics.columns)
summary = df_statistics['statistic'].describe()
print(summary)

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.boxplot(x='bucket_size', y='statistic', data=df_statistics)
plt.xscale('log', base=2)  # Log scale for bucket sizes
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Statistic Value')
plt.title('Distribution of Statistics Across Bucket Sizes')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'stat_distribution_boxplot.png'))
plt.close()

# Compute average statistic per bucket size
avg_stats = df_statistics.groupby('bucket_size')['statistic'].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(x='bucket_size', y='statistic', data=avg_stats, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Average Statistic Value')
plt.title('Average Statistic Across Bucket Sizes')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'avg_stat_trend.png'))
plt.close()

# Compute average statistic per bucket size and category
avg_stats_cat = df_statistics.groupby(['bucket_size', 'category'])['statistic'].mean().reset_index()

plt.figure(figsize=(14, 10))
sns.lineplot(x='bucket_size', y='statistic', hue='category', data=avg_stats_cat, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Average Statistic Value')
plt.title('Average Statistic Across Bucket Sizes by Category')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'avg_stat_trend_by_category.png'))
plt.close()

plt.figure(figsize=(12, 8))
sns.histplot(df_statistics['statistic'], bins=50, kde=True)
plt.xlabel('Statistic Value')
plt.ylabel('Frequency')
plt.title('Histogram of Statistic Values')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'stat_histogram.png'))
plt.close()

# Select specific bucket sizes for scatter plot
selected_bucket_sizes = [2**i for i in range(1, 15, 2)]  # e.g., 2, 8, 32, ..., 8192

df_selected = df_statistics[df_statistics['bucket_size'].isin(selected_bucket_sizes)]

plt.figure(figsize=(14, 10))
sns.scatterplot(x='bucket_size', y='statistic', hue='category', data=df_selected, alpha=0.5)
plt.xscale('log', base=2)
plt.xlabel('Number of Buckets (Log Scale)')
plt.ylabel('Statistic Value')
plt.title('Scatter Plot of Statistic Values for Selected Bucket Sizes')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'stat_scatter_selected_buckets.png'))
plt.close()


# Pivot the data for heatmap
heatmap_data = df_statistics.groupby(['bucket_size', 'category'])['statistic'].mean().unstack()

plt.figure(figsize=(16, 12))
sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f")
plt.xscale('log', base=2)
plt.xlabel('Category')
plt.ylabel('Number of Buckets (Log Scale)')
plt.title('Heatmap of Average Statistics Across Bucket Sizes and Categories')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'stat_heatmap.png'))
plt.close()
