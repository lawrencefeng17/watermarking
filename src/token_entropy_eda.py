import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
input_csv = '/home/lawrence/prc/src/statistics/llama-3.2-1B-instruct/token_entropy.csv'  # Replace with your entropy CSV file path
plots_dir = '/home/lawrence/prc/plots/'  # Replace with your desired output directory

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the entropy CSV
df = pd.read_csv(input_csv)

# Ensure the DataFrame has the required columns
if 'entropy' not in df.columns or 'category' not in df.columns:
    raise ValueError("The CSV must contain at least 'entropy' and 'category' columns.")

# ------------------------------
# Plotting Functions
# ------------------------------

# 1. Histogram of Entropy
def plot_entropy_histogram(df, output_path):
    plt.figure(figsize=(12, 8))
    sns.histplot(df['entropy'], bins=50, kde=True, color='blue')
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Entropy')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 2. Violin Plot of Entropy by Category
def plot_entropy_violin(df, output_path):
    plt.figure(figsize=(14, 10))
    sns.violinplot(x='category', y='entropy', data=df, scale='width', inner='quartile', color='lightblue')
    plt.xlabel('Category')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 3. Box Plot of Entropy by Category
def plot_entropy_boxplot(df, output_path):
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='category', y='entropy', data=df)
    plt.xlabel('Category')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy Boxplot by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 4. Summary Statistics Table
def print_summary_statistics(df):
    summary = df['entropy'].describe()
    print("Summary Statistics for Entropy:")
    print(summary)

# ------------------------------
# Execute Plotting
# ------------------------------

print("Generating entropy plots...")

# Histogram
plot_entropy_histogram(df, os.path.join(plots_dir, 'entropy_histogram.png'))

# Violin plot
plot_entropy_violin(df, os.path.join(plots_dir, 'entropy_violin_by_category.png'))

# Box plot
plot_entropy_boxplot(df, os.path.join(plots_dir, 'entropy_boxplot_by_category.png'))

# Summary statistics
print_summary_statistics(df)

print(f"Plots saved to {plots_dir}")
