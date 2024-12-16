import pandas as pd

df1 = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

# Verify the dataset structure
print(df1.columns)

# Load the token_statistics.csv
csv_path = '/home/lawrence/prc/token_statistics_original.csv'
df2 = pd.read_csv(csv_path)
df2.drop(columns = ['category'], inplace=True)
print(df2.columns)

result = pd.merge(df1, df2, left_on='instruction', right_on='prompt', how='left')

# Save the merged DataFrame to a new CSV file
result.to_csv('/home/lawrence/prc/token_statistics_merged.csv', index=False)
