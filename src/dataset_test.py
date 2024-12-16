import pandas as pd

df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

print(df.head())    
print(df.columns)