import pandas as pd
from pathlib import Path

df = pd.read_csv("/home/lawrence/prc/src/statistics/llama-3.2-1B-instruct/token_statistics.csv")

# print first 5 rows
print(df.head(50))