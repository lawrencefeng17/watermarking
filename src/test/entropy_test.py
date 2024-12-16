import pandas as pd

path = "/home/lawrence/prc/src/statistics/llama-3.2-1B-instruct/dolly15k_token_entropy.csv"

df = pd.read_csv(path)

# print first 5 rows
print(df.head(50))